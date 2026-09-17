# Public frictional-contact API + solver dispatch


"""
    solve_contact_friction!(prob; δ=0.0, tol=1e-8, maxiter=50)

Frictional contact iteration for type-4 pairs (from ContatoMultiCorpos2 logic).

States per pair: `1=open`, `2=slip`, `3=stick`.

# Laplace (scalar)
Contact is frictionless unilateral: gap ≥ 0, q ≤ 0 (compression), complementarity.
`μ` ignored for scalar.

# Elasticity
Coulomb: stick ``u_t^a = u_t^b``, ``|t_t| ≤ μ |t_n|``;
slip ``t_t = ±μ t_n``, gap closed in normal direction.
`δ` = additional rigid normal approach.
"""
function solve_contact_friction!(prob::MultiRegionProblem{<:Laplace};
    δ=0.0, tol=1e-8, maxiter=40)
    isempty(prob.contacts) && pair_contacts!(prob)
    # Frictionless unilateral for scalar potential/heat: treat like Signorini
    # on the normal flux.  We iterate active set.
    for dad in prob.regions
        has_cache(dad, :H) || H_G_full_direct(dad, 16)
    end

    for _it in 1:maxiter
        # set BC from contact state
        for cp in prob.contacts
            da, db = prob.regions[cp.reg_a], prob.regions[cp.reg_b]
            if cp.state == 1  # open: Neumann 0 both
                da.BC[cp.node_a] = BC_NEUMANN; da.BV[cp.node_a] = 0.0
                db.BC[cp.node_b] = BC_NEUMANN; db.BV[cp.node_b] = 0.0
            else  # closed: interface-like continuity of T, balance of q
                da.BC[cp.node_a] = BC_INTERFACE
                db.BC[cp.node_b] = BC_INTERFACE
            end
        end
        # rebuild interfaces from closed contacts + type-3
        pair_interfaces!(prob)
        # also add closed contacts as interfaces
        for cp in prob.contacts
            if cp.state != 1
                push!(prob.interfaces, InterfacePair(cp.reg_a, cp.node_a, cp.reg_b, cp.node_b))
            end
        end
        solve_multiregion!(prob)

        changed = false
        for cp in prob.contacts
            da, db = prob.regions[cp.reg_a], prob.regions[cp.reg_b]
            Ta, Tb = da.T[cp.node_a], db.T[cp.node_b]
            qa, qb = da.q[cp.node_a], db.q[cp.node_b]
            # gap estimate: geometric + (Tb - Ta) as relative "penetration" proxy
            # For potential this is not a mechanical gap; use flux sign
            gap = cp.gap0 - δ + (Tb - Ta)  # heuristic
            compression = -(qa)  # positive if flux into a from contact
            if cp.state == 1  # open
                if gap < -tol
                    cp.state = 3; changed = true
                end
            else  # closed
                if compression < -tol  # tension → open
                    cp.state = 1; changed = true
                end
            end
        end
        !changed && break
    end
    return prob
end

"""
Elasticity frictional contact — Contato (MATLAB) multi-body **active-set** scheme.

Builds a **coupled** global system in the nodal (n,t) frame (Leonardo / Contato):

1. Each region: ``Ĥ, Ĝ`` via local rotation; exterior BCs applied; contact faces
   keep free ``u`` with contact tractions as extra unknowns.
2. Contact pairs contribute 4 algebraic rows (open / slip / stick) as in
   `aplica_contato_com_atrito_multicorpos.m`.
3. **Explicit active-set loop** (Contato with frozen set = one linear solve):
   ```text
   state ← verify(x)
   A, b  ← assemble(state)
   x     ← A \\ b
   until ‖Δx‖ small and/or states stable
   ```
   Equivalent to one Newton step on ``R = Ax - b`` with ``J = A``.

States: `1=open`, `±2=slip`, `3=stick` (MATLAB codes).

Solvers:
- `:activeset` (default) — Contato verify → assemble → solve `A*x = b`
- `:ssn` — semi-smooth Newton on Alart–Curnier residual (same unknowns)
- `:proj_gnm` — projected Coulomb residual GNM (Rodríguez-Tembleque & Abascal 2013)
- `:proj_newton` — accelerated projected Newton: frozen-state solve +
  quasi-complementarity reduction + line search (same paper §6.1)
- `:gnmls` — GNM with line search, single contact traction ``Λ`` per pair
  (Rodríguez-Tembleque & Abascal, *Comput. Struct.* 2010: ``p¹=Λ``, ``p²=−R^{-1}Λ``)


For robustness under large approach/load prefer
[`solve_contact_friction_stepped!`](@ref) (outer load loop + warm start).

# Keywords
- `δ` — rigid approach (``h = g₀ - δ``)
- `solver` — `:activeset` | `:ssn` | `:proj_gnm` | `:proj_newton` | `:gnmls`
- `rn`, `rt` — augmentation / AC scales (default: auto from ``E/L``); SSN / projected Newton
- `x0` — optional warm-start unknown vector
- `reset_states` — if `true` (default), all pairs start open
- `return_x` — also return the unknown vector for warm starts
- `near_factor` — H,G far-lumping cutoff in units of element length
  (default `2`; `Inf` integrates every pair). With Gauss–Legendre
  collocation, nodal lumping matches full integration on the bulk twin.
"""
function solve_contact_friction!(prob::MultiRegionProblem{<:Elasticity};
        δ=0.0, δt=0.0, tol=1e-8, maxiter=40, npg=12, verbose=false,
        method::Symbol=:ntn,
        solver::Symbol=:activeset,
        rn::Union{Nothing,Real}=nothing,
        rt::Union{Nothing,Real}=nothing,
        x0::Union{Nothing,AbstractVector}=nothing,
        reset_states::Bool=true,
        return_x::Bool=false,
        common_normal::Bool=false,
        nsteps::Int=1,
        near_factor::Real=1.5)
    ctx = _contact_friction_setup(prob; method=method, npg=npg,
        common_normal=common_normal, near_factor=near_factor)
    ctx === nothing && return return_x ? (prob, Float64[]) : prob
    prep, pairs = ctx.prep, ctx.pairs
    h = [cp.gap0 - δ for cp in pairs]
    ht = _contact_ht_vec(h, δt)
    N = ctx.N
    x_init = if x0 === nothing
        zeros(N)
    else
        length(x0) == N || throw(DimensionMismatch("x0 length $(length(x0)) ≠ $N"))
        collect(Float64, x0)
    end
    if reset_states
        for cp in pairs
            cp.state = 1
            cp.ut_lock = 0.0
        end
    end
    x, ok = _contact_inner_solve!(prep, pairs, h, x_init;
        ht=ht, solver=solver, tol=tol, maxiter=maxiter, verbose=verbose, rn=rn, rt=rt,
        nsteps=nsteps)
    ok || @warn "solve_contact_friction! did not fully converge" δ=δ δt=δt solver=solver
    _verify_contact_states!(pairs, prep, h, x; ht=ht, epsc=1e-7)
    _update_contact_ut_locks!(pairs, prep, h, x; ht=ht)
    _scatter_contact_solution!(prob, prep, pairs, x)
    if !isempty(prob.regions)
        set_cache!(prob.regions[1]; contact_x=copy(x))
    end
    return return_x ? (prob, x) : prob
end

"""
    solve_contact_friction_stepped!(prob; δ_end, nsteps=10, ...)

**Load-stepped** frictional contact: outer loop on rigid approach ``δ``, inner
solver (Contato active-set, SSN, or projected Newton).

```text
for s = 1:nsteps
    δ_s = δ_end * s/nsteps
    x ← inner_solve(δ_s; warm-start x)   # :activeset | :ssn | :proj_gnm | :proj_newton | :gnmls
end
```

# Keywords
- `δ_end` / `δ_start` / `nsteps` / `δ_path` — approach schedule
- `solver` — `:activeset` | `:ssn` | `:proj_gnm` | `:proj_newton` | `:gnmls`
- `rn`, `rt` — augmentation scales for SSN / projected Newton (default auto)
- `adaptive` — bisect a failed step once and retry
- `tol`, `maxiter`, `method`, `verbose`, `npg`

History: `contact_δ_hist`, `contact_tn_hist`, `contact_x` on `prob.regions[1]`.
"""
function solve_contact_friction_stepped!(prob::MultiRegionProblem{<:Elasticity};
        δ_end::Union{Nothing,Real}=nothing,
        δ_start::Real=0.0,
        nsteps::Int=10,
        δ_path::Union{Nothing,AbstractVector}=nothing,
        δt_path::Union{Nothing,AbstractVector}=nothing,
        δt_end::Real=0.0,
        adaptive::Bool=true,
        tol=1e-8,
        maxiter=40,
        npg=12,
        verbose=false,
        method::Symbol=:ntn,
        solver::Symbol=:activeset,
        rn::Union{Nothing,Real}=nothing,
        rt::Union{Nothing,Real}=nothing,
        max_bisect::Int=6,
        common_normal::Bool=false,
        near_factor::Real=1.5)
    ctx = _contact_friction_setup(prob; method=method, npg=npg,
        common_normal=common_normal, near_factor=near_factor)
    ctx === nothing && return prob
    prep, pairs = ctx.prep, ctx.pairs
    N = ctx.N

    if δ_path !== nothing
        path = collect(Float64, δ_path)
    else
        δ_end === nothing && throw(ArgumentError("pass δ_end or δ_path"))
        nsteps >= 1 || throw(ArgumentError("nsteps ≥ 1"))
        path = collect(range(float(δ_start), float(δ_end); length=nsteps + 1))[2:end]
    end
    if δt_path !== nothing
        tpath = collect(Float64, δt_path)
        length(tpath) == length(path) || throw(DimensionMismatch(
            "δt_path length $(length(tpath)) ≠ δ_path length $(length(path))"))
    else
        tpath = fill(float(δt_end), length(path))
    end

    x = zeros(N)
    if has_cache(prob.regions[1], :contact_x)
        xw = prob.regions[1].contact_x
        length(xw) == N && (x = collect(Float64, xw))
    end
    for cp in pairs
        cp.state = 1
    end

    δ_hist = Float64[]
    δt_hist = Float64[]
    tn_hist = Float64[]
    tt_hist = Float64[]
    s = 1
    n_bisect = 0
    δ_ref = δ_end === nothing ? (isempty(path) ? 1.0 : path[end]) : float(δ_end)
    while s <= length(path)
        δ = path[s]
        δt = tpath[s]
        h = [cp.gap0 - δ for cp in pairs]
        ht = _contact_ht_vec(h, δt)
        x_try, ok = _contact_inner_solve!(prep, pairs, h, x;
            ht=ht, solver=solver, tol=tol, maxiter=maxiter, verbose=verbose, rn=rn, rt=rt)
        if !ok && adaptive && n_bisect < max_bisect && abs(δt) < 1e-15
            # bisection only on pure-normal ramps (fretting corners keep δt)
            δ_prev = s == 1 ? float(δ_start) : path[s - 1]
            δ_mid = 0.5 * (δ_prev + δ)
            if abs(δ_mid - δ_prev) > 1e-14 * max(abs(δ_ref), 1.0)
                verbose && @info "contact step failed; bisecting" δ=δ δ_mid=δ_mid solver=solver
                insert!(path, s, δ_mid)
                insert!(tpath, s, δt)
                n_bisect += 1
                continue
            end
        end
        if !ok
            @warn "solve_contact_friction_stepped! step failed" s=s δ=δ δt=δt solver=solver
        end
        x = x_try
        _verify_contact_states!(pairs, prep, h, x; ht=ht, epsc=1e-7)
        _update_contact_ut_locks!(pairs, prep, h, x; ht=ht)
        push!(δ_hist, δ)
        push!(δt_hist, δt)
        tn_mean = mean(abs(cp.tn) for cp in pairs)
        tt_mean = mean(abs(cp.tt) for cp in pairs)
        push!(tn_hist, tn_mean)
        push!(tt_hist, tt_mean)
        verbose && @info "contact step" s=s δ=δ δt=δt solver=solver n_closed=count(cp -> abs(cp.state) != 1, pairs) tn_mean=tn_mean
        s += 1
    end

    _scatter_contact_solution!(prob, prep, pairs, x)
    set_cache!(prob.regions[1]; contact_x=copy(x),
        contact_δ_hist=δ_hist, contact_δt_hist=δt_hist,
        contact_tn_hist=tn_hist, contact_tt_hist=tt_hist)
    return prob
end

"""
    solve_contact_friction_fretting!(prob; δn, δt_max, n_normal=8, top_reg=2, ...)

Cattaneo–Mindlin-style multi-body fretting at fixed normal approach `δn`.

**Important:** tangential loading is applied as a **far-field Dirichlet**
``u_x = δt`` on the top block far face (region `top_reg`), *not* as a contact
gap shift. A gap-shift ``g_t = Δu_t - δt`` forces stick nodes back to
``Δu_t = 0`` on unload and wipes Mindlin residual shear. Far-field bulk slip
lets stick zones lock residual traction when ``δt → 0``.

```text
ramp  δn ↑ ,  u_x = 0
A     δn   ,  u_x = 0
B     δn   ,  u_x = +δt_max
C     δn   ,  u_x = 0          ← residual shear
D     δn   ,  u_x = −δt_max
E     δn   ,  u_x = 0
```

Requires `set_farfield_displacement!` (from the two-block fixture) or an
equivalent BC helper on the top region.
"""
function solve_contact_friction_fretting!(prob::MultiRegionProblem{<:Elasticity};
        δn::Real,
        δt_max::Real,
        n_normal::Int=8,
        top_reg::Int=2,
        tol=1e-8,
        maxiter=40,
        npg=12,
        verbose=false,
        method::Symbol=:ntn,
        solver::Symbol=:ssn,
        rn::Union{Nothing,Real}=nothing,
        rt::Union{Nothing,Real}=nothing)
    δn = float(δn)
    δt_max = float(δt_max)
    n_normal >= 1 || throw(ArgumentError("n_normal ≥ 1"))
    1 <= top_reg <= length(prob.regions) || throw(ArgumentError("bad top_reg"))
    top = prob.regions[top_reg]
    xw = nothing
    δt_hist = Float64[]
    δ_hist = Float64[]
    ux_corners = Dict{Symbol,Float64}()

    # --- normal ramp (far-field u_x = 0) ---
    set_farfield_displacement!(top; ux=0.0, uy=0.0, face=:top)
    for δ in range(δn / n_normal, δn; length=n_normal)
        _, xw = solve_contact_friction!(prob; δ=δ, δt=0.0, solver=solver,
            method=method, tol=tol, maxiter=maxiter, npg=npg,
            x0=xw, reset_states=(xw === nothing), return_x=true, rn=rn, rt=rt,
            verbose=verbose)
        push!(δ_hist, δ); push!(δt_hist, 0.0)
    end

    # --- fretting corners A–E via bulk u_x ---
    fretting = (
        (:A, 0.0),
        (:B, +δt_max),
        (:C, 0.0),
        (:D, -δt_max),
        (:E, 0.0),
    )
    for (name, uxt) in fretting
        set_farfield_displacement!(top; ux=uxt, uy=0.0, face=:top)
        _, xw = solve_contact_friction!(prob; δ=δn, δt=0.0, solver=solver,
            method=method, tol=tol, maxiter=maxiter, npg=npg,
            x0=xw, reset_states=false, return_x=true, rn=rn, rt=rt,
            verbose=verbose)
        if solver !== :activeset
            _, xw = solve_contact_friction!(prob; δ=δn, δt=0.0, solver=:activeset,
                method=method, tol=tol, maxiter=max(20, maxiter ÷ 2), npg=npg,
                x0=xw, reset_states=false, return_x=true, verbose=verbose)
        end
        push!(δ_hist, δn); push!(δt_hist, uxt)
        ux_corners[name] = uxt
        verbose && @info "fretting corner" name=name ux=uxt n_closed=count(cp -> abs(cp.state) != 1, prob.contacts)
    end

    set_cache!(prob.regions[1]; contact_x=copy(xw),
        contact_δ_hist=δ_hist, contact_δt_hist=δt_hist,
        contact_fretting_names=[:A, :B, :C, :D, :E],
        contact_fretting_ux=ux_corners)
    return prob
end

"""Dispatch inner contact solve."""
function _contact_inner_solve!(prep, pairs, h, x_init;
        ht=nothing, solver::Symbol=:activeset, tol=1e-8, maxiter=40, verbose=false,
        rn=nothing, rt=nothing, nsteps::Int=1)
    if solver === :activeset
        return _contact_activeset!(prep, pairs, h, x_init; ht=ht, tol=tol, maxiter=maxiter,
            verbose=verbose, nsteps=nsteps)
    elseif solver === :ssn
        return _contact_ssn!(prep, pairs, h, x_init; ht=ht, tol=tol, maxiter=maxiter,
            verbose=verbose, rn=rn, rt=rt)
    elseif solver === :proj_gnm
        return _contact_proj_gnm!(prep, pairs, h, x_init; ht=ht, tol=tol, maxiter=maxiter,
            verbose=verbose, rn=rn, rt=rt)
    elseif solver === :proj_newton
        return _contact_proj_newton!(prep, pairs, h, x_init; ht=ht, tol=tol, maxiter=maxiter,
            verbose=verbose, rn=rn, rt=rt)
    elseif solver === :gnmls
        return _contact_gnmls!(prep, pairs, h, x_init; ht=ht, tol=tol, maxiter=maxiter,
            verbose=verbose, rn=rn, rt=rt)
    else
        throw(ArgumentError(
            "unknown contact solver $(repr(solver)); use :activeset, :ssn, " *
            ":proj_gnm, :proj_newton, or :gnmls"))
    end
end
