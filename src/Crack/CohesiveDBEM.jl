# =============================================================================
# Nonlinear cohesive-contact Dual BEM (Cordeiro et al. 2024)
# =============================================================================
# Builds on [`DualMesh`](@ref) / [`assemble_dual!`](@ref). Crack face pairs
# (twins) carry a cohesive law with four surface conditions:
#   (i) contact  (ii) softening  (iii) unloading/reloading  (iv) complete failure
#
# Discrete system (compact form of Cordeiro eq. 24):
#   A(u) u = f(λ)
# where cohesive stiffness K(δ) couples tractions on ± faces to the opening
# Δu = u⁺ − u⁻ through the already-assembled H, G operators.
#
# Solution: Newton–Raphson with optional single-DOF displacement control
# (arc-length-like continuation on a monitored DOF).

export CohesivePair, CohesiveDBEMProblem
export build_cohesive_pairs, solve_cohesive_dbem!
export cohesive_tractions, cohesive_openings
export modeI_patch_mesh, modeII_patch_mesh, contact_compression_mesh
export extend_cohesive_process_zone!

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

"""
    CohesivePair

One collocation pair on opposite crack faces (twins) with local history.
"""
mutable struct CohesivePair
    node_plus::Int              # face + (typically eq_type == 2, disp BIE)
    node_minus::Int             # face − (eq_type == 3, traction BIE)
    n̂::SVector{2, Float64}      # normal from − toward + (opening positive)
    hist::CohesiveHistory
    # last evaluated local response
    tn::Float64
    tt::Float64
    kn::Float64
    kt::Float64
end

"""
    CohesiveDBEMProblem

Nonlinear cohesive DBEM problem on a pre-assembled [`DualMesh`](@ref).
"""
mutable struct CohesiveDBEMProblem{L<:AbstractCohesiveLaw}
    mesh::DualMesh
    law::L
    pairs::Vector{CohesivePair}
    kn_pen::Float64                 # contact penalty
    # load
    λ::Float64                      # load factor
    load_dofs::Vector{Int}          # global DOFs with proportional Dirichlet
    load_ū::Vector{Float64}         # prescribed displacement per load_dof at λ=1
    # solver options
    tol::Float64
    maxiter::Int
    verbose::Bool
end

function CohesiveDBEMProblem(
        mesh::DualMesh,
        law::AbstractCohesiveLaw;
        kn_pen = 1e12,
        tol = 1e-6,
        maxiter = 40,
        verbose = false,
    )
    pairs = build_cohesive_pairs(mesh)
    return CohesiveDBEMProblem(
        mesh, law, pairs, float(kn_pen),
        0.0, Int[], Float64[],
        float(tol), Int(maxiter), verbose,
    )
end

"""
    build_cohesive_pairs(mesh) -> Vector{CohesivePair}

Collect unique twin pairs. Face **+** is preferably `eq_type==2` (disp BIE),
face **−** is `eq_type==3` (traction BIE).

# Opening normal
Mesh normals on crack faces are **solid-outward** (into the gap). The cohesive
opening normal `n̂` points **from − toward + through the gap**:

``n̂ = - n_{+}^{out}``  (≈ `n₋^{out}`),

so that under mode-I tension ``δn = n̂ · (u⁺ - u⁻) > 0``.
"""
function build_cohesive_pairs(mesh::DualMesh)
    pairs = CohesivePair[]
    seen = Set{Tuple{Int, Int}}()
    for (i, nd) in enumerate(mesh.nodes)
        tw = nd.twin
        tw == 0 && continue
        a, b = minmax(i, tw)
        (a, b) in seen && continue
        push!(seen, (a, b))
        na, nb = mesh.nodes[a], mesh.nodes[b]
        # prefer eq=2 as +, eq=3 as −
        if na.eq_type == 2 || nb.eq_type == 3
            np, nm = a, b
        elseif nb.eq_type == 2 || na.eq_type == 3
            np, nm = b, a
        else
            np, nm = a, b
        end
        n_plus_out = mesh.nodes[np].normal
        nn = norm(n_plus_out)
        n_plus_out = nn > eps() ? n_plus_out / nn : Point2D(0.0, 1.0)
        n_m = mesh.nodes[nm].normal
        nmn = norm(n_m)
        n_m = nmn > eps() ? n_m / nmn : -n_plus_out
        # Opening normal from − → + through the gap.
        # Solid-outward faces: n₊ ≈ −n₋ and both point into the gap → n̂ = −n₊ ≈ n₋.
        n̂ = -n_plus_out
        if dot(n̂, n_m) < 0
            # prefer alignment with minus solid-outward (also into gap)
            n̂ = n_m
        end
        n̂ = n̂ / (norm(n̂) + eps())
        push!(pairs, CohesivePair(np, nm, n̂, CohesiveHistory(), 0.0, 0.0, 0.0, 0.0))
    end
    return pairs
end

# ---------------------------------------------------------------------------
# Kinematics / constitutive at pairs
# ---------------------------------------------------------------------------

function _pair_opening(prob::CohesiveDBEMProblem, cp::CohesivePair, u::AbstractVector)
    up = SVector(u[2cp.node_plus - 1], u[2cp.node_plus])
    um = SVector(u[2cp.node_minus - 1], u[2cp.node_minus])
    Δu = up - um
    R = local_to_global_R(cp.n̂)
    δ = opening_local(R, Δu)
    return δ[1], δ[2], R, Δu
end

"""Update all pair tractions/stiffnesses from current `u`; return global K blocks info."""
function _update_constitutive!(prob::CohesiveDBEMProblem, u::AbstractVector)
    for cp in prob.pairs
        δn, δt, _, _ = _pair_opening(prob, cp, u)
        tn, tt, kn, kt, st = evaluate_surface!(
            prob.law, δn, δt, cp.hist; kn_pen = prob.kn_pen,
        )
        cp.tn = tn
        cp.tt = tt
        cp.kn = kn
        cp.kt = kt
        cp.hist.state = st
    end
    return nothing
end

"""
Assemble `Kc` mapping displacements → BEM nodal tractions on crack faces.

With opening normal `n̂` (from − to + through the gap):
- `δ = Rᵀ (u⁺ − u⁻)`, `R = [n̂ t̂]`
- cohesive `t_loc = (tn, tt)`
- BEM traction on **+** face (solid-outward basis): `t⁺ = -R t_loc`
  (tension tn>0 pulls + toward −)
- `t⁻ = -t⁺`
"""
function _assemble_Kc(prob::CohesiveDBEMProblem)
    n = length(prob.mesh.nodes)
    Kc = zeros(2n, 2n)
    for cp in prob.pairs
        R = local_to_global_R(cp.n̂)
        Kloc = SMatrix{2, 2, Float64}(cp.kn, 0.0, 0.0, cp.kt)
        # t⁺ = -R K Rᵀ (u⁺ − u⁻)
        Kg = -(R * Kloc * R')
        ip, im = cp.node_plus, cp.node_minus
        rp, rm = 2ip-1:2ip, 2im-1:2im
        Kc[rp, rp] .+= Kg
        Kc[rp, rm] .-= Kg
        Kc[rm, rp] .-= Kg
        Kc[rm, rm] .+= Kg
    end
    return Kc
end

"""Nodal cohesive traction vector `t_coh` of length 2n (global components)."""
function _cohesive_traction_vector(prob::CohesiveDBEMProblem, u::AbstractVector)
    n = length(prob.mesh.nodes)
    t = zeros(2n)
    for cp in prob.pairs
        δn, δt, R, _ = _pair_opening(prob, cp, u)
        tn, tt, kn, kt, _ = evaluate_surface!(
            prob.law, δn, δt, cp.hist; kn_pen = prob.kn_pen,
        )
        cp.tn, cp.tt, cp.kn, cp.kt = tn, tt, kn, kt
        # BEM traction on + face
        tg = -(R * SVector(tn, tt))
        t[2cp.node_plus - 1] += tg[1]
        t[2cp.node_plus] += tg[2]
        t[2cp.node_minus - 1] -= tg[1]
        t[2cp.node_minus] -= tg[2]
    end
    return t
end

# ---------------------------------------------------------------------------
# Linear operators from DualMesh with exterior BCs
# ---------------------------------------------------------------------------

"""
Build free/fixed partitioning for exterior BCs (crack faces free of
prescribed traction — cohesive supplies them).

Returns named tuple with:
- `free`, `fixed` DOF index sets
- `u_fixed` values at λ=1 (scaled by λ later)
- `Hff, Hffix` blocks
- `G` full, and map from nodal traction 2n → element traction 6nel via lumping
"""
function _bc_partition(mesh::DualMesh, load_dofs::Vector{Int}, load_ū::Vector{Float64})
    n = length(mesh.nodes)
    ndof = 2n
    is_dir = falses(ndof)
    u_bar = zeros(ndof)
    # Outer-boundary Dirichlet only. Crack-face nodes stay free (cohesive traction).
    for el in mesh.elements
        el.eq_type == 1 || continue
        for loc in 1:3, dir in 1:2
            j = el.fis[loc]
            dof = 2(j - 1) + dir
            if el.bc_type[loc, dir] == 0
                is_dir[dof] = true
                u_bar[dof] = el.bc_val[loc, dir]
            end
        end
    end
    # force crack-face DOFs free even if element table says Dirichlet
    for (j, nd) in enumerate(mesh.nodes)
        if nd.eq_type in (2, 3)
            is_dir[2j - 1] = false
            is_dir[2j] = false
        end
    end
    # proportional load dofs override
    for (k, dof) in enumerate(load_dofs)
        is_dir[dof] = true
        u_bar[dof] = load_ū[k]
    end
    free = findall(!, is_dir)
    fixed = findall(identity, is_dir)
    return (; is_dir, u_bar, free, fixed)
end

"""
Map nodal tractions `t_nodal` (2n) onto element traction vector (6nel)
by scattering to owning element local nodes (first owner wins, same as dual BC).
"""
function _nodal_to_element_traction(mesh::DualMesh, t_nodal::AbstractVector)
    n = length(mesh.nodes)
    nel = length(mesh.elements)
    t_el = zeros(6nel)
    owner = fill((0, 0), n)
    for (ie, el) in enumerate(mesh.elements)
        for loc in 1:3
            j = el.fis[loc]
            owner[j] == (0, 0) && (owner[j] = (ie, loc))
        end
    end
    for j in 1:n
        ie, loc = owner[j]
        ie == 0 && continue
        for dir in 1:2
            gcol = 6(ie - 1) + 2(loc - 1) + dir
            t_el[gcol] = t_nodal[2(j - 1) + dir]
        end
    end
    return t_el
end

# ---------------------------------------------------------------------------
# Residual and Jacobian
# ---------------------------------------------------------------------------

"""
Residual r(u_free; λ) = (H u − G t_ext − G t_coh(u))[free]
with u[fixed] = λ * u_bar[fixed].
"""
function _residual!(
        r::AbstractVector,
        prob::CohesiveDBEMProblem,
        u_free::AbstractVector,
        λ::Float64,
        part,
    )
    mesh = prob.mesh
    n = length(mesh.nodes)
    u = zeros(2n)
    u[part.free] .= u_free
    u[part.fixed] .= λ .* part.u_bar[part.fixed]

    _update_constitutive!(prob, u)
    t_coh = _cohesive_traction_vector(prob, u)
    t_el = _nodal_to_element_traction(mesh, t_coh)

    # Prescribed Neumann on *outer* boundary only (eq_type==1).
    # Crack faces (eq 2/3) get traction solely from the cohesive law.
    for (ie, el) in enumerate(mesh.elements)
        el.eq_type == 1 || continue
        for loc in 1:3, dir in 1:2
            if el.bc_type[loc, dir] == 1
                gcol = 6(ie - 1) + 2(loc - 1) + dir
                t_el[gcol] += el.bc_val[loc, dir]
            end
        end
    end

    res = mesh.H * u - mesh.G * t_el
    r .= res[part.free]
    return r, u
end

"""
Approximate Jacobian ∂r/∂u_free using cohesive tangent Kc:
  J = H_ff − (G * ∂t_el/∂u)_ff
with ∂t_nodal/∂u ≈ Kc (assembled from local kn, kt).
"""
function _jacobian(prob::CohesiveDBEMProblem, u::AbstractVector, part)
    mesh = prob.mesh
    _update_constitutive!(prob, u)
    Kc = _assemble_Kc(prob)
    # t_el = P * t_nodal, t_nodal ≈ Kc u  (for cohesive part)
    # residual = H u − G P Kc u − G t_ext
    # J_full = H − G P Kc
    n = length(mesh.nodes)
    # build P (6nel × 2n)
    nel = length(mesh.elements)
    P = zeros(6nel, 2n)
    owner = fill((0, 0), n)
    for (ie, el) in enumerate(mesh.elements)
        for loc in 1:3
            j = el.fis[loc]
            owner[j] == (0, 0) && (owner[j] = (ie, loc))
        end
    end
    for j in 1:n
        ie, loc = owner[j]
        ie == 0 && continue
        for dir in 1:2
            gcol = 6(ie - 1) + 2(loc - 1) + dir
            P[gcol, 2(j - 1) + dir] = 1.0
        end
    end
    Jfull = mesh.H - mesh.G * (P * Kc)
    return Jfull[part.free, part.free]
end

# ---------------------------------------------------------------------------
# Newton solver + load stepping / DOF control
# ---------------------------------------------------------------------------

"""
    solve_cohesive_dbem!(prob; nsteps=20, λ_end=1.0,
        control_dof=0, Δu_control=0.0,
        method=:load, Δs=0.0, ψ=0.0)

Solve the nonlinear cohesive DBEM problem.

# Keywords
- `method = :load` — proportional Dirichlet load factor `λ`
- `method = :dof` — single-DOF displacement control (`control_dof`, `Δu_control`)
- `method = :arclength` — spherical arc-length (Crisfield) with step `Δs`
  (auto `Δs` if `Δs≤0`); `ψ` weights load vs displacement (0 = pure disp. arc)
- `nsteps`, `λ_end` — load stepping
- returns `(u_hist, λ_hist)`
"""
function solve_cohesive_dbem!(
        prob::CohesiveDBEMProblem;
        nsteps::Int = 20,
        λ_end::Float64 = 1.0,
        control_dof::Int = 0,
        Δu_control::Float64 = 0.0,
        method::Symbol = :load,
        Δs::Float64 = 0.0,
        ψ::Float64 = 0.0,
    )
    mesh = prob.mesh
    @assert size(mesh.H, 1) > 0 "call assemble_dual! before solve_cohesive_dbem!"
    part = _bc_partition(mesh, prob.load_dofs, prob.load_ū)
    nfree = length(part.free)
    nfree == 0 && error("no free DOFs")

    u_free = fill(1e-12, nfree)
    u_hist = Vector{Vector{Float64}}()
    λ_hist = Float64[]
    n = length(mesh.nodes)

    # legacy alias
    if control_dof > 0 && method === :load
        method = :dof
    end

    if method === :arclength
        return _solve_arclength!(prob, part, u_free, nsteps, Δs, ψ, λ_end, u_hist, λ_hist)
    elseif method === :dof || control_dof > 0
        cd = control_dof
        cd in part.free || error("control_dof=$cd is not free")
        idx_c = findfirst(==(cd), part.free)
        u_target = 0.0
        for step in 1:nsteps
            u_target += Δu_control
            free2 = [d for d in part.free if d != cd]
            ok = _newton_step_control!(
                prob, part, free2, cd, u_target, u_free; idx_c,
            )
            u = zeros(2n)
            u[part.free] .= u_free
            u[part.fixed] .= prob.λ .* part.u_bar[part.fixed]
            u[cd] = u_target
            mesh.u = u
            t_coh = _cohesive_traction_vector(prob, u)
            mesh.t_el = _nodal_to_element_traction(mesh, t_coh)
            push!(u_hist, copy(u))
            push!(λ_hist, prob.λ)
            prob.verbose && @info "DOF-control step" step u_target ok
        end
    else
        dλ = λ_end / nsteps
        for step in 1:nsteps
            λ = step * dλ
            prob.λ = λ
            conv = _newton_step!(prob, part, u_free, λ)
            u = zeros(2n)
            u[part.free] .= u_free
            u[part.fixed] .= λ .* part.u_bar[part.fixed]
            mesh.u = u
            t_coh = _cohesive_traction_vector(prob, u)
            mesh.t_el = _nodal_to_element_traction(mesh, t_coh)
            push!(u_hist, copy(u))
            push!(λ_hist, λ)
            prob.verbose && @info "load step" step λ conv
            conv || @warn "Newton did not converge at step $step (λ=$λ)"
        end
    end
    return u_hist, λ_hist
end

# -------------------- spherical arc-length (Crisfield) ---------------------

function _solve_arclength!(
        prob, part, u_free, nsteps, Δs, ψ, λ_end, u_hist, λ_hist,
    )
    mesh = prob.mesh
    n = length(mesh.nodes)
    nf = length(u_free)
    λ = 0.0
    prob.λ = λ
    r = zeros(nf)
    r0 = zeros(nf)
    r1 = zeros(nf)

    function full_u(uf, lam)
        u = zeros(2n)
        u[part.free] .= uf
        u[part.fixed] .= lam .* part.u_bar[part.fixed]
        return u
    end
    function drdλ_at(uf, lam)
        _residual!(r1, prob, uf, lam + 1e-8, part)
        _residual!(r0, prob, uf, lam, part)
        return (r1 .- r0) ./ 1e-8
    end
    function regularize!(J)
        @inbounds for i in 1:size(J, 1)
            J[i, i] += 1e-14 * (abs(J[i, i]) + 1)
        end
        return J
    end

    u = full_u(u_free, λ)
    J = regularize!(_jacobian(prob, u, part))
    du_pred = J \ (-drdλ_at(u_free, λ))
    s_auto = Δs > 0 ? Δs : max(0.05 * norm(du_pred), 1e-6)
    sign_dir = 1.0

    for step in 1:nsteps
        # --- predictor ---
        if ψ > 0
            nrm = max(sqrt(dot(du_pred, du_pred) + ψ^2), 1e-30)
            Δu = sign_dir * s_auto * du_pred / nrm
            Δλ = sign_dir * s_auto * ψ / nrm
        else
            nrm = max(norm(du_pred), 1e-30)
            Δu = sign_dir * s_auto * du_pred / nrm
            Δλ = sign_dir * s_auto / nrm
        end
        u0 = copy(u_free)
        λ0 = λ
        u_free .+= Δu
        λ += Δλ

        # --- corrector ---
        conv = false
        for it in 1:prob.maxiter
            _, u = _residual!(r, prob, u_free, λ, part)
            nr = norm(r)
            du_s = u_free .- u0
            dλ_s = λ - λ0
            g = dot(du_s, du_s) + ψ^2 * dλ_s^2 - s_auto^2
            if (nr / max(norm(u_free), 1.0) < prob.tol) &&
               (abs(g) < max(prob.tol, 1e-8) * max(s_auto^2, 1e-16))
                conv = true
                break
            end
            J = regularize!(_jacobian(prob, u, part))
            fλ = drdλ_at(u_free, λ)
            a = J \ (-r)
            b = J \ (-fλ)
            denom = 2 * dot(du_s, b) + 2 * ψ^2 * dλ_s
            δλ = abs(denom) < 1e-30 ? 0.0 : (-g - 2 * dot(du_s, a)) / denom
            u_free .+= a .+ δλ .* b
            λ += δλ
        end

        u = full_u(u_free, λ)
        J = regularize!(_jacobian(prob, u, part))
        du_new = J \ (-drdλ_at(u_free, λ))
        if dot(du_new, u_free .- u0) + ψ^2 * (λ - λ0) < 0
            sign_dir = -sign_dir
        end
        du_pred = du_new

        prob.λ = λ
        mesh.u = u
        mesh.t_el = _nodal_to_element_traction(mesh, _cohesive_traction_vector(prob, u))
        push!(u_hist, copy(u))
        push!(λ_hist, λ)
        prob.verbose && @info "arclength step" step λ conv s = s_auto

        if λ_end > 0 && λ >= λ_end
            break
        elseif λ_end < 0 && λ <= λ_end
            break
        end
    end
    return u_hist, λ_hist
end

function _newton_step!(
        prob::CohesiveDBEMProblem,
        part,
        u_free::Vector{Float64},
        λ::Float64,
    )
    r = zeros(length(u_free))
    for it in 1:prob.maxiter
        _, u = _residual!(r, prob, u_free, λ, part)
        nr = norm(r)
        nref = max(norm(u_free), 1.0)
        if nr / nref < prob.tol || nr < prob.tol
            return true
        end
        J = _jacobian(prob, u, part)
        # regularize
        for i in 1:size(J, 1)
            J[i, i] += 1e-14 * (abs(J[i, i]) + 1.0)
        end
        du = J \ (-r)
        # line search
        α = 1.0
        r2 = similar(r)
        u_trial = copy(u_free)
        for ls in 1:8
            u_trial .= u_free .+ α .* du
            _residual!(r2, prob, u_trial, λ, part)
            if norm(r2) < (1 - 1e-4 * α) * nr
                break
            end
            α *= 0.5
        end
        u_free .= u_trial
    end
    _, u = _residual!(r, prob, u_free, λ, part)
    return norm(r) / max(norm(u_free), 1.0) < 10 * prob.tol
end

function _newton_step_control!(
        prob, part, free2, cd, u_cd, u_free_full; idx_c,
    )
    # Rebuild partition treating cd as fixed
    n = length(prob.mesh.nodes)
    is_dir = copy(part.is_dir)
    is_dir[cd] = true
    u_bar = copy(part.u_bar)
    u_bar[cd] = u_cd
    free = findall(!, is_dir)
    fixed = findall(identity, is_dir)
    part2 = (; is_dir, u_bar, free, fixed)
    # extract free unknowns (without control)
    u2 = zeros(length(free))
    # map from full free vector
    for (k, d) in enumerate(free)
        j = findfirst(==(d), part.free)
        j !== nothing && (u2[k] = u_free_full[j])
    end
    λ = 1.0
    prob.λ = λ
    ok = _newton_step!(prob, part2, u2, λ)
    # write back
    u_free_full .= 0
    for (k, d) in enumerate(part.free)
        if d == cd
            u_free_full[k] = u_cd
        else
            j = findfirst(==(d), free)
            j !== nothing && (u_free_full[k] = u2[j])
        end
    end
    return ok
end

# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

"""Local openings (δn, δt) for every cohesive pair."""
function cohesive_openings(prob::CohesiveDBEMProblem)
    u = prob.mesh.u
    return [_pair_opening(prob, cp, u)[1:2] for cp in prob.pairs]
end

"""Local tractions (tn, tt) for every cohesive pair."""
function cohesive_tractions(prob::CohesiveDBEMProblem)
    return [(cp.tn, cp.tt) for cp in prob.pairs]
end

# ---------------------------------------------------------------------------
# Patch-test meshes (Cordeiro §5.1 mode I / II)
# ---------------------------------------------------------------------------

"""
    modeI_patch_mesh(; L=0.1, E=32e9, ν=0.2, n_coh=4, n_side=3) -> DualMesh

Square plate with a horizontal cohesive interface at mid-height (mode I).
Bottom fixed, top Dirichlet uy.
"""
function modeI_patch_mesh(; L = 0.1, E = 32e9, ν = 0.2, n_coh = 4, n_side = 3, plane_strain = true)
    geo = Point2D[]
    nodes = DualNode[]
    elems = DualElement[]
    # Helper to push a linear edge as one discontinuous quadratic element
    # with collocation at ±2/3,0 mapped to the segment.
    function add_edge!(p1, p2, eq, bc_type, bc_val; crack_face = 0)
        g0 = length(geo)
        mid = 0.5 * (p1 + p2)
        push!(geo, p1); push!(geo, mid); push!(geo, p2)
        # physical nodes
        n0 = length(nodes)
        t̂ = (p2 - p1) / (norm(p2 - p1) + eps())
        n̂ = Point2D(t̂[2], -t̂[1])   # right-hand normal
        for (k, ξ) in enumerate((-2 / 3, 0.0, 2 / 3))
            N = N_cont(ξ)
            x = N[1] * p1 + N[2] * mid + N[3] * p2
            push!(nodes, DualNode(n0 + k, Point2D(x), eq, n̂, 0))
        end
        bt = Matrix{Int}(undef, 3, 2)
        bv = Matrix{Float64}(undef, 3, 2)
        for loc in 1:3, dir in 1:2
            bt[loc, dir] = bc_type[dir]
            bv[loc, dir] = bc_val[dir]
        end
        push!(elems, DualElement(length(elems) + 1,
            (g0 + 1, g0 + 2, g0 + 3),
            (n0 + 1, n0 + 2, n0 + 3),
            eq, bt, bv))
        return length(elems)
    end

    h = L / 2
    # outer boundary (eq=1): bottom, right lower, right upper, top, left upper, left lower
    # bottom y=0
    xs = range(0, L; length = n_side + 1)
    for i in 1:n_side
        add_edge!(Point2D(xs[i], 0.0), Point2D(xs[i + 1], 0.0), 1, (0, 0), (0.0, 0.0))
    end
    # right x=L lower + upper
    ys = range(0, h; length = max(n_side ÷ 2, 2))
    for i in 1:length(ys)-1
        add_edge!(Point2D(L, ys[i]), Point2D(L, ys[i + 1]), 1, (1, 1), (0.0, 0.0))
    end
    ys2 = range(h, L; length = max(n_side ÷ 2, 2))
    for i in 1:length(ys2)-1
        add_edge!(Point2D(L, ys2[i]), Point2D(L, ys2[i + 1]), 1, (1, 1), (0.0, 0.0))
    end
    # top y=L (Dirichlet uy will be set as load)
    xs_top = range(L, 0; length = n_side + 1)
    top_elems = Int[]
    for i in 1:n_side
        e = add_edge!(Point2D(xs_top[i], L), Point2D(xs_top[i + 1], L), 1, (1, 0), (0.0, 0.0))
        push!(top_elems, e)
    end
    # left x=0 upper + lower
    ys3 = range(L, h; length = max(n_side ÷ 2, 2))
    for i in 1:length(ys3)-1
        add_edge!(Point2D(0.0, ys3[i]), Point2D(0.0, ys3[i + 1]), 1, (1, 1), (0.0, 0.0))
    end
    ys4 = range(h, 0; length = max(n_side ÷ 2, 2))
    for i in 1:length(ys4)-1
        add_edge!(Point2D(0.0, ys4[i]), Point2D(0.0, ys4[i + 1]), 1, (1, 1), (0.0, 0.0))
    end

    # Cohesive interface at y=h.
    # Normals = solid-outward (into the gap), same convention as Gmsh dual cracks:
    #   face + (upper body bottom, eq=2): n = (0, -1)
    #   face − (lower body top,    eq=3): n = (0, +1)
    xs_c = range(0, L; length = n_coh + 1)
    face_a = Int[]
    face_b = Int[]
    nodes_plus = Int[]
    nodes_minus = Int[]
    for i in 1:n_coh
        # upper face: left → right, solid above → outward into gap is down
        e = add_edge!(Point2D(xs_c[i], h), Point2D(xs_c[i + 1], h), 2, (1, 1), (0.0, 0.0))
        el = elems[e]
        for loc in 1:3
            nodes[el.fis[loc]].normal = Point2D(0.0, -1.0)
            nodes[el.fis[loc]].eq_type = 2
            push!(nodes_plus, el.fis[loc])
        end
        push!(face_a, e)
    end
    for i in n_coh:-1:1
        # lower face: right → left, solid below → outward into gap is up
        e = add_edge!(Point2D(xs_c[i + 1], h), Point2D(xs_c[i], h), 3, (1, 1), (0.0, 0.0))
        el = elems[e]
        for loc in 1:3
            nodes[el.fis[loc]].normal = Point2D(0.0, 1.0)
            nodes[el.fis[loc]].eq_type = 3
            push!(nodes_minus, el.fis[loc])
        end
        push!(face_b, e)
    end
    # twin pairing by x-coordinate
    for ip in nodes_plus
        xp = nodes[ip].pos
        best, dmin = 0, Inf
        for im in nodes_minus
            d = abs(nodes[im].pos[1] - xp[1])
            if d < dmin
                dmin, best = d, im
            end
        end
        if best > 0
            nodes[ip].twin = best
            nodes[best].twin = ip
        end
    end

    mesh = DualMesh(geo, nodes, elems; E = E, ν = ν, plane_strain = plane_strain)
    mesh.crack_face_a = face_a
    mesh.crack_face_b = face_b
    return mesh, top_elems
end

"""
    modeII_patch_mesh(; ...) -> DualMesh

Square with horizontal cohesive interface, shear loading (mode II).
"""
function modeII_patch_mesh(; kwargs...)
    mesh, top = modeI_patch_mesh(; kwargs...)
    # change top BC: free uy, prescribe ux (shear) — done via load_dofs in problem setup
    return mesh, top
end

# ---------------------------------------------------------------------------
# Contact compression patch + process-zone growth
# ---------------------------------------------------------------------------

"""
    contact_compression_mesh(; L=0.1, gap0=0.0, ...) -> (DualMesh, top_elems)

Mode-I geometry with a closed (or nearly closed) cohesive interface for
**compression / contact** tests. Top face Dirichlet `uy < 0` drives contact.
"""
function contact_compression_mesh(; gap0 = 0.0, kwargs...)
    mesh, top = modeI_patch_mesh(; kwargs...)
    if gap0 != 0
        for elid in mesh.crack_face_a
            el = mesh.elements[elid]
            for j in el.fis
                p = mesh.nodes[j].pos
                mesh.nodes[j].pos = Point2D(p[1], p[2] + gap0 / 2)
            end
        end
        for elid in mesh.crack_face_b
            el = mesh.elements[elid]
            for j in el.fis
                p = mesh.nodes[j].pos
                mesh.nodes[j].pos = Point2D(p[1], p[2] - gap0 / 2)
            end
        end
    end
    return mesh, top
end

"""
    extend_cohesive_process_zone!(mesh, tip_node, da; n_new=2) -> new_face_a_ids

Extend both crack faces by length `da` ahead of `tip_node`, adding `n_new`
discontinuous cohesive elements per face and twin-pairing the new nodes.
Call `assemble_dual!` again after extension.
"""
function extend_cohesive_process_zone!(
        mesh::DualMesh,
        tip_node::Int,
        da::Float64;
        n_new::Int = 2,
    )
    tip = mesh.nodes[tip_node]
    faceA_nodes = Int[]
    for elid in mesh.crack_face_a
        append!(faceA_nodes, mesh.elements[elid].fis)
    end
    unique!(faceA_nodes)
    centroid = sum(mesh.nodes[i].pos for i in faceA_nodes) / max(length(faceA_nodes), 1)
    d = tip.pos - centroid
    if norm(d) < 1e-14
        d = tip.normal
    end
    d = d / (norm(d) + eps())
    n̂_plus = tip.normal / (norm(tip.normal) + eps())

    xs = [tip.pos + (k / n_new) * da * d for k in 0:n_new]
    new_a = Int[]
    nodes_plus = Int[]
    nodes_minus = Int[]

    for i in 1:n_new
        p1, p2 = xs[i], xs[i + 1]
        mid = 0.5 * (p1 + p2)
        g0 = length(mesh.geo_nodes)
        push!(mesh.geo_nodes, p1, mid, p2)
        n0 = length(mesh.nodes)
        for ξ in (-2 / 3, 0.0, 2 / 3)
            N = N_cont(ξ)
            x = N[1] * p1 + N[2] * mid + N[3] * p2
            push!(mesh.nodes, DualNode(length(mesh.nodes) + 1, Point2D(x), 2, n̂_plus, 0))
            push!(nodes_plus, length(mesh.nodes))
        end
        bt = ones(Int, 3, 2)
        bv = zeros(3, 2)
        el = DualElement(
            length(mesh.elements) + 1,
            (g0 + 1, g0 + 2, g0 + 3),
            (n0 + 1, n0 + 2, n0 + 3),
            2, bt, bv,
        )
        push!(mesh.elements, el)
        push!(mesh.crack_face_a, el.id)
        push!(new_a, el.id)

        g0 = length(mesh.geo_nodes)
        push!(mesh.geo_nodes, p2, mid, p1)
        n0 = length(mesh.nodes)
        n̂_minus = -n̂_plus
        for ξ in (-2 / 3, 0.0, 2 / 3)
            N = N_cont(ξ)
            x = N[1] * p2 + N[2] * mid + N[3] * p1
            push!(mesh.nodes, DualNode(length(mesh.nodes) + 1, Point2D(x), 3, n̂_minus, 0))
            push!(nodes_minus, length(mesh.nodes))
        end
        el = DualElement(
            length(mesh.elements) + 1,
            (g0 + 1, g0 + 2, g0 + 3),
            (n0 + 1, n0 + 2, n0 + 3),
            3, bt, bv,
        )
        push!(mesh.elements, el)
        push!(mesh.crack_face_b, el.id)
    end

    for ip in nodes_plus
        xp = mesh.nodes[ip].pos
        best, dmin = 0, Inf
        for im in nodes_minus
            dd = norm(mesh.nodes[im].pos - xp)
            dd < dmin && ((dmin, best) = (dd, im))
        end
        if best > 0
            mesh.nodes[ip].twin = best
            mesh.nodes[best].twin = ip
        end
    end

    nn = length(mesh.nodes)
    nel = length(mesh.elements)
    mesh.H = zeros(2nn, 2nn)
    mesh.G = zeros(2nn, 6nel)
    mesh.u = zeros(2nn)
    mesh.t_el = zeros(6nel)
    mesh.tip_nodes = [nodes_plus[end]]
    return new_a
end
