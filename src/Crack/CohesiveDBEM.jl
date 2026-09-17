# =============================================================================
# Nonlinear cohesive-contact Dual BEM (Cordeiro et al. 2024)
# =============================================================================
# Builds on BEMdata dual assembly ([`assemble_dual!`](@ref)). Crack face pairs
# (twins) carry a cohesive law with four surface conditions:
#   (i) contact  (ii) softening  (iii) unloading/reloading  (iv) complete failure
#
# Discrete system (compact form of Cordeiro eq. 24):
#   A(u) u = f(λ)
# where cohesive stiffness K(δ) couples tractions on ± faces to the opening
# Δu = u⁺ − u⁻ through the already-assembled nodal H, G operators.
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

Nonlinear cohesive DBEM problem on a pre-assembled elasticity [`BEMdata`](@ref).
"""
mutable struct CohesiveDBEMProblem{L<:AbstractCohesiveLaw}
    mesh::BEMdata{<:Elasticity}
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
        mesh::BEMdata{<:Elasticity},
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
function build_cohesive_pairs(mesh::BEMdata{<:Elasticity})
    parentmodule(@__MODULE__).has_cache(mesh, :twin) || prepare_crack!(mesh)
    pairs = CohesivePair[]
    seen = Set{Tuple{Int, Int}}()
    twin = mesh.twin
    eq = mesh.eq_type
    for i in 1:mesh.n
        tw = twin[i]
        tw == 0 && continue
        a, b = minmax(i, tw)
        (a, b) in seen && continue
        push!(seen, (a, b))
        ea, eb = eq[a], eq[b]
        if ea == 2 || eb == 3
            np, nm = a, b
        elseif eb == 2 || ea == 3
            np, nm = b, a
        else
            np, nm = a, b
        end
        n_plus_out = mesh.Normal[np]
        nn = norm(n_plus_out)
        n_plus_out = nn > eps() ? n_plus_out / nn : Point2D(0.0, 1.0)
        n_m = mesh.Normal[nm]
        nmn = norm(n_m)
        n_m = nmn > eps() ? n_m / nmn : -n_plus_out
        # Opening normal from − → + through the gap.
        # Solid-outward faces: n₊ ≈ −n₋ and both point into the gap → n̂ = −n₊ ≈ n₋.
        n̂ = -n_plus_out
        if dot(n̂, n_m) < 0
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
    n = prob.mesh.n
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
    n = prob.mesh.n
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
# Linear operators from BEMdata with exterior BCs
# ---------------------------------------------------------------------------

"""
Build free/fixed partitioning for exterior BCs (crack faces free of
prescribed traction — cohesive supplies them).

Returns named tuple with `is_dir`, `u_bar`, `free`, `fixed`.
"""
function _bc_partition(mesh::BEMdata{<:Elasticity}, load_dofs::Vector{Int},
        load_ū::Vector{Float64})
    n = mesh.n
    ndof = 2n
    is_dir = falses(ndof)
    u_bar = zeros(ndof)
    eq = mesh.eq_type
    @inbounds for i in 1:n
        if eq[i] in (2, 3)
            continue
        end
        for dir in 1:2
            dof = 2(i - 1) + dir
            if mesh.BC[dof] == 0
                is_dir[dof] = true
                u_bar[dof] = mesh.BV[dof]
            end
        end
    end
    for (k, dof) in enumerate(load_dofs)
        is_dir[dof] = true
        u_bar[dof] = load_ū[k]
    end
    free = findall(!, is_dir)
    fixed = findall(identity, is_dir)
    return (; is_dir, u_bar, free, fixed)
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
    n = mesh.n
    u = zeros(2n)
    u[part.free] .= u_free
    u[part.fixed] .= λ .* part.u_bar[part.fixed]

    _update_constitutive!(prob, u)
    t = _cohesive_traction_vector(prob, u)
    eq = mesh.eq_type
    @inbounds for i in 1:n
        eq[i] == 1 || continue
        for dir in 1:2
            dof = 2(i - 1) + dir
            if mesh.BC[dof] == 1
                t[dof] += mesh.BV[dof]
            end
        end
    end

    res = mesh.H * u - mesh.G * t
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
    Jfull = mesh.H - mesh.G * Kc
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
    n = mesh.n

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
            mesh.traction = t_coh
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
            mesh.traction = t_coh
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
    n = mesh.n
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
        mesh.traction = _cohesive_traction_vector(prob, u)
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
    n = prob.mesh.n
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
    modeI_patch_mesh(; L=0.1, E=32e9, ν=0.2, n_coh=4, n_side=3) -> (BEMdata, top_elems)

Square plate with a horizontal cohesive interface at mid-height (mode I).
Bottom fixed, top Dirichlet uy (set by the caller via `load_dofs`).
"""
function modeI_patch_mesh(; L=0.1, E=32e9, ν=0.2, n_coh=4, n_side=3,
        plane_strain=true, ordem=2)
    B = parentmodule(@__MODULE__)
    p = Int(ordem)
    qsi, wi = B.discontinuous_nodes_weights(p)
    n_per = p + 1
    poly_geo = B.Equispaced(p)
    Ngeo, dNgeo = B.shapefun(poly_geo, qsi)

    nodes = Point2D[]
    normals = Point2D[]
    elems = B.Element[]
    BC = Int[]
    BV = Float64[]
    eq_acc = Int[]

    function add_edge!(p1, p2, eq, bc_type, bc_val)
        X = [p1 + ((k - 1) / p) * (p2 - p1) for k in 1:n_per]
        NOS = Ngeo * X
        dx = dNgeo * X
        J = norm.(dx)
        nrm = B.tan2normal.(dx ./ J)
        Lseg = abs(dot(J, wi))
        n0 = length(nodes)
        idx = collect((n0 + 1):(n0 + n_per))
        append!(nodes, Point2D.(NOS))
        append!(normals, nrm)
        append!(eq_acc, fill(eq, n_per))
        for _ in 1:n_per
            append!(BC, Int[bc_type[1], bc_type[2]])
            append!(BV, Float64[bc_val[1], bc_val[2]])
        end
        push!(elems, B.Element(idx, collect(Float64, J), Float64(Lseg), 0))
        return length(elems)
    end

    h = L / 2
    xs = range(0, L; length=n_side + 1)
    for i in 1:n_side
        add_edge!(Point2D(xs[i], 0.0), Point2D(xs[i + 1], 0.0), 1, (0, 0), (0.0, 0.0))
    end
    ys = range(0, h; length=max(n_side ÷ 2, 2))
    for i in 1:length(ys)-1
        add_edge!(Point2D(L, ys[i]), Point2D(L, ys[i + 1]), 1, (1, 1), (0.0, 0.0))
    end
    ys2 = range(h, L; length=max(n_side ÷ 2, 2))
    for i in 1:length(ys2)-1
        add_edge!(Point2D(L, ys2[i]), Point2D(L, ys2[i + 1]), 1, (1, 1), (0.0, 0.0))
    end
    xs_top = range(L, 0; length=n_side + 1)
    top_elems = Int[]
    for i in 1:n_side
        e = add_edge!(Point2D(xs_top[i], L), Point2D(xs_top[i + 1], L), 1, (1, 0), (0.0, 0.0))
        push!(top_elems, e)
    end
    ys3 = range(L, h; length=max(n_side ÷ 2, 2))
    for i in 1:length(ys3)-1
        add_edge!(Point2D(0.0, ys3[i]), Point2D(0.0, ys3[i + 1]), 1, (1, 1), (0.0, 0.0))
    end
    ys4 = range(h, 0; length=max(n_side ÷ 2, 2))
    for i in 1:length(ys4)-1
        add_edge!(Point2D(0.0, ys4[i]), Point2D(0.0, ys4[i + 1]), 1, (1, 1), (0.0, 0.0))
    end

    xs_c = range(0, L; length=n_coh + 1)
    face_a = Int[]
    face_b = Int[]
    nodes_plus = Int[]
    nodes_minus = Int[]
    for i in 1:n_coh
        e = add_edge!(Point2D(xs_c[i], h), Point2D(xs_c[i + 1], h), 2, (1, 1), (0.0, 0.0))
        el = elems[e]
        for j in el.index
            normals[j] = Point2D(0.0, -1.0)
            eq_acc[j] = 2
            push!(nodes_plus, j)
        end
        push!(face_a, e)
    end
    for i in n_coh:-1:1
        e = add_edge!(Point2D(xs_c[i + 1], h), Point2D(xs_c[i], h), 3, (1, 1), (0.0, 0.0))
        el = elems[e]
        for j in el.index
            normals[j] = Point2D(0.0, 1.0)
            eq_acc[j] = 3
            push!(nodes_minus, j)
        end
        push!(face_b, e)
    end

    n = length(nodes)
    twin = zeros(Int, n)
    for ip in nodes_plus
        xp = nodes[ip]
        best, dmin = 0, Inf
        for im in nodes_minus
            d = abs(nodes[im][1] - xp[1])
            if d < dmin
                dmin, best = d, im
            end
        end
        if best > 0
            twin[ip] = best
            twin[best] = ip
        end
    end

    dad = B.BEMdata(
        "modeI_patch",
        2,
        elems,
        B.Legendre(p),
        SVector{n_per}(Float64.(wi)),
        nodes,
        normals,
        B.Elasticity(E, ν, 1.0; plane_strain=plane_strain),
        BC,
        BV,
        n,
        0,
        n,
        B.BEMCache(),
    )
    B.set_cache!(dad; eq_type=eq_acc, twin=twin,
        crack_face_a=face_a, crack_face_b=face_b, crack_bc=:traction_free,
        tip_nodes=isempty(nodes_plus) ? Int[] : [nodes_plus[1], nodes_plus[end]])
    return dad, top_elems
end

"""
    modeII_patch_mesh(; ...) -> (BEMdata, top_elems)

Square with horizontal cohesive interface, shear loading (mode II).
"""
function modeII_patch_mesh(; kwargs...)
    mesh, top = modeI_patch_mesh(; kwargs...)
    return mesh, top
end

# ---------------------------------------------------------------------------
# Contact compression patch + process-zone growth
# ---------------------------------------------------------------------------

"""
    contact_compression_mesh(; L=0.1, gap0=0.0, ...) -> (BEMdata, top_elems)

Mode-I geometry with a closed (or nearly closed) cohesive interface for
**compression / contact** tests. Top face Dirichlet `uy < 0` drives contact.
"""
function contact_compression_mesh(; gap0=0.0, kwargs...)
    mesh, top = modeI_patch_mesh(; kwargs...)
    if gap0 != 0
        coll = getfield(mesh, :collocation)
        for elid in mesh.crack_face_a
            for j in mesh.elements[elid].index
                p = coll[j]
                coll[j] = Point2D(p[1], p[2] + gap0 / 2)
            end
        end
        for elid in mesh.crack_face_b
            for j in mesh.elements[elid].index
                p = coll[j]
                coll[j] = Point2D(p[1], p[2] - gap0 / 2)
            end
        end
    end
    return mesh, top
end

"""
    extend_cohesive_process_zone!(dad, tip_node, da; n_new=2) -> new_face_a_ids

Extend both crack faces by length `da` ahead of `tip_node`, adding `n_new`
discontinuous cohesive elements per face and twin-pairing the new nodes.
Call `assemble_dual!` again after extension. Requires `dad.ni == 0`.
"""
function extend_cohesive_process_zone!(
        mesh::BEMdata{<:Elasticity},
        tip_node::Int,
        da::Float64;
        n_new::Int=2,
    )
    B = parentmodule(@__MODULE__)
    mesh.ni == 0 || error("extend_cohesive_process_zone!: no internal nodes")
    p = length(mesh.elements[1]) - 1
    n_per = p + 1
    qsi, wi = B.discontinuous_nodes_weights(p)
    poly_geo = B.Equispaced(p)
    Ngeo, dNgeo = B.shapefun(poly_geo, qsi)

    faceA_nodes = Int[]
    for elid in mesh.crack_face_a
        append!(faceA_nodes, mesh.elements[elid].index)
    end
    unique!(faceA_nodes)
    tip_pos = mesh.Nodes[tip_node]
    centroid = sum(mesh.Nodes[i] for i in faceA_nodes) / max(length(faceA_nodes), 1)
    d = tip_pos - centroid
    if norm(d) < 1e-14
        d = mesh.Normal[tip_node]
    end
    d = d / (norm(d) + eps())
    n̂_plus = mesh.Normal[tip_node]
    n̂_plus = n̂_plus / (norm(n̂_plus) + eps())
    n̂_minus = -n̂_plus

    xs = [tip_pos + (k / n_new) * da * d for k in 0:n_new]
    new_a = Int[]
    nodes_plus = Int[]
    nodes_minus = Int[]
    eq_type = copy(mesh.eq_type)
    twin = copy(mesh.twin)
    face_a = copy(mesh.crack_face_a)
    face_b = copy(mesh.crack_face_b)
    coll = getfield(mesh, :collocation)

    function add_seg!(p1, p2, eq, nfix)
        X = [p1 + ((k - 1) / p) * (p2 - p1) for k in 1:n_per]
        NOS = Ngeo * X
        dx = dNgeo * X
        J = norm.(dx)
        Lseg = abs(dot(J, wi))
        n0 = mesh.n
        idx = collect((n0 + 1):(n0 + n_per))
        append!(coll, Point2D.(NOS))
        append!(mesh.Normal, fill(nfix, n_per))
        for _ in 1:n_per
            append!(mesh.BC, [1, 1])
            append!(mesh.BV, [0.0, 0.0])
        end
        append!(eq_type, fill(eq, n_per))
        append!(twin, zeros(Int, n_per))
        mesh.n = n0 + n_per
        mesh.nt = mesh.n
        push!(mesh.elements, B.Element(idx, collect(Float64, J), Float64(Lseg), 0))
        return idx, length(mesh.elements)
    end

    for i in 1:n_new
        p1, p2 = xs[i], xs[i + 1]
        idx, ie = add_seg!(p1, p2, 2, n̂_plus)
        append!(nodes_plus, idx)
        push!(face_a, ie)
        push!(new_a, ie)
        idx, ie = add_seg!(p2, p1, 3, n̂_minus)
        append!(nodes_minus, idx)
        push!(face_b, ie)
    end

    for ip in nodes_plus
        xp = coll[ip]
        best, dmin = 0, Inf
        for im in nodes_minus
            dd = norm(coll[im] - xp)
            dd < dmin && ((dmin, best) = (dd, im))
        end
        if best > 0
            twin[ip] = best
            twin[best] = ip
        end
    end

    B.set_cache!(mesh; eq_type=eq_type, twin=twin,
        crack_face_a=face_a, crack_face_b=face_b,
        tip_nodes=[nodes_plus[end]],
        H=nothing, G=nothing, A=nothing, B=nothing, b=nothing, u=nothing)
    return new_a
end

