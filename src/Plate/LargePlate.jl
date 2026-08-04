# Large-deflection (von Kármán) thin plates — placa_grande / placa_large.jl
#
# Couples ThinPlate Kirchhoff BEM with plane-stress membrane elasticity BEM.
# Nonlinear residual solved by NonlinearSolve + ForwardDiff (AutoForwardDiff).

export LargePlateProblem, solve_large_plate!, build_large_plate_problem
export membrane_stiffness_CB, extract_w_field, geometric_load
export linear_wmax_reference

# =============================================================================
# Problem container
# =============================================================================

"""Precomputed operators for large-deflection plate analysis."""
mutable struct LargePlateProblem
    plate::Any
    dad_pe::BEMdata{<:Elasticity}
    A_pl::Matrix{Float64}
    b_bc::Vector{Float64}
    q_load::Vector{Float64}
    is_kin::BitVector
    known::Vector{Float64}
    A_pe::Matrix{Float64}
    b_pe::Vector{Float64}
    M_pe::Matrix{Float64}
    pts::Vector{SVector{2,Float64}}
    Fx::Matrix{Float64}
    Fy::Matrix{Float64}
    w_index::Vector{Int}
    λ_hist::Vector{Float64}
    w_hist::Matrix{Float64}
    w_center_hist::Vector{Float64}
end

membrane_stiffness_CB(E, ν, h) = E * h / (1 - ν^2)

function build_large_plate_problem(plate, dad_pe::BEMdata{<:Elasticity};
    npg_plate=10, npg_pe=10)

    if isempty(plate.H)
        assemble_plate!(plate; npg=npg_plate)
    end
    A_pl, b_full, is_kin, known = apply_bc_plate(plate)
    b_bc = b_full .- plate.q
    q_load = copy(plate.q)

    has_cache(dad_pe, :H) || H_G_full_direct(dad_pe; npg=npg_pe, threaded=false)
    applyBC(dad_pe)
    A_pe = copy(dad_pe.A)
    b_pe = copy(dad_pe.b)
    has_cache(dad_pe, :M) || dibem_elasticity!(dad_pe; npg=npg_pe)
    M_pe = Matrix{Float64}(dad_pe.M)

    n = length(plate.nodes)
    ni = length(plate.internal)
    nc = length(plate.corners)
    pts = SVector{2,Float64}[]
    w_index = Int[]
    for i in 1:n
        push!(pts, plate.nodes[i].pos)
        push!(w_index, 2i - 1)
    end
    for k in 1:ni
        push!(pts, plate.internal[k])
        push!(w_index, 2n + k)
    end
    for c in 1:nc
        push!(pts, plate.corners[c].pos)
        push!(w_index, 2n + ni + c)
    end

    ops = rbf_gradient_ops(pts; rbf=PHS(3; poly_deg=1))
    return LargePlateProblem(plate, dad_pe, A_pl, b_bc, q_load, is_kin, known,
        A_pe, b_pe, M_pe, pts, ops.Fx, ops.Fy, w_index,
        Float64[], zeros(0, 0), Float64[])
end

# =============================================================================
# Dual-safe field helpers (ForwardDiff-friendly, no mutation of BEMdata)
# =============================================================================

function extract_w_field(prob::LargePlateProblem, u::AbstractVector)
    T = eltype(u)
    w = Vector{T}(undef, length(prob.w_index))
    @inbounds for k in eachindex(prob.w_index)
        w[k] = u[prob.w_index[k]]
    end
    return w
end

function pack_plate_u(prob::LargePlateProblem, x::AbstractVector)
    T = eltype(x)
    u = Vector{T}(undef, length(prob.is_kin))
    @inbounds for dof in eachindex(prob.is_kin)
        u[dof] = prob.is_kin[dof] ? T(prob.known[dof]) : x[dof]
    end
    return u
end

function membrane_N_from_w(prob::LargePlateProblem, w::AbstractVector)
    E = prob.dad_pe.properties.E
    ν = prob.dad_pe.properties.nu
    h = prob.plate.props.h
    CB = membrane_stiffness_CB(E, ν, h)
    wx = prob.Fx * w
    wy = prob.Fy * w
    ex = 0.5 .* wx .^ 2
    ey = 0.5 .* wy .^ 2
    gxy = wx .* wy
    Nxx = CB .* (ex .+ ν .* ey)
    Nyy = CB .* (ey .+ ν .* ex)
    Nxy = CB .* ((1 - ν) / 2) .* gxy
    return Nxx, Nyy, Nxy
end

function geometric_load(prob::LargePlateProblem, w::AbstractVector, Nxx, Nyy, Nxy)
    wxx = prob.Fx * (prob.Fx * w)
    wyy = prob.Fy * (prob.Fy * w)
    wxy = prob.Fx * (prob.Fy * w)
    return Nxx .* wxx .+ 2 .* Nxy .* wxy .+ Nyy .* wyy
end

function geo_rhs_vector(prob::LargePlateProblem, qg::AbstractVector)
    T = eltype(qg)
    plate = prob.plate
    n = length(plate.nodes)
    ni = length(plate.internal)
    nc = length(plate.corners)
    ndof = 2n + ni + nc
    rhs = zeros(T, ndof)
    # Domain-load particular integral scales as ∼ L²/(8π D) for Kirchhoff FS.
    D = bending_stiffness(plate.props)
    xs = getindex.(prob.pts, 1)
    ys = getindex.(prob.pts, 2)
    Lref = max(maximum(xs) - minimum(xs), maximum(ys) - minimum(ys), eps())
    wt = (Lref^2) / (8 * π * D + eps())
    @inbounds for (k, dof) in enumerate(prob.w_index)
        # free w-equations only (Dirichlet w already known)
        prob.is_kin[dof] && continue
        rhs[dof] = qg[k] * wt
    end
    return rhs
end

function membrane_bodyforce_from_N(prob::LargePlateProblem, Nxx, Nyy, Nxy)
    T = eltype(Nxx)
    dNxx_dx = prob.Fx * Nxx
    dNyy_dy = prob.Fy * Nyy
    dNxy_dx = prob.Fx * Nxy
    dNxy_dy = prob.Fy * Nxy
    bx_pts = dNxx_dx .+ dNxy_dy
    by_pts = dNxy_dx .+ dNyy_dy

    dad = prob.dad_pe
    bn = zeros(T, 2 * dad.nt)
    for i in 1:dad.nt
        p = i <= dad.n ? dad.Nodes[i] : dad.internalNodes[i-dad.n]
        best, bd = 1, Inf
        @inbounds for k in eachindex(prob.pts)
            d = (prob.pts[k][1] - p[1])^2 + (prob.pts[k][2] - p[2])^2
            if d < bd
                bd = d
                best = k
            end
        end
        bn[2i-1] = bx_pts[best]
        bn[2i] = by_pts[best]
    end
    return bn
end

"""Dual-safe membrane solve — no mutation of `dad_pe`."""
function solve_membrane_N(prob::LargePlateProblem, Nxx_nl, Nyy_nl, Nxy_nl)
    T = eltype(Nxx_nl)
    bn = membrane_bodyforce_from_N(prob, Nxx_nl, Nyy_nl, Nxy_nl)
    # promote RHS to Dual if needed
    rhs = T.(prob.b_pe) .+ prob.M_pe * bn
    x = prob.A_pe \ rhs
    dad = prob.dad_pe
    n = dad.n
    BC = dad.BC
    BV = dad.BV
    # reconstruct boundary displacements without mutating cache
    u = zeros(T, 2n)
    @inbounds for dof in 1:2n
        if BC[dof] == 0
            u[dof] = T(BV[dof])
        else
            u[dof] = x[dof]
        end
    end

    ux_pts = zeros(T, length(prob.pts))
    uy_pts = zeros(T, length(prob.pts))
    for (k, p) in enumerate(prob.pts)
        best, bd = 1, Inf
        @inbounds for i in 1:n
            d = (dad.Nodes[i][1] - p[1])^2 + (dad.Nodes[i][2] - p[2])^2
            if d < bd
                bd = d
                best = i
            end
        end
        ux_pts[k] = u[2best-1]
        uy_pts[k] = u[2best]
    end
    E = dad.properties.E
    ν = dad.properties.nu
    h = prob.plate.props.h
    CB = membrane_stiffness_CB(E, ν, h)
    dux_dx = prob.Fx * ux_pts
    dux_dy = prob.Fy * ux_pts
    duy_dx = prob.Fx * uy_pts
    duy_dy = prob.Fy * uy_pts
    Nxx_l = CB .* (dux_dx .+ ν .* duy_dy)
    Nyy_l = CB .* (duy_dy .+ ν .* dux_dx)
    Nxy_l = CB .* ((1 - ν) / 2) .* (dux_dy .+ duy_dx)
    return Nxx_l .+ Nxx_nl, Nyy_l .+ Nyy_nl, Nxy_l .+ Nxy_nl
end

# =============================================================================
# Residual on FREE dofs only (out-of-place, ForwardDiff-safe)
# =============================================================================

"""Scatter free unknown vector `y` into full mixed unknown `x`."""
function _scatter_free(prob::LargePlateProblem, y::AbstractVector, x_lin::AbstractVector)
    T = promote_type(eltype(y), eltype(x_lin))
    x = T.(x_lin)
    k = 0
    @inbounds for dof in eachindex(prob.is_kin)
        if !prob.is_kin[dof]
            k += 1
            x[dof] = y[k]
        end
    end
    return x
end

function _gather_free(prob::LargePlateProblem, x::AbstractVector)
    T = eltype(x)
    nfree = count(!, prob.is_kin)
    y = Vector{T}(undef, nfree)
    k = 0
    @inbounds for dof in eachindex(prob.is_kin)
        if !prob.is_kin[dof]
            k += 1
            y[k] = x[dof]
        end
    end
    return y
end

function plate_residual_free(y, p)
    prob, λ, x_base = p.prob, p.λ, p.x_base
    T = eltype(y)
    x = _scatter_free(prob, y, x_base)
    u = pack_plate_u(prob, x)
    w = extract_w_field(prob, u)
    Nxx_nl, Nyy_nl, Nxy_nl = membrane_N_from_w(prob, w)
    Nxx, Nyy, Nxy = solve_membrane_N(prob, Nxx_nl, Nyy_nl, Nxy_nl)
    qg = geometric_load(prob, w, Nxx, Nyy, Nxy)
    rhs_geo = geo_rhs_vector(prob, qg)
    Rfull = prob.A_pl * x .- (T.(prob.b_bc) .+ T(λ) .* T.(prob.q_load) .+ rhs_geo)
    return _gather_free(prob, Rfull)
end

function _picard_plate_step(prob::LargePlateProblem, λ, x0; niter=12, e=0.5)
    x = copy(x0)
    for _ in 1:niter
        u = pack_plate_u(prob, x)
        w = extract_w_field(prob, u)
        Nxx_nl, Nyy_nl, Nxy_nl = membrane_N_from_w(prob, w)
        Nxx, Nyy, Nxy = solve_membrane_N(prob, Nxx_nl, Nyy_nl, Nxy_nl)
        qg = geometric_load(prob, w, Nxx, Nyy, Nxy)
        rhs = prob.b_bc .+ λ .* prob.q_load .+ Float64.(geo_rhs_vector(prob, qg))
        x_new = prob.A_pl \ rhs
        x .= e .* x_new .+ (1 - e) .* x
    end
    return x
end

"""
    solve_large_plate!(prob; nsteps=20, λ_max=1.0, ...)

Load-controlled von Kármán solution with
`NewtonRaphson(; autodiff=AutoForwardDiff())` on free DOFs only.
"""
function solve_large_plate!(prob::LargePlateProblem;
    nsteps::Int=20, λ_max=1.0, e_relax=0.5,
    abstol=1e-8, reltol=1e-8, maxiters=30, alg=nothing)

    ndof = size(prob.A_pl, 1)
    free = findall(!, prob.is_kin)
    nfree = length(free)
    x = zeros(ndof)
    x_prev = zeros(ndof)

    λs = Float64[]
    wcs = Float64[]
    wh = zeros(ndof, nsteps)

    # Forward-mode AD (Dual numbers) — default over FiniteDiff
    solver = alg === nothing ? NewtonRaphson(; autodiff=AutoForwardDiff()) : alg

    @showprogress "Large plate load steps" for step in 1:nsteps
        λ = λ_max * step / nsteps
        x_lin = prob.A_pl \ (prob.b_bc .+ λ .* prob.q_load)
        if step == 1
            x .= x_lin
        else
            x .= e_relax .* x_prev .+ (1 - e_relax) .* x_lin
        end
        # Picard warm-start (placa_grande), then Newton-AD polish
        x .= _picard_plate_step(prob, λ, x; niter=6, e=e_relax)
        y0 = _gather_free(prob, x)
        x_base = copy(x)
        p = (prob=prob, λ=λ, x_base=x_base)
        nlprob = NonlinearProblem(plate_residual_free, y0, p)
        sol = NonlinearSolve.solve(nlprob, solver; abstol=abstol, reltol=reltol,
            maxiters=maxiters)
        rc = string(sol.retcode)
        if occursin("Success", rc) || occursin("Default", rc)
            x .= _scatter_free(prob, sol.u, x_base)
        else
            # keep Picard result (already applied)
            @debug "Newton-AD did not converge; keeping Picard" λ=λ retcode=sol.retcode
        end
        any(!isfinite, x) && (x .= x_lin)
        x_prev .= x
        u = pack_plate_u(prob, x)
        prob.plate.u .= Float64.(u)
        wh[:, step] .= Float64.(u)
        push!(λs, λ)
        wc = length(prob.plate.internal) > 0 ? Float64(plate_w_int(prob.plate, 1)) : NaN
        push!(wcs, wc)
    end

    prob.λ_hist = λs
    prob.w_hist = wh
    prob.w_center_hist = wcs
    return (λ=λs, w_center=wcs, u_final=prob.plate.u)
end

function linear_wmax_reference(prob::LargePlateProblem; λ=1.0)
    x = prob.A_pl \ (prob.b_bc .+ λ .* prob.q_load)
    u = pack_plate_u(prob, x)
    n = length(prob.plate.nodes)
    length(prob.plate.internal) == 0 && return NaN
    return u[2n+1]
end
