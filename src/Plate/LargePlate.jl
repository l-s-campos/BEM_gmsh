# Large-deflection (von Kármán) thin plates — placa_grande / placa_large.jl
#
# Couples ThinPlate Kirchhoff BEM with plane-stress membrane elasticity BEM.
# Nonlinear residual solved by NonlinearSolve + ForwardDiff (AutoForwardDiff).

export LargePlateProblem, solve_large_plate!, build_large_plate_problem
export membrane_stiffness_CB, extract_w_field, geometric_load
export linear_wmax_reference, solve_large_plate_anm!

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
    Mx::Matrix{Float64}
    My::Matrix{Float64}
    λ_hist::Vector{Float64}
    w_hist::Matrix{Float64}
    w_center_hist::Vector{Float64}
end

membrane_stiffness_CB(E, ν, h) = E * h / (1 - ν^2)

function build_large_plate_problem(plate, dad_pe::BEMdata{<:Elasticity};
    npg_plate=10, npg_pe=10, plate_dibem::Bool=false, rbf=PHS())

    has_cache(plate, :H) || assemble_plate!(plate; npg=npg_plate)
    if plate_dibem
        has_cache(plate, :M) || dibem_plate!(plate; npg=npg_plate, rbf=rbf,
            apply_load=true)
    end
    A_pl, b_full, is_kin, known = apply_bc_plate(plate)
    q_load = has_cache(plate, :plate_q) ? copy(plate.plate_q) : zeros(length(b_full))
    b_bc = b_full .- q_load

    has_cache(dad_pe, :H) || H_G_full_direct(dad_pe; npg=npg_pe, threaded=false)
    applyBC(dad_pe)
    A_pe = copy(dad_pe.A)
    b_pe = copy(dad_pe.b)
    has_cache(dad_pe, :M) || dibem_elasticity!(dad_pe; npg=npg_pe)
    M_pe = Matrix{Float64}(dad_pe.M)

    pts, w_index = ThinPlate._plate_w_samples(plate)

    ops = rbf_gradient_ops(pts; rbf=PHS(3; poly_deg=1))
    Mx = has_cache(plate, :Mx) ? Matrix{Float64}(plate.Mx) : zeros(0, 0)
    My = has_cache(plate, :My) ? Matrix{Float64}(plate.My) : zeros(0, 0)
    return LargePlateProblem(plate, dad_pe, A_pl, b_bc, q_load, is_kin, known,
        A_pe, b_pe, M_pe, pts, ops.Fx, ops.Fy, w_index, Mx, My,
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
    h = prob.plate.properties.h
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

function geometric_flux(prob::LargePlateProblem, w::AbstractVector, Nxx, Nyy, Nxy)
    wx = prob.Fx * w
    wy = prob.Fy * w
    vx = Nxx .* wx .+ Nxy .* wy
    vy = Nxy .* wx .+ Nyy .* wy
    return vx, vy
end

function geometric_load(prob::LargePlateProblem, w::AbstractVector, Nxx, Nyy, Nxy)
    # Fallback scalar ∇·(N∇w) when `Mx` is empty (no DIBEM IBP maps).
    wx = prob.Fx * w
    wy = prob.Fy * w
    wxx = prob.Fx * wx
    wyy = prob.Fy * wy
    wxy = prob.Fx * wy
    Nv = Nxx .* wxx .+ 2 .* Nxy .* wxy .+ Nyy .* wyy
    bx = (prob.Fx * Nxx) .+ (prob.Fy * Nxy)
    by = (prob.Fx * Nxy) .+ (prob.Fy * Nyy)
    return Nv .+ bx .* wx .+ by .* wy
end

function geo_rhs_vector(prob::LargePlateProblem, w, Nxx, Nyy, Nxy)
    T = eltype(w)
    ndof = size(prob.A_pl, 1)
    vx, vy = geometric_flux(prob, w, Nxx, Nyy, Nxy)
    qg = prob.Fx * vx .+ prob.Fy * vy
    plate = prob.plate
    rhs = zeros(T, ndof)
    if has_cache(plate, :M)
        qw = zeros(T, size(plate.M, 2))
        @inbounds for (k, dof) in enumerate(prob.w_index)
            dof <= length(qw) && (qw[dof] = qg[k])
        end
        rhs = plate.M * qw
        return rhs
    end
    D = bending_stiffness(plate.properties)
    xs = getindex.(prob.pts, 1)
    ys = getindex.(prob.pts, 2)
    Lref = max(maximum(xs) - minimum(xs), maximum(ys) - minimum(ys), eps())
    wt = (Lref^2) / (8 * π * D + eps())
    @inbounds for (k, dof) in enumerate(prob.w_index)
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
        p = point(dad, i)
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
    h = prob.plate.properties.h
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
    rhs_geo = geo_rhs_vector(prob, w, Nxx, Nyy, Nxy)
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
        rhs = prob.b_bc .+ λ .* prob.q_load .+
              Float64.(geo_rhs_vector(prob, w, Nxx, Nyy, Nxy))
        x_new = prob.A_pl \ rhs
        x .= e .* x_new .+ (1 - e) .* x
    end
    return x
end

"""
    solve_large_plate!(prob; nsteps=20, λ_max=1.0, nonlinear=:newton, ...)

Load-controlled von Kármán.

`nonlinear`:
- `:newton` — Picard warm-start, then Newton with ForwardDiff Jacobian of `R(x,λ)`
- `:picard` — relaxed fixed-point only (`placa_grande` style)
- `:anm` — asymptotic numerical method + Padé (Cochelin); see
  [`solve_large_plate_anm!`](@ref)
"""
function solve_large_plate!(prob::LargePlateProblem;
    nsteps::Int=20, λ_max=1.0, λ_path=nothing, e_relax=0.5,
    abstol=1e-8, reltol=1e-8, maxiters=30, alg=nothing,
    nonlinear::Symbol=:newton)
    if nonlinear === :anm
        return solve_large_plate_anm!(prob; λ_max=λ_max,
            maxsteps=max(nsteps, 8), rtol=max(reltol, 0.05))
    end

    λ_steps = λ_path === nothing ?
        collect(λ_max .* (1:nsteps) ./ nsteps) : collect(Float64, λ_path)
    nsteps = length(λ_steps)
    ndof = size(prob.A_pl, 1)
    x = zeros(ndof)
    x_prev = zeros(ndof)

    λs = Float64[]
    wcs = Float64[]
    wh = zeros(ndof, nsteps)

    @showprogress "Large plate load steps" for step in 1:nsteps
        λ = λ_steps[step]
        x_lin = prob.A_pl \ (prob.b_bc .+ λ .* prob.q_load)
        if step == 1
            x .= x_lin
        else
            # Continue from the last accepted nonlinear state. Mixing in the
            # linear predictor at large λ overshoots (linear w ~ 5 h at Q=402).
            x .= x_prev
        end
        npic = nonlinear === :picard ? max(maxiters, 6) : 2
        x .= _picard_plate_step(prob, λ, x; niter=npic, e=e_relax)
        x_pic = copy(x)
        if nonlinear !== :picard
            # AD Jacobian of the full mixed residual (same J as ANM)
            if alg === nothing
                xN = _newton_ad!(prob, x, λ; maxiters=maxiters, atol=abstol)
            else
                y0 = _gather_free(prob, x)
                x_base = copy(x)
                p = (prob=prob, λ=λ, x_base=x_base)
                nlprob = NonlinearProblem(plate_residual_free, y0, p)
                sol = NonlinearSolve.solve(nlprob, alg; abstol=abstol, reltol=reltol,
                    maxiters=maxiters)
                xN = _scatter_free(prob, sol.u, x_base)
            end
            ok = all(isfinite, xN) && all(isfinite, x_pic) &&
                norm(xN) < 20 * (norm(x_pic) + 1)
            x .= ok ? xN : x_pic
        end
        any(!isfinite, x) && (x .= x_pic)
        x_prev .= x
        u = pack_plate_u(prob, x)
        set_cache!(prob.plate; u=Float64.(u), T=Float64.(u))
        wh[:, step] .= Float64.(u)
        push!(λs, λ)
        wc = prob.plate.ni > 0 ? Float64(plate_w_int(prob.plate, 1)) : NaN
        push!(wcs, wc)
    end

    prob.λ_hist = λs
    prob.w_hist = wh
    prob.w_center_hist = wcs
    return (λ=λs, w_center=wcs, u_final=prob.plate.u)
end

# =============================================================================
# ANM + Padé (Cochelin, Damil, Potier-Ferry)
# =============================================================================

"""Mixed residual `A x − b − λ q − f(N(w(x)))` on all swapped DOFs."""
function _residual_xλ(prob::LargePlateProblem, x, λ)
    T = eltype(x)
    u = pack_plate_u(prob, x)
    w = extract_w_field(prob, u)
    Nxx_nl, Nyy_nl, Nxy_nl = membrane_N_from_w(prob, w)
    Nxx, Nyy, Nxy = solve_membrane_N(prob, Nxx_nl, Nyy_nl, Nxy_nl)
    rhs_geo = geo_rhs_vector(prob, w, Nxx, Nyy, Nxy)
    return prob.A_pl * x .- (T.(prob.b_bc) .+ T(λ) .* T.(prob.q_load) .+ rhs_geo)
end

"""Forward-mode AD Jacobian `∂R/∂x` at `(x, λ)` (ForwardDiff Duals)."""
function _ad_jacobian(prob::LargePlateProblem, x, λ)
    return ForwardDiff.jacobian(z -> _residual_xλ(prob, z, λ), x)
end

"""Newton on the full mixed residual with an AD Jacobian."""
function _newton_ad!(prob::LargePlateProblem, x, λ;
        maxiters::Int=8, atol=1e-8)
    x = copy(x)
    nq = norm(prob.q_load) + eps()
    for _ in 1:maxiters
        R = _residual_xλ(prob, x, λ)
        nR = norm(R)
        nR < atol * nq && return x
        J = _ad_jacobian(prob, x, λ)
        any(!isfinite, J) && return x
        dx = try
            J \ R
        catch
            return x
        end
        any(!isfinite, dx) && return x
        α = 1.0
        accepted = false
        for _ in 1:10
            xt = x .- α .* dx
            any(!isfinite, xt) && (α *= 0.5; continue)
            if norm(_residual_xλ(prob, xt, λ)) < (1 - 1e-4 * α) * nR
                x .= xt
                accepted = true
                break
            end
            α *= 0.5
        end
        accepted || return x
    end
    return x
end

"""Taylor polynomial `Σ c[k] a^{k-1}`."""
function _poly_eval(c::AbstractVector, a)
    s = zero(a) * (isempty(c) ? 0.0 : c[1])
    ak = one(a)
    @inbounds for k in eachindex(c)
        s += c[k] * ak
        ak *= a
    end
    return s
end

"""Denominator `1 + q₁ a + … + q_M a^M` of a near-diagonal Padé of `c` (a⁰…aᴺ)."""
function _pade_den(c::AbstractVector{<:Real}, M::Int)
    N = length(c) - 1
    M = clamp(M, 0, N ÷ 2)
    M < 1 && return [1.0]
    L = N - M
    A = zeros(M, M)
    b = zeros(M)
    @inbounds for i in 1:M
        pow = L + i
        for j in 1:M
            idx = pow - j
            A[i, j] = (0 <= idx <= N) ? c[idx + 1] : 0.0
        end
        b[i] = -((0 <= pow <= N) ? c[pow + 1] : 0.0)
    end
    qtail = try
        A \ b
    catch
        return [1.0]
    end
    any(!isfinite, qtail) && return [1.0]
    return [1.0; qtail]
end

function _pade_num(c::AbstractVector{<:Real}, q::AbstractVector{<:Real}, L::Int)
    M = length(q) - 1
    p = zeros(L + 1)
    @inbounds for i in 0:L
        s = 0.0
        for j in 0:min(i, M)
            s += c[i - j + 1] * q[j + 1]
        end
        p[i + 1] = s
    end
    return p
end

function _pade_eval_scalar(c::AbstractVector{<:Real}, q::AbstractVector{<:Real}, a)
    M = length(q) - 1
    N = length(c) - 1
    L = N - M
    L < 0 && return _poly_eval(c, a)
    p = _pade_num(c, q, L)
    den = _poly_eval(q, a)
    abs(den) < 1e-14 && return _poly_eval(c, a)
    return _poly_eval(p, a) / den
end

"""Smallest positive real root of `Q(a) = Σ q[k] a^{k-1}`, or `Inf`."""
function _pade_pole(q::AbstractVector{<:Real})
    M = length(q) - 1
    M < 1 && return Inf
    abs(q[end]) < 1e-14 && return Inf
    C = zeros(M, M)
    @inbounds for i in 1:M-1
        C[i, i + 1] = 1.0
    end
    @inbounds for j in 1:M
        C[M, j] = -q[j] / q[end]
    end
    vals = try
        eigvals(C)
    catch
        return Inf
    end
    amin = Inf
    @inbounds for z in vals
        abs(imag(z)) < 1e-8 * max(abs(real(z)), 1.0) || continue
        r = real(z)
        r > 1e-12 && r < amin && (amin = r)
    end
    return amin
end

"""
ANM series of order `N` at `(x0, λ0)` on the full mixed vector:
`x(a)=Σ a^p x_p`, `λ(a)=Σ a^p λ_p`.

Orders `p≥2` from one residual sample at a small `ε` (Cochelin). Jacobian
once per expansion.
"""
function _anm_series(prob::LargePlateProblem, x0, λ0; order::Int=6, ε=0.05)
    # Load parametrization: λ(a) = λ0 + a  (no turning points on this path).
    N = max(order, 1)
    J = _ad_jacobian(prob, x0, λ0)
    any(!isfinite, J) && error("ANM Jacobian is not finite at λ=$λ0")
    F = lu(J)
    x1 = F \ prob.q_load               # J x1 = q
    X = Vector{Vector{Float64}}(undef, N + 1)
    Λ = zeros(N + 1)
    X[1] = collect(Float64, x0)
    Λ[1] = float(λ0)
    X[2] = x1
    Λ[2] = 1.0
    for p in 2:N
        xε = copy(x0)
        εk = ε
        @inbounds for k in 1:p-1
            xε .+= εk .* X[k + 1]
            εk *= ε
        end
        λε = λ0 + ε                     # λ1=1, λp=0
        Fp = _residual_xλ(prob, xε, λε) ./ (ε^p)
        X[p + 1] = F \ (-Fp)
        Λ[p + 1] = 0.0
    end
    return X, Λ
end

function _anm_eval_taylor(X, Λ, a)
    x = zero(X[1])
    λ = 0.0
    ak = 1.0
    @inbounds for p in 1:length(X)
        x .+= ak .* X[p]
        λ += ak * Λ[p]
        ak *= a
    end
    return x, λ
end

function _anm_pade_q(X, Λ)
    i0 = argmax(abs.(X[2]))
    c = [X[p][i0] for p in 1:length(X)]
    q = _pade_den(c, (length(c) - 1) ÷ 2)
    length(q) > 1 && return q
    return _pade_den(Λ, (length(Λ) - 1) ÷ 2)
end

function _anm_eval_pade(X, Λ, a)
    q = _anm_pade_q(X, Λ)
    λ = Λ[1] + a * Λ[2]               # load parametrization
    x = zero(X[1])
    @inbounds for i in eachindex(x)
        c = [X[p][i] for p in 1:length(X)]
        x[i] = _pade_eval_scalar(c, q, a)
    end
    return x, λ
end

function _anm_radius(X)
    N = length(X) - 1
    a = Inf
    @inbounds for p in 1:N-1
        n0 = norm(X[p + 1])
        n1 = norm(X[p + 2])
        n1 > 1e-14 * max(n0, 1.0) || continue
        a = min(a, n0 / n1)
    end
    return isfinite(a) ? a : 0.25
end

function _anm_amax(prob, X, Λ; rtol=0.5, a_hi=10.0)
    qn = norm(prob.q_load) + eps()
    tol = rtol * qn
    amin = min(1e-3, a_hi)
    hi = a_hi
    ar = 0.8 * _anm_radius(X)
    ar >= 0.05 * a_hi && (hi = min(hi, ar))
    apole = _pade_pole(_anm_pade_q(X, Λ))
    isfinite(apole) && apole >= 0.05 * a_hi && (hi = min(hi, 0.8 * apole))
    a = max(hi, amin)
    best = amin
    for _ in 1:16
        xp, λp = _anm_eval_pade(X, Λ, a)
        xt, λt = _anm_eval_taylor(X, Λ, a)
        xbound = norm(X[1]) + abs(a) * norm(X[2]) + 1
        if !all(isfinite, xp) || !isfinite(λp) || norm(xp) > 10 * xbound
            xp, λp = xt, λt
        end
        if all(isfinite, xp) && isfinite(λp)
            nr = norm(_residual_xλ(prob, xp, λp))
            if nr <= tol
                return a
            end
            best = a
        end
        a < amin && break
        a *= 0.5
    end
    return clamp(best, amin, a_hi)
end

"""
    solve_large_plate_anm!(prob; λ_max=1.0, order=6, …)

Asymptotic numerical method (power series of the path) + Padé of that
series (Cochelin). One Jacobian per continuation step; higher-order RHS
from residual samples. Optional Picard corrector after each Padé step
(`corrector=true`; off by default — the lumped geometric load can kick
the Picard iterate off the path).
"""
function solve_large_plate_anm!(prob::LargePlateProblem;
        λ_max=1.0, order::Int=3, ε=0.1, rtol=0.5,
        maxsteps::Int=20, corrector::Bool=false)

    x = zeros(length(prob.is_kin))
    λ = 0.0
    λs = Float64[λ]
    n = prob.plate.n
    u0 = pack_plate_u(prob, x)
    w0 = prob.plate.ni > 0 ? Float64(u0[2n+1]) : 0.0
    wcs = Float64[w0]
    ndof = length(x)
    wh = zeros(ndof, maxsteps + 1)
    wh[:, 1] .= Float64.(u0)

    @showprogress "ANM–Padé steps" for step in 1:maxsteps
        λ >= λ_max - 1e-10 && break
        X, Λ = _anm_series(prob, x, λ; order=order, ε=ε)
        a_hi = max(λ_max - λ, 0.0)
        amax = _anm_amax(prob, X, Λ; rtol=rtol, a_hi=max(a_hi, 1e-6))
        amax = min(amax, a_hi)
        amax <= 1e-10 && break
        xp, λp = _anm_eval_pade(X, Λ, amax)
        xt, λt = _anm_eval_taylor(X, Λ, amax)
        xbound = norm(X[1]) + abs(amax) * norm(X[2]) + 1
        if !all(isfinite, xp) || !isfinite(λp) || norm(xp) > 10 * xbound
            xp, λp = xt, λt
        end
        if corrector
            xp = _newton_ad!(prob, xp, λp; maxiters=4, atol=1e-8)
        end
        any(!isfinite, xp) && break
        x = xp
        λ = λp
        λ > λ_max && (λ = λ_max)
        push!(λs, λ)
        u = pack_plate_u(prob, x)
        set_cache!(prob.plate; u=Float64.(u), T=Float64.(u))
        col = min(step + 1, size(wh, 2))
        wh[:, col] .= Float64.(u)
        wc = prob.plate.ni > 0 ? Float64(u[2n+1]) : NaN
        push!(wcs, wc)
    end
    nrec = length(λs)
    prob.λ_hist = λs
    prob.w_hist = wh[:, 1:min(nrec, size(wh, 2))]
    prob.w_center_hist = wcs
    return (λ=λs, w_center=wcs, u_final=prob.plate.u)
end

function linear_wmax_reference(prob::LargePlateProblem; λ=1.0)
    x = prob.A_pl \ (prob.b_bc .+ λ .* prob.q_load)
    u = pack_plate_u(prob, x)
    n = prob.plate.n
    prob.plate.ni == 0 && return NaN
    return u[2n+1]
end
