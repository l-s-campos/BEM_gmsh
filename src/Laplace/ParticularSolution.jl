# =============================================================================
# Particular-solution BEM (Dual Reciprocity / RBF)
# =============================================================================
# Poisson:   ∇²u = f   →  u = u_h + u_p,  ∇²u_p ≈ f,  ∇²u_h = 0
# Transient: ∂u/∂t = κ ∇²u + f  (DRM + θ-method)

export fit_source_field, particular_from_coeffs, particular_grad
export solve_poisson_rbf_bem!, solve_transient_drm!, solve_drm_poisson!
export compare_poisson_rbf_bem, build_drm_matrices, galerkin_drm_mixed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_all_pts(dad::BEMdata) = Point[dad.Nodes; dad.internalNodes]

function _eval_field(f, pts::AbstractVector{<:Point})
    if f isa Function
        return Float64[float(f(p)) for p in pts]
    elseif f isa AbstractVector
        length(f) == length(pts) || throw(DimensionMismatch("f length"))
        return float.(f)
    else
        return fill(float(f), length(pts))
    end
end

function _eval_field_t(f, pts, t::Real)
    if f isa Function
        # try f(p,t) then f(p)
        try
            return Float64[float(f(p, t)) for p in pts]
        catch
            return Float64[float(f(p)) for p in pts]
        end
    else
        return _eval_field(f, pts)
    end
end

"""
    fit_source_field(pts, fvals; method=:global, basis=PHS(3;poly_deg=1)) -> fit

`method` ∈ `:global, :local, :pu, :rational`.
"""
function fit_source_field(
        pts::AbstractVector{<:Point},
        fvals::AbstractVector{<:Real};
        method::Symbol = :global,
        basis::AbstractRadialBasis = PHS(3; poly_deg = 1),
        k_local::Int = 15,
    )
    pts = collect(Point, pts)
    y = float.(fvals)
    if method === :global
        rbf = RBF(pts, basis)
        n = length(pts)
        rhs = rbf.npoly == 0 ? y : vcat(y, zeros(rbf.npoly))
        coef = rbf.fat \ rhs
        return (; method, basis, pts, α = coef[1:n], β = coef[(n + 1):end],
            h = rbf.h, rbf = rbf, deg = poly_deg(basis), y)
    elseif method === :local
        lr = local_rbf_fit(pts, basis; k = k_local)
        return (; method, basis, pts, y, lr, h = rbf_length_scale(pts), α = y)
    elseif method === :pu
        pu = pu_rbf_fit(pts, basis; k_local = k_local)
        return (; method, basis, pts, y, pu, h = rbf_length_scale(pts), α = y)
    elseif method === :rational
        b = basis isa IMQ ? basis : IMQ(1.0; poly_deg = -1)
        rr = rational_rbf_fit(pts, y, b)
        return (; method, basis = b, pts, y, rr, h = rr.h, α = y)
    else
        throw(ArgumentError("unknown method $method; use :global,:local,:pu,:rational"))
    end
end

"""Particular solution of ``∇²Ψ = p`` for low-order monomials of the source interpolant.

2D: ``∇²(r²/4)=1``, ``∇²(x r²/8)=x``, ``∇²(y r²/8)=y``.
3D: ``∇²(r²/6)=1``, ``∇²(x r²/10)=x`` (and cyclic).
Without this, CPD ``f=Fα+Pβ`` with ``\\mathrm{deg}(P)≤1`` parks constants/linears in ``β``
and ``u_p=Ψα`` misses them (``q_p≈0``).
"""
function _poly_laplace_particular(β::AbstractVector, x::Point; dim::Int)
    isempty(β) && return 0.0
    r2 = 0.0
    @inbounds for d in 1:dim
        r2 += x[d] * x[d]
    end
    s = 0.0
    if dim == 2
        length(β) >= 1 && (s += β[1] * r2 / 4)
        length(β) >= 2 && (s += β[2] * x[1] * r2 / 8)
        length(β) >= 3 && (s += β[3] * x[2] * r2 / 8)
    elseif dim == 3
        length(β) >= 1 && (s += β[1] * r2 / 6)
        length(β) >= 2 && (s += β[2] * x[1] * r2 / 10)
        length(β) >= 3 && (s += β[3] * x[2] * r2 / 10)
        length(β) >= 4 && (s += β[4] * x[3] * r2 / 10)
    end
    return s
end

function particular_from_coeffs(fit, x::Point; dim::Int = length(x))
    if fit.method === :global
        s = 0.0
        @inbounds for j in eachindex(fit.pts)
            r = norm(x - fit.pts[j])
            # φ_h(r) = φ(r/h) ⇒ ∇²_x φ_h = h^{-2} (∇²φ)(r/h)
            # Ψ_h with ∇² Ψ_h = φ_h ⇒ Ψ_h(r) = h² Ψ(r/h)
            s += fit.α[j] * laplace_particular(fit.basis, r / fit.h; dim = dim) * fit.h^2
        end
        if hasproperty(fit, :β) && !isempty(fit.β)
            s += _poly_laplace_particular(fit.β, x; dim = dim)
        end
        return s
    else
        y = :y in propertynames(fit) ? fit.y : fit.α
        gfit = fit_source_field(fit.pts, y; method = :global, basis = PHS(3; poly_deg = 1))
        return particular_from_coeffs(gfit, x; dim = dim)
    end
end

particular_from_coeffs(fit, xs::AbstractVector{<:Point}; dim = 2) =
    [particular_from_coeffs(fit, x; dim = dim) for x in xs]

function particular_grad(fit, x::Point; dim::Int = length(x))
    ε = 1e-7
    g = zeros(dim)
    @inbounds for d in 1:dim
        if dim == 2
            xp = Point2D(x[1] + (d == 1 ? ε : 0), x[2] + (d == 2 ? ε : 0))
            xm = Point2D(x[1] - (d == 1 ? ε : 0), x[2] - (d == 2 ? ε : 0))
        else
            xp = Point3D(x[1] + (d == 1 ? ε : 0), x[2] + (d == 2 ? ε : 0), x[3] + (d == 3 ? ε : 0))
            xm = Point3D(x[1] - (d == 1 ? ε : 0), x[2] - (d == 2 ? ε : 0), x[3] - (d == 3 ? ε : 0))
        end
        g[d] = (particular_from_coeffs(fit, xp; dim = dim) -
                particular_from_coeffs(fit, xm; dim = dim)) / (2ε)
    end
    return g
end

function _dn_particular(fit, x::Point, n̂; dim::Int = length(x))
    g = particular_grad(fit, x; dim = dim)
    s = 0.0
    @inbounds for d in 1:dim
        s += g[d] * n̂[d]
    end
    return s
end

# ---------------------------------------------------------------------------
# Poisson
# ---------------------------------------------------------------------------

"""
    solve_poisson_rbf_bem!(dad, f; method=:global, basis=PHS(3;poly_deg=1), npg=16)

Solve ``∇²u = f`` by particular solution + homogeneous BEM.
"""
function solve_poisson_rbf_bem!(
        dad::BEMdata{<:LaplaceLike},
        f;
        method::Symbol = :global,
        basis::AbstractRadialBasis = PHS(3; poly_deg = 1),
        npg::Int = 16,
        k_local::Int = 15,
    )
    dim = dad.dimension
    pts = _all_pts(dad)
    fvals = _eval_field(f, pts)
    fit = fit_source_field(pts, fvals; method = method, basis = basis, k_local = k_local)
    up = particular_from_coeffs(fit, pts; dim = dim)

    BC0 = copy(dad.BC)
    BV0 = copy(dad.BV)
    kcond = float(dad.properties isa Laplace ? dad.properties.k : 1.0)

    @inbounds for i in 1:dad.n
        if dad.BC[i] == 0
            dad.BV[i] = BV0[i] - up[i]
        else
            ∂n = _dn_particular(fit, dad.Nodes[i], dad.Normal[i]; dim = dim)
            dad.BV[i] = BV0[i] - (-kcond * ∂n)   # q_p = -k ∂u_p/∂n
        end
    end

    H_G_full_direct(dad, npg)
    solve(dad)
    uh = copy(dad.T)
    qh = has_cache(dad, :q) ? copy(dad.q) : zeros(dad.n)

    dad.BC .= BC0
    dad.BV .= BV0

    u = uh[1:dad.nt] .+ up
    q = zeros(dad.n)
    @inbounds for i in 1:dad.n
        ∂n = _dn_particular(fit, dad.Nodes[i], dad.Normal[i]; dim = dim)
        q[i] = qh[i] + (-kcond * ∂n)
    end
    set_cache!(dad; T = u, q = q, uh = uh, up = up, rbf_fit = fit, poisson_f = fvals)
    return u
end

function compare_poisson_rbf_bem(
        dad::BEMdata,
        f,
        u_exact;
        methods = (:global, :local, :pu),
        basis = PHS(3; poly_deg = 1),
        npg = 12,
    )
    results = NamedTuple[]
    pts = _all_pts(dad)
    uex = _eval_field(u_exact, pts)
    for m in methods
        dad_c = deepcopy(dad)
        t0 = time()
        try
            u = solve_poisson_rbf_bem!(dad_c, f; method = m, basis = basis, npg = npg)
            dt = time() - t0
            err = u .- uex
            push!(results, (; method = m, rmse = sqrt(mean(abs2, err)),
                maxerr = maximum(abs, err), time = dt))
        catch e
            @warn "method $m failed" exception = e
            push!(results, (; method = m, rmse = Inf, maxerr = Inf, time = NaN))
        end
    end
    return results
end

# ---------------------------------------------------------------------------
# DRM matrices
# ---------------------------------------------------------------------------

"""
    build_drm_matrices(dad, basis; npg=12) -> (; H, G, F, Ψ, η, M, h)

Build dual-reciprocity operators:
- ``F_{ij} = φ(|x_i-x_j|/h)``
- ``Ψ_{ij} = h² Ψ̂(|x_i-x_j|/h)`` with ``∇²Ψ̂ = φ``
- ``η_{ij} = q_j(x_i) = -k ∂Ψ_j/∂n(x_i)`` on boundary nodes
  (package flux, same convention as `G`)
- ``C = H Ψ - G η``, ``M = C F^{-1}`` so ``H u - G q = M b`` when ``∇²u = b``
"""
function build_drm_matrices(
        dad::BEMdata{<:Laplace},
        basis::AbstractRadialBasis = PHS(3; poly_deg = 1);
        npg::Int = 12,
        galerkin::Bool = false,
    )
    dim = dad.dimension
    pts = _all_pts(dad)
    nt = dad.nt
    n = dad.n
    if galerkin
        H_G_galerkin(dad; npg=npg, threaded=false)
    else
        has_cache(dad, :H) || H_G_full_direct(dad, npg)
    end
    H = Matrix(dad.H)
    G = Matrix(dad.G)


    hh = max(rbf_length_scale(pts), 1e-14)
    F = zeros(nt, nt)
    Ψ = zeros(nt, nt)
    @inbounds for j in 1:nt, i in 1:nt
        rij = norm(pts[i] - pts[j])
        F[i, j] = basis(_scale_r(rij, hh))
        Ψ[i, j] = laplace_particular(basis, rij / hh; dim = dim) * hh^2
    end
    ε = 1e-12 * (tr(F) / nt + 1)
    @inbounds for i in 1:nt
        F[i, i] += ε
    end

    kcond = float(dad.properties.k)
    η = zeros(n, nt)
    @inbounds for j in 1:nt, i in 1:n
        εn = 1e-7
        xi = dad.Nodes[i]
        ni = dad.Normal[i]
        if dim == 2
            xp = Point2D(xi[1] + εn * ni[1], xi[2] + εn * ni[2])
            xm = Point2D(xi[1] - εn * ni[1], xi[2] - εn * ni[2])
        else
            xp = Point3D(xi[1] + εn * ni[1], xi[2] + εn * ni[2], xi[3] + εn * ni[3])
            xm = Point3D(xi[1] - εn * ni[1], xi[2] - εn * ni[2], xi[3] - εn * ni[3])
        end
        rj = pts[j]
        Ψp = laplace_particular(basis, norm(xp - rj) / hh; dim = dim) * hh^2
        Ψm = laplace_particular(basis, norm(xm - rj) / hh; dim = dim) * hh^2
        η[i, j] = -kcond * (Ψp - Ψm) / (2εn)
    end

    pdeg = max(poly_deg(basis), -1)
    npoly = pdeg < 0 ? 0 : rbf_npoly(dim, pdeg)
    if npoly > 0 && dim == 2
        P = _dibem_poly_P(pts, basis)
        Ψp = zeros(nt, npoly)
        ηp = zeros(n, npoly)
        @inbounds for k in 1:npoly, i in 1:nt
            Ψp[i, k] = _laplace_poly_psi(pts[i], k, pdeg)
        end
        @inbounds for k in 1:npoly, i in 1:n
            g = _laplace_poly_grad(dad.Nodes[i], k, pdeg)
            ηp[i, k] = -kcond * (g[1] * dad.Normal[i][1] + g[2] * dad.Normal[i][2])
        end
        K = [F P; P' zeros(npoly, npoly)]
        Cfull = [H * Ψ - G * η   H * Ψp - G * ηp]
        W = K \ [Matrix{Float64}(I, nt, nt); zeros(npoly, nt)]
        M = Cfull * W
        C = H * Ψ - G * η
        set_cache!(dad; M = M)
        return (; H, G, F, Ψ, η, M, h = hh, C, P, npoly, Ψp, ηp, W)
    end

    C = H * Ψ - G * η
    M = C / F
    set_cache!(dad; M = M)
    return (; H, G, F, Ψ, η, M, h = hh, C, npoly = 0, Ψp = nothing, ηp = nothing, W = nothing)
end

# ∇² Ψ = monomial  (2-D). Order matches MonomialBasis{2,deg}:
# deg0: 1
# deg1: 1, x, y
# deg2: 1, x, y, xy, x², y²
function _laplace_poly_psi(x, k::Int, deg::Int)
    X, Y = x[1], x[2]
    r2 = X * X + Y * Y
    k == 1 && return r2 / 4
    deg >= 1 && k == 2 && return X * r2 / 8
    deg >= 1 && k == 3 && return Y * r2 / 8
    deg >= 2 && k == 4 && return X * Y * r2 / 12
    deg >= 2 && k == 5 && return X^4 / 12
    deg >= 2 && k == 6 && return Y^4 / 12
    return 0.0
end

function _laplace_poly_grad(x, k::Int, deg::Int)
    X, Y = x[1], x[2]
    r2 = X * X + Y * Y
    k == 1 && return SVector(X / 2, Y / 2)
    if deg >= 1 && k == 2
        return SVector((3 * X * X + Y * Y) / 8, (X * Y) / 4)
    end
    if deg >= 1 && k == 3
        return SVector((X * Y) / 4, (X * X + 3 * Y * Y) / 8)
    end
    if deg >= 2 && k == 4
        # Ψ = xy r²/12 = (x³y + xy³)/12
        return SVector((3 * X * X * Y + Y^3) / 12, (X^3 + 3 * X * Y * Y) / 12)
    end
    deg >= 2 && k == 5 && return SVector(X^3 / 3, 0.0)
    deg >= 2 && k == 6 && return SVector(0.0, Y^3 / 3)
    return SVector(0.0, 0.0)
end

"""
    galerkin_drm_mixed(dad, basis; npg=12) -> (; A, b, Mdrm, C, D, N, drm, mode)

Pérez-Gavilán / Aliabadi mixed DRM. `A` is the symmetric Costabel block
(their eq. 11). DRM body columns `C` are eq. 19, `Mdrm = C F⁻¹` so the
mixed residual is `A x = b_bc + Mdrm b`. Internal DRM points enlarge `F`
only; `A` stays on boundary unknowns `(q_D, u_N)`.
"""
function galerkin_drm_mixed(
        dad::BEMdata{<:Laplace},
        basis::AbstractRadialBasis = PHS(3; poly_deg = 1);
        npg::Int = 12,
        threaded::Bool = false,
    )
    drm = build_drm_matrices(dad, basis; npg=npg, galerkin=true)
    n = dad.n
    V = Matrix(dad.galerkin_V)
    Hb = Matrix(dad.H[1:n, 1:n])
    W = Matrix(dad.galerkin_W)
    B = Matrix(dad.galerkin_B)
    sys = galerkin_mixed_system(dad; npg=npg, threaded=threaded)
    D, Nset = sys.D, sys.N
    nD, nN = length(D), length(Nset)
    m = nD + nN
    Ψb = drm.Ψ[1:n, :]
    ηb = drm.η
    C = zeros(m, dad.nt)
    if nD > 0
        C[1:nD, :] .= V[D, :] * ηb .- Hb[D, :] * Ψb
    end
    if nN > 0
        C[nD+1:m, :] .= .-B[Nset, :] * ηb .+ W[Nset, :] * Ψb
    end
    Mdrm = C / drm.F
    set_cache!(dad; M=drm.M, galerkin_drm=Mdrm)
    return (; A=sys.A, b=sys.b, D, N=Nset, C, Mdrm, drm, mode=sys.mode, ops=sys)
end

"""
    solve_drm_poisson!(dad, f; basis=PHS(3), npg=12, galerkin=false)

``H u - G q = M f`` with DRM mass ``M = (HΨ - Gη) F⁻¹``.
"""
function solve_drm_poisson!(
        dad::BEMdata{<:Laplace},
        f;
        basis::AbstractRadialBasis = PHS(3; poly_deg = 1),
        npg::Int = 12,
        galerkin::Bool = false,
    )
    drm = build_drm_matrices(dad, basis; npg=npg, galerkin=galerkin)
    set_cache!(dad; H=drm.H, G=drm.G, M=drm.M)
    if has_cache(dad, :A)
        dad.cache.A = nothing
    end
    applyBC(dad)
    fv = _eval_field(f, _all_pts(dad))
    dad.b .+= drm.M * fv
    x = bem_linsolve(dad.A, dad.b)
    Tfull = zeros(dad.nt)
    qfull = zeros(dad.n)
    Tfull[1:length(x)] .= x
    split_sol!(dad, Tfull, qfull)
    set_cache!(dad; T=Tfull, q=qfull)
    return Tfull
end



# ---------------------------------------------------------------------------
# Transient DRM
# ---------------------------------------------------------------------------

"""
    solve_transient_drm!(dad, u0; κ=1.0, Δt, t_end, f=0, basis=PHS(3), npg=12)

DRM time stepping for ``∂u/∂t = κ ∇²u + f``.

Discrete (backward Euler, θ=1):
``(H - M/(κ Δt)) u^{n+1} - G q^{n+1} = -M f/κ - M u^n/(κ Δt)``
with ``M = (HΨ - Gη) F^{-1}`` so ``H u - G q = M b`` when ``∇²u = b = (ú - f)/κ``.

BCs on `dad` are applied each step via [`applyBC`](@ref).
"""
function solve_transient_drm!(
        dad::BEMdata{<:Laplace},
        u0::AbstractVector{<:Real};
        κ::Float64 = 1.0,
        Δt::Float64 = 0.01,
        t_end::Float64 = 1.0,
        f = 0.0,
        θ::Float64 = 1.0,
        basis::AbstractRadialBasis = PHS(3; poly_deg = 1),
        npg::Int = 12,
        galerkin::Bool = false,
    )
    0 < θ <= 1 || throw(ArgumentError("θ ∈ (0,1]"))
    nt = dad.nt
    n = dad.n
    length(u0) == nt || throw(DimensionMismatch("u0 length $(length(u0)) ≠ nt=$nt"))
    pts = _all_pts(dad)
    dim = dad.dimension

    drm = build_drm_matrices(dad, basis; npg = npg, galerkin = galerkin)

    H0, G0, Mraw = drm.H, drm.G, drm.M
    # H u - G q = Mraw * b with b = ∇²u = (ú - f)/κ
    # ⇒ H u - G q = (Mraw/κ) (ú - f)
    M = Mraw ./ κ

    nsteps = max(1, Int(ceil(t_end / Δt)))
    Δt = t_end / nsteps
    U = zeros(nt, nsteps + 1)
    U[:, 1] = float.(u0)
    t_hist = collect(range(0, t_end; length = nsteps + 1))
    q_hist = zeros(n, nsteps + 1)

    u = float.(u0)
    # factor left-hand operator once if BCs time-independent
    H_eff = H0 .- (θ / Δt) .* M
    set_cache!(dad; H = H_eff, G = G0)
    if has_cache(dad, :A)
        dad.cache.A = nothing
    end
    applyBC(dad)
    Afac = lu(dad.A)
    b_bc_template = copy(dad.b)   # from BC only (domain RHS added each step)

    for step in 1:nsteps
        tnp = t_hist[step + 1]
        tn = t_hist[step]
        fv = _eval_field_t(f, pts, tnp)
        # RHS: H_eff u^{n+1} - G q = b_BC + b_dom
        # From:
        #   H u^{n+1} - G q - (θ/Δt) M u^{n+1} = (1-θ)/Δt M? 
        # Backward Euler θ=1:
        #   H u - G q - M/Δt u = -M f - M/Δt u^n
        #   b_dom = -M*f - M*u^n/Δt
        #
        # Crank–Nicolson θ=0.5:
        #   H u^{n+1} - G q^{n+1} - (θ/Δt) M u^{n+1}
        #     = (1-θ)/Δt? Standard split:
        #   H (θ u^{n+1}+(1-θ)u^n) - G q_avg - M (u^{n+1}-u^n)/Δt = -M f
        # Simplified BE only for robustness when θ=1; for θ<1 include explicit part:
        b_dom = -M * fv .- (M * u) ./ Δt
        if θ < 1 - 1e-14
            # add explicit residual of previous step: (1-θ) terms
            # H (1-θ) u^n contributes to RHS when moved:
            # (H - θ M/Δt) u^{n+1} - G q = -M f - M u^n/Δt - (1-θ) H u^n + (1-θ) M/Δt? 
            # Use pure BE for θ=1; for general θ use:
            b_dom .-= (1 - θ) .* (H0 * u)
            b_dom .+= (1 - θ) / Δt .* (M * u)
        end

        b = b_bc_template .+ b_dom
        x = Afac \ b
        Tfull = zeros(nt)
        qfull = zeros(n)
        Tfull[1:length(x)] .= x
        split_sol!(dad, Tfull, qfull)
        u = copy(Tfull)
        U[:, step + 1] = u
        q_hist[:, step + 1] = qfull
    end

    set_cache!(dad; H = H0, G = G0, T = U[:, end], q = q_hist[:, end],
        time = t_hist, U_hist = U, q_hist = q_hist, drm = drm)
    if has_cache(dad, :A)
        dad.cache.A = nothing
    end
    return t_hist, U
end
