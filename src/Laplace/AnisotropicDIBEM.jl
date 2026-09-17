# Strategy 1: anisotropic Laplace as isotropic Poisson + DIBEM residual.
# k = (det K)^{1/d}, ΔK = K − k I, f* = (1/k) ∇·(ΔK ∇u) ≈ A_f u  (RBF Hess).
# One linear system (no Picard).

export solve_anisotropic_dibem!, solve_anisotropic_ibp!, anisotropic_wave_shift!
export rbf_diff_ops, isotropic_scale

isotropic_scale(K::AbstractMatrix) = det(K)^(1 / size(K, 1))

_shift_coord(x::SVector{D,T}, β, h) where {D,T} =
    SVector{D,T}(ntuple(k -> x[k] + (k == β ? T(h) : zero(T)), D))

"""
RBF first and second derivative matrices on `pts`: `∂_α u ≈ D[α]*u`,
`∂_{αβ} u ≈ H[α,β]*u`. `nlocal=nothing` is a global interpolant; a positive
`nlocal` uses RBF-FD on that many neighbours (needed on domains with holes).
"""
function rbf_diff_ops(pts::AbstractVector{<:Point}, rbf::AbstractRadialBasis;
        nlocal::Union{Nothing,Integer}=nothing)
    nlocal === nothing && return _rbf_diff_ops_global(pts, rbf)
    return _rbf_diff_ops_local(pts, rbf, Int(nlocal))
end

function _rbf_diff_ops_global(pts::AbstractVector{<:Point}, rbf::AbstractRadialBasis)
    n = length(pts)
    dim = length(pts[1])
    deg = poly_deg(rbf)
    npoly = rbf_npoly(dim, deg)
    m = n + npoly
    A = zeros(m, m)
    @inbounds for j in 1:n, i in 1:n
        A[i, j] = rbf(norm(pts[i] - pts[j]))
    end
    if npoly > 0
        mon = MonomialBasis(dim, deg)
        @inbounds for i in 1:n
            p = mon(pts[i])
            for k in 1:npoly
                A[i, n + k] = p[k]
                A[n + k, i] = p[k]
            end
        end
    end
    ε = 1e-12 * (sum(abs, view(A, 1:n, 1:n)) / max(n^2, 1) + 1)
    @inbounds for i in 1:n
        A[i, i] += ε
    end
    rhs = zeros(m, n)
    @inbounds for i in 1:n
        rhs[i, i] = 1.0
    end
    W = A \ rhs                       # (n+npoly) × n   interpolant weights for each nodal u
    D = ntuple(_ -> zeros(n, n), dim)
    Hess = Matrix{Float64}[zeros(n, n) for _ in 1:dim, _ in 1:dim]
    Fx = [zeros(n, m) for _ in 1:dim]
    Hx = [zeros(n, m) for _ in 1:dim, _ in 1:dim]
    @inbounds for j in 1:n, i in 1:n
        for α in 1:dim
            Fx[α][i, j] = ∂(rbf, α, pts[i], pts[j])
        end
        Hij = _rbf_hess(rbf, pts[i], pts[j], dim)
        for α in 1:dim, β in 1:dim
            Hx[α, β][i, j] = Hij[α, β]
        end
    end
    if npoly > 0
        mon = MonomialBasis(dim, deg)
        @inbounds for i in 1:n
            for α in 1:dim
                dp = ∂(mon, α, pts[i])
                for k in 1:npoly
                    Fx[α][i, n + k] = dp[k]
                end
            end
            _monomial_hess!(Hx, i, n, dim, deg)
        end
    end
    @inbounds for α in 1:dim
        D[α] .= view(Fx[α] * W, 1:n, 1:n)
        for β in 1:dim
            Hess[α, β] .= view(Hx[α, β] * W, 1:n, 1:n)
        end
    end
    return D, Hess
end

function _rbf_diff_ops_local(pts::AbstractVector{<:Point}, rbf::AbstractRadialBasis,
        nlocal::Int)
    n = length(pts)
    dim = length(pts[1])
    deg0 = poly_deg(rbf)
    npoly0 = rbf_npoly(dim, deg0)
    nlocal = clamp(nlocal, max(npoly0 + 1, dim + 2), n)
    tree = _rbf_kdtree(pts)
    D = ntuple(_ -> zeros(n, n), dim)
    Hess = Matrix{Float64}[zeros(n, n) for _ in 1:dim, _ in 1:dim]
    @inbounds for i in 1:n
        ids = rbf_neighbors(tree, pts[i], nlocal)
        m = length(ids)
        deg = deg0
        thin = _stencil_is_planar(pts, ids, dim)
        thin && (deg = min(deg, 1))
        while deg >= 0 && rbf_npoly(dim, deg) > m
            deg -= 1
        end
        np = rbf_npoly(dim, deg)
        A = zeros(m + np, m + np)
        @inbounds for b in 1:m, a in 1:m
            A[a, b] = rbf(norm(pts[ids[a]] - pts[ids[b]]))
        end
        ε = 1e-12 * (sum(abs, view(A, 1:m, 1:m)) / max(m^2, 1) + 1)
        @inbounds for a in 1:m
            A[a, a] += ε
        end
        if np > 0
            mon = MonomialBasis(dim, deg)
            @inbounds for a in 1:m
                p = mon(pts[ids[a]])
                for k in 1:np
                    A[a, m + k] = p[k]
                    A[m + k, a] = p[k]
                end
            end
        end
        xi = pts[i]
        rhs = zeros(m + np)
        for α in 1:dim
            fill!(rhs, 0.0)
            @inbounds for b in 1:m
                rhs[b] = ∂(rbf, α, xi, pts[ids[b]])
            end
            if np > 0
                dp = ∂(MonomialBasis(dim, deg), α, xi)
                @inbounds for k in 1:np
                    rhs[m + k] = dp[k]
                end
            end
            coef = _local_rbf_solve(A, rhs)
            @inbounds for (t, j) in enumerate(ids)
                D[α][i, j] = coef[t]
            end
            # PHS Hessian on a thin 3D stencil is ~1e5 and wrecks S1; only
            # fit second derivatives when the knn ball is genuinely 3-D.
            thin && continue
            for β in α:dim
                fill!(rhs, 0.0)
                @inbounds for b in 1:m
                    Hij = _rbf_hess(rbf, xi, pts[ids[b]], dim)
                    rhs[b] = Hij[α, β]
                end
                _poly_hess_rhs!(rhs, m, dim, deg, α, β)
                coef = _local_rbf_solve(A, rhs)
                @inbounds for (t, j) in enumerate(ids)
                    Hess[α, β][i, j] = coef[t]
                    α != β && (Hess[β, α][i, j] = coef[t])
                end
            end
        end
    end
    return D, Hess
end

function _stencil_is_planar(pts, ids, dim::Int)
    dim < 3 && return false
    c = zero(pts[ids[1]])
    @inbounds for j in ids
        c += pts[j]
    end
    c /= length(ids)
    C = zeros(dim, dim)
    @inbounds for j in ids
        v = pts[j] - c
        for α in 1:dim, β in 1:dim
            C[α, β] += v[α] * v[β]
        end
    end
    ev = eigvals(Symmetric(C))
    # Boundary knn stencils in 3D are thin slabs (face + a few interiors).
    # Quadratic 3D monomials on those pancakes are ill-conditioned (~1e15 Hess).
    return ev[1] < 0.05 * (ev[end] + 1e-30)
end

function _local_rbf_solve(A, rhs)
    try
        return A \ rhs
    catch e
        (e isa SingularException || e isa LinearAlgebra.LAPACKException) || rethrow()
        return pinv(A) * rhs
    end
end

function _poly_hess_rhs!(rhs, m, dim, deg, α, β)
    deg < 2 && return nothing
    if dim == 2
        # [1, x, y, xy, x², y²]
        if α == 1 && β == 1
            rhs[m + 5] = 2.0
        elseif α == 2 && β == 2
            rhs[m + 6] = 2.0
        elseif α != β
            rhs[m + 4] = 1.0
        end
    elseif dim == 3 && length(rhs) >= m + 10
        if α == 1 && β == 2
            rhs[m + 5] = 1.0
        elseif α == 1 && β == 3
            rhs[m + 6] = 1.0
        elseif α == 2 && β == 3
            rhs[m + 7] = 1.0
        elseif α == 1 && β == 1
            rhs[m + 8] = 2.0
        elseif α == 2 && β == 2
            rhs[m + 9] = 2.0
        elseif α == 3 && β == 3
            rhs[m + 10] = 2.0
        end
    end
    return nothing
end

function _rbf_hess(rbf, x::Point, xi::Point, dim::Int)
    H = zeros(dim, dim)
    d = x - xi
    r = norm(d)
    r < 1e-14 && return H
    if rbf isa PHS3
        invr = 1 / r
        @inbounds for α in 1:dim, β in 1:dim
            H[α, β] = 3 * ((α == β ? r : 0.0) + d[α] * d[β] * invr)
        end
    elseif rbf isa PHS1
        invr = 1 / r
        invr3 = invr^3
        @inbounds for α in 1:dim, β in 1:dim
            H[α, β] = (α == β ? invr : 0.0) - d[α] * d[β] * invr3
        end
    else
        h = 1e-6 * max(r, 1.0)
        @inbounds for β in 1:dim
            xp = _shift_coord(x, β, h)
            xm = _shift_coord(x, β, -h)
            for α in 1:dim
                H[α, β] = (∂(rbf, α, xp, xi) - ∂(rbf, α, xm, xi)) / (2h)
            end
        end
    end
    return H
end

"""Fill interpolant Hessian columns of the polynomial tail (`deg≥2`)."""
function _monomial_hess!(Hx, i::Int, n::Int, dim::Int, deg::Int)
    deg < 2 && return nothing
    if dim == 2
        # [1, x, y, xy, x², y²]
        Hx[1, 2][i, n + 4] = 1.0
        Hx[2, 1][i, n + 4] = 1.0
        Hx[1, 1][i, n + 5] = 2.0
        Hx[2, 2][i, n + 6] = 2.0
    elseif dim == 3
        # [1, x, y, z, xy, xz, yz, x², y², z²]
        Hx[1, 2][i, n + 5] = 1.0
        Hx[2, 1][i, n + 5] = 1.0
        Hx[1, 3][i, n + 6] = 1.0
        Hx[3, 1][i, n + 6] = 1.0
        Hx[2, 3][i, n + 7] = 1.0
        Hx[3, 2][i, n + 7] = 1.0
        Hx[1, 1][i, n + 8] = 2.0
        Hx[2, 2][i, n + 9] = 2.0
        Hx[3, 3][i, n + 10] = 2.0
    end
    return nothing
end

"""Quadratic least-squares Hessian at interior nodes. Boundary rows stay 0.

No PHS: surface knn + PHS Hessian is O(1e5) in 3D. Monomial Hess of
``[1,x,y,z,xy,xz,yz,x²,y²,z²]`` (and the 2D analogue) is exact on
quadratics and zero on linears.
"""
function _interior_quadratic_hess(pts, n::Int, nlocal::Int)
    nt = length(pts)
    dim = length(pts[1])
    Hess = Matrix{Float64}[zeros(nt, nt) for _ in 1:dim, _ in 1:dim]
    n >= nt && return Hess
    deg = 2
    np = rbf_npoly(dim, deg)
    nlocal = clamp(max(Int(nlocal), np + dim), np + 2, nt)
    tree = _rbf_kdtree(pts)
    mon = MonomialBasis(dim, deg)
    hrow = [zeros(np) for _ in 1:dim, _ in 1:dim]
    if dim == 2
        hrow[1, 2][4] = 1.0; hrow[2, 1][4] = 1.0
        hrow[1, 1][5] = 2.0
        hrow[2, 2][6] = 2.0
    else
        hrow[1, 2][5] = 1.0; hrow[2, 1][5] = 1.0
        hrow[1, 3][6] = 1.0; hrow[3, 1][6] = 1.0
        hrow[2, 3][7] = 1.0; hrow[3, 2][7] = 1.0
        hrow[1, 1][8] = 2.0
        hrow[2, 2][9] = 2.0
        hrow[3, 3][10] = 2.0
    end
    @inbounds for i in (n + 1):nt
        ids = rbf_neighbors(tree, pts[i], nlocal)
        _stencil_is_planar(pts, ids, dim) && continue
        m = length(ids)
        m < np && continue
        V = zeros(m, np)
        for (a, j) in enumerate(ids)
            p = mon(pts[j])
            for k in 1:np
                V[a, k] = p[k]
            end
        end
        P = pinv(V)
        for α in 1:dim, β in α:dim
            w = P' * hrow[α, β]
            for (t, j) in enumerate(ids)
                Hess[α, β][i, j] = w[t]
                α != β && (Hess[β, α][i, j] = w[t])
            end
        end
    end
    return Hess
end

"""
    solve_anisotropic_dibem!(dad, K; b=0, rbf=PHS(3; poly_deg=2), kiso=nothing) -> T

Strategy 1: ``∇·(K∇u)=-b`` as isotropic Poisson ``∇²u = -b/k - f*`` with
``k=(det K)^{1/d}`` (override with `kiso` when ``K`` is singular),
``f* = (1/k)∇·(ΔK∇u)``.

Package DIBEM is ``H u - G q = M\\,∇²u``, so the residual enters as
``H u - G q = -M(b/k + f*)``. `dad` is `Laplace(1)` with `H,G,M`.
2D: local RBF-FD Hessian (`nlocal`) so a hole does not pollute a global
PHS interpolant. 3D: quadratic least-squares Hessian at **interior**
nodes only (PHS Hessian on surface knn slabs is ~1e5). Distinct from
strategy 3, which never forms ``\\mathrm{Hess}(u)``.
Pass `kiso` to override ``(det K)^{1/d}`` (needed when ``K`` is singular
but ``∇·(K∇u)=0`` still holds, e.g. ``K=[1 1; 1 1]`` with ``kiso=1``).
"""
function solve_anisotropic_dibem!(dad::BEMdata{<:Laplace}, K;
        b=0, rbf=PHS(3; poly_deg=2), npg::Int=12, nlocal::Int=21,
        kiso::Union{Nothing,Real}=nothing)
    dim = dad.dimension
    Km = _as_Kmat(K, dim)
    kiso = kiso === nothing ? isotropic_scale(Km) : float(kiso)
    abs(kiso) > 1e-15 || throw(ArgumentError(
        "solve_anisotropic_dibem!: isotropic scale must be nonzero (got $kiso)"))
    ΔK = Matrix(Km) - kiso * Matrix{Float64}(I, dim, dim)
    has_cache(dad, :H) || assemble!(dad; npg=npg, threaded=false)
    has_cache(dad, :M) || DIBEM(dad; rbf=rbf)
    pts = all_points(dad)
    nt = dad.nt
    n = dad.n
    Af = zeros(nt, nt)
    Hess = if dim == 3
        _interior_quadratic_hess(pts, n, max(nlocal, 40))
    else
        rbf_diff_ops(pts, rbf; nlocal=nlocal)[2]
    end
    @inbounds for α in 1:dim, β in 1:dim
        abs(ΔK[α, β]) < 1e-15 && continue
        Af .+= (ΔK[α, β] / kiso) .* Hess[α, β]
    end
    H = Matrix{Float64}(dad.H)
    G = Matrix{Float64}(dad.G)
    M = Matrix{Float64}(dad.M)
    # ∇²u = -b/k - f*  ⇒  H u - G q_iso = -M(b/k) - M Af u
    # q_iso from q_pkg and shape-function ∇_Γ (same split as S3; a global
    # RBF gradient is unreliable on a hole).
    L = H + M * Af
    A = copy(L)
    fv = _eval_aniso_source(b, pts) ./ kiso
    rhs = .-M * fv
    S = _boundary_tangential_grad(dad)
    nΔn = zeros(n)
    ΔKn = [ΔK * dad.Normal[j] for j in 1:n]
    @inbounds for j in 1:n
        nΔn[j] = dot(dad.Normal[j], ΔKn[j])
    end
    SΓ = zeros(n, n)
    @inbounds for β in 1:dim, j in 1:n
        SΓ[j, :] .+= ΔKn[j][β] .* view(S[β], j, :)
    end
    @inbounds for j in 1:n
        dad.BC[j] == 1 || continue
        d = kiso + nΔn[j]
        col = abs(d) < 1e-15 ? G[:, j] : G[:, j] ./ d
        A[:, 1:n] .-= col * transpose(view(SΓ, j, :))
        rhs .+= col .* dad.BV[j]
    end
    Au = copy(A)
    @inbounds for j in 1:n
        dad.BC[j] == 0 || continue
        rhs .-= Au[:, j] .* dad.BV[j]
        A[:, j] .= .-G[:, j]
    end
    x = bem_linsolve(A, rhs)
    Tfull = zeros(nt)
    qfull = zeros(n)
    Tfull[1:length(x)] .= x
    split_sol!(dad, Tfull, qfull)
    _dirichlet_q_iso_to_pkg!(dad, kiso, ΔK, Tfull, qfull)
    set_cache!(dad; T=Tfull, q=qfull, A=A, b=rhs, aniso_Af=Af)
    return dad.T
end

"""Dirichlet unknowns in the isotropic BIE are `q_iso=-∂u/∂n`. Convert to
package flux `q=-n·K∇u = (kiso + n·ΔK n) q_iso − n·ΔK ∇_Γ u`."""
function _dirichlet_q_iso_to_pkg!(dad, kiso, ΔK, Tfull, qfull)
    n = dad.n
    dim = dad.dimension
    S = _boundary_tangential_grad(dad)
    u = view(Tfull, 1:n)
    @inbounds for j in 1:n
        dad.BC[j] == 0 || continue
        nj = dad.Normal[j]
        ΔKn = ΔK * nj
        nΔn = dot(nj, ΔKn)
        sΓ = 0.0
        for β in 1:dim
            sΓ += ΔKn[β] * dot(view(S[β], j, :), u)
        end
        qfull[j] = (kiso + nΔn) * qfull[j] - sΓ
    end
    return qfull
end

# ---------------------------------------------------------------------------
# Strategy 3: one IBP, then DIBEM (no Hess(u), no ∂φ in the RIM).
#
#   ∫_Ω Φ (b/k + f*) dΩ
#     = (1/k) ∫_Ω Φ b dΩ
#     + (1/k) ∫_Γ Φ (n·ΔK ∇u) dΓ
#     − (1/k) ∫_Ω ∇Φ · (ΔK ∇u) dΩ
#
# ∫ Φ b  → existing M (IF from int(rbf, x, xj)).
# ∫_Γ Φ (n·ΔK ∇u) → G v,  v from q and shape-function ∇_Γ.
# ∫ ∇Φ · w, w = ΔK ∇u → Loeffler DIBEM with kernel ∇Φ: same c (int(rbf)),
#   ID from the analytic RIM primitive of ∇Φ.
# ---------------------------------------------------------------------------

"""
    solve_anisotropic_ibp!(dad, K; b=0, rbf=PHS(3; poly_deg=2)) -> T

Strategy 3: integrate ``∇·(ΔK∇u)`` by parts once, then DIBEM. Interpolates
``u`` / the vector ``ΔK∇u`` by values; radial integrals are `int(rbf,x,xj)`
and the closed-form RIM primitive of ``∇Φ``. No Hessian of ``u``.
"""
function solve_anisotropic_ibp!(dad::BEMdata{<:Laplace}, K;
        b=0, rbf=PHS(3; poly_deg=2), npg::Int=12,
        nlocal::Union{Nothing,Integer}=21)
    dim = dad.dimension
    Km = _as_Kmat(K, dim)
    kiso = isotropic_scale(Km)
    ΔK = Matrix(Km) - kiso * Matrix{Float64}(I, dim, dim)
    has_cache(dad, :H) || assemble!(dad; npg=npg, threaded=false)
    has_cache(dad, :M) || DIBEM(dad; rbf=rbf)
    has_cache(dad, :dibem_c) || DIBEM(dad; rbf=rbf)
    has_cache(dad, :qsi) || _init_quadrature!(dad, npg)
    n = dad.n
    nt = dad.nt
    H = Matrix{Float64}(dad.H)
    G = Matrix{Float64}(dad.G)
    M = Matrix{Float64}(dad.M)
    invk = 1 / kiso
    fv = _eval_aniso_source(b, all_points(dad))
    rhs = .-(invk) .* (M * fv)
    A = copy(H)
    if maximum(abs, ΔK) < 1e-15
        Gq = G
        Au = copy(A)
        @inbounds for j in 1:n
            if dad.BC[j] == 0
                rhs .-= Au[:, j] .* dad.BV[j]
                A[:, j] .= .-Gq[:, j]
            else
                rhs .+= Gq[:, j] .* dad.BV[j]
            end
        end
        x = bem_linsolve(A, rhs)
        Tfull = zeros(nt)
        qfull = zeros(n)
        Tfull[1:length(x)] .= x
        split_sol!(dad, Tfull, qfull)
        set_cache!(dad; T=Tfull, q=qfull, A=A, b=rhs)
        return dad.T
    end
    S = _boundary_tangential_grad(dad)
    nΔn = zeros(n)
    ΔKn = [ΔK * dad.Normal[j] for j in 1:n]
    @inbounds for j in 1:n
        nΔn[j] = dot(dad.Normal[j], ΔKn[j])
    end
    SΓ = zeros(n, n)
    @inbounds for β in 1:dim, j in 1:n
        SΓ[j, :] .+= ΔKn[j][β] .* view(S[β], j, :)
    end
    # Volume: w = ΔK ∇u with ∇u from a local RBF interpolant (global PHS
    # through a hole pollutes ∇u the same way it pollutes the Hessian).
    N = _dibem_gradPhi_operators(dad)
    nloc = if nlocal === nothing
        nothing
    elseif dim == 3 && dad.ni == 0
        nothing
    elseif dim == 3
        max(Int(nlocal), 50)
    else
        Int(nlocal)
    end
    Du, _ = rbf_diff_ops(all_points(dad), rbf; nlocal=nloc)
    AV = zeros(nt, nt)
    @inbounds for α in 1:dim, β in 1:dim
        abs(ΔK[α, β]) < 1e-15 && continue
        AV .+= ΔK[α, β] .* (N[α] * Du[β])
    end
    # H u − G q + (1/k) G (SΓ u − nΔn q) − (1/k) AV u = −M b/k
    A[:, 1:n] .+= invk .* (G * SΓ)
    A .-= invk .* AV
    Gq = copy(G)
    @inbounds for j in 1:n
        Gq[:, j] .*= (1 + invk * nΔn[j])
    end
    @inbounds for j in 1:n
        dad.BC[j] == 1 || continue
        d = kiso + nΔn[j]
        col = abs(d) < 1e-15 ? Gq[:, j] : Gq[:, j] ./ d
        A[:, 1:n] .-= col * transpose(view(SΓ, j, :))
        rhs .+= col .* dad.BV[j]
    end
    Au = copy(A)
    @inbounds for j in 1:n
        dad.BC[j] == 0 || continue
        rhs .-= Au[:, j] .* dad.BV[j]
        A[:, j] .= .-Gq[:, j]
    end
    x = bem_linsolve(A, rhs)
    Tfull = zeros(nt)
    qfull = zeros(n)
    Tfull[1:length(x)] .= x
    split_sol!(dad, Tfull, qfull)
    _dirichlet_q_iso_to_pkg!(dad, kiso, ΔK, Tfull, qfull)
    set_cache!(dad; T=Tfull, q=qfull, A=A, b=rhs)
    return dad.T
end

"""
    anisotropic_wave_shift!(dad, K; strategy=:ibp) -> dad

Rewrite isotropic DIBEM wave operators for ``p_{tt} = ∇·(K∇p) + f``:

    H_eff p − G_eff q = (M/k) (p_{tt} − f)

Call after `H_G_full_direct` / `assemble!` and `DIBEM`. `dad` is `Laplace(1)`.

* `:ibp` (default) — Green's theorem once on ``∇·(ΔK∇p)``: interpolates
  ``ΔK∇p`` by values, no Hessian of ``p``. Same volume/boundary split as
  [`solve_anisotropic_ibp!`](@ref).
* `:hess` — RBF Hessian residual of [`solve_anisotropic_dibem!`](@ref).
"""
function anisotropic_wave_shift!(dad::BEMdata{<:Laplace}, K;
        strategy::Symbol=:ibp, rbf=PHS(3; poly_deg=2),
        nlocal::Union{Nothing,Integer}=nothing, npg::Int=12)
    strategy === :ibp || strategy === :hess ||
        throw(ArgumentError("strategy must be :ibp or :hess (got $strategy)"))
    dim = dad.dimension
    Km = _as_Kmat(K, dim)
    kiso = isotropic_scale(Km)
    ΔK = Matrix(Km) - kiso * Matrix{Float64}(I, dim, dim)
    has_cache(dad, :H) || error("call assemble! / H_G_full_direct first")
    has_cache(dad, :M) || DIBEM(dad; rbf=rbf)
    n = dad.n
    nt = dad.nt
    if strategy === :hess
        Af = zeros(nt, nt)
        if maximum(abs, ΔK) > 1e-15
            Hess = if dim == 3
                _interior_quadratic_hess(all_points(dad), n, max(nlocal === nothing ? 40 : Int(nlocal), 40))
            else
                rbf_diff_ops(all_points(dad), rbf; nlocal=nlocal)[2]
            end
            @inbounds for α in 1:dim, β in 1:dim
                abs(ΔK[α, β]) < 1e-15 && continue
                Af .+= (ΔK[α, β] / kiso) .* Hess[α, β]
            end
            set_cache!(dad; H=dad.H + dad.M * Af)
        end
        set_cache!(dad; M=dad.M ./ kiso, aniso_Af=Af, aniso_kiso=kiso,
            aniso_strategy=:hess)
        return dad
    end
    has_cache(dad, :dibem_c) || DIBEM(dad; rbf=rbf)
    has_cache(dad, :qsi) || _init_quadrature!(dad, npg)
    H = Matrix{Float64}(dad.H)
    G = Matrix{Float64}(dad.G)
    M = Matrix{Float64}(dad.M)
    invk = 1 / kiso
    if maximum(abs, ΔK) > 1e-15
        S = _boundary_tangential_grad(dad)
        nΔn = zeros(n)
        ΔKn = [ΔK * dad.Normal[j] for j in 1:n]
        @inbounds for j in 1:n
            nΔn[j] = dot(dad.Normal[j], ΔKn[j])
        end
        SΓ = zeros(n, n)
        @inbounds for β in 1:dim, j in 1:n
            SΓ[j, :] .+= ΔKn[j][β] .* view(S[β], j, :)
        end
        N = _dibem_gradPhi_operators(dad)
        nloc = if dim == 3
            dad.ni == 0 ? nothing : (nlocal === nothing ? 50 : max(Int(nlocal), 50))
        else
            nlocal
        end
        Du, _ = rbf_diff_ops(all_points(dad), rbf; nlocal=nloc)
        AV = zeros(nt, nt)
        @inbounds for α in 1:dim, β in 1:dim
            abs(ΔK[α, β]) < 1e-15 && continue
            AV .+= ΔK[α, β] .* (N[α] * Du[β])
        end
        H[:, 1:n] .+= invk .* (G * SΓ)
        H .-= invk .* AV
        @inbounds for j in 1:n
            G[:, j] .*= (1 + invk * nΔn[j])
        end
        set_cache!(dad; aniso_AV=AV)
    end
    set_cache!(dad; H=H, G=G, M=M ./ kiso, aniso_kiso=kiso, aniso_strategy=:ibp)
    return dad
end

"""Loeffler ``N_α``: ``(N_α β) ≈ ∫_Ω ∂_α Φ\\,β\\,dΩ``. Same `c` as DIBEM (`int(rbf)`)."""
function _dibem_gradPhi_operators(dad::BEMdata{<:Laplace})
    dim = dad.dimension
    nt = dad.nt
    c = collect(Float64, dad.dibem_c)
    length(c) == nt || throw(DimensionMismatch("dibem_c length $(length(c)) ≠ nt=$nt"))
    D∇, ID∇ = _gradPhi_D_ID(dad)
    return ntuple(α -> _loeffler_op(D∇[α], c, ID∇[α]), dim)
end

function _loeffler_op(D, c, ID)
    M = D .* c'
    @inbounds for i in eachindex(ID)
        M[i, i] = 0.0
        M[i, i] = -sum(view(M, i, :)) + ID[i]
    end
    return M
end

"""Pointwise ``∇_X Φ(ξ_i, x_j)`` and RIM ``∫ ∇_X Φ\\,dΩ``. Primitive ``-R e/(c k)``."""
function _gradPhi_D_ID(dad::BEMdata{<:Laplace})
    dim = dad.dimension
    nt = dad.nt
    kcond = float(dad.properties.k)
    pts = all_points(dad)
    D∇ = ntuple(_ -> zeros(nt, nt), dim)
    invck = dim == 2 ? 1 / (2π * kcond) : 1 / (4π * kcond)
    @inbounds for j in 1:nt, i in 1:nt
        i == j && continue
        r = pts[j] - pts[i]
        R = norm(r)
        R < 1e-15 && continue
        e = r / R
        coef = dim == 2 ? -invck / R : -invck / (R * R)
        for α in 1:dim
            D∇[α][i, j] = coef * e[α]
        end
    end
    ID∇ = ntuple(_ -> zeros(nt), dim)
    _accumulate_IgradPhi!(ID∇, dad, invck)
    return D∇, ID∇
end

function _accumulate_IgradPhi!(ID∇, dad, invck)
    geos = _rim_build_elements(dad)
    dim = dad.dimension
    dimv = Val(Int(dim))
    _dibem_src_loop!(dad.nt, true) do i
        x = point(dad, i)
        @inbounds for g in geos
            if _near_element(x, g.nodes, g.el)
                for q in eachindex(g.wJ)
                    wJ = g.wJ[q]
                    wJ == 0 && continue
                    r = g.y[q] - x
                    R = norm(r)
                    R < 1e-14 && continue
                    e = r / R
                    s = -R * invck * (wJ * _rim_factor(g.n[q], r, R, dimv))
                    for α in 1:dim
                        ID∇[α][i] += s * e[α]
                    end
                end
            else
                for a in eachindex(g.xj)
                    r = g.xj[a] - x
                    R = norm(r)
                    R < 1e-10 && continue
                    e = r / R
                    s = -R * invck * (g.wj[a] * _rim_factor(g.nj[a], r, R, dimv))
                    for α in 1:dim
                        ID∇[α][i] += s * e[α]
                    end
                end
            end
        end
    end
    return nothing
end

"""Tangential ``∇_Γ`` at boundary nodes from element shape functions."""
function _boundary_tangential_grad(dad)
    n = dad.n
    dim = dad.dimension
    S = ntuple(_ -> zeros(n, n), dim)
    poly = dad.element_type
    if dim == 2
        ξs = poly.nodes
        @inbounds for el in dad.elements
            idx = el.index
            nn = length(idx)
            for k in 1:nn
                i = idx[k]
                ξ = ξs[min(k, length(ξs))]
                _, dN = shapefun(poly, ξ)
                J = el.Jacobian[k]
                J < 1e-16 && continue
                n̂ = dad.Normal[i]
                # t̂ with tan2normal(t̂)=n̂ = (t_y, -t_x); J from geometry (quadratic OK)
                t1 = -n̂[2]
                t2 = n̂[1]
                invJ = 1 / J
                for j in 1:nn
                    coef = dN[1, j] * invJ
                    S[1][i, idx[j]] += t1 * coef
                    S[2][i, idx[j]] += t2 * coef
                end
            end
        end
    else
        ξs = poly.nodes
        nξ = length(ξs)
        @inbounds for el in dad.elements
            idx = el.index
            X = dad.Nodes[idx]
            nn = length(idx)
            nN = nn
            for k in 1:nn
                iξ = (k - 1) % nξ + 1
                iη = (k - 1) ÷ nξ + 1
                L, Lξ, Lη = shapefun2D(poly, poly, ξs[iξ], ξs[min(iη, nξ)])
                nN = size(L, 2)
                xξ = zero(X[1])
                xη = zero(X[1])
                for j in 1:nN
                    xξ += Lξ[1, j] * X[j]
                    xη += Lη[1, j] * X[j]
                end
                Gmat = hcat(SVector(xξ[1], xξ[2], xξ[3]),
                    SVector(xη[1], xη[2], xη[3]))
                GtG = Gmat' * Gmat
                det(GtG) < 1e-20 && continue
                i = idx[k]
                # ∇_Γ u = G (GtG)\ (u_ξ, u_η)
                for j in 1:nN
                    rhsξ = SVector(Lξ[1, j], Lη[1, j])
                    αcoef = GtG \ rhsξ
                    g = Gmat * αcoef
                    for α in 1:3
                        S[α][i, idx[j]] += g[α]
                    end
                end
            end
        end
    end
    return S
end

function _as_Kmat(K::AnisotropicLaplace, dim::Int)
    size(K.K, 1) == dim || throw(ArgumentError("K dimension mismatch"))
    return Matrix(K.K)
end
function _as_Kmat(K::OrthotropicLaplace, dim::Int)
    dim == 2 || throw(ArgumentError("OrthotropicLaplace is 2D"))
    return [K.k1 0.0; 0.0 K.k2]
end
function _as_Kmat(K::AbstractMatrix, dim::Int)
    size(K, 1) == dim && size(K, 2) == dim || throw(ArgumentError("K must be $dim×$dim"))
    return Matrix{Float64}(K)
end

function _eval_aniso_source(b, pts)
    n = length(pts)
    b isa Number && return fill(float(b), n)
    b isa AbstractVector && return collect(Float64, b)
    return Float64[float(b(p)) for p in pts]
end
