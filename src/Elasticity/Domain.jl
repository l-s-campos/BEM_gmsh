export DIBEM_dense, dibem_elasticity!, dibem_matrix

"""
    DIBEM_dense(dad::BEMdata{<:Elasticity}; rbf=PHS(), npg=12)

Dense **Direct Interpolation BEM** mass `M` for isotropic elasticity.
Port of `Monta_M_RIMd` / `calc_md`: the whole integrand ``U* b`` is
interpolated with RBFs, then the domain integrals of the RBF and of
Kelvin ``U*`` are taken to the boundary by radial integration (Gauss
on every element).

```
F c = IF ,   M-block_ij = c_j U*(x_i, x_j)  (i≠j) ,
M-block_ii = ID_i − ∑_{j≠i} M-block_ij
```

`M` is `(d·nt)×(d·nt)` (`d = dad.dimension`), stored in `dad.cache.M`.
Compressed backends (2-D) use the same `c` / `ID` with a structured `U*`.
"""
function DIBEM_dense(dad::BEMdata{<:Elasticity}; rbf=PHS(), npg::Int=12,
        centers::Symbol=:collocation, threaded::Bool=true, rim::Symbol=:lumped,
        kwargs...)
    centers in (:collocation, :cells) || throw(ArgumentError(
        "DIBEM centers must be :collocation or :cells (got $centers)"))
    if centers === :cells
        dad.dimension == 2 || error("DIBEM elasticity centers=:cells is 2D only")
        return _DIBEM_dense_volume_centers(dad, rbf; npg=npg)
    end
    nt = dad.nt
    dim = dad.dimension
    n0 = dad.Normal[1]
    IF, ID, pts = _dibem_elast_IF_ID(dad, rbf; npg=npg, threaded=threaded, rim=rim)

    ndof = dim * nt
    F = zeros(nt, nt)
    D = zeros(ndof, ndof)
    _dibem_src_loop!(nt, threaded) do j
        xj = pts[j]
        @inbounds for i in 1:nt
            rvec = xj - pts[i]
            R = norm(rvec)
            F[i, j] = rbf(R)
            R > 0 || continue
            U, _ = fundamental(dad, rvec, n0)
            D[expand(i, dim), expand(j, dim)] .= U
        end
    end
    _dibem_ridge_F!(F)

    IP = _dibem_monomial_IP(dad, rbf)
    c = _dibem_poly_c(F, IF, pts, rbf; IP=IP)
    M = zeros(ndof, ndof)
    @inbounds for j in 1:nt
        a = c[j]
        cols = expand(j, dim)
        for d in 1:dim
            M[:, cols[d]] .= a .* D[:, cols[d]]
        end
    end
    @inbounds for i in 1:nt
        rows = expand(i, dim)
        M[rows, rows] .= 0
        S = zeros(dim, dim)
        for d in 1:dim
            S[:, d] = vec(sum(view(M, rows, d:dim:ndof); dims=2))
        end
        M[rows, rows] .= .-S .+ ID[rows, :]
    end
    M .*= dad.properties.rho
    set_cache!(dad; M, dibem_F=F, dibem_rbf=rbf, dibem_method=:dense,
        dibem_c=c, dibem_ID=ID, dibem_D=D, dibem_centers=:collocation)
    return M
end

function _DIBEM_dense_volume_centers(dad::BEMdata{<:Elasticity}, rbf; npg::Int=12)
    ξ = _dibem_volume_centers(dad)
    pts = all_points(dad)
    nc = length(ξ)
    nt = dad.nt
    n0 = dad.Normal[1]
    F = zeros(nc, nc)
    D = zeros(2nt, 2nc)
    @showprogress "Assembling F (cell centers) and D (elasticity)" for k in 1:nc
        @inbounds for j in 1:nc
            r = norm(ξ[j] - ξ[k])
            r > 0 && (F[j, k] = rbf(r))
        end
        @inbounds for i in 1:nt
            r = ξ[k] - pts[i]
            R = norm(r)
            R < 1e-15 && continue
            U, _ = fundamental(dad, r, n0)
            D[2i-1:2i, 2k-1:2k] .= U
        end
    end
    _dibem_ridge_F!(F)
    IF = _dibem_rbf_IF(dad, rbf, ξ; npg=npg)
    _, ID, _ = _dibem_elast_IF_ID(dad, rbf; npg=npg)
    IP = _dibem_monomial_IP(dad, rbf)
    c = _dibem_poly_c(F, IF, ξ, rbf; IP=IP)
    Q = _dibem_center_Q(pts, ξ)
    M = _dibem_M_volume_elast(D, c, Q, ID)
    M .*= dad.properties.rho
    set_cache!(dad; M, dibem_F=F, dibem_c=c, dibem_ID=ID, dibem_rbf=rbf,
        dibem_Q=Q, dibem_method=:dense, dibem_centers=:cells)
    return M
end

"""
    dibem_elasticity!(dad; rbf=PHS(), kwargs...)

Alias for [`DIBEM_dense`](@ref)`(dad::BEMdata{<:Elasticity})`. Kept for
callers in thermoelasticity / plates.
"""
function dibem_elasticity!(dad::BEMdata{<:Elasticity}; rbf=PHS(), kwargs...)
    return DIBEM_dense(dad; rbf=rbf, kwargs...)
end

"""
    dibem_matrix(dad::BEMdata{<:Elasticity}; rbf=PHS(), rebuild=false, method=:dense)

Return the elasticity DIBEM operator `M`. Rebuilds via [`DIBEM`](@ref) if
missing or `rebuild=true`.
"""
function dibem_matrix(dad::BEMdata{<:Elasticity}; rbf=PHS(), rebuild::Bool=false,
        method::Symbol=:dense, kwargs...)
    if rebuild || !has_cache(dad, :M)
        return DIBEM(dad; method=method, rbf=rbf, kwargs...)
    end
    return dad.M
end

"""
    _galerkin_Ustar(props, R, e) → SMatrix{d,d}

Radial particular of the Kelvin single layer along `e = r/R`.
2-D: ``∫_0^R U*(ρ e) ρ dρ``. 3-D: ``∫_0^R U*(ρ e) ρ² dρ``.

```
Û*(R,e) = ∫_0^R U*(ρ e) ρ dρ
```

so that `∫_Ω U* dΩ = ∫_Γ Û* (n·r/R²) dΓ`.

With
`U* = [(3-4ν) log(1/ρ) I + e⊗e] / (8π μ (1-ν))`:

```
Û* = [ (3-4ν)(R²/4 − R²/2 log R) I + (R²/2) e⊗e ] / (8π μ (1-ν))
```
"""
function _galerkin_Ustar(props::Elasticity, R::Real, e::SVector{2})
    ν = effective_nu(props)
    μ = props.mu
    base = 8 * π * μ * (1 - ν)
    R2 = R * R
    logR = log(R)
    c_iso = (3 - 4ν) * (R2 / 4 - R2 / 2 * logR)
    c_dir = R2 / 2
    # (c_iso * I + c_dir * e⊗e) / base
    e1, e2 = e[1], e[2]
    return @SMatrix [
        (c_iso + c_dir * e1 * e1)/base    (c_dir * e1 * e2)/base
        (c_dir * e2 * e1)/base            (c_iso + c_dir * e2 * e2)/base
    ]
end

"""Radial particular of 3-D Kelvin: ``∫_0^R U*(ρ e) ρ² dρ``.

``U* = [(3-4ν)I + e⊗e] / (16π μ (1-ν) R)``, so the primitive is
``[(3-4ν)I + e⊗e] R² / (32π μ (1-ν))``.
"""
function _galerkin_Ustar(props::Elasticity, R::Real, e::SVector{3})
    ν = props.nu
    μ = props.mu
    base = 16 * π * μ * (1 - ν)
    c = (R * R / 2) / base
    e1, e2, e3 = e[1], e[2], e[3]
    ciso = (3 - 4ν) * c
    return @SMatrix [
        ciso + c * e1 * e1   c * e1 * e2         c * e1 * e3
        c * e2 * e1          ciso + c * e2 * e2  c * e2 * e3
        c * e3 * e1          c * e3 * e2         ciso + c * e3 * e3
    ]
end
_cell_U_primitive(props::Elasticity, R, e) = _galerkin_Ustar(props, R, e)
# ---------------------------------------------------------------------------
# Dual Reciprocity (DRM) for 2D elasticity
#
# kernel = :r          f = r              (û ~ r³, t̂ ~ r²)
# kernel = :one_plus_r f = 1+r            (û ~ r² + r³)
# kernel = :mq         f = √(r²+C²)       (Galerkin of ∇⁴g = f; C=0.01 paper)
#
# Book / Galerkin û, t̂ satisfy L(û) = +f I. Navier here is L(u)+b = 0, so they
# enter HΨ − Gη with a global minus. Traction is closed-form, not FD.
# ---------------------------------------------------------------------------

"""
    build_drm_matrices(dad::BEMdata{<:Elasticity}; npg=12, kernel=:r, C=0.01)

DRM mass for 2D elasticity.

- `:r` — ``f = r``
- `:one_plus_r` — ``f = 1+r``
- `:mq` — Samaan & Rashed (2007) ``f = √(r²+C²)`` (default `C=0.01`)

Stores `M` so ``H u - G t = M b``.
"""
function build_drm_matrices(dad::BEMdata{<:Elasticity}; npg::Int = 12,
        kernel::Symbol = :r, C::Real = 0.01, poly_deg::Int = -1)
    dad.dimension == 3 && return _build_drm_matrices_3d(dad; npg=npg)
    kernel === :r || kernel === :one_plus_r || kernel === :mq ||
        throw(ArgumentError("kernel must be :r, :one_plus_r, or :mq (got $kernel)"))
    ker = Val(kernel)
    has_cache(dad, :H) || H_G_full_direct(dad; npg = npg, threaded = false)
    H = Matrix(dad.H)
    G = Matrix(dad.G)
    nt = dad.nt
    n = dad.n
    props = dad.properties
    nu = effective_nu(props)
    Gmod = shear_modulus(props)

    ndof = 2nt
    nb = 2n
    pts = all_points(dad)

    F = zeros(ndof, ndof)
    Ψ = zeros(ndof, ndof)
    @inbounds for j = 1:nt, i = 1:nt
        rvec = pts[i] - pts[j]
        φ = _drm_phi(norm(rvec), ker, C)
        F[2i-1:2i, 2j-1:2j] .= φ * I(2)
        Ψ[2i-1:2i, 2j-1:2j] .= .-_drm_u(rvec, nu, Gmod, ker, C)
    end

    ε = 1e-12 * (sum(abs, F) / max(ndof^2, 1) + 1)
    @inbounds for k = 1:ndof
        F[k, k] += ε
    end

    η = zeros(nb, ndof)
    @inbounds for j = 1:nt, i = 1:n
        rvec = dad.Nodes[i] - pts[j]
        η[2i-1:2i, 2j-1:2j] .= .-_drm_t(rvec, dad.Normal[i], nu, Gmod, ker, C)
    end

    npoly = poly_deg < 0 ? 0 : rbf_npoly(2, poly_deg)
    if npoly > 0
        ncol = 2 * npoly
        P = zeros(ndof, ncol)
        Ψp = zeros(ndof, ncol)
        ηp = zeros(nb, ncol)
        @inbounds for k in 1:npoly, d in 1:2
            col = 2 * (k - 1) + d
            for i in 1:nt
                pk = _elast_mono_value(pts[i], k, poly_deg)
                P[2 * (i - 1) + d, col] = pk
                u = _navier_mono_u(pts[i], nu, Gmod, k, d)
                # L(u)+p e_d = 0, so H u − G t = ∫ U (p e_d) — no extra minus
                Ψp[2i-1:2i, col] .= u
            end
            for i in 1:n
                t = _navier_mono_t(dad.Nodes[i], dad.Normal[i], nu, Gmod, k, d)
                ηp[2i-1:2i, col] .= t
            end
        end
        K = [F P; P' zeros(ncol, ncol)]
        Cfull = [H * Ψ - G * η   H * Ψp - G * ηp]
        W = K \ [Matrix{Float64}(I, ndof, ndof); zeros(ncol, ndof)]
        M = Cfull * W
        M .*= props.rho
        set_cache!(dad; M = M)
        return (; H, G, F, Ψ, η, M, kernel, C, poly_deg, npoly)
    end

    Mop = H * Ψ - G * η
    M = Mop / F
    M .*= props.rho
    set_cache!(dad; M = M)
    return (; H, G, F, Ψ, η, M, kernel, C)
end

"""DRM mass for 3D elasticity (isotropic or anisotropic). Particular solution ``û = (R+R³)I`` (BESLE)."""
function build_drm_matrices(dad::BEMdata{<:AnisotropicElasticity3D}; npg::Int=12, kwargs...)
    return _build_drm_matrices_3d(dad; npg=npg)
end

function _build_drm_matrices_3d(dad; npg::Int=12)
    has_cache(dad, :H) || H_G_full_direct(dad; npg=npg, threaded=false)
    H = Matrix(dad.H)
    G = Matrix(dad.G)
    nt = dad.nt
    n = dad.n
    dim = 3
    ndof = dim * nt
    nb = dim * n
    pts = all_points(dad)
    C4 = dad.properties isa AnisotropicElasticity3D ? dad.properties.C4 :
         voigt_to_tensor(voigt_stiffness(dad.properties, 3))
    ρ = dad.properties.rho

    F = zeros(ndof, ndof)
    Ψ = zeros(ndof, ndof)
    @inbounds for j in 1:nt, i in 1:nt
        rvec = pts[i] - pts[j]
        φ, Uhat = _drm3_phi_U(rvec)
        bi = 3i-2:3i
        bj = 3j-2:3j
        F[bi, bj] .= φ * I(3)
        Ψ[bi, bj] .= .-Uhat
    end
    ε = 1e-12 * (sum(abs, F) / max(ndof^2, 1) + 1)
    @inbounds for k in 1:ndof
        F[k, k] += ε
    end
    η = zeros(nb, ndof)
    @inbounds for j in 1:nt, i in 1:n
        rvec = dad.Nodes[i] - pts[j]
        That = _drm3_T(rvec, dad.Normal[i], C4)
        η[3i-2:3i, 3j-2:3j] .= .-That
    end
    Mop = H * Ψ - G * η
    M = Mop / F
    M .*= ρ
    set_cache!(dad; M=M)
    return (; H, G, F, Ψ, η, M, kernel=:besle)
end

function _drm3_phi_U(rvec::SVector{3})
    R = norm(rvec)
    R < 1e-14 && return 0.0, zeros(3, 3)
    φ = R + R^3
    return φ, φ * I(3)
end

"""Traction particular for ``û = (R+R³)I``. ``ε_{pq}^{(k)} = ½(1+3R²)(δ_{pk} n_q + δ_{qk} n_p)``."""
function _drm3_T(rvec::SVector{3}, nrm::SVector{3}, C4::Array{Float64,4})
    R = norm(rvec)
    T = zeros(3, 3)
    R < 1e-14 && return T
    nh = rvec / R
    fac = 1 + 3 * R^2
    @inbounds for k in 1:3, a in 1:3
        s = 0.0
        for p in 1:3, q in 1:3, b in 1:3
            εpq = 0.5 * fac * (((p == k) ? nh[q] : 0.0) + ((q == k) ? nh[p] : 0.0))
            s += nrm[b] * C4[b, a, p, q] * εpq
        end
        T[a, k] = s
    end
    return T
end

function _elast_mono_value(x, k::Int, deg::Int)
    k == 1 && return 1.0
    deg >= 1 && k == 2 && return x[1]
    deg >= 1 && k == 3 && return x[2]
    deg >= 2 && k == 4 && return x[1] * x[2]
    deg >= 2 && k == 5 && return x[1]^2
    deg >= 2 && k == 6 && return x[2]^2
    return 0.0
end

"""Navier particular for monomial body force `p_k e_d` (2-D)."""
function _navier_mono_u(x, nu::Real, μ::Real, k::Int, d::Int)
    λ = 2μ * nu / (1 - 2nu)
    X, Y = x[1], x[2]
    u = zeros(2)
    if k == 1
        d == 1 ? (u[1] = -Y^2 / (2μ)) : (u[2] = -X^2 / (2μ))
    elseif k == 2
        d == 1 ? (u[1] = -X^3 / (6 * (λ + 2μ))) : (u[2] = -X^3 / (6μ))
    elseif k == 3
        d == 1 ? (u[1] = -Y^3 / (6μ)) : (u[2] = -Y^3 / (6 * (λ + 2μ)))
    end
    return u
end

function _navier_mono_t(x, nrm, nu::Real, μ::Real, k::Int, d::Int)
    λ = 2μ * nu / (1 - 2nu)
    X, Y = x[1], x[2]
    nx, ny = nrm[1], nrm[2]
    if k == 1 && d == 1
        return SVector(-Y * ny, -Y * nx)
    elseif k == 1 && d == 2
        return SVector(-X * ny, -X * nx)
    elseif k == 2 && d == 1
        σxx = -X^2 / 2
        σyy = -λ * X^2 / (2 * (λ + 2μ))
        return SVector(σxx * nx, σyy * ny)
    elseif k == 2 && d == 2
        sxy = -X^2 / 2
        return SVector(sxy * ny, sxy * nx)
    elseif k == 3 && d == 1
        sxy = -Y^2 / 2
        return SVector(sxy * ny, sxy * nx)
    elseif k == 3 && d == 2
        σyy = -Y^2 / 2
        σxx = -λ * Y^2 / (2 * (λ + 2μ))
        return SVector(σxx * nx, σyy * ny)
    end
    return SVector(0.0, 0.0)
end

_drm_phi(r, ::Val{:r}, C=0.01) = r
_drm_phi(r, ::Val{:one_plus_r}, C=0.01) = 1 + r
_drm_phi(r, ::Val{:mq}, C=0.01) = sqrt(r * r + C * C)

# Back-compat: 1+r kernels
_drm_u(rvec::SVector{2}, nu::Real, G::Real) = _drm_u(rvec, nu, G, Val(:one_plus_r))
_drm_t(rvec::SVector{2}, nrm::SVector{2}, nu::Real, G::Real) =
    _drm_t(rvec, nrm, nu, G, Val(:one_plus_r))
_drm_u(rvec::SVector{2}, nu::Real, G::Real, v::Val{:r}, ::Real) =
    _drm_u(rvec, nu, G, v)
_drm_t(rvec::SVector{2}, nrm::SVector{2}, nu::Real, G::Real, v::Val{:r}, ::Real) =
    _drm_t(rvec, nrm, nu, G, v)
_drm_u(rvec::SVector{2}, nu::Real, G::Real, v::Val{:one_plus_r}, ::Real) =
    _drm_u(rvec, nu, G, v)
_drm_t(rvec::SVector{2}, nrm::SVector{2}, nu::Real, G::Real, v::Val{:one_plus_r}, ::Real) =
    _drm_t(rvec, nrm, nu, G, v)
_drm_u(rvec::SVector{2}, nu::Real, G::Real, v::Val{:mq}) =
    _drm_u(rvec, nu, G, v, 0.01)
_drm_t(rvec::SVector{2}, nrm::SVector{2}, nu::Real, G::Real, v::Val{:mq}) =
    _drm_t(rvec, nrm, nu, G, v, 0.01)

"""
Particular displacement for ``f = r``:

```
û_ij = [1/(30(1-ν)μ)] [(3-10ν/3) δ_ij − r_,i r_,j] r³
```
"""
function _drm_u(rvec::SVector{2}, nu::Real, G::Real, ::Val{:r})
    r = norm(rvec)
    r < 1e-14 && return zeros(2, 2)
    rm = rvec / r
    r3 = r * r * r
    c = 1 / (30 * (1 - nu) * G)
    U = zeros(2, 2)
    @inbounds for i in 1:2, j in 1:2
        δ = (i == j ? 1.0 : 0.0)
        U[i, j] = c * ((3 - 10nu / 3) * δ - rm[i] * rm[j]) * r3
    end
    return U
end

"""
Particular traction for ``f = r`` (``∂r/∂n = n·r_``):

```
p̂_ij = [1/(15(1-ν))] [(4-5ν) r_,i n_j − (1-5ν) r_,j n_i
        + ((4-5ν) δ_ij − r_,j r_,i) ∂r/∂n] r²
```
"""
function _drm_t(rvec::SVector{2}, nrm::SVector{2}, nu::Real, G::Real, ::Val{:r})
    r = norm(rvec)
    r < 1e-14 && return zeros(2, 2)
    rm = rvec / r
    drdn = dot(rm, nrm)
    c = 1 / (15 * (1 - nu))
    r2 = r * r
    T = zeros(2, 2)
    @inbounds for i in 1:2, j in 1:2
        δ = (i == j ? 1.0 : 0.0)
        T[i, j] = c * (
            (4 - 5nu) * rm[i] * nrm[j] -
            (1 - 5nu) * rm[j] * nrm[i] +
            ((4 - 5nu) * δ - rm[j] * rm[i]) * drdn
        ) * r2
    end
    return T
end

"""
Particular displacement for ``f = 1+r`` (body force in direction ``k``):

```
û_mk = [(1-2ν)/((5-4ν)G)] r_,m r_,k r²
     + [1/(30(1-ν)G)] [(3-10ν/3) δ_mk − r_,m r_,k] r³
```
"""
function _drm_u(rvec::SVector{2}, nu::Real, G::Real, ::Val{:one_plus_r})
    r = norm(rvec)
    r < 1e-14 && return zeros(2, 2)
    rm = rvec / r
    r2 = r * r
    r3 = r2 * r
    coef1 = (1 - 2nu) / ((5 - 4nu) * G)
    coef2 = 1 / (30 * (1 - nu) * G)
    U = zeros(2, 2)
    @inbounds for m in 1:2, k in 1:2
        rmk = rm[m] * rm[k]
        δ = (m == k ? 1.0 : 0.0)
        U[m, k] = coef1 * rmk * r2 + coef2 * ((3 - 10nu / 3) * δ - rmk) * r3
    end
    return U
end

"""
Particular traction for ``f = 1+r``. Independent of ``G``.
The ``(1-2ν)`` in the first block is cancelled so ν→1/2 is finite.
"""
function _drm_t(rvec::SVector{2}, nrm::SVector{2}, nu::Real, G::Real, ::Val{:one_plus_r})
    r = norm(rvec)
    r < 1e-14 && return zeros(2, 2)
    rm = rvec / r
    drdn = dot(rm, nrm)
    den = 5 - 4nu
    coef2 = 1 / (15 * (1 - nu))
    T = zeros(2, 2)
    @inbounds for m in 1:2, k in 1:2
        rmk = rm[m] * rm[k]
        δ = (m == k ? 1.0 : 0.0)
        term1 = (2 * (1 + nu) / den * rm[m] * nrm[k] +
                 (1 - 2nu) / den * (rm[k] * nrm[m] + δ * drdn)) * r
        inside2 = (4 - 5nu) * rm[k] * nrm[m] - (1 - 5nu) * rm[m] * nrm[k] +
                  ((4 - 5nu) * δ - rmk) * drdn
        T[m, k] = term1 + coef2 * inside2 * r * r
    end
    return T
end

# ---------------------------------------------------------------------------
# MQ  f = √(r²+C²)  — Galerkin particular of ∇⁴g = f (2D)
#
# p = ∇²g,  ∇²p = f:
#   p = s³/9 + (C²/3)s − (C³/3) ln(C+s),   s = √(r²+C²)
# g from ∇²g = p (radial):  A = g'/r,  B = g''−g'/r = p−2A
#   û_ij = (1/μ)[ p δ_ij − α g,ij ],  α = 1/(2(1−ν))
# Regular at r = 0 (no Kelvin smoothing). Traction from Hooke on û.
# ---------------------------------------------------------------------------

"""I = ∫_0^r t p(t) dt for the MQ Poisson potential p above."""
function _mq_I(r::Float64, C::Float64)
    s = hypot(r, C)
    C2 = C * C
    C3 = C2 * C
    C5 = C2 * C3
    lns = log(C + s)
    return s^5 / 45 + C2 * s^3 / 9 + C3 * s^2 / 12 - C2 * C2 * s / 6 -
           (C3 * s^2 / 6) * lns + (C5 / 6) * lns - C5 / 20
end

"""Radial scalars (p, A, B, p′, A′, B′) of the MQ Galerkin potential."""
function _mq_galerkin_scalars(r::Float64, C::Float64)
    C > 0 || throw(ArgumentError("MQ shape C must be > 0"))
    if r <= 1e-14
        ln2C = log(2C)
        p0 = C^3 * (4 / 9 - ln2C / 3)
        return (p = p0, A = p0 / 2, B = 0.0, pp = 0.0, Ap = 0.0, Bp = 0.0)
    end
    s = hypot(r, C)
    C3 = C * C * C
    p = s^3 / 9 + (C * C / 3) * s - (C3 / 3) * log(C + s)
    A = _mq_I(r, C) / (r * r)
    B = p - 2A
    pp = (s^3 - C3) / (3r)
    Ap = B / r
    Bp = pp - 2B / r
    return (p = p, A = A, B = B, pp = pp, Ap = Ap, Bp = Bp)
end

"""
Particular displacement for ``f = √(r²+C²)`` (Samaan & Rashed 2007).
"""
function _drm_u(rvec::SVector{2}, nu::Real, G::Real, ::Val{:mq}, C::Real)
    r = norm(rvec)
    sc = _mq_galerkin_scalars(float(r), float(C))
    α = 1 / (2 * (1 - nu))
    β = (sc.p - α * sc.A) / G
    γ = -α * sc.B / G
    r < 1e-14 && return [β 0.0; 0.0 β]
    rm = rvec / r
    U = zeros(2, 2)
    @inbounds for i in 1:2, j in 1:2
        δ = (i == j ? 1.0 : 0.0)
        U[i, j] = β * δ + γ * rm[i] * rm[j]
    end
    return U
end

"""
Particular traction for ``f = √(r²+C²)``. Independent of a global G-scale
on û after Hooke (`σ = λ (div) I + μ (∇u+∇uᵀ)`). Zero at ``r = 0``.
"""
function _drm_t(rvec::SVector{2}, nrm::SVector{2}, nu::Real, G::Real, ::Val{:mq}, C::Real)
    r = norm(rvec)
    r < 1e-14 && return zeros(2, 2)
    sc = _mq_galerkin_scalars(float(r), float(C))
    α = 1 / (2 * (1 - nu))
    μ = G
    βp = (sc.pp - α * sc.Ap) / μ
    γ = -α * sc.B / μ
    γp = -α * sc.Bp / μ
    rm = rvec / r
    drdn = dot(rm, nrm)
    δdiv = βp + γp + γ / r
    λ = 2μ * nu / (1 - 2nu)
    T = zeros(2, 2)
    @inbounds for i in 1:2, j in 1:2
        δij = (i == j ? 1.0 : 0.0)
        ni, nj = nrm[i], nrm[j]
        ri, rj = rm[i], rm[j]
        T[i, j] = λ * δdiv * ni * rj +
                  μ * βp * drdn * δij + μ * βp * nj * ri +
                  2μ * γp * drdn * ri * rj +
                  μ * (γ / r) * (2 * ni * rj + nj * ri + drdn * δij -
                                 4 * drdn * ri * rj)
    end
    return T
end
