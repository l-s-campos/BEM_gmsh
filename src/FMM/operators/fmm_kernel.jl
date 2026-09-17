# =============================================================================
# FMM-backed kernel matrix: fast matvec + entry access (Martinsson 2011)
# =============================================================================

"""
    struct FMMKernelMatrix{T} <: AbstractMatrix{T}

Dense-looking kernel matrix `A[i,j] = K(x_i, x_j)` with:
- `getindex` / block extraction via direct kernel evaluation
- `mul!` via FMM (O(N) or O(N log N))

Used as an `AbstractMatrix` with FMM `mul!` (GMRES, sampling, …).
"""
struct FMMKernelMatrix{T} <: AbstractKernelMatrix{T}
    points::Matrix{Float64}          # d × N (local tree order)
    entry::Function                  # (i,j) -> T  (1-based local indices)
    matvec!::Function                # (y, x) -> y  with y = A*x (local order)
    matvec_adj!::Function            # (y, x) -> y  with y = A'*x
    n::Int
end

Base.size(A::FMMKernelMatrix) = (A.n, A.n)
Base.IndexStyle(::Type{<:FMMKernelMatrix}) = IndexCartesian()

function Base.getindex(A::FMMKernelMatrix{T}, i::Int, j::Int) where {T}
    return A.entry(i, j)::T
end

function LinearAlgebra.mul!(y::AbstractVector, A::FMMKernelMatrix, x::AbstractVector,
                            a::Number=1, b::Number=0)
    length(x) == A.n && length(y) == A.n || throw(DimensionMismatch())
    if iszero(b)
        A.matvec!(y, x)
        a != 1 && rmul!(y, a)
    else
        tmp = similar(y)
        A.matvec!(tmp, x)
        y .= b .* y .+ a .* tmp
    end
    return y
end

"""Multi-RHS FMM apply (column loop — FMM kernels are single-RHS)."""
function LinearAlgebra.mul!(Y::AbstractMatrix, A::FMMKernelMatrix, X::AbstractMatrix,
                            a::Number=1, b::Number=0)
    size(X, 1) == A.n && size(Y, 1) == A.n || throw(DimensionMismatch())
    size(Y, 2) == size(X, 2) || throw(DimensionMismatch())
    @inbounds for k in 1:size(X, 2)
        mul!(view(Y, :, k), A, view(X, :, k), a, b)
    end
    return Y
end

"""Sample `A * Ω` using FMM matvecs (Martinsson §2.3)."""
function sample_matvec(A::FMMKernelMatrix{T}, Ω::AbstractMatrix{T}) where {T}
    S = Matrix{T}(undef, size(Ω, 1), size(Ω, 2))
    mul!(S, A, Ω)
    return S
end

"""Sample `A' * Ω` using adjoint FMM matvecs."""
function sample_matvec_adj(A::FMMKernelMatrix{T}, Ω::AbstractMatrix{T}) where {T}
    S = Matrix{T}(undef, size(Ω, 1), size(Ω, 2))
    mul!(S, adjoint(A), Ω)
    return S
end

# Adjoint FMM matvec (used by GMRES / sampling).
function LinearAlgebra.mul!(y::AbstractVector, At::Adjoint{<:Any,<:FMMKernelMatrix}, x::AbstractVector)
    return parent(At).matvec_adj!(y, x)
end

function LinearAlgebra.mul!(Y::AbstractMatrix, At::Adjoint{<:Any,<:FMMKernelMatrix}, X::AbstractMatrix)
    size(X, 2) == size(Y, 2) || throw(DimensionMismatch())
    A = parent(At)
    @inbounds for k in 1:size(X, 2)
        A.matvec_adj!(view(Y, :, k), view(X, :, k))
    end
    return Y
end

# ---------- constructors from FMM2D kernels ----------

"""
    fmm_laplace3d_matrix(points; eps=1e-8, nmax=-1, exclude_self=true, p=nothing)

Symmetric Laplace kernel matrix `1/(4π|x_i-x_j|)` with FMM matvec.
`points` is `3 × N`. Cubic octree + spherical-harmonic M2L (FMM3D lists).
`nmax < 0` selects `laplace3d_ndiv(eps)` (`200` at `1e-8`, Flatiron `lndiv`). `p=nothing` uses
`laplace3d_nterms(eps)` (capped at 16); pass `p=` to lower the order.
"""
function fmm_laplace3d_matrix(
    points::AbstractMatrix{<:Real};
    eps::Float64=1e-8,
    nmax::Int=-1,
    η::Float64=1.0,
    exclude_self::Bool=true,
    tree=nothing,
    full_fmm::Bool=true,
    p::Union{Nothing,Int}=nothing,
)
    d, n = size(points)
    d == 3 || throw(ArgumentError("expected 3 × N points"))
    pts = Matrix{Float64}(points)
    plan3 = build_laplace3d_plan(pts; eps=eps, nmax=nmax, η=η, tree=tree,
        full_fmm=full_fmm, p=p)

    function entry(i, j)
        i == j && exclude_self && return 0.0
        r = hypot(pts[1, i] - pts[1, j], pts[2, i] - pts[2, j], pts[3, i] - pts[3, j])
        r < 1e-30 && return 0.0
        return INV4PI / r
    end

    function matvec!(y, x)
        apply_laplace3d!(plan3, y; charges=x)
        return y
    end
    # symmetric kernel
    return FMMKernelMatrix{Float64}(pts, entry, matvec!, matvec!, n)
end

"""
    fmm_laplace2d_matrix(points; eps=1e-8, nmax=-1)

Symmetric 2D Laplace `log|x_i-x_j|` (real) with FMM matvec.
`nmax < 0` selects Flatiron `lndiv2d` leaf size from `eps`.
"""
function fmm_laplace2d_matrix(
    points::AbstractMatrix{<:Real};
    eps::Float64=1e-8,
    nmax::Int=-1,
    η::Float64=1.0,
    tree=nothing,
)
    d, n = size(points)
    d == 2 || throw(ArgumentError("expected 2 × N points"))
    pts = Matrix{Float64}(points)
    # Cache tree + expansions — matvec used many times (GMRES).
    plan = build_laplace2d_plan(pts; eps=eps, nmax=nmax, η=η, pg=1, tree=tree)
    pot_buf = Vector{Float64}(undef, n)

    function entry(i, j)
        i == j && return 0.0
        r = hypot(pts[1, i] - pts[1, j], pts[2, i] - pts[2, j])
        r < 1e-30 && return 0.0
        return log(r)
    end
    function matvec!(y, x)
        apply_laplace2d!(plan, pot_buf; charges=x)
        copyto!(y, pot_buf)
        return y
    end
    return FMMKernelMatrix{Float64}(pts, entry, matvec!, matvec!, n)
end

"""
    fmm_laplace2d_double_layer_matrix(points, normals; n_boundary, eps=1e-8, …)

Bare 2D Laplace double-layer matrix matching BEM `fundamental.T`:

```
Dq_ij = (r · n_j) / (2π R²),   r = x_j − x_i   (source j − field i)
```
with `Dq_ii = 0` and `Dq_i,j = 0` for `j > n_boundary` (internal columns).

Matvec uses dipole FMM. The FMM dipole potential is calibrated so that
scaling by `+1/(2π)` matches `fundamental.T` (verified against `LaplaceDqKernel`).

`points` is `2 × N` (all collocation), `normals` is `2 × N` (zeros past boundary).
"""
function fmm_laplace2d_double_layer_matrix(
    points::AbstractMatrix{<:Real},
    normals::AbstractMatrix{<:Real};
    n_boundary::Int=size(points, 2),
    eps::Float64=1e-8,
    nmax::Int=50,
    η::Float64=1.0,
    tree=nothing,
)
    d, n = size(points)
    d == 2 || throw(ArgumentError("expected 2 × N points"))
    size(normals) == (2, n) || throw(DimensionMismatch("normals must be 2 × N"))
    0 < n_boundary <= n || throw(ArgumentError("n_boundary out of range"))
    pts = Matrix{Float64}(points)
    nrm = Matrix{Float64}(normals)
    # zero internal normals (and any garbage)
    @inbounds for j in (n_boundary + 1):n
        nrm[1, j] = 0.0
        nrm[2, j] = 0.0
    end
    α = 1 / (2π)    # FMM dipole pot → BEM T  (sign checked vs LaplaceDqKernel)
    plan_dl = build_laplace2d_plan(pts; eps=eps, nmax=nmax, η=η, pg=1, dipvec=nrm, tree=tree)
    plan_adj = build_laplace2d_plan(pts; eps=eps, nmax=nmax, η=η, pg=2, tree=tree)
    dens_buf = Vector{Float64}(undef, n)
    pot_buf = Vector{Float64}(undef, n)
    grad_buf = zeros(Float64, 2, n)

    function entry(i, j)
        (i == j || j > n_boundary) && return 0.0
        rx = pts[1, j] - pts[1, i]
        ry = pts[2, j] - pts[2, i]
        R2 = rx * rx + ry * ry
        R2 < 1e-30 && return 0.0
        return (rx * nrm[1, j] + ry * nrm[2, j]) / (2π * R2)
    end

    function matvec!(y, x)
        # zero internal densities so only boundary columns contribute
        @inbounds for j in 1:n
            dens_buf[j] = j <= n_boundary ? Float64(x[j]) : 0.0
        end
        apply_laplace2d!(plan_dl, pot_buf; dipstr=dens_buf)
        @inbounds for i in 1:n
            y[i] = α * pot_buf[i]
        end
        return y
    end

    # adjoint: Dq is not symmetric — (Dq')_ij = Dq_ji
    # Dq' sampling: entry-wise i↔j is wrong for DL.
    # Dq_ij = ( (x_j-x_i)·n_j )/(2π R²)
    # (Dq')_ij = Dq_ji = ( (x_i-x_j)·n_i )/(2π R²) = −( (x_j-x_i)·n_i )/(2π R²)
    # Matvec Dq' x: like dipole FMM with normals at *targets* — not standard.
    # Use entry-based transpose matvec for moderate n; for large n use FMM on swapped roles.
    function matvec_adj!(y, x)
        # y_i = Σ_j Dq_ji x_j = (n_i · ∇φ)/2π with φ = Σ x_j log R
        @inbounds for j in 1:n
            dens_buf[j] = j <= n_boundary ? Float64(x[j]) : 0.0
        end
        apply_laplace2d!(plan_adj, pot_buf; charges=dens_buf, grad=grad_buf)
        @inbounds for i in 1:n_boundary
            y[i] = (nrm[1, i] * grad_buf[1, i] + nrm[2, i] * grad_buf[2, i]) / (2π)
        end
        @inbounds for i in (n_boundary + 1):n
            y[i] = 0.0
        end
        return y
    end

    return FMMKernelMatrix{Float64}(pts, entry, matvec!, matvec_adj!, n)
end

"""
    fmm_yukawa3d_matrix(points, κ; eps=1e-8, nmax=40)

Yukawa `e^{-κr}/(4πr)` with FMM matvec.
"""
function fmm_yukawa3d_matrix(
    points::AbstractMatrix{<:Real},
    κ::Float64;
    eps::Float64=1e-8,
    nmax::Int=40,
    η::Float64=1.2,
)
    d, n = size(points)
    d == 3 || throw(ArgumentError("expected 3 × N points"))
    pts = Matrix{Float64}(points)
    function entry(i, j)
        i == j && return 0.0
        r = hypot(pts[1, i] - pts[1, j], pts[2, i] - pts[2, j], pts[3, i] - pts[3, j])
        r < 1e-30 && return 0.0
        return exp(-κ * r) * INV4PI / r
    end
    function matvec!(y, x)
        vals = yfmm3d(eps, κ, pts; charges=collect(x), pg=1, nmax=nmax, η=η)
        copyto!(y, vals.pot)
        return y
    end
    return FMMKernelMatrix{Float64}(pts, entry, matvec!, matvec!, n)
end

# =============================================================================
# 2D Kelvin (plane strain / stress) — three Laplace FMMs
# =============================================================================
#
# Bare Stokeslet (Flatiron, no 1/2π):  G = -½ log(R) I + ½ e⊗e
# Kelvin (BEM Fundamental.jl):         U = A [-(3-4ν) log(R) I + e⊗e]
#                                    A = 1/(8π μ (1-ν))
#
# Identity:  U = 2A G + 2A(2ν-1) (log R) I
# Equivalently, with φx,φy,φm the log potentials of fx, fy, y·f:
#   u = A (4ν-3)(φx,φy) + A (x₁∇φx + x₂∇φy − ∇φm)
# =============================================================================

"""
    KelvinFMMMatrix

Matrix-free 2D Kelvin single-layer operator on collocation poles.
Size `(2n)×(2n)`, node-major DOF order `[uₓ₁,uᵧ₁,uₓ₂,uᵧ₂,…]`.
Diagonal is 0 (self-interaction skipped). Matvec via three real Laplace FMMs.
"""
struct KelvinFMMMatrix <: AbstractMatrix{Float64}
    points::Matrix{Float64}   # 2 × n
    n::Int
    A::Float64                # 1/(8π μ (1-ν))
    ν::Float64                # effective Poisson ratio
    eps::Float64
    nmax::Int
    η::Float64
    plan::Laplace2DFMMPlan    # cached tree/expansions (pg=2)
    fx::Vector{Float64}
    fy::Vector{Float64}
    mchg::Vector{Float64}
    potx::Vector{Float64}
    poty::Vector{Float64}
    potm::Vector{Float64}
    gradx::Matrix{Float64}
    grady::Matrix{Float64}
    gradm::Matrix{Float64}
end

Base.size(K::KelvinFMMMatrix) = (2K.n, 2K.n)
Base.size(K::KelvinFMMMatrix, d) = size(K)[d]
Base.IndexStyle(::Type{KelvinFMMMatrix}) = IndexCartesian()

function Base.getindex(K::KelvinFMMMatrix, α::Int, β::Int)
    i = (α + 1) ÷ 2
    j = (β + 1) ÷ 2
    a = α - 2(i - 1)
    b = β - 2(j - 1)
    i == j && return 0.0
    rx = K.points[1, i] - K.points[1, j]
    ry = K.points[2, i] - K.points[2, j]
    R2 = rx * rx + ry * ry
    R2 < 1e-30 && return 0.0
    R = sqrt(R2)
    # U = A [ (3-4ν) log(1/R) I + e⊗e ]
    e1 = rx / R
    e2 = ry / R
    c_iso = (3 - 4K.ν) * log(1 / R)
    if a == 1 && b == 1
        return K.A * (c_iso + e1 * e1)
    elseif a == 1 && b == 2
        return K.A * (e1 * e2)
    elseif a == 2 && b == 1
        return K.A * (e2 * e1)
    else
        return K.A * (c_iso + e2 * e2)
    end
end

function LinearAlgebra.mul!(y::AbstractVector, K::KelvinFMMMatrix, x::AbstractVector)
    n = K.n
    length(x) == 2n && length(y) == 2n || throw(DimensionMismatch())
    fx, fy, mchg = K.fx, K.fy, K.mchg
    pts = K.points
    @inbounds for j in 1:n
        fx[j] = x[2j-1]
        fy[j] = x[2j]
        mchg[j] = pts[1, j] * fx[j] + pts[2, j] * fy[j]
    end

    # Three Laplace FMMs on the same cached plan (tree built once).
    apply_laplace2d!(K.plan, K.potx; charges=fx, grad=K.gradx)
    apply_laplace2d!(K.plan, K.poty; charges=fy, grad=K.grady)
    apply_laplace2d!(K.plan, K.potm; charges=mchg, grad=K.gradm)

    A = K.A
    cφ = A * (4K.ν - 3)   # coefficient of (φx, φy)
    @inbounds for j in 1:n
        x1 = pts[1, j]
        x2 = pts[2, j]
        φx, φy = K.potx[j], K.poty[j]
        gxx, gxy = K.gradx[1, j], K.gradx[2, j]
        gyx, gyy = K.grady[1, j], K.grady[2, j]
        gmx, gmy = K.gradm[1, j], K.gradm[2, j]
        # u = cφ (φx,φy) + A (x1 ∇φx + x2 ∇φy − ∇φm)
        y[2j-1] = cφ * φx + A * (x1 * gxx + x2 * gyx - gmx)
        y[2j]   = cφ * φy + A * (x1 * gxy + x2 * gyy - gmy)
    end
    return y
end

function LinearAlgebra.mul!(y::AbstractVector, K::KelvinFMMMatrix, x::AbstractVector,
        α::Number, β::Number)
    if iszero(β)
        mul!(y, K, x)
        α != 1 && rmul!(y, α)
    else
        t = similar(y)
        mul!(t, K, x)
        y .= β .* y .+ α .* t
    end
    return y
end

Base.:*(K::KelvinFMMMatrix, x::AbstractVector) = mul!(similar(x, Float64), K, x)

# Kelvin single-layer is symmetric → adjoint matvec = forward
function LinearAlgebra.mul!(y::AbstractVector, Kt::Adjoint{<:Any,<:KelvinFMMMatrix}, x::AbstractVector)
    return mul!(y, parent(Kt), x)
end

function LinearAlgebra.mul!(Y::AbstractMatrix, K::KelvinFMMMatrix, X::AbstractMatrix)
    size(Y, 2) == size(X, 2) || throw(DimensionMismatch())
    @inbounds for p in 1:size(X, 2)
        mul!(view(Y, :, p), K, view(X, :, p))
    end
    return Y
end

function LinearAlgebra.mul!(Y::AbstractMatrix, Kt::Adjoint{<:Any,<:KelvinFMMMatrix}, X::AbstractMatrix)
    return mul!(Y, parent(Kt), X)
end

"""
    fmm_kelvin2d_matrix(points; μ, ν, eps=1e-8, nmax=50, η=1.0)

2D Kelvin single-layer matrix on `points` (`2 × N`) with FMM matvec.
`ν` must be the **effective** Poisson ratio (plane strain / mapped plane stress).
`μ` is the shear modulus. Diagonal is zero.
"""
function fmm_kelvin2d_matrix(
    points::AbstractMatrix{<:Real};
    μ::Real,
    ν::Real,
    eps::Float64=1e-8,
    nmax::Int=50,
    η::Float64=1.0,
    tree=nothing,
)
    d, n = size(points)
    d == 2 || throw(ArgumentError("expected 2 × N points for Kelvin 2D"))
    μ = float(μ)
    ν = float(ν)
    A = 1 / (8π * μ * (1 - ν))
    pts = Matrix{Float64}(points)
    plan = build_laplace2d_plan(pts; eps=eps, nmax=nmax, η=η, pg=2, tree=tree)
    return KelvinFMMMatrix(
        pts, n, A, ν, eps, nmax, η, plan,
        zeros(n), zeros(n), zeros(n),
        zeros(n), zeros(n), zeros(n),
        zeros(2, n), zeros(2, n), zeros(2, n),
    )
end

# =============================================================================
# 3D Kelvin — four Laplace-3D FMMs (Papkovich–Neuber)
# =============================================================================
#
# U = B/R [(3-4ν) I + e⊗e],  B = 1/(16π μ (1-ν))
# φ_k = ∫ f_k /(4π R) dy,  ψ = ∫ (y·f)/(4π R) dy   (lfmm3d)
# u_i = C [(3-4ν) φ_i − x_k ∂_i φ_k + ∂_i ψ],  C = 1/(4 μ (1-ν)) = 4π B
# =============================================================================

"""
    KelvinFMMMatrix3D

Matrix-free 3D Kelvin single-layer on collocation poles.
Size `(3n)×(3n)`, node-major `[uₓ,uᵧ,u_z,… ]`. Diagonal 0.
Matvec via four real Laplace-3D FMMs (`pg=2`) on the octree plan.
"""
struct KelvinFMMMatrix3D <: AbstractMatrix{Float64}
    points::Matrix{Float64}   # 3 × n
    n::Int
    C::Float64                # 1/(4 μ (1-ν))
    ν::Float64
    eps::Float64
    nmax::Int
    η::Float64
    plan::Laplace3DFMMPlan
    fx::Vector{Float64}
    fy::Vector{Float64}
    fz::Vector{Float64}
    mchg::Vector{Float64}
    potx::Vector{Float64}
    poty::Vector{Float64}
    potz::Vector{Float64}
    potm::Vector{Float64}
    gradx::Matrix{Float64}
    grady::Matrix{Float64}
    gradz::Matrix{Float64}
    gradm::Matrix{Float64}
end

Base.size(K::KelvinFMMMatrix3D) = (3K.n, 3K.n)
Base.size(K::KelvinFMMMatrix3D, d) = size(K)[d]
Base.IndexStyle(::Type{KelvinFMMMatrix3D}) = IndexCartesian()

function Base.getindex(K::KelvinFMMMatrix3D, α::Int, β::Int)
    i = (α - 1) ÷ 3 + 1
    j = (β - 1) ÷ 3 + 1
    a = α - 3 * (i - 1)
    b = β - 3 * (j - 1)
    i == j && return 0.0
    rx = K.points[1, j] - K.points[1, i]
    ry = K.points[2, j] - K.points[2, i]
    rz = K.points[3, j] - K.points[3, i]
    R2 = rx * rx + ry * ry + rz * rz
    R2 < 1e-30 && return 0.0
    R = sqrt(R2)
    B = K.C / (4π * R)          # 1/(16π μ (1-ν) R)
    e = (rx / R, ry / R, rz / R)
    iso = (3 - 4K.ν) * (a == b ? 1.0 : 0.0)
    return B * (iso + e[a] * e[b])
end

function _lfmm3d_charge_grad!(K::KelvinFMMMatrix3D, charges, pot, grad)
    apply_laplace3d!(K.plan, pot; charges=charges, grad=grad)
    return nothing
end

function LinearAlgebra.mul!(y::AbstractVector, K::KelvinFMMMatrix3D, x::AbstractVector)
    n = K.n
    length(x) == 3n && length(y) == 3n || throw(DimensionMismatch())
    fx, fy, fz, mchg = K.fx, K.fy, K.fz, K.mchg
    pts = K.points
    @inbounds for j in 1:n
        fx[j] = x[3j - 2]
        fy[j] = x[3j - 1]
        fz[j] = x[3j]
        mchg[j] = pts[1, j] * fx[j] + pts[2, j] * fy[j] + pts[3, j] * fz[j]
    end
    _lfmm3d_charge_grad!(K, fx, K.potx, K.gradx)
    _lfmm3d_charge_grad!(K, fy, K.poty, K.grady)
    _lfmm3d_charge_grad!(K, fz, K.potz, K.gradz)
    _lfmm3d_charge_grad!(K, mchg, K.potm, K.gradm)

    C = K.C
    cφ = C * (3 - 4K.ν)
    @inbounds for j in 1:n
        x1, x2, x3 = pts[1, j], pts[2, j], pts[3, j]
        # u_i = C[(3-4ν) φ_i − x_k ∂_i φ_k + ∂_i ψ]
        gx1, gx2, gx3 = K.gradx[1, j], K.gradx[2, j], K.gradx[3, j]
        gy1, gy2, gy3 = K.grady[1, j], K.grady[2, j], K.grady[3, j]
        gz1, gz2, gz3 = K.gradz[1, j], K.gradz[2, j], K.gradz[3, j]
        gm1, gm2, gm3 = K.gradm[1, j], K.gradm[2, j], K.gradm[3, j]
        y[3j - 2] = cφ * K.potx[j] + C * (gm1 - (x1 * gx1 + x2 * gy1 + x3 * gz1))
        y[3j - 1] = cφ * K.poty[j] + C * (gm2 - (x1 * gx2 + x2 * gy2 + x3 * gz2))
        y[3j]     = cφ * K.potz[j] + C * (gm3 - (x1 * gx3 + x2 * gy3 + x3 * gz3))
    end
    return y
end

function LinearAlgebra.mul!(y::AbstractVector, K::KelvinFMMMatrix3D, x::AbstractVector,
        α::Number, β::Number)
    if iszero(β)
        mul!(y, K, x)
        α != 1 && rmul!(y, α)
    else
        t = similar(y)
        mul!(t, K, x)
        y .= β .* y .+ α .* t
    end
    return y
end

Base.:*(K::KelvinFMMMatrix3D, x::AbstractVector) = mul!(similar(x, Float64), K, x)

function LinearAlgebra.mul!(y::AbstractVector, Kt::Adjoint{<:Any,<:KelvinFMMMatrix3D}, x::AbstractVector)
    return mul!(y, parent(Kt), x)
end

function LinearAlgebra.mul!(Y::AbstractMatrix, K::KelvinFMMMatrix3D, X::AbstractMatrix)
    size(Y, 2) == size(X, 2) || throw(DimensionMismatch())
    @inbounds for p in 1:size(X, 2)
        mul!(view(Y, :, p), K, view(X, :, p))
    end
    return Y
end

function LinearAlgebra.mul!(Y::AbstractMatrix, Kt::Adjoint{<:Any,<:KelvinFMMMatrix3D}, X::AbstractMatrix)
    return mul!(Y, parent(Kt), X)
end

"""
    fmm_kelvin3d_matrix(points; μ, ν, eps=1e-8, nmax=-1, η=1.0, p=nothing)

3D Kelvin single-layer on `points` (`3 × N`) with FMM matvec.
`ν` is the true Poisson ratio. Diagonal is zero.
Each of the four Laplace applies evaluates potential and gradient from
local expansions on the cubic octree. `p=` is forwarded to
`build_laplace3d_plan`.
"""
function fmm_kelvin3d_matrix(
    points::AbstractMatrix{<:Real};
    μ::Real,
    ν::Real,
    eps::Float64=1e-8,
    nmax::Int=-1,
    η::Float64=1.0,
    tree=nothing,
    full_fmm::Bool=true,
    p::Union{Nothing,Int}=nothing,
)
    d, n = size(points)
    d == 3 || throw(ArgumentError("expected 3 × N points for Kelvin 3D"))
    μ = float(μ)
    ν = float(ν)
    C = 1 / (4 * μ * (1 - ν))
    pts = Matrix{Float64}(points)
    plan = build_laplace3d_plan(pts; eps=eps, nmax=nmax, η=η, tree=tree,
        full_fmm=full_fmm, p=p)
    return KelvinFMMMatrix3D(
        pts, n, C, ν, eps, nmax, η, plan,
        zeros(n), zeros(n), zeros(n), zeros(n),
        zeros(n), zeros(n), zeros(n), zeros(n),
        zeros(3, n), zeros(3, n), zeros(3, n), zeros(3, n),
    )
end
