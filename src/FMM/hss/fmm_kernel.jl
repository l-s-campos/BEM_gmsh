# =============================================================================
# FMM-backed kernel matrix: fast matvec + entry access (Martinsson 2011)
# =============================================================================

"""
    struct FMMKernelMatrix{T} <: AbstractMatrix{T}

Dense-looking kernel matrix `A[i,j] = K(x_i, x_j)` with:
- `getindex` / block extraction via direct kernel evaluation
- `mul!` via FMM (O(N) or O(N log N))

Built for Martinsson's randomized HSS compression (SIAM J. Matrix Anal. Appl. 2011).
"""
struct FMMKernelMatrix{T} <: AbstractMatrix{T}
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

# Adjoint matvec for HMatrices.assemble_hss / hara_h2 (...; method=:fmm)
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
    fmm_laplace3d_matrix(points; eps=1e-8, nmax=40, exclude_self=true)

Symmetric Laplace kernel matrix `1/(4π|x_i-x_j|)` with FMM matvec.
`points` is `3 × N`.
"""
function fmm_laplace3d_matrix(
    points::AbstractMatrix{<:Real};
    eps::Float64=1e-8,
    nmax::Int=40,
    η::Float64=1.0,
    exclude_self::Bool=true,
)
    d, n = size(points)
    d == 3 || throw(ArgumentError("expected 3 × N points"))
    pts = Matrix{Float64}(points)
    thresh = exclude_self ? 1e-14 : 0.0

    function entry(i, j)
        i == j && exclude_self && return 0.0
        r = hypot(pts[1, i] - pts[1, j], pts[2, i] - pts[2, j], pts[3, i] - pts[3, j])
        r < 1e-30 && return 0.0
        return INV4PI / r
    end

    function matvec!(y, x)
        # FMM of charges x at sources=targets=pts, then zero diagonal if needed
        vals = lfmm3d(eps, pts; charges=collect(x), pg=1, nmax=nmax, η=η)
        copyto!(y, vals.pot)
        if exclude_self
            # diagonal of Laplace is singular; FMM already skips self
            nothing
        end
        return y
    end
    # symmetric kernel
    return FMMKernelMatrix{Float64}(pts, entry, matvec!, matvec!, n)
end

"""
    fmm_laplace2d_matrix(points; eps=1e-8, nmax=50)

Symmetric 2D Laplace `log|x_i-x_j|` (real) with FMM matvec.
"""
function fmm_laplace2d_matrix(
    points::AbstractMatrix{<:Real};
    eps::Float64=1e-8,
    nmax::Int=50,
    η::Float64=1.0,
)
    d, n = size(points)
    d == 2 || throw(ArgumentError("expected 2 × N points"))
    pts = Matrix{Float64}(points)

    function entry(i, j)
        i == j && return 0.0
        r = hypot(pts[1, i] - pts[1, j], pts[2, i] - pts[2, j])
        r < 1e-30 && return 0.0
        return log(r)
    end
    function matvec!(y, x)
        vals = rfmm2d(eps, pts; charges=collect(x), pg=1, nmax=nmax, η=η)
        copyto!(y, vals.pot)
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
        xc = collect(Float64, x)
        @inbounds for j in (n_boundary + 1):n
            xc[j] = 0.0
        end
        vals = rfmm2d(eps, pts; dipstr=xc, dipvec=nrm, pg=1, nmax=nmax, η=η)
        @inbounds for i in 1:n
            y[i] = α * vals.pot[i]
        end
        # force exact diagonal 0 (FMM already skips self)
        return y
    end

    # adjoint: Dq is not symmetric — (Dq')_ij = Dq_ji
    # For HSS sampling we need Dq'. Entry-wise: same formula with i↔j is wrong for DL.
    # Dq_ij = ( (x_j-x_i)·n_j )/(2π R²)
    # (Dq')_ij = Dq_ji = ( (x_i-x_j)·n_i )/(2π R²) = −( (x_j-x_i)·n_i )/(2π R²)
    # Matvec Dq' x: like dipole FMM with normals at *targets* — not standard.
    # Use entry-based transpose matvec for moderate n; for large n use FMM on swapped roles.
    function matvec_adj!(y, x)
        # y_i = Σ_j Dq_ji x_j = Σ_j ( (x_i-x_j)·n_i )/(2π R²) x_j
        #      = n_i · Σ_j (x_i-x_j) x_j / (2π R²)
        # This is n_i · ∇-type field from charges x at sources — charge FMM grad.
        xc = collect(Float64, x)
        @inbounds for j in (n_boundary + 1):n
            xc[j] = 0.0
        end
        # pot/grad of charges: φ = Σ x_j log|r|, but we need Σ x_j (x_i-x_j)/R² = −∇φ_charge
        # if φ = Σ x_j log R with R=|x_i-x_j|, ∇_i φ = Σ x_j (x_i-x_j)/R²
        vals = rfmm2d(eps, pts; charges=xc, pg=2, nmax=nmax, η=η)
        @inbounds for i in 1:n_boundary
            # (Dq' x)_i = ( n_i · ∇φ ) / (2π)
            y[i] = (nrm[1, i] * vals.grad[1, i] + nrm[2, i] * vals.grad[2, i]) / (2π)
        end
        @inbounds for i in (n_boundary + 1):n
            y[i] = 0.0   # no normal → Dq_ji = 0 for all j when i internal? 
            # Dq_ji zero when i > n_boundary because normals[i]=0 in formula with n_i
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
    fx = Vector{Float64}(undef, n)
    fy = Vector{Float64}(undef, n)
    @inbounds for j in 1:n
        fx[j] = x[2j-1]
        fy[j] = x[2j]
    end
    pts = K.points
    mchg = pts[1, :] .* fx .+ pts[2, :] .* fy

    vx = rfmm2d(K.eps, pts; charges=fx, pg=2, nmax=K.nmax, η=K.η)
    vy = rfmm2d(K.eps, pts; charges=fy, pg=2, nmax=K.nmax, η=K.η)
    vm = rfmm2d(K.eps, pts; charges=mchg, pg=2, nmax=K.nmax, η=K.η)

    A = K.A
    cφ = A * (4K.ν - 3)   # coefficient of (φx, φy)
    @inbounds for j in 1:n
        x1 = pts[1, j]
        x2 = pts[2, j]
        φx, φy = vx.pot[j], vy.pot[j]
        gxx, gxy = vx.grad[1, j], vx.grad[2, j]
        gyx, gyy = vy.grad[1, j], vy.grad[2, j]
        gmx, gmy = vm.grad[1, j], vm.grad[2, j]
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
)
    d, n = size(points)
    d == 2 || throw(ArgumentError("expected 2 × N points for Kelvin 2D"))
    μ = float(μ)
    ν = float(ν)
    A = 1 / (8π * μ * (1 - ν))
    return KelvinFMMMatrix(Matrix{Float64}(points), n, A, ν, eps, nmax, η)
end
