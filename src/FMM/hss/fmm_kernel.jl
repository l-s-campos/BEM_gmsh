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

"""Sample `A * Ω` using FMM matvecs (Martinsson §2.3)."""
function sample_matvec(A::FMMKernelMatrix{T}, Ω::AbstractMatrix{T}) where {T}
    n, rs = size(Ω)
    S = Matrix{T}(undef, n, rs)
    @inbounds for k in 1:rs
        mul!(view(S, :, k), A, view(Ω, :, k))
    end
    return S
end

"""Sample `A' * Ω` using adjoint FMM matvecs."""
function sample_matvec_adj(A::FMMKernelMatrix{T}, Ω::AbstractMatrix{T}) where {T}
    n, rs = size(Ω)
    S = Matrix{T}(undef, n, rs)
    x = Vector{T}(undef, n)
    y = Vector{T}(undef, n)
    @inbounds for k in 1:rs
        copyto!(x, view(Ω, :, k))
        A.matvec_adj!(y, x)
        copyto!(view(S, :, k), y)
    end
    return S
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
