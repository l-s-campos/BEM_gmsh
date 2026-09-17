# Depth-1 HODLR LU of [A B; C D] (hm-toolbox hodlr_lu at one level).
# Off-diagonals B ≈ UB*VB', C ≈ UC*VC' by partial ACA (no SVD).
#   L11 U11 = P A
#   U12 = L11 \ (P UB),   V12 = VB
#   U21 = UC,             V21 = U11^{-T} VC
#   S = D - U21 (V21' U12) V12'
#   L22 U22 = P2 S

"""LU factors of a 2×2 HODLR node `[A B; C D]`."""
struct HODLR2x2LU{T, TF1, TF2}
    FA::TF1
    FS::TF2
    U12::Matrix{T}
    V12::Matrix{T}
    U21::Matrix{T}
    V21::Matrix{T}
    nu::Int
    nq::Int
end

Base.size(F::HODLR2x2LU) = (F.nu + F.nq, F.nu + F.nq)
Base.size(F::HODLR2x2LU, d::Integer) = d == 1 || d == 2 ? F.nu + F.nq : 1
Base.eltype(::HODLR2x2LU{T}) where {T} = T
function Base.show(io::IO, F::HODLR2x2LU)
    return print(io, "HODLR2x2LU{", eltype(F), "} n=", size(F, 1),
        " rank(B)=", size(F.U12, 2), " rank(C)=", size(F.U21, 2))
end
Base.show(io::IO, ::MIME"text/plain", F::HODLR2x2LU) = show(io, F)

function _hodlr_lr(M::AbstractMatrix{T}; rtol) where {T}
    m, n = size(M)
    (m == 0 || n == 0) && return zeros(T, m, 0), zeros(T, n, 0)
    R = PartialACA(; rtol = float(rtol))(M)
    return R.A, R.B
end

"""
    hodlr_lu_2x2(A, B, C, D; rtol=1e-8) -> HODLR2x2LU

HODLR LU of `[A B; C D]`. `B` and `C` are ACA-compressed; `A` and the Schur
complement of `D` are dense LU.
"""
function hodlr_lu_2x2(A::AbstractMatrix, B::AbstractMatrix, C::AbstractMatrix,
        D::AbstractMatrix; rtol = 1e-8)
    T = promote_type(eltype(A), eltype(B), eltype(C), eltype(D))
    Am = Matrix{T}(A)
    Bm = Matrix{T}(B)
    Cm = Matrix{T}(C)
    Dm = Matrix{T}(D)
    nu, nq = size(Am, 1), size(Dm, 1)
    size(Am) == (nu, nu) || throw(DimensionMismatch("A must be square"))
    size(Dm) == (nq, nq) || throw(DimensionMismatch("D must be square"))
    size(Bm) == (nu, nq) && size(Cm) == (nq, nu) ||
        throw(DimensionMismatch("B,C shapes"))
    FA = lu(Am)
    UB, VB = _hodlr_lr(Bm; rtol = rtol)
    UC, VC = _hodlr_lr(Cm; rtol = rtol)
    U12 = FA.L \ UB[FA.p, :]
    V12 = VB
    U21 = UC
    V21 = FA.U' \ VC
    S = Dm .- U21 * ((V21' * U12) * V12')
    FS = lu(S)
    return HODLR2x2LU{T, typeof(FA), typeof(FS)}(FA, FS, U12, V12, U21, V21, nu, nq)
end

function _hodlr_solve!(x::AbstractVector, F::HODLR2x2LU, b::AbstractVector)
    n = F.nu + F.nq
    length(x) == n && length(b) == n || throw(DimensionMismatch())
    b1 = view(b, 1:F.nu)
    b2 = view(b, (F.nu + 1):n)
    y1 = F.FA.L \ b1[F.FA.p]
    y2 = F.FS.L \ (b2[F.FS.p] .- F.U21 * (F.V21' * y1))
    x2 = F.FS.U \ y2
    x1 = F.FA.U \ (y1 .- F.U12 * (F.V12' * x2))
    copyto!(view(x, 1:F.nu), x1)
    copyto!(view(x, (F.nu + 1):n), x2)
    return x
end

function LinearAlgebra.ldiv!(F::HODLR2x2LU, b::AbstractVector)
    x = similar(b)
    _hodlr_solve!(x, F, b)
    copyto!(b, x)
    return b
end

function LinearAlgebra.ldiv!(x::AbstractVector, F::HODLR2x2LU, b::AbstractVector)
    return _hodlr_solve!(x, F, b)
end

function Base.:\(F::HODLR2x2LU, b::AbstractVector)
    return _hodlr_solve!(similar(b), F, b)
end

# HODLR LU with HSS on all four tiles. ULV on A and on S = D - C A^{-1} B.
# B, C stay rectangular HSS (no ACA). Schur matvec: S*v = D*v - C*(A \ (B*v)).

"""HODLR 2×2 with HSS `A,B,C,D` and [`ulv`](@ref) of `A` and of the Schur of `D`."""
struct HODLR2x2ULV{T, TFA, TFS, TB, TC}
    FA::TFA
    FS::TFS
    B::TB
    C::TC
    nu::Int
    nq::Int
end

Base.size(F::HODLR2x2ULV) = (F.nu + F.nq, F.nu + F.nq)
Base.size(F::HODLR2x2ULV, d::Integer) = d == 1 || d == 2 ? F.nu + F.nq : 1
Base.eltype(::HODLR2x2ULV{T}) where {T} = T
function Base.show(io::IO, F::HODLR2x2ULV)
    return print(io, "HODLR2x2ULV{", eltype(F), "} n=", size(F, 1),
        " B=", size(F.B), " C=", size(F.C))
end
Base.show(io::IO, ::MIME"text/plain", F::HODLR2x2ULV) = show(io, F)

"""Lazy `S = D - C (A \\ B)` for HSS assembly of the Schur complement."""
struct HSSSchurHODLR{T, TD, TFA, TB, TC} <: AbstractMatrix{T}
    D::TD
    FA::TFA
    B::TB
    C::TC
end
Base.size(S::HSSSchurHODLR) = size(S.D)
Base.eltype(::HSSSchurHODLR{T}) where {T} = T

function _schur_col(S::HSSSchurHODLR{T}, j::Int) where {T}
    e = zeros(T, size(S, 2))
    e[j] = one(T)
    return S.D * e .- S.C * (S.FA \ (S.B * e))
end

function Base.getindex(S::HSSSchurHODLR, i::Int, j::Int)
    return _schur_col(S, j)[i]
end

function Base.Matrix(S::HSSSchurHODLR{T}) where {T}
    n = size(S, 1)
    M = Matrix{T}(undef, n, n)
    @inbounds for j in 1:n
        M[:, j] = _schur_col(S, j)
    end
    return M
end

function _kernel_block!(out::AbstractMatrix{T}, S::HSSSchurHODLR{T}, I::Vector{Int},
        J::Vector{Int}) where {T}
    m, n = length(I), length(J)
    (m == 0 || n == 0) && return out
    nq = size(S, 1)
    Z = Matrix{T}(undef, nq, n)
    @inbounds for k in 1:n
        Z[:, k] = _schur_col(S, J[k])
    end
    @inbounds for k in 1:n, t in 1:m
        out[t, k] = Z[I[t], k]
    end
    return out
end

function _hss_square(A::AbstractMatrix, pts; rtol, nmax, method, Tmax)
    tree = ClusterTree(collect(pts), PrincipalComponentSplitter(; nmax=Int(nmax)))
    return assemble_hss(A, tree; rtol=float(rtol), method=method, Tmax=Tmax, symm=:n)
end

function _factor_square(A::AbstractMatrix, pts; rtol, nmax, method, Tmax)
    n = size(A, 1)
    n <= Int(nmax) && return lu(Matrix(A))
    return ulv(_hss_square(A, pts; rtol=rtol, nmax=nmax, method=method, Tmax=Tmax))
end

"""
    hodlr_ulv_2x2(A, B, C, D, pts_u, pts_q; rtol=1e-8, nmax=32) -> HODLR2x2ULV

HODLR LU of `[A B; C D]`: HSS on all four tiles, ULV of `A` and of
`S = D - C (A \\ B)`. `B` and `C` stay rectangular HSS.
"""
function hodlr_ulv_2x2(A::AbstractMatrix, B::AbstractMatrix, C::AbstractMatrix,
        D::AbstractMatrix, pts_u, pts_q;
        rtol = 1e-8, nmax::Integer = 32, method::Symbol = :id, Tmax = Inf)
    T = promote_type(eltype(A), eltype(B), eltype(C), eltype(D))
    Am, Bm = Matrix{T}(A), Matrix{T}(B)
    Cm, Dm = Matrix{T}(C), Matrix{T}(D)
    nu, nq = size(Am, 1), size(Dm, 1)
    kw = (; rtol=float(rtol), nmax=nmax, method=method, Tmax=Tmax)
    FA = _factor_square(Am, pts_u; kw...)
    tree_u = ClusterTree(collect(pts_u), PrincipalComponentSplitter(; nmax=Int(nmax)))
    tree_q = ClusterTree(collect(pts_q), PrincipalComponentSplitter(; nmax=Int(nmax)))
    HB = assemble_hss(Bm, tree_u, tree_q; rtol=float(rtol), method=method,
        Tmax=Tmax, symm=:n)
    HC = assemble_hss(Cm, tree_q, tree_u; rtol=float(rtol), method=method,
        Tmax=Tmax, symm=:n)
    HD = _hss_square(Dm, pts_q; kw...)
    Sop = HSSSchurHODLR{T, typeof(HD), typeof(FA), typeof(HB), typeof(HC)}(HD, FA, HB, HC)
    FS = _factor_square(Sop, pts_q; kw...)
    return HODLR2x2ULV{T, typeof(FA), typeof(FS), typeof(HB), typeof(HC)}(
        FA, FS, HB, HC, nu, nq)
end

function _hodlr_ulv_solve!(x::AbstractVector, F::HODLR2x2ULV{T}, b::AbstractVector) where {T}
    n = F.nu + F.nq
    length(x) == n && length(b) == n || throw(DimensionMismatch())
    b1 = Vector{T}(view(b, 1:F.nu))
    b2 = Vector{T}(view(b, (F.nu + 1):n))
    w = F.FA \ b1
    x2 = F.FS \ (b2 .- F.C * w)
    x1 = F.FA \ (b1 .- F.B * x2)
    copyto!(view(x, 1:F.nu), x1)
    copyto!(view(x, (F.nu + 1):n), x2)
    return x
end

function LinearAlgebra.ldiv!(F::HODLR2x2ULV, b::AbstractVector)
    x = similar(b)
    _hodlr_ulv_solve!(x, F, b)
    copyto!(b, x)
    return b
end

function LinearAlgebra.ldiv!(x::AbstractVector, F::HODLR2x2ULV, b::AbstractVector)
    return _hodlr_ulv_solve!(x, F, b)
end

function Base.:\(F::HODLR2x2ULV, b::AbstractVector)
    return _hodlr_ulv_solve!(similar(b), F, b)
end
