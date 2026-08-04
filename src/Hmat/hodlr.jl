"""
    mutable struct HODLRMatrix{R,T} <: AbstractStructuredMatrix{T}

Hierarchically Off-Diagonal Low-Rank matrix on a binary [`ClusterTree`](@ref).

At a non-leaf node the block partitioning is
```
[ A00    B01 ]
[ B10    A11 ]
```
where `A00, A11` are themselves `HODLRMatrix` children and the off-diagonal
blocks `B01, B10` are stored as [`RkMatrix`](@ref). Leaves hold a dense diagonal
block `D`.
"""
mutable struct HODLRMatrix{R, T} <: AbstractStructuredMatrix{T}
    tree::R
    D::Union{Matrix{T}, Nothing}
    B01::Union{RkMatrix{T}, Nothing}
    B10::Union{RkMatrix{T}, Nothing}
    children::Vector{HODLRMatrix{R, T}}
    parentnode::HODLRMatrix{R, T}
    function HODLRMatrix{R, T}(tree, D, B01, B10, children, parentnode) where {R, T}
        H = new{R, T}(tree, D, B01, B10)
        H.children = isnothing(children) ? HODLRMatrix{R, T}[] : children
        H.parentnode = isnothing(parentnode) ? H : parentnode
        return H
    end
end

isleaf(H::HODLRMatrix) = isempty(H.children)
isroot(H::HODLRMatrix) = H.parentnode === H
children(H::HODLRMatrix) = H.children
parentnode(H::HODLRMatrix) = H.parentnode
index_range(H::HODLRMatrix) = index_range(H.tree)
Base.size(H::HODLRMatrix) = (length(H.tree), length(H.tree))
Base.eltype(::HODLRMatrix{R, T}) where {R, T} = T
rowperm(H::HODLRMatrix) = loc2glob(H.tree)
colperm(H::HODLRMatrix) = loc2glob(H.tree)

function Base.show(io::IO, H::HODLRMatrix)
    return print(io, "HODLRMatrix{$(eltype(H))} of size $(size(H,1)) × $(size(H,2))")
end
Base.show(io::IO, ::MIME"text/plain", H::HODLRMatrix) = show(io, H)

"""
    assemble_hodlr([T,], K, tree; comp=PartialACA(), global_index=true)

Assemble a square [`HODLRMatrix`](@ref) approximation of `K` on the binary
cluster `tree`. Off-diagonal blocks are compressed with `comp`; diagonal leaves
are stored densely.
"""
function assemble_hodlr(
        ::Type{T},
        K,
        tree::R;
        comp = PartialACA(),
        global_index = use_global_index(),
    ) where {T, R}
    global_index && (K = PermutedMatrix(K, loc2glob(tree), loc2glob(tree)))
    root = HODLRMatrix{R, T}(tree, nothing, nothing, nothing, nothing, nothing)
    _assemble_hodlr!(root, K, comp)
    return root
end

function assemble_hodlr(K::AbstractMatrix, tree; kwargs...)
    return assemble_hodlr(eltype(K), K, tree; kwargs...)
end

function _assemble_hodlr!(H::HODLRMatrix{R, T}, K, comp) where {R, T}
    node = H.tree
    if isleaf(node)
        irange = index_range(node)
        D = Matrix{T}(undef, length(irange), length(irange))
        getblock!(D, K, irange, irange)
        H.D = D
    else
        ch = children(node)
        length(ch) == 2 || error("HODLR requires a binary ClusterTree")
        c0 = HODLRMatrix{R, T}(ch[1], nothing, nothing, nothing, nothing, H)
        c1 = HODLRMatrix{R, T}(ch[2], nothing, nothing, nothing, nothing, H)
        H.children = [c0, c1]
        _assemble_hodlr!(c0, K, comp)
        _assemble_hodlr!(c1, K, comp)
        r0 = index_range(ch[1])
        r1 = index_range(ch[2])
        H.B01 = comp(K, r0, r1)
        H.B10 = comp(K, r1, r0)
    end
    return H
end

# ---- conversion / queries ---------------------------------------------------

function Base.Matrix(H::HODLRMatrix{R, T}; global_index = true) where {R, T}
    n = size(H, 1)
    M = zeros(T, n, n)
    _hodlr_to_matrix!(M, H)
    if global_index && isroot(H)
        p = rowperm(H)
        P = PermutedMatrix(M, invperm(p), invperm(p))
        return Matrix(P)
    else
        return M
    end
end

"""Fill local matrix `M` (size of `H`) from the HODLR structure."""
function _hodlr_to_matrix!(M, H::HODLRMatrix)
    if isleaf(H)
        n = size(H, 1)
        M[1:n, 1:n] = H.D
    else
        c0, c1 = H.children
        n0 = size(c0, 1)
        n1 = size(c1, 1)
        _hodlr_to_matrix!(view(M, 1:n0, 1:n0), c0)
        _hodlr_to_matrix!(view(M, (n0 + 1):(n0 + n1), (n0 + 1):(n0 + n1)), c1)
        M[1:n0, (n0 + 1):(n0 + n1)] = Matrix(H.B01)
        M[(n0 + 1):(n0 + n1), 1:n0] = Matrix(H.B10)
    end
    return M
end

function compression_ratio(H::HODLRMatrix)
    ns = Base.summarysize(H)
    nr = length(H) * sizeof(eltype(H))
    return nr / ns
end

function maxrank(H::HODLRMatrix)
    if isleaf(H)
        return 0
    else
        r = max(rank(H.B01), rank(H.B10))
        return max(r, maximum(maxrank, H.children))
    end
end

function Base.deepcopy_internal(H::HODLRMatrix{R, T}, stackdict::IdDict) where {R, T}
    haskey(stackdict, H) && return stackdict[H]
    H2 = HODLRMatrix{R, T}(H.tree, nothing, nothing, nothing, nothing, nothing)
    stackdict[H] = H2
    H2.D = isnothing(H.D) ? nothing : deepcopy(H.D)
    H2.B01 = isnothing(H.B01) ? nothing : deepcopy(H.B01)
    H2.B10 = isnothing(H.B10) ? nothing : deepcopy(H.B10)
    if !isempty(H.children)
        H2.children = [Base.deepcopy_internal(c, stackdict) for c in H.children]
        for c in H2.children
            c.parentnode = H2
        end
    end
    return H2
end

# ---- matvec -----------------------------------------------------------------

function LinearAlgebra.mul!(
        y::AbstractVector,
        H::HODLRMatrix,
        x::AbstractVector,
        a::Number = 1,
        b::Number = 0;
        global_index = use_global_index(),
    )
    if global_index
        p = colperm(H)
        x = x[p]
        y = permute!(y, rowperm(H))
        rmul!(x, a)
    elseif a != 1
        x = a * x
    end
    iszero(b) ? fill!(y, zero(eltype(y))) : rmul!(y, b)
    _hodlr_gemv!(y, H, x)
    global_index && invpermute!(y, rowperm(H))
    return y
end

function LinearAlgebra.mul!(
        Y::AbstractMatrix,
        H::HODLRMatrix,
        X::AbstractMatrix,
        a::Number = 1,
        b::Number = 0;
        kwargs...,
    )
    size(Y, 2) == size(X, 2) || throw(DimensionMismatch("size(Y,2) != size(X,2)"))
    for k in 1:size(Y, 2)
        mul!(view(Y, :, k), H, view(X, :, k), a, b; kwargs...)
    end
    return Y
end

function _hodlr_gemv!(y, H::HODLRMatrix, x)
    if isleaf(H)
        irange = index_range(H)
        mul!(view(y, irange), H.D, view(x, irange), true, true)
    else
        c0, c1 = H.children
        r0 = index_range(c0)
        r1 = index_range(c1)
        _hodlr_gemv!(y, c0, x)
        _hodlr_gemv!(y, c1, x)
        mul!(view(y, r0), H.B01, view(x, r1), true, true)
        mul!(view(y, r1), H.B10, view(x, r0), true, true)
    end
    return y
end

# ---- recursive factorisation ------------------------------------------------

const HODLR_LU = LU{<:Any, <:HODLRMatrix}

"""
    lu!(H::HODLRMatrix, compressor=PartialACA())

In-place recursive HODLR factorization (no pivoting). After factorization the
diagonal leaves store dense LU factors and off-diagonal blocks store the
corresponding L/U factors in low-rank form.
"""
function LinearAlgebra.lu!(H::HODLRMatrix, compressor; kwargs...)
    _hodlr_factor!(H, compressor)
    return LU(H, Int[], 0)
end

function LinearAlgebra.lu!(
        H::HODLRMatrix;
        atol = 0,
        rank = typemax(Int),
        rtol = atol > 0 || rank < typemax(Int) ? 0 : sqrt(eps(Float64)),
        kwargs...,
    )
    return lu!(H, PartialACA(; atol, rank, rtol); kwargs...)
end

LinearAlgebra.lu(H::HODLRMatrix, args...; kwargs...) = lu!(deepcopy(H), args...; kwargs...)

function _hodlr_factor!(H::HODLRMatrix{R, T}, compressor) where {R, T}
    if isleaf(H)
        lu!(H.D, NOPIVOT())
    else
        c0, c1 = H.children
        _hodlr_factor!(c0, compressor)
        # U01 = L00 \\ B01
        _hodlr_ldiv_L_local!(c0, H.B01)
        # L10 = B10 / U00
        _hodlr_rdiv_U_local!(H.B10, c0)
        # A11 <- A11 - L10 * U01
        _hodlr_schur!(c1, H.B10, H.B01, compressor)
        _hodlr_factor!(c1, compressor)
    end
    return H
end

function _hodlr_ldiv_L_local!(H::HODLRMatrix, R::RkMatrix)
    @assert size(R.A, 1) == size(H, 1)
    _hodlr_solve_L_mat_local!(H, R.A)
    return R
end

function _hodlr_solve_L_mat_local!(H::HODLRMatrix, X::AbstractMatrix)
    if isleaf(H)
        ldiv!(UnitLowerTriangular(H.D), X)
    else
        c0, c1 = H.children
        n0 = size(c0, 1)
        X0 = view(X, 1:n0, :)
        X1 = view(X, (n0 + 1):size(X, 1), :)
        _hodlr_solve_L_mat_local!(c0, X0)
        mul!(X1, H.B10, X0, -1, 1)
        _hodlr_solve_L_mat_local!(c1, X1)
    end
    return X
end

function _hodlr_rdiv_U_local!(R::RkMatrix, H::HODLRMatrix)
    @assert size(R.B, 1) == size(H, 1)
    _hodlr_solve_Uadjoint_mat_local!(H, R.B)
    return R
end

function _hodlr_solve_Uadjoint_mat_local!(H::HODLRMatrix, X::AbstractMatrix)
    if isleaf(H)
        ldiv!(UpperTriangular(H.D)', X)
    else
        c0, c1 = H.children
        n0 = size(c0, 1)
        X0 = view(X, 1:n0, :)
        X1 = view(X, (n0 + 1):size(X, 1), :)
        _hodlr_solve_Uadjoint_mat_local!(c0, X0)
        mul!(X1, H.B01.B, H.B01.At * X0, -1, 1)
        _hodlr_solve_Uadjoint_mat_local!(c1, X1)
    end
    return X
end

"""
A11 ← A11 - L10*U01 without densifying the whole A11 block.

Forms the product as a single [`RkMatrix`](@ref) and subtracts it level-by-level:
diagonal leaves get a dense rank-k update; off-diagonal blocks are merged as
low-rank factors and recompressed.
"""
function _hodlr_schur!(
        A11::HODLRMatrix{R, T},
        L10::RkMatrix{T},
        U01::RkMatrix{T},
        compressor,
    ) where {R, T}
    # L10*U01 = (L10.A * (L10.B' * U01.A)) * U01.B'
    mid = L10.Bt * U01.A
    P = RkMatrix(Matrix(L10.A * mid), Matrix(U01.B))
    _hodlr_axpy_rk!(A11, -one(T), P, compressor)
    return A11
end

"""H ← H + α P with P an RkMatrix on the local index set of H."""
function _hodlr_axpy_rk!(
        H::HODLRMatrix{R, T},
        α::Number,
        P::RkMatrix{T},
        compressor,
    ) where {R, T}
    @assert size(P, 1) == size(H, 1) && size(P, 2) == size(H, 2)
    if isleaf(H)
        mul!(H.D, P.A, P.Bt, α, true)
    else
        c0, c1 = H.children
        n0 = size(c0, 1)
        A0 = P.A[1:n0, :]
        A1 = P.A[(n0 + 1):end, :]
        B0 = P.B[1:n0, :]
        B1 = P.B[(n0 + 1):end, :]
        _hodlr_axpy_rk!(c0, α, RkMatrix(Matrix(A0), Matrix(B0)), compressor)
        _hodlr_axpy_rk!(c1, α, RkMatrix(Matrix(A1), Matrix(B1)), compressor)
        # off-diagonals: B01 += α A0 B1', B10 += α A1 B0'
        H.B01 = _rk_axpy_recompress(H.B01, α, A0, B1, compressor)
        H.B10 = _rk_axpy_recompress(H.B10, α, A1, B0, compressor)
    end
    return H
end

"""R ← recompress(R + α A B')."""
function _rk_axpy_recompress(
        R::RkMatrix{T},
        α::Number,
        A::AbstractMatrix{T},
        B::AbstractMatrix{T},
        compressor,
    ) where {T}
    A2 = hcat(R.A, α .* A)
    B2 = hcat(R.B, B)
    return _recompress_rk(RkMatrix(Matrix(A2), Matrix(B2)), compressor)
end

function _recompress_rk(R::RkMatrix{T}, compressor) where {T}
    m, n = size(R)
    r = rank(R)
    # cheap path: truncated SVD of the small Gram core via QR
    if r == 0
        return R
    end
    FA = qr(R.A)
    FB = qr(R.B)
    # thin R factors
    RA = FA.R
    RB = FB.R
    rA = min(size(RA, 1), size(RA, 2))
    rB = min(size(RB, 1), size(RB, 2))
    rA == 0 && return RkMatrix(zeros(T, m, 0), zeros(T, n, 0))
    core = RA[1:rA, 1:rA] * adjoint(RB[1:rB, 1:rB])
    # pad if rectangular ranks differ
    if rA != rB
        return compressor(Matrix(R))  # rare; safe fallback
    end
    F = svd(core)
    tol = compressor isa PartialACA ? max(compressor.atol, compressor.rtol * F.S[1]) : sqrt(eps(real(T))) * F.S[1]
    rnew = count(s -> s > tol, F.S)
    rnew = max(rnew, 0)
    rmax = compressor isa PartialACA ? min(compressor.rank, rA) : rA
    rnew = min(rnew, rmax)
    if rnew == 0
        return RkMatrix(zeros(T, m, 0), zeros(T, n, 0))
    end
    QA = Matrix(FA.Q)
    QB = Matrix(FB.Q)
    Anew = QA[:, 1:rA] * (F.U[:, 1:rnew] * Diagonal(sqrt.(F.S[1:rnew])))
    Bnew = QB[:, 1:rB] * (F.V[:, 1:rnew] * Diagonal(sqrt.(F.S[1:rnew])))
    if rnew * (m + n) >= m * n
        return compressor(Matrix(R)) isa RkMatrix ? compressor(Matrix(R)) : RkMatrix(Anew, Bnew)
    end
    return RkMatrix(Anew, Bnew)
end

# ---- solve ------------------------------------------------------------------

function LinearAlgebra.ldiv!(F::HODLR_LU, y::AbstractVector; global_index = true)
    H = F.factors
    global_index && permute!(y, colperm(H))
    _hodlr_solve_L!(H, y)
    _hodlr_solve_U!(H, y)
    global_index && invpermute!(y, rowperm(H))
    return y
end

function _hodlr_solve_L!(H::HODLRMatrix, y::AbstractVector)
    if isleaf(H)
        irange = index_range(H)
        ldiv!(UnitLowerTriangular(H.D), view(y, irange))
    else
        c0, c1 = H.children
        r0 = index_range(c0)
        r1 = index_range(c1)
        _hodlr_solve_L!(c0, y)
        mul!(view(y, r1), H.B10, view(y, r0), -1, 1)
        _hodlr_solve_L!(c1, y)
    end
    return y
end

function _hodlr_solve_U!(H::HODLRMatrix, y::AbstractVector)
    if isleaf(H)
        irange = index_range(H)
        ldiv!(UpperTriangular(H.D), view(y, irange))
    else
        c0, c1 = H.children
        r0 = index_range(c0)
        r1 = index_range(c1)
        _hodlr_solve_U!(c1, y)
        mul!(view(y, r0), H.B01, view(y, r1), -1, 1)
        _hodlr_solve_U!(c0, y)
    end
    return y
end
