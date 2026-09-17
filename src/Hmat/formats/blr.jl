"""
    mutable struct BLRMatrix{T} <: AbstractStructuredMatrix{T}

Block Low-Rank matrix: a flat (one-level) partitioning of a matrix into tiles.
Each tile is stored either as a dense `Matrix{T}` or an [`RkMatrix`](@ref).

# Fields
- `row_partition` / `col_partition`: contiguous index ranges for each block-row/column
- `blocks`: `nbrows × nbcols` array of tiles
- `admissible`: which tiles were compressed as low-rank
- `rowperm` / `colperm`: permutation from local (tree) ordering to the original
  global ordering; identity when no cluster tree is used
"""
mutable struct BLRMatrix{T} <: AbstractStructuredMatrix{T}
    row_partition::Vector{UnitRange{Int}}
    col_partition::Vector{UnitRange{Int}}
    blocks::Matrix{Union{Matrix{T}, RkMatrix{T}}}
    admissible::BitMatrix
    rowperm::Vector{Int}
    colperm::Vector{Int}
end

function BLRMatrix{T}(
        row_partition::Vector{UnitRange{Int}},
        col_partition::Vector{UnitRange{Int}};
        rowperm = collect(1:sum(length, row_partition; init = 0)),
        colperm = collect(1:sum(length, col_partition; init = 0)),
    ) where {T}
    nbrows = length(row_partition)
    nbcols = length(col_partition)
    blocks = Matrix{Union{Matrix{T}, RkMatrix{T}}}(undef, nbrows, nbcols)
    admissible = falses(nbrows, nbcols)
    return BLRMatrix{T}(row_partition, col_partition, blocks, admissible, rowperm, colperm)
end

function Base.size(B::BLRMatrix)
    return (sum(length, B.row_partition; init = 0), sum(length, B.col_partition; init = 0))
end
Base.eltype(::BLRMatrix{T}) where {T} = T

row_partition(B::BLRMatrix) = B.row_partition
col_partition(B::BLRMatrix) = B.col_partition
rowperm(B::BLRMatrix) = B.rowperm
colperm(B::BLRMatrix) = B.colperm

function Base.show(io::IO, B::BLRMatrix)
    m, n = size(B)
    nb = length(B.blocks)
    nlr = count(B.admissible)
    nbrows, nbcols = size(B.blocks)
    return print(
        io,
        "BLRMatrix{$(eltype(B))} of size $m × $n with $(nbrows)×$(nbcols) tiles ($nlr low-rank, $(nb - nlr) dense)",
    )
end
Base.show(io::IO, ::MIME"text/plain", B::BLRMatrix) = show(io, B)

"""
    leaf_ranges(tree) -> Vector{UnitRange{Int}}

Index ranges of the leaves of a [`ClusterTree`](@ref), in leaf order.
"""
leaf_ranges(tree) = map(index_range, leaves(tree))

"""
    assemble_blr([T,], K, row_partition, col_partition; kwargs...)

Assemble a [`BLRMatrix`](@ref) approximation of the matrix-like object `K`.

# Arguments
- `K`: object supporting `getindex` / [`getblock!`](@ref)
- `row_partition`, `col_partition`: `Vector{UnitRange{Int}}` of tile ranges, **or**
  `ClusterTree`s (leaf ranges are used)

# Keywords
- `adm`: admissibility predicate. For cluster-tree partitions, called on leaf
  nodes `(row_leaf, col_leaf)`. For raw ranges, called as `adm(i,j)` on tile indices.
  Default: `StrongAdmissibilityStd()` when trees are given, otherwise all off-diagonal
  tiles are compressed (`(i,j) -> i != j`).
- `comp`: low-rank compressor (default `PartialACA()`)
- `global_index`: permute `K` into the local ordering of the trees (default `true`
  when trees are passed)
- `threads`: assemble tiles in parallel
"""
function assemble_blr(::Type{T}, K, row_part, col_part; kwargs...) where {T}
    return _assemble_blr(T, K, row_part, col_part; kwargs...)
end

function assemble_blr(K::AbstractMatrix, row_part, col_part; kwargs...)
    return assemble_blr(eltype(K), K, row_part, col_part; kwargs...)
end

function _assemble_blr(
        ::Type{T},
        K,
        rowtree::ClusterTree,
        coltree::ClusterTree;
        adm = StrongAdmissibilityStd(),
        comp = PartialACA(),
        global_index = use_global_index(),
        threads = use_threads(),
    ) where {T}
    row_leaves = leaves(rowtree)
    col_leaves = leaves(coltree)
    row_part = map(index_range, row_leaves)
    col_part = map(index_range, col_leaves)
    rp = loc2glob(rowtree)
    cp = loc2glob(coltree)
    global_index && (K = PermutedMatrix(K, rp, cp))
    B = BLRMatrix{T}(row_part, col_part; rowperm = copy(rp), colperm = copy(cp))
    _fill_blr_tiles!(B, K, comp, threads) do i, j
        return adm(row_leaves[i], col_leaves[j])
    end
    return B
end

function _assemble_blr(
        ::Type{T},
        K,
        row_part::Vector{UnitRange{Int}},
        col_part::Vector{UnitRange{Int}};
        adm = nothing,
        comp = PartialACA(),
        global_index = false,
        threads = use_threads(),
        rowperm = collect(1:sum(length, row_part; init = 0)),
        colperm = collect(1:sum(length, col_part; init = 0)),
    ) where {T}
    adm_fun = isnothing(adm) ? ((i, j) -> i != j) : adm
    if global_index
        K = PermutedMatrix(K, rowperm, colperm)
    end
    B = BLRMatrix{T}(row_part, col_part; rowperm, colperm)
    _fill_blr_tiles!(B, K, comp, threads) do i, j
        return adm_fun(i, j)
    end
    return B
end

function _fill_blr_tiles!(adm_ij::F, B::BLRMatrix{T}, K, comp, threads) where {F, T}
    nbrows, nbcols = size(B.blocks)
    if threads
        acc = Threads.Atomic{Int}(1)
        np = Threads.nthreads()
        ntiles = nbrows * nbcols
        @sync for _ in 1:np
            Threads.@spawn begin
                buf = allocate_buffer(comp, T)
                while true
                    t = Threads.atomic_add!(acc, 1)
                    t > ntiles && break
                    i = mod1(t, nbrows)
                    j = div(t - 1, nbrows) + 1
                    _assemble_blr_tile!(B, K, comp, buf, adm_ij, i, j)
                end
            end
        end
    else
        buf = allocate_buffer(comp, T)
        for j in 1:nbcols, i in 1:nbrows
            _assemble_blr_tile!(B, K, comp, buf, adm_ij, i, j)
        end
    end
    return B
end

function _assemble_blr_tile!(B::BLRMatrix{T}, K, comp, buf, adm_ij, i, j) where {T}
    irange = B.row_partition[i]
    jrange = B.col_partition[j]
    if adm_ij(i, j)
        R = comp(K, irange, jrange, buf)
        m, n = length(irange), length(jrange)
        r = rank(R)
        if r * (m + n) >= m * n
            B.blocks[i, j] = Matrix(R)
            B.admissible[i, j] = false
        else
            B.blocks[i, j] = R
            B.admissible[i, j] = true
        end
    else
        out = Matrix{T}(undef, length(irange), length(jrange))
        getblock!(out, K, irange, jrange)
        B.blocks[i, j] = out
        B.admissible[i, j] = false
    end
    return nothing
end

# ---- conversion / queries ---------------------------------------------------

function Base.Matrix(B::BLRMatrix{T}; global_index = true) where {T}
    M = zeros(T, size(B)...)
    for j in 1:length(B.col_partition), i in 1:length(B.row_partition)
        irange = B.row_partition[i]
        jrange = B.col_partition[j]
        M[irange, jrange] = Matrix(B.blocks[i, j])
    end
    if global_index
        P = PermutedMatrix(M, invperm(B.rowperm), invperm(B.colperm))
        return Matrix(P)
    else
        return M
    end
end

function compression_ratio(B::BLRMatrix)
    ns = Base.summarysize(B)
    nr = length(B) * sizeof(eltype(B))
    return nr / ns
end

function maxrank(B::BLRMatrix)
    rmax = 0
    for j in 1:size(B.blocks, 2), i in 1:size(B.blocks, 1)
        blk = B.blocks[i, j]
        if blk isa RkMatrix
            rmax = max(rmax, rank(blk))
        end
    end
    return rmax
end

function Base.deepcopy_internal(B::BLRMatrix{T}, stackdict::IdDict) where {T}
    haskey(stackdict, B) && return stackdict[B]
    B2 = BLRMatrix{T}(
        copy(B.row_partition),
        copy(B.col_partition);
        rowperm = copy(B.rowperm),
        colperm = copy(B.colperm),
    )
    stackdict[B] = B2
    B2.admissible = copy(B.admissible)
    for j in 1:size(B.blocks, 2), i in 1:size(B.blocks, 1)
        B2.blocks[i, j] = deepcopy(B.blocks[i, j])
    end
    return B2
end

# ---- matvec -----------------------------------------------------------------

function LinearAlgebra.mul!(
        y::AbstractVector,
        B::BLRMatrix,
        x::AbstractVector,
        a::Number = 1,
        b::Number = 0;
        global_index = use_global_index(),
    )
    if global_index
        x = x[B.colperm]
        y = permute!(y, B.rowperm)
        rmul!(x, a)
    elseif a != 1
        x = a * x
    end
    iszero(b) ? fill!(y, zero(eltype(y))) : rmul!(y, b)
    for j in 1:length(B.col_partition)
        jrange = B.col_partition[j]
        xj = view(x, jrange)
        for i in 1:length(B.row_partition)
            irange = B.row_partition[i]
            mul!(view(y, irange), B.blocks[i, j], xj, true, true)
        end
    end
    global_index && invpermute!(y, B.rowperm)
    return y
end

function LinearAlgebra.mul!(
        Y::AbstractMatrix,
        B::BLRMatrix,
        X::AbstractMatrix,
        a::Number = 1,
        b::Number = 0;
        kwargs...,
    )
    size(Y, 2) == size(X, 2) || throw(DimensionMismatch("size(Y,2) != size(X,2)"))
    for k in 1:size(Y, 2)
        mul!(view(Y, :, k), B, view(X, :, k), a, b; kwargs...)
    end
    return Y
end

# ---- block LU factorisation (no pivoting) -----------------------------------

const BLR_LU = LU{<:Any, <:BLRMatrix}

function Base.getproperty(F::BLR_LU, s::Symbol)
    B = getfield(F, :factors)
    if s == :L
        return UnitLowerTriangular(B)
    elseif s == :U
        return UpperTriangular(B)
    else
        return getfield(F, s)
    end
end

"""
    lu!(B::BLRMatrix, compressor=PartialACA(); ...)

In-place block LU factorization of a square [`BLRMatrix`](@ref) without pivoting.
Off-diagonal low-rank tiles are updated with recompression via `compressor`.
"""
function LinearAlgebra.lu!(B::BLRMatrix, compressor; kwargs...)
    nblocks = size(B.blocks, 1)
    size(B.blocks, 1) == size(B.blocks, 2) ||
        throw(DimensionMismatch("BLR LU requires a square block layout"))
    size(B, 1) == size(B, 2) || throw(DimensionMismatch("BLR LU requires a square matrix"))
    for k in 1:nblocks
        Akk = B.blocks[k, k]
        Akk isa Matrix || error("diagonal BLR tiles must be dense for LU")
        lu!(Akk, NOPIVOT())
        for i in (k + 1):nblocks
            _blr_rdiv_U!(B.blocks[i, k], Akk)
            _blr_ldiv_L!(Akk, B.blocks[k, i])
        end
        for i in (k + 1):nblocks, j in (k + 1):nblocks
            _blr_schur_update!(B, i, j, k, compressor)
        end
    end
    return LU(B, Int[], 0)
end

function LinearAlgebra.lu!(
        B::BLRMatrix;
        atol = 0,
        rank = typemax(Int),
        rtol = atol > 0 || rank < typemax(Int) ? 0 : sqrt(eps(Float64)),
        kwargs...,
    )
    return lu!(B, PartialACA(; atol, rank, rtol); kwargs...)
end

LinearAlgebra.lu(B::BLRMatrix, args...; kwargs...) = lu!(deepcopy(B), args...; kwargs...)

function _blr_rdiv_U!(A::Matrix, U::Matrix)
    rdiv!(A, UpperTriangular(U))
    return A
end
function _blr_rdiv_U!(R::RkMatrix, U::Matrix)
    # R/U = A*(B' * inv(U)) = A*(inv(U')*B)'  => B <- U' \\ B
    ldiv!(UpperTriangular(U)', R.B)
    return R
end

function _blr_ldiv_L!(L::Matrix, A::Matrix)
    ldiv!(UnitLowerTriangular(L), A)
    return A
end
function _blr_ldiv_L!(L::Matrix, R::RkMatrix)
    ldiv!(UnitLowerTriangular(L), R.A)
    return R
end

function _blr_schur_update!(B::BLRMatrix{T}, i, j, k, compressor) where {T}
    Aik = B.blocks[i, k]
    Akj = B.blocks[k, j]
    Aij = B.blocks[i, j]
    if Aij isa Matrix
        _mul_tile!(Aij, Aik, Akj, -one(T), one(T))
    else
        M = Matrix(Aij)
        _mul_tile!(M, Aik, Akj, -one(T), one(T))
        R = compressor(M)
        m, n = size(M)
        r = rank(R)
        if r * (m + n) >= m * n
            B.blocks[i, j] = M
            B.admissible[i, j] = false
        else
            B.blocks[i, j] = R
            B.admissible[i, j] = true
        end
    end
    return nothing
end

function _mul_tile!(C::Matrix, A, B, α, β)
    if A isa Matrix && B isa Matrix
        mul!(C, A, B, α, β)
    elseif A isa RkMatrix && B isa Matrix
        tmp = A.Bt * B
        mul!(C, A.A, tmp, α, β)
    elseif A isa Matrix && B isa RkMatrix
        tmp = A * B.A
        mul!(C, tmp, B.Bt, α, β)
    else
        mid = A.Bt * B.A
        tmp = A.A * mid
        mul!(C, tmp, B.Bt, α, β)
    end
    return C
end

function LinearAlgebra.ldiv!(F::BLR_LU, y::AbstractVector; global_index = true)
    B = F.factors
    global_index && permute!(y, B.colperm)
    nblocks = size(B.blocks, 1)
    # L z = b
    for i in 1:nblocks
        ir = B.row_partition[i]
        yi = view(y, ir)
        for j in 1:(i - 1)
            jr = B.row_partition[j]
            mul!(yi, B.blocks[i, j], view(y, jr), -1, 1)
        end
        ldiv!(UnitLowerTriangular(B.blocks[i, i]), yi)
    end
    # U x = z
    for i in nblocks:-1:1
        ir = B.row_partition[i]
        yi = view(y, ir)
        for j in (i + 1):nblocks
            jr = B.row_partition[j]
            mul!(yi, B.blocks[i, j], view(y, jr), -1, 1)
        end
        ldiv!(UpperTriangular(B.blocks[i, i]), yi)
    end
    global_index && invpermute!(y, B.rowperm)
    return y
end
