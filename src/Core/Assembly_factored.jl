# Hierarchical / factored H,G infrastructure (physics-agnostic operators)
#
#   Bare kernels Du, Dq compressed separately; quadrature weights outside:
#     G x = Du (w ∘ x) + diag_G ∘ x
#     H x = Dq (w_H ∘ x) + diag_H ∘ x
#
# Same pattern as DIBEM  M x = D (c ∘ x) + diag ∘ x.
# Physics-specific kernels and H_G_Hmat live under Laplace/ (or Elasticity/).

export ColWeightedOp, node_weights, free_term
export BCIndexSets, bc_index_sets, nu, nq
export HGBlocks, partition_HG, BlockMixedOperator, IndexMapOp
export build_block_mixed_system, scatter_block_sol!, block_mixed_rhs
export BlockHLU, factor_block_hlu!, factor_block_hlu, ldiv_block_hlu!
export BlockULV, factor_block_ulv, ldiv_block_ulv!

"""
    node_weights(dad::BEMdata) -> Vector{Float64}

Integration weights for each boundary collocation node (`Jacobian × quad weight`).
"""
function node_weights(dad::BEMdata)
    w = zeros(dad.n)
    for elem in dad.elements
        for (k, node) in enumerate(elem.index)
            w[node] += elem.Jacobian[k] * dad.elem_weight[k]
        end
    end
    return w
end

"""
    free_term(dad, i) -> Float64

Diagonal free-term entry placed in ``H_ii`` for discontinuous collocation
(scalar problems).

| location | ``c`` | ``H_ii = free_term`` |
|----------|-------|----------------------|
| boundary | ``1/2`` | ``-1/2`` |
| internal | ``1`` | ``-1`` |
"""
@inline free_term(dad::BEMdata, i::Integer) = i <= dad.n ? -0.5 : -1.0

# ---------------------------------------------------------------------------
# Column-weighted operator  A x = K (w ∘ x) + diag ∘ x
# ---------------------------------------------------------------------------

"""
    ColWeightedOp

Matrix-free
```
A x = K (w ∘ x) + d ∘ x
```
with compressed bare kernel `K` and column weights `w` (length = `ncols`).
Diagonal free-term / regularisation lives in `d` (length = `nrows`).

DIBEM uses the same type: `M x = D (c ∘ x) + d ∘ x` (`w` ↔ `c`).
"""
mutable struct ColWeightedOp{TK} <: AbstractMatrix{Float64}
    K::TK
    w::Vector{Float64}
    d::Vector{Float64}
    nrows::Int
    ncols::Int
    """Optional sparse near-field correction: full ∫ − pointwise kernel×w."""
    corr::Union{Nothing, SparseMatrixCSC{Float64,Int}}
end

ColWeightedOp(K, w, d, nrows, ncols) =
    ColWeightedOp(K, w, d, nrows, ncols, nothing)

Base.size(A::ColWeightedOp) = (A.nrows, A.ncols)
Base.IndexStyle(::Type{<:ColWeightedOp}) = IndexCartesian()

function Base.getindex(A::ColWeightedOp, i::Int, j::Int)
    val = A.w[j] * A.K[i, j]
    if i == j && j <= length(A.d)
        val += A.d[i]
    end
    if A.corr !== nothing && 1 <= i <= size(A.corr, 1) && 1 <= j <= size(A.corr, 2)
        val += A.corr[i, j]
    end
    return val
end

function LinearAlgebra.mul!(y::AbstractVector, A::ColWeightedOp, x::AbstractVector)
    length(x) == A.ncols && length(y) == A.nrows || throw(DimensionMismatch())
    mul!(y, A.K, A.w .* x)
    n = min(A.nrows, A.ncols, length(A.d))
    @inbounds for i in 1:n
        y[i] += A.d[i] * x[i]
    end
    if A.corr !== nothing
        mul!(y, A.corr, x, 1, 1)  # y += corr * x
    end
    return y
end

function LinearAlgebra.mul!(y::AbstractVector, A::ColWeightedOp, x::AbstractVector,
        α::Number, β::Number)
    if iszero(β)
        mul!(y, A, x)
        α != 1 && rmul!(y, α)
    else
        t = similar(y)
        mul!(t, A, x)
        y .= β .* y .+ α .* t
    end
    return y
end

Base.:*(A::ColWeightedOp, x::AbstractVector) = mul!(similar(x, Float64, A.nrows), A, x)

# =============================================================================
# BC index sets and H/G(/M) block partition
# =============================================================================
# Collocation split (scalar):
#   u  — Neumann boundary + all internals  → unknown T  (known q on boundary)
#   q  — Dirichlet boundary                → unknown q  (known T)
#
# Full BIE  H T − G q = f  becomes the packed mixed system
#
#   [ Huu  −Guq ] [ T_u ]   [  Guu q_u − Huq T_q ]     (+ optional mass on T)
#   [ Hqu  −Gqq ] [ q_q ] = [  Gqu q_u − Hqq T_q ]
#
# With domain mass M (Helmholtz correlato H+κ²M, etc.):
#   Huu ← Huu+κ² Muu,  Huq ← Huq+κ² Muq, …
# =============================================================================

"""
    BCIndexSets

Index partition of collocation DOFs for mixed BCs.

- `u`: Neumann boundary nodes ∪ internal collocation (unknown potential)
- `q`: Dirichlet boundary nodes (unknown flux)

`u` and `q` are complementary over `1:nt` for H rows/cols; G columns only
exist on the boundary, so `u ∩ (1:n)` and `q` partition `1:n`.
"""
struct BCIndexSets
    u::Vector{Int}   # unknown T
    q::Vector{Int}   # unknown q (Dirichlet boundary)
    n::Int           # boundary size
    nt::Int          # total collocation
end

Base.length(idx::BCIndexSets) = idx.nt
nu(idx::BCIndexSets) = length(idx.u)
nq(idx::BCIndexSets) = length(idx.q)

"""
    bc_index_sets(dad) -> BCIndexSets
    bc_index_sets(BC, n, nt) -> BCIndexSets

Build the `u`/`q` collocation partition from BC flags
(`0` = Dirichlet, `1` = Neumann; internals always `u`).
"""
function bc_index_sets(BC::AbstractVector{<:Integer}, n::Integer, nt::Integer)
    length(BC) >= n || throw(DimensionMismatch("BC length $(length(BC)) < n=$n"))
    u = Int[]
    q = Int[]
    sizehint!(u, nt)
    sizehint!(q, n)
    @inbounds for j in 1:n
        if BC[j] == 0
            push!(q, j)
        else
            push!(u, j)
        end
    end
    @inbounds for j in (n + 1):nt
        push!(u, j)
    end
    return BCIndexSets(u, q, Int(n), Int(nt))
end

bc_index_sets(dad::BEMdata) = bc_index_sets(dad.BC, dad.n, dad.nt)

"""
    HGBlocks

Four-block partition of influence operators:

```
H = [Huu Huq; Hqu Hqq] ,   G = [Guu Guq; Gqu Gqq]
```

Optional mass blocks `M**` (same layout as `H`) when a domain operator is present.
Blocks may be dense `SubArray`s (views) or matrix-free [`IndexMapOp`](@ref)s.
"""
struct HGBlocks{HUU,HUQ,HQU,HQQ,GUU,GUQ,GQU,GQQ,MUU,MUQ,MQU,MQQ}
    idx::BCIndexSets
    Huu::HUU; Huq::HUQ; Hqu::HQU; Hqq::HQQ
    Guu::GUU; Guq::GUQ; Gqu::GQU; Gqq::GQQ
    Muu::MUU; Muq::MUQ; Mqu::MQU; Mqq::MQQ
    has_M::Bool
end

# ---------------------------------------------------------------------------
# Index-mapped matvec wrapper (H-matrix / FMM safe — no getindex)
# ---------------------------------------------------------------------------

"""
    IndexMapOp(A, rows, cols)

Matrix-free view `B = A[rows, cols]` implemented by scatter → `A*v` → gather.
Works for any `A` with `mul!(y, A, x)` (dense, `ColWeightedOp`, `HMatrix`, …).
"""
struct IndexMapOp{TA} <: AbstractMatrix{Float64}
    A::TA
    rows::Vector{Int}
    cols::Vector{Int}
    # work buffers (length = full A sizes)
    xfull::Vector{Float64}
    yfull::Vector{Float64}
end

function IndexMapOp(A, rows::AbstractVector{<:Integer}, cols::AbstractVector{<:Integer})
    mA, nA = size(A)
    return IndexMapOp{typeof(A)}(
        A, collect(Int, rows), collect(Int, cols),
        zeros(nA), zeros(mA),
    )
end

Base.size(B::IndexMapOp) = (length(B.rows), length(B.cols))
Base.eltype(::IndexMapOp) = Float64

function LinearAlgebra.mul!(y::AbstractVector, B::IndexMapOp, x::AbstractVector)
    length(x) == length(B.cols) && length(y) == length(B.rows) ||
        throw(DimensionMismatch("IndexMapOp mul size"))
    fill!(B.xfull, 0.0)
    @inbounds for (j, jc) in enumerate(B.cols)
        B.xfull[jc] = x[j]
    end
    mul!(B.yfull, B.A, B.xfull)
    @inbounds for (i, ir) in enumerate(B.rows)
        y[i] = B.yfull[ir]
    end
    return y
end

function LinearAlgebra.mul!(y::AbstractVector, B::IndexMapOp, x::AbstractVector,
        α::Number, β::Number)
    if iszero(β)
        mul!(y, B, x)
        α != 1 && rmul!(y, α)
    else
        t = similar(y)
        mul!(t, B, x)
        y .= β .* y .+ α .* t
    end
    return y
end

Base.:*(B::IndexMapOp, x::AbstractVector) = mul!(similar(x, Float64, size(B, 1)), B, x)

function Base.getindex(B::IndexMapOp, i::Int, j::Int)
    return Float64(B.A[B.rows[i], B.cols[j]])
end

# dense path: true views (column swap / factorize friendly)
# Only Array-backed matrices — HMatrix/ColWeightedOp are AbstractMatrix but
# must not use getindex-based views.
_block_view(A::Array, rows, cols) = view(A, rows, cols)
_block_view(A::SubArray, rows, cols) = view(A, rows, cols)
_block_view(A, rows, cols) = IndexMapOp(A, rows, cols)

"""
    partition_HG(H, G, idx::BCIndexSets; M=nothing) -> HGBlocks

Split `H` (`nt×nt`) and `G` (`nt×n`) into the four BC blocks. When `M` is given
(same size as `H`), mass blocks are included.

Dense `Matrix` inputs yield `SubArray` views (zero-copy). Hierarchical /
factored operators yield [`IndexMapOp`](@ref) wrappers.
"""
function partition_HG(H, G, idx::BCIndexSets; M=nothing)
    u, q = idx.u, idx.q
    # G columns only on boundary: u_b = u ∩ (1:n)
    u_b = filter(j -> j <= idx.n, u)

    Huu = _block_view(H, u, u)
    Huq = _block_view(H, u, q)
    Hqu = _block_view(H, q, u)
    Hqq = _block_view(H, q, q)

    Guu = _block_view(G, u, u_b)
    Guq = _block_view(G, u, q)
    Gqu = _block_view(G, q, u_b)
    Gqq = _block_view(G, q, q)

    if M === nothing
        z = nothing
        return HGBlocks(idx, Huu, Huq, Hqu, Hqq, Guu, Guq, Gqu, Gqq,
                        z, z, z, z, false)
    end
    Muu = _block_view(M, u, u)
    Muq = _block_view(M, u, q)
    Mqu = _block_view(M, q, u)
    Mqq = _block_view(M, q, q)
    return HGBlocks(idx, Huu, Huq, Hqu, Hqq, Guu, Guq, Gqu, Gqq,
                    Muu, Muq, Mqu, Mqq, true)
end

partition_HG(H, G, dad::BEMdata; M=nothing) =
    partition_HG(H, G, bc_index_sets(dad); M=M)

# ---------------------------------------------------------------------------
# Packed mixed operator from blocks
# ---------------------------------------------------------------------------

"""
    BlockMixedOperator

Packed mixed-BC operator acting on `x = [T_u; q_q]`:

```
A x = [ Huu T_u − Guq q_q ] + κ² [ Muu T_u ]     (rows u)
      [ Hqu T_u − Gqq q_q ] + κ² [ Mqu T_u ]     (rows q)
```

Size `(n_u + n_q)² = nt × nt`. Prefer this over column-swap when `H`/`G` are
hierarchical or FMM-backed — whole subblocks are applied via matvec.
"""
struct BlockMixedOperator{B<:HGBlocks} <: AbstractMatrix{Float64}
    blocks::B
    κ2::Float64
    # workspaces
    Tu::Vector{Float64}
    qq::Vector{Float64}
    yu::Vector{Float64}
    yq::Vector{Float64}
    tmpu::Vector{Float64}
    tmpq::Vector{Float64}
end

function BlockMixedOperator(blocks::HGBlocks; κ2::Real=0.0)
    nu_ = nu(blocks.idx)
    nq_ = nq(blocks.idx)
    return BlockMixedOperator(blocks, float(κ2),
        zeros(nu_), zeros(nq_), zeros(nu_), zeros(nq_), zeros(nu_), zeros(nq_))
end

function Base.size(A::BlockMixedOperator)
    n = nu(A.blocks.idx) + nq(A.blocks.idx)
    return (n, n)
end

function LinearAlgebra.mul!(y::AbstractVector, A::BlockMixedOperator, x::AbstractVector)
    B = A.blocks
    nu_ = nu(B.idx)
    nq_ = nq(B.idx)
    length(x) == nu_ + nq_ && length(y) == nu_ + nq_ || throw(DimensionMismatch())

    Tu = A.Tu; qq = A.qq; yu = A.yu; yq = A.yq
    copyto!(Tu, 1, x, 1, nu_)
    copyto!(qq, 1, x, nu_ + 1, nq_)

    # yu = Huu*Tu - Guq*qq
    mul!(yu, B.Huu, Tu)
    mul!(A.tmpu, B.Guq, qq)
    yu .-= A.tmpu
    # yq = Hqu*Tu - Gqq*qq
    mul!(yq, B.Hqu, Tu)
    mul!(A.tmpq, B.Gqq, qq)
    yq .-= A.tmpq

    if B.has_M && A.κ2 != 0
        mul!(A.tmpu, B.Muu, Tu)
        yu .+= A.κ2 .* A.tmpu
        mul!(A.tmpq, B.Mqu, Tu)
        yq .+= A.κ2 .* A.tmpq
    end

    copyto!(y, 1, yu, 1, nu_)
    copyto!(y, nu_ + 1, yq, 1, nq_)
    return y
end

function LinearAlgebra.mul!(y::AbstractVector, A::BlockMixedOperator, x::AbstractVector,
        α::Number, β::Number)
    if iszero(β)
        mul!(y, A, x)
        α != 1 && rmul!(y, α)
    else
        t = similar(y)
        mul!(t, A, x)
        y .= β .* y .+ α .* t
    end
    return y
end

Base.:*(A::BlockMixedOperator, x::AbstractVector) =
    mul!(similar(x, Float64, size(A, 1)), A, x)

"""
    block_mixed_rhs(blocks, dad; κ2=0) -> b

RHS for [`BlockMixedOperator`](@ref):

```
b_u =  Guu q_u − Huq T_q − κ² Muq T_q
b_q =  Gqu q_u − Hqq T_q − κ² Mqq T_q
```
"""
function block_mixed_rhs(blocks::HGBlocks, dad::BEMdata; κ2::Real=0.0)
    idx = blocks.idx
    nu_ = nu(idx); nq_ = nq(idx)
    κ2 = float(κ2)

    # known vectors in block coordinates
    Tu_known = zeros(nu_)          # always 0 (unknowns live in x)
    Tq_known = zeros(nq_)
    qu_known = zeros(count(j -> j <= idx.n, idx.u))
    # map known boundary values
    u_b = filter(j -> j <= idx.n, idx.u)
    @inbounds for (k, j) in enumerate(idx.q)
        Tq_known[k] = dad.BV[j]          # Dirichlet potential
    end
    @inbounds for (k, j) in enumerate(u_b)
        qu_known[k] = dad.BV[j]          # Neumann flux
    end

    bu = zeros(nu_)
    bq = zeros(nq_)
    tmpu = zeros(nu_)
    tmpq = zeros(nq_)

    # b = G * q_known - H * T_known  (+ −κ² M T_known)
    if !isempty(qu_known)
        mul!(bu, blocks.Guu, qu_known)
        mul!(bq, blocks.Gqu, qu_known)
    end
    if !isempty(Tq_known)
        mul!(tmpu, blocks.Huq, Tq_known)
        bu .-= tmpu
        mul!(tmpq, blocks.Hqq, Tq_known)
        bq .-= tmpq
        if blocks.has_M && κ2 != 0
            mul!(tmpu, blocks.Muq, Tq_known)
            bu .-= κ2 .* tmpu
            mul!(tmpq, blocks.Mqq, Tq_known)
            bq .-= κ2 .* tmpq
        end
    end

    b = zeros(nu_ + nq_)
    copyto!(b, 1, bu, 1, nu_)
    copyto!(b, nu_ + 1, bq, 1, nq_)
    return b
end

"""
    build_block_mixed_system(H, G, dad; M=nothing, κ2=0) -> (; A, b, blocks, idx)

Construct packed mixed system `A x = b` with `x = [T_u; q_q]`.
"""
function build_block_mixed_system(H, G, dad::BEMdata; M=nothing, κ2::Real=0.0)
    idx = bc_index_sets(dad)
    blocks = partition_HG(H, G, idx; M=M)
    A = BlockMixedOperator(blocks; κ2=κ2)
    b = block_mixed_rhs(blocks, dad; κ2=κ2)
    return (; A, b, blocks, idx)
end

# =============================================================================
# One-level hierarchical LU on the 2×2 BC block structure
# =============================================================================
# Same recursion as HMatrices `_lu!` on a single parent with 2×2 children:
#
#   for i = 1:2
#       lu!(A[i,i])
#       for j > i
#           ldiv!(UnitLowerTriangular(A[i,i]), A[i,j])   # U12 = L11 \ A12
#           rdiv!(A[j,i], UpperTriangular(A[i,i]))       # L21 = A21 / U11
#       end
#       for j,k > i
#           A[j,k] -= L21 * U12                          # Schur
#       end
#   end
#
# Packed mixed BC operator (never assembled as one matrix):
#
#   [ A11  A12 ] = [ Huu(+κ²Muu)   −Guq ]
#   [ A21  A22 ]   [ Hqu(+κ²Mqu)   −Gqq ]
#
# After `_block_lu!` the four blocks hold L/U factors in place (MATLAB-style
# combined LU on diagonals; L21 and U12 overwrite A21 and A12).
# =============================================================================

"""
    BlockHLU

One-level hierarchical LU of the packed mixed BC operator, stored as the four
BC blocks only — the same idea as [`HMatrices._lu!`](@ref) with `m=n=2` children.

| field | after factor |
|-------|----------------|
| `A11` | combined LU of Huu(+κ²Muu) |
| `U12` | `L11 \\ (−Guq)` |
| `L21` | `(Hqu(+κ²Mqu)) / U11` |
| `A22` | combined LU of Schur `−Gqq − L21*U12` |

Never builds the full `(n_u+n_q)²` matrix.
"""
struct BlockHLU{TA11,TU12,TL21,TA22}
    idx::BCIndexSets
    A11::TA11
    U12::TU12          # nothing if nq=0
    L21::TL21
    A22::TA22
    κ2::Float64
end

Base.size(F::BlockHLU) = (F.idx.nt, F.idx.nt)
Base.size(F::BlockHLU, d::Integer) =
    d == 1 || d == 2 ? F.idx.nt : throw(ArgumentError("invalid dimension $d"))

function Base.show(io::IO, ::MIME"text/plain", F::BlockHLU)
    nu_, nq_ = nu(F.idx), nq(F.idx)
    print(io, "BlockHLU one-level H-LU  A11=$(size(F.A11))")
    nq_ > 0 && print(io, "  U12=$(size(F.U12))  L21=$(size(F.L21))  A22=$(size(F.A22))")
    print(io, "  (nu=$nu_, nq=$nq_)")
end

"""Own dense copy of a block. Hierarchical / IndexMapOp densified by matvecs."""
_as_matrix(A::Array{Float64}) = copy(A)
_as_matrix(A::SubArray) = Matrix{Float64}(A)
_as_matrix(A::Matrix) = copy(A)
function _as_matrix(A)   # IndexMapOp, ColWeightedOp, HMatrix, …
    m, n = size(A)
    out = zeros(Float64, m, n)
    x = zeros(Float64, n)
    @inbounds for j in 1:n
        fill!(x, 0.0); x[j] = 1.0
        mul!(view(out, :, j), A, x)
    end
    return out
end

"""
    factor_block_hlu!(A11, A12, A21, A22) -> nothing

In-place one-level H-LU on a 2×2 block array (mirrors `_lu!` for `m=n=2`):

1. `lu!(A11)`
2. `A12 ← L11 \\ A12`,  `A21 ← A21 / U11`
3. `A22 ← A22 − A21*A12`
4. `lu!(A22)`

All four arguments must be dense `Matrix`s owned by the caller.
`A12`/`A21`/`A22` may be empty (`nq=0`): only step 1 runs.
"""
function factor_block_hlu!(A11::AbstractMatrix, A12::AbstractMatrix,
                           A21::AbstractMatrix, A22::AbstractMatrix)
    # No pivoting — same convention as HMatrices._lu! leaf factorization,
    # so UnitLowerTriangular / UpperTriangular can wrap the combined storage.
    nopiv = VERSION >= v"1.7" ? NoPivot() : Val(false)
    # --- i = 1 diagonal ---
    lu!(A11, nopiv)
    nq_ = size(A12, 2)
    if nq_ == 0
        return nothing
    end
    # --- i = 1 off-diagonals ---
    ldiv!(UnitLowerTriangular(A11), A12)              # U12 = L11 \ A12
    rdiv!(A21, UpperTriangular(A11))                  # L21 = A21 / U11
    # --- Schur update on child[2,2] ---
    mul!(A22, A21, A12, -1.0, 1.0)                    # A22 -= L21 * U12
    # --- i = 2 diagonal ---
    lu!(A22, nopiv)
    return nothing
end

"""
    factor_block_hlu(A::BlockMixedOperator) -> BlockHLU
    factor_block_hlu(blocks; κ2=0) -> BlockHLU

Copy the four BC blocks, form
`A11=Huu(+κ²Muu)`, `A12=−Guq`, `A21=Hqu(+κ²Mqu)`, `A22=−Gqq`,
then run [`factor_block_hlu!`](@ref). Dense Array/SubArray blocks only.
"""
function factor_block_hlu(A::BlockMixedOperator)
    return factor_block_hlu(A.blocks; κ2=A.κ2)
end

function factor_block_hlu(blocks::HGBlocks; κ2::Real=0.0)
    κ2 = float(κ2)
    idx = blocks.idx
    nq_ = nq(idx)

    # Dense or hierarchical: materialize each block via matvecs when needed.
    # Hierarchical Huu still gets classical (one-level) LU on the densified tile —
    # same 2×2 H-LU recursion; full recursive H-LU on Huu alone is a future step.
    A11 = _as_matrix(blocks.Huu)
    if blocks.has_M && κ2 != 0
        A11 .+= κ2 .* _as_matrix(blocks.Muu)
    end

    if nq_ == 0
        factor_block_hlu!(A11, zeros(size(A11, 1), 0), zeros(0, size(A11, 1)), zeros(0, 0))
        return BlockHLU(idx, A11, nothing, nothing, nothing, κ2)
    end

    A12 = .-_as_matrix(blocks.Guq)          # becomes U12
    A21 = _as_matrix(blocks.Hqu)            # becomes L21
    A22 = .-_as_matrix(blocks.Gqq)          # becomes Schur LU
    if blocks.has_M && κ2 != 0
        A21 .+= κ2 .* _as_matrix(blocks.Mqu)
    end

    factor_block_hlu!(A11, A12, A21, A22)
    return BlockHLU(idx, A11, A12, A21, A22, κ2)
end

"""
    ldiv_block_hlu!(x, F::BlockHLU, b) -> x

Forward/back substitution on the one-level factors (same as H-LU `ldiv!`
with 2 block-rows):

```
# L y = b
y1 = L11 \\ b1
y2 = L22 \\ (b2 − L21 y1)
# U x = y
x2 = U22 \\ y2
x1 = U11 \\ (y1 − U12 x2)
```
"""
function ldiv_block_hlu!(x::AbstractVector, F::BlockHLU, b::AbstractVector)
    nu_ = nu(F.idx); nq_ = nq(F.idx)
    length(x) == nu_ + nq_ && length(b) == nu_ + nq_ || throw(DimensionMismatch())

    x1 = view(x, 1:nu_)
    b1 = view(b, 1:nu_)

    if nq_ == 0
        # single block: A11 holds combined LU
        ldiv!(UnitLowerTriangular(F.A11), copyto!(x1, b1))
        ldiv!(UpperTriangular(F.A11), x1)
        return x
    end

    x2 = view(x, nu_+1:nu_+nq_)
    b2 = view(b, nu_+1:nu_+nq_)

    # --- L y = b ---
    y1 = copy(b1)
    ldiv!(UnitLowerTriangular(F.A11), y1)            # y1 = L11 \ b1
    y2 = b2 .- F.L21 * y1                            # y2 = b2 − L21 y1
    ldiv!(UnitLowerTriangular(F.A22), y2)            # y2 = L22 \ y2

    # --- U x = y ---
    copyto!(x2, y2)
    ldiv!(UpperTriangular(F.A22), x2)                # x2 = U22 \ y2
    copyto!(x1, y1)
    mul!(x1, F.U12, x2, -1.0, 1.0)                   # x1 = y1 − U12 x2
    ldiv!(UpperTriangular(F.A11), x1)                # x1 = U11 \ x1
    return x
end

function ldiv_block_hlu(F::BlockHLU, b::AbstractVector)
    x = similar(b)
    return ldiv_block_hlu!(x, F, b)
end

Base.:\(F::BlockHLU, b::AbstractVector) = ldiv_block_hlu(F, b)

# =============================================================================
# Mixed-BC 2×2 with HSS on every tile, ULV on the square diagonals
# =============================================================================
#   [ A  B ] [x1]   [ Huu(+κ²Muu)   −Guq ] [ T_u ]
#   [ C  D ] [x2] = [ Hqu(+κ²Mqu)   −Gqq ] [ q_q ]
#
# All four tiles are HSS. A and D are square → ulv. Off-diagonals B, C are
# rectangular HSS (matvec only). Nested substitution in x1:
#
#   x1 = A \\ (b1 − B (D \\ (b2 − C x1)))
#   x2 = D \\ (b2 − C x1)
#
# i.e. (A − B D⁻¹ C) x1 = b1 − B D⁻¹ b2, then recover x2.
# =============================================================================

"""
    BlockULV

Mixed-BC 2×2 `[A B; C D]` factored like HODLR LU: HSS on all four tiles,
ULV of `A` and of `S = D - C (A \\ B)`. `B` and `C` stay rectangular HSS.
"""
struct BlockULV{TF}
    idx::BCIndexSets
    F::TF
    κ2::Float64
end

Base.size(F::BlockULV) = (F.idx.nt, F.idx.nt)
Base.size(F::BlockULV, d::Integer) =
    d == 1 || d == 2 ? F.idx.nt : throw(ArgumentError("invalid dimension $d"))

function Base.show(io::IO, ::MIME"text/plain", F::BlockULV)
    nu_, nq_ = nu(F.idx), nq(F.idx)
    print(io, "BlockULV ", F.F, "  (nu=$nu_, nq=$nq_)")
end

_hss_shift(p::SVector{2, T}, δ) where {T} = SVector{2, T}(p[1] + δ, p[2])
_hss_shift(p::SVector{3, T}, δ) where {T} = SVector{3, T}(p[1] + δ, p[2], p[3])
_hss_shift(p, δ) = p  # fallback: no shift

function _hss_block_tree(A::AbstractMatrix, rpts, cpts=rpts; rtol, nmax, method)
    rt = ClusterTree(collect(rpts), PrincipalComponentSplitter(; nmax=Int(nmax)))
    if cpts === rpts || rpts === cpts
        return HMatrices.assemble_hss(A, rt; rtol=float(rtol), method=method)
    end
    ct = ClusterTree(collect(cpts), PrincipalComponentSplitter(; nmax=Int(nmax)))
    return HMatrices.assemble_hss(A, rt, ct; rtol=float(rtol), method=method)
end

"""
    factor_block_ulv(A::BlockMixedOperator, pts; rtol=1e-8, nmax=32)

HODLR LU of the mixed 2×2 with ULV on HSS diagonals.
"""
function factor_block_ulv(A::BlockMixedOperator, pts; kwargs...)
    return factor_block_ulv(A.blocks, pts; κ2=A.κ2, kwargs...)
end

function factor_block_ulv(blocks::HGBlocks, pts;
        κ2::Real=0.0, rtol=1e-8, nmax::Integer=32)
    κ2 = float(κ2)
    idx = blocks.idx
    nq_ = nq(idx)
    pts_u = [pts[i] for i in idx.u]
    A11 = _as_matrix(blocks.Huu)
    if blocks.has_M && κ2 != 0
        A11 .+= κ2 .* _as_matrix(blocks.Muu)
    end
    if nq_ == 0
        tree_u = ClusterTree(pts_u, PrincipalComponentSplitter(; nmax=Int(nmax)))
        HA = HMatrices.assemble_hss(A11, tree_u; rtol=float(rtol), method=:id,
            Tmax=Inf, symm=:n)
        return BlockULV(idx, HMatrices.ulv(HA), κ2)
    end
    pts_q = [pts[i] for i in idx.q]
    A12 = .-_as_matrix(blocks.Guq)
    A21 = _as_matrix(blocks.Hqu)
    A22 = .-_as_matrix(blocks.Gqq)
    if blocks.has_M && κ2 != 0
        A21 .+= κ2 .* _as_matrix(blocks.Mqu)
    end
    F = HMatrices.hodlr_ulv_2x2(A11, A12, A21, A22, pts_u, pts_q;
        rtol=float(rtol), nmax=nmax)
    return BlockULV(idx, F, κ2)
end

function ldiv_block_ulv!(x::AbstractVector, F::BlockULV, b::AbstractVector)
    length(x) == F.idx.nt && length(b) == F.idx.nt || throw(DimensionMismatch())
    copyto!(x, F.F \ Vector{eltype(b)}(b))
    return x
end

function ldiv_block_ulv(F::BlockULV, b::AbstractVector)
    x = similar(b)
    return ldiv_block_ulv!(x, F, b)
end

Base.:\(F::BlockULV, b::AbstractVector) = ldiv_block_ulv(F, b)

function LinearAlgebra.ldiv!(x::AbstractVector, F::BlockULV, b::AbstractVector)
    return ldiv_block_ulv!(x, F, b)
end

"""
    assemble_hss_BDC(F::BlockULV, pts; nmax=32, kwargs...) -> HSSMatrix

Square HSS of `B * (D \\ C)` on the Neumann/internal cluster (`F.B`, `F.F22`, `F.C`).
"""


function LinearAlgebra.ldiv!(x::AbstractVector, F::BlockHLU, b::AbstractVector)
    return ldiv_block_hlu!(x, F, b)
end

"""
    scatter_block_sol!(dad, x, idx) -> (T, q)

Unpack packed unknown `x = [T_u; q_q]` into full `T` (`nt`) and `q` (`n`),
inserting known BC values from `dad.BV`.
"""
function scatter_block_sol!(dad::BEMdata, x::AbstractVector, idx::BCIndexSets)
    nu_ = nu(idx); nq_ = nq(idx)
    length(x) == nu_ + nq_ || throw(DimensionMismatch("x length $(length(x)) ≠ nt=$(idx.nt)"))
    T = zeros(eltype(x), idx.nt)
    q = zeros(eltype(x), idx.n)
    @inbounds for (k, j) in enumerate(idx.u)
        T[j] = x[k]
    end
    @inbounds for (k, j) in enumerate(idx.q)
        q[j] = x[nu_ + k]
        T[j] = dad.BV[j]                 # known Dirichlet potential
    end
    @inbounds for j in 1:idx.n
        if dad.BC[j] != 0
            q[j] = dad.BV[j]             # known Neumann flux
        end
    end
    return T, q
end

"""Write diagonal of an H-matrix using `fdiag(global_index) -> value`."""
function _set_diagonal!(fdiag, Hmat::HMatrix)
    piv = HMatrices.pivot(Hmat)
    for block in HMatrices.nodes(Hmat)
        HMatrices.hasdata(block) || continue
        HMatrices.isadmissible(block) && continue
        data = HMatrices.data(block)
        data isa Matrix || continue

        irange = HMatrices.rowrange(block) .- piv[1] .+ 1
        jrange = HMatrices.colrange(block) .- piv[2] .+ 1
        irangeg = HMatrices.rowperm(Hmat)[irange]
        jrangeg = HMatrices.colperm(Hmat)[jrange]
        for (iloc, ig) in enumerate(irangeg)
            ig > size(Hmat, 2) && continue
            for (jloc, jg) in enumerate(jrangeg)
                if ig == jg && jloc <= size(data, 2) && iloc <= size(data, 1)
                    data[iloc, jloc] = fdiag(ig)
                end
            end
        end
    end
    return nothing
end
