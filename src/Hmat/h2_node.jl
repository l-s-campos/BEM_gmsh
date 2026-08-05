# Recursive H² block tree (H2Lib-style sons)
#
# Flat [`H2Matrix`](@ref) stores generators + interaction lists (H2Pack layout).
# [`H2Node`](@ref) repackages the same data as a recursive block tree:
#
#   node = dense F  |  uniform coupling S  |  matrix of son nodes
#
# matching H2Lib `struct _h2matrix { u | f | son[rsons*csons] }`.

"""
    H2BlockKind

Leaf/node kind in a recursive H² tree.
- `H2DenseLeaf` — inadmissible nearfield (`F`)
- `H2UniformLeaf` — admissible far block (`S` with nested bases)
- `H2Split` — subdivided into `sons`
"""
@enum H2BlockKind begin
    H2DenseLeaf
    H2UniformLeaf
    H2Split
end

"""
    mutable struct H2Node{R,T}

Recursive H² block (H2Lib `h2matrix` analogue).

# Fields
- `row_id`, `col_id`: cluster indices into the source [`H2TreeIndex`](@ref)
- `kind`: [`H2BlockKind`](@ref)
- `S`: coupling for uniform leaves (`Matrix` or [`RkMatrix`](@ref)); else `nothing`
- `F`: dense nearfield; else `nothing`
- `sons`: `rsons × csons` child blocks when `kind === H2Split`
- `pack`: shared [`H2Pack`](@ref) with nested bases and original flat data
"""
mutable struct H2Node{R, T}
    row_id::Int
    col_id::Int
    kind::H2BlockKind
    S::Any                    # Nothing | Matrix{T} | RkMatrix{T}
    F::Union{Nothing, Matrix{T}}
    sons::Matrix{H2Node{R, T}}
    pack::Any                 # H2Pack{R,T} (set after construct)
end

"""
    struct H2Pack{R,T}

Shared payload for a recursive H² tree: flat generators, interaction maps,
and a pointer to the source [`H2Matrix`](@ref).
"""
struct H2Pack{R, T}
    src::H2Matrix{R, T}
    tidx::H2TreeIndex{R}
    U::Vector{Matrix{T}}
    Bfar::Dict{Tuple{Int, Int}, Any}
    Ddiag::Dict{Int, Matrix{T}}
    Dnear::Dict{Tuple{Int, Int}, Matrix{T}}
    n::Int
    rowperm::Vector{Int}
    colperm::Vector{Int}
end

function H2Pack(H::H2Matrix{R, T}) where {R, T}
    return H2Pack{R, T}(
        H,
        H.tidx,
        H.U,
        H.Bfar,
        H.Ddiag,
        H.Dnear,
        H.n,
        H.rowperm,
        H.colperm,
    )
end

# ---- predicates / sizes -------------------------------------------------------

isleaf_h2(N::H2Node) = N.kind !== H2Split
isuniform(N::H2Node) = N.kind === H2UniformLeaf
isdense_h2(N::H2Node) = N.kind === H2DenseLeaf
issplit(N::H2Node) = N.kind === H2Split

function rowcluster(N::H2Node)
    return N.pack.tidx.nodes[N.row_id]
end
function colcluster(N::H2Node)
    return N.pack.tidx.nodes[N.col_id]
end

function rowrange(N::H2Node)
    return index_range(rowcluster(N))
end
function colrange(N::H2Node)
    return index_range(colcluster(N))
end

Base.size(N::H2Node) = (length(rowrange(N)), length(colrange(N)))
Base.size(N::H2Node, d::Integer) = d == 1 ? size(N)[1] : d == 2 ? size(N)[2] : 1
Base.eltype(::H2Node{<:Any, T}) where {T} = T
Base.length(N::H2Node) = prod(size(N))

function Base.show(io::IO, N::H2Node)
    m, n = size(N)
    return print(io, "H2Node{$(eltype(N))} $(m)×$(n) $(N.kind) ",
        "(row=$(N.row_id), col=$(N.col_id))")
end
Base.show(io::IO, ::MIME"text/plain", N::H2Node) = show(io, N)

function nsons(N::H2Node)
    issplit(N) || return (0, 0)
    return size(N.sons)
end

"""Count nodes in the recursive block tree."""
function h2_nnodes(N::H2Node)
    issplit(N) || return 1
    return 1 + sum(h2_nnodes, N.sons; init = 0)
end

function h2_nleaves(N::H2Node)
    issplit(N) || return 1
    return sum(h2_nleaves, N.sons; init = 0)
end

# ---- constructors -------------------------------------------------------------

function _h2node_dense(ri::Int, ci::Int, F::Matrix{T}, pack::H2Pack{R, T}) where {R, T}
    Z = Matrix{H2Node{R, T}}(undef, 0, 0)
    return H2Node{R, T}(ri, ci, H2DenseLeaf, nothing, F, Z, pack)
end

function _h2node_uniform(ri::Int, ci::Int, S, pack::H2Pack{R, T}) where {R, T}
    Z = Matrix{H2Node{R, T}}(undef, 0, 0)
    return H2Node{R, T}(ri, ci, H2UniformLeaf, S, nothing, Z, pack)
end

function _h2node_split(ri::Int, ci::Int, sons::Matrix{H2Node{R, T}}, pack::H2Pack{R, T}) where {R, T}
    return H2Node{R, T}(ri, ci, H2Split, nothing, nothing, sons, pack)
end

# ---- repackage ----------------------------------------------------------------

"""
    h2_repackage(H::H2Matrix) -> H2Node

Build a recursive H2Lib-style block tree from a flat [`H2Matrix`](@ref).

| Flat storage | Recursive node |
|--------------|----------------|
| `Bfar[(i,j)]` | uniform leaf at `(i,j)` (transpose if only `(j,i)` stored) |
| `Dnear` / `Ddiag` | dense leaf |
| refined pairs | `sons` via the same split rule as H2Lib / `_h2_intersect!` |

Nested bases stay in `node.pack.U` (shared). No copy of coupling/near data
beyond shallow references / adjoints when the opposite triangle is needed.
"""
function h2_repackage(H::H2Matrix{R, T}) where {R, T}
    pack = H2Pack(H)
    return _h2_build_node(pack, 1, 1)
end

function _h2_build_node(pack::H2Pack{R, T}, ri::Int, ci::Int) where {R, T}
    tidx = pack.tidx

    # --- uniform (admissible) leaf ---
    if haskey(pack.Bfar, (ri, ci))
        return _h2node_uniform(ri, ci, pack.Bfar[(ri, ci)], pack)
    elseif haskey(pack.Bfar, (ci, ri))
        return _h2node_uniform(ri, ci, _h2_adjoint_block(pack.Bfar[(ci, ri)]), pack)
    end

    # --- dense near / diagonal leaf ---
    rch = tidx.children[ri]
    cch = tidx.children[ci]
    if ri == ci && isempty(rch)
        haskey(pack.Ddiag, ri) || throw(ErrorException(
            "h2_repackage: missing Ddiag for leaf cluster $ri"))
        return _h2node_dense(ri, ci, pack.Ddiag[ri], pack)
    end
    if haskey(pack.Dnear, (ri, ci))
        return _h2node_dense(ri, ci, pack.Dnear[(ri, ci)], pack)
    elseif haskey(pack.Dnear, (ci, ri))
        return _h2node_dense(ri, ci, collect(adjoint(pack.Dnear[(ci, ri)])), pack)
    end

    # --- subdivide (H2Lib block-tree sons rule) ---
    if isempty(rch) && isempty(cch)
        # Partition incomplete: fall back to dense block via source matvec
        return _h2node_dense(ri, ci, _h2_extract_dense(pack, ri, ci), pack)
    end

    rsons = isempty(rch) ? Int[ri] : rch
    csons = isempty(cch) ? Int[ci] : cch
    nr, nc = length(rsons), length(csons)
    sons = Matrix{H2Node{R, T}}(undef, nr, nc)
    @inbounds for j in 1:nc, i in 1:nr
        sons[i, j] = _h2_build_node(pack, rsons[i], csons[j])
    end
    return _h2node_split(ri, ci, sons, pack)
end

function _h2_adjoint_block(S)
    if S isa RkMatrix
        # A*B' adjoint is B*A'
        return RkMatrix(copy(S.B), copy(S.A))
    else
        return collect(adjoint(S))
    end
end

function _h2_extract_dense(pack::H2Pack{R, T}, ri::Int, ci::Int) where {R, T}
    Ir = index_range(pack.tidx.nodes[ri])
    Jr = index_range(pack.tidx.nodes[ci])
    n = pack.n
    X = zeros(T, n, length(Jr))
    @inbounds for (k, j) in enumerate(Jr)
        X[j, k] = one(T)
    end
    Y = zeros(T, n, length(Jr))
    mul!(Y, pack.src, X, one(T), zero(T); global_index = false)
    return Matrix{T}(view(Y, Ir, :))
end

# ---- nested basis expansion ---------------------------------------------------

"""
    h2_basis_matrix(pack, cluster_id) -> Matrix

Full cluster basis ``V_t`` on `index_range(t)` by expanding nested `U`
(leaf generators + non-leaf transfers), matching H2Lib nested `V_t = V_{sons} E`.
"""
function h2_basis_matrix(pack::H2Pack{R, T}, id::Int) where {R, T}
    tidx = pack.tidx
    U = pack.U
    ch = tidx.children[id]
    if isempty(ch)
        return copy(U[id])
    end
    # V = blkdiag(V_c) * U[id]
    Vch = [h2_basis_matrix(pack, c) for c in ch]
    heights = [size(V, 1) for V in Vch]
    ranks = [size(V, 2) for V in Vch]
    m = sum(heights)
    rpar = size(U[id], 2)
    size(U[id], 1) == sum(ranks) || return copy(U[id])  # inconsistent → raw
    V = zeros(T, m, rpar)
    roff = 0
    coff = 0  # row offset into U[id]
    for (Vci, hi, ri) in zip(Vch, heights, ranks)
        ri == 0 && continue
        Eblock = U[id][(coff + 1):(coff + ri), :]
        V[(roff + 1):(roff + hi), :] = Vci * Eblock
        roff += hi
        coff += ri
    end
    return V
end

function h2_basis_matrix(N::H2Node, id::Int = N.row_id)
    return h2_basis_matrix(N.pack, id)
end

# ---- apply uniform leaf: y[I] += V_r * S * V_c' * x[J] ------------------------

function _h2_apply_uniform!(
        y::AbstractVector{T},
        N::H2Node{<:Any, T},
        x::AbstractVector{T},
    ) where {T}
    Ir = rowrange(N)
    Jr = colrange(N)
    Vr = h2_basis_matrix(N.pack, N.row_id)
    Vc = h2_basis_matrix(N.pack, N.col_id)
    # handle depth-mismatch couplings stored against full index sets
    S = N.S
    xr = view(x, Jr)
    yr = view(y, Ir)

    if S isa RkMatrix
        # S = A*B' on skeleton spaces — still multiply as dense small op via Matrix
        Sm = Matrix(S)
    else
        Sm = S
    end

    # Cases from flat H²: same-depth skeletons, or mixed full/skeleton
    if size(Sm, 1) == size(Vr, 2) && size(Sm, 2) == size(Vc, 2)
        yr .+= Vr * (Sm * (Vc' * xr))
    elseif size(Sm, 1) == size(Vr, 2) && size(Sm, 2) == length(Jr)
        yr .+= Vr * (Sm * xr)
    elseif size(Sm, 1) == length(Ir) && size(Sm, 2) == size(Vc, 2)
        yr .+= Sm * (Vc' * xr)
    elseif size(Sm, 1) == length(Ir) && size(Sm, 2) == length(Jr)
        yr .+= Sm * xr
    else
        # last resort: densify via source
        Fd = _h2_extract_dense(N.pack, N.row_id, N.col_id)
        yr .+= Fd * xr
    end
    return y
end

# ---- recursive matvec ---------------------------------------------------------

"""
    mul!(y, N::H2Node, x; global_index=true)

Apply recursive H² tree (for verification against flat [`H2Matrix`](@ref)).
"""
function LinearAlgebra.mul!(
        y::AbstractVector,
        N::H2Node,
        x::AbstractVector,
        a::Number = 1,
        b::Number = 0;
        global_index = use_global_index(),
    )
    pack = N.pack
    T = eltype(N)
    if global_index
        xloc = x[pack.colperm]
        # scale into work vector
        xw = a == 1 ? collect(T, xloc) : collect(T, a .* xloc)
        yw = zeros(T, pack.n)
        _h2node_matvec!(yw, N, xw)
        yloc = iszero(b) ? yw : b .* y[pack.rowperm] .+ yw
        y[pack.rowperm] .= yloc
    else
        xw = a == 1 ? collect(T, x) : collect(T, a .* x)
        if iszero(b)
            fill!(y, zero(T))
        else
            rmul!(y, b)
        end
        _h2node_matvec!(y, N, xw)
    end
    return y
end

function _h2node_matvec!(y::AbstractVector{T}, N::H2Node{<:Any, T}, x::AbstractVector{T}) where {T}
    if isdense_h2(N)
        mul!(view(y, rowrange(N)), N.F, view(x, colrange(N)), true, true)
    elseif isuniform(N)
        _h2_apply_uniform!(y, N, x)
    else
        for s in N.sons
            _h2node_matvec!(y, s, x)
        end
    end
    return y
end

function Base.:*(N::H2Node, x::AbstractVector)
    y = zeros(eltype(N), size(N, 1))
    return mul!(y, N, x)
end

"""Materialize recursive H² to dense (small n / tests)."""
function Base.Matrix(N::H2Node{R, T}; global_index = true) where {R, T}
    n = N.pack.n
    M = zeros(T, n, n)
    ej = zeros(T, n)
    @inbounds for j in 1:n
        fill!(ej, 0)
        ej[j] = 1
        mul!(view(M, :, j), N, ej; global_index = false)
    end
    global_index || return M
    p = N.pack.rowperm
    return Matrix(PermutedMatrix(M, invperm(p), invperm(p)))
end

# ---- walk helpers -------------------------------------------------------------

"""Depth-first visitor `f(node)` over the recursive tree."""
function h2_foreach(f, N::H2Node)
    f(N)
    if issplit(N)
        for s in N.sons
            h2_foreach(f, s)
        end
    end
    return nothing
end

function h2_leaves(N::H2Node)
    out = H2Node[]
    h2_foreach(N) do node
        isleaf_h2(node) && push!(out, node)
    end
    return out
end
