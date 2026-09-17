# Recursive H² block tree (H2Lib-style sons)
#
# [`H2Node`](@ref) is a recursive block tree:

#
#   node = dense F  |  uniform coupling S  |  matrix of son nodes
#
# matching H2Lib `struct _h2matrix { u | f | son[rsons*csons] }`.

"""Flat view of a [`ClusterTree`](@ref) for H² node indexing (1-based)."""
struct H2TreeIndex{R}
    nodes::Vector{R}
    parent::Vector{Int}
    children::Vector{Vector{Int}}
    levels::Vector{Vector{Int}}
    leafnodes::Vector{Int}
    depth_of::Vector{Int}
end

function H2TreeIndex(root::R) where {R}
    nodes = R[]
    parent_ids = Int[]
    child_ids = Vector{Int}[]
    function add!(n, p)
        push!(nodes, n)
        push!(parent_ids, p)
        push!(child_ids, Int[])
        id = length(nodes)
        if p > 0
            push!(child_ids[p], id)
        end
        for c in children(n)
            add!(c, id)
        end
        return id
    end
    add!(root, 0)
    nnode = length(nodes)
    depth_of = zeros(Int, nnode)
    for i in 2:nnode
        depth_of[i] = depth_of[parent_ids[i]] + 1
    end
    maxd = maximum(depth_of; init = 0)
    levels = [Int[] for _ in 0:maxd]
    for i in 1:nnode
        push!(levels[depth_of[i] + 1], i)
    end
    leafnodes = [i for i in 1:nnode if isempty(child_ids[i])]
    return H2TreeIndex{R}(nodes, parent_ids, child_ids, levels, leafnodes, depth_of)
end

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
- `pack`: shared [`H2Pack`](@ref) with nested bases and couplings
"""
mutable struct H2Node{R, T}
    row_id::Int
    col_id::Int
    kind::H2BlockKind
    S::Any                    # Nothing | Matrix{T} | RkMatrix{T}
    F::Union{Nothing, Matrix{T}}
    sons::Matrix{H2Node{R, T}}
    pack::Any                 # H2Pack{R,T} (set after construct)
    """If true, `S` is a full-block operator on `rowrange×colrange` (e.g. after rkupdate).
    If false, `S` is a nested/skeleton coupling used with `pack.U` bases."""
    s_full::Bool
end

"""
    struct H2Pack{R,T}

Shared payload for a recursive H² tree: nested bases `U`, far couplings
`Bfar`, near/diag dense blocks, and an optional `src` for fallback matvecs.
"""
struct H2Pack{R, T}
    src::Any
    tidx::H2TreeIndex{R}
    U::Vector{Matrix{T}}
    Bfar::Dict{Tuple{Int, Int}, Any}
    Ddiag::Dict{Int, Matrix{T}}
    Dnear::Dict{Tuple{Int, Int}, Matrix{T}}
    n::Int
    rowperm::Vector{Int}
    colperm::Vector{Int}
    Vcache::Vector{Union{Nothing, Matrix{T}}}
    Gcache::Vector{Union{Nothing, Matrix{T}}}
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
    return H2Node{R, T}(ri, ci, H2DenseLeaf, nothing, F, Z, pack, false)
end

function _h2node_uniform(ri::Int, ci::Int, S, pack::H2Pack{R, T}; s_full::Bool = false) where {R, T}
    Z = Matrix{H2Node{R, T}}(undef, 0, 0)
    return H2Node{R, T}(ri, ci, H2UniformLeaf, S, nothing, Z, pack, s_full)
end

function _h2node_split(ri::Int, ci::Int, sons::Matrix{H2Node{R, T}}, pack::H2Pack{R, T}) where {R, T}
    return H2Node{R, T}(ri, ci, H2Split, nothing, nothing, sons, pack, false)
end

# ---- repackage ----------------------------------------------------------------

"""
    h2_repackage(pack::H2Pack) -> H2Node

Build a recursive H2Lib-style block tree from packed nested generators.

| Flat storage | Recursive node |
|--------------|----------------|
| `Bfar[(i,j)]` | uniform leaf at `(i,j)` (transpose if only `(j,i)` stored) |
| `Dnear` / `Ddiag` | dense leaf |
| refined pairs | `sons` via the same split rule as H2Lib / `_h2_intersect!` |

Nested bases stay in `node.pack.U` (shared). No copy of coupling/near data
beyond shallow references / adjoints when the opposite triangle is needed.
"""
function h2_repackage(pack::H2Pack)
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

    # H2Lib: a block is a leaf if either cluster is a leaf (no hanging 1×k splits).
    if isempty(rch) || isempty(cch)
        return _h2node_dense(ri, ci, _h2_collect_block(pack, ri, ci), pack)
    end

    rsons = sort(rch; by = id -> first(index_range(tidx.nodes[id])))
    csons = sort(cch; by = id -> first(index_range(tidx.nodes[id])))
    nr, nc = length(rsons), length(csons)
    sons = Matrix{H2Node{R, T}}(undef, nr, nc)
    @inbounds for j in 1:nc, i in 1:nr
        sons[i, j] = _h2_build_node(pack, rsons[i], csons[j])
    end
    return _h2node_split(ri, ci, sons, pack)
end

"""Assemble a dense |ri|×|ci| block from stored couplings / children / src."""
function _h2_collect_block(pack::H2Pack{R, T}, ri::Int, ci::Int) where {R, T}
    tidx = pack.tidx
    if haskey(pack.Bfar, (ri, ci))
        return _h2_uniform_from_S(pack, ri, ci, pack.Bfar[(ri, ci)])
    elseif haskey(pack.Bfar, (ci, ri))
        return collect(adjoint(_h2_uniform_from_S(pack, ci, ri, pack.Bfar[(ci, ri)])))
    end
    if ri == ci && haskey(pack.Ddiag, ri)
        return copy(pack.Ddiag[ri])
    end
    if haskey(pack.Dnear, (ri, ci))
        return copy(pack.Dnear[(ri, ci)])
    elseif haskey(pack.Dnear, (ci, ri))
        return collect(adjoint(pack.Dnear[(ci, ri)]))
    end
    rch = tidx.children[ri]
    cch = tidx.children[ci]
    Ir = index_range(tidx.nodes[ri])
    Jr = index_range(tidx.nodes[ci])
    if isempty(rch) && isempty(cch)
        return _h2_extract_dense(pack, ri, ci)
    end
    M = zeros(T, length(Ir), length(Jr))
    i0, j0 = first(Ir), first(Jr)
    rlist = isempty(rch) ? Int[ri] : rch
    clist = isempty(cch) ? Int[ci] : cch
    for r in rlist, c in clist
        B = _h2_collect_block(pack, r, c)
        ir = index_range(tidx.nodes[r])
        jr = index_range(tidx.nodes[c])
        rows = (first(ir) - i0 + 1):(last(ir) - i0 + 1)
        cols = (first(jr) - j0 + 1):(last(jr) - j0 + 1)
        size(B) == (length(rows), length(cols)) || continue
        M[rows, cols] .= B
    end
    return M
end

function _h2_uniform_from_S(pack::H2Pack{R, T}, ri::Int, ci::Int, S) where {R, T}
    Vr = h2_basis_matrix(pack, ri)
    Vc = h2_basis_matrix(pack, ci)
    Sm = S isa RkMatrix ? Matrix(S) : Matrix{T}(S)
    m = length(index_range(pack.tidx.nodes[ri]))
    n = length(index_range(pack.tidx.nodes[ci]))
    if size(Sm, 1) == size(Vr, 2) && size(Sm, 2) == size(Vc, 2) &&
            size(Vr, 1) == m && size(Vc, 1) == n
        return Vr * (Sm * adjoint(Vc))
    elseif size(Sm, 1) == m && size(Sm, 2) == n
        return Sm
    end
    return _h2_extract_dense(pack, ri, ci)
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
    src = pack.src
    Yblk = zeros(T, length(Ir), length(Jr))
    x = zeros(T, n)
    y = zeros(T, n)
    @inbounds for (k, j) in enumerate(Jr)
        fill!(x, zero(T))
        x[j] = one(T)
        fill!(y, zero(T))
        mul!(y, src, x; global_index = false)
        Yblk[:, k] .= view(y, Ir)
    end
    return Yblk
end

# ---- nested basis expansion ---------------------------------------------------

"""
    h2_basis_matrix(pack, cluster_id) -> Matrix

Full cluster basis ``V_t`` on `index_range(t)` by expanding nested `U`
(leaf generators + non-leaf transfers), matching H2Lib nested `V_t = V_{sons} E`.
"""
function h2_clear_basis_cache!(pack::H2Pack)
    fill!(pack.Vcache, nothing)
    fill!(pack.Gcache, nothing)
    return pack
end

function h2_basis_matrix(pack::H2Pack{R, T}, id::Int) where {R, T}
    if 1 <= id <= length(pack.Vcache)
        got = pack.Vcache[id]
        got !== nothing && return got
    end
    tidx = pack.tidx
    U = pack.U
    ch = tidx.children[id]
    V = if isempty(ch)
        copy(U[id])
    else
        Vch = [h2_basis_matrix(pack, c) for c in ch]
        heights = [size(Vc, 1) for Vc in Vch]
        ranks = [size(Vc, 2) for Vc in Vch]
        m = sum(heights)
        rpar = size(U[id], 2)
        if size(U[id], 1) != sum(ranks)
            copy(U[id])
        else
            V = zeros(T, m, rpar)
            roff = 0
            coff = 0
            for (Vci, hi, ri) in zip(Vch, heights, ranks)
                ri == 0 && continue
                Eblock = U[id][(coff + 1):(coff + ri), :]
                V[(roff + 1):(roff + hi), :] = Vci * Eblock
                roff += hi
                coff += ri
            end
            V
        end
    end
    if 1 <= id <= length(pack.Vcache)
        pack.Vcache[id] = V
    end
    return V
end

"""Cluster Gram ``V_t' V_t`` (``k×k``), nested-cached."""
function h2_gram_matrix(pack::H2Pack{R, T}, id::Int) where {R, T}
    if 1 <= id <= length(pack.Gcache)
        got = pack.Gcache[id]
        got !== nothing && return got
    end
    tidx = pack.tidx
    U = pack.U
    ch = tidx.children[id]
    G = if isempty(ch)
        u = U[id]
        isempty(u) ? zeros(T, 0, 0) : adjoint(u) * u
    else
        Gs = [h2_gram_matrix(pack, c) for c in ch]
        rpar = size(U[id], 2)
        if rpar == 0
            zeros(T, 0, 0)
        else
        # G = E' blkdiag(G_c) E with E = U[id]
        tmp = zeros(T, size(U[id], 1), rpar)
        coff = 0
        for Gc in Gs
            ri = size(Gc, 1)
            ri == 0 && continue
            tmp[(coff + 1):(coff + ri), :] = Gc * U[id][(coff + 1):(coff + ri), :]
            coff += ri
        end
        adjoint(U[id]) * tmp
        end
    end
    if 1 <= id <= length(pack.Gcache)
        pack.Gcache[id] = G
    end
    return G
end

function h2_basis_matrix(N::H2Node, id::Int = N.row_id)
    return h2_basis_matrix(N.pack, id)
end

# ---- apply uniform leaf: y[I] += V_r * S * V_c' * x[J] ------------------------

function _h2_apply_uniform!(
        y::AbstractVector{T},
        N::H2Node{<:Any, T},
        x::AbstractVector{T},
        α::Number = one(T),
    ) where {T}
    Ir = rowrange(N)
    Jr = colrange(N)
    S = N.S
    xr = view(x, Jr)
    yr = view(y, Ir)

    # Full-block operator (rkupdate recompression): yI += α S * xJ  (no nested V)
    if N.s_full
        mul!(yr, S, xr, α, true)
        return y
    end

    Vr = h2_basis_matrix(N.pack, N.row_id)
    Vc = h2_basis_matrix(N.pack, N.col_id)
    # Nested/skeleton coupling (default from flat H²)
    if S isa RkMatrix && size(S, 1) == size(Vr, 2) && size(S, 2) == size(Vc, 2)
        tmp = Vc' * xr
        mid = S * tmp
        mul!(yr, Vr, mid, α, true)
        return y
    end
    Sm = S isa RkMatrix ? Matrix(S) : S

    if size(Sm, 1) == size(Vr, 2) && size(Sm, 2) == size(Vc, 2)
        mul!(yr, Vr, Sm * (Vc' * xr), α, true)
    elseif size(Sm, 1) == size(Vr, 2) && size(Sm, 2) == length(Jr)
        mul!(yr, Vr, Sm * xr, α, true)
    elseif size(Sm, 1) == length(Ir) && size(Sm, 2) == size(Vc, 2)
        mul!(yr, Sm, Vc' * xr, α, true)
    elseif size(Sm, 1) == length(Ir) && size(Sm, 2) == length(Jr)
        mul!(yr, Sm, xr, α, true)
    else
        throw(DimensionMismatch(
            "uniform H2 apply: S $(size(Sm)) vs Vr $(size(Vr)) Vc $(size(Vc)) " *
            "block $(length(Ir))×$(length(Jr)) (row=$(N.row_id), col=$(N.col_id))",
        ))
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

# ---- NNCA → nested H² (keep couplings, do not expand to H-matrix) ------------

"""
    h2node(A::NNCAMatrix) -> H2Node

Repackage square scalar NNCA as a recursive H2Lib block tree: nested `L2P`
stays nested (`U`), M2L stays `k×k` couplings, near/self stay dense.
"""
function h2node(A::NNCAMatrix{T}) where {T}
    A.n == A.n_col || throw(ArgumentError("h2node needs square NNCA; got $(size(A))"))
    A.p == 1 || throw(ArgumentError("h2node is implemented for scalar NNCA (p=1)"))
    pack = _nnca_h2pack(A)
    return h2_repackage(pack)
end

function _nnca_h2pack(A::NNCAMatrix{T}) where {T}
    tidx = H2TreeIndex(A.tree)
    nnode = length(tidx.nodes)
    idmap = Dict{Int, Int}()
    for i in 1:nnode
        nid = node_id(tidx.nodes[i])
        nid == 0 && (nid = i)
        idmap[nid] = i
    end
    U = Vector{Matrix{T}}(undef, nnode)
    Ddiag = Dict{Int, Matrix{T}}()
    Dnear = Dict{Tuple{Int, Int}, Matrix{T}}()
    Bfar = Dict{Tuple{Int, Int}, Any}()
    for i in 1:nnode
        nid = node_id(tidx.nodes[i])
        nid == 0 && (nid = i)
        b = A.boxes[nid]
        U[i] = isempty(b.L2P) ? zeros(T, 0, 0) : copy(b.L2P)
        if !isempty(b.self)
            Ddiag[i] = copy(b.self)
        end
    end
    for nid in 1:length(A.boxes)
        haskey(idmap, nid) || continue
        i = idmap[nid]
        a = A.m2l_dsptr[nid]
        stop = A.m2l_dsptr[nid + 1]
        @inbounds while a < stop
            cj = A.m2l_src[a]
            if haskey(idmap, cj)
                m = A.m2l_m[a]
                n = A.m2l_n[a]
                p0 = A.m2l_ptr[a]
                Bfar[(i, idmap[cj])] = Matrix{T}(reshape(A.m2l_data[p0:(p0 + m * n - 1)], m, n))
            end
            a += 1
        end
        a = A.near_dsptr[nid]
        stop = A.near_dsptr[nid + 1]
        @inbounds while a < stop
            cj = A.near_src[a]
            if haskey(idmap, cj)
                m = A.near_m[a]
                n = A.near_n[a]
                p0 = A.near_ptr[a]
                Dnear[(i, idmap[cj])] = Matrix{T}(reshape(A.near_data[p0:(p0 + m * n - 1)], m, n))
            end
            a += 1
        end
    end
    nn = length(tidx.nodes)
    return H2Pack{typeof(A.tree), T}(
        A, tidx, U, Bfar, Ddiag, Dnear, A.n, copy(A.perm), copy(A.colperm_pts),
        Vector{Union{Nothing, Matrix{T}}}(nothing, nn),
        Vector{Union{Nothing, Matrix{T}}}(nothing, nn),
    )
end
