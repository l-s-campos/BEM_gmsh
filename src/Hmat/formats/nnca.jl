# Gujjula–Ambikasaran NNCA H² (arXiv:2203.14832 / SAFRAN-LAB/NNCA).
# Geometric quad/oct tree, FMM interaction list, full-IL ACA, nested L2P,
# dense M2L on skeletons, dense near field. Apply is M2M / M2L / L2L / near.

mutable struct NNCABox{T}
    check::Vector{Int}
    charge::Vector{Int}
    L2P::Matrix{T}
    self::Matrix{T}
    outgoing::Vector{T}
    incoming::Vector{T}
    potential::Vector{T}
end

function NNCABox{T}() where {T}
    return NNCABox{T}(Int[], Int[], zeros(T, 0, 0), zeros(T, 0, 0), T[], T[], T[])
end

"""
    NNCAMatrix{T} <: AbstractStructuredMatrix{T}

Square H² from [`assemble_h2`](@ref) / [`assemble_nnca`](@ref). Storage is
tree-local (leaf order of `tree`).

For a scalar kernel, `size` is `n×n` with `n` points. For a tensor kernel
(`SMatrix{p,p}`), interaction lists and skeletons stay on **points** (block
NNCA); `L2P` / `M2L` / near are `(p n)×(p r)` scalar matrices and `size` is
`(p n)×(p n)`.
"""
mutable struct NNCAMatrix{T} <: AbstractStructuredMatrix{T}
    tree::ClusterTree
    boxes::Vector{NNCABox{T}}
    id2node::Vector{Any}
    neighbors::Vector{Vector{Int}}
    il::Vector{Vector{Int}}
    levels::Vector{Vector{Int}}       # node_ids per depth, coarse → fine
    leaf_ids::Vector{Int}
    n::Int                            # row points
    p::Int                            # tensor block size (1 for scalar)
    perm::Vector{Int}                 # loc2glob (rows)
    invperm::Vector{Int}
    avg_rank::Float64
    # packed M2L (CSR by dest box; data is column-major)
    m2l_dsptr::Vector{Int}
    m2l_src::Vector{Int}
    m2l_ptr::Vector{Int}
    m2l_m::Vector{Int}
    m2l_n::Vector{Int}
    m2l_data::Vector{T}
    # packed near (CSR by dest leaf)
    near_dsptr::Vector{Int}
    near_src::Vector{Int}
    near_ptr::Vector{Int}
    near_m::Vector{Int}
    near_n::Vector{Int}
    near_data::Vector{T}
    n_col::Int
    coltree::Any
    boxes_col::Vector{NNCABox{T}}
    id2col::Vector{Any}
    levels_col::Vector{Vector{Int}}
    leaf_ids_col::Vector{Int}
    colperm_pts::Vector{Int}
    invcolperm::Vector{Int}
end

Base.size(A::NNCAMatrix) = (A.p * A.n, A.p * A.n_col)
Base.eltype(::NNCAMatrix{T}) where {T} = T
rowperm(A::NNCAMatrix) = A.p == 1 ? A.perm : _expand_perm(A.perm, A.p)
colperm(A::NNCAMatrix) = A.p == 1 ? A.colperm_pts : _expand_perm(A.colperm_pts, A.p)
_nnca_square(A::NNCAMatrix) = A.n == A.n_col && A.boxes_col === A.boxes

function maxrank(A::NNCAMatrix)
    r = 0
    p = A.p
    for b in A.boxes
        r = max(r, size(b.L2P, 2) ÷ p, length(b.check))
    end
    if A.boxes_col !== A.boxes
        for b in A.boxes_col
            r = max(r, size(b.L2P, 2) ÷ p, length(b.check))
        end
    end
    return r
end

function compression_ratio(A::NNCAMatrix)
    ns = Base.summarysize(A)
    return (length(A) * sizeof(eltype(A))) / ns
end

function Base.show(io::IO, A::NNCAMatrix)
    sz = size(A)
    blk = A.p == 1 ? "" : " block p=$(A.p)"
    shp = _nnca_square(A) ? "" : " rectangular"
    return print(io, "NNCAMatrix{", eltype(A), "} $(sz[1])×$(sz[2])", blk, shp, " ",
        "avg rank=$(round(A.avg_rank; digits=1)) maxrank=$(maxrank(A))")
end
Base.show(io::IO, ::MIME"text/plain", A::NNCAMatrix) = show(io, A)

"""
    assemble_h2([T,], K, tree; rtol=1e-6, rank=typemax(Int), global_index=true, device=:host)
    assemble_nnca(...)

NNCA H² on a geometric quad/oct [`ClusterTree`](@ref) (`DyadicSplitter` with
`tight=false`). `K` supports `getindex`. `method=:cheb` (needs a
[`KernelMatrix`](@ref)) uses tensor Chebyshev nested interpolation of order
`order` (default 6 in 2-D, 4 in 3-D) instead of ACA skeletons, then H2Lib-style weighted truncation of nested
bases and M2L to `rtol` (`rtol=0` skips recompress).

If `eltype(K)` is `SMatrix{p,p}`, this is **block NNCA**: FMM interaction lists
and ACA skeletons are on points; `L2P`/`M2L`/near are stored as scalar
`(p n)×(p r)` blocks. The result is a `(p n)×(p n)` matrix of the scalar
eltype.

A second tree `assemble_h2(K, rowtree, coltree)` builds a rectangular
`(p n_row)×(p n_col)` operator (dual-tree IL). Share a root
`container` so same-depth boxes have equal size.
"""
function assemble_h2(K::AbstractMatrix, tree::ClusterTree; kwargs...)
    return assemble_h2(eltype(K), K, tree; kwargs...)
end
function assemble_h2(K::AbstractMatrix, rowtree::ClusterTree, coltree::ClusterTree; kwargs...)
    rowtree === coltree && return assemble_h2(K, rowtree; kwargs...)
    return assemble_h2(eltype(K), K, rowtree, coltree; kwargs...)
end
const assemble_nnca = assemble_h2

function assemble_h2(K::AbstractMatrix, pts::AbstractVector; nmax::Int=32, kwargs...)
    tree = ClusterTree(pts, DyadicSplitter(; nmax=nmax, tight=false); cube=true)
    return assemble_h2(K, tree; kwargs...)
end

function assemble_h2(::Type{T}, K, tree::ClusterTree;
        rtol=1e-6, rank=typemax(Int), global_index=true, threads=true,
        device=:host, method::Symbol=:nnca, order::Int=0) where {T}
    if method === :cheb || method === :chebyshev
        is_tensor_eltype(T) && throw(ArgumentError(
            "Chebyshev H² is scalar only; got eltype $T"))
        A = _assemble_h2_cheb(T, K, tree; order = order, rtol = rtol, rank = rank,
            global_index = global_index, threads = threads)
        return gpu_wrap(A, device)
    end
    (method === :nnca || method === :aca) || throw(ArgumentError(
        "assemble_h2 method must be :nnca or :cheb; got $method"))
    A = if is_tensor_eltype(T)
        S = eltype(T)
        p, q = tensor_blocksize(T)
        p == q || throw(ArgumentError(
            "block NNCA requires square SMatrix{p,p}, got $T"))
        _assemble_h2(S, K, tree, p; rtol=rtol, rank=rank,
            global_index=global_index, threads=threads)
    else
        _assemble_h2(T, K, tree, 1; rtol=rtol, rank=rank,
            global_index=global_index, threads=threads)
    end
    return gpu_wrap(A, device)
end

function assemble_h2(::Type{T}, K, rowtree::ClusterTree, coltree::ClusterTree;
        rtol=1e-6, rank=typemax(Int), global_index=true, threads=true,
        device=:host) where {T}
    A = if is_tensor_eltype(T)
        S = eltype(T)
        p, q = tensor_blocksize(T)
        p == q || throw(ArgumentError(
            "block NNCA requires square SMatrix{p,p}, got $T"))
        _assemble_h2_rect(S, K, rowtree, coltree, p; rtol=rtol, rank=rank,
            global_index=global_index, threads=threads)
    else
        _assemble_h2_rect(T, K, rowtree, coltree, 1; rtol=rtol, rank=rank,
            global_index=global_index, threads=threads)
    end
    return gpu_wrap(A, device)
end

function _assemble_h2(::Type{T}, K, tree::ClusterTree, p::Int;
        rtol=1e-6, rank=typemax(Int), global_index=true, threads=true) where {T}
    node_id(tree) == 0 && assign_node_ids!(tree)
    n = length(tree)
    neighbors, il, id2node = neighbor_il_lists(tree)
    nn = nnodes(tree)
    boxes = [NNCABox{T}() for _ in 1:nn]
    perm = copy(loc2glob(tree))
    iperm = copy(glob2loc(tree))
    # Local-order kernel (avoids PermutedMatrix gathers on every ACA/M2L sample).
    Kg = global_index ? _nnca_local_kernel(K, perm) : K

    levels_nodes = nodes_by_depth(tree)
    # skeletons, coarse from fine (skip depth 0–1 like SAFRAN). Same-level
    # boxes only read finer children's checks, so siblings are independent.
    dmax = length(levels_nodes) - 1
    rtolf, ranki = float(rtol), Int(rank)
    for d in dmax:-1:2
        lev = levels_nodes[d + 1]
        if threads && Threads.nthreads() > 1 && length(lev) > 1
            Threads.@threads for k in eachindex(lev)
                _nnca_get_nodes!(boxes, lev[k], Kg, il, id2node, T, p;
                    rtol=rtolf, rank=ranki)
            end
        else
            for node in lev
                _nnca_get_nodes!(boxes, node, Kg, il, id2node, T, p;
                    rtol=rtolf, rank=ranki)
            end
        end
    end
    # depth 1: no IL; identity on concatenated child checks (or leaf DOFs)
    if dmax >= 1
        for node in levels_nodes[2]
            _nnca_shallow_basis!(boxes, node, T, p)
        end
    end

    m2l = _nnca_assemble_m2l!(boxes, il, Kg, T, p; threads=threads)
    leaf_ids = [node_id(L) for L in leaves(tree)]
    near = _nnca_assemble_near!(boxes, neighbors, id2node, Kg, T, p, leaf_ids; threads=threads)

    nrank = 0
    nbox = 0
    for d in 2:dmax
        for node in levels_nodes[d + 1]
            nbox += 1
            nrank += length(boxes[node_id(node)].check)
        end
    end
    avg = nbox == 0 ? 0.0 : nrank / nbox

    levels = [Int[node_id(n) for n in lev] for lev in levels_nodes]
    id2any = Any[id2node...]
    return NNCAMatrix{T}(tree, boxes, id2any, neighbors, il, levels,
        leaf_ids, n, p, perm, iperm, avg,
        m2l.dsptr, m2l.src, m2l.ptr, m2l.m, m2l.n, m2l.data,
        near.dsptr, near.src, near.ptr, near.m, near.n, near.data,
        n, tree, boxes, id2any, levels, leaf_ids, perm, iperm)
end

function _nnca_skeleton_level!(boxes, lev, Kg, il, id2, T, p, rtolf, ranki, threads;
        boxes_other=boxes)
    isempty(lev) && return
    if threads && Threads.nthreads() > 1 && length(lev) > 1
        Threads.@threads for k in eachindex(lev)
            _nnca_get_nodes!(boxes, lev[k], Kg, il, id2, T, p;
                rtol=rtolf, rank=ranki, boxes_other=boxes_other)
        end
    else
        for node in lev
            _nnca_get_nodes!(boxes, node, Kg, il, id2, T, p;
                rtol=rtolf, rank=ranki, boxes_other=boxes_other)
        end
    end
    return
end

function _assemble_h2_rect(::Type{T}, K, rowtree::ClusterTree, coltree::ClusterTree, p::Int;
        rtol=1e-6, rank=typemax(Int), global_index=true, threads=true) where {T}
    node_id(rowtree) == 0 && assign_node_ids!(rowtree)
    node_id(coltree) == 0 && assign_node_ids!(coltree)
    n_row = length(rowtree)
    n_col = length(coltree)
    neighbors, il, id2row, id2col_t = dual_neighbor_il_lists(rowtree, coltree)
    nnR = nnodes(rowtree)
    nnC = nnodes(coltree)
    boxes_row = [NNCABox{T}() for _ in 1:nnR]
    boxes_col = [NNCABox{T}() for _ in 1:nnC]
    rperm = copy(loc2glob(rowtree))
    cperm = copy(loc2glob(coltree))
    Kg = global_index ? _nnca_local_kernel(K, rperm, cperm) : K
    Kt = NNCATranspose(Kg)
    id2row_v = Any[id2row...]
    id2col = Any[id2col_t...]
    il_col = [Int[] for _ in 1:nnC]
    for xid in 1:nnR
        for yid in il[xid]
            push!(il_col[yid], xid)
        end
    end
    rtolf, ranki = float(rtol), Int(rank)
    levels_col_n = nodes_by_depth(coltree)
    levels_row_n = nodes_by_depth(rowtree)
    dmax = max(length(levels_row_n), length(levels_col_n)) - 1
    for d in dmax:-1:2
        if d + 1 <= length(levels_col_n)
            _nnca_skeleton_level!(boxes_col, levels_col_n[d + 1], Kt, il_col, id2row, T, p,
                rtolf, ranki, threads; boxes_other=boxes_row)
        end
        if d + 1 <= length(levels_row_n)
            _nnca_skeleton_level!(boxes_row, levels_row_n[d + 1], Kg, il, id2col_t, T, p,
                rtolf, ranki, threads; boxes_other=boxes_col)
        end
    end
    if length(levels_col_n) >= 2
        for node in levels_col_n[2]
            _nnca_shallow_basis!(boxes_col, node, T, p)
        end
    end
    if length(levels_row_n) >= 2
        for node in levels_row_n[2]
            _nnca_shallow_basis!(boxes_row, node, T, p)
        end
    end
    m2l = _nnca_assemble_m2l!(boxes_row, il, Kg, T, p; threads=threads, boxes_src=boxes_col)
    leaf_ids = [node_id(L) for L in leaves(rowtree)]
    leaf_ids_col = [node_id(L) for L in leaves(coltree)]
    near = _nnca_assemble_near_rect!(boxes_row, neighbors, id2row, id2col_t, Kg, T, p,
        leaf_ids; threads=threads)
    nrank = 0
    nbox = 0
    dmax = length(levels_row_n) - 1
    for d in 2:dmax
        for node in levels_row_n[d + 1]
            nbox += 1
            nrank += length(boxes_row[node_id(node)].check)
        end
    end
    avg = nbox == 0 ? 0.0 : nrank / nbox
    levels = [Int[node_id(n) for n in lev] for lev in levels_row_n]
    levels_col = [Int[node_id(n) for n in lev] for lev in levels_col_n]
    return NNCAMatrix{T}(rowtree, boxes_row, id2row_v, neighbors, il, levels,
        leaf_ids, n_row, p, rperm, invperm(rperm), avg,
        m2l.dsptr, m2l.src, m2l.ptr, m2l.m, m2l.n, m2l.data,
        near.dsptr, near.src, near.ptr, near.m, near.n, near.data,
        n_col, coltree, boxes_col, id2col, levels_col, leaf_ids_col, cperm, invperm(cperm))
end

function _nnca_local_kernel(K::KernelMatrix, perm::Vector{Int})
    X = rowelements(K)
    Y = colelements(K)
    Xp = X[perm]
    Yp = Y === X ? Xp : Y[perm]
    return KernelMatrix(kernel(K), Xp, Yp)
end
_nnca_local_kernel(K, perm::Vector{Int}) = PermutedMatrix(K, perm, perm)

function _nnca_local_kernel(K::KernelMatrix, rperm::Vector{Int}, cperm::Vector{Int})
    X = rowelements(K)
    Y = colelements(K)
    return KernelMatrix(kernel(K), X[rperm], Y[cperm])
end
_nnca_local_kernel(K, rperm::Vector{Int}, cperm::Vector{Int}) =
    PermutedMatrix(K, rperm, cperm)

struct NNCATranspose{K, T} <: AbstractMatrix{T}
    parent::K
end
NNCATranspose(K::AbstractMatrix) = NNCATranspose{typeof(K), eltype(K)}(K)
Base.size(A::NNCATranspose) = reverse(size(A.parent))
Base.getindex(A::NNCATranspose, i::Int, j::Int) = A.parent[j, i]

function _nnca_cand_dual(boxes, node, il, id2node, boxes_other=boxes)
    id = node_id(node)
    if isleaf(node)
        cand = collect(index_range(node))
    else
        cand = Int[]
        for c in children(node)
            append!(cand, boxes[node_id(c)].check)
        end
    end
    dual = Int[]
    for qid in il[id]
        Q = id2node[qid]
        if isleaf(Q)
            append!(dual, collect(index_range(Q)))
        else
            for c in children(Q)
                append!(dual, boxes_other[node_id(c)].check)
            end
        end
    end
    return cand, dual
end

function _nnca_get_nodes!(boxes, node, K, il, id2node, ::Type{T}, p::Int; rtol, rank,
        boxes_other=boxes) where {T}
    cand, dual = _nnca_cand_dual(boxes, node, il, id2node, boxes_other)
    b = boxes[node_id(node)]
    function _identity_basis!(ids)
        n = length(ids)
        b.check = ids
        b.charge = Int[]
        b.L2P = Matrix{T}(I, p * n, p * n)
        b.outgoing = zeros(T, p * n)
        b.incoming = zeros(T, p * n)
        return
    end
    if isempty(cand) || isempty(dual)
        _identity_basis!(cand)
        return
    end
    rb, cb, Ac, _, L, R = nnca_aca(K, cand, dual, T; tol=float(rtol), rank=rank)
    r = length(rb)
    if r == 0
        _identity_basis!(cand)
        return
    end
    b.check = cand[rb]
    b.charge = dual[cb]
    temp = copy(Ac)
    if p == 1
        # L2P = (Ac / R) / L   (Eigen OnTheRight solves)
        rdiv!(temp, UpperTriangular(R))
        rdiv!(temp, LowerTriangular(L))
    else
        # R is the skeleton intersection; L2P = Ac / R
        temp = try
            Ac / R
        catch
            Ac * LinearAlgebra.pinv(R)
        end
        if !all(isfinite, temp)
            temp = Ac * LinearAlgebra.pinv(R)
        end
    end
    b.L2P = temp
    b.outgoing = zeros(T, p * r)
    b.incoming = zeros(T, p * r)
    return
end

function _nnca_shallow_basis!(boxes, node, ::Type{T}, p::Int) where {T}
    b = boxes[node_id(node)]
    isempty(b.check) || return
    if isleaf(node)
        cand = collect(index_range(node))
    else
        cand = Int[]
        for c in children(node)
            append!(cand, boxes[node_id(c)].check)
        end
        if isempty(cand)
            for c in children(node)
                append!(cand, collect(index_range(c)))
            end
        end
    end
    n = length(cand)
    b.check = cand
    b.L2P = Matrix{T}(I, p * n, p * n)
    b.outgoing = zeros(T, p * n)
    b.incoming = zeros(T, p * n)
    return
end

function _nnca_assemble_m2l!(boxes, il, K, ::Type{T}, p::Int; threads, boxes_src=boxes) where {T}
    nn = length(boxes)
    counts = zeros(Int, nn)
    ents = zeros(Int, nn)
    @inbounds for i in 1:nn
        b = boxes[i]
        isempty(b.check) && continue
        ni = p * length(b.check)
        c = 0
        e = 0
        for qid in il[i]
            qid > length(boxes_src) && continue
            isempty(boxes_src[qid].check) && continue
            c += 1
            e += ni * (p * length(boxes_src[qid].check))
        end
        counts[i] = c
        ents[i] = e
    end
    dsptr = Vector{Int}(undef, nn + 1)
    eptr = Vector{Int}(undef, nn + 1)
    dsptr[1] = 1
    eptr[1] = 1
    @inbounds for i in 1:nn
        dsptr[i + 1] = dsptr[i] + counts[i]
        eptr[i + 1] = eptr[i] + ents[i]
    end
    npair = dsptr[end] - 1
    src = Vector{Int}(undef, npair)
    ptr = Vector{Int}(undef, npair)
    mm = Vector{Int}(undef, npair)
    nnv = Vector{Int}(undef, npair)
    data = Vector{T}(undef, max(eptr[end] - 1, 0))
    function fill_one(i)
        b = boxes[i]
        isempty(b.check) && return
        ni = p * length(b.check)
        k = dsptr[i]
        off = eptr[i]
        for qid in il[i]
            qid > length(boxes_src) && continue
            qb = boxes_src[qid]
            isempty(qb.check) && continue
            nj = p * length(qb.check)
            M = reshape(view(data, off:(off + ni * nj - 1)), ni, nj)
            _kernel_block!(M, K, b.check, qb.check)
            src[k] = qid
            ptr[k] = off
            mm[k] = ni
            nnv[k] = nj
            k += 1
            off += ni * nj
        end
        return
    end
    if threads && Threads.nthreads() > 1
        Threads.@threads for i in 1:nn
            fill_one(i)
        end
    else
        for i in 1:nn
            fill_one(i)
        end
    end
    return (; dsptr, src, ptr, m=mm, n=nnv, data)
end

function _nnca_assemble_near!(boxes, neighbors, id2node, K, ::Type{T}, p::Int, leaf_ids; threads) where {T}
    nn = length(boxes)
    islf = falses(nn)
    @inbounds for id in leaf_ids
        islf[id] = true
    end
    counts = zeros(Int, nn)
    ents = zeros(Int, nn)
    @inbounds for id in leaf_ids
        node = id2node[id]
        ni = p * length(index_range(node))
        c = 0
        e = 0
        for nid in neighbors[id]
            islf[nid] || continue
            c += 1
            e += ni * (p * length(index_range(id2node[nid])))
        end
        counts[id] = c
        ents[id] = e
    end
    dsptr = Vector{Int}(undef, nn + 1)
    eptr = Vector{Int}(undef, nn + 1)
    dsptr[1] = 1
    eptr[1] = 1
    @inbounds for i in 1:nn
        dsptr[i + 1] = dsptr[i] + counts[i]
        eptr[i + 1] = eptr[i] + ents[i]
    end
    npair = dsptr[end] - 1
    src = Vector{Int}(undef, npair)
    ptr = Vector{Int}(undef, npair)
    mm = Vector{Int}(undef, npair)
    nnv = Vector{Int}(undef, npair)
    data = Vector{T}(undef, max(eptr[end] - 1, 0))
    function fill_leaf(id)
        node = id2node[id]
        b = boxes[id]
        I = collect(index_range(node))
        ni = p * length(I)
        S = Matrix{T}(undef, ni, ni)
        _kernel_block!(S, K, I, I)
        b.self = S
        b.potential = zeros(T, ni)
        k = dsptr[id]
        off = eptr[id]
        for nid in neighbors[id]
            islf[nid] || continue
            J = collect(index_range(id2node[nid]))
            nj = p * length(J)
            M = reshape(view(data, off:(off + ni * nj - 1)), ni, nj)
            _kernel_block!(M, K, I, J)
            src[k] = nid
            ptr[k] = off
            mm[k] = ni
            nnv[k] = nj
            k += 1
            off += ni * nj
        end
        return
    end
    if threads && Threads.nthreads() > 1
        Threads.@threads for k in eachindex(leaf_ids)
            fill_leaf(leaf_ids[k])
        end
    else
        for id in leaf_ids
            fill_leaf(id)
        end
    end
    return (; dsptr, src, ptr, m=mm, n=nnv, data)
end

function _nnca_assemble_near_rect!(boxes, neighbors, id2row, id2col, K, ::Type{T}, p::Int,
        leaf_ids; threads) where {T}
    nn = length(boxes)
    counts = zeros(Int, nn)
    ents = zeros(Int, nn)
    @inbounds for id in leaf_ids
        node = id2row[id]
        ni = p * length(index_range(node))
        c = 0
        e = 0
        for yid in neighbors[id]
            Q = id2col[yid]
            isleaf(Q) || continue
            c += 1
            e += ni * (p * length(index_range(Q)))
        end
        counts[id] = c
        ents[id] = e
    end
    dsptr = Vector{Int}(undef, nn + 1)
    eptr = Vector{Int}(undef, nn + 1)
    dsptr[1] = 1
    eptr[1] = 1
    @inbounds for i in 1:nn
        dsptr[i + 1] = dsptr[i] + counts[i]
        eptr[i + 1] = eptr[i] + ents[i]
    end
    npair = dsptr[end] - 1
    src = Vector{Int}(undef, npair)
    ptr = Vector{Int}(undef, npair)
    mm = Vector{Int}(undef, npair)
    nnv = Vector{Int}(undef, npair)
    data = Vector{T}(undef, max(eptr[end] - 1, 0))
    function fill_leaf(id)
        node = id2row[id]
        b = boxes[id]
        I = collect(index_range(node))
        ni = p * length(I)
        b.potential = zeros(T, ni)
        k = dsptr[id]
        off = eptr[id]
        for yid in neighbors[id]
            Q = id2col[yid]
            isleaf(Q) || continue
            J = collect(index_range(Q))
            nj = p * length(J)
            M = reshape(view(data, off:(off + ni * nj - 1)), ni, nj)
            _kernel_block!(M, K, I, J)
            src[k] = yid
            ptr[k] = off
            mm[k] = ni
            nnv[k] = nj
            k += 1
            off += ni * nj
        end
        return
    end
    if threads && Threads.nthreads() > 1
        Threads.@threads for k in eachindex(leaf_ids)
            fill_leaf(leaf_ids[k])
        end
    else
        for id in leaf_ids
            fill_leaf(id)
        end
    end
    return (; dsptr, src, ptr, m=mm, n=nnv, data)
end

@inline function _pack_gemv!(y::AbstractVector{T}, data::Vector{T}, p0::Int,
        m::Int, n::Int, x::AbstractVector{T}) where {T}
    @inbounds for j in 0:(n - 1)
        xj = x[j + 1]
        base = p0 + j * m - 1
        for i in 1:m
            y[i] += data[base + i] * xj
        end
    end
    return y
end

function LinearAlgebra.mul!(y::AbstractVector, A::NNCAMatrix{T}, x::AbstractVector,
        a::Number=1, b::Number=0; global_index=use_global_index()) where {T}
    if eltype(x) <: SVector
        return _nnca_mul_svec!(y, A, x, a, b; global_index=global_index)
    end
    nrow = A.p * A.n
    ncol = A.p * A.n_col
    length(x) == ncol && length(y) == nrow || throw(DimensionMismatch(
        "NNCA matvec: got $(length(y))×$(length(x)), expected $(nrow)×$(ncol)"))
    xt = global_index ? _nnca_permute_in(x, A.colperm_pts, A.p) : (a == 1 ? x : a .* x)
    if global_index && a != 1
        xt = a .* xt
    end
    yt = _nnca_apply(A, xt)
    if global_index
        _nnca_permute_out!(y, yt, A.perm, A.p, b)
    elseif iszero(b)
        copyto!(y, yt)
    else
        y .= T(b) .* y .+ yt
    end
    return y
end

function _nnca_mul_svec!(y::AbstractVector{SVector{p, T}}, A::NNCAMatrix{T},
        x::AbstractVector{SVector{p, T}}, a::Number, b::Number;
        global_index) where {p, T}
    A.p == p || throw(DimensionMismatch("NNCA block size $(A.p) vs SVector{$p}"))
    length(x) == A.n_col && length(y) == A.n || throw(DimensionMismatch())
    xf = scalarize(x)
    yf = iszero(b) ? similar(xf) : scalarize(y)
    mul!(yf, A, xf, a, b; global_index=global_index)
    y .= descalarize(yf, SVector{p, T})
    return y
end

function Base.:*(A::NNCAMatrix{T}, x::AbstractVector) where {T}
    if eltype(x) <: SVector
        return mul!(similar(x), A, x)
    end
    return mul!(Vector{T}(undef, size(A, 1)), A, x)
end

function _nnca_permute_in(x::AbstractVector, perm::Vector{Int}, p::Int)
    p == 1 && return x[perm]
    n = length(perm)
    xt = similar(x, p * n)
    @inbounds for i in 1:n
        i0 = p * (i - 1)
        g0 = p * (perm[i] - 1)
        for a in 1:p
            xt[i0 + a] = x[g0 + a]
        end
    end
    return xt
end

function _nnca_permute_out!(y::AbstractVector{T}, yt::AbstractVector{T},
        perm::Vector{Int}, p::Int, b) where {T}
    n = length(perm)
    if iszero(b)
        @inbounds for i in 1:n
            i0 = p * (i - 1)
            g0 = p * (perm[i] - 1)
            for a in 1:p
                y[g0 + a] = yt[i0 + a]
            end
        end
    else
        @inbounds for i in 1:n
            i0 = p * (i - 1)
            g0 = p * (perm[i] - 1)
            for a in 1:p
                y[g0 + a] = T(b) * y[g0 + a] + yt[i0 + a]
            end
        end
    end
    return y
end

function _nnca_apply(A::NNCAMatrix{T}, x::AbstractVector{T}) where {T}
    boxes = A.boxes
    id2 = A.id2node
    p = A.p
    for id in A.leaf_ids
        ir = index_range(id2[id])
        b = boxes[id]
        nloc = p * length(ir)
        length(b.potential) == nloc || (b.potential = zeros(T, nloc))
    end
    _nnca_m2m!(A, x)
    _nnca_m2l!(A)
    _nnca_l2l!(A)
    _nnca_near!(A, x)
    y = zeros(T, p * A.n)
    for id in A.leaf_ids
        copyto!(view(y, expand_range(index_range(id2[id]), p)), boxes[id].potential)
    end
    return y
end

@inline function _nnca_foreach_level(ids, f)
    if Threads.nthreads() > 1 && length(ids) >= 8
        Threads.@threads for k in eachindex(ids)
            f(ids[k])
        end
    else
        for id in ids
            f(id)
        end
    end
    return
end

function _nnca_m2m!(A::NNCAMatrix{T}, x::AbstractVector{T}) where {T}
    boxes = A.boxes_col
    id2 = A.id2col
    p = A.p
    for lev in Iterators.reverse(A.levels_col)
        isempty(lev) && continue
        _nnca_foreach_level(lev, id -> begin
            node = id2[id]
            b = boxes[id]
            isempty(b.L2P) && return
            if isleaf(node)
                ir = expand_range(index_range(node), p)
                size(b.L2P, 1) == length(ir) || return
                mul!(b.outgoing, adjoint(b.L2P), view(x, ir))
            else
                need = size(b.L2P, 1)
                stack = Vector{T}(undef, need)
                off = 0
                for c in children(node)
                    co = boxes[node_id(c)].outgoing
                    n = length(co)
                    n == 0 && continue
                    off + n > need && return
                    copyto!(view(stack, (off + 1):(off + n)), co)
                    off += n
                end
                off == need || return
                mul!(b.outgoing, adjoint(b.L2P), stack)
            end
            return
        end)
    end
    return
end

function _nnca_m2l!(A::NNCAMatrix{T}) where {T}
    boxes = A.boxes
    dsptr = A.m2l_dsptr
    src = A.m2l_src
    ptr = A.m2l_ptr
    mm = A.m2l_m
    nnv = A.m2l_n
    data = A.m2l_data
    for lev in A.levels
        isempty(lev) && continue
        _nnca_foreach_level(lev, id -> begin
            b = boxes[id]
            fill!(b.incoming, zero(T))
            isempty(b.incoming) && return
            a = dsptr[id]
            stop = dsptr[id + 1]
            while a < stop
                qo = A.boxes_col[src[a]].outgoing
                n = nnv[a]
                m = mm[a]
                if length(qo) == n && length(b.incoming) == m
                    _pack_gemv!(b.incoming, data, ptr[a], m, n, qo)
                end
                a += 1
            end
            return
        end)
    end
    return
end

function _nnca_l2l!(A::NNCAMatrix{T}) where {T}
    boxes = A.boxes
    id2 = A.id2node
    for lev in A.levels
        isempty(lev) && continue
        _nnca_foreach_level(lev, id -> begin
            node = id2[id]
            b = boxes[id]
            isempty(b.L2P) && return
            if isleaf(node)
                ir = expand_range(index_range(node), A.p)
                size(b.L2P, 1) == length(ir) || return
                length(b.incoming) == size(b.L2P, 2) || return
                mul!(b.potential, b.L2P, b.incoming)
            else
                need = size(b.L2P, 1)
                length(b.incoming) == size(b.L2P, 2) || return
                temp = b.L2P * b.incoming
                off = 0
                for c in children(node)
                    cb = boxes[node_id(c)]
                    n = length(cb.incoming)
                    n == 0 && continue
                    off + n > length(temp) && return
                    cb.incoming .+= view(temp, (off + 1):(off + n))
                    off += n
                end
            end
            return
        end)
    end
    return
end

function _nnca_near!(A::NNCAMatrix{T}, x::AbstractVector{T}) where {T}
    boxes = A.boxes
    id2 = A.id2node
    p = A.p
    dsptr = A.near_dsptr
    src = A.near_src
    ptr = A.near_ptr
    mm = A.near_m
    nnv = A.near_n
    data = A.near_data
    _nnca_foreach_level(A.leaf_ids, id -> begin
        node = id2[id]
        b = boxes[id]
        ir = expand_range(index_range(node), p)
        if !isempty(b.self) && size(b.self, 1) == length(ir)
            mul!(b.potential, b.self, view(x, ir), true, true)
        end
        a = dsptr[id]
        stop = dsptr[id + 1]
        while a < stop
            jr = expand_range(index_range(A.id2col[src[a]]), p)
            n = nnv[a]
            m = mm[a]
            if length(jr) == n && length(b.potential) == m
                _pack_gemv!(b.potential, data, ptr[a], m, n, view(x, jr))
            end
            a += 1
        end
        return
    end)
    return
end
