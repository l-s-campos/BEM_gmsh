# Interpolative H²: tensor Chebyshev nested bases (Hackbusch–Börm).
# Leaf L2P interpolates cluster points from Chebyshev nodes in the AABB;
# parent L2P interpolates child Chebyshev nodes from parent nodes; M2L is
# the kernel on those nodes. Apply reuses NNCA M2M / M2L / L2L / near.

function _cheb_nodes_1d(n::Int, lo::Float64, hi::Float64)
    n < 1 && (n = 1)
    mid = 0.5 * (lo + hi)
    n == 1 && return [mid]
    sc = 0.499 * (hi - lo)          # stay strictly inside the box
    iszero(sc) && return [mid]
    return [mid - sc * cospi((2j - 1) / (2n)) for j in 1:n]
end

function _cheb_weights_1d(n::Int)
    n == 1 && return [1.0]
    return [(-1.0)^(j - 1) * sinpi((2j - 1) / (2n)) for j in 1:n]
end

function _barycentric_1d!(ℓ::Vector{Float64}, x::Float64, nodes::Vector{Float64},
        weights::Vector{Float64})
    n = length(nodes)
    s = 0.0
    @inbounds for j in 1:n
        d = x - nodes[j]
        if abs(d) <= 1e-15 * (1.0 + abs(x))
            fill!(ℓ, 0.0)
            ℓ[j] = 1.0
            return ℓ
        end
        w = weights[j] / d
        ℓ[j] = w
        s += w
    end
    invs = 1.0 / s
    @inbounds for j in 1:n
        ℓ[j] *= invs
    end
    return ℓ
end

function _cheb_nax(nd::Int, order::Int)
    o = order > 0 ? order : (nd >= 3 ? 4 : 6)
    return ntuple(_ -> o, nd)
end

function _cheb_axes(box, nax::NTuple{N, Int}) where {N}
    lo = low_corner(box)
    hi = high_corner(box)
    nodes = ntuple(d -> _cheb_nodes_1d(nax[d], Float64(lo[d]), Float64(hi[d])), N)
    weights = ntuple(d -> _cheb_weights_1d(nax[d]), N)
    return nodes, weights
end

function _cheb_grid(nodes::NTuple{N, Vector{Float64}}) where {N}
    ntot = prod(length, nodes)
    pts = Vector{SVector{N, Float64}}(undef, ntot)
    _cheb_fill_grid!(pts, nodes)
    return pts
end

function _cheb_fill_grid!(pts::Vector{SVector{1, Float64}}, nodes)
    xs = nodes[1]
    @inbounds for i in eachindex(xs)
        pts[i] = SVector(xs[i])
    end
    return pts
end

function _cheb_fill_grid!(pts::Vector{SVector{2, Float64}}, nodes)
    xs, ys = nodes[1], nodes[2]
    t = 0
    @inbounds for y in ys, x in xs
        t += 1
        pts[t] = SVector(x, y)
    end
    return pts
end

function _cheb_fill_grid!(pts::Vector{SVector{3, Float64}}, nodes)
    xs, ys, zs = nodes[1], nodes[2], nodes[3]
    t = 0
    @inbounds for z in zs, y in ys, x in xs
        t += 1
        pts[t] = SVector(x, y, z)
    end
    return pts
end

function _cheb_interp_rows!(L::Matrix{Float64}, xpts, nodes, weights)
    nd = length(nodes)
    nd == 2 && return _cheb_interp_rows2!(L, xpts, nodes, weights)
    nd == 3 && return _cheb_interp_rows3!(L, xpts, nodes, weights)
    return _cheb_interp_rows1!(L, xpts, nodes, weights)
end

function _cheb_interp_rows1!(L, xpts, nodes, weights)
    xs, wx = nodes[1], weights[1]
    lx = zeros(length(xs))
    @inbounds for i in eachindex(xpts)
        _barycentric_1d!(lx, Float64(xpts[i][1]), xs, wx)
        for j in eachindex(lx)
            L[i, j] = lx[j]
        end
    end
    return L
end

function _cheb_interp_rows2!(L, xpts, nodes, weights)
    xs, ys = nodes[1], nodes[2]
    wx, wy = weights[1], weights[2]
    nx, ny = length(xs), length(ys)
    lx = zeros(nx)
    ly = zeros(ny)
    @inbounds for i in eachindex(xpts)
        p = xpts[i]
        _barycentric_1d!(lx, Float64(p[1]), xs, wx)
        _barycentric_1d!(ly, Float64(p[2]), ys, wy)
        col = 1
        for iy in 1:ny, ix in 1:nx
            L[i, col] = lx[ix] * ly[iy]
            col += 1
        end
    end
    return L
end

function _cheb_interp_rows3!(L, xpts, nodes, weights)
    xs, ys, zs = nodes[1], nodes[2], nodes[3]
    wx, wy, wz = weights[1], weights[2], weights[3]
    nx, ny, nz = length(xs), length(ys), length(zs)
    lx = zeros(nx)
    ly = zeros(ny)
    lz = zeros(nz)
    @inbounds for i in eachindex(xpts)
        p = xpts[i]
        _barycentric_1d!(lx, Float64(p[1]), xs, wx)
        _barycentric_1d!(ly, Float64(p[2]), ys, wy)
        _barycentric_1d!(lz, Float64(p[3]), zs, wz)
        col = 1
        for iz in 1:nz, iy in 1:ny, ix in 1:nx
            L[i, col] = lx[ix] * ly[iy] * lz[iz]
            col += 1
        end
    end
    return L
end

function _assemble_h2_cheb(::Type{T}, K, tree::ClusterTree{N};
        order::Int = 0, rtol::Real = 0, rank::Integer = typemax(Int),
        global_index::Bool = true, threads::Bool = true) where {T, N}
    K isa KernelMatrix || throw(ArgumentError(
        "assemble_h2 method=:cheb needs a KernelMatrix (kernel + coordinates)"))
    f = kernel(K)
    pts = rowelements(K)
    node_id(tree) == 0 && assign_node_ids!(tree)
    n = length(tree)
    neighbors, il, id2node = neighbor_il_lists(tree)
    nn = nnodes(tree)
    boxes = [NNCABox{T}() for _ in 1:nn]
    perm = copy(loc2glob(tree))
    iperm = copy(glob2loc(tree))
    Kg = global_index ? _nnca_local_kernel(K, perm) : K
    nax = _cheb_nax(N, order)
    cheb = Vector{Vector{SVector{N, Float64}}}(undef, nn)
    axes = Vector{Any}(undef, nn)
    par = threads && Threads.nthreads() > 1
    levels_nodes = nodes_by_depth(tree)
    dmax = length(levels_nodes) - 1

    function fill_box(node)
        id = node_id(node)
        b = boxes[id]
        B = container(node)
        nds, wts = _cheb_axes(B, nax)
        axes[id] = (nds, wts)
        Ξ = _cheb_grid(nds)
        cheb[id] = Ξ
        k = length(Ξ)
        if isleaf(node)
            I = collect(index_range(node))
            xloc = [pts[perm[i]] for i in I]
            L = Matrix{T}(undef, length(I), k)
            _cheb_interp_rows!(L, xloc, nds, wts)
            b.L2P = L
        else
            ch = children(node)
            rows = vcat([cheb[node_id(c)] for c in ch]...)
            L = Matrix{T}(undef, length(rows), k)
            _cheb_interp_rows!(L, rows, nds, wts)
            b.L2P = L
        end
        b.check = collect(1:k)
        b.outgoing = zeros(T, k)
        b.incoming = zeros(T, k)
        return
    end

    for d in dmax:-1:0
        lev = levels_nodes[d + 1]
        isempty(lev) && continue
        if par && length(lev) > 1
            nblas = BLAS.get_num_threads()
            try
                BLAS.set_num_threads(1)
                Threads.@threads for k in eachindex(lev)
                    fill_box(lev[k])
                end
            finally
                BLAS.set_num_threads(nblas)
            end
        else
            for node in lev
                fill_box(node)
            end
        end
    end

    m2l = _cheb_assemble_m2l!(boxes, il, f, cheb, T; threads = threads)
    leaf_ids = [node_id(L) for L in leaves(tree)]
    near = _nnca_assemble_near!(boxes, neighbors, id2node, Kg, T, 1, leaf_ids; threads = threads)

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
    A = NNCAMatrix{T}(tree, boxes, id2any, neighbors, il, levels,
        leaf_ids, n, 1, perm, iperm, avg,
        m2l.dsptr, m2l.src, m2l.ptr, m2l.m, m2l.n, m2l.data,
        near.dsptr, near.src, near.ptr, near.m, near.n, near.data,
        n, tree, boxes, id2any, levels, leaf_ids, perm, iperm)
    if float(rtol) > 0
        _h2_cheb_recompress!(A, float(rtol); rank = Int(rank), threads = threads)
    end
    return A
end

"""
    _h2_cheb_recompress!(A, rtol; rank=typemax(Int), threads=true)

H2Lib-style weighted truncation (`weight_clusterbasis` + `localweights` +
`truncate_clusterbasis`). Nested QR weights of `L2P`/`E` and QR of stacked
`W_s S'` couplings; SVD of `Vhat Z'` at relative precision `rtol`.
Then `S ← U_t' S U_s` and the nested bases are replaced by the left
singular vectors.
"""
function _h2_cheb_recompress!(A::NNCAMatrix{T}, rtol::Float64;
        rank::Int = typemax(Int), threads::Bool = true) where {T}
    A.p == 1 || throw(ArgumentError("Chebyshev recompress is scalar only"))
    _nnca_square(A) || throw(ArgumentError("Chebyshev recompress needs a square H²"))
    nn = length(A.boxes)
    W = _h2_cheb_basis_weights(A)
    Lw = _h2_cheb_local_weights(A, W; threads = threads)
    Utr, Qs = _h2_cheb_weighted_truncate(A, W, Lw, rtol, rank)
    _h2_cheb_transform_m2l!(A, Utr)
    _h2_cheb_install_bases!(A, Qs)
    nrank = 0
    nbox = 0
    @inbounds for id in 1:nn
        k = length(A.boxes[id].check)
        k == 0 && continue
        nbox += 1
        nrank += k
    end
    A.avg_rank = nbox == 0 ? 0.0 : nrank / nbox
    return A
end

function _h2_thin_R(A::AbstractMatrix{T}) where {T}
    m, n = size(A)
    (m == 0 || n == 0) && return zeros(T, 0, n)
    F = qr(A)
    r = min(m, n)
    return Matrix{T}(F.R[1:r, :])
end

function _h2_cheb_basis_weights(A::NNCAMatrix{T}) where {T}
    nn = length(A.boxes)
    W = Vector{Matrix{T}}(undef, nn)
    id2 = A.id2node
    for lev in Iterators.reverse(A.levels)
        for id in lev
            b = A.boxes[id]
            k = size(b.L2P, 2)
            if k == 0
                W[id] = zeros(T, 0, 0)
                continue
            end
            node = id2[id]
            if isleaf(node)
                W[id] = _h2_thin_R(b.L2P)
                continue
            end
            E = b.L2P
            row0 = 0
            blocks = Matrix{T}[]
            for c in children(node)
                cid = node_id(c)
                kc = size(A.boxes[cid].L2P, 2)
                if kc == 0
                    continue
                end
                Wc = W[cid]
                Ec = E[(row0 + 1):(row0 + kc), :]
                push!(blocks, isempty(Wc) ? Ec : Wc * Ec)
                row0 += kc
            end
            W[id] = isempty(blocks) ? _h2_thin_R(E) : _h2_thin_R(vcat(blocks...))
        end
    end
    return W
end

function _h2_cheb_local_weights(A::NNCAMatrix{T}, W::Vector{Matrix{T}};
        threads::Bool = true) where {T}
    nn = length(A.boxes)
    Lw = Vector{Matrix{T}}(undef, nn)
    dsptr, src, ptr, mm, nnv, data = A.m2l_dsptr, A.m2l_src, A.m2l_ptr, A.m2l_m, A.m2l_n, A.m2l_data
    npair = dsptr[end] - 1
    dest_of = zeros(Int, max(npair, 0))
    rev = [Int[] for _ in 1:nn]
    @inbounds for i in 1:nn
        a = dsptr[i]
        stop = dsptr[i + 1]
        while a < stop
            dest_of[a] = i
            push!(rev[src[a]], a)
            a += 1
        end
    end
    function one(i)
        k = size(A.boxes[i].L2P, 2)
        if k == 0
            Lw[i] = zeros(T, 0, 0)
            return
        end
        parts = Matrix{T}[]
        a = dsptr[i]
        stop = dsptr[i + 1]
        @inbounds while a < stop
            s = src[a]
            m, n = mm[a], nnv[a]
            S = reshape(view(data, ptr[a]:(ptr[a] + m * n - 1)), m, n)
            Ws = W[s]
            if m == k && size(Ws, 2) == n
                push!(parts, isempty(Ws) ? Matrix(transpose(S)) : Ws * transpose(S))
            end
            a += 1
        end
        @inbounds for a in rev[i]
            m, n = mm[a], nnv[a]
            n == k || continue
            S = reshape(view(data, ptr[a]:(ptr[a] + m * n - 1)), m, n)
            dest = dest_of[a]
            dest == 0 && continue
            Wd = W[dest]
            if size(Wd, 2) == m
                push!(parts, isempty(Wd) ? Matrix(S) : Wd * S)
            end
        end
        Lw[i] = isempty(parts) ? zeros(T, 0, k) : _h2_thin_R(vcat(parts...))
        return
    end
    if threads && Threads.nthreads() > 1 && nn > 1
        nblas = BLAS.get_num_threads()
        try
            BLAS.set_num_threads(1)
            Threads.@threads for i in 1:nn
                one(i)
            end
        finally
            BLAS.set_num_threads(nblas)
        end
    else
        for i in 1:nn
            one(i)
        end
    end
    return Lw
end

function _h2_cheb_merge_weight(W::AbstractMatrix{T}, Lw::AbstractMatrix{T}, k::Int) where {T}
    if k == 0
        return zeros(T, 0, 0)
    end
    if size(W, 2) != k
        W = isempty(W) ? zeros(T, 0, k) : W
    end
    if size(Lw, 2) != k
        Lw = isempty(Lw) ? zeros(T, 0, k) : Lw
    end
    if size(W, 1) == 0 && size(Lw, 1) == 0
        return Matrix{T}(I, k, k)
    elseif size(Lw, 1) == 0
        return W
    elseif size(W, 1) == 0
        return Lw
    end
    return _h2_thin_R(vcat(W, Lw))
end

function _h2_cheb_weighted_truncate(A::NNCAMatrix{T}, W::Vector{Matrix{T}},
        Lw::Vector{Matrix{T}}, rtol::Float64, rankmax::Int) where {T}
    nn = length(A.boxes)
    Utr = Vector{Matrix{T}}(undef, nn)
    Qs = Vector{Matrix{T}}(undef, nn)
    id2 = A.id2node
    for lev in Iterators.reverse(A.levels)
        for id in lev
            b = A.boxes[id]
            kold = size(b.L2P, 2)
            if kold == 0
                Utr[id] = zeros(T, 0, 0)
                Qs[id] = zeros(T, 0, 0)
                continue
            end
            node = id2[id]
            if isleaf(node)
                Vhat = b.L2P
            else
                E = b.L2P
                row0 = 0
                blocks = Matrix{T}[]
                for c in children(node)
                    cid = node_id(c)
                    kc = size(A.boxes[cid].L2P, 2)
                    kc == 0 && continue
                    Ec = E[(row0 + 1):(row0 + kc), :]
                    Uc = Utr[cid]
                    push!(blocks, isempty(Uc) ? Ec[1:0, :] : transpose(Uc) * Ec)
                    row0 += kc
                end
                Vhat = isempty(blocks) ? E[1:0, :] : vcat(blocks...)
            end
            Z = _h2_cheb_merge_weight(W[id], Lw[id], kold)
            VhatZ = isempty(Z) ? Matrix(Vhat) : Vhat * transpose(Z)
            if size(VhatZ, 1) == 0 || size(VhatZ, 2) == 0
                r = 0
                Qs[id] = zeros(T, size(Vhat, 1), 0)
                Utr[id] = zeros(T, kold, 0)
                continue
            end
            F = svd(VhatZ; full = false)
            τ = rtol * abs(F.S[1])
            r = 0
            @inbounds for σ in F.S
                σ > τ || break
                r += 1
            end
            r = clamp(max(r, 1), 1, min(length(F.S), rankmax, size(Vhat, 1), kold))
            Q = F.U[:, 1:r]
            Qs[id] = Q
            Utr[id] = transpose(Vhat) * Q
        end
    end
    return Utr, Qs
end

function _h2_cheb_install_bases!(A::NNCAMatrix{T}, Qs::Vector{Matrix{T}}) where {T}
    nn = length(A.boxes)
    @inbounds for id in 1:nn
        b = A.boxes[id]
        size(b.L2P, 2) == 0 && continue
        Q = Qs[id]
        b.L2P = Q
        r = size(Q, 2)
        b.check = collect(1:r)
        b.outgoing = zeros(T, r)
        b.incoming = zeros(T, r)
    end
    return A
end

function _h2_cheb_transform_m2l!(A::NNCAMatrix{T}, Utr::Vector{Matrix{T}}) where {T}
    nn = length(A.boxes)
    dsptr, src, ptr, mm, nnv, data = A.m2l_dsptr, A.m2l_src, A.m2l_ptr, A.m2l_m, A.m2l_n, A.m2l_data
    npair = dsptr[end] - 1
    newS = Vector{Matrix{T}}(undef, max(npair, 0))
    ents = 0
    @inbounds for i in 1:nn
        a = dsptr[i]
        stop = dsptr[i + 1]
        Ui = Utr[i]
        while a < stop
            s = src[a]
            m, n = mm[a], nnv[a]
            S = reshape(view(data, ptr[a]:(ptr[a] + m * n - 1)), m, n)
            Us = Utr[s]
            S2 = (isempty(Ui) || isempty(Us)) ? S[1:0, 1:0] : Ui' * S * Us
            newS[a] = S2
            ents += length(S2)
            a += 1
        end
    end
    ndata = Vector{T}(undef, ents)
    nptr = Vector{Int}(undef, max(npair, 0))
    nmm = Vector{Int}(undef, max(npair, 0))
    nnn = Vector{Int}(undef, max(npair, 0))
    off = 1
    @inbounds for a in 1:npair
        S2 = newS[a]
        m, n = size(S2)
        nptr[a] = off
        nmm[a] = m
        nnn[a] = n
        copyto!(view(ndata, off:(off + m * n - 1)), vec(S2))
        off += m * n
    end
    A.m2l_ptr = nptr
    A.m2l_m = nmm
    A.m2l_n = nnn
    A.m2l_data = ndata
    return A
end

function _cheb_assemble_m2l!(boxes, il, f, cheb, ::Type{T}; threads,
        boxes_src = boxes, cheb_src = cheb) where {T}
    nn = length(boxes)
    counts = zeros(Int, nn)
    ents = zeros(Int, nn)
    @inbounds for i in 1:nn
        b = boxes[i]
        isempty(b.check) && continue
        isassigned(cheb, i) || continue
        ni = length(cheb[i])
        c = 0
        e = 0
        for qid in il[i]
            qid > length(boxes_src) && continue
            isassigned(cheb_src, qid) || continue
            isempty(boxes_src[qid].check) && continue
            c += 1
            e += ni * length(cheb_src[qid])
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
        isassigned(cheb, i) || return
        Xt = cheb[i]
        ni = length(Xt)
        ni == 0 && return
        k = dsptr[i]
        off = eptr[i]
        for qid in il[i]
            qid > length(boxes_src) && continue
            isassigned(cheb_src, qid) || continue
            isempty(boxes_src[qid].check) && continue
            Xs = cheb_src[qid]
            nj = length(Xs)
            @inbounds for j in 1:nj, ii in 1:ni
                data[off + (j - 1) * ni + (ii - 1)] = T(f(Xt[ii], Xs[j]))
            end
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
        nblas = BLAS.get_num_threads()
        try
            BLAS.set_num_threads(1)
            Threads.@threads for i in 1:nn
                fill_one(i)
            end
        finally
            BLAS.set_num_threads(nblas)
        end
    else
        for i in 1:nn
            fill_one(i)
        end
    end
    return (; dsptr, src, ptr, m = mm, n = nnv, data)
end
