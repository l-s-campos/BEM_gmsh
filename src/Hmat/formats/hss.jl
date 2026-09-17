# Hierarchically semi-separable (HSS) matrices on a binary ClusterTree
# (PrincipalComponentSplitter). Nested interpolative generators + sibling B.
# Default construction is FLAM rskel / Xia nested ID against neighbors+proxy,
# not SVD of the dense complement A(t, t^c).
# Xia et al., Numer. Linear Algebra Appl. 17 (2010);
# Ho–Greengard, SIAM J. Sci. Comput. 34 (2012) (rskel).

"""Binary HSS node. Generators `U,V` live on leaves; `R,W,B` on parents.
`m`,`n` are the row and column sizes of this block; `nl` is the left child's rows."""
mutable struct HSSNode{T}
    leaf::Bool
    root::Bool
    m::Int
    n::Int
    D::Matrix{T}
    U::Matrix{T}
    V::Matrix{T}
    Rl::Matrix{T}
    Rr::Matrix{T}
    Wl::Matrix{T}
    Wr::Matrix{T}
    B12::Matrix{T}
    B21::Matrix{T}
    left::Union{HSSNode{T}, Nothing}
    right::Union{HSSNode{T}, Nothing}
    nl::Int
end

"""HSS matrix in tree-local order. `perm`/`iperm` are row loc2glob; `cperm`/`icperm` columns."""
struct HSSMatrix{T} <: AbstractMatrix{T}
    root::HSSNode{T}
    m::Int
    n::Int
    perm::Vector{Int}
    iperm::Vector{Int}
    cperm::Vector{Int}
    icperm::Vector{Int}
end

Base.size(A::HSSMatrix) = (A.m, A.n)
function Base.size(A::HSSMatrix, d::Integer)
    d == 1 && return A.m
    d == 2 && return A.n
    return 1
end
Base.eltype(::HSSMatrix{T}) where {T} = T

function Base.show(io::IO, A::HSSMatrix)
    k = _hss_maxrank(A.root)
    return print(io, "HSSMatrix{", eltype(A), "} ", A.m, "×", A.n, " maxrank=", k)
end
Base.show(io::IO, ::MIME"text/plain", A::HSSMatrix) = show(io, A)

function _hss_maxrank(N::HSSNode)
    N.leaf && return max(size(N.U, 2), size(N.V, 2))
    return max(size(N.B12, 1), size(N.B12, 2), size(N.Rl, 2),
        _hss_maxrank(N.left), _hss_maxrank(N.right))
end
maxrank(A::HSSMatrix) = _hss_maxrank(A.root)

"""Wall-time split of [`assemble_hss`](@ref). Pass `stats=` to accumulate."""
mutable struct HSSBuildStats
    t_kernel::Float64
    t_svd::Float64
    t_pinv::Float64
    t_lsq::Float64
    n_kernel::Int
    n_svd::Int
    numel_kernel::Int
end
HSSBuildStats() = HSSBuildStats(0.0, 0.0, 0.0, 0.0, 0, 0, 0)

const _HSS_STATS = Ref{Union{Nothing, HSSBuildStats}}(nothing)
const _HSS_STATS_LOCK = ReentrantLock()

function _hss_note_kernel!(dt, nI, nJ)
    s = _HSS_STATS[]
    s === nothing && return
    lock(_HSS_STATS_LOCK) do
        s.t_kernel += dt
        s.n_kernel += 1
        s.numel_kernel += nI * nJ
    end
    return
end

function _hss_note_id!(dt)
    s = _HSS_STATS[]
    s === nothing && return
    lock(_HSS_STATS_LOCK) do
        s.t_svd += dt
        s.n_svd += 1
    end
    return
end

function _hss_block(K, I::AbstractVector{Int}, J::AbstractVector{Int}, ::Type{T}) where {T}
    M = Matrix{T}(undef, length(I), length(J))
    (isempty(I) || isempty(J)) && return M
    s = _HSS_STATS[]
    t0 = s === nothing ? UInt64(0) : time_ns()
    _kernel_block!(M, K, Vector{Int}(I), Vector{Int}(J))
    s !== nothing && _hss_note_kernel!(1e-9 * (time_ns() - t0), length(I), length(J))
    return M
end

function _hss_id(Kid::AbstractMatrix{T}; rtol, rank, Tmax) where {T}
    s = _HSS_STATS[]
    t0 = s === nothing ? UInt64(0) : time_ns()
    sk, rd, Tm = interpolative_decomp(Kid; rtol = rtol, rank = rank, Tmax = Tmax)
    s !== nothing && _hss_note_id!(1e-9 * (time_ns() - t0))
    return sk, rd, Tm
end

function _hss_aca(Kid::AbstractMatrix{T}; rtol, rank) where {T}
    s = _HSS_STATS[]
    t0 = s === nothing ? UInt64(0) : time_ns()
    sk, rd, Tm = _aca_column_id(Kid; rtol = rtol, rank = rank)
    s !== nothing && _hss_note_id!(1e-9 * (time_ns() - t0))
    return sk, rd, Tm
end

function _hss_skel(Kid, method; rtol, rank, Tmax)
    return method === :aca ? _hss_aca(Kid; rtol = rtol, rank = rank) :
        _hss_id(Kid; rtol = rtol, rank = rank, Tmax = Tmax)
end

function _hss_interp(::Type{T}, n::Int, sk::Vector{Int}, rd::Vector{Int},
        Tm::AbstractMatrix) where {T}
    k = length(sk)
    k == 0 && return zeros(T, n, 0)
    U = zeros(T, n, k)
    @inbounds for (j, s) in enumerate(sk)
        U[s, j] = one(T)
    end
    if !isempty(rd) && !isempty(Tm)
        U[rd, :] = adjoint(Tm)
    end
    return U
end

function _hss_Z(::Type{T}) where {T}
    return zeros(T, 0, 0)
end

function _hss_default_pxyfun(K, tree::ClusterTree{N}) where {N}
    K isa KernelMatrix || return nothing
    N == 2 && return circle_proxy(K)
    N == 3 && return sphere_proxy(K)
    return nothing
end

function _hss_nbr_skel(skel::Vector{Vector{Int}}, neigh::Vector{Vector{Int}},
        id::Int, slf::Vector{Int})
    nids = neigh[id]
    n = 0
    @inbounds for nid in nids
        n += length(skel[nid])
    end
    out = Vector{Int}(undef, n)
    t = 0
    seen = Set(slf)
    @inbounds for nid in nids
        for i in skel[nid]
            i in seen && continue
            t += 1
            out[t] = i
            push!(seen, i)
        end
    end
    resize!(out, t)
    return out
end

function _hss_near_skel(skel::Vector{Vector{Int}}, neigh::Vector{Vector{Int}},
        box, slf::Vector{Int})
    id = node_id(box)
    nbr = _hss_nbr_skel(skel, neigh, id, slf)
    P = parentnode(box)
    P === box && return nbr
    seen = Set(slf)
    for i in nbr
        push!(seen, i)
    end
    for S in children(P)
        S === box && continue
        for i in skel[node_id(S)]
            i in seen && continue
            push!(nbr, i)
            push!(seen, i)
        end
    end
    return nbr
end

function _hss_sample(K, slf::Vector{Int}, nbr::Vector{Int}, box, pxyfun, depth_,
        ::Type{T}, symm::Symbol) where {T}
    Kpxy = zeros(T, 0, length(slf))
    if depth_ >= 2 && pxyfun !== nothing && !isempty(slf)
        Kpxy, nbr = pxyfun(slf, nbr, box)
    end
    Anbr = isempty(nbr) ? zeros(T, 0, length(slf)) : _hss_block(K, nbr, slf, T)
    if symm === :n && !isempty(nbr)
        Anbr = vcat(Anbr, adjoint(_hss_block(K, slf, nbr, T)))
    end
    return isempty(Kpxy) ? Anbr : vcat(Anbr, Kpxy)
end

"""
    assemble_hss(K, tree; rtol=1e-6, rank=typemax(Int), pxyfun=nothing, Tmax=2, symm=:s)
    assemble_hss(K, rowtree, coltree; kwargs...)

HSS representation of kernel `K` on a **binary** [`ClusterTree`](@ref)
(`PrincipalComponentSplitter` or `GeometricSplitter`). One tree is square
(same row/column partition). Two trees give a rectangular HSS (`U` on
`rowtree`, `V` on `coltree`). Default `method=:id` is FLAM `rskel`: nested
interpolative decomposition of neighbor + proxy samples. `method=:aca` uses
partial ACA on the same sample. `pxyfun` defaults to [`circle_proxy`](@ref) /
[`sphere_proxy`](@ref) for a square [`KernelMatrix`](@ref). Factor square
HSS with [`ulv`](@ref). `method=:svd` is Xia dense-complement (square or rectangular).
Same-level boxes are independent: `threads=true` (default) IDs them in
parallel (BLAS is pinned to 1 thread in that loop). `recompress=true`
QR-proper + relative tsvd of sibling `B` after the ID sweep (hm-toolbox);
off by default because coarse SVD is costly.
"""
function assemble_hss(K::AbstractMatrix, tree::ClusterTree; kwargs...)
    return assemble_hss(eltype(K), K, tree, tree; kwargs...)
end

function assemble_hss(K::AbstractMatrix, rowtree::ClusterTree, coltree::ClusterTree; kwargs...)
    return assemble_hss(eltype(K), K, rowtree, coltree; kwargs...)
end

function assemble_hss(::Type{T}, K, rowtree::ClusterTree, coltree::ClusterTree;
        rtol = 1e-6, rank = typemax(Int),
        stats::Union{Nothing, HSSBuildStats} = nothing,
        pxyfun = nothing,
        Tmax = 2,
        symm::Symbol = :s,
        method::Symbol = :id,
        threads::Bool = true,
        recompress::Bool = false,
    ) where {T}
    for (lab, tr) in (("row", rowtree), ("col", coltree))
        ch = children(tr)
        (isempty(ch) || length(ch) == 2) ||
            throw(ArgumentError("assemble_hss $lab tree must be binary (PrincipalComponentSplitter / GeometricSplitter); got $(length(ch)) children"))
    end
    square = rowtree === coltree
    m = length(rowtree)
    n = length(coltree)
    rperm = copy(loc2glob(rowtree))
    irperm = copy(glob2loc(rowtree))
    cperm = square ? rperm : copy(loc2glob(coltree))
    icperm = square ? irperm : copy(glob2loc(coltree))
    old = _HSS_STATS[]
    _HSS_STATS[] = stats
    try
        if method === :svd
            if square
                root, _, _ = _hss_build_svd(rowtree, K, rperm, m, true, float(rtol), Int(rank), T)
            else
                root, _, _ = _hss_build_svd_rect(rowtree, coltree, K, rperm, cperm,
                    m, n, true, float(rtol), Int(rank), T)
            end
            return HSSMatrix{T}(root, m, n, rperm, irperm, cperm, icperm)
        end
        (method === :id || method === :aca) ||
            throw(ArgumentError("assemble_hss method must be :id, :aca, or :svd"))
        if square
            pxy = pxyfun === nothing ? _hss_default_pxyfun(K, rowtree) : pxyfun
            root = _hss_build_id(rowtree, K, rperm, float(rtol), Int(rank), float(Tmax),
                pxy, symm, method, T; threads = threads)
        else
            pxyU, pxyV = _hss_default_pxy_rect(K, rowtree, coltree)
            root = _hss_build_rect(rowtree, coltree, K, rperm, cperm,
                float(rtol), Int(rank), float(Tmax), pxyU, pxyV, method, true, T)
        end
        if recompress && !root.leaf
            _hss_recompress!(root, float(rtol))
        end
        return HSSMatrix{T}(root, m, n, rperm, irperm, cperm, icperm)
    finally
        _HSS_STATS[] = old
    end
end

function _hss_build_id(tree, K, perm, rtol, rank, Tmax, pxyfun, symm, method,
        ::Type{T}; threads::Bool = true) where {T}
    node_id(tree) == 0 && assign_node_ids!(tree)
    neigh, _, id2 = neighbor_il_lists(tree)
    nn = nnodes(tree)
    skel = [Int[] for _ in 1:nn]
    nodes_h = Vector{Union{HSSNode{T}, Nothing}}(nothing, nn)
    Z = _hss_Z(T)
    for L in leaves(tree)
        I = collect(index_range(L))
        skel[node_id(L)] = perm[I]
    end
    levels = nodes_by_depth(tree)
    par = threads && Threads.nthreads() > 1
    for lev in Iterators.reverse(levels)
        isempty(lev) && continue
        for box in lev
            isempty(children(box)) && continue
            id = node_id(box)
            ch = children(box)
            skel[id] = vcat(skel[node_id(ch[1])], skel[node_id(ch[2])])
        end
        nlev = length(lev)
        nds = Vector{HSSNode{T}}(undef, nlev)
        sko = Vector{Vector{Int}}(undef, nlev)
        if par && nlev > 1
            nblas = BLAS.get_num_threads()
            try
                BLAS.set_num_threads(1)
                Threads.@threads for k in 1:nlev
                    nds[k], sko[k] = _hss_id_box(lev[k], tree, K, perm, rtol, rank, Tmax,
                        pxyfun, symm, method, T, neigh, skel, nodes_h, Z)
                end
            finally
                BLAS.set_num_threads(nblas)
            end
        else
            for k in 1:nlev
                nds[k], sko[k] = _hss_id_box(lev[k], tree, K, perm, rtol, rank, Tmax,
                    pxyfun, symm, method, T, neigh, skel, nodes_h, Z)
            end
        end
        for k in 1:nlev
            id = node_id(lev[k])
            nodes_h[id] = nds[k]
            skel[id] = sko[k]
        end
    end
    return nodes_h[node_id(tree)]
end

function _hss_id_box(box, tree, K, perm, rtol, rank, Tmax, pxyfun, symm, method,
        ::Type{T}, neigh, skel, nodes_h, Z) where {T}
    isrt = box === tree
    ch = children(box)
    if isempty(ch)
        I = collect(index_range(box))
        m = length(I)
        slf = perm[I]
        D = _hss_block(K, slf, slf, T)
        if isrt
            nd = HSSNode{T}(true, true, m, m, D, zeros(T, m, 0), zeros(T, m, 0),
                Z, Z, Z, Z, Z, Z, nothing, nothing, m)
            return nd, slf
        end
        nbr = pxyfun === nothing ? _hss_complement(slf, length(perm)) :
            _hss_near_skel(skel, neigh, box, slf)
        Kid = _hss_sample(K, slf, nbr, box, pxyfun, depth(box), T, symm)
        if size(Kid, 1) == 0 || length(slf) < 2
            U = Matrix{T}(LinearAlgebra.I, m, m)
            nd = HSSNode{T}(true, false, m, m, D, U, copy(U),
                Z, Z, Z, Z, Z, Z, nothing, nothing, m)
            return nd, slf
        end
        sk, rd, Tm = _hss_skel(Kid, method; rtol = rtol, rank = rank, Tmax = Tmax)
        U = _hss_interp(T, m, sk, rd, Tm)
        nd = HSSNode{T}(true, false, m, m, D, U, copy(U),
            Z, Z, Z, Z, Z, Z, nothing, nothing, m)
        return nd, slf[sk]
    end
    length(ch) == 2 || throw(ArgumentError("HSS node must be binary"))
    lid, rid = node_id(ch[1]), node_id(ch[2])
    Lnd, Rnd = nodes_h[lid], nodes_h[rid]
    skl, skr = skel[lid], skel[rid]
    B12 = _hss_block(K, skl, skr, T)
    B21 = symm === :s ? Matrix{T}(transpose(B12)) : _hss_block(K, skr, skl, T)
    m = length(index_range(box))
    nl = Lnd.m
    slf = vcat(skl, skr)
    if isrt
        nd = HSSNode{T}(false, true, m, m, Z, Z, Z, Z, Z, Z, Z,
            B12, B21, Lnd, Rnd, nl)
        return nd, slf
    end
    kl = length(skl)
    nbr = pxyfun === nothing ? _hss_complement(slf, length(perm)) :
        _hss_near_skel(skel, neigh, box, slf)
    Kid = _hss_sample(K, slf, nbr, box, pxyfun, depth(box), T, symm)
    ns = length(slf)
    if size(Kid, 1) == 0 || ns < 2
        R = Matrix{T}(LinearAlgebra.I, ns, ns)
        Rl, Rr = R[1:kl, :], R[(kl + 1):end, :]
        nd = HSSNode{T}(false, false, m, m, Z, Z, Z, Rl, Rr, copy(Rl),
            copy(Rr), B12, B21, Lnd, Rnd, nl)
        return nd, slf
    end
    sk, rd, Tm = _hss_skel(Kid, method; rtol = rtol, rank = rank, Tmax = Tmax)
    R = _hss_interp(T, ns, sk, rd, Tm)
    Rl = R[1:kl, :]
    Rr = R[(kl + 1):end, :]
    nd = HSSNode{T}(false, false, m, m, Z, Z, Z, Rl, Rr, copy(Rl),
        copy(Rr), B12, B21, Lnd, Rnd, nl)
    return nd, slf[sk]
end

function _hss_proxy_pref(nd, npts, radius)
    if nd == 2
        θ = range(0.0, 2π; length = npts + 1)[1:(end - 1)]
        return [SVector(radius * cos(t), radius * sin(t)) for t in θ]
    elseif nd == 3
        pref = Vector{SVector{3, Float64}}(undef, npts)
        φ = π * (3 - sqrt(5))
        @inbounds for i in 0:(npts - 1)
            y = 1 - 2 * (i + 0.5) / npts
            rxy = sqrt(max(0.0, 1 - y^2))
            th = φ * i
            pref[i + 1] = radius * SVector(rxy * cos(th), y, rxy * sin(th))
        end
        return pref
    end
    throw(ArgumentError("HSS proxy needs 2D or 3D points"))
end

function _hss_geom_proxy(f, slf_pts, nbr_pts; npts::Int = 64, radius = 1.5, swap::Bool = false)
    nd = length(slf_pts[1])
    pref = _hss_proxy_pref(nd, npts, radius)
    r2 = float(radius)^2
    function pxyfun(slf::Vector{Int}, nbr::Vector{Int}, box::ClusterTree, ctr = nothing)
        isempty(slf) && return zeros(typeof(float(f(slf_pts[1], slf_pts[1]))), length(pref), 0), Int[]
        c = ctr === nothing ? center(container(box)) : ctr
        ℓ = high_corner(container(box)) - low_corner(container(box))
        T = typeof(float(f(slf_pts[slf[1]], slf_pts[slf[1]])))
        Kpxy = Matrix{T}(undef, length(pref), length(slf))
        @inbounds for (j, i) in enumerate(slf)
            yi = slf_pts[i]
            for (ii, p) in enumerate(pref)
                q = nd == 2 ? SVector(c[1] + p[1] * ℓ[1], c[2] + p[2] * ℓ[2]) :
                    SVector(c[1] + p[1] * ℓ[1], c[2] + p[2] * ℓ[2], c[3] + p[3] * ℓ[3])
                Kpxy[ii, j] = swap ? T(f(yi, q)) : T(f(q, yi))
            end
        end
        nbr2 = Int[]
        cc = nd == 2 ? SVector(c[1], c[2]) : SVector(c[1], c[2], c[3])
        @inbounds for i in nbr
            d = (nbr_pts[i] - cc) ./ ℓ
            sum(abs2, d) < r2 && push!(nbr2, i)
        end
        return Kpxy, nbr2
    end
    return pxyfun
end

function _hss_default_pxy_rect(K, rowtree, coltree)
    K isa KernelMatrix || return nothing, nothing
    f = kernel(K)
    X = rowelements(K)
    Y = colelements(K)
    return _hss_geom_proxy(f, X, Y; swap = true), _hss_geom_proxy(f, Y, X; swap = false)
end

function _hss_near_pts(node, perm, neigh, id2)
    out = Int[]
    seen = Set{Int}()
    P = parentnode(node)
    if P !== node
        for S in children(P)
            S === node && continue
            for i in perm[index_range(S)]
                i in seen && continue
                push!(out, i)
                push!(seen, i)
            end
        end
    end
    id = node_id(node)
    if 1 <= id <= length(neigh)
        for nid in neigh[id]
            nd = id2[nid]
            for i in perm[index_range(nd)]
                i in seen && continue
                push!(out, i)
                push!(seen, i)
            end
        end
    end
    return out
end

function _hss_sample_U(K, I, nbrJ, rbox, pxyU, depth_, ::Type{T}) where {T}
    Kpxy = zeros(T, 0, length(I))
    nbr = nbrJ
    if depth_ >= 2 && pxyU !== nothing && !isempty(I)
        Kpxy, nbr = pxyU(I, nbr, rbox)
    end
    At = isempty(nbr) ? zeros(T, 0, length(I)) : transpose(_hss_block(K, I, nbr, T))
    return isempty(Kpxy) ? At : vcat(At, Kpxy)
end

function _hss_sample_V(K, J, nbrI, cbox, pxyV, depth_, ::Type{T}) where {T}
    Kpxy = zeros(T, 0, length(J))
    nbr = nbrI
    if depth_ >= 2 && pxyV !== nothing && !isempty(J)
        Kpxy, nbr = pxyV(J, nbr, cbox)
    end
    Anbr = isempty(nbr) ? zeros(T, 0, length(J)) : _hss_block(K, nbr, J, T)
    return isempty(Kpxy) ? Anbr : vcat(Anbr, Kpxy)
end

function _hss_compress_side(Kid, mloc, method, rtol, rank, Tmax, ::Type{T}) where {T}
    if size(Kid, 1) == 0 || mloc < 2
        return Matrix{T}(LinearAlgebra.I, mloc, mloc), collect(1:mloc)
    end
    sk, rd, Tm = _hss_skel(Kid, method; rtol = rtol, rank = rank, Tmax = Tmax)
    U = _hss_interp(T, mloc, sk, rd, Tm)
    return U, sk
end

function _hss_build_rect(rt, ct, K, rperm, cperm, rtol, rank, Tmax, pxyU, pxyV,
        method, isroot, ::Type{T}, neighr = nothing, id2r = nothing,
        neighc = nothing, id2c = nothing) where {T}
    if neighr === nothing
        node_id(rt) == 0 && assign_node_ids!(rt)
        node_id(ct) == 0 && assign_node_ids!(ct)
        neighr, _, id2r = neighbor_il_lists(rt)
        neighc, _, id2c = neighbor_il_lists(ct)
    end
    rch = children(rt)
    cch = children(ct)
    I0 = rperm[collect(index_range(rt))]
    J0 = cperm[collect(index_range(ct))]
    m = length(I0)
    n = length(J0)
    Z = _hss_Z(T)
    leaf = isempty(rch) || isempty(cch) || length(rch) != 2 || length(cch) != 2
    if leaf
        D = _hss_block(K, I0, J0, T)
        if isroot
            return HSSNode{T}(true, true, m, n, D, zeros(T, m, 0), zeros(T, n, 0),
                Z, Z, Z, Z, Z, Z, nothing, nothing, m)
        end
        if pxyU === nothing
            nbrJ = _hss_complement(J0, length(cperm))
            nbrI = _hss_complement(I0, length(rperm))
        else
            nbrJ = _hss_near_pts(ct, cperm, neighc, id2c)
            nbrI = _hss_near_pts(rt, rperm, neighr, id2r)
        end
        KidU = _hss_sample_U(K, I0, nbrJ, rt, pxyU, depth(rt), T)
        KidV = _hss_sample_V(K, J0, nbrI, ct, pxyV, depth(ct), T)
        U, sku = _hss_compress_side(KidU, m, method, rtol, rank, Tmax, T)
        V, skv = _hss_compress_side(KidV, n, method, rtol, rank, Tmax, T)
        return HSSNode{T}(true, false, m, n, D, U, V, Z, Z, Z, Z, Z, Z,
            nothing, nothing, m), I0[sku], J0[skv]
    end
    Lnd, skIl, skJl = _hss_build_rect(rch[1], cch[1], K, rperm, cperm, rtol, rank,
        Tmax, pxyU, pxyV, method, false, T, neighr, id2r, neighc, id2c)
    Rnd, skIr, skJr = _hss_build_rect(rch[2], cch[2], K, rperm, cperm, rtol, rank,
        Tmax, pxyU, pxyV, method, false, T, neighr, id2r, neighc, id2c)
    B12 = _hss_block(K, skIl, skJr, T)
    B21 = _hss_block(K, skIr, skJl, T)
    if isroot
        return HSSNode{T}(false, true, m, n, Z, Z, Z, Z, Z, Z, Z, B12, B21,
            Lnd, Rnd, Lnd.m)
    end
    Isk = vcat(skIl, skIr)
    Jsk = vcat(skJl, skJr)
    kl = length(skIl)
    kv = length(skJl)
    if pxyU === nothing
        nbrJ = _hss_complement(J0, length(cperm))
        nbrI = _hss_complement(I0, length(rperm))
    else
        nbrJ = _hss_near_pts(ct, cperm, neighc, id2c)
        nbrI = _hss_near_pts(rt, rperm, neighr, id2r)
    end
    KidU = _hss_sample_U(K, Isk, nbrJ, rt, pxyU, depth(rt), T)
    KidV = _hss_sample_V(K, Jsk, nbrI, ct, pxyV, depth(ct), T)
    Rfull, sku = _hss_compress_side(KidU, length(Isk), method, rtol, rank, Tmax, T)
    Wfull, skv = _hss_compress_side(KidV, length(Jsk), method, rtol, rank, Tmax, T)
    Rl = Rfull[1:kl, :]
    Rr = Rfull[(kl + 1):end, :]
    Wl = Wfull[1:kv, :]
    Wr = Wfull[(kv + 1):end, :]
    nd = HSSNode{T}(false, false, m, n, Z, Z, Z, Rl, Rr, Wl, Wr, B12, B21,
        Lnd, Rnd, Lnd.m)
    return nd, Isk[sku], Jsk[skv]
end

# Dense-complement SVD (naive Xia). Kept as method=:svd.
function _hss_svd(A::AbstractMatrix)
    s = _HSS_STATS[]
    t0 = s === nothing ? UInt64(0) : time_ns()
    F = svd(A; full = false)
    if s !== nothing
        s.t_svd += 1e-9 * (time_ns() - t0)
        s.n_svd += 1
    end
    return F
end

function _hss_row_basis(B::AbstractMatrix{T}; rtol, rank) where {T}
    m, n = size(B)
    m == 0 && return zeros(T, 0, 0)
    n == 0 && return Matrix{T}(I, m, m)
    F = _hss_svd(B)
    isempty(F.S) && return zeros(T, m, 0)
    τ = float(rtol) * F.S[1]
    k = count(s -> s > τ, F.S)
    k = clamp(k, 1, min(m, n, Int(rank), length(F.S)))
    return F.U[:, 1:k]
end

function _hss_B(Ul::AbstractMatrix{T}, A::AbstractMatrix{T}, Vr::AbstractMatrix{T}) where {T}
    (size(Ul, 2) == 0 || size(Vr, 2) == 0) &&
        return zeros(T, size(Ul, 2), size(Vr, 2))
    s = _HSS_STATS[]
    t0 = s === nothing ? UInt64(0) : time_ns()
    B = pinv(Ul) * A * pinv(transpose(Vr))
    if s !== nothing
        s.t_pinv += 1e-9 * (time_ns() - t0)
    end
    return B
end

function _hss_lsq(U::AbstractMatrix{T}, A::AbstractMatrix{T}) where {T}
    size(U, 2) == 0 && return zeros(T, 0, size(A, 2))
    s = _HSS_STATS[]
    t0 = s === nothing ? UInt64(0) : time_ns()
    G = U \ A
    if s !== nothing
        s.t_lsq += 1e-9 * (time_ns() - t0)
    end
    return G
end

function _hss_iglob(perm, I)
    return perm[I]
end

function _hss_complement(I::Vector{Int}, n::Int)
    inI = falses(n)
    @inbounds for i in I
        inI[i] = true
    end
    J = Vector{Int}(undef, n - length(I))
    t = 0
    @inbounds for i in 1:n
        inI[i] && continue
        t += 1
        J[t] = i
    end
    return J
end

function _hss_build_svd(node, K, perm, n, isroot, rtol, rank, ::Type{T}) where {T}
    I = collect(index_range(node))
    m = length(I)
    J = _hss_complement(I, n)
    ch = children(node)
    if isempty(ch)
        D = _hss_block(K, _hss_iglob(perm, I), _hss_iglob(perm, I), T)
        if isempty(J)
            nd = HSSNode{T}(true, isroot, m, m, D, zeros(T, m, 0), zeros(T, m, 0),
                zeros(T, 0, 0), zeros(T, 0, 0), zeros(T, 0, 0), zeros(T, 0, 0),
                zeros(T, 0, 0), zeros(T, 0, 0), nothing, nothing, m)
            return nd, zeros(T, m, 0), zeros(T, m, 0)
        end
        Brow = _hss_block(K, _hss_iglob(perm, I), _hss_iglob(perm, J), T)
        Fb = _hss_svd(Brow)
        k = 0
        if !isempty(Fb.S)
            τ = float(rtol) * Fb.S[1]
            k = clamp(count(s -> s > τ, Fb.S), 1,
                min(m, Int(rank), length(Fb.S)))
        end
        U = k == 0 ? zeros(T, m, 0) : Fb.U[:, 1:k]
        V = copy(U)
        nd = HSSNode{T}(true, isroot, m, m, D, U, V,
            zeros(T, 0, 0), zeros(T, 0, 0), zeros(T, 0, 0), zeros(T, 0, 0),
            zeros(T, 0, 0), zeros(T, 0, 0), nothing, nothing, m)
        return nd, U, V
    end
    length(ch) == 2 || throw(ArgumentError("HSS node must be binary"))
    nl, Ul, Vl = _hss_build_svd(ch[1], K, perm, n, false, rtol, rank, T)
    nr, Ur, Vr = _hss_build_svd(ch[2], K, perm, n, false, rtol, rank, T)
    Il = collect(index_range(ch[1]))
    Ir = collect(index_range(ch[2]))
    A12 = _hss_block(K, _hss_iglob(perm, Il), _hss_iglob(perm, Ir), T)
    A21 = _hss_block(K, _hss_iglob(perm, Ir), _hss_iglob(perm, Il), T)
    B12 = _hss_B(Ul, A12, Vr)
    B21 = _hss_B(Ur, A21, Vl)
    if isempty(J)
        nd = HSSNode{T}(false, true, m, m, zeros(T, 0, 0), zeros(T, 0, 0), zeros(T, 0, 0),
            zeros(T, 0, 0), zeros(T, 0, 0), zeros(T, 0, 0), zeros(T, 0, 0),
            B12, B21, nl, nr, nl.m)
        return nd, zeros(T, m, 0), zeros(T, m, 0)
    end
    A_lJ = _hss_block(K, _hss_iglob(perm, Il), _hss_iglob(perm, J), T)
    A_rJ = _hss_block(K, _hss_iglob(perm, Ir), _hss_iglob(perm, J), T)
    Gl = _hss_lsq(Ul, A_lJ)
    Gr = _hss_lsq(Ur, A_rJ)
    G = vcat(Gl, Gr)
    Rfull = _hss_row_basis(G; rtol = rtol, rank = rank)
    kl = size(Ul, 2)
    Rl = Rfull[1:kl, :]
    Rr = Rfull[(kl + 1):end, :]
    Wl = copy(Rl)
    Wr = copy(Rr)
    U = vcat(Ul * Rl, Ur * Rr)
    V = vcat(Vl * Wl, Vr * Wr)
    nd = HSSNode{T}(false, isroot, m, m, zeros(T, 0, 0), zeros(T, 0, 0), zeros(T, 0, 0),
        Rl, Rr, Wl, Wr, B12, B21, nl, nr, nl.m)
    return nd, U, V
end

function _hss_build_svd_rect(rt, ct, K, rperm, cperm, mtot, ntot, isroot, rtol, rank,
        ::Type{T}) where {T}
    I = collect(index_range(rt))
    J = collect(index_range(ct))
    mi, nj = length(I), length(J)
    Ic = _hss_complement(I, mtot)
    Jc = _hss_complement(J, ntot)
    rch, cch = children(rt), children(ct)
    Z = _hss_Z(T)
    D = _hss_block(K, rperm[I], cperm[J], T)
    leaf = isempty(rch) || isempty(cch) || length(rch) != 2 || length(cch) != 2
    if leaf
        U = isempty(Jc) ? zeros(T, mi, 0) :
            _hss_row_basis(_hss_block(K, rperm[I], cperm[Jc], T); rtol = rtol, rank = rank)
        V = isempty(Ic) ? zeros(T, nj, 0) :
            _hss_row_basis(transpose(_hss_block(K, rperm[Ic], cperm[J], T));
                rtol = rtol, rank = rank)
        nd = HSSNode{T}(true, isroot, mi, nj, D, U, V, Z, Z, Z, Z, Z, Z,
            nothing, nothing, mi)
        return nd, U, V
    end
    Lnd, Ul, Vl = _hss_build_svd_rect(rch[1], cch[1], K, rperm, cperm, mtot, ntot,
        false, rtol, rank, T)
    Rnd, Ur, Vr = _hss_build_svd_rect(rch[2], cch[2], K, rperm, cperm, mtot, ntot,
        false, rtol, rank, T)
    Il, Ir = collect(index_range(rch[1])), collect(index_range(rch[2]))
    Jl, Jr = collect(index_range(cch[1])), collect(index_range(cch[2]))
    B12 = _hss_B(Ul, _hss_block(K, rperm[Il], cperm[Jr], T), Vr)
    B21 = _hss_B(Ur, _hss_block(K, rperm[Ir], cperm[Jl], T), Vl)
    if isroot || (isempty(Jc) && isempty(Ic))
        nd = HSSNode{T}(false, true, mi, nj, Z, Z, Z, Z, Z, Z, Z, B12, B21,
            Lnd, Rnd, Lnd.m)
        return nd, zeros(T, mi, 0), zeros(T, nj, 0)
    end
    Gl = _hss_lsq(Ul, _hss_block(K, rperm[Il], cperm[Jc], T))
    Gr = _hss_lsq(Ur, _hss_block(K, rperm[Ir], cperm[Jc], T))
    Rfull = _hss_row_basis(vcat(Gl, Gr); rtol = rtol, rank = rank)
    kl = size(Ul, 2)
    Rl = size(Rfull, 1) >= kl ? Rfull[1:kl, :] : zeros(T, kl, size(Rfull, 2))
    Rr = size(Rfull, 1) > kl ? Rfull[(kl + 1):end, :] :
        zeros(T, size(Ur, 2), size(Rfull, 2))
    Cl = size(Vl, 2) == 0 ? zeros(T, length(Ic), 0) :
        transpose(_hss_lsq(Vl, transpose(_hss_block(K, rperm[Ic], cperm[Jl], T))))
    Cr = size(Vr, 2) == 0 ? zeros(T, length(Ic), 0) :
        transpose(_hss_lsq(Vr, transpose(_hss_block(K, rperm[Ic], cperm[Jr], T))))
    C = hcat(Cl, Cr)
    if size(C, 1) == 0 || size(C, 2) == 0
        Wl = zeros(T, size(Vl, 2), size(Rl, 2))
        Wr = zeros(T, size(Vr, 2), size(Rl, 2))
    else
        Fw = _hss_svd(C)
        kv = 0
        if !isempty(Fw.S)
            τ = float(rtol) * Fw.S[1]
            kv = clamp(count(s -> s > τ, Fw.S), 1,
                min(size(C, 2), Int(rank), length(Fw.S)))
        end
        W = kv == 0 ? zeros(T, size(C, 2), 0) : Fw.V[:, 1:kv]
        kvl = size(Vl, 2)
        Wl = W[1:kvl, :]
        Wr = W[(kvl + 1):end, :]
    end
    U = vcat(Ul * Rl, Ur * Rr)
    V = vcat(size(Wl, 2) == 0 ? zeros(T, size(Vl, 1), 0) : Vl * Wl,
        size(Wr, 2) == 0 ? zeros(T, size(Vr, 1), 0) : Vr * Wr)
    nd = HSSNode{T}(false, isroot, mi, nj, Z, Z, Z, Rl, Rr, Wl, Wr, B12, B21,
        Lnd, Rnd, Lnd.m)
    return nd, U, V
end

function _hss_up!(N::HSSNode{T}, x::AbstractVector{T}) where {T}
    if N.leaf
        z = N.V' * x
        return (z = z, l = nothing, r = nothing)
    end
    nc = N.left.n
    gl = _hss_up!(N.left, view(x, 1:nc))
    gr = _hss_up!(N.right, view(x, (nc + 1):N.n))
    z = N.root ? similar(gl.z, 0) : (N.Wl' * gl.z .+ N.Wr' * gr.z)
    return (z = z, l = gl, r = gr)
end

function _hss_down!(N::HSSNode{T}, x::AbstractVector{T}, g, f::AbstractVector{T}) where {T}
    if N.leaf
        y = N.D * x
        if !isempty(f) && !isempty(N.U)
            y .+= N.U * f
        end
        return y
    end
    if N.root
        fl = N.B12 * g.r.z
        fr = N.B21 * g.l.z
    else
        fl = N.B12 * g.r.z + N.Rl * f
        fr = N.B21 * g.l.z + N.Rr * f
    end
    nc = N.left.n
    yl = _hss_down!(N.left, view(x, 1:nc), g.l, fl)
    yr = _hss_down!(N.right, view(x, (nc + 1):N.n), g.r, fr)
    return vcat(yl, yr)
end

function _hss_mul(N::HSSNode{T}, x::AbstractVector{T}) where {T}
    N.leaf && return N.D * x
    g = _hss_up!(N, x)
    return _hss_down!(N, x, g, T[])
end

function LinearAlgebra.mul!(y::AbstractVector, A::HSSMatrix{T}, x::AbstractVector;
        global_index = use_global_index()) where {T}
    length(x) == A.n && length(y) == A.m || throw(DimensionMismatch())
    xl = global_index ? x[A.cperm] : x
    yl = _hss_mul(A.root, xl)
    if global_index
        y[A.perm] = yl
    else
        copyto!(y, yl)
    end
    return y
end

function LinearAlgebra.mul!(y::AbstractVector, A::HSSMatrix, x::AbstractVector,
        a::Number, b::Number)
    tmp = similar(y)
    mul!(tmp, A, x)
    if iszero(b)
        y .= a .* tmp
    else
        y .= b .* y .+ a .* tmp
    end
    return y
end

Base.:*(A::HSSMatrix, x::AbstractVector) = mul!(Vector{eltype(A)}(undef, A.m), A, x)

function _hss_mul_mat(A::HSSMatrix{T}, X::AbstractMatrix) where {T}
    size(X, 1) == A.n || throw(DimensionMismatch())
    Y = Matrix{T}(undef, A.m, size(X, 2))
    @inbounds for j in 1:size(X, 2)
        mul!(view(Y, :, j), A, view(X, :, j))
    end
    return Y
end

"""
    HSSBDC

Lazy `B * (D \\ C)` with rectangular HSS `B`, `C` and [`ULVFactor`](@ref) of `D`.
Used only to assemble the product as square HSS via [`assemble_hss_BDC`](@ref).
"""
struct HSSBDC{T, TB, TF, TC} <: AbstractMatrix{T}
    B::TB
    F::TF
    C::TC
end
function HSSBDC(B::HSSMatrix{T}, F, C::HSSMatrix{T}) where {T}
    size(B, 2) == size(F, 1) == size(C, 1) || throw(DimensionMismatch(
        "B D^{-1} C needs size(B,2)=size(D)=size(C,1); got $(size(B)), $(size(F)), $(size(C))"))
    size(F, 1) == size(F, 2) || throw(DimensionMismatch("D must be square"))
    return HSSBDC{T, typeof(B), typeof(F), typeof(C)}(B, F, C)
end

Base.size(P::HSSBDC) = (size(P.B, 1), size(P.C, 2))
Base.eltype(::HSSBDC{T}) where {T} = T

function Base.getindex(P::HSSBDC{T}, i::Int, j::Int) where {T}
    e = zeros(T, size(P.C, 2))
    e[j] = one(T)
    return (P.B * (P.F \ (P.C * e)))[i]
end

function _kernel_block!(out::AbstractMatrix{T}, P::HSSBDC{T}, I::Vector{Int},
        J::Vector{Int}) where {T}
    m, n = length(I), length(J)
    size(out, 1) == m && size(out, 2) == n || throw(DimensionMismatch())
    (m == 0 || n == 0) && return out
    E = zeros(T, size(P.C, 2), n)
    @inbounds for k in 1:n
        E[J[k], k] = one(T)
    end
    Y = _hss_mul_mat(P.C, E)
    @inbounds for k in 1:n
        Y[:, k] = P.F \ view(Y, :, k)
    end
    Z = _hss_mul_mat(P.B, Y)
    @inbounds for k in 1:n, t in 1:m
        out[t, k] = Z[I[t], k]
    end
    return out
end

"""
    assemble_hss_BDC(B, F, C, tree; kwargs...) -> HSSMatrix

Assemble the square HSS of `P = B * (D \\ C)` on the **u-tree** (`tree` matches
the row partition of `B` / column partition of `C`). `F = ulv(D)`. Blocks of
`P` are filled by HSS matvecs and ULV, never by forming dense `P`.
"""
function assemble_hss_BDC(B::HSSMatrix, F, C::HSSMatrix, tree::ClusterTree;
        kwargs...)
    length(tree) == size(B, 1) == size(C, 2) || throw(DimensionMismatch(
        "u-tree length $(length(tree)) ≠ size(B,1)=$(size(B,1)) or size(C,2)=$(size(C,2))"))
    return assemble_hss(HSSBDC(B, F, C), tree; kwargs...)
end

"""
    HSSSchur

Lazy `A - B*(D \\ C)` on the u-tree. `A` is square HSS; `B`,`C` rectangular HSS;
`F = ulv(D)`.
"""
struct HSSSchur{T, TA, TB, TF, TC} <: AbstractMatrix{T}
    A::TA
    B::TB
    F::TF
    C::TC
end
function HSSSchur(A::AbstractMatrix{T}, B::HSSMatrix{T}, F, C::HSSMatrix{T}) where {T}
    size(A, 1) == size(A, 2) == size(B, 1) == size(C, 2) || throw(DimensionMismatch(
        "Schur A-B D\\C needs square A matching B rows and C cols"))
    size(B, 2) == size(F, 1) == size(C, 1) || throw(DimensionMismatch(
        "B D\\C inner sizes"))
    return HSSSchur{T, typeof(A), typeof(B), typeof(F), typeof(C)}(A, B, F, C)
end

Base.size(S::HSSSchur) = size(S.A)
Base.eltype(::HSSSchur{T}) where {T} = T

function Base.getindex(S::HSSSchur{T}, i::Int, j::Int) where {T}
    e = zeros(T, size(S.A, 2))
    e[j] = one(T)
    return ((S.A * e) - S.B * (S.F \ (S.C * e)))[i]
end

function _kernel_block!(out::AbstractMatrix{T}, S::HSSSchur{T}, I::Vector{Int},
        J::Vector{Int}) where {T}
    m, n = length(I), length(J)
    size(out, 1) == m && size(out, 2) == n || throw(DimensionMismatch())
    (m == 0 || n == 0) && return out
    E = zeros(T, size(S.C, 2), n)
    @inbounds for k in 1:n
        E[J[k], k] = one(T)
    end
    Y = _hss_mul_mat(S.C, E)
    @inbounds for k in 1:n
        Y[:, k] = S.F \ view(Y, :, k)
    end
    Z = _hss_mul_mat(S.B, Y)
    if S.A isa HSSMatrix
        AZ = _hss_mul_mat(S.A, E)
        @inbounds for k in 1:n, t in 1:m
            out[t, k] = AZ[I[t], k] - Z[I[t], k]
        end
    else
        @inbounds for k in 1:n, t in 1:m
            out[t, k] = S.A[I[t], J[k]] - Z[I[t], k]
        end
    end
    return out
end

"""
    assemble_hss_schur(A, B, F, C, tree; kwargs...) -> HSSMatrix

Square HSS of `S = A - B*(D \\ C)` on the u-tree. Factor with [`ulv`](@ref).
"""
function assemble_hss_schur(A::AbstractMatrix, B::HSSMatrix, F, C::HSSMatrix,
        tree::ClusterTree; kwargs...)
    length(tree) == size(A, 1) || throw(DimensionMismatch(
        "u-tree length $(length(tree)) ≠ size(A,1)=$(size(A,1))"))
    return assemble_hss(HSSSchur(A, B, F, C), tree; kwargs...)
end

"""
    HSSBlock2x2

Lazy square `[A B; C D]` from four HSS tiles (`A`,`D` square, `B`,`C` rectangular).
"""
struct HSSBlock2x2{T, TA, TB, TC, TD} <: AbstractMatrix{T}
    A::TA
    B::TB
    C::TC
    D::TD
    nu::Int
    nq::Int
end
function HSSBlock2x2(A::AbstractMatrix{T}, B::AbstractMatrix{T}, C::AbstractMatrix{T},
        D::AbstractMatrix{T}) where {T}
    nu, nq = size(A, 1), size(D, 1)
    size(A) == (nu, nu) && size(D) == (nq, nq) || throw(DimensionMismatch("A,D square"))
    size(B) == (nu, nq) && size(C) == (nq, nu) || throw(DimensionMismatch("B,C shape"))
    return HSSBlock2x2{T, typeof(A), typeof(B), typeof(C), typeof(D)}(A, B, C, D, nu, nq)
end

Base.size(M::HSSBlock2x2) = (M.nu + M.nq, M.nu + M.nq)
Base.eltype(::HSSBlock2x2{T}) where {T} = T

function LinearAlgebra.mul!(y::AbstractVector, M::HSSBlock2x2{T},
        x::AbstractVector) where {T}
    length(x) == M.nu + M.nq && length(y) == length(x) || throw(DimensionMismatch())
    x1 = view(x, 1:M.nu)
    x2 = view(x, (M.nu + 1):(M.nu + M.nq))
    y1 = view(y, 1:M.nu)
    y2 = view(y, (M.nu + 1):(M.nu + M.nq))
    mul!(y1, M.A, x1)
    mul!(y1, M.B, x2, one(T), one(T))
    mul!(y2, M.C, x1)
    mul!(y2, M.D, x2, one(T), one(T))
    return y
end

function Base.getindex(M::HSSBlock2x2{T}, i::Int, j::Int) where {T}
    e = zeros(T, size(M, 2))
    e[j] = one(T)
    y = similar(e)
    mul!(y, M, e)
    return y[i]
end

function _kernel_block!(out::AbstractMatrix{T}, M::HSSBlock2x2{T}, I::Vector{Int},
        J::Vector{Int}) where {T}
    m, n = length(I), length(J)
    size(out, 1) == m && size(out, 2) == n || throw(DimensionMismatch())
    (m == 0 || n == 0) && return out
    E = zeros(T, M.nu + M.nq, n)
    @inbounds for k in 1:n
        E[J[k], k] = one(T)
    end
    Z = Matrix{T}(undef, M.nu + M.nq, n)
    @inbounds for k in 1:n
        mul!(view(Z, :, k), M, view(E, :, k))
    end
    @inbounds for k in 1:n, t in 1:m
        out[t, k] = Z[I[t], k]
    end
    return out
end

"""
    assemble_hss_2x2(A, B, C, D, tree; kwargs...) -> HSSMatrix

One square HSS for `[A B; C D]`. Factor with [`ulv`](@ref).
"""
function assemble_hss_2x2(A::AbstractMatrix, B::AbstractMatrix, C::AbstractMatrix,
        D::AbstractMatrix, tree::ClusterTree; kwargs...)
    M = HSSBlock2x2(A, B, C, D)
    length(tree) == size(M, 1) || throw(DimensionMismatch(
        "tree length $(length(tree)) ≠ packed size $(size(M, 1))"))
    return assemble_hss(M, tree; kwargs...)
end

function ulv(A::AbstractMatrix, B::AbstractMatrix, C::AbstractMatrix,
        D::AbstractMatrix, tree::ClusterTree; kwargs...)
    return ulv(assemble_hss_2x2(A, B, C, D, tree; kwargs...))
end
