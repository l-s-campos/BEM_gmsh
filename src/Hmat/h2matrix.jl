# =============================================================================
# H² matrix with Proxy Point Method (following H2Pack-Matlab)
# =============================================================================

"""
    struct H2BoxAdmissibility

Box-based admissibility used by H2Pack: two clusters are admissible when they
are well-separated relative to `alpha` times the box edge length
(``L_\\infty``-style separation on axis-aligned boxes).
"""
Base.@kwdef struct H2BoxAdmissibility
    alpha::Float64 = 1.0
end

function (adm::H2BoxAdmissibility)(a::ClusterTree, b::ClusterTree)
    return _boxes_admissible(container(a), container(b), adm.alpha)
end

# Expanded (block-DOF) trees: admissibility is purely geometric
function (adm::H2BoxAdmissibility)(a::ExpandedClusterTree, b::ExpandedClusterTree)
    return _boxes_admissible(container(a), container(b), adm.alpha)
end
function (adm::H2BoxAdmissibility)(a::ExpandedClusterTree, b::ClusterTree)
    return _boxes_admissible(container(a), container(b), adm.alpha)
end
function (adm::H2BoxAdmissibility)(a::ClusterTree, b::ExpandedClusterTree)
    return _boxes_admissible(container(a), container(b), adm.alpha)
end

function _boxes_admissible(box1::HyperRectangle{N}, box2::HyperRectangle{N}, α::Float64) where {N}
    # Port of H2Pack isadmissible_box (L∞ separation with scale α)
    c1 = low_corner(box1)
    L1 = high_corner(box1) .- low_corner(box1)
    c2 = low_corner(box2)
    L2 = high_corner(box2) .- low_corner(box2)
    for i in 1:N
        if L1[i] < L2[i]
            dist = c1[i] - c2[i]
            if dist < 0
                abs(dist) >= (1 + α - 1e-8) * L1[i] && return true
            else
                dist >= (α - 1e-8) * L1[i] + L2[i] && return true
            end
        else
            dist = c2[i] - c1[i]
            if dist < 0
                abs(dist) >= (1 + α - 1e-8) * L2[i] && return true
            else
                dist >= (α - 1e-8) * L2[i] + L1[i] && return true
            end
        end
    end
    return false
end

# =============================================================================
# Proxy points
# =============================================================================

"""
    proxy_surface_unit(dim, nsample) -> Vector{SVector{dim,Float64}}

Uniform-ish samples on the surface of the cube `[-1,1]^dim` (H2Pack
`gridpoint_on_boxsurface`).
"""
function proxy_surface_unit(dim::Int, nsample::Int)
    dim == 2 && return _proxy_surface_2d(nsample)
    dim == 3 && return _proxy_surface_3d(nsample)
    # fallback: random on expanded sphere
    return [SVector{dim, Float64}(ntuple(k -> 2 * rand() - 1, dim)) for _ in 1:nsample]
end

function _proxy_surface_2d(nsample::Int)
    Nx = max(2, ceil(Int, 0.5 * nsample / 2))
    Ny = Nx
    intX = (1:Nx) ./ (Nx + 1)
    intY = (1:Ny) ./ (Ny + 1)
    pts = SVector{2, Float64}[]
    # bottom / top (y = ±1), x in (-1,1)
    for x in intX
        push!(pts, SVector(2x - 1, -1.0))
        push!(pts, SVector(2x - 1, 1.0))
    end
    # left / right (x = ±1), y in (-1,1)
    for y in intY
        push!(pts, SVector(-1.0, 2y - 1))
        push!(pts, SVector(1.0, 2y - 1))
    end
    return pts
end

function _proxy_surface_3d(nsample::Int)
    # allocate proportionally to face areas of unit cube
    n_face = max(4, ceil(Int, nsample / 6))
    n1 = max(2, ceil(Int, sqrt(n_face)))
    n2 = n1
    int1 = (1:n1) ./ (n1 + 1)
    int2 = (1:n2) ./ (n2 + 1)
    pts = SVector{3, Float64}[]
    faces = (
        (1, 0, 0, -1.0), (1, 0, 0, 1.0),
        (0, 1, 0, -1.0), (0, 1, 0, 1.0),
        (0, 0, 1, -1.0), (0, 0, 1, 1.0),
    )
    # simpler: six faces of [-1,1]^3
    for s in (-1.0, 1.0)
        for u in int1, v in int2
            push!(pts, SVector(s, 2u - 1, 2v - 1))
            push!(pts, SVector(2u - 1, s, 2v - 1))
            push!(pts, SVector(2u - 1, 2v - 1, s))
        end
    end
    return pts
end

"""
    scale_proxy_to_box(Yp_unit, box, alpha) -> Vector{SVector}

Place unit-cube surface proxies around `box` at radius `(1+2α)·half_edge`
(H2Pack surface proxy scaling).
"""
function scale_proxy_to_box(
        Yp_unit::Vector{SVector{N, Float64}},
        box::HyperRectangle{N},
        alpha::Float64,
    ) where {N}
    ctr = center(box)
    half = maximum(high_corner(box) .- low_corner(box)) / 2
    L2 = (1 + 2 * max(alpha, 1e-3)) * half
    return [ctr .+ L2 .* y for y in Yp_unit]
end

"""
    proxy_points_per_level(tree, alpha; nsample=200)

Proxy surface point sets for each tree depth (index `d` = depth from root,
depth 0 unused). Same geometry at a given level for all boxes of equal size.
"""
proxy_points_per_level(tree::ExpandedClusterTree, alpha::Float64; nsample::Int=200) =
    proxy_points_per_level(tree.tree, alpha; nsample=nsample)

function proxy_points_per_level(
        tree::ClusterTree{N, T},
        alpha::Float64;
        nsample::Int = 200,
    ) where {N, T}
    maxd = maximum(depth, leaves(tree))
    Yp_unit = proxy_surface_unit(N, nsample)
    # representative box half-length per depth via root split count
    Yp = Vector{Vector{SVector{N, Float64}}}(undef, maxd + 1)
    Yp[1] = SVector{N, Float64}[]  # root level unused
    # walk one path to leaves collecting boxes
    node = tree
    boxes_by_depth = Vector{HyperRectangle{N, T}}(undef, maxd + 1)
    boxes_by_depth[1] = container(tree)
    d = 0
    while !isleaf(node) && d < maxd
        d += 1
        node = children(node)[1]
        boxes_by_depth[d + 1] = container(node)
    end
    for d in 0:maxd
        box = d + 1 <= length(boxes_by_depth) ? boxes_by_depth[d + 1] : container(tree)
        # unit proxies scaled around origin-sized box of this level, then used
        # relative to each node's center at assembly time
        half = maximum(high_corner(box) .- low_corner(box)) / 2
        L2 = (1 + 2 * max(alpha, 1e-3)) * Float64(half)
        Yp[d + 1] = [L2 .* y for y in Yp_unit]
    end
    return Yp
end

# =============================================================================
# Flattened cluster tree
# =============================================================================

"""Flat view of a [`ClusterTree`](@ref) for H² node indexing (1-based)."""
struct H2TreeIndex{R}
    nodes::Vector{R}
    parent::Vector{Int}
    children::Vector{Vector{Int}}
    levels::Vector{Vector{Int}}   # levels[1] = root, levels[end] = deepest
    leafnodes::Vector{Int}
    depth_of::Vector{Int}         # depth_of[i] = 0 for root
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
        for c in HMatrices.children(n)
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

# =============================================================================
# H2Matrix
# =============================================================================

"""
    mutable struct H2Matrix{R,T} <: AbstractStructuredMatrix{T}

``\\mathcal{H}^2`` matrix with nested bases from the **proxy point method**
(H2Pack-Matlab `Mat2H2_ID_Proxy`).

# Generators
- `U[i]`: nested basis / translator at cluster node `i` (`m×r` at leaves,
  `(∑ r_child)×r` at non-leaves)
- `skeleton[i]`: skeleton index set (absolute local ordering)
- `B[(i,j)]`: far-field coupling (skeleton–skeleton)
- `Ddiag` / `Dnear`: **always dense** near-field blocks
- `Bfar`: far couplings — dense `Matrix` or low-rank [`RkMatrix`](@ref)
  when assembled with `far_method=:aca`

Build with [`assemble_h2`](@ref). Matvec uses the classical up / intermediate /
down sweeps of H2Pack.
"""
mutable struct H2Matrix{R, T} <: AbstractStructuredMatrix{T}
    tidx::H2TreeIndex{R}
    U::Vector{Matrix{T}}
    skeleton::Vector{Vector{Int}}
    near::Vector{Tuple{Int, Int}}
    far::Vector{Tuple{Int, Int}}
    Ddiag::Dict{Int, Matrix{T}}
    Dnear::Dict{Tuple{Int, Int}, Matrix{T}}
    Bfar::Dict{Tuple{Int, Int}, Any}   # Matrix{T} or RkMatrix{T}
    rowperm::Vector{Int}
    colperm::Vector{Int}
    minlvl::Int          # 1-based level index (into tidx.levels)
    alpha::Float64
    n::Int
end

Base.size(H::H2Matrix) = (H.n, H.n)
Base.eltype(::H2Matrix{<:Any, T}) where {T} = T
rowperm(H::H2Matrix) = H.rowperm
colperm(H::H2Matrix) = H.colperm

function Base.show(io::IO, H::H2Matrix)
    nf, nn = length(H.far), length(H.near)
    return print(
        io,
        "H2Matrix{$(eltype(H))} of size $(H.n)×$(H.n) ",
        "($(nf) far, $(nn) near blocks, maxrank=$(maxrank(H)))",
    )
end
Base.show(io::IO, ::MIME"text/plain", H::H2Matrix) = show(io, H)

function maxrank(H::H2Matrix)
    r = 0
    for U in H.U
        r = max(r, size(U, 2))
    end
    return r
end

function compression_ratio(H::H2Matrix)
    return (length(H) * sizeof(eltype(H))) / Base.summarysize(H)
end

# =============================================================================
# Assembly
# =============================================================================

"""Block size encoded by the cluster tree (`1` for points, `p` for `expand_tree`)."""
h2_blocksize(::ClusterTree) = 1
h2_blocksize(t::ExpandedClusterTree) = t.p

"""
    assemble_h2([T,], K, tree; kwargs...)

Assemble a square [`H2Matrix`](@ref) with the proxy-point ID method.

Supports:
- scalar `KernelMatrix` on a plain [`ClusterTree`](@ref)
- block / vectorial kernels via [`ScalarizedMatrix`](@ref) on
  [`expand_tree`](@ref)`(tree, p)` (e.g. 2D Kelvin): each proxy contributes
  `p` sample columns (unit loads in each Cartesian direction)

# Keywords
- `alpha=1.0`: box-admissibility parameter (H2Pack)
- `rtol=1e-6`, `rank=typemax(Int)`: ID tolerance / max rank for nested bases
- `nsample=200`: number of unit-surface proxy samples
- `adm=H2BoxAdmissibility(alpha)`: admissibility predicate on cluster pairs
- `global_index=true`: permute `K` into the tree ordering
- `symmetric=true`: store only upper far/near pairs and apply transpose in matvec
- `far_method=:dense | :aca`: how to store **far** coupling blocks `B`.
  Near-field (`Ddiag`, `Dnear`) is **always dense**.
  `:aca` runs **partial ACA on the fly** (entry samples only — never builds the
  full dense far block) and stores an [`RkMatrix`](@ref).
- `comp=PartialACA(;rtol)`: compressor for `far_method=:aca`
"""
function assemble_h2(
        ::Type{T},
        K,
        tree::R;
        alpha = 1.0,
        rtol = 1e-6,
        rank = typemax(Int),
        nsample = 200,
        adm = nothing,
        global_index = use_global_index(),
        symmetric = true,
        far_method::Symbol = :dense,
        comp = nothing,
    ) where {T, R}
    α = Float64(alpha)
    adm_fun = isnothing(adm) ? H2BoxAdmissibility(α) : adm
    rp = loc2glob(tree)
    global_index && (K = PermutedMatrix(K, rp, rp))
    tidx = H2TreeIndex(tree)
    nnode = length(tidx.nodes)
    n = length(tree)
    pblk = h2_blocksize(tree)
    U = [zeros(T, 0, 0) for _ in 1:nnode]
    skeleton = [Int[] for _ in 1:nnode]
    # initial leaf skeletons = full cluster index ranges
    for i in tidx.leafnodes
        skeleton[i] = collect(index_range(tidx.nodes[i]))
    end
    # proxy points per depth (geometry from underlying point tree if expanded)
    Yp_level = proxy_points_per_level(tree, α; nsample)
    # bottom-up ID compression
    nlevel = length(tidx.levels)
    for lvl in nlevel:-1:1
        for node in tidx.levels[lvl]
            # gather candidate indices
            if isempty(tidx.children[node])
                cand = skeleton[node]
            else
                cand = reduce(vcat, (skeleton[c] for c in tidx.children[node]); init = Int[])
            end
            isempty(cand) && continue
            box = container(tidx.nodes[node])
            ctr = center(box)
            d = tidx.depth_of[node]
            Yrel = Yp_level[min(d + 1, length(Yp_level))]
            pts = root_elements(tree)
            # Each proxy × each of pblk load directions → sample columns
            ncol = length(Yrel) * pblk
            A_sam = Matrix{T}(undef, length(cand), max(ncol, 0))
            if ncol > 0
                jj = 0
                for yr in Yrel
                    yp = ctr .+ yr
                    for b in 1:pblk
                        jj += 1
                        @inbounds for (ii, gi) in enumerate(cand)
                            A_sam[ii, jj] = h2_proxy_entry(K, gi, yp, b, pts, pblk)
                        end
                    end
                end
            end
            # if Yrel empty (root), skip compression
            if isempty(Yrel) || size(A_sam, 2) == 0
                U[node] = Matrix{T}(I, length(cand), length(cand))
                skeleton[node] = cand
                continue
            end
            basis, Jloc = _row_id(A_sam, float(rtol), min(Int(rank), size(A_sam)...))
            U[node] = Matrix(basis)   # dense nested / leaf basis
            skeleton[node] = isempty(Jloc) ? Int[] : cand[Jloc]
        end
    end
    # interaction lists
    near, far = _h2_block_partition(tidx, adm_fun; symmetric)
    minlvl = isempty(far) ? nlevel :
             minimum(min(tidx.depth_of[i], tidx.depth_of[j]) for (i, j) in far) + 1
    far_method in (:dense, :aca) || throw(ArgumentError(
        "far_method must be :dense or :aca; got $far_method"))
    aca = far_method === :aca ? something(comp, PartialACA(; rtol=float(rtol), rank=rank)) : nothing

    # Near-field: always dense (leaf self + inadmissible pairs)
    Ddiag = Dict{Int, Matrix{T}}()
    Dnear = Dict{Tuple{Int, Int}, Matrix{T}}()
    Bfar = Dict{Tuple{Int, Int}, Any}()
    for i in tidx.leafnodes
        ir = index_range(tidx.nodes[i])
        D = Matrix{T}(undef, length(ir), length(ir))
        getblock!(D, K, ir, ir)
        Ddiag[i] = D
    end
    for (i, j) in near
        ir = index_range(tidx.nodes[i])
        jr = index_range(tidx.nodes[j])
        D = Matrix{T}(undef, length(ir), length(jr))
        getblock!(D, K, ir, jr)
        Dnear[(i, j)] = D
    end
    # Far couplings: dense fill OR on-the-fly ACA (no full dense intermediate)
    for (i, j) in far
        di, dj = tidx.depth_of[i], tidx.depth_of[j]
        if di == dj
            Ii, Ij = skeleton[i], skeleton[j]
        elseif di > dj
            Ii = skeleton[i]
            Ij = collect(index_range(tidx.nodes[j]))
        else
            Ii = collect(index_range(tidx.nodes[i]))
            Ij = skeleton[j]
        end
        Bfar[(i, j)] = _h2_far_block(K, Ii, Ij, T, aca)
    end
    return H2Matrix{R, T}(
        tidx, U, skeleton, near, far, Ddiag, Dnear, Bfar,
        copy(rp), copy(rp), minlvl, α, n,
    )
end

# ---------------------------------------------------------------------------
# Far-block assembly: dense entry fill OR partial ACA (entry samples only)
# ---------------------------------------------------------------------------

"""
View of `parent[I[i], J[j]]` as an `m×n` matrix without copying.
Used so [`PartialACA`](@ref) can pivot through arbitrary (non-contiguous)
skeleton index lists without assembling the full block first.
"""
struct _IndexMapMatrix{T, PK} <: AbstractMatrix{T}
    parent::PK
    I::Vector{Int}
    J::Vector{Int}
end
_IndexMapMatrix(parent, I::Vector{Int}, J::Vector{Int}) =
    _IndexMapMatrix{eltype(parent), typeof(parent)}(parent, I, J)

Base.size(M::_IndexMapMatrix) = (length(M.I), length(M.J))
Base.IndexStyle(::Type{<:_IndexMapMatrix}) = IndexCartesian()
@inline Base.getindex(M::_IndexMapMatrix, i::Int, j::Int) = M.parent[M.I[i], M.J[j]]

function getblock!(out, M::_IndexMapMatrix, irange_, jrange_)
    irange = irange_ isa Colon ? axes(M, 1) : irange_
    jrange = jrange_ isa Colon ? axes(M, 2) : jrange_
    @inbounds for (jloc, j) in enumerate(jrange), (iloc, i) in enumerate(irange)
        out[iloc, jloc] = M.parent[M.I[i], M.J[j]]
    end
    return out
end

function getblock!(out, Madj::Adjoint{<:Any,<:_IndexMapMatrix}, irange_, j::Int)
    M = parent(Madj)
    # column j of M' = row j of M → parent[I[j], J[irange]]
    irange = irange_ isa Colon ? axes(Madj, 1) : irange_
    @inbounds for (iloc, i) in enumerate(irange)
        out[iloc] = conj(M.parent[M.I[j], M.J[i]])
    end
    return out
end

"""
Build far block `K[I,J]`.
- `aca === nothing`: dense entry loop (still no preallocated full-matrix path beyond the block itself).
- `aca::PartialACA`: **on-the-fly** partial ACA via [`_IndexMapMatrix`](@ref) — only pivots/crosses are evaluated.
"""
function _h2_far_block(K, I, J, ::Type{T}, aca) where {T}
    I = I isa Vector{Int} ? I : collect(Int, I)
    J = J isa Vector{Int} ? J : collect(Int, J)
    m, n = length(I), length(J)
    (m == 0 || n == 0) && return zeros(T, m, n)

    if aca === nothing
        B = Matrix{T}(undef, m, n)
        _getblock_idx!(B, K, I, J)
        return B
    end

    # Partial ACA samples K[I[·], J[·]] only at pivot rows/cols — never fills m×n first
    W = _IndexMapMatrix(K, I, J)
    return aca(W, 1:m, 1:n)
end

function assemble_h2(K::AbstractMatrix, tree; kwargs...)
    return assemble_h2(eltype(K), K, tree; kwargs...)
end

"""
    h2_proxy_entry(K, i, yp, b, pts, p) -> Number

Free-space sample for H² nested-basis construction.

- `i`: row index in **K's current ordering** (tree-local after `PermutedMatrix`)
- `yp`: proxy source location (geometry)
- `b`: load direction `1:p` (always `1` for scalar kernels)
- `pts`: `root_elements(tree)` — geometric points in underlying local order
- `p`: block size (`h2_blocksize(tree)`)

Scalar `KernelMatrix` uses `f(pts[i], yp)`. Block kernels
([`ScalarizedMatrix`](@ref) of `SMatrix{p,p}`) use
[`h2_proxy_block`](@ref) on the parent block matrix.
"""
function h2_proxy_entry end

# ---- scalar KernelMatrix (geometry via pts; local index i) ----
function h2_proxy_entry(K::KernelMatrix, i::Int, yp, b::Int, pts, p::Int)
    p == 1 && b == 1 || throw(ArgumentError("scalar KernelMatrix expects p=b=1"))
    return float(K.f(pts[i], yp))
end

function h2_proxy_entry(K::PermutedMatrix{<:KernelMatrix}, i::Int, yp, b::Int, pts, p::Int)
    # geometric: pts already in tree-local order; f ignores global index
    return h2_proxy_entry(K.data, i, yp, b, pts, p)
end

# ---- block / ScalarizedMatrix (DOF index → point + component) ----
function h2_proxy_entry(S::ScalarizedMatrix{<:Any,T,p0,q0}, i_dof::Int, yp, b::Int, pts, p::Int) where {T,p0,q0}
    p == p0 || throw(DimensionMismatch("tree blocksize p=$p ≠ matrix p=$p0"))
    1 <= b <= p0 || throw(BoundsError(1:p0, b))
    ib, a = block_index(i_dof, p0)
    return float(h2_proxy_block(parent(S), ib, a, yp, b))
end

function h2_proxy_entry(K::PermutedMatrix{<:ScalarizedMatrix}, i::Int, yp, b::Int, pts, p::Int)
    # map tree-local DOF → original DOF index into ScalarizedMatrix
    return h2_proxy_entry(K.data, K.rowperm[i], yp, b, pts, p)
end

function h2_proxy_entry(K::PermutedMatrix, i::Int, yp, b::Int, pts, p::Int)
    return h2_proxy_entry(K.data, K.rowperm[i], yp, b, pts, p)
end

"""
    h2_proxy_block(Kblock, i_point, a, yp, b) -> Number

Entry `U[a,b](x_i, yp)` of a point-indexed block kernel at field point index
`i_point` and proxy source `yp`, for use in H² sampling.

Define this for custom block kernels (e.g. Kelvin `SMatrix{2,2}`).
"""
function h2_proxy_block(Kblock, i_point::Int, a::Int, yp, b::Int)
    throw(ArgumentError(
        "H2 block proxy eval not defined for $(typeof(Kblock)); " *
        "implement h2_proxy_block(K, i_point, a, yp, b) or use a KernelMatrix",
    ))
end

# backward-compatible aliases
_kernel_eval(K, i::Int, yp, pts) = h2_proxy_entry(K, i, yp, 1, pts, 1)

function _getblock_idx!(B::Matrix{T}, K, I::Vector{Int}, J::Vector{Int}) where {T}
    @inbounds for jj in eachindex(J), ii in eachindex(I)
        B[ii, jj] = K[I[ii], J[jj]]
    end
    return B
end

# ---- block partition (H2Pack style) ----------------------------------------

function _h2_block_partition(tidx::H2TreeIndex, adm; symmetric = true)
    near = Tuple{Int, Int}[]
    far = Tuple{Int, Int}[]
    root = 1
    _h2_self!(near, far, tidx, root, adm)
    if !symmetric
        # add symmetric counterparts for directed far/near
        append!(near, [(j, i) for (i, j) in near])
        append!(far, [(j, i) for (i, j) in far])
    end
    return near, far
end

function _h2_self!(near, far, tidx, p, adm)
    ch = tidx.children[p]
    isempty(ch) && return nothing
    for c in ch
        _h2_self!(near, far, tidx, c, adm)
    end
    for a in 1:length(ch)
        for b in (a + 1):length(ch)
            _h2_intersect!(near, far, tidx, ch[a], ch[b], adm)
        end
    end
    return nothing
end

function _h2_intersect!(near, far, tidx, p1, p2, adm)
    n1, n2 = tidx.nodes[p1], tidx.nodes[p2]
    if adm(n1, n2)
        push!(far, (p1, p2))
        return nothing
    end
    c1, c2 = tidx.children[p1], tidx.children[p2]
    if isempty(c1) && isempty(c2)
        push!(near, (p1, p2))
        return nothing
    end
    if isempty(c1) && !isempty(c2)
        for c in c2
            _h2_intersect!(near, far, tidx, p1, c, adm)
        end
        return nothing
    end
    if !isempty(c1) && isempty(c2)
        for c in c1
            _h2_intersect!(near, far, tidx, c, p2, adm)
        end
        return nothing
    end
    for a in c1, b in c2
        _h2_intersect!(near, far, tidx, a, b, adm)
    end
    return nothing
end

# =============================================================================
# Matvec (H2Pack H2_matvec sweeps)
# =============================================================================

function LinearAlgebra.mul!(
        y::AbstractVector,
        H::H2Matrix,
        x::AbstractVector,
        a::Number = 1,
        b::Number = 0;
        global_index = use_global_index(),
    )
    if global_index
        x = x[H.colperm]
        y = permute!(y, H.rowperm)
        rmul!(x, a)
    elseif a != 1
        x = a * x
    end
    iszero(b) ? fill!(y, zero(eltype(y))) : rmul!(y, b)
    y .+= _h2_matvec(H, collect(x))
    global_index && invpermute!(y, H.rowperm)
    return y
end

function LinearAlgebra.mul!(
        Y::AbstractMatrix, H::H2Matrix{R, T}, X::AbstractMatrix,
        a::Number = 1, b::Number = 0;
        global_index = use_global_index(),
    ) where {R, T}
    size(Y, 2) == size(X, 2) || throw(DimensionMismatch())
    size(X, 2) == 1 && return mul!(view(Y, :, 1), H, view(X, :, 1), a, b; global_index)

    # Blocked multi-RHS: reuse level sweeps with Matrix projections
    if global_index
        Xp = Matrix{T}(X[H.colperm, :])
        rmul!(Xp, a)
        Yp = iszero(b) ? zeros(T, size(Y, 1), size(Y, 2)) :
             Matrix{T}(b .* Y[H.rowperm, :])
    else
        Xp = a == 1 ? Matrix{T}(X) : Matrix{T}(a .* X)
        Yp = iszero(b) ? zeros(T, size(Y, 1), size(Y, 2)) : Matrix{T}(b .* Y)
    end
    Yp .+= _h2_matvec_multi(H, Xp)
    if global_index
        Y[:, :] .= Yp[invperm(H.rowperm), :]
    else
        Y[:, :] .= Yp
    end
    return Y
end

function _h2_matvec(H::H2Matrix{R, T}, x::Vector{T}) where {R, T}
    U = _h2_matvec_multi(H, reshape(x, :, 1))
    return vec(U)
end

"""H² apply for one or many RHS (`X` is `n×s`). Level sweeps use Matrix blocks."""
function _h2_matvec_multi(H::H2Matrix{R, T}, X::AbstractMatrix{T}) where {R, T}
    tidx = H.tidx
    nnode = length(tidx.nodes)
    n, s = size(X)
    u = zeros(T, n, s)
    empty0 = zeros(T, 0, s)
    yproj = Vector{Matrix{T}}(undef, nnode)
    for i in 1:nnode
        yproj[i] = empty0
    end
    nlevel = length(tidx.levels)
    minlvl = H.minlvl
    for lvl in nlevel:-1:minlvl
        for node in tidx.levels[lvl]
            Un = H.U[node]
            size(Un, 2) == 0 && continue
            ch = tidx.children[node]
            if isempty(ch)
                ir = index_range(tidx.nodes[node])
                yproj[node] = Un' * view(X, ir, :)
            else
                stack = reduce(vcat, (yproj[c] for c in ch if size(yproj[c], 1) > 0); init = empty0)
                size(stack, 1) == size(Un, 1) || continue
                yproj[node] = Un' * stack
            end
        end
    end
    inter = [zeros(T, size(H.U[i], 2), s) for i in 1:nnode]
    for (c1, c2) in H.far
        B = H.Bfar[(c1, c2)]
        d1, d2 = tidx.depth_of[c1], tidx.depth_of[c2]
        if d1 == d2
            if size(yproj[c2], 1) == size(B, 2)
                inter[c1] = inter[c1] + B * yproj[c2]
            end
            if size(yproj[c1], 1) == size(B, 1)
                inter[c2] = inter[c2] + B' * yproj[c1]
            end
        elseif d1 > d2
            jr = index_range(tidx.nodes[c2])
            if size(B, 2) == length(jr)
                inter[c1] = inter[c1] + B * view(X, jr, :)
            end
            if size(yproj[c1], 1) == size(B, 1)
                view(u, jr, :) .+= B' * yproj[c1]
            end
        else
            ir = index_range(tidx.nodes[c1])
            if size(B, 1) == length(ir) && size(yproj[c2], 1) == size(B, 2)
                view(u, ir, :) .+= B * yproj[c2]
            end
            if size(B, 1) == length(ir)
                inter[c2] = inter[c2] + B' * view(X, ir, :)
            end
        end
    end
    for lvl in minlvl:nlevel
        for node in tidx.levels[lvl]
            size(inter[node], 1) == 0 && continue
            size(H.U[node], 2) == size(inter[node], 1) || continue
            contrib = H.U[node] * inter[node]
            ch = tidx.children[node]
            if isempty(ch)
                ir = index_range(tidx.nodes[node])
                length(ir) == size(contrib, 1) && (view(u, ir, :) .+= contrib)
            else
                off = 1
                for c in ch
                    rc = size(H.U[c], 2)
                    rc == 0 && continue
                    chunk = contrib[off:(off + rc - 1), :]
                    off += rc
                    if size(inter[c], 1) == rc
                        inter[c] = inter[c] + chunk
                    elseif size(inter[c], 1) == 0
                        inter[c] = chunk
                    end
                end
            end
        end
    end
    for (i, D) in H.Ddiag
        ir = index_range(tidx.nodes[i])
        view(u, ir, :) .+= D * view(X, ir, :)
    end
    for (i, j) in H.near
        D = H.Dnear[(i, j)]
        ir = index_range(tidx.nodes[i])
        jr = index_range(tidx.nodes[j])
        view(u, ir, :) .+= D * view(X, jr, :)
        view(u, jr, :) .+= D' * view(X, ir, :)
    end
    return u
end

function Base.Matrix(H::H2Matrix{R, T}; global_index = true) where {R, T}
    n = size(H, 1)
    M, ej = zeros(T, n, n), zeros(T, n)
    @inbounds for j in 1:n
        fill!(ej, 0); ej[j] = 1
        M[:, j] = _h2_matvec(H, ej)
    end
    global_index || return M
    p = H.rowperm
    return Matrix(PermutedMatrix(M, invperm(p), invperm(p)))
end

function Base.deepcopy_internal(H::H2Matrix{R, T}, sd::IdDict) where {R, T}
    haskey(sd, H) && return sd[H]
    H2 = H2Matrix{R, T}(
        H.tidx,
        [copy(U) for U in H.U],
        [copy(s) for s in H.skeleton],
        copy(H.near),
        copy(H.far),
        Dict(k => copy(v) for (k, v) in H.Ddiag),
        Dict(k => copy(v) for (k, v) in H.Dnear),
        Dict(k => copy(v) for (k, v) in H.Bfar),
        copy(H.rowperm),
        copy(H.colperm),
        H.minlvl,
        H.alpha,
        H.n,
    )
    sd[H] = H2
    return H2
end
