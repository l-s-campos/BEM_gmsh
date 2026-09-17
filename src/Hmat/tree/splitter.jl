"""
    abstract type AbstractSplitter

An `AbstractSplitter` is used to split a [`ClusterTree`](@ref). The interface
requires the following methods:
- `should_split(clt,splitter)` : return a `Bool` determining if the
  `ClusterTree` should be further divided
- `split!(clt,splitter)` : perform the splitting of the `ClusterTree` handling
  the necessary data sorting.

See [`GeometricSplitter`](@ref) for an example of an implementation.
"""
abstract type AbstractSplitter end

"""
    should_split(clt::ClusterTree, depth, splitter::AbstractSplitter)

Determine whether or not a `ClusterTree` should be further divided.
"""
function should_split end

"""
    split!(clt::ClusterTree,splitter::AbstractSplitter)

Divide `clt` using the strategy implemented by `splitter`. This function is
reponsible of assigning the `children` and `parent` fields, as well as of
permuting the data of `clt`.
"""
function split! end

"""
    struct GeometricSplitter <: AbstractSplitter

Used to split a `ClusterTree` in half along the largest axis. The children boxes
are shrank to tighly fit the data.
"""
Base.@kwdef struct GeometricSplitter <: AbstractSplitter
    nmax::Int = 50
end

function should_split(node::ClusterTree, depth, splitter::GeometricSplitter)
    return length(node) > splitter.nmax
end

function split!(cluster::ClusterTree, ::GeometricSplitter)
    rec = cluster.container
    wmax, imax = findmax(high_corner(rec) - low_corner(rec))
    mid = low_corner(rec)[imax] + wmax / 2
    predicate = (x) -> x[imax] < mid
    left_node, right_node = binary_split!(cluster, predicate)
    cluster.children = [left_node, right_node]
    return cluster
end

"""Alias of [`GeometricSplitter`](@ref) (tight AABB children)."""
const GeometricMinimalSplitter = GeometricSplitter

"""
    struct DyadicSplitter <: AbstractSplitter

Quadtree (2D) / octree (3D) split: cut every axis at the box midpoint and keep
non-empty orthants (up to `2^N` children). Matches Flatiron `pts_tree2d` /
`pts_tree3d`. If every point falls in one orthant, falls back to
[`GeometricSplitter`](@ref) so the tree still refines.

`tight=true` (default, FMM) shrinks each child to the point AABB.
`tight=false` keeps the geometric orthant of the parent — required for
same-level neighbor / interaction lists (NNCA, equal-size FMM boxes).
"""
Base.@kwdef struct DyadicSplitter <: AbstractSplitter
    nmax::Int = 50
    tight::Bool = true
end

"""Geometric quadtree (2-D) / octree (3-D) for H-matrix and NNCA trees."""
hmatrix_splitter(; nmax::Int=32) = DyadicSplitter(; nmax=nmax, tight=false)

function should_split(node::ClusterTree, depth, splitter::DyadicSplitter)
    return length(node) > splitter.nmax
end

function split!(cluster::ClusterTree{N,T}, splitter::DyadicSplitter) where {N,T}
    rec = cluster.container
    mid = (low_corner(rec) + high_corner(rec)) / 2
    els = root_elements(cluster)
    irange = index_range(cluster)
    l2g = loc2glob(cluster)
    nort = 1 << N
    nloc = length(irange)
    codes = Vector{Int}(undef, nloc)
    counts = zeros(Int, nort)
    @inbounds for (k, i) in enumerate(irange)
        pt = els[l2g[i]]
        code = 0
        for d in 1:N
            pt[d] >= mid[d] && (code |= 1 << (d - 1))
        end
        c = code + 1
        codes[k] = c
        counts[c] += 1
    end
    nfilled = count(>(0), counts)
    if nfilled <= 1
        return split!(cluster, GeometricSplitter(nmax=splitter.nmax))
    end
    offsets = Vector{Int}(undef, nort)
    acc = 0
    @inbounds for k in 1:nort
        offsets[k] = acc
        acc += counts[k]
    end
    buff = view(cluster.glob2loc, irange)
    cursor = copy(offsets)
    xls = [high_corner(rec) for _ in 1:nort]
    xus = [low_corner(rec) for _ in 1:nort]
    @inbounds for (k, i) in enumerate(irange)
        gi = l2g[i]
        pt = els[gi]
        c = codes[k]
        cursor[c] += 1
        buff[cursor[c]] = gi
        xls[c] = min.(xls[c], pt)
        xus[c] = max.(xus[c], pt)
    end
    copy!(view(l2g, irange), buff)
    children = ClusterTree{N,T}[]
    sizehint!(children, nfilled)
    lo = low_corner(rec)
    hi = high_corner(rec)
    @inbounds for k in 1:nort
        counts[k] == 0 && continue
        a = irange.start + offsets[k]
        b = a + counts[k] - 1
        box = if splitter.tight
            HyperRectangle(xls[k], xus[k])
        else
            _orthant_box(lo, hi, mid, k - 1, N)
        end
        push!(children, ClusterTree(
            els, box, a:b, l2g, cluster.glob2loc, nothing, cluster,
        ))
    end
    cluster.children = children
    return cluster
end

"""
    struct PrincipalComponentSplitter <: AbstractSplitter
"""
Base.@kwdef struct PrincipalComponentSplitter <: AbstractSplitter
    nmax::Int = 50
end

function should_split(node::ClusterTree, depth, splitter::PrincipalComponentSplitter)
    return length(node) > splitter.nmax
end

function split!(cluster::ClusterTree, ::PrincipalComponentSplitter)
    pts = cluster._elements
    irange = cluster.index_range
    xc = center_of_mass(cluster)
    # compute covariance matrix for principal direction
    l2g = loc2glob(cluster)
    cov = sum(irange) do i
        x = center(pts[l2g[i]])
        return (x - xc) * transpose(x - xc)
    end
    v = eigvecs(cov)[:, end]
    predicate = (x) -> dot(x - xc, v) < 0
    left_node, right_node = binary_split!(cluster, predicate)
    cluster.children = [left_node, right_node]
    return cluster
end

function center_of_mass(clt::ClusterTree)
    pts = clt._elements
    loc_idxs = clt.index_range
    l2g = loc2glob(clt)
    # w    = clt.weights
    n = length(loc_idxs)
    # M    = isempty(w) ? n : sum(i->w[i],glob_idxs)
    # xc   = isempty(w) ? sum(i->pts[i]/M,glob_idxs) : sum(i->w[i]*pts[i]/M,glob_idxs)
    M = n
    xc = sum(i -> center(pts[l2g[i]]) / M, loc_idxs)
    return xc
end

"""
    struct CardinalitySplitter <: AbstractSplitter

Used to split a `ClusterTree` along the largest dimension if
`length(tree)>nmax`. The split is performed so the `data` is evenly distributed
amongst all children.

## See also: [`AbstractSplitter`](@ref)
"""
Base.@kwdef struct CardinalitySplitter <: AbstractSplitter
    nmax::Int = 50
end

function should_split(node::ClusterTree, depth, splitter::CardinalitySplitter)
    return length(node) > splitter.nmax
end

function split!(cluster::ClusterTree, ::CardinalitySplitter)
    points = cluster._elements
    irange = cluster.index_range
    rec = container(cluster)
    _, imax = findmax(high_corner(rec) - low_corner(rec))
    l2g = loc2glob(cluster)
    # sometimes the median can fail to split the data, for example
    # median([0,0,0,1]) = 0, so no point will be sorted on the left as per the
    # predicate x->x<med, causing an infinite recursion. This is rare, but can
    # happen In such cases, we use the mean instead of the median.
    med = median((points[l2g[i]])[imax] for i in irange) # the median along largest axis `imax`
    npts = sum(i -> points[l2g[i]][imax] < med, irange)
    if abs(npts - length(irange) / 2) > 1
        med = mean((points[l2g[i]])[imax] for i in irange)
    end
    predicate = (x) -> x[imax] < med
    left_node, right_node = binary_split!(cluster, predicate)
    cluster.children = [left_node, right_node]
    return cluster
end

"""
    binary_split!(cluster::ClusterTree,predicate)

Split a `ClusterTree` into two, sorting all elements in the process according to
predicate. `cluster` is assigned as parent to each children.

Each point is sorted according to whether `f(x)` returns `true` (point sorted on
the "left" node) or `false` (point sorted on the "right" node). At the end a
minimal `HyperRectangle` containing all left/right points is created.
"""
function binary_split!(cluster::ClusterTree{N, T}, predicate::Function) where {N, T}
    f = predicate
    rec = container(cluster)
    els = root_elements(cluster)
    irange = index_range(cluster)
    n = length(irange)
    buff = view(cluster.glob2loc, irange) # use as a temporary buffer
    l2g = loc2glob(cluster)
    npts_left = 0
    npts_right = 0
    xl_left = xl_right = high_corner(rec)
    xu_left = xu_right = low_corner(rec)
    # sort the points into left and right rectangle
    for i in irange
        pt = els[l2g[i]]
        if f(pt)
            xl_left = min.(xl_left, pt)
            xu_left = max.(xu_left, pt)
            npts_left += 1
            buff[npts_left] = l2g[i]
        else
            xl_right = min.(xl_right, pt)
            xu_right = max.(xu_right, pt)
            buff[n - npts_right] = l2g[i]
            npts_right += 1
        end
    end
    # bounding boxes
    left_rec = HyperRectangle(xl_left, xu_left)
    right_rec = HyperRectangle(xl_right, xu_right)
    @assert npts_left + npts_right == length(irange) "elements lost during split"
    # new ranges for children cluster
    copy!(view(l2g, irange), buff)
    left_indices = (irange.start):((irange.start) + npts_left - 1)
    right_indices = (irange.start + npts_left):(irange.stop)
    # create children
    clt1 = ClusterTree(els, left_rec, left_indices, l2g, cluster.glob2loc, nothing, cluster)
    clt2 =
        ClusterTree(els, right_rec, right_indices, l2g, cluster.glob2loc, nothing, cluster)
    return clt1, clt2
end

function _orthant_box(lo::SVector{N,T}, hi::SVector{N,T}, mid::SVector{N,T},
        code::Int, ::Int) where {N,T}
    newlo = ntuple(d -> ((code >> (d - 1)) & 1) == 1 ? mid[d] : lo[d], N)
    newhi = ntuple(d -> ((code >> (d - 1)) & 1) == 1 ? hi[d] : mid[d], N)
    return HyperRectangle(SVector{N,T}(newlo), SVector{N,T}(newhi))
end
