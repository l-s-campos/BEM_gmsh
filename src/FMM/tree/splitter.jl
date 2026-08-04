"""
    abstract type AbstractSplitter

Strategy used to split a [`ClusterTree`](@ref).
"""
abstract type AbstractSplitter end

function should_split end
function split! end

"""
    GeometricSplitter(; nmax=50)

Split along the largest axis at the midpoint; shrink children to the data.
"""
Base.@kwdef struct GeometricSplitter <: AbstractSplitter
    nmax::Int = 50
end

should_split(node::ClusterTree, depth, s::GeometricSplitter) = length(node) > s.nmax

function split!(cluster::ClusterTree, ::GeometricSplitter)
    rec = cluster.container
    wmax, imax = findmax(high_corner(rec) - low_corner(rec))
    mid = low_corner(rec)[imax] + wmax / 2
    left_node, right_node = binary_split!(cluster, x -> x[imax] < mid)
    cluster.children = [left_node, right_node]
    return cluster
end

"""
    CardinalitySplitter(; nmax=50)

Split along the largest axis near the median so children have similar sizes.
"""
Base.@kwdef struct CardinalitySplitter <: AbstractSplitter
    nmax::Int = 50
end

should_split(node::ClusterTree, depth, s::CardinalitySplitter) = length(node) > s.nmax

function split!(cluster::ClusterTree, ::CardinalitySplitter)
    points = cluster._elements
    irange = cluster.index_range
    rec = container(cluster)
    _, imax = findmax(high_corner(rec) - low_corner(rec))
    l2g = loc2glob(cluster)
    med = median((points[l2g[i]])[imax] for i in irange)
    npts = sum(i -> points[l2g[i]][imax] < med, irange)
    if abs(npts - length(irange) / 2) > 1
        med = mean((points[l2g[i]])[imax] for i in irange)
    end
    left_node, right_node = binary_split!(cluster, x -> x[imax] < med)
    cluster.children = [left_node, right_node]
    return cluster
end

"""
    binary_split!(cluster, predicate)

Partition elements of `cluster` into two children according to `predicate`.
"""
function binary_split!(cluster::ClusterTree{N,T}, predicate::Function) where {N,T}
    rec = container(cluster)
    els = root_elements(cluster)
    irange = index_range(cluster)
    n = length(irange)
    buff = view(cluster.glob2loc, irange)
    l2g = loc2glob(cluster)
    npts_left = 0
    npts_right = 0
    xl_left = xl_right = high_corner(rec)
    xu_left = xu_right = low_corner(rec)
    for i in irange
        pt = els[l2g[i]]
        if predicate(pt)
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
    left_rec = HyperRectangle(xl_left, xu_left)
    right_rec = HyperRectangle(xl_right, xu_right)
    @assert npts_left + npts_right == length(irange) "elements lost during split"
    copy!(view(l2g, irange), buff)
    left_indices = (irange.start):(irange.start + npts_left - 1)
    right_indices = (irange.start + npts_left):(irange.stop)
    clt1 = ClusterTree(els, left_rec, left_indices, l2g, cluster.glob2loc, nothing, cluster)
    clt2 = ClusterTree(els, right_rec, right_indices, l2g, cluster.glob2loc, nothing, cluster)
    return clt1, clt2
end
