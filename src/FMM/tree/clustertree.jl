"""
    mutable struct ClusterTree{N,T}

Binary cluster tree over points of type `SVector{N,T}` (ported from BEM/Hmat).

# Fields
- `_elements` : sorted elements (local order)
- `container` : bounding `HyperRectangle`
- `index_range` : local indices of elements in this node
- `loc2glob` / `glob2loc` : index permutations
- `children` / `parentnode`
"""
mutable struct ClusterTree{N,T}
    _elements::Vector{SVector{N,T}}
    container::HyperRectangle{N,T}
    index_range::UnitRange{Int}
    loc2glob::Vector{Int}
    glob2loc::Vector{Int}
    children::Vector{ClusterTree{N,T}}
    parentnode::ClusterTree{N,T}
    function ClusterTree(
        els::Vector{SVector{N,T}},
        container,
        loc_idxs,
        loc2glob,
        glob2loc,
        children,
        parentnode,
    ) where {N,T}
        clt = new{N,T}(els, container, loc_idxs, loc2glob, glob2loc)
        clt.children = isnothing(children) ? Vector{typeof(clt)}() : children
        clt.parentnode = isnothing(parentnode) ? clt : parentnode
        return clt
    end
end

root_elements(clt::ClusterTree) = clt._elements
index_range(clt::ClusterTree) = clt.index_range
children(clt::ClusterTree) = clt.children
parentnode(clt::ClusterTree) = clt.parentnode
container(clt::ClusterTree) = clt.container
elements(clt::ClusterTree) = view(root_elements(clt), index_range(clt))
loc2glob(clt::ClusterTree) = clt.loc2glob
glob2loc(clt::ClusterTree) = clt.glob2loc

isleaf(clt::ClusterTree) = isempty(clt.children)
isroot(clt::ClusterTree) = parentnode(clt) === clt

diameter(node::ClusterTree) = diameter(container(node))
radius(node::ClusterTree) = radius(container(node))
distance(a::ClusterTree, b::ClusterTree) = distance(container(a), container(b))
Base.length(node::ClusterTree) = length(index_range(node))

function ClusterTree(
    elements,
    splitter=GeometricSplitter();
    copy_elements=true,
    threads=false,
)
    copy_elements && (elements = deepcopy(elements))
    bbox = bounding_box(elements)
    n = length(elements)
    irange = 1:n
    l2g = collect(irange)
    g2l = collect(irange)
    root = ClusterTree(elements, bbox, irange, l2g, g2l, nothing, nothing)
    _build_cluster_tree!(root, splitter, threads)
    g2l .= invperm(l2g)
    copy!(elements, elements[l2g])
    return root
end

function _build_cluster_tree!(current_node, splitter, threads, depth=0)
    if should_split(current_node, depth, splitter)
        split!(current_node, splitter)
        if threads
            Threads.@threads for child in children(current_node)
                _build_cluster_tree!(child, splitter, threads, depth + 1)
            end
        else
            for child in children(current_node)
                _build_cluster_tree!(child, splitter, threads, depth + 1)
            end
        end
    end
    return current_node
end

function Base.show(io::IO, tree::ClusterTree{N,T}) where {N,T}
    print(io, "ClusterTree{$N,$T} with $(length(tree)) points")
end

function Base.summary(clt::ClusterTree)
    @printf "Cluster tree with %i elements" length(clt)
    ns = nodes(clt)
    @printf "\n\t number of nodes: %i" length(ns)
    ls = leaves(clt)
    @printf "\n\t number of leaves: %i" length(ls)
    ppl = map(length, ls)
    @printf "\n\t min elements/leaf: %i" minimum(ppl)
    @printf "\n\t max elements/leaf: %i" maximum(ppl)
    dpl = map(depth, ls)
    @printf "\n\t min leaf depth: %i" minimum(dpl)
    @printf "\n\t max leaf depth: %i" maximum(dpl)
end
