# Geometric same-level neighbors (Moore) and FMM interaction list on a
# ClusterTree. Boxes at one level are neighbors when their containers touch
# (η = √2 equal-size squares/cubes). IL(B) = children of neighbors of parent(B)
# that are not neighbors of B. Works with missing orthants.

"""Group nodes of `root` by `depth` (index `d+1` holds depth `d`)."""
function nodes_by_depth(root::ClusterTree)
    nds = nodes(root)
    dmax = maximum(depth, nds; init=0)
    levels = [Vector{typeof(root)}() for _ in 0:dmax]
    for n in nds
        push!(levels[depth(n) + 1], n)
    end
    return levels
end

@inline function boxes_touch(a::ClusterTree, b::ClusterTree; atol=0.0)
    da = diameter(a)
    db = diameter(b)
    return distance(a, b) <= atol + 1e-12 * (1 + max(da, db))
end

"""
    neighbor_il_lists(root) -> (neighbors, il, id2node)

`neighbors[i]` / `il[i]` are `node_id` vectors for the box with `node_id==i`.
`id2node[i]` is that `ClusterTree`. Root has empty lists. Depth-1 boxes are
mutual neighbors (all siblings of a square/cube touch).
"""
function neighbor_il_lists(root::ClusterTree)
    node_id(root) == 0 && assign_node_ids!(root)
    nn = nnodes(root)
    neighbors = [Int[] for _ in 1:nn]
    il = [Int[] for _ in 1:nn]
    id2node = Vector{typeof(root)}(undef, nn)
    for n in nodes(root)
        id2node[node_id(n)] = n
    end
    levels = nodes_by_depth(root)
    length(levels) < 2 && return neighbors, il, id2node

    # depth 1: siblings
    for node in levels[2]
        P = parentnode(node)
        id = node_id(node)
        for S in children(P)
            S === node && continue
            boxes_touch(node, S) && push!(neighbors[id], node_id(S))
        end
    end

    for d in 2:(length(levels) - 1)
        for node in levels[d + 1]
            P = parentnode(node)
            id = node_id(node)
            # children of parent and of parent's neighbors
            for C in children(P)
                C === node && continue
                if boxes_touch(node, C)
                    push!(neighbors[id], node_id(C))
                else
                    push!(il[id], node_id(C))
                end
            end
            for qid in neighbors[node_id(P)]
                Q = id2node[qid]
                for C in children(Q)
                    if boxes_touch(node, C)
                        push!(neighbors[id], node_id(C))
                    else
                        push!(il[id], node_id(C))
                    end
                end
            end
        end
    end
    return neighbors, il, id2node
end

"""
    dual_neighbor_il_lists(X, Y) -> (neighbors, il, id2X, id2Y)

Dual-tree FMM lists for a rectangular kernel on row tree `X` and column tree
`Y`. `neighbors[i]` / `il[i]` are **column** `node_id`s for row box `i`.
Well-separated pairs at depth ≥ 2 become M2L; touching leaves become near.
Prefer a shared root [`container`](@ref) so same-depth boxes have equal size.
"""
function dual_neighbor_il_lists(X::ClusterTree, Y::ClusterTree)
    node_id(X) == 0 && assign_node_ids!(X)
    node_id(Y) == 0 && assign_node_ids!(Y)
    nnX = nnodes(X)
    neighbors = [Int[] for _ in 1:nnX]
    il = [Int[] for _ in 1:nnX]
    id2X = Vector{typeof(X)}(undef, nnX)
    id2Y = Vector{typeof(Y)}(undef, nnodes(Y))
    for n in nodes(X)
        id2X[node_id(n)] = n
    end
    for n in nodes(Y)
        id2Y[node_id(n)] = n
    end
    function visit!(A, B)
        ta = node_id(A)
        tb = node_id(B)
        if !boxes_touch(A, B)
            if depth(A) >= 2 && depth(B) >= 2
                push!(il[ta], tb)
                return
            elseif isleaf(A) && isleaf(B)
                push!(neighbors[ta], tb)
                return
            end
        elseif isleaf(A) && isleaf(B)
            push!(neighbors[ta], tb)
            return
        end
        if isleaf(A)
            for b in children(B)
                visit!(A, b)
            end
        elseif isleaf(B)
            for a in children(A)
                visit!(a, B)
            end
        elseif diameter(A) >= diameter(B)
            for a in children(A)
                visit!(a, B)
            end
        else
            for b in children(B)
                visit!(A, b)
            end
        end
        return
    end
    visit!(X, Y)
    return neighbors, il, id2X, id2Y
end
