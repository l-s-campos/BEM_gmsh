"""
    filter_tree(f, tree, isterminal=true)

Return a vector of nodes of `tree` for which `f(node)` is true.
"""
function filter_tree(f, tree, isterminal=true)
    nodes = Vector{typeof(tree)}()
    return filter_tree!(f, nodes, tree, isterminal)
end

function filter_tree!(f, nodes, tree, isterminal=true)
    if f(tree)
        push!(nodes, tree)
        isterminal || foreach(c -> filter_tree!(f, nodes, c, isterminal), children(tree))
    else
        foreach(c -> filter_tree!(f, nodes, c, isterminal), children(tree))
    end
    return nodes
end

leaves(tree) = filter_tree(isleaf, tree, true)
nodes(tree) = filter_tree(_ -> true, tree, false)

function depth(tree, acc=0)
    isroot(tree) && return acc
    return depth(parentnode(tree), acc + 1)
end
