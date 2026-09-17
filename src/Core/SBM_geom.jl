# Geometry helpers for SBM origin intensity factors.
# Collocation is `dad.Nodes`: GL points placed by `format2d`.
# Γ_m is that collocation's parent `dad.elements` entry (same nodes + polynomial).
# L_n = parent-element length / n_loc (so 2-node elements: Length/2).

"""GL collocation parameters on [-1,1], same as `format2d` / `discontinuous_nodes_weights`."""
sbm_gl_nodes(nel_nodes::Integer) = discontinuous_nodes_weights(nel_nodes - 1)[1]

"""Map each collocation to its BEM element / local GL index; neighbor lengths."""
function sbm_bind_geometry(dad::BEMdata)
    n = dad.n
    col_el = zeros(Int, n)
    col_loc = zeros(Int, n)
    lengths = zeros(n)
    @inbounds for (ie, el) in enumerate(dad.elements)
        nloc = length(el.index)
        share = el.Length / max(nloc, 1)
        for (k, j) in enumerate(el.index)
            col_el[j] = ie
            col_loc[j] = k
            lengths[j] = share
        end
    end
    pos = lengths[lengths .> 0]
    μ = isempty(pos) ? 1.0 : mean(pos)
    @inbounds for i in 1:n
        lengths[i] < 1e-14 && (lengths[i] = μ)
    end
    return (; col_el, col_loc, lengths)
end

"""Parent BEM element of collocation `m`: GL nodes as geometry, ξ = format2d GL node."""
function sbm_parent_geom(dad::BEMdata, col_el, col_loc, m::Integer)
    el = dad.elements[col_el[m]]
    geo = Point2D[dad.Nodes[j] for j in el.index]
    k = col_loc[m]
    ξm = float(sbm_gl_nodes(length(el))[k])
    return geo, dad.element_type, ξm
end

"""Parent BEM element used for the Γ_m integral."""
function sbm_element_geom(d, m::Integer)
    return sbm_parent_geom(d.dad, d.col_el, d.col_loc, m)
end

"""Voronoi interval of collocation `m` in the parent-element parameter ξ ∈ [-1,1]."""
function sbm_xi_interval(d, m::Integer)
    dad = d.dad
    el = dad.elements[d.col_el[m]]
    ξs = sbm_gl_nodes(length(el))
    k = d.col_loc[m]
    nk = length(ξs)
    a = k == 1 ? -1.0 : 0.5 * (float(ξs[k - 1]) + float(ξs[k]))
    b = k == nk ? 1.0 : 0.5 * (float(ξs[k]) + float(ξs[k + 1]))
    return a, b
end

@inline function sbm_geom_at(geo::AbstractVector{<:Point2D}, poly, ξ)
    N, dN = shapefun(poly, ξ)
    pg = zero(geo[1])
    dx = zero(geo[1])
    @inbounds for k in eachindex(geo)
        pg += N[1, k] * geo[k]
        dx += dN[1, k] * geo[k]
    end
    J = norm(dx)
    J < 1e-30 && return pg, 0.0, dx
    nrm = tan2normal(dx / J)
    return pg, J, nrm
end
