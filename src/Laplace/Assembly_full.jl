export H_G_full_direct, H_G_full_direct!

"""
    H_G_full_direct(dad; npg=20, threaded=true)

Assemble dense influence matrices `H` and `G`.

**2D near-field integration** (`integrate_element`):
- Laplace → **Dumont** (GL + C_G / C₁)
- 2D Elasticity (Kelvin) → **Dumont** (log U + H1/H2 on T)
- 2D AnisotropicElasticity (Lekhnitskii) → **SST** (Cordeiro & Leonel 2020)
- other scalars → adaptive **sinh**
- far pairs → one-point nodal rule (unchanged)

**3D near-field**: polar+radial sinh when close, else tensor sinh.

Parallelism: `threaded=true` uses `Threads.@threads` over collocation rows.
"""
function H_G_full_direct(dad::BEMdata{<:Scalar}; npg=20, threaded=true)
    if dad.dimension == 3
        return H_G_full_direct_3d!(dad; npg=npg, threaded=threaded)
    end
    return H_G_full_direct_2d!(dad; npg=npg, threaded=threaded)
end

# keep positional npg for backward compatibility
H_G_full_direct(dad::BEMdata{<:Scalar}, npg::Integer; threaded=true) =
    H_G_full_direct(dad; npg=npg, threaded=threaded)

function H_G_full_direct(dad::BEMdata{<:Vectorial}; npg=20, threaded=true)
    if dad.dimension == 3
        return H_G_full_direct_3d_vec!(dad; npg=npg, threaded=threaded)
    end
    return H_G_full_direct_2d_vec!(dad; npg=npg, threaded=threaded)
end
H_G_full_direct(dad::BEMdata{<:Vectorial}, npg::Integer; threaded=true) =
    H_G_full_direct(dad; npg=npg, threaded=threaded)

# =============================================================================
# 2D scalar
# =============================================================================

function H_G_full_direct_2d!(dad::BEMdata{<:Scalar}; npg=20, threaded=true)
    qsi, w = gausslegendre(npg)
    H = zeros(dad.nt, dad.nt)
    G = zeros(dad.nt, dad.n)
    set_cache!(dad; H, G, qsi, w)

    elems = dad.elements
    n_el = length(elems)
    # Precompute element geometry coords (Lagrange nodes or Bézier controls)
    Xel = [element_geometry(dad, e) for e in elems]

    if threaded && Threads.nthreads() > 1
        Threads.@threads for i in 1:dad.nt
            _assemble_row_2d_scalar!(dad, H, G, i, elems, Xel)
        end
    else
        @showprogress "Assembling H and G" for i in 1:dad.nt
            _assemble_row_2d_scalar!(dad, H, G, i, elems, Xel)
        end
    end

    @inbounds for i in 1:dad.nt
        H[i, i] = 0.0
        H[i, i] = -sum(H[i, :])
    end
    return H, G
end

function _assemble_row_2d_scalar!(dad, H, G, i, elems, Xel)
    pf = point(dad, i)
    @inbounds for (ej, elem_j) in enumerate(elems)
        xj = Xel[ej]
        r0 = euclidean(pf, xj[1])
        if r0 < 2 * elem_j.Length
            h = @view H[i, elem_j.index]
            g = @view G[i, elem_j.index]
            integrate_element(dad, elem_j, xj, pf, h, g)
        else
            for k in eachindex(elem_j.index)
                node = elem_j.index[k]
                # collocation location (not Bézier control)
                r_node = dad.Nodes[node] - pf
                n_node = dad.Normal[node]
                Tast_k, Qast_k = fundamental(dad, r_node, n_node)
                wjk = elem_j.Jacobian[k] * dad.elem_weight[k]
                H[i, node] = Qast_k * wjk
                G[i, node] = Tast_k * wjk
            end
        end
    end
    return nothing
end

# =============================================================================
# 3D scalar
# =============================================================================

function H_G_full_direct_3d!(dad::BEMdata{<:Scalar}; npg=12, threaded=true)
    qsi, w = gausslegendre(npg)
    H = zeros(dad.nt, dad.nt)
    G = zeros(dad.nt, dad.n)
    set_cache!(dad; H, G, qsi, w)
    elems = dad.elements
    Xel = [element_geometry(dad, e) for e in elems]

    if threaded && Threads.nthreads() > 1
        Threads.@threads for i in 1:dad.nt
            _assemble_row_3d_scalar!(dad, H, G, i, elems, Xel, qsi, w)
        end
    else
        @showprogress "Assembling H and G (3D)" for i in 1:dad.nt
            _assemble_row_3d_scalar!(dad, H, G, i, elems, Xel, qsi, w)
        end
    end
    @inbounds for i in 1:dad.nt
        H[i, i] = 0.0
        H[i, i] = -sum(H[i, :])
    end
    return H, G
end

function _assemble_row_3d_scalar!(dad, H, G, i, elems, Xel, qsi, w)
    pf = point(dad, i)
    @inbounds for (ej, elem_j) in enumerate(elems)
        xj = Xel[ej]
        r0 = euclidean(pf, xj[1])
        if r0 < 2 * elem_j.Length
            h = @view H[i, elem_j.index]
            g = @view G[i, elem_j.index]
            integrate_element(dad, elem_j, xj, pf, qsi, w, h, g)
        else
            for k in eachindex(elem_j.index)
                node = elem_j.index[k]
                # collocation location (not Bézier control)
                r_node = dad.Nodes[node] - pf
                n_node = dad.Normal[node]
                Tast_k, Qast_k = fundamental(dad, r_node, n_node)
                wjk = elem_j.Jacobian[k] * dad.elem_weight[k]
                H[i, node] = Qast_k * wjk
                G[i, node] = Tast_k * wjk
            end
        end
    end
    return nothing
end

# =============================================================================
# 2D / 3D vectorial
# =============================================================================

function H_G_full_direct_2d_vec!(dad::BEMdata{<:Vectorial}; npg=20, threaded=true)
    qsi, w = gausslegendre(npg)
    dim = dad.dimension
    H = zeros(dim * dad.nt, dim * dad.nt)
    G = zeros(dim * dad.nt, dim * dad.n)
    set_cache!(dad; H, G, qsi, w)
    elems = dad.elements
    Xel = [element_geometry(dad, e) for e in elems]

    if threaded && Threads.nthreads() > 1
        Threads.@threads for i in 1:dad.nt
            _assemble_row_2d_vec!(dad, H, G, i, elems, Xel)
        end
    else
        @showprogress "Assembling H and G" for i in 1:dad.nt
            _assemble_row_2d_vec!(dad, H, G, i, elems, Xel)
        end
    end
    @views for i in 1:dad.nt
        ii = dim*(i-1)+1:dim*i
        H[ii, ii] .= 0.0
        for j in 1:dim
            H[ii, ii[j]] .= -sum(H[ii, j:dim:end]; dims=2)
        end
    end
    return H, G
end

function _assemble_row_2d_vec!(dad, H, G, i, elems, Xel)
    dim = dad.dimension
    pf = point(dad, i)
    ii = dim*(i-1)+1:dim*i
    @inbounds for (ej, elem_j) in enumerate(elems)
        xj = Xel[ej]
        jj = dim*(elem_j.index[1]-1)+1:elem_j.index[end]*dim
        r0 = euclidean(pf, xj[1])
        if r0 < 2 * elem_j.Length
            h = @view H[ii, jj]
            g = @view G[ii, jj]
            integrate_element(dad, elem_j, xj, pf, h, g)
        else
            for k in eachindex(elem_j.index)
                node = elem_j.index[k]
                r_node = dad.Nodes[node] - pf
                n_node = dad.Normal[node]
                jji = dim*(node-1)+1:node*dim
                U, T = fundamental(dad, r_node, n_node)
                wjk = elem_j.Jacobian[k] * dad.elem_weight[k]
                H[ii, jji] .= T .* wjk
                G[ii, jji] .= U .* wjk
            end
        end
    end
    return nothing
end

function H_G_full_direct_3d_vec!(dad::BEMdata{<:Vectorial}; npg=12, threaded=true)
    qsi, w = gausslegendre(npg)
    dim = dad.dimension
    H = zeros(dim * dad.nt, dim * dad.nt)
    G = zeros(dim * dad.nt, dim * dad.n)
    set_cache!(dad; H, G, qsi, w)
    elems = dad.elements
    Xel = [element_geometry(dad, e) for e in elems]
    if threaded && Threads.nthreads() > 1
        Threads.@threads for i in 1:dad.nt
            _assemble_row_3d_vec!(dad, H, G, i, elems, Xel, qsi, w)
        end
    else
        @showprogress "Assembling H and G (3D vec)" for i in 1:dad.nt
            _assemble_row_3d_vec!(dad, H, G, i, elems, Xel, qsi, w)
        end
    end
    @views for i in 1:dad.nt
        ii = dim*(i-1)+1:dim*i
        H[ii, ii] .= 0.0
        for j in 1:dim
            H[ii, ii[j]] .= -sum(H[ii, j:dim:end]; dims=2)
        end
    end
    return H, G
end

function _assemble_row_3d_vec!(dad, H, G, i, elems, Xel, qsi, w)
    dim = dad.dimension
    pf = point(dad, i)
    ii = dim*(i-1)+1:dim*i
    @inbounds for (ej, elem_j) in enumerate(elems)
        xj = Xel[ej]
        jj = dim*(elem_j.index[1]-1)+1:elem_j.index[end]*dim
        r0 = euclidean(pf, xj[1])
        if r0 < 2 * elem_j.Length
            h = @view H[ii, jj]
            g = @view G[ii, jj]
            integrate_element(dad, elem_j, xj, pf, qsi, w, h, g)
        else
            for k in eachindex(elem_j.index)
                node = elem_j.index[k]
                r_node = dad.Nodes[node] - pf
                n_node = dad.Normal[node]
                jji = dim*(node-1)+1:node*dim
                U, T = fundamental(dad, r_node, n_node)
                wjk = elem_j.Jacobian[k] * dad.elem_weight[k]
                H[ii, jji] .= T .* wjk
                G[ii, jji] .= U .* wjk
            end
        end
    end
    return nothing
end

# =============================================================================
# Element integration
# =============================================================================

"""
    integrate_element(dad, elem, x, pf, h, g; f=fundamental)

Integrate double/single layer contributions of one boundary element for source
`pf` into local vectors `h`, `g` (length = `#nodes` of `elem`).

Uses Dumont singular integration when applicable, otherwise sinh-transformed
Gauss quadrature. Formerly named `integraelem`.
"""
function integrate_element(dad::BEMdata{<:Scalar}, elem, x::Vector{Point2D}, pf::Point2D, h, g, f=fundamental)
    qsi, w = _quad_rule(dad)
    iso_iga = elem.controls !== nothing && dad.element_type isa Bernstein &&
        length(elem) == length(elem.controls)
    if supports_dumont(dad) && f === fundamental && (elem.controls === nothing || iso_iga)
        # Dumont: geometry nodes == field DOF nodes (Lagrange or isoparametric Bézier)
        xgeom = iso_iga ? elem.controls : x
        integraelem_dumont!(h, g, dad, elem, xgeom, pf, qsi, w)
        return nothing
    end
    # sinh near-field (elasticity-like scalars / custom fundamentals / IGA)
    eta, ww = transform(dad, elem, pf)
    N, dN, pg, dxdqsi = _elem_field_and_geom(dad, elem, x, eta)
    r = pg .- Ref(pf)
    dgamadqsi = norm.(dxdqsi)
    t = dxdqsi ./ dgamadqsi
    @inbounds for i in eachindex(eta)
        Tasti, Qasti = f(dad, r[i], tan2normal(t[i]))
        wi = dgamadqsi[i] * ww[i]
        for j in 1:length(elem)
            h[j] += N[i, j] * Qasti * wi
            g[j] += N[i, j] * Tasti * wi
        end
    end
    return nothing
end

"""
Field shape functions `N` (n_q × n_dof) and geometry `pg`, `dx/dξ`.

# IGA / Bézier
When `elem.controls` is set and `dad.element_type isa Bernstein` with
`length(elem) == length(controls)` (**isoparametric**): the same Bernstein
basis is used for field and geometry.
"""
function _elem_field_and_geom(dad, elem, x, eta)
    if elem.controls === nothing
        N, dN = element_shapefun(dad.element_type, elem, eta)
        pg = N * x
        dx = dN * x
        return N, dN, pg, dx
    end
    ctrl = elem.controls
    Ng, dNg = element_shapefun(dad.element_type, elem, eta)
    pg = [sum(Ng[i, a] * ctrl[a] for a in eachindex(ctrl)) for i in 1:size(Ng, 1)]
    dx = [sum(dNg[i, a] * ctrl[a] for a in eachindex(ctrl)) for i in 1:size(dNg, 1)]

    n_dof = length(elem)
    isoparametric = dad.element_type isa Bernstein && n_dof == length(ctrl) &&
        size(Ng, 2) == n_dof
    if isoparametric
        # field ≡ geometry basis (Bezier isoparametric collocation)
        return Ng, dNg, pg, dx
    end
    # hybrid fallback: Legendre field on collocation count
    poly_f = Legendre(max(n_dof - 1, 0))
    N, dN = shapefun(poly_f, eta)
    if size(N, 2) != n_dof
        ncol = min(size(N, 2), n_dof)
        N = N[:, 1:ncol]
        dN = dN[:, 1:ncol]
    end
    return N, dN, pg, dx
end

"""Gauss rule from cache, or default."""
function _quad_rule(dad)
    if has_cache(dad, :qsi) && has_cache(dad, :w)
        return dad.qsi, dad.w
    end
    return gausslegendre(12)
end

function integrate_element(dad::BEMdata{<:Scalar}, elem, x::Vector{Point3D}, pf::Point3D, qsi, w, h, g)
    # flattened near-field: polar+sinh when d/L small, else tensor sinh
    η1, η2, ww = transform_surface(dad, elem, pf; qsi2=qsi)
    N, dN1, dN2 = shapefun2D_points(dad.element_type, η1, η2)
    pg = N * x
    r = pg .- Ref(pf)
    dx1 = dN1 * x
    dx2 = dN2 * x
    dx = cross.(dx1, dx2)
    dgamadqsi = norm.(dx)
    nrm = dx ./ dgamadqsi
    @inbounds for ind in eachindex(ww)
        Tasti, Qasti = fundamental(dad, r[ind], nrm[ind])
        wi = dgamadqsi[ind] * ww[ind]
        for j in 1:length(elem)
            h[j] += N[ind, j] * Qasti * wi
            g[j] += N[ind, j] * Tasti * wi
        end
    end
    return nothing
end

function integrate_element(dad::BEMdata{<:Vectorial}, elem, x::Vector{Point2D}, pf::Point2D, h, g, f=fundamental)
    qsi, w = _quad_rule(dad)
    if supports_dumont(dad) && dad.dimension == 2 && f === fundamental && elem.controls === nothing
        integraelem_dumont!(h, g, dad, elem, x, pf, qsi, w)
        return nothing
    end
    # adaptive sinh (3D vectorial or custom fundamental / IGA)
    eta, ww = transform(dad, elem, pf)
    N, dN, pg, dxdqsi = _elem_field_and_geom(dad, elem, x, eta)
    r = pg .- Ref(pf)
    dgamadqsi = norm.(dxdqsi)
    t = dxdqsi ./ dgamadqsi
    dim = dad.dimension
    @inbounds for i in eachindex(eta)
        U, Tker = f(dad, r[i], tan2normal(t[i]))
        wi = dgamadqsi[i] * ww[i]
        for j in 1:length(elem)
            cols = dim*(j-1)+1:dim*j
            for d in 1:dim
                h[d, cols] .+= N[i, j] * Tker[d, :] * wi
                g[d, cols] .+= N[i, j] * U[d, :] * wi
            end
        end
    end
    return nothing
end

function integrate_element(dad::BEMdata{<:Vectorial}, elem, x::Vector{Point3D}, pf::Point3D, qsi, w, h, g)
    η1, η2, ww = transform_surface(dad, elem, pf; qsi2=qsi)
    N, dN1, dN2 = shapefun2D_points(dad.element_type, η1, η2)
    pg = N * x
    r = pg .- Ref(pf)
    dx1 = dN1 * x
    dx2 = dN2 * x
    dx = cross.(dx1, dx2)
    dgamadqsi = norm.(dx)
    nrm = dx ./ dgamadqsi
    dim = dad.dimension
    @inbounds for ind in eachindex(ww)
        U, Tker = fundamental(dad, r[ind], nrm[ind])
        wi = dgamadqsi[ind] * ww[ind]
        for j in 1:length(elem)
            cols = dim*(j-1)+1:dim*j
            for d in 1:dim
                h[d, cols] .+= N[ind, j] * Tker[d, :] * wi
                g[d, cols] .+= N[ind, j] * U[d, :] * wi
            end
        end
    end
    return nothing
end

# backward-compatible alias
const integraelem = integrate_element
