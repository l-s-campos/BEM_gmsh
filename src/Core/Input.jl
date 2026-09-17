export format2d, format3d, formatdata, discontinuous_nodes_weights
export DomainCell, extract_domain_cells, polygon_area, point_in_polygon, tan2normal
export with_gmsh, gmsh_ensure!, gmsh_release!

# ---------------------------------------------------------------------------
# Gmsh session (refcount so nested format2d / export share one API)
# ---------------------------------------------------------------------------

const _GMSH_REFS = Ref(0)
const _GMSH_OWNED = Ref(false)

"""
    gmsh_ensure!(; verbosity=1, terminal=nothing)

Initialize Gmsh if needed and bump the refcount. Pair with [`gmsh_release!`](@ref).
Adopts an already-initialized session without calling `gmsh.initialize` again.
"""
function gmsh_ensure!(; verbosity::Integer = 1, terminal = nothing)
    if _GMSH_REFS[] == 0
        already = try
            Bool(gmsh.isInitialized())
        catch
            false
        end
        if already
            _GMSH_OWNED[] = false
        else
            gmsh.initialize()
            _GMSH_OWNED[] = true
        end
        try
            gmsh.option.setNumber("General.Verbosity", Float64(verbosity))
        catch
        end
        if terminal !== nothing
            try
                gmsh.option.setNumber("General.Terminal", Float64(terminal))
            catch
            end
        end
    end
    _GMSH_REFS[] += 1
    return nothing
end

"""
    gmsh_release!()

Drop one refcount from [`gmsh_ensure!`](@ref). Finalizes Gmsh when the count
hits zero, unless the session was adopted (not owned by us).
"""
function gmsh_release!()
    _GMSH_REFS[] <= 0 && return nothing
    _GMSH_REFS[] -= 1
    if _GMSH_REFS[] == 0 && _GMSH_OWNED[]
        try
            gmsh.finalize()
        catch
        end
        _GMSH_OWNED[] = false
    end
    return nothing
end

"""
    with_gmsh(f; verbosity=1, terminal=nothing)

Run `f()` inside a refcounted Gmsh session. Nested `with_gmsh` calls share one
API instance; finalize runs only when the outermost block exits.
"""
function with_gmsh(f; kwargs...)
    gmsh_ensure!(; kwargs...)
    try
        return f()
    finally
        gmsh_release!()
    end
end

"""
    discontinuous_nodes_weights(p) -> (ξ, w)

Gauss–Legendre nodes/weights on ``[-1,1]`` (`p+1` points) for discontinuous
collocation.
"""
function discontinuous_nodes_weights(p::Integer)
    p >= 1 || error("degree must be ≥ 1, got $p")
    return gausslegendre(p + 1)
end

"""
    format2d(filename, properties; tipo=nothing, …) -> BEMdata

Build 2D [`BEMdata`](@ref) from a Gmsh `.msh` / `.geo`.

Discontinuous Gauss–Legendre collocation on mesh edges (`Legendre(p)`).

`tipo` is the field degree. Default (`nothing`): use the mesh 1-D order.
If set and different from the mesh, calls `gmsh.model.mesh.setOrder(tipo)`.
That upgrade does **not** snap new nodes onto CAD — mesh curved edges at
`ordem=tipo` *before* writing the `.msh`, otherwise quadratic collocation
sits on straight chords (constant `n` / `gap0` per element).

BCs come from physical group names on 1-D entities (`"0;T"` / `"1;q"` scalar;
`"tx;vx;ty;vy"` elasticity; 4-token Kirchhoff; 6-token FSDT). Dual-BEM
crack faces use type 5 (`"5;2"` / `"5;3"`, repeated once per DOF pair).
"""
function format2d(
        filename::AbstractString,
        properties::Problem;
        tipo = nothing,
        pontointerno = true,
        finalize::Bool = true,
        reopen::Bool = true,
        discretization::Symbol = :lagrange,
        collocation::Symbol = :legendre,
    )
    disc = Symbol(discretization)
    disc in (:lagrange, :Lagrange) || throw(ArgumentError(
        "unknown discretization=$(repr(discretization)); only :lagrange is supported",
    ))
    return format2d_lagrange(
        filename, properties; tipo, pontointerno, finalize, reopen,
        collocation=collocation,
    )
end

"""Alias of [`format2d`](@ref) (mesh → `BEMdata`)."""
const formatdata = format2d






function format2d_lagrange(
        filename::AbstractString,
        properties::Problem;
        tipo = nothing,
        pontointerno = true,
        finalize::Bool = true,
        reopen::Bool = true,
        collocation::Symbol = :legendre,
    )
    _gmsh_open_model!(filename; reopen)
    try
        elemTypes, elemTags, elemNodeTags, gmsh_nodes = _gmsh_boundary_lines!(tipo)
        nelem = length(elemTags[1])
        p, n_geo, reorder = _gmsh_line_layout(Int(elemTypes[1]))
        n_per = p + 1
        n_col = n_per * nelem
        dof = n_dof(properties, 2)

        nodes = Vector{Point2D}(undef, n_col)
        normal = Vector{Point2D}(undef, n_col)
        elements = Vector{Element}(undef, nelem)
        BC = ones(Int, dof * n_col)
        BV = zeros(Float64, dof * n_col)

        col = Symbol(collocation)
        col === :legendre || throw(ArgumentError(
            "collocation must be :legendre (Gauss–Legendre); got $(repr(col))"))
        qsi, wi = discontinuous_nodes_weights(p)
        poly = Legendre(p)
        Ngeo, dNgeo = shapefun(Equispaced(p), qsi)

        for (i_elem, el_tag) in enumerate(elemTags[1])
            tags = _line_node_tags(elemNodeTags[1], i_elem, n_geo, reorder)
            X = Point2D.(getindex.(gmsh_nodes[tags], Ref(1:2)))
            idx = ((i_elem - 1) * n_per + 1):(i_elem * n_per)
            nodes[idx] .= Ngeo * X
            dx = dNgeo * X
            J = norm.(dx)
            normal[idx] = tan2normal.(dx ./ J)
            L = abs(dot(J, wi))
            bc_type, bc_value = _entity_bc(el_tag, dof)
            _assign_bc!(BC, BV, idx, bc_type, bc_value, dof)
            _, _, _, etag = gmsh.model.mesh.getElement(el_tag)
            elements[i_elem] = Element(;
                index = collect(Int64, idx),
                Jacobian = collect(Float64, J),
                Length = Float64(L),
                Region = Int64(etag),
                geo = collect(X))
        end

        cells = _gmsh_domain_cells(gmsh_nodes)
        _orient_normals_outward!(normal, elements, nodes, cells)
        internal = pontointerno ? [c.centroid for c in cells] : Point2D[]
        _maybe_write_msh(filename)
        dad = _bemdata_2d(filename, elements, poly, wi, nodes, normal, properties, BC, BV, internal)
        isempty(cells) || set_cache!(dad; cells=cells)
        return dad
    finally
        finalize && gmsh_release!()
    end
end


# ---------------------------------------------------------------------------
# Gmsh helpers
# ---------------------------------------------------------------------------

function _gmsh_open_model!(filename::AbstractString; reopen::Bool)
    gmsh_ensure!(verbosity = 1)
    if reopen
        gmsh.clear()
        gmsh.open(filename)
    end
    try
        gmsh.model.geo.synchronize()
    catch
    end
    return nothing
end

"""1-D boundary elements, optionally re-ordered to `tipo`. Returns mesh arrays + nodes."""
function _gmsh_boundary_lines!(tipo)
    elemTypes, elemTags, elemNodeTags = gmsh.model.mesh.getElements(1, -1)
    if isempty(elemTypes) || isempty(elemTags) || isempty(elemTags[1])
        try
            gmsh.model.mesh.generate(1)
        catch
        end
        elemTypes, elemTags, elemNodeTags = gmsh.model.mesh.getElements(1, -1)
    end
    if isempty(elemTypes) || isempty(elemTags) || isempty(elemTags[1])
        error("format2d: no 1D mesh elements")
    end

    p_mesh = _gmsh_line_layout(Int(elemTypes[1]))[1]
    if tipo !== nothing
        p_req = Int(tipo)
        p_req >= 1 || throw(ArgumentError("tipo must be ≥ 1, got $p_req"))
        if p_req != p_mesh
            try
                gmsh.model.mesh.setOrder(p_req)
            catch e
                error("format2d: cannot setOrder($p_req) from mesh order $p_mesh: $e")
            end
            elemTypes, elemTags, elemNodeTags = gmsh.model.mesh.getElements(1, -1)
            p_mesh = _gmsh_line_layout(Int(elemTypes[1]))[1]
            p_mesh == p_req || error("format2d: setOrder($p_req) left mesh at order $p_mesh")
        end
    end

    _, coords, _ = gmsh.model.mesh.getNodes()
    gmsh_nodes = reinterpret(SVector{3,Float64}, coords) |> collect
    return elemTypes, elemTags, elemNodeTags, gmsh_nodes
end

"""Gmsh 1-D type → `(order, n_nodes, parametric reorder | nothing)`."""
function _gmsh_line_layout(etype::Integer)
    etype == 1 && return 1, 2, nothing
    etype == 8 && return 2, 3, (1, 3, 2)           # ends, mid
    etype == 26 && return 3, 4, (1, 3, 4, 2)
    etype == 27 && return 4, 5, (1, 3, 4, 5, 2)
    etype == 28 && return 5, 6, (1, 3, 4, 5, 6, 2)
    try
        _, _, order, numv, _, _ = gmsh.model.mesh.getElementProperties(etype)
        p, nn = Int(order), Int(numv)
        nn >= 3 && return p, nn, Tuple(vcat(1, 3:nn, 2))
        return p, nn, nothing
    catch
        error("Unsupported 1D Gmsh element type $etype")
    end
end

@inline function _line_node_tags(flat, i_elem, n_geo, reorder)
    raw = flat[((i_elem - 1) * n_geo + 1):(i_elem * n_geo)]
    return reorder === nothing ? raw : raw[collect(reorder)]
end

function _entity_bc(el_tag, dof)
    _, _, dim, etag = gmsh.model.mesh.getElement(el_tag)
    groups = gmsh.model.getPhysicalGroupsForEntity(dim, etag)
    if isempty(groups)
        return ones(Int, dof), zeros(dof)
    end
    return parse_pairs(gmsh.model.getPhysicalName(dim, groups[1]))
end


"""Piecewise-constant domain cell (Gmsh 2-D element, corner polygon)."""
struct DomainCell
    verts::Vector{Point2D}
    centroid::Point2D
    area::Float64
end

"""Signed area of a closed polyline (CCW > 0). `verts` is not repeated."""
function polygon_area(verts::AbstractVector{<:SVector{2}})
    n = length(verts)
    n < 3 && return 0.0
    a = 0.0
    @inbounds for i in 1:n
        j = i == n ? 1 : i + 1
        a += verts[i][1] * verts[j][2] - verts[j][1] * verts[i][2]
    end
    return 0.5 * a
end
const _polygon_area = polygon_area

"""Corner vertices of a Gmsh 2-D element (`etype` + nodal coords)."""
function _cell_corners(etype::Integer, pts::AbstractVector{Point2D})
    npe = length(pts)
    # triangles: 2 (lin), 9 (quad), 21/23 (higher) — first 3 are corners
    # quads: 3 (lin), 10/16 (higher) — first 4 are corners
    if etype == 2 || npe == 3
        return Point2D[pts[1], pts[2], pts[3]]
    elseif etype == 3 || npe == 4
        return Point2D[pts[1], pts[2], pts[3], pts[4]]
    elseif etype in (9, 21, 23) || npe == 6
        return Point2D[pts[1], pts[2], pts[3]]
    elseif etype in (10, 16) || npe >= 8
        return Point2D[pts[1], pts[2], pts[3], pts[4]]
    end
    return collect(pts)
end

"""Even–odd ray test. Boundary counts as outside. Used to orient `format2d` normals."""
function point_in_polygon(p::SVector{2}, verts::AbstractVector{<:SVector{2}})
    nv = length(verts)
    nv < 3 && return false
    inside = false
    j = nv
    @inbounds for i in 1:nv
        yi, yj = verts[i][2], verts[j][2]
        if (yi > p[2]) != (yj > p[2])
            xint = (verts[j][1] - verts[i][1]) * (p[2] - yi) / (yj - yi + 1e-30) + verts[i][1]
            p[1] < xint && (inside = !inside)
        end
        j = i
    end
    return inside
end
const _point_in_poly = point_in_polygon

function _point_in_cells(p::SVector{2}, cells::Vector{DomainCell})
    @inbounds for c in cells
        point_in_polygon(p, c.verts) && return true
    end
    return false
end

"""Flip a 3-D element if ``n`` points toward the nearest volume centroid."""
function _orient_normals_outward_3d!(normal, elements, nodes, centroids)
    isempty(centroids) && return nothing
    @inbounds for el in elements
        i0 = el.index[1]
        p = nodes[i0]
        c = _nearest_point(centroids, p)
        dot(normal[i0], p - c) < 0 || continue
        for i in el.index
            normal[i] = -normal[i]
        end
    end
    return nothing
end

function _nearest_point(pts, p)
    imin = 1
    dmin = Inf
    @inbounds for k in eachindex(pts)
        q = pts[k]
        d = abs2(p[1] - q[1]) + abs2(p[2] - q[2]) + abs2(p[3] - q[3])
        if d < dmin
            dmin = d
            imin = k
        end
    end
    return pts[imin]
end

"""Flip per-element normals so a step ``-n`` lands in a 2-D cell (outward ``n``)."""
function _orient_normals_outward!(normal, elements, nodes, cells)
    isempty(cells) && return nothing
    @inbounds for el in elements
        i0 = el.index[1]
        δ = 0.1 * float(el.Length)
        δ < 1e-16 && continue
        ptest = nodes[i0] - δ * normal[i0]
        _point_in_cells(ptest, cells) && continue
        for i in el.index
            normal[i] = -normal[i]
        end
    end
    return nothing
end

"""Gmsh surface elements as [`DomainCell`](@ref)s. Empty if no 2-D mesh."""
function _gmsh_domain_cells(gmsh_nodes)
    et, etags, enodes = gmsh.model.mesh.getElements(2, -1)
    (isempty(et) || isempty(etags)) && return DomainCell[]
    cells = DomainCell[]
    for (ity, etype) in enumerate(et)
        tags = etags[ity]
        nod = enodes[ity]
        isempty(tags) && continue
        npe = length(nod) ÷ length(tags)
        npe < 3 && continue
        for i in eachindex(tags)
            raw = nod[((i - 1) * npe + 1):(i * npe)]
            pts = Point2D[Point2D(gmsh_nodes[t][1], gmsh_nodes[t][2]) for t in raw]
            verts = _cell_corners(Int(etype), pts)
            c = mean(verts)
            a = abs(_polygon_area(verts))
            a < 1e-18 && continue
            push!(cells, DomainCell(verts, c, a))
        end
    end
    return cells
end

"""If no 3-D elements exist, add a volume from the closed surface and `generate(3)`."""
function _ensure_volume_mesh!()
    _has_3d_elements() && return true
    if !isempty(gmsh.model.getEntities(3))
        try
            gmsh.model.mesh.generate(3)
        catch
        end
        _has_3d_elements() && return true
    end
    surfs = gmsh.model.getEntities(2)
    isempty(surfs) && return false
    tags = Int[Int(s[2]) for s in surfs]
    ok = false
    try
        sl = gmsh.model.geo.addSurfaceLoop(tags)
        gmsh.model.geo.addVolume([sl])
        gmsh.model.geo.synchronize()
        ok = true
    catch
    end
    if !ok
        try
            sl = gmsh.model.occ.addSurfaceLoop(tags)
            gmsh.model.occ.addVolume([sl])
            gmsh.model.occ.synchronize()
            ok = true
        catch e
            @warn "format3d: cannot build a volume from the surface mesh" exception = e
            return false
        end
    end
    try
        gmsh.model.mesh.generate(3)
    catch e
        @warn "format3d: gmsh generate(3) failed; no interior points" exception = e
        return false
    end
    return _has_3d_elements()
end

function _has_3d_elements()
    _, etags, _ = gmsh.model.mesh.getElements(3, -1)
    return any(!isempty, etags)
end

"""Centroids of 3-D mesh elements (all types). Empty if none."""
function _gmsh_volume_centroids()
    et, etags, enodes = gmsh.model.mesh.getElements(3, -1)
    (isempty(et) || isempty(etags)) && return Point3D[]
    ntags, coords, _ = gmsh.model.mesh.getNodes()
    pts = reinterpret(SVector{3,Float64}, coords)
    tag2pt = Dict{Int,Point3D}(Int(ntags[i]) => Point3D(pts[i]) for i in eachindex(ntags))
    out = Point3D[]
    for ity in eachindex(et)
        tags = etags[ity]
        nod = enodes[ity]
        isempty(tags) && continue
        npe = length(nod) ÷ length(tags)
        npe < 4 && continue
        for i in eachindex(tags)
            raw = nod[((i - 1) * npe + 1):(i * npe)]
            c = zero(Point3D)
            for t in raw
                c += tag2pt[Int(t)]
            end
            push!(out, c / npe)
        end
    end
    return out
end

"""Centroids of mesh entities of topological dimension `dim` (Point2D)."""
function _surface_centroids(gmsh_nodes, dim::Integer)
    dim == 2 && return [c.centroid for c in _gmsh_domain_cells(gmsh_nodes)]
    et, etags, enodes = gmsh.model.mesh.getElements(dim, -1)
    (isempty(et) || isempty(etags) || isempty(etags[1])) && return Point2D[]
    npe = length(enodes[1]) ÷ max(length(etags[1]), 1)
    out = Vector{Point2D}(undef, length(etags[1]))
    for i in eachindex(etags[1])
        tags = enodes[1][((i - 1) * npe + 1):(i * npe)]
        pts = Point2D.(getindex.(gmsh_nodes[tags], Ref(1:2)))
        out[i] = mean(pts)
    end
    return out
end

function extract_domain_cells(dad::BEMdata)
    has_cache(dad, :cells) && return dad.cells::Vector{DomainCell}
    return DomainCell[]
end

function _maybe_write_msh(filename)
    lowercase(splitext(filename)[2]) == ".geo" || return nothing
    try
        gmsh.write(splitext(filename)[1] * ".msh")
    catch
    end
    return nothing
end

function _bemdata_2d(filename, ELEM, poly, wi, NOS, normal, properties, BC, BV, internal)
    n = length(NOS)
    ni = length(internal)
    return BEMdata(
        basename(splitext(filename)[1]),
        2,
        ELEM,
        poly,
        SVector{length(wi)}(Float64.(wi)),
        ni == 0 ? NOS : vcat(NOS, internal),
        normal,
        properties,
        BC,
        BV,
        n,
        ni,
        n + ni,
        BEMCache(),
    )
end

function _assign_bc!(BC, BV, idx, tipoCDC, valorCDC, dof)
    if dof == 1
        t0 = Int(tipoCDC[1])
        v0 = Float64(valorCDC[1])
        for i in idx
            BC[i] = t0
            BV[i] = v0
        end
    else
        tc = ones(Int, dof)
        vc = zeros(Float64, dof)
        np = min(length(tipoCDC), dof)
        tc[1:np] .= Int.(tipoCDC[1:np])
        vc[1:np] .= Float64.(valorCDC[1:np])
        for i in idx
            for d in 1:dof
                BC[dof * (i - 1) + d] = tc[d]
                BV[dof * (i - 1) + d] = vc[d]
            end
        end
    end
    return nothing
end




function parse_pairs(s::AbstractString)
    parts = strip.(split(s, ';'; keepempty=false))
    # allow bare labels without BC data
    if length(parts) < 2 || isnothing(tryparse(Int, parts[1]))
        return [1], [0.0]
    end
    n_pairs = length(parts) ÷ 2
    ints = Vector{Int}(undef, n_pairs)
    floats = Vector{Float64}(undef, n_pairs)
    for (k, i_part) in enumerate(1:2:2n_pairs)
        num_i = tryparse(Int, parts[i_part])
        num_f = tryparse(Float64, parts[i_part+1])
        (isnothing(num_i) || isnothing(num_f)) &&
            throw(ArgumentError("Bad BC pair in '$s'"))
        ints[k] = num_i
        floats[k] = num_f
    end
    return ints, floats
end

tan2normal(t::Point2D) = SVector{2,Float64}(t[2], -t[1])

"""
    format3d(filename, properties; tipo=1, pontointerno=true) -> BEMdata

Read a 3D Gmsh surface mesh and build [`BEMdata`](@ref).

Supported surface elements: 4-node quads (type 3), 9-node quads (type 10),
and linear triangles (type 2) as collapsed 4-node quads (`tipo=1` only).

`pontointerno=true` (default): if the model has no 3-D elements, wrap the
closed surface into a volume and call `gmsh.model.mesh.generate(3)`, then
place an internal collocation point at each volume-element centroid.
"""
function format3d(
        filename::AbstractString,
        properties::Problem;
        tipo = 1,
        pontointerno = true,
        finalize::Bool = true,
        reopen::Bool = true,
    )
    _gmsh_open_model!(filename; reopen)
    try
    _, coords, _ = gmsh.model.mesh.getNodes()
    gmsh_nodes = reinterpret(SVector{3,Float64}, coords) |> collect

    elemTypes, elemTags, elemNodeTags = gmsh.model.mesh.getElements(2, -1)
    name, dim, order, numv, parv, _ = gmsh.model.mesh.getElementProperties(elemTypes[1])
    et0 = Int(elemTypes[1])

    dof = n_dof(properties, 3)
    n_per = (tipo + 1)^2
    nelem = length(elemTags[1])
    n_col = n_per * nelem

    nodes = Vector{Point3D}(undef, n_col)
    normal = Vector{Point3D}(undef, n_col)
    BC = ones(Int, dof * n_col)
    BV = zeros(Float64, dof * n_col)
    elements = Vector{Element}(undef, nelem)
    qsi, wi = gausslegendre(tipo + 1)

    for (i_elem, el_tag) in enumerate(elemTags[1])
        if et0 == 3  # 4-node quad
            el_node_tags = elemNodeTags[1][((i_elem-1)*numv+1):(i_elem*numv)]
            el_node_tags = el_node_tags[[1, 2, 4, 3]]
            N, dNx, dNy = shapefun2D(Equispaced(1), qsi)
        elseif et0 == 10  # 9-node quad
            el_node_tags = elemNodeTags[1][((i_elem-1)*numv+1):(i_elem*numv)]
            el_node_tags = el_node_tags[[1, 5, 2, 8, 9, 6, 4, 7, 3]]
            N, dNx, dNy = shapefun2D(Equispaced(2), qsi)
        elseif et0 == 2  # 3-node triangle → collapsed 4-node quad (repeat last vertex)
            tipo == 1 || throw(ArgumentError(
                "linear triangles (Gmsh type 2) require tipo=1; got tipo=$tipo"))
            el_node_tags = elemNodeTags[1][((i_elem-1)*numv+1):(i_elem*numv)]
            el_node_tags = [el_node_tags[1], el_node_tags[2], el_node_tags[3], el_node_tags[3]]
            N, dNx, dNy = shapefun2D(Equispaced(1), qsi)
        elseif et0 == 9
            throw(ArgumentError(
                "quadratic triangles (Gmsh type 9) are not supported; use linear triangles or quads"))
        else
            error("Unsupported 2D element type $(elemTypes[1]) in 3D mesh")
        end
        el_local_nodes = Point3D.(getindex.(gmsh_nodes[el_node_tags], Ref(1:3)))
        idx = ((i_elem - 1) * n_per + 1):(i_elem * n_per)
        nodes[idx] .= N * el_local_nodes
        dx1 = dNx * el_local_nodes
        dx2 = dNy * el_local_nodes
        dgamadqsi = norm.(cross.(dx1, dx2))
        normal[idx] = cross.(dx1, dx2) ./ dgamadqsi
        tamanho = norm(el_local_nodes[1] - el_local_nodes[end])

        _, _, elementEntityDim, elementEntityTag = gmsh.model.mesh.getElement(el_tag)
        physicalgroups = gmsh.model.getPhysicalGroupsForEntity(elementEntityDim, elementEntityTag)
        if !isempty(physicalgroups)
            phys_name = gmsh.model.getPhysicalName(elementEntityDim, physicalgroups[1])
            bc_type, bc_value = parse_pairs(phys_name)
        else
            @warn "No physical group for element $el_tag"
            bc_type = ones(Int, dof)
            bc_value = zeros(dof)
        end
        _assign_bc!(BC, BV, idx, bc_type, bc_value, dof)
        elements[i_elem] = Element(collect(idx), dgamadqsi, tamanho, elementEntityTag)
    end

    have3d = _has_3d_elements()
    if pontointerno && !have3d
        have3d = _ensure_volume_mesh!()
    end
    cents = have3d ? _gmsh_volume_centroids() : Point3D[]
    _orient_normals_outward_3d!(normal, elements, nodes, cents)
    internalNodes = pontointerno ? cents : Point3D[]

    if lowercase(splitext(filename)[2]) == ".geo"
        msh_filename = splitext(filename)[1] * ".msh"
        gmsh.write(msh_filename)
        @info "Mesh saved to $msh_filename"
    end
    n = length(nodes)
    ni = length(internalNodes)
    nt = n + ni
    w2 = kron(wi, wi)
    return BEMdata(
        basename(splitext(filename)[1]),
        3,
        elements,
        Legendre(tipo),
        SVector{length(w2)}(w2),
        isempty(internalNodes) ? nodes : vcat(nodes, internalNodes),
        normal,
        properties,
        BC,
        BV,
        n,
        ni,
        nt,
        BEMCache(),
    )
    finally
        finalize && gmsh_release!()
    end
end
