export format2d, format3d, format2d_iga, discontinuous_nodes_weights

"""
    discontinuous_nodes_weights(tipo) -> (ξ, w)

Gauss–Legendre nodes and weights on ``[-1,1]`` used as **discontinuous**
collocation / quadrature points by [`format2d`](@ref).

`tipo` only sets the **number of nodes**:
- `tipo = 1` → 2 Gauss points
- `tipo = 2` → 3 Gauss points
- `tipo = n` → `n+1` Gauss points

Geometry is interpolated from Gmsh continuous nodes via `Equispaced`.
"""
function discontinuous_nodes_weights(tipo::Integer)
    tipo >= 1 || error("tipo must be ≥ 1, got $tipo")
    return gausslegendre(tipo + 1)
end

"""
    format2d(filename, properties; tipo=1, pontointerno=true,
             discretization=:lagrange, iga_mode=:cad, iga_degree=nothing,
             iga_nel=nothing, finalize=true, reopen=true) -> BEMdata

Read a 2D Gmsh mesh (`.msh` / `.geo`) and build a [`BEMdata`](@ref).

# Discretization
- `discretization = :lagrange` (default) — discontinuous collocation: each
  boundary element owns `tipo+1` Gauss nodes (never shared). Geometry from
  Gmsh continuous Lagrange edges via [`Equispaced`](@ref).
- `discretization = :iga` | `:bezier` | `:nurbs` — **isogeometric** path:
  Bézier-extracted elements with [`Bernstein`](@ref) basis and the same
  [`Element`](@ref) interface (`index`, `Jacobian`, `Length`, `Region`) plus
  `extraction` / `nurbs_weights`. See `iga_mode`.

# IGA options (`discretization = :iga`)
- `iga_mode = :cad` (default) — for every physical boundary curve, sample the
  Gmsh CAD parametrization, build an open B-spline of degree `iga_degree`,
  apply [`bezier_extraction`](@ref), and place collocation at Gauss images.
  This is how Gmsh NURBS/BSpline geometry enters the BEM.
- `iga_mode = :bezier_mesh` — mesh with order `iga_degree`, convert each
  Lagrange edge to Bézier controls via [`lagrange_to_bezier_matrix`](@ref)
  (exact for polynomial edges; approximates CAD when Gmsh curved-meshed it).
- `iga_degree` — spline / Bézier degree (default `max(tipo, 2)`)
- `iga_nel` — elements per curve in `:cad` mode (default: from mesh size / `ndiv`)
- `tipo` — still sets the number of collocation Gauss points per element (`tipo+1`)

Boundary conditions are taken from **physical group names** on 1-D entities:

# Scalar (`Laplace`)
Name format `"type;value"`:
- `"0;T"` Dirichlet
- `"1;q"` Neumann (`q = -k ∂T/∂n`)
- `"3;0"` **interface** (perfect bond between subregions)
- `"4;μ"` **contact** candidate (value = friction coefficient)

# Vectorial (`Elasticity`)
Name format `"tx;vx;ty;vy"` with one pair per component, e.g.
`"0;0;0;0"` (fixed), `"1;0;1;0"` (traction-free),
`"3;0;3;0"` (interface), `"4;μ;4;μ"` (contact, μ in value slots),
`"5;2;5;2"` / `"5;3;5;3"` (**crack** dual-BEM faces: type 5,
value = equation 2 displacement BIE or 3 traction BIE).
`BC`/`BV` are stored per DOF: `[ux₁, uy₁, ux₂, uy₂, …]`.
"""
function format2d(
        filename::AbstractString,
        properties::Problem;
        tipo = 1,
        pontointerno = true,
        finalize::Bool = true,
        reopen::Bool = true,
        discretization::Symbol = :lagrange,
        iga_mode::Symbol = :cad,
        iga_degree = nothing,
        iga_nel = nothing,
    )
    disc = Symbol(discretization)
    if disc in (:iga, :bezier, :nurbs, :IGA, :Bezier, :NURBS)
        return format2d_iga(
            filename, properties;
            tipo, pontointerno, finalize, reopen,
            iga_mode = Symbol(iga_mode),
            iga_degree = iga_degree === nothing ? max(Int(tipo), 2) : Int(iga_degree),
            iga_nel,
        )
    end
    disc === :lagrange || disc === :Lagrange || throw(ArgumentError(
        "unknown discretization=$(repr(discretization)); use :lagrange or :iga",
    ))
    return format2d_lagrange(
        filename, properties; tipo, pontointerno, finalize, reopen,
    )
end

function format2d_lagrange(
        filename::AbstractString,
        properties::Problem;
        tipo = 1,
        pontointerno = true,
        finalize::Bool = true,
        reopen::Bool = true,
    )
    # `finalize=false` keeps Gmsh alive for multi-step pipelines.
    # `reopen=false` reads the model already loaded in the current session.
    if reopen
        if finalize
            gmsh.initialize()
            gmsh.option.setNumber("General.Verbosity", 1)
            gmsh.clear()
            gmsh.open(filename)
        else
            gmsh_ensure!(verbosity = 1)
            gmsh.clear()
            gmsh.open(filename)
        end
    else
        gmsh_is_alive() || gmsh_ensure!(verbosity = 1)
    end
    try
        gmsh.model.geo.synchronize()
    catch
        # OCC models are already synchronized
    end

    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    gmsh_nodes = reinterpret(SVector{3,Float64}, coords) |> collect

    elemTypes, elemTags, elemNodeTags = gmsh.model.mesh.getElements(1, -1)
    if isempty(elemTypes) || isempty(elemTags) || isempty(elemTags[1])
        error("format2d: no 1D mesh elements in $(repr(filename)). " *
              "Provide a meshed boundary (.msh) or a geometry the Gmsh API can parse.")
    end
    nelem = length(elemTags[1])
    n_per_elem = tipo + 1
    n_col = n_per_elem * nelem

    is_vec = properties isa Vectorial
    dof = is_vec ? 2 : 1

    NOS = Vector{Point2D}(undef, n_col)
    normal = Vector{Point2D}(undef, n_col)
    ELEM = Vector{Element}(undef, nelem)
    BC = ones(Int, dof * n_col)
    BV = zeros(Float64, dof * n_col)

    # Gauss nodes = element collocation nodes (discontinuous)
    qsi, wi = discontinuous_nodes_weights(tipo)

    for (i_elem, el_tag) in enumerate(elemTags[1])
        # Geometry from continuous Gmsh nodes; DOFs at Gauss qsi
        if elemTypes[1] == 1  # 2-node line (linear geometry)
            el_node_tags = elemNodeTags[1][((i_elem-1)*2+1):(i_elem*2)]
            N, dN = shapefun(Equispaced(1), qsi)
        elseif elemTypes[1] == 8  # 3-node quadratic line
            el_node_tags = elemNodeTags[1][((i_elem-1)*3+1):(i_elem*3)]
            el_node_tags = el_node_tags[[1, 3, 2]]
            N, dN = shapefun(Equispaced(2), qsi)
        else
            error("Unsupported 1D element type $(elemTypes[1])")
        end
        el_local_nodes = Point2D.(getindex.(gmsh_nodes[el_node_tags], Ref(1:2)))

        # each element owns its Gauss collocation nodes (no sharing)
        idx = ((i_elem - 1) * n_per_elem + 1):(i_elem * n_per_elem)
        NOS[idx] .= N * el_local_nodes
        dxdqsi = dN * el_local_nodes
        dgamadqsi = norm.(dxdqsi)
        t = dxdqsi ./ dgamadqsi
        normal[idx] = tan2normal.(t)
        tamanho = abs(dot(dgamadqsi, wi))

        _, _, elementEntityDim, elementEntityTag = gmsh.model.mesh.getElement(el_tag)
        physicalgroups = gmsh.model.getPhysicalGroupsForEntity(elementEntityDim, elementEntityTag)
        if !isempty(physicalgroups)
            phys_name = gmsh.model.getPhysicalName(elementEntityDim, physicalgroups[1])
            tipoCDC, valorCDC = parse_pairs(phys_name)
        else
            @warn "No physical group for element $el_tag — defaulting to Neumann 0"
            tipoCDC = ones(Int, dof)
            valorCDC = zeros(dof)
        end
        _assign_bc!(BC, BV, idx, tipoCDC, valorCDC, dof)
        ELEM[i_elem] = Element(collect(idx), collect(dgamadqsi), tamanho, elementEntityTag)
    end

    if pontointerno
        et, etags, enodes = gmsh.model.mesh.getElements(2, -1)
        internalNodes = Vector{Point2D}(undef, length(etags[1]))
        # try common quad (4) then triangle (3)
        npe = length(enodes[1]) ÷ max(length(etags[1]), 1)
        for (i_elem, _) in enumerate(etags[1])
            el_node_tags = enodes[1][((i_elem-1)*npe+1):(i_elem*npe)]
            el_local = Point2D.(getindex.(gmsh_nodes[el_node_tags], Ref(1:2)))
            internalNodes[i_elem] = mean(el_local)
        end
    else
        internalNodes = Point2D[]
    end

    if lowercase(splitext(filename)[2]) == ".geo"
        msh_filename = splitext(filename)[1] * ".msh"
        gmsh.write(msh_filename)
        @info "Mesh saved to $msh_filename"
    end
    # finalize=true → legacy own session; finalize=false → leave Gmsh up for caller
    finalize && gmsh.finalize()

    n = length(NOS)
    ni = length(internalNodes)
    nt = n + ni
    # Legendre(tipo) carries Dmat for shapefun / closest-point; collocation
    # positions themselves are the discontinuous nodes `qsi` stored on Nodes.
    return BEMdata(
        basename(splitext(filename)[1]),
        2,
        ELEM,
        Legendre(tipo),
        SVector{length(wi)}(Float64.(wi)),
        isempty(internalNodes) ? NOS : vcat(NOS, internalNodes),
        normal,
        properties,
        BC,
        BV,
        n,
        ni,
        nt,
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
        # ensure we have `dof` pairs; pad with Neumann 0
        tc = ones(Int, dof)
        vc = zeros(Float64, dof)
        np = min(length(tipoCDC), dof)
        tc[1:np] .= Int.(tipoCDC[1:np])
        vc[1:np] .= Float64.(valorCDC[1:np])
        for i in idx
            for d in 1:dof
                BC[dof*(i-1)+d] = tc[d]
                BV[dof*(i-1)+d] = vc[d]
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
"""
function format3d(
        filename::AbstractString,
        properties::Problem;
        tipo = 1,
        pontointerno = true,
        finalize::Bool = true,
        reopen::Bool = true,
    )
    if reopen
        if finalize
            gmsh.initialize()
            gmsh.option.setNumber("General.Verbosity", 1)
            gmsh.clear()
            gmsh.open(filename)
        else
            gmsh_is_alive() || gmsh_ensure!(verbosity = 1)
            gmsh.clear()
            gmsh.open(filename)
        end
    else
        gmsh_is_alive() || gmsh_ensure!(verbosity = 1)
    end
    try
        gmsh.model.geo.synchronize()
    catch
    end

    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    gmsh_nodes = reinterpret(SVector{3,Float64}, coords) |> collect

    elemTypes, elemTags, elemNodeTags = gmsh.model.mesh.getElements(2, -1)
    name, dim, order, numv, parv, _ = gmsh.model.mesh.getElementProperties(elemTypes[1])

    is_vec = properties isa Vectorial
    dof = is_vec ? 3 : 1
    n_per = (tipo + 1)^2
    nelem = length(elemTags[1])
    n_col = n_per * nelem

    NOS = Vector{Point3D}(undef, n_col)
    normal = Vector{Point3D}(undef, n_col)
    BC = ones(Int, dof * n_col)
    BV = zeros(Float64, dof * n_col)
    ELEM = Vector{Element}(undef, nelem)
    qsi, wi = gausslegendre(tipo + 1)

    for (i_elem, el_tag) in enumerate(elemTags[1])
        if elemTypes[1] == 3  # 4-node quad
            el_node_tags = elemNodeTags[1][((i_elem-1)*numv+1):(i_elem*numv)]
            el_node_tags = el_node_tags[[1, 2, 4, 3]]
            N, dNx, dNy = shapefun2D(Equispaced(1), qsi)
        elseif elemTypes[1] == 10  # 9-node quad
            el_node_tags = elemNodeTags[1][((i_elem-1)*numv+1):(i_elem*numv)]
            el_node_tags = el_node_tags[[1, 5, 2, 8, 9, 6, 4, 7, 3]]
            N, dNx, dNy = shapefun2D(Equispaced(2), qsi)
        else
            error("Unsupported 2D element type $(elemTypes[1]) in 3D mesh")
        end
        el_local_nodes = Point3D.(getindex.(gmsh_nodes[el_node_tags], Ref(1:3)))
        idx = ((i_elem - 1) * n_per + 1):(i_elem * n_per)
        NOS[idx] .= N * el_local_nodes
        dx1 = dNx * el_local_nodes
        dx2 = dNy * el_local_nodes
        dgamadqsi = norm.(cross.(dx1, dx2))
        normal[idx] = cross.(dx1, dx2) ./ dgamadqsi
        tamanho = euclidean(el_local_nodes[1], el_local_nodes[end])

        _, _, elementEntityDim, elementEntityTag = gmsh.model.mesh.getElement(el_tag)
        physicalgroups = gmsh.model.getPhysicalGroupsForEntity(elementEntityDim, elementEntityTag)
        if !isempty(physicalgroups)
            phys_name = gmsh.model.getPhysicalName(elementEntityDim, physicalgroups[1])
            tipoCDC, valorCDC = parse_pairs(phys_name)
        else
            @warn "No physical group for element $el_tag"
            tipoCDC = ones(Int, dof)
            valorCDC = zeros(dof)
        end
        _assign_bc!(BC, BV, idx, tipoCDC, valorCDC, dof)
        ELEM[i_elem] = Element(collect(idx), dgamadqsi, tamanho, elementEntityTag)
    end

    if pontointerno
        et, etags, enodes = gmsh.model.mesh.getElements(3, -1)
        if !isempty(etags)
            _, _, _, numv3, _, _ = gmsh.model.mesh.getElementProperties(et[1])
            internalNodes = Vector{Point3D}(undef, length(etags[1]))
            for (i_elem, _) in enumerate(etags[1])
                el_node_tags = enodes[1][((i_elem-1)*numv3+1):(i_elem*numv3)]
                el_local = Point3D.(getindex.(gmsh_nodes[el_node_tags], Ref(1:3)))
                internalNodes[i_elem] = mean(el_local)
            end
        else
            internalNodes = Point3D[]
        end
    else
        internalNodes = Point3D[]
    end

    if lowercase(splitext(filename)[2]) == ".geo"
        msh_filename = splitext(filename)[1] * ".msh"
        gmsh.write(msh_filename)
        @info "Mesh saved to $msh_filename"
    end
    finalize && gmsh.finalize()

    n = length(NOS)
    ni = length(internalNodes)
    nt = n + ni
    w2 = kron(wi, wi)
    return BEMdata(
        basename(splitext(filename)[1]),
        3,
        ELEM,
        Legendre(tipo),
        SVector{length(w2)}(w2),
        isempty(internalNodes) ? NOS : vcat(NOS, internalNodes),
        normal,
        properties,
        BC,
        BV,
        n,
        ni,
        nt,
        BEMCache(),
    )
end


# =============================================================================
# Isogeometric / Bézier format2d
# =============================================================================

"""
    format2d_iga(filename, properties; kwargs...) -> BEMdata

Build a 2D IGA-BEM model with Bézier-extracted elements. Prefer calling
[`format2d`](@ref)(...; `discretization=:iga`).
"""
function format2d_iga(
        filename::AbstractString,
        properties::Problem;
        tipo = 1,
        pontointerno = true,
        finalize::Bool = true,
        reopen::Bool = true,
        iga_mode::Symbol = :cad,
        iga_degree::Int = 2,
        iga_nel = nothing,
    )
    if reopen
        if finalize
            gmsh.initialize()
            gmsh.option.setNumber("General.Verbosity", 1)
            gmsh.clear()
            gmsh.open(filename)
        else
            gmsh_is_alive() || gmsh_ensure!(verbosity = 1)
            gmsh.clear()
            gmsh.open(filename)
        end
    else
        gmsh_is_alive() || gmsh_ensure!(verbosity = 1)
    end
    try
        gmsh.model.geo.synchronize()
    catch
    end
    try
        gmsh.model.occ.synchronize()
    catch
    end

    p = Int(iga_degree)
    p >= 1 || throw(ArgumentError("iga_degree must be ≥ 1"))
    n_col_per = tipo + 1
    qsi, wi = discontinuous_nodes_weights(tipo)
    poly_B = Bernstein(p)

    is_vec = properties isa Vectorial
    dof = is_vec ? 2 : 1

    dad = if iga_mode in (:cad, :nurbs, :CAD)
        _format2d_iga_cad(
            filename, properties;
            p, n_col_per, qsi, wi, poly_B, dof, pontointerno, iga_nel,
        )
    elseif iga_mode in (:bezier_mesh, :mesh)
        _format2d_iga_bezier_mesh(
            filename, properties;
            p, n_col_per, qsi, wi, poly_B, dof, pontointerno,
        )
    else
        error("unknown iga_mode=$(repr(iga_mode)); use :cad or :bezier_mesh")
    end

    if lowercase(splitext(filename)[2]) == ".geo"
        msh_filename = splitext(filename)[1] * ".msh"
        try
            gmsh.write(msh_filename)
        catch
        end
    end
    finalize && gmsh.finalize()
    return dad
end

function _curve_param_bounds(ctag::Integer)
    b = gmsh.model.getParametrizationBounds(1, ctag)
    if b isa Tuple
        return Float64(b[1][1]), Float64(b[2][1])
    else
        return Float64(b[1]), Float64(b[2])
    end
end

function _format2d_iga_cad(
        filename, properties;
        p, n_col_per, qsi, wi, poly_B, dof, pontointerno, iga_nel,
    )
    curve_bc = _physical_curve_bcs()
    isempty(curve_bc) && error(
        "IGA :cad mode needs physical curve groups (got none in $filename)",
    )

    NOS = Point2D[]
    normal = Point2D[]
    ELEM = Element[]

    for (ctag, _phys_name) in curve_bc
        umin, umax = _curve_param_bounds(ctag)
        n_el = iga_nel === nothing ? _default_nel_for_curve(ctag, p) : Int(iga_nel)
        n_el = max(Int(n_el), 1)
        Ξ = open_knot_vector(n_el, p; a = umin, b = umax)
        ncp = length(Ξ) - p - 1
        γ = greville_abscissae(Ξ, p)

        Xg = Matrix{Float64}(undef, ncp, 2)
        @inbounds for i in 1:ncp
            xyz = gmsh.model.getValue(1, ctag, [γ[i]])
            Xg[i, 1] = xyz[1]
            Xg[i, 2] = xyz[2]
        end
        A = zeros(ncp, ncp)
        @inbounds for i in 1:ncp, j in 1:ncp
            A[i, j] = bspline_basis(j, p, Ξ, γ[i])
        end
        Areg = A + 1e-12 * I
        cx = Areg \ Xg[:, 1]
        cy = Areg \ Xg[:, 2]
        Pcurve = [Point2D(cx[i], cy[i]) for i in 1:ncp]

        Cs, spans = bezier_extraction(Ξ, p)
        for (e, (first_cp, _, _)) in enumerate(spans)
            C = Cs[e]
            loc = collect(first_cp:(first_cp + p))
            P_loc = Pcurve[loc]
            w_loc = ones(Float64, p + 1)
            Pb = _mat_mul_points(C, P_loc)
            wb = map(x -> abs(x) < 1e-14 ? 1.0 : abs(x), C * w_loc)
            Ceye = Matrix{Float64}(I, p + 1, p + 1)

            # ---- isoparametric Bézier ----
            # DOFs at Bernstein Greville images; field ≡ geometry = Bernstein(Pb)
            Jg = Float64[]
            base = length(NOS)
            idx = collect((base + 1):(base + p + 1))
            for a in 1:(p + 1)
                ξg = poly_B.nodes[a]
                Ng, dNg = rationalize_shape(shapefun(poly_B, ξg)..., wb)
                xg = sum(Ng[1, b] * Pb[b] for b in 1:(p + 1))
                dxg = sum(dNg[1, b] * Pb[b] for b in 1:(p + 1))
                J = norm(dxg)
                push!(Jg, J)
                push!(NOS, xg)
                push!(normal, tan2normal(J > 0 ? dxg / J : Point2D(1.0, 0.0)))
            end
            Nq, dNq = rationalize_shape(shapefun(poly_B, qsi)..., wb)
            L = sum(norm(sum(dNq[k, b] * Pb[b] for b in 1:(p + 1))) * wi[k] for k in eachindex(wi))
            push!(ELEM, BezierElement(idx, Jg, Float64(L), Int(ctag), Ceye;
                weights = wb, controls = Pb))
        end
    end

    n = length(NOS)
    BC = ones(Int, dof * n)
    BV = zeros(Float64, dof * n)
    tag2bc = Dict{Int, String}(ctag => name for (ctag, name) in curve_bc)
    for elem in ELEM
        phys_name = get(tag2bc, elem.Region, "1;0")
        tipoCDC, valorCDC = parse_pairs(phys_name)
        _assign_bc!(BC, BV, elem.index, tipoCDC, valorCDC, dof)
    end

    internalNodes = pontointerno ? _internal_nodes_2d() : Point2D[]
    ni = length(internalNodes)
    # quadrature weights for far-field lumping at Greville DOFs
    w_el = fill(2 / (p + 1), p + 1)
    return BEMdata(
        basename(splitext(filename)[1]),
        2,
        ELEM,
        poly_B,
        SVector{p + 1}(w_el),
        isempty(internalNodes) ? NOS : vcat(NOS, internalNodes),
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

function _format2d_iga_bezier_mesh(
        filename, properties;
        p, n_col_per, qsi, wi, poly_B, dof, pontointerno,
    )
    try
        gmsh.model.mesh.setOrder(p)
    catch
    end
    et0, etags0, _ = gmsh.model.mesh.getElements(1, -1)
    if isempty(et0) || isempty(etags0) || isempty(etags0[1])
        gmsh.model.mesh.generate(1)
        try
            gmsh.model.mesh.setOrder(p)
        catch
        end
    end

    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    gmsh_nodes = reinterpret(SVector{3, Float64}, coords) |> collect

    elemTypes, elemTags, elemNodeTags = gmsh.model.mesh.getElements(1, -1)
    isempty(elemTypes) && error("no 1-D elements for IGA bezier_mesh in $filename")
    _, _, order, numv, _, _ = gmsh.model.mesh.getElementProperties(elemTypes[1])
    p_use = Int(order)
    poly_use = Bernstein(p_use)

    NOS = Point2D[]
    normal = Point2D[]
    ELEM = Element[]

    for (i_elem, el_tag) in enumerate(elemTags[1])
        el_node_tags = elemNodeTags[1][((i_elem - 1) * numv + 1):(i_elem * numv)]
        el_local = Point2D.(getindex.(gmsh_nodes[el_node_tags], Ref(1:2)))
        el_local = _gmsh_line_nodes_parametric(el_local, p_use)

        Pb = bezier_controls_from_lagrange(el_local, p_use)
        wb = ones(p_use + 1)
        Ceye = Matrix{Float64}(I, p_use + 1, p_use + 1)

        # isoparametric: DOFs at Greville images of Bernstein basis
        Jg = Float64[]
        base = length(NOS)
        idx = collect((base + 1):(base + p_use + 1))
        for a in 1:(p_use + 1)
            ξg = poly_use.nodes[a]
            Ng, dNg = rationalize_shape(shapefun(poly_use, ξg)..., wb)
            xg = sum(Ng[1, b] * Pb[b] for b in 1:(p_use + 1))
            dxg = sum(dNg[1, b] * Pb[b] for b in 1:(p_use + 1))
            J = norm(dxg)
            push!(Jg, J)
            push!(NOS, xg)
            push!(normal, tan2normal(J > 0 ? dxg / J : Point2D(1.0, 0.0)))
        end
        Nq, dNq = rationalize_shape(shapefun(poly_use, qsi)..., wb)
        L = sum(norm(sum(dNq[k, b] * Pb[b] for b in 1:(p_use + 1))) * wi[k] for k in eachindex(wi))

        _, _, _, entityTag = gmsh.model.mesh.getElement(el_tag)
        push!(ELEM, BezierElement(idx, Jg, Float64(L), Int(entityTag), Ceye;
            weights = wb, controls = Pb))
    end

    n = length(NOS)
    BC = ones(Int, dof * n)
    BV = zeros(Float64, dof * n)
    for (i_elem, el_tag) in enumerate(elemTags[1])
        _, _, _, entityTag = gmsh.model.mesh.getElement(el_tag)
        physicalgroups = gmsh.model.getPhysicalGroupsForEntity(1, entityTag)
        if !isempty(physicalgroups)
            phys_name = gmsh.model.getPhysicalName(1, physicalgroups[1])
            tipoCDC, valorCDC = parse_pairs(phys_name)
        else
            tipoCDC = ones(Int, dof)
            valorCDC = zeros(dof)
        end
        _assign_bc!(BC, BV, ELEM[i_elem].index, tipoCDC, valorCDC, dof)
    end

    internalNodes = pontointerno ? _internal_nodes_2d() : Point2D[]
    ni = length(internalNodes)
    p_use = isempty(ELEM) ? p : (length(ELEM[1]) - 1)
    w_el = fill(2 / (p_use + 1), p_use + 1)
    return BEMdata(
        basename(splitext(filename)[1]),
        2,
        ELEM,
        poly_use,
        SVector{p_use + 1}(w_el),
        isempty(internalNodes) ? NOS : vcat(NOS, internalNodes),
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

function _physical_curve_bcs()
    out = Tuple{Int, String}[]
    for (dim, tag) in gmsh.model.getPhysicalGroups(1)
        name = gmsh.model.getPhysicalName(dim, tag)
        for et in gmsh.model.getEntitiesForPhysicalGroup(dim, tag)
            push!(out, (Int(et), String(name)))
        end
    end
    return out
end

function _default_nel_for_curve(ctag::Integer, p::Integer)
    try
        et, etags, _ = gmsh.model.mesh.getElements(1, ctag)
        if !isempty(etags) && !isempty(etags[1])
            return max(length(etags[1]), 1)
        end
    catch
    end
    try
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(1, ctag)
        L = hypot(xmax - xmin, ymax - ymin)
        return max(Int(ceil(L / 0.1)), 2)
    catch
        return 4
    end
end

function _internal_nodes_2d()
    et, etags, enodes = gmsh.model.mesh.getElements(2, -1)
    (isempty(et) || isempty(etags) || isempty(etags[1])) && return Point2D[]
    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    gmsh_nodes = reinterpret(SVector{3, Float64}, coords) |> collect
    npe = length(enodes[1]) ÷ max(length(etags[1]), 1)
    internalNodes = Vector{Point2D}(undef, length(etags[1]))
    for (i_elem, _) in enumerate(etags[1])
        el_node_tags = enodes[1][((i_elem - 1) * npe + 1):(i_elem * npe)]
        el_local = Point2D.(getindex.(gmsh_nodes[el_node_tags], Ref(1:2)))
        internalNodes[i_elem] = mean(el_local)
    end
    return internalNodes
end

function _mat_mul_points(C::AbstractMatrix, P::Vector{Point2D})
    n = size(C, 1)
    m = size(C, 2)
    length(P) == m || throw(DimensionMismatch())
    out = Vector{Point2D}(undef, n)
    @inbounds for i in 1:n
        acc = Point2D(0.0, 0.0)
        for j in 1:m
            acc += C[i, j] * P[j]
        end
        out[i] = acc
    end
    return out
end

"""Reorder Gmsh line nodes to increasing parametric order (equispaced)."""
function _gmsh_line_nodes_parametric(nodes::Vector{Point2D}, p::Integer)
    length(nodes) == p + 1 || return nodes
    p == 1 && return nodes
    if p == 2 && length(nodes) == 3
        return Point2D[nodes[1], nodes[3], nodes[2]]
    end
    if p == 3 && length(nodes) == 4
        return Point2D[nodes[1], nodes[3], nodes[4], nodes[2]]
    end
    corners = nodes[1:2]
    interior = nodes[3:end]
    return vcat(Point2D[corners[1]], interior, Point2D[corners[2]])
end
