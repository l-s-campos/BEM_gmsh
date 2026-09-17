# Pacheco (2020) heat-conductor benchmarks and Coelho (2021) elasticity cases 3–5.
# Gmsh is used only to build the *initial* mesh; the optimizer then works on loops.

export pacheco_problem, pacheco_options
export pacheco_inverted_v, pacheco_asymmetric, pacheco_bridge, pacheco_cross
export mesh_pacheco_inverted_v, mesh_pacheco_asymmetric, mesh_pacheco_bridge, mesh_pacheco_cross
export coelho_problem, coelho_options
export coelho_cantilever, coelho_cc_beam, coelho_ss_beam
export mesh_coelho_cantilever, mesh_coelho_cc_beam, mesh_coelho_ss_beam
export portela_plate_hole, portela_heat_hole

function _straight(p0, p1, n_el; bc, value)
    n_el = max(Int(n_el), 1)
    verts = [Point2D(p0[1], p0[2]), Point2D(p1[1], p1[2])]
    return BoundarySegment(verts; bc=bc, value=value, n_el=n_el)
end

"""Sample a circular arc from `p0` to `p1` with signed radius (PG2 convention:
negative ⇒ centre to the right of p0→p1)."""
function _arc_verts(p0::Point2D, p1::Point2D, R::Float64, n::Int)
    chord = p1 - p0
    c = norm(chord)
    c < 1e-14 && return Point2D[p0, p1]
    Rabs = abs(R)
    Rabs < c / 2 && (Rabs = c / 2 + 1e-12)
    h = sqrt(max(Rabs^2 - (c / 2)^2, 0.0))
    mid = 0.5 * (p0 + p1)
    # unit left-of-chord
    t = chord / c
    n̂ = Point2D(-t[2], t[1])           # left
    # PG2: R < 0 → centre to the right
    centre = mid + (R < 0 ? -h : h) * n̂
    a0 = atan(p0[2] - centre[2], p0[1] - centre[1])
    a1 = atan(p1[2] - centre[2], p1[1] - centre[1])
    if R < 0
        # clockwise from p0 to p1
        a1 <= a0 && (a1 += 2π)
        # wait: right-of-chord centre is clockwise for typical bottom arc
        da = a1 - a0
        if da > π
            a1 -= 2π
        end
    else
        a1 < a0 && (a1 += 2π)
        da = a1 - a0
        if da > π
            a1 -= 2π
        end
    end
    θ = range(a0, a1; length=max(n, 3))
    return Point2D[Point2D(centre[1] + Rabs * cos(t), centre[2] + Rabs * sin(t)) for t in θ]
end

function _outer(segs; degree=2, k=1.0, nint=20, name="pacheco", properties=nothing)
    TopologyDesign([segs]; degree=degree, k=k, nx_int=nint, ny_int=nint, name=name,
        properties=properties)
end

# ----- problem 1: inverted V -------------------------------------------------

"""
    pacheco_inverted_v(; ne=20, nint=16, degree=2, k=1.0) -> TopologyDesign

Pacheco (2020) inverted-V heat conductor on the unit square: hot patches
on the bottom corners (`T=1`), cold patch on the top (`T=0`), insulated
elsewhere. Dirichlet edges are frozen against motion.
"""
function pacheco_inverted_v(; ne=20, nint=16, degree=2, k=1.0)
    Th, Tl = 1.0, 0.0
    n = x -> max(Int(round(x)), 2)
    segs = [
        _straight((0.0, 0.0), (0.2, 0.0), n(0.2ne); bc=0, value=Th),
        _straight((0.2, 0.0), (0.8, 0.0), n(0.6ne); bc=1, value=0.0),
        _straight((0.8, 0.0), (1.0, 0.0), n(0.2ne); bc=0, value=Th),
        _straight((1.0, 0.0), (1.0, 1.0), n(ne);     bc=1, value=0.0),
        _straight((1.0, 1.0), (0.65, 1.0), n(0.35ne); bc=1, value=0.0),
        _straight((0.65, 1.0), (0.35, 1.0), n(0.3ne); bc=0, value=Tl),
        _straight((0.35, 1.0), (0.0, 1.0), n(0.35ne); bc=1, value=0.0),
        _straight((0.0, 1.0), (0.0, 0.0), n(ne);     bc=1, value=0.0),
    ]
    d = _outer(segs; degree=degree, k=k, nint=nint, name="pacheco_inverted_v")
    freeze_dirichlet!(d)
    return d
end

function pacheco_asymmetric(; ne=20, nint=16, degree=2, k=1.0)
    n = x -> max(Int(round(x)), 2)
    segs = [
        _straight((0.0, 0.0), (0.8, 0.0), n(0.8ne); bc=1, value=0.0),
        _straight((0.8, 0.0), (1.0, 0.0), n(0.2ne); bc=0, value=1.0),
        _straight((1.0, 0.0), (1.0, 0.8), n(0.8ne); bc=1, value=0.0),
        _straight((1.0, 0.8), (1.0, 1.0), n(0.2ne); bc=0, value=0.0),
        _straight((1.0, 1.0), (0.2, 1.0), n(0.8ne); bc=1, value=0.0),
        _straight((0.2, 1.0), (0.0, 1.0), n(0.2ne); bc=0, value=1.0),
        _straight((0.0, 1.0), (0.0, 0.2), n(0.8ne); bc=1, value=0.0),
        _straight((0.0, 0.2), (0.0, 0.0), n(0.2ne); bc=0, value=0.0),
    ]
    d = _outer(segs; degree=degree, k=k, nint=nint, name="pacheco_asymmetric")
    freeze_dirichlet!(d)
    return d
end

function pacheco_bridge(; ne=20, nint=16, degree=2, k=1.0)
    n = x -> max(Int(round(x)), 2)
    p0 = Point2D(0.3125, 0.0)
    p1 = Point2D(0.6875, 0.0)
    arc = _arc_verts(p0, p1, -0.15, n(0.6ne) + 1)
    segs = [
        _straight((0.0, 0.0), (0.3125, 0.0), n(0.35ne); bc=1, value=0.0),
        BoundarySegment(arc; bc=1, value=0.0, n_el=n(0.6ne)),
        _straight((0.6875, 0.0), (1.0, 0.0), n(0.35ne); bc=1, value=0.0),
        _straight((1.0, 0.0), (1.0, 0.1875), n(0.15ne); bc=0, value=0.0),
        _straight((1.0, 0.1875), (1.0, 0.5), n(0.35ne); bc=1, value=0.0),
        _straight((1.0, 0.5), (0.0, 0.5), n(ne);        bc=1, value=0.0),
        _straight((0.0, 0.5), (0.0, 0.1875), n(0.35ne); bc=1, value=0.0),
        _straight((0.0, 0.1875), (0.0, 0.0), n(0.15ne); bc=0, value=1.0),
    ]
    d = _outer(segs; degree=degree, k=k, nint=nint, name="pacheco_bridge")
    freeze_dirichlet!(d)
    return d
end

function pacheco_cross(; ne=20, nint=16, degree=2, k=1.0)
    n = x -> max(Int(round(x)), 2)
    segs = [
        _straight((0.6, 0.0), (1.0, 0.0), n(0.4ne); bc=1, value=0.0),
        _straight((1.0, 0.0), (1.0, 0.4), n(0.4ne); bc=1, value=0.0),
        _straight((1.0, 0.4), (1.0, 0.6), n(0.2ne); bc=0, value=0.0),
        _straight((1.0, 0.6), (1.0, 1.0), n(0.4ne); bc=1, value=0.0),
        _straight((1.0, 1.0), (0.6, 1.0), n(0.4ne); bc=1, value=0.0),
        _straight((0.6, 1.0), (0.4, 1.0), n(0.2ne); bc=0, value=1.0),
        _straight((0.4, 1.0), (0.0, 1.0), n(0.4ne); bc=1, value=0.0),
        _straight((0.0, 1.0), (0.0, 0.6), n(0.4ne); bc=1, value=0.0),
        _straight((0.0, 0.6), (0.0, 0.4), n(0.2ne); bc=0, value=0.0),
        _straight((0.0, 0.4), (0.0, 0.0), n(0.4ne); bc=1, value=0.0),
        _straight((0.0, 0.0), (0.4, 0.0), n(0.4ne); bc=1, value=0.0),
        _straight((0.4, 0.0), (0.6, 0.0), n(0.2ne); bc=0, value=1.0),
    ]
    d = _outer(segs; degree=degree, k=k, nint=nint, name="pacheco_cross")
    freeze_dirichlet!(d)
    return d
end

function pacheco_options(id::Integer)
    id == 1 && return PachecoOptions(ΔA=0.50, vmax=0.10, pct=0.90, nv=0.04)
    id == 2 && return PachecoOptions(ΔA=0.20, vmax=0.10, pct=0.50, nv=0.04)
    id == 3 && return PachecoOptions(ΔA=0.35, vmax=0.05, pct=0.95, nv=0.04)
    id == 4 && return PachecoOptions(ΔA=0.40, vmax=0.05, pct=0.90, nv=0.04)
    throw(ArgumentError("pacheco_options id must be 1:4"))
end

function pacheco_problem(id::Integer; ne=20, nint=16, degree=2, via_gmsh::Bool=false, ndiv=nothing)
    ne_m = ndiv === nothing ? ne : ndiv
    if via_gmsh
        msh = if id == 1
            mesh_pacheco_inverted_v(; ndiv=ne_m)
        elseif id == 2
            mesh_pacheco_asymmetric(; ndiv=ne_m)
        elseif id == 3
            mesh_pacheco_bridge(; ndiv=ne_m)
        elseif id == 4
            mesh_pacheco_cross(; ndiv=ne_m)
        else
            throw(ArgumentError("pacheco_problem id must be 1:4"))
        end
        dad = format2d(msh, Laplace(1.0); tipo=degree, pontointerno=false, finalize=true)
        d = extract_loops(dad; nx_int=nint, ny_int=nint)
        d.degree = degree
        return d, pacheco_options(id)
    end
    d = if id == 1
        pacheco_inverted_v(; ne=ne_m, nint=nint, degree=degree)
    elseif id == 2
        pacheco_asymmetric(; ne=ne_m, nint=nint, degree=degree)
    elseif id == 3
        pacheco_bridge(; ne=ne_m, nint=nint, degree=degree)
    elseif id == 4
        pacheco_cross(; ne=ne_m, nint=nint, degree=degree)
    else
        throw(ArgumentError("pacheco_problem id must be 1:4"))
    end
    return d, pacheco_options(id)
end

# ----- Gmsh initial meshes ---------------------------------------------------

function _pacheco_gmsh_poly(name, pts, segs_idx, bc, n_el; ordem=2)
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(name)
    lc = 0.1
    tags = Int[]
    for (i, p) in enumerate(pts)
        push!(tags, gmsh.model.geo.addPoint(p[1], p[2], 0.0, lc, i))
    end
    n = length(pts)
    lines = Int[]
    for i in 1:n
        a = tags[i]
        b = tags[i == n ? 1 : i + 1]
        push!(lines, gmsh.model.geo.addLine(a, b, i))
    end
    cl = gmsh.model.geo.addCurveLoop(lines, 1)
    s1 = gmsh.model.geo.addPlaneSurface([cl], 1)
    gmsh.model.geo.synchronize()
    for (ℓ, ne) in zip(lines, n_el)
        gmsh.model.mesh.setTransfiniteCurve(ℓ, max(ne + 1, 2))
    end
    # physical groups by BC string
    groups = Dict{String,Vector{Int}}()
    for (ℓ, (t, v)) in zip(lines, bc)
        key = string(t, ";", v)
        push!(get!(groups, key, Int[]), ℓ)
    end
    for (key, ls) in groups
        gmsh.model.addPhysicalGroup(1, ls, -1, key)
    end
    gmsh.model.addPhysicalGroup(2, [s1], -1, "Domain")
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(ordem)
    out = datadir("Laplace", name * ".msh")
    mkpath(dirname(out))
    gmsh.write(out)
    gmsh.finalize()
    return out
end

function mesh_pacheco_inverted_v(; ndiv=20, ordem=2, nome="pacheco_inverted_v")
    n = x -> max(Int(round(x)), 2)
    pts = [(0.0, 0.0), (0.2, 0.0), (0.8, 0.0), (1.0, 0.0),
           (1.0, 1.0), (0.65, 1.0), (0.35, 1.0), (0.0, 1.0)]
    bc = [(0, 1.0), (1, 0.0), (0, 1.0), (1, 0.0), (1, 0.0), (0, 0.0), (1, 0.0), (1, 0.0)]
    nel = [n(0.2ndiv), n(0.6ndiv), n(0.2ndiv), n(ndiv), n(0.35ndiv), n(0.3ndiv), n(0.35ndiv), n(ndiv)]
    return _pacheco_gmsh_poly(nome, pts, nothing, bc, nel; ordem=ordem)
end

function mesh_pacheco_asymmetric(; ndiv=20, ordem=2, nome="pacheco_asymmetric")
    n = x -> max(Int(round(x)), 2)
    pts = [(0.0, 0.0), (0.8, 0.0), (1.0, 0.0), (1.0, 0.8),
           (1.0, 1.0), (0.2, 1.0), (0.0, 1.0), (0.0, 0.2)]
    bc = [(1, 0.0), (0, 1.0), (1, 0.0), (0, 0.0), (1, 0.0), (0, 1.0), (1, 0.0), (0, 0.0)]
    nel = [n(0.8ndiv), n(0.2ndiv), n(0.8ndiv), n(0.2ndiv), n(0.8ndiv), n(0.2ndiv), n(0.8ndiv), n(0.2ndiv)]
    return _pacheco_gmsh_poly(nome, pts, nothing, bc, nel; ordem=ordem)
end

function mesh_pacheco_cross(; ndiv=20, ordem=2, nome="pacheco_cross")
    n = x -> max(Int(round(x)), 2)
    pts = [(0.6, 0.0), (1.0, 0.0), (1.0, 0.4), (1.0, 0.6),
           (1.0, 1.0), (0.6, 1.0), (0.4, 1.0), (0.0, 1.0),
           (0.0, 0.6), (0.0, 0.4), (0.0, 0.0), (0.4, 0.0)]
    bc = [(1, 0.0), (1, 0.0), (0, 0.0), (1, 0.0), (1, 0.0), (0, 1.0),
          (1, 0.0), (1, 0.0), (0, 0.0), (1, 0.0), (1, 0.0), (0, 1.0)]
    nel = [n(0.4ndiv), n(0.4ndiv), n(0.2ndiv), n(0.4ndiv), n(0.4ndiv), n(0.2ndiv),
           n(0.4ndiv), n(0.4ndiv), n(0.2ndiv), n(0.4ndiv), n(0.4ndiv), n(0.2ndiv)]
    return _pacheco_gmsh_poly(nome, pts, nothing, bc, nel; ordem=ordem)
end

function mesh_pacheco_bridge(; ndiv=20, ordem=2, nome="pacheco_bridge")
    n = x -> max(Int(round(x)), 2)
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(nome)
    lc = 0.1
    p = Dict{Int,Int}()
    coords = (
        (1, 0.0, 0.0), (2, 0.3125, 0.0), (3, 0.6875, 0.0), (4, 1.0, 0.0),
        (5, 1.0, 0.1875), (6, 1.0, 0.5), (7, 0.0, 0.5), (8, 0.0, 0.1875),
    )
    for (i, x, y) in coords
        p[i] = gmsh.model.geo.addPoint(x, y, 0.0, lc, i)
    end
    # arc centre (PG2 R = -0.15, p2→p3 along bottom)
    # sample to get centre
    a0 = Point2D(0.3125, 0.0)
    a1 = Point2D(0.6875, 0.0)
    av = _arc_verts(a0, a1, -0.15, 5)
    mid = av[3]
    pc = gmsh.model.geo.addPoint(mid[1], mid[2], 0.0, lc)
    l1 = gmsh.model.geo.addLine(p[1], p[2])
    l2 = gmsh.model.geo.addCircleArc(p[2], pc, p[3])
    l3 = gmsh.model.geo.addLine(p[3], p[4])
    l4 = gmsh.model.geo.addLine(p[4], p[5])
    l5 = gmsh.model.geo.addLine(p[5], p[6])
    l6 = gmsh.model.geo.addLine(p[6], p[7])
    l7 = gmsh.model.geo.addLine(p[7], p[8])
    l8 = gmsh.model.geo.addLine(p[8], p[1])
    lines = [l1, l2, l3, l4, l5, l6, l7, l8]
    cl = gmsh.model.geo.addCurveLoop(lines)
    s1 = gmsh.model.geo.addPlaneSurface([cl])
    gmsh.model.geo.synchronize()
    nel = [n(0.35ndiv), n(0.6ndiv), n(0.35ndiv), n(0.15ndiv),
           n(0.35ndiv), n(ndiv), n(0.35ndiv), n(0.15ndiv)]
    for (ℓ, ne) in zip(lines, nel)
        gmsh.model.mesh.setTransfiniteCurve(ℓ, max(ne + 1, 2))
    end
    gmsh.model.addPhysicalGroup(1, [l1, l2, l3, l5, l6, l7], -1, "1;0")
    gmsh.model.addPhysicalGroup(1, [l4], -1, "0;0")
    gmsh.model.addPhysicalGroup(1, [l8], -1, "0;1")
    gmsh.model.addPhysicalGroup(2, [s1], -1, "Domain")
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(ordem)
    out = datadir("Laplace", nome * ".msh")
    mkpath(dirname(out))
    gmsh.write(out)
    gmsh.finalize()
    return out
end

# =============================================================================
# Coelho (2021) plane-stress elasticity — cases 3–5
# =============================================================================

_coelho_props(E, ν) = Elasticity(E, ν, 1.0; plane_stress=true)

function coelho_cantilever(; ne=15, nint=20, degree=2, E=100e9, ν=0.3, F=1000.0)
    n = x -> max(Int(round(x)), 2)
    h, b = 1.5, 1.0
    segs = [
        _straight((0.0, 0.0), (h, 0.0), n(ne); bc=[1, 1], value=[0.0, 0.0]),
        _straight((h, 0.0), (h, 0.45b), n(ne); bc=[1, 1], value=[0.0, 0.0]),
        _straight((h, 0.45b), (h, 0.55b), n(ne); bc=[0, 1], value=[0.0, -F]),
        _straight((h, 0.55b), (h, b), n(ne); bc=[1, 1], value=[0.0, 0.0]),
        _straight((h, b), (0.0, b), n(ne); bc=[1, 1], value=[0.0, 0.0]),
        _straight((0.0, b), (0.0, 0.9b), n(ne); bc=[0, 0], value=[0.0, 0.0]),
        _straight((0.0, 0.9b), (0.0, 0.1b), n(ne); bc=[1, 1], value=[0.0, 0.0]),
        _straight((0.0, 0.1b), (0.0, 0.0), n(ne); bc=[0, 0], value=[0.0, 0.0]),
    ]
    d = _outer(segs; degree=degree, nint=nint, name="coelho_cantilever",
        properties=_coelho_props(E, ν))
    freeze_dirichlet!(d)
    return d
end

function coelho_cc_beam(; ne=15, nint=20, degree=2, E=100e9, ν=0.3, F=1000.0)
    n = x -> max(Int(round(x)), 2)
    h, b = 2.0, 1.0
    segs = [
        _straight((0.0, 0.0), (0.1h, 0.0), n(ne); bc=[0, 0], value=[0.0, 0.0]),
        _straight((0.1h, 0.0), (0.45h, 0.0), n(ne); bc=[1, 1], value=[0.0, 0.0]),
        _straight((0.45h, 0.0), (0.55h, 0.0), n(ne); bc=[1, 1], value=[0.0, -F]),
        _straight((0.55h, 0.0), (0.9h, 0.0), n(ne); bc=[1, 1], value=[0.0, 0.0]),
        _straight((0.9h, 0.0), (h, 0.0), n(ne); bc=[0, 0], value=[0.0, 0.0]),
        _straight((h, 0.0), (h, b), n(ne); bc=[1, 1], value=[0.0, 0.0]),
        _straight((h, b), (0.55h, b), n(ne); bc=[1, 1], value=[0.0, 0.0]),
        _straight((0.55h, b), (0.45h, b), n(ne); bc=[1, 1], value=[0.0, -F]),
        _straight((0.45h, b), (0.0, b), n(ne); bc=[1, 1], value=[0.0, 0.0]),
        _straight((0.0, b), (0.0, 0.0), n(ne); bc=[1, 1], value=[0.0, 0.0]),
    ]
    d = _outer(segs; degree=degree, nint=nint, name="coelho_cc_beam",
        properties=_coelho_props(E, ν))
    freeze_dirichlet!(d)
    return d
end

"""Clamp + roller beam (Coelho caso 5). The source CCSeg clamps both bottom ends."""
function coelho_ss_beam(; ne=15, nint=20, degree=2, E=200e9, ν=0.3, F=nothing)
    n = x -> max(Int(round(x)), 2)
    h, b = 2.0, 1.0
    Fv = F === nothing ? 3 * 100 / (2 * b) : Float64(F)
    segs = [
        _straight((0.0, 0.0), (0.1h, 0.0), n(ne); bc=[0, 0], value=[0.0, 0.0]),
        _straight((0.1h, 0.0), (0.45h, 0.0), n(ne); bc=[1, 1], value=[0.0, 0.0]),
        _straight((0.45h, 0.0), (0.55h, 0.0), n(ne); bc=[1, 1], value=[0.0, -Fv]),
        _straight((0.55h, 0.0), (0.9h, 0.0), n(ne); bc=[1, 1], value=[0.0, 0.0]),
        _straight((0.9h, 0.0), (h, 0.0), n(ne); bc=[0, 0], value=[0.0, 0.0]),
        _straight((h, 0.0), (h, b), n(ne); bc=[1, 1], value=[0.0, 0.0]),
        _straight((h, b), (0.0, b), n(ne); bc=[1, 1], value=[0.0, 0.0]),
        _straight((0.0, b), (0.0, 0.0), n(ne); bc=[1, 1], value=[0.0, 0.0]),
    ]
    d = _outer(segs; degree=degree, nint=nint, name="coelho_ss_beam",
        properties=_coelho_props(E, ν))
    freeze_dirichlet!(d)
    return d
end

function coelho_options(id::Integer)
    id == 3 && return PachecoOptions(ΔA=0.65, vmax=0.03, pct=0.75, nv=0.10)
    id == 4 && return PachecoOptions(ΔA=0.60, vmax=0.03, pct=0.75, nv=0.10)
    id == 5 && return PachecoOptions(ΔA=0.65, vmax=0.03, pct=0.75, nv=0.10)
    throw(ArgumentError("coelho_options id must be 3, 4, or 5"))
end

function coelho_problem(id::Integer; ne=15, nint=20, degree=2, via_gmsh::Bool=false, ndiv=nothing)
    ne_m = ndiv === nothing ? ne : ndiv
    props = if id == 3 || id == 4
        _coelho_props(100e9, 0.3)
    elseif id == 5
        _coelho_props(200e9, 0.3)
    else
        throw(ArgumentError("coelho_problem id must be 3, 4, or 5"))
    end
    if via_gmsh
        msh = if id == 3
            mesh_coelho_cantilever(; ndiv=ne_m)
        elseif id == 4
            mesh_coelho_cc_beam(; ndiv=ne_m)
        else
            mesh_coelho_ss_beam(; ndiv=ne_m)
        end
        dad = format2d(msh, props; tipo=degree, pontointerno=false, finalize=true)
        d = extract_loops(dad; nx_int=nint, ny_int=nint)
        d.degree = degree
        return d, coelho_options(id)
    end
    d = if id == 3
        coelho_cantilever(; ne=ne_m, nint=nint, degree=degree)
    elseif id == 4
        coelho_cc_beam(; ne=ne_m, nint=nint, degree=degree)
    else
        coelho_ss_beam(; ne=ne_m, nint=nint, degree=degree)
    end
    return d, coelho_options(id)
end

function _poly_gmsh(name, pts, bc_keys, n_el; ordem=2, folder="elastico")
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(name)
    lc = 0.1
    tags = Int[]
    for (i, p) in enumerate(pts)
        push!(tags, gmsh.model.geo.addPoint(p[1], p[2], 0.0, lc, i))
    end
    n = length(pts)
    lines = Int[]
    for i in 1:n
        a = tags[i]
        b = tags[i == n ? 1 : i + 1]
        push!(lines, gmsh.model.geo.addLine(a, b, i))
    end
    cl = gmsh.model.geo.addCurveLoop(lines, 1)
    s1 = gmsh.model.geo.addPlaneSurface([cl], 1)
    gmsh.model.geo.synchronize()
    for (ℓ, ne) in zip(lines, n_el)
        gmsh.model.mesh.setTransfiniteCurve(ℓ, max(ne + 1, 2))
    end
    groups = Dict{String,Vector{Int}}()
    for (ℓ, key) in zip(lines, bc_keys)
        push!(get!(groups, key, Int[]), ℓ)
    end
    for (key, ls) in groups
        gmsh.model.addPhysicalGroup(1, ls, -1, key)
    end
    gmsh.model.addPhysicalGroup(2, [s1], -1, "Domain")
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(ordem)
    out = datadir(folder, name * ".msh")
    mkpath(dirname(out))
    gmsh.write(out)
    gmsh.finalize()
    return out
end

function mesh_coelho_cantilever(; ndiv=15, ordem=2, nome="coelho_cantilever", F=1000.0)
    n = x -> max(Int(round(x)), 2)
    h, b = 1.5, 1.0
    pts = [(0.0, 0.0), (h, 0.0), (h, 0.45b), (h, 0.55b),
           (h, b), (0.0, b), (0.0, 0.9b), (0.0, 0.1b)]
    bc = ["1;0;1;0", "1;0;1;0", "0;0;1;$(-F)", "1;0;1;0",
          "1;0;1;0", "0;0;0;0", "1;0;1;0", "0;0;0;0"]
    nel = fill(n(ndiv), 8)
    return _poly_gmsh(nome, pts, bc, nel; ordem=ordem)
end

function mesh_coelho_cc_beam(; ndiv=15, ordem=2, nome="coelho_cc_beam", F=1000.0)
    n = x -> max(Int(round(x)), 2)
    h, b = 2.0, 1.0
    pts = [(0.0, 0.0), (0.1h, 0.0), (0.45h, 0.0), (0.55h, 0.0), (0.9h, 0.0),
           (h, 0.0), (h, b), (0.55h, b), (0.45h, b), (0.0, b)]
    bc = ["0;0;0;0", "1;0;1;0", "1;0;1;$(-F)", "1;0;1;0", "0;0;0;0",
          "1;0;1;0", "1;0;1;0", "1;0;1;$(-F)", "1;0;1;0", "1;0;1;0"]
    nel = fill(n(ndiv), 10)
    return _poly_gmsh(nome, pts, bc, nel; ordem=ordem)
end

function mesh_coelho_ss_beam(; ndiv=15, ordem=2, nome="coelho_ss_beam", F=nothing)
    n = x -> max(Int(round(x)), 2)
    h, b = 2.0, 1.0
    Fv = F === nothing ? 3 * 100 / (2 * b) : Float64(F)
    pts = [(0.0, 0.0), (0.1h, 0.0), (0.45h, 0.0), (0.55h, 0.0),
           (0.9h, 0.0), (h, 0.0), (h, b), (0.0, b)]
    bc = ["0;0;0;0", "1;0;1;0", "1;0;1;$(-Fv)", "1;0;1;0",
          "0;0;0;0", "1;0;1;0", "1;0;1;0", "1;0;1;0"]
    nel = fill(n(ndiv), 8)
    return _poly_gmsh(nome, pts, bc, nel; ordem=ordem)
end

# =============================================================================
# Portela (2012) plate-with-hole (elasticity) and insulated-hole heat analogue
# =============================================================================

"""
    portela_plate_hole(; ratio=1.0, a_over_h=0.125, h=1.0, ne_outer=6, ne_hole=4,
                       nint=8, degree=2, E=1.0, ν=0.3, t2=1.0)

Quarter of a square plate of side `h` with a centred square hole of side
`a = a_over_h * h` (Portela 2012). Symmetry on `x=0` and `y=0`; tension
`ℓ1 = ratio * t2` on `x = h/2`, `ℓ2 = t2` on `y = h/2`; hole is traction-free
design. Plane stress. Radial origin is `(0,0)`.
"""
function portela_plate_hole(; ratio=1.0, a_over_h=0.125, h=1.0, ne_outer=6, ne_hole=4,
        nint=8, degree=2, E=1.0, ν=0.3, t2=1.0)
    a = a_over_h * h
    ℓ1 = float(ratio) * float(t2)
    ℓ2 = float(t2)
    n = x -> max(Int(round(x)), 2)
    L = h / 2
    r = a / 2
    segs = [
        _straight((r, 0.0), (L, 0.0), n(ne_outer); bc=[1, 0], value=[0.0, 0.0]),
        _straight((L, 0.0), (L, L), n(ne_outer); bc=[1, 1], value=[ℓ1, 0.0]),
        _straight((L, L), (0.0, L), n(ne_outer); bc=[1, 1], value=[0.0, ℓ2]),
        _straight((0.0, L), (0.0, r), n(ne_outer); bc=[0, 1], value=[0.0, 0.0]),
        _straight((0.0, r), (r, r), n(ne_hole); bc=[1, 1], value=[0.0, 0.0]),
        _straight((r, r), (r, 0.0), n(ne_hole); bc=[1, 1], value=[0.0, 0.0]),
    ]
    d = TopologyDesign([segs]; degree=degree, nx_int=nint, ny_int=nint,
        name="portela_plate_hole", properties=Elasticity(E, ν, 1.0; plane_stress=true))
    freeze_dirichlet!(d)
    return d
end

"""
    portela_heat_hole(; a=0.25, ne=8, nint=8, degree=2, k=1.0)

Unit square, hot bottom (`T=1`), cold top (`T=0`), insulated sides and a
centred insulated square hole of side `a`. Laplace analogue of Portela's
hole-shape problem. Radial origin is `(0.5, 0.5)`.
"""
function portela_heat_hole(; a=0.25, ne=8, nint=8, degree=2, k=1.0)
    lo = 0.5 - a / 2
    hi = 0.5 + a / 2
    n = x -> max(Int(round(x)), 2)
    outer = [
        _straight((0.0, 0.0), (1.0, 0.0), n(ne); bc=0, value=1.0),
        _straight((1.0, 0.0), (1.0, 1.0), n(ne); bc=1, value=0.0),
        _straight((1.0, 1.0), (0.0, 1.0), n(ne); bc=0, value=0.0),
        _straight((0.0, 1.0), (0.0, 0.0), n(ne); bc=1, value=0.0),
    ]
    hole = [
        _straight((lo, lo), (hi, lo), n(ne); bc=1, value=0.0),
        _straight((hi, lo), (hi, hi), n(ne); bc=1, value=0.0),
        _straight((hi, hi), (lo, hi), n(ne); bc=1, value=0.0),
        _straight((lo, hi), (lo, lo), n(ne); bc=1, value=0.0),
    ]
    d = TopologyDesign([outer, hole]; degree=degree, k=k, nx_int=nint, ny_int=nint,
        name="portela_heat_hole")
    freeze_dirichlet!(d)
    # Paper: only the hole is Γ_d. Insulated outer sides stay put.
    for s in d.loops[1]
        any(==(0), s.bc) && continue
        fill!(s.frozen, true)
    end
    return d
end
