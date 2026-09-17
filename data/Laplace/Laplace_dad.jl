# Geometry generators for Laplace benchmarks.
# `datadir` comes from DrWatson (no longer reexported by `using BEM`).
using DrWatson: datadir

"""
    placa_com_furo(; kwargs...)

Plate with a circular hole. Writes `datadir("Laplace", nome * ".msh")`.
"""
function placa_com_furo(;
    nome="placa_com_furo",
    ordem=2,
    Lx=1.0,
    Ly=0.4,
    cx=0.5,
    cy=0.2,
    r=0.08,
    lc_hole=0.02,
    lc_plate=0.05,
    show=true,
)
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(nome)

    plate = gmsh.model.occ.addRectangle(0.0, 0.0, 0.0, Lx, Ly)
    hole = gmsh.model.occ.addDisk(cx, cy, 0.0, r, r)
    gmsh.model.occ.cut([(2, plate)], [(2, hole)])
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.setRecombine(2, 1)

    gmsh.model.addPhysicalGroup(2, [1], 1, "Plate")
    leftCurves = Int[7]
    rightCurves = Int[8]
    bottomCurves = Int[6]
    topCurves = Int[9]
    holeCurves = Int[5]
    gmsh.model.addPhysicalGroup(1, leftCurves, -1, "0;0")       # Dirichlet
    gmsh.model.addPhysicalGroup(1, rightCurves, -1, "0;1")
    gmsh.model.addPhysicalGroup(1, bottomCurves, -1, "1;0")
    gmsh.model.addPhysicalGroup(1, topCurves, -1, "1;0")
    gmsh.model.addPhysicalGroup(1, holeCurves, -1, "1;0")

    f1 = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(f1, "CurvesList", holeCurves)
    gmsh.model.mesh.field.setNumber(f1, "Sampling", 100)
    f2 = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(f2, "InField", f1)
    gmsh.model.mesh.field.setNumber(f2, "SizeMin", lc_hole)
    gmsh.model.mesh.field.setNumber(f2, "SizeMax", lc_plate)
    gmsh.model.mesh.field.setNumber(f2, "DistMin", 0.01)
    gmsh.model.mesh.field.setNumber(f2, "DistMax", 0.08)
    gmsh.model.mesh.field.setAsBackgroundMesh(f2)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)

    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(ordem)

    out = datadir("Laplace", nome * ".msh")
    mkpath(dirname(out))
    gmsh.write(out)
    show && gmsh.fltk.run()
    gmsh.finalize()
    return out
end

"""Classify 1-D OCC curves of the unit plate-with-hole by bounding box."""
function _plate_hole_curve_groups(; L=1.0, tol=1e-6)
    groups = Dict(:left => Int[], :right => Int[], :bottom => Int[], :top => Int[], :hole => Int[])
    for (dim, tag) in gmsh.model.getEntities(1)
        xmin, ymin, _z0, xmax, ymax, _z1 = gmsh.model.getBoundingBox(dim, tag)
        if abs(xmax) < tol && abs(xmin) < tol
            push!(groups[:left], tag)
        elseif abs(xmin - L) < tol && abs(xmax - L) < tol
            push!(groups[:right], tag)
        elseif abs(ymax) < tol && abs(ymin) < tol
            push!(groups[:bottom], tag)
        elseif abs(ymin - L) < tol && abs(ymax - L) < tol
            push!(groups[:top], tag)
        else
            push!(groups[:hole], tag)
        end
    end
    return groups
end

"""
    placa_furo_orto(; L=1, r=0.25, lc=0.05, Tleft=0, qright=5) -> path

Unit square minus a disk, FEniCS orthotropic heat plate geometry.
Package flux ``q = -n·K∇u``: FEniCS ``K∇T·n = 5`` on the right is `qright=-5`.
Left Dirichlet `Tleft`, rest insulated.
"""
function placa_furo_orto(;
        nome="placa_furo_orto",
        L=1.0,
        r=0.25,
        cx=0.5,
        cy=0.5,
        lc=0.05,
        Tleft=0.0,
        qright=-5.0,
        ordem=1,
        show=false,
    )
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(nome)
    plate = gmsh.model.occ.addRectangle(0.0, 0.0, 0.0, L, L)
    hole = gmsh.model.occ.addDisk(cx, cy, 0.0, r, r)
    gmsh.model.occ.cut([(2, plate)], [(2, hole)])
    gmsh.model.occ.synchronize()
    g = _plate_hole_curve_groups(; L=L)
    gmsh.model.addPhysicalGroup(1, g[:left], -1, "0;$Tleft")
    gmsh.model.addPhysicalGroup(1, g[:right], -1, "1;$qright")
    gmsh.model.addPhysicalGroup(1, vcat(g[:bottom], g[:top], g[:hole]), -1, "1;0")
    surfs = [t for (_d, t) in gmsh.model.getEntities(2)]
    gmsh.model.addPhysicalGroup(2, surfs, -1, "plate")
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), lc)
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(ordem)
    # Hole curves from OCC cut are oriented inward to Ω; BEM needs n outward
    # (into the hole). Reverse after setOrder so quadratic mids keep the flag.
    try
        gmsh.model.mesh.reverse([(1, t) for t in g[:hole]])
    catch
    end
    out = datadir("Laplace", nome * ".msh")
    mkpath(dirname(out))
    gmsh.write(out)
    show && gmsh.fltk.run()
    gmsh.finalize()
    return out
end

"""
    placa_furo_orto_3d(; dz=0.2, lc=0.08, ...) -> path

Extrusion of [`placa_furo_orto`](@ref) (cylinder hole). `z`-faces insulated.
"""
function placa_furo_orto_3d(;
        nome="placa_furo_orto_3d",
        L=1.0,
        r=0.25,
        cx=0.5,
        cy=0.5,
        dz=0.2,
        lc=0.08,
        Tleft=0.0,
        qright=-5.0,
        recombine::Bool=true,
        show=false,
    )
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(nome)
    plate = gmsh.model.occ.addRectangle(0.0, 0.0, 0.0, L, L)
    hole = gmsh.model.occ.addDisk(cx, cy, 0.0, r, r)
    gmsh.model.occ.cut([(2, plate)], [(2, hole)])
    gmsh.model.occ.synchronize()
    surfs2 = gmsh.model.getEntities(2)
    gmsh.model.occ.extrude(surfs2, 0.0, 0.0, dz)
    gmsh.model.occ.synchronize()
    left = Int[]; right = Int[]; rest = Int[]
    tol = 1e-6
    for (dim, tag) in gmsh.model.getEntities(2)
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(dim, tag)
        if abs(xmax) < tol && abs(xmin) < tol
            push!(left, tag)
        elseif abs(xmin - L) < tol && abs(xmax - L) < tol
            push!(right, tag)
        else
            push!(rest, tag)
        end
    end
    gmsh.model.addPhysicalGroup(2, left, -1, "0;$Tleft")
    gmsh.model.addPhysicalGroup(2, right, -1, "1;$qright")
    gmsh.model.addPhysicalGroup(2, rest, -1, "1;0")
    vols = [t for (_d, t) in gmsh.model.getEntities(3)]
    isempty(vols) || gmsh.model.addPhysicalGroup(3, vols, -1, "solid")
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), lc)
    recombine && gmsh.option.setNumber("Mesh.RecombineAll", 1)
    # Volume mesh while OCC is live — a surface-only .msh will not
    # tetrahedralize after reload (generate(3) then yields 0 elements).
    gmsh.model.mesh.generate(3)
    # reverse cylinder walls (not outer box faces)
    for (dim, tag) in gmsh.model.getEntities(2)
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(dim, tag)
        zface = abs(zmax - zmin) < tol
        outer = (abs(xmax) < tol && abs(xmin) < tol) ||
                (abs(xmin - L) < tol && abs(xmax - L) < tol) ||
                (abs(ymax) < tol && abs(ymin) < tol) ||
                (abs(ymin - L) < tol && abs(ymax - L) < tol) || zface
        if !outer
            try
                gmsh.model.mesh.reverse([(2, tag)])
            catch
            end
        end
    end
    out = datadir("Laplace", nome * ".msh")
    mkpath(dirname(out))
    gmsh.write(out)
    show && gmsh.fltk.run()
    gmsh.finalize()
    return out
end

"""
    cubo_furo_orto(; L=1, r=0.25, lc=0.12, ...) -> path

Unit cube ``[0,L]³`` minus a cylinder of radius `r` along ``z``
(through-hole). Same in-plane BCs as [`placa_furo_orto`](@ref):
left Dirichlet, right Neumann, other faces (including ``z=0,L`` and the
bore) insulated.

This is the 3D companion of the FEniCS plate: ``T(x,y,z)=T_{2D}(x,y)``.
Unlike [`placa_furo_orto_3d`](@ref) with default `dz=0.2` (a thin slab,
one element through the thickness), the cube has aspect ratio 1 so
``z``-faces are a distance `L` apart.
"""
cubo_furo_orto(; L=1.0, nome="cubo_furo_orto", kwargs...) =
    placa_furo_orto_3d(; L=L, dz=L, nome=nome, kwargs...)

"""
    quadrado(; nome="quadrado", Lx=1, Ly=1, ordem=1, ndiv=10, show=true))

Unit square with default BCs matching ``T = x`` under the package flux
convention ``q = -k ∂T/∂n`` (k=1):
- left (`x=0`): Dirichlet ``T=0``
- right (`x=L`): Neumann ``q = -1``  (since ∂T/∂n=+1 ⇒ q=-1)
- top/bottom: Neumann ``q = 0``

Writes `datadir("Laplace", nome * ".msh")` and returns that path.
"""
function quadrado(; nome="quadrado", Lx=1.0, Ly=1.0, ordem=1, ndiv=10, show=true)
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(nome)
    lc = 0.1

    p1 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, lc, 1)
    p2 = gmsh.model.geo.addPoint(Lx, 0.0, 0.0, lc, 2)
    p3 = gmsh.model.geo.addPoint(Lx, Ly, 0.0, lc, 3)
    p4 = gmsh.model.geo.addPoint(0.0, Ly, 0.0, lc, 4)

    l1 = gmsh.model.geo.addLine(p1, p2, 1)  # bottom
    l2 = gmsh.model.geo.addLine(p2, p3, 2)  # right
    l3 = gmsh.model.geo.addLine(p3, p4, 3)  # top
    l4 = gmsh.model.geo.addLine(p4, p1, 4)  # left

    cl = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4], 1)
    s1 = gmsh.model.geo.addPlaneSurface([cl], 1)
    gmsh.model.geo.synchronize()

    gmsh.model.mesh.setTransfiniteCurve(l1, ndiv)
    gmsh.model.mesh.setTransfiniteCurve(l2, ndiv)
    gmsh.model.mesh.setTransfiniteCurve(l3, ndiv)
    gmsh.model.mesh.setTransfiniteCurve(l4, ndiv)
    gmsh.model.mesh.setTransfiniteSurface(s1)
    gmsh.model.mesh.setRecombine(2, s1)

    # BCs → analytical T = x  (k=1, q = -k ∂T/∂n)
    gmsh.model.addPhysicalGroup(1, [l1, l3], -1, "1;0")   # insulated
    gmsh.model.addPhysicalGroup(1, [l2], -1, "1;-1")      # q = -1
    gmsh.model.addPhysicalGroup(1, [l4], -1, "0;0")       # T = 0
    gmsh.model.addPhysicalGroup(2, [s1], -1, "Domain")

    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(ordem)

    out = datadir("Laplace", nome * ".msh")
    mkpath(dirname(out))
    gmsh.write(out)
    show && gmsh.fltk.run()
    gmsh.finalize()
    return out
end

"""
    quadrado_elasticity(; nome="quadrado_elast", ndiv=10, show=false)

Square mesh for 2D elasticity patch tests.
Physical names use 4-token BC strings `tx;vx;ty;vy`.
Default: left edge clamped (`0;0;0;0`), other edges traction-free (`1;0;1;0`).
"""
function quadrado_elasticity(; nome="quadrado_elast", Lx=1.0, Ly=1.0, ordem=1, ndiv=10, show=false)
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(nome)
    lc = 0.1
    p1 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, lc, 1)
    p2 = gmsh.model.geo.addPoint(Lx, 0.0, 0.0, lc, 2)
    p3 = gmsh.model.geo.addPoint(Lx, Ly, 0.0, lc, 3)
    p4 = gmsh.model.geo.addPoint(0.0, Ly, 0.0, lc, 4)
    l1 = gmsh.model.geo.addLine(p1, p2, 1)
    l2 = gmsh.model.geo.addLine(p2, p3, 2)
    l3 = gmsh.model.geo.addLine(p3, p4, 3)
    l4 = gmsh.model.geo.addLine(p4, p1, 4)
    cl = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4], 1)
    s1 = gmsh.model.geo.addPlaneSurface([cl], 1)
    gmsh.model.geo.synchronize()
    for l in (l1, l2, l3, l4)
        gmsh.model.mesh.setTransfiniteCurve(l, ndiv)
    end
    gmsh.model.mesh.setTransfiniteSurface(s1)
    gmsh.model.mesh.setRecombine(2, s1)

    gmsh.model.addPhysicalGroup(1, [l4], -1, "0;0;0;0")       # left: fixed
    gmsh.model.addPhysicalGroup(1, [l1, l2, l3], -1, "1;0;1;0") # free
    gmsh.model.addPhysicalGroup(2, [s1], -1, "Domain")
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(ordem)

    out = datadir("elastico", nome * ".msh")
    mkpath(dirname(out))
    gmsh.write(out)
    show && gmsh.fltk.run()
    gmsh.finalize()
    return out
end

"""Type-5 Dual-BEM physical name, `ndof` pairs (`"5;2"` CBIE / `"5;3"` HBIE)."""
function _crack_bc_name(eq::Integer, ndof::Integer)
    pair = eq == 2 ? "5;2" : "5;3"
    return join(fill(pair, Int(ndof)), ";")
end

"""Two coincident Dual-BEM faces (Portela). Do **not** `removeAllDuplicates`."""
function _gmsh_embed_center_crack!(s, cx, cy, a, α, ndiv_crack, lc, nameA, nameB)
    a > 0 || return nothing
    cα, sα = cos(α), sin(α)
    ptL = gmsh.model.geo.addPoint(cx - a * cα, cy - a * sα, 0.0, lc / 2)
    ptR = gmsh.model.geo.addPoint(cx + a * cα, cy + a * sα, 0.0, lc / 2)
    cA = gmsh.model.geo.addLine(ptL, ptR)
    cB = gmsh.model.geo.addLine(ptR, ptL)
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.embed(1, [cA, cB], 2, s)
    ncr = max(Int(ndiv_crack), 3)
    gmsh.model.mesh.setTransfiniteCurve(cA, ncr)
    gmsh.model.mesh.setTransfiniteCurve(cB, ncr)
    gmsh.model.addPhysicalGroup(1, [cA], -1, nameA)
    gmsh.model.addPhysicalGroup(1, [cB], -1, nameB)
    return nothing
end

"""
    quadrado_plate(; a=1, ndiv=9, ordem=2, bc="SSSS", nome="quadrado_plate")

Rectangle Kirchhoff mesh (`[x0,x0+Lx]×[y0,y0+Ly]`, default unit square).
Physical names are 4-token plate BCs `type_w;val_w;type_Mn;val_Mn` per
edge (bottom, right, top, left):

- `C` clamped → `0;0;0;0`
- `S` simply supported → `0;0;1;0`
- `F` free → `1;0;1;0`

`ndiv` is Gmsh transfinite points per edge (`n_el = ndiv-1` elements), or a
4-tuple. A centre crack (`crack>0`) is two coincident embedded curves
tagged `"5;2;5;2"` (CBIE) / `"5;3;5;3"` (HBIE) — Dual BEM after
[`formatdata`](@ref) + `prepare_crack!`.
"""
function quadrado_plate(; nome="quadrado_plate", a=1.0, Lx=a, Ly=a, x0=0.0, y0=0.0,
        ndiv=9, ordem=2, bc="SSSS", crack=0.0, crack_α=0.0, ndiv_crack=16,
        show=false)
    length(bc) == 4 || error("quadrado_plate: bc must have 4 characters (C/S/F)")
    function tok(c)
        c = uppercase(c)
        c == 'C' && return "0;0;0;0"
        c == 'S' && return "0;0;1;0"
        c == 'F' && return "1;0;1;0"
        error("unknown plate BC '$c' (use C/S/F)")
    end
    ndivs = ndiv isa Integer ? (ndiv, ndiv, ndiv, ndiv) : Tuple(ndiv)
    length(ndivs) == 4 || error("ndiv must be an integer or 4-tuple")
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(nome)
    lc = min(Lx, Ly) / max(maximum(ndivs), 4)
    p1 = gmsh.model.geo.addPoint(x0, y0, 0.0, lc)
    p2 = gmsh.model.geo.addPoint(x0 + Lx, y0, 0.0, lc)
    p3 = gmsh.model.geo.addPoint(x0 + Lx, y0 + Ly, 0.0, lc)
    p4 = gmsh.model.geo.addPoint(x0, y0 + Ly, 0.0, lc)
    l1 = gmsh.model.geo.addLine(p1, p2)  # bottom
    l2 = gmsh.model.geo.addLine(p2, p3)  # right
    l3 = gmsh.model.geo.addLine(p3, p4)  # top
    l4 = gmsh.model.geo.addLine(p4, p1)  # left
    cl = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    s1 = gmsh.model.geo.addPlaneSurface([cl])
    gmsh.model.geo.synchronize()
    for (l, nd) in zip((l1, l2, l3, l4), ndivs)
        gmsh.model.mesh.setTransfiniteCurve(l, Int(nd))
    end
    has_crack = float(crack) > 0
    if !has_crack
        gmsh.model.mesh.setTransfiniteSurface(s1)
        gmsh.model.mesh.setRecombine(2, s1)
    end
    # One physical group per distinct BC string (Gmsh names must be unique).
    bytok = Dict{String,Vector{Int}}()
    for (l, ch) in zip((l1, l2, l3, l4), bc)
        push!(get!(bytok, tok(ch), Int[]), l)
    end
    for (name, ls) in bytok
        gmsh.model.addPhysicalGroup(1, ls, -1, name)
    end
    if has_crack
        _gmsh_embed_center_crack!(s1, x0 + Lx / 2, y0 + Ly / 2, float(crack),
            float(crack_α), ndiv_crack, lc, "5;2;5;2", "5;3;5;3")
    end
    gmsh.model.addPhysicalGroup(2, [s1], -1, "Domain")
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(ordem)
    out = datadir("Laplace", nome * ".msh")
    mkpath(dirname(out))
    gmsh.write(out)
    show && gmsh.fltk.run()
    gmsh.finalize()
    return out
end

"""
    quadrado_fsdt(; a=1, Lx=a, Ly=a, ndiv=9, ordem=2, bc="SSSS",
                    vn=(0,0,0,0), nome="quadrado_fsdt")

Rectangle FSDT mesh (`[x0,x0+Lx]×[y0,y0+Ly]`). Physical names are 6-token
plate BCs `type_ψx;val;type_ψy;val;type_w;val` per edge (bottom, right,
top, left):

- `C` clamped → `0;0;0;0;0;0`
- `S` simply supported (soft) → `1;0;1;0;0;0` (moments free, ``w=0``)
- `F` free → `1;0;1;0;1;vn`

`ndiv` is Gmsh transfinite points per edge (`n_el = ndiv-1` elements), or a
4-tuple of point counts. `vn[e]` is the known ``Q_n`` on a free edge.

A centre crack (`crack>0`) is two coincident embedded curves tagged
`"5;2"` / `"5;3"` repeated `ndof` times (3-DOF Reissner or 5-DOF Hsu–Hwu).
`ndof=5` uses SS-1: ``u=0`` on y=const, ``v=0`` on x=const, ``w=0``.
"""
function quadrado_fsdt(; nome="quadrado_fsdt", a=1.0, Lx=a, Ly=a, x0=0.0, y0=0.0,
        ndiv=9, ordem=2, bc="SSSS", vn::NTuple{4,Float64}=(0.0, 0.0, 0.0, 0.0),
        crack=0.0, crack_α=0.0, ndiv_crack=16, ndof::Integer=3, show=false)
    length(bc) == 4 || error("quadrado_fsdt: bc must have 4 characters (C/S/F)")
    ndof in (3, 5) || error("quadrado_fsdt: ndof must be 3 or 5, got $ndof")
    function tok(c, v, edge)
        c = uppercase(c)
        if ndof == 5
            c == 'C' && return "0;0;0;0;0;0;0;0;0;0"
            if c == 'S'
                edge == 1 || edge == 3 ?
                    (return "0;0;1;0;1;0;1;0;0;0") :  # u, w
                    (return "1;0;0;0;1;0;1;0;0;0")    # v, w
            end
            c == 'F' && return "1;0;1;0;1;0;1;0;1;$v"
            error("unknown FSDT BC '$c' (use C/S/F)")
        end
        c == 'C' && return "0;0;0;0;0;0"
        c == 'S' && return "1;0;1;0;0;0"
        c == 'F' && return "1;0;1;0;1;$v"
        error("unknown FSDT BC '$c' (use C/S/F)")
    end
    ndivs = ndiv isa Integer ? (ndiv, ndiv, ndiv, ndiv) : Tuple(ndiv)
    length(ndivs) == 4 || error("ndiv must be an integer or 4-tuple")
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(nome)
    lc = min(Lx, Ly) / max(maximum(ndivs), 4)
    p1 = gmsh.model.geo.addPoint(x0, y0, 0.0, lc)
    p2 = gmsh.model.geo.addPoint(x0 + Lx, y0, 0.0, lc)
    p3 = gmsh.model.geo.addPoint(x0 + Lx, y0 + Ly, 0.0, lc)
    p4 = gmsh.model.geo.addPoint(x0, y0 + Ly, 0.0, lc)
    l1 = gmsh.model.geo.addLine(p1, p2)  # bottom
    l2 = gmsh.model.geo.addLine(p2, p3)  # right
    l3 = gmsh.model.geo.addLine(p3, p4)  # top
    l4 = gmsh.model.geo.addLine(p4, p1)  # left
    cl = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    s1 = gmsh.model.geo.addPlaneSurface([cl])
    gmsh.model.geo.synchronize()
    for (l, nd) in zip((l1, l2, l3, l4), ndivs)
        gmsh.model.mesh.setTransfiniteCurve(l, Int(nd))
    end
    has_crack = float(crack) > 0
    if !has_crack
        gmsh.model.mesh.setTransfiniteSurface(s1)
        gmsh.model.mesh.setRecombine(2, s1)
    end
    bytok = Dict{String,Vector{Int}}()
    for (e, (l, ch, v)) in enumerate(zip((l1, l2, l3, l4), bc, vn))
        push!(get!(bytok, tok(ch, v, e), Int[]), l)
    end
    for (name, ls) in bytok
        gmsh.model.addPhysicalGroup(1, ls, -1, name)
    end
    if has_crack
        _gmsh_embed_center_crack!(s1, x0 + Lx / 2, y0 + Ly / 2, float(crack),
            float(crack_α), ndiv_crack, lc, _crack_bc_name(2, ndof),
            _crack_bc_name(3, ndof))
    end
    gmsh.model.addPhysicalGroup(2, [s1], -1, "Domain")
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(ordem)
    out = datadir("Laplace", nome * ".msh")
    mkpath(dirname(out))
    gmsh.write(out)
    show && gmsh.fltk.run()
    gmsh.finalize()
    return out
end
