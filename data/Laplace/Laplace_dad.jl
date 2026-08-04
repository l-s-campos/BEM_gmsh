# Geometry generators for Laplace benchmarks.
# Prefer calling these after `using BEM` / `@quickactivate :BEM` so `datadir` works.

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

"""
    quadrado(; nome="quadrado", Lx=1, Ly=1, ordem=1, ndiv=10, show=true)

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
