# Large plate with corner hole — remote tension in x (anisotropic materials)
# Geometry: hole r=1, plate width=height=10 (quarter). Materials Table 8.3.
# include(datadir("elastico", "aniso", "plate_hole_tension_x.jl"))

r_hole = 1.0
L = 10.0
σ∞ = 1.0
tension_dir = :x
nurbs_p = 4
n_sources_per_edge = 30

# Materials (Daniels; Ishai 2006)
boron_aluminum = (name="boron_aluminum", E1=235.0, E2=137.0, G12=47.0, ν12=0.30)
carbon_phenolic = (name="carbon_phenolic", E1=20.0, E2=19.0, G12=6.8, ν12=0.23)
sic_aluminum = (name="sic_aluminum", E1=204.0, E2=118.0, G12=41.0, ν12=0.27)
materials = (boron_aluminum, carbon_phenolic, sic_aluminum)

# BC quarter: bottom uy=0; right tx=σ∞; top free; left ux=0; hole free
function mesh_plate_hole_tension_x(; ndiv=20, nome="plate_hole_tension_x", show=false)
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(nome)
    lc = L / max(ndiv, 8)
    c = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, lc)
    ph1 = gmsh.model.geo.addPoint(r_hole, 0.0, 0.0, lc)
    ph2 = gmsh.model.geo.addPoint(0.0, r_hole, 0.0, lc)
    po1 = gmsh.model.geo.addPoint(L, 0.0, 0.0, lc)
    po2 = gmsh.model.geo.addPoint(L, L, 0.0, lc)
    po3 = gmsh.model.geo.addPoint(0.0, L, 0.0, lc)
    lb = gmsh.model.geo.addLine(ph1, po1)
    lr = gmsh.model.geo.addLine(po1, po2)
    lt = gmsh.model.geo.addLine(po2, po3)
    ll = gmsh.model.geo.addLine(po3, ph2)
    ah = gmsh.model.geo.addCircleArc(ph2, c, ph1)
    cl = gmsh.model.geo.addCurveLoop([lb, lr, lt, ll, ah])
    s = gmsh.model.geo.addPlaneSurface([cl])
    gmsh.model.geo.synchronize()
    for crv in (lb, lr, lt, ll, ah)
        gmsh.model.mesh.setTransfiniteCurve(crv, max(ndiv, 4))
    end
    gmsh.model.addPhysicalGroup(1, [lb], -1, "1;0;0;0")
    gmsh.model.addPhysicalGroup(1, [lr], -1, "1;$σ∞;1;0")
    gmsh.model.addPhysicalGroup(1, [lt], -1, "1;0;1;0")
    gmsh.model.addPhysicalGroup(1, [ll], -1, "0;0;1;0")
    gmsh.model.addPhysicalGroup(1, [ah], -1, "1;0;1;0")
    gmsh.model.addPhysicalGroup(2, [s], -1, "Domain")
    gmsh.model.mesh.generate(2)
    out = datadir("elastico", "aniso", nome * ".msh")
    mkpath(dirname(out))
    gmsh.write(out)
    show && gmsh.fltk.run()
    gmsh.finalize()
    return out
end
