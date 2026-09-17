# Escudero 2018 §6 meshes. Cylinder numbers from appendix O (not the σY=0.8 typo).
# include(datadir("elastico", "iso", "escudero2018.jl"))

using DrWatson: datadir

"""Quarter plate with a hole, Escudero §6.1. Box `[0,L]×[0,W]`, hole radius `R` at the origin."""
function mesh_escudero_plate_hole(; L=180.0, W=100.0, R=50.0, ndiv=8,
        nome="escudero_plate_hole", show=false, ordem=1)
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(nome)
    lc = min(L, W) / max(ndiv, 4)
    o = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, lc)
    p1 = gmsh.model.geo.addPoint(R, 0.0, 0.0, lc)
    p2 = gmsh.model.geo.addPoint(L, 0.0, 0.0, lc)
    p3 = gmsh.model.geo.addPoint(L, W, 0.0, lc)
    p4 = gmsh.model.geo.addPoint(0.0, W, 0.0, lc)
    p5 = gmsh.model.geo.addPoint(0.0, R, 0.0, lc)
    lbot = gmsh.model.geo.addLine(p1, p2)
    lright = gmsh.model.geo.addLine(p2, p3)
    ltop = gmsh.model.geo.addLine(p3, p4)
    lleft = gmsh.model.geo.addLine(p4, p5)
    ahole = gmsh.model.geo.addCircleArc(p5, o, p1)
    cl = gmsh.model.geo.addCurveLoop([lbot, lright, ltop, lleft, ahole])
    s = gmsh.model.geo.addPlaneSurface([cl])
    gmsh.model.geo.synchronize()
    nR = max(ndiv ÷ 2, 4)
    gmsh.model.mesh.setTransfiniteCurve(lbot, ndiv)
    gmsh.model.mesh.setTransfiniteCurve(lright, nR)
    gmsh.model.mesh.setTransfiniteCurve(ltop, ndiv)
    gmsh.model.mesh.setTransfiniteCurve(lleft, nR)
    gmsh.model.mesh.setTransfiniteCurve(ahole, ndiv)
    gmsh.model.addPhysicalGroup(1, [lbot], -1, "1;0;0;0")     # y=0 roller
    gmsh.model.addPhysicalGroup(1, [lleft], -1, "0;0;1;0")    # x=0 roller
    gmsh.model.addPhysicalGroup(1, [lright], -1, "1;0;1;0")   # traction set later
    gmsh.model.addPhysicalGroup(1, [ltop, ahole], -1, "1;0;1;0")
    gmsh.model.addPhysicalGroup(2, [s], -1, "Domain")
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(ordem)
    out = datadir("elastico", "iso", nome * ".msh")
    mkpath(dirname(out))
    gmsh.write(out)
    show && gmsh.fltk.run()
    gmsh.finalize()
    return out
end
