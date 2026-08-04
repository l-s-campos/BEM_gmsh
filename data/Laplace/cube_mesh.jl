"""
    mesh_cube(; L=1.0, ndiv=4, nome=\"cubo_bem\", show=false) -> path

Unit cube surface mesh with physical BC names for Laplace:
- bottom z=0: Dirichlet T=0
- top z=L: Dirichlet T=1
- sides: Neumann q=0
"""
function mesh_cube(; L=1.0, ndiv=4, nome="cubo_bem", show=false)
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(nome)
    lc = L / ndiv

    # 8 corners
    p = [
        gmsh.model.geo.addPoint(0, 0, 0, lc),
        gmsh.model.geo.addPoint(L, 0, 0, lc),
        gmsh.model.geo.addPoint(L, L, 0, lc),
        gmsh.model.geo.addPoint(0, L, 0, lc),
        gmsh.model.geo.addPoint(0, 0, L, lc),
        gmsh.model.geo.addPoint(L, 0, L, lc),
        gmsh.model.geo.addPoint(L, L, L, lc),
        gmsh.model.geo.addPoint(0, L, L, lc),
    ]
    # bottom
    l1 = gmsh.model.geo.addLine(p[1], p[2])
    l2 = gmsh.model.geo.addLine(p[2], p[3])
    l3 = gmsh.model.geo.addLine(p[3], p[4])
    l4 = gmsh.model.geo.addLine(p[4], p[1])
    # top
    l5 = gmsh.model.geo.addLine(p[5], p[6])
    l6 = gmsh.model.geo.addLine(p[6], p[7])
    l7 = gmsh.model.geo.addLine(p[7], p[8])
    l8 = gmsh.model.geo.addLine(p[8], p[5])
    # vertical
    l9 = gmsh.model.geo.addLine(p[1], p[5])
    l10 = gmsh.model.geo.addLine(p[2], p[6])
    l11 = gmsh.model.geo.addLine(p[3], p[7])
    l12 = gmsh.model.geo.addLine(p[4], p[8])

    cl_bot = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    cl_top = gmsh.model.geo.addCurveLoop([l5, l6, l7, l8])
    cl_f = gmsh.model.geo.addCurveLoop([l1, l10, -l5, -l9])
    cl_r = gmsh.model.geo.addCurveLoop([l2, l11, -l6, -l10])
    cl_b = gmsh.model.geo.addCurveLoop([l3, l12, -l7, -l11])
    cl_l = gmsh.model.geo.addCurveLoop([l4, l9, -l8, -l12])

    s_bot = gmsh.model.geo.addPlaneSurface([cl_bot])
    s_top = gmsh.model.geo.addPlaneSurface([cl_top])
    s_f = gmsh.model.geo.addPlaneSurface([cl_f])
    s_r = gmsh.model.geo.addPlaneSurface([cl_r])
    s_b = gmsh.model.geo.addPlaneSurface([cl_b])
    s_l = gmsh.model.geo.addPlaneSurface([cl_l])
    gmsh.model.geo.synchronize()

    for ℓ in (l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12)
        gmsh.model.mesh.setTransfiniteCurve(ℓ, ndiv + 1)
    end
    for s in (s_bot, s_top, s_f, s_r, s_b, s_l)
        gmsh.model.mesh.setTransfiniteSurface(s)
        gmsh.model.mesh.setRecombine(2, s)
    end

    # Physical BC on surfaces (2D entities for 3D BEM)
    gmsh.model.addPhysicalGroup(2, [s_bot], -1, "0;0")   # T=0
    gmsh.model.addPhysicalGroup(2, [s_top], -1, "0;1")   # T=1
    gmsh.model.addPhysicalGroup(2, [s_f, s_r, s_b, s_l], -1, "1;0")  # insulated

    gmsh.model.mesh.generate(2)
    out = datadir("Laplace", nome * ".msh")
    mkpath(dirname(out))
    gmsh.write(out)
    show && gmsh.fltk.run()
    gmsh.finalize()
    return out
end
