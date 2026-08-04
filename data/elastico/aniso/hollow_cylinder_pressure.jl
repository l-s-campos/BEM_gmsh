# Hollow pressurized cylinder (quarter model), quasi-isotropic / orthotropic
# include(datadir("elastico", "aniso", "hollow_cylinder_pressure.jl"))
#
# BC (quarter): N1 bottom uy=0 & tx free; N2 outer free; N3 left ux=0;
#               N4 inner tn = -P

ra = 0.05            # m inner radius
rb = 0.10            # m outer radius
E1 = 1000.0          # GPa  (longitudinal; ≈ isotropic when E2≈E1)
E2 = 1000.1          # GPa  (transverse — slight anisotropy for Lekhnitskii path)
G12 = 384.61         # GPa
ν12 = 0.3
P = 100.0            # N (normal pressure magnitude on inner wall)
plane_strain = true
nurbs_p = 2          # thesis IGABEM order (info only)

# Isotropic Lamé closed form (use E≈E1, ν=ν12 when E1≈E2)
function ur_hollow(r; E=E1, ν=ν12)
    return ((1 + ν) * P * ra^2) / ((rb^2 - ra^2) * E) * ((1 - 2ν) * r + rb^2 / r)
end
function σr_hollow(r)
    return (P * ra^2) / (rb^2 - ra^2) * (1 - rb^2 / r^2)
end
function σθ_hollow(r)
    return (P * ra^2) / (rb^2 - ra^2) * (1 + rb^2 / r^2)
end

function mesh_hollow_cylinder(; ndiv=12, nome="hollow_cylinder_pressure", show=false)
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(nome)
    lc = (rb - ra) / max(ndiv, 4)
    c = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, lc)
    p1 = gmsh.model.geo.addPoint(ra, 0.0, 0.0, lc)
    p2 = gmsh.model.geo.addPoint(rb, 0.0, 0.0, lc)
    p3 = gmsh.model.geo.addPoint(0.0, rb, 0.0, lc)
    p4 = gmsh.model.geo.addPoint(0.0, ra, 0.0, lc)
    lbot = gmsh.model.geo.addLine(p1, p2)
    aout = gmsh.model.geo.addCircleArc(p2, c, p3)
    lleft = gmsh.model.geo.addLine(p3, p4)
    ain = gmsh.model.geo.addCircleArc(p4, c, p1)
    cl = gmsh.model.geo.addCurveLoop([lbot, aout, lleft, ain])
    s = gmsh.model.geo.addPlaneSurface([cl])
    gmsh.model.geo.synchronize()
    nr, nθ = max(ndiv ÷ 2, 4), ndiv
    for (crv, n) in ((lbot, nr), (aout, nθ), (lleft, nr), (ain, nθ))
        gmsh.model.mesh.setTransfiniteCurve(crv, n)
    end
    gmsh.model.mesh.setTransfiniteSurface(s)
    gmsh.model.mesh.setRecombine(2, s)
    gmsh.model.addPhysicalGroup(1, [lbot], -1, "1;0;0;0")
    gmsh.model.addPhysicalGroup(1, [lleft], -1, "0;0;1;0")
    gmsh.model.addPhysicalGroup(1, [aout], -1, "1;0;1;0")
    gmsh.model.addPhysicalGroup(1, [ain], -1, "1;0;1;0")  # fill tn=-P after format2d
    gmsh.model.addPhysicalGroup(2, [s], -1, "Domain")
    gmsh.model.mesh.generate(2)
    out = datadir("elastico", "aniso", nome * ".msh")
    mkpath(dirname(out))
    gmsh.write(out)
    show && gmsh.fltk.run()
    gmsh.finalize()
    return out
end
