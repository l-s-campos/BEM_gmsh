# Rotating orthotropic disk (centrifugal body force), quarter model
# include(datadir("elastico", "aniso", "rotating_orthotropic_disk.jl"))
#
# BC: N1 (x-axis) uy=0; N2 outer free (after body-force particular solution);
#     N3 (y-axis) ux=0. Body force ρ ω² r êr — use DIBEM / particular integrals.

R = 1.0              # m
E1 = 17.24           # GPa (longitudinal)
E2 = 48.26           # GPa (transverse)
G12 = 6.89           # GPa
ν12 = 0.29
ω = 20.0             # rad/s
ρ = 1.0              # kg/m³
fiber_angle_deg = 0.0
n_layers = 2
nurbs_p = 2

# Lekhnitskii / BEM props helper (when using AnisotropicElasticity)
# p = lekhnitskii_params(E1, E2, G12, ν12; θ_deg=fiber_angle_deg)

function mesh_rotating_disk(; ndiv=16, nome="rotating_orthotropic_disk", show=false)
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(nome)
    lc = R / max(ndiv, 8)
    c = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, lc)
    p1 = gmsh.model.geo.addPoint(R, 0.0, 0.0, lc)
    p2 = gmsh.model.geo.addPoint(0.0, R, 0.0, lc)
    o = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, lc)
    # quarter disk: radius arc + two radii
    # use center point shared
    lbot = gmsh.model.geo.addLine(o, p1)           # N1
    arc = gmsh.model.geo.addCircleArc(p1, c, p2)   # N2 outer
    lleft = gmsh.model.geo.addLine(p2, o)          # N3
    cl = gmsh.model.geo.addCurveLoop([lbot, arc, lleft])
    s = gmsh.model.geo.addPlaneSurface([cl])
    gmsh.model.geo.synchronize()
    nθ = max(ndiv, 8)
    nr = max(ndiv ÷ 2, 4)
    gmsh.model.mesh.setTransfiniteCurve(lbot, nr)
    gmsh.model.mesh.setTransfiniteCurve(arc, nθ)
    gmsh.model.mesh.setTransfiniteCurve(lleft, nr)
    gmsh.model.mesh.setTransfiniteSurface(s)
    gmsh.model.addPhysicalGroup(1, [lbot], -1, "1;0;0;0")   # uy=0
    gmsh.model.addPhysicalGroup(1, [arc], -1, "1;0;1;0")    # free (+ body force via domain)
    gmsh.model.addPhysicalGroup(1, [lleft], -1, "0;0;1;0")  # ux=0
    gmsh.model.addPhysicalGroup(2, [s], -1, "Domain")
    gmsh.model.mesh.generate(2)
    out = datadir("elastico", "aniso", nome * ".msh")
    mkpath(dirname(out))
    gmsh.write(out)
    show && gmsh.fltk.run()
    gmsh.finalize()
    return out
end

# centrifugal body force density (N/m³ if consistent SI): f = ρ ω² (x, y)
body_force_centrifugal(x, y) = (ρ * ω^2 * x, ρ * ω^2 * y)
