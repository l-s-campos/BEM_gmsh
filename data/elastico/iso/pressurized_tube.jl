# Pressurized thick-walled cylinder (quarter model), plane strain
# include(datadir("elastico", "iso", "pressurized_tube.jl"))

Ra = 50.0          # mm inner radius
Rb = 100.0         # mm outer radius
E = 200_000.0      # MPa
ν = 0.32
P = 100.0          # N/mm internal pressure
plane_strain = true

# Lamé (internal P, external free)
function σr_tube(r)
    c = Ra^2 * P / (Rb^2 - Ra^2)
    return c * (1 - Rb^2 / r^2)
end
function σθ_tube(r)
    c = Ra^2 * P / (Rb^2 - Ra^2)
    return c * (1 + Rb^2 / r^2)
end
function ur_tube(r)
    c = Ra^2 * P / (Rb^2 - Ra^2)
    return ((1 + ν) / E) * c * ((1 - 2ν) * r + Rb^2 / r)
end

internal_points = [
    (0.0667, 0.0333),
    (0.0333, 0.0667),
    (0.0667, 0.0667),
]

function mesh_pressurized_tube(; a=Ra, b=Rb, ndiv=12, nome="pressurized_tube",
        show=false, ordem=1, nr=nothing, nθ=nothing, progression=1.0)
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(nome)
    lc = (b - a) / max(ndiv, 4)
    c = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, lc)
    p1 = gmsh.model.geo.addPoint(a, 0.0, 0.0, lc)
    p2 = gmsh.model.geo.addPoint(b, 0.0, 0.0, lc)
    p3 = gmsh.model.geo.addPoint(0.0, b, 0.0, lc)
    p4 = gmsh.model.geo.addPoint(0.0, a, 0.0, lc)
    lbot = gmsh.model.geo.addLine(p1, p2)
    aout = gmsh.model.geo.addCircleArc(p2, c, p3)
    lleft = gmsh.model.geo.addLine(p3, p4)
    ain = gmsh.model.geo.addCircleArc(p4, c, p1)
    cl = gmsh.model.geo.addCurveLoop([lbot, aout, lleft, ain])
    s = gmsh.model.geo.addPlaneSurface([cl])
    gmsh.model.geo.synchronize()
    nr = something(nr, max(ndiv ÷ 2, 4))
    nθ = something(nθ, ndiv)
    prog = float(progression)
    gmsh.model.mesh.setTransfiniteCurve(lbot, nr, "Progression", prog)
    gmsh.model.mesh.setTransfiniteCurve(aout, nθ)
    gmsh.model.mesh.setTransfiniteCurve(lleft, nr, "Progression",
        abs(prog) > 1e-12 ? 1 / prog : 1.0)
    gmsh.model.mesh.setTransfiniteCurve(ain, nθ)
    gmsh.model.mesh.setTransfiniteSurface(s)
    gmsh.model.mesh.setRecombine(2, s)
    gmsh.model.addPhysicalGroup(1, [lbot], -1, "1;0;0;0")
    gmsh.model.addPhysicalGroup(1, [lleft], -1, "0;0;1;0")
    gmsh.model.addPhysicalGroup(1, [aout], -1, "1;0;1;0")
    gmsh.model.addPhysicalGroup(1, [ain], -1, "1;0;1;0")
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
