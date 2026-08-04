# Circular cavity under internal pressure (infinite medium, truncated annulus)
# include(datadir("elastico", "iso", "pressurized_cavity.jl"))

Ra = 3.0             # m cavity radius
E = 207_900.0        # Pa
ν = 0.1
P = 100.0            # Pa
plane_strain = true
R_inf = 20.0 * Ra    # outer truncation

σr_cavity(r) = -P * Ra^2 / r^2
ur_cavity(r) = P * Ra^2 * (1 + ν) / (E * r)

probe_radii = (3.0, 4.0, 6.0, 10.0, 20.0)

function mesh_pressurized_cavity(; ndiv=24, nome="pressurized_cavity", show=false, b=R_inf)
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(nome)
    lc = (b - Ra) / max(ndiv, 8)
    c = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, lc)
    θs = (0.0, π / 2, π, 3π / 2)
    pin = [gmsh.model.geo.addPoint(Ra * cos(θ), Ra * sin(θ), 0.0, lc) for θ in θs]
    pout = [gmsh.model.geo.addPoint(b * cos(θ), b * sin(θ), 0.0, lc) for θ in θs]
    ain = [
        gmsh.model.geo.addCircleArc(pin[1], c, pin[2]),
        gmsh.model.geo.addCircleArc(pin[2], c, pin[3]),
        gmsh.model.geo.addCircleArc(pin[3], c, pin[4]),
        gmsh.model.geo.addCircleArc(pin[4], c, pin[1]),
    ]
    aout = [
        gmsh.model.geo.addCircleArc(pout[1], c, pout[2]),
        gmsh.model.geo.addCircleArc(pout[2], c, pout[3]),
        gmsh.model.geo.addCircleArc(pout[3], c, pout[4]),
        gmsh.model.geo.addCircleArc(pout[4], c, pout[1]),
    ]
    cl_out = gmsh.model.geo.addCurveLoop(aout)
    cl_in = gmsh.model.geo.addCurveLoop([ain[4], ain[3], ain[2], ain[1]])
    s = gmsh.model.geo.addPlaneSurface([cl_out, cl_in])
    gmsh.model.geo.synchronize()
    nseg = max(ndiv ÷ 4, 4)
    for crv in vcat(ain, aout)
        gmsh.model.mesh.setTransfiniteCurve(crv, nseg)
    end
    gmsh.model.addPhysicalGroup(1, ain, -1, "1;0;1;0")
    gmsh.model.addPhysicalGroup(1, aout, -1, "1;0;1;0")
    gmsh.model.addPhysicalGroup(2, [s], -1, "Domain")
    gmsh.model.mesh.generate(2)
    out = datadir("elastico", "iso", nome * ".msh")
    mkpath(dirname(out))
    gmsh.write(out)
    show && gmsh.fltk.run()
    gmsh.finalize()
    return out
end
