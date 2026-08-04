# Plate with circular hole under remote uniaxial tension (Kirsch), plane stress
# include(datadir("elastico", "iso", "plate_with_hole.jl"))

R = 50.0             # mm hole radius
E = 1.0e5            # Pa
ν = 0.25
P = 1.0              # N/mm remote tension (x)
plane_strain = false
L = 200.0            # mm outer half-width (finite plate ≈ 4R)

ν_eq = ν / (1 + ν)
E_eq = E * (1 + 2ν) / (1 + ν)^2

function σ11_hole(r, θ)
    a = R
    return (P / 2) * (1 - a^2 / r^2) + (P / 2) * (1 - 4a^2 / r^2 + 3a^4 / r^4) * cos(2θ)
end
function σ22_hole(r, θ)
    a = R
    return (P / 2) * (1 + a^2 / r^2) - (P / 2) * (1 + 3a^4 / r^4) * cos(2θ)
end
function σ12_hole(r, θ)
    a = R
    return -(P / 2) * (1 + 2a^2 / r^2 - 3a^4 / r^4) * sin(2θ)
end

internal_points = [
    (1.6667, 1.6667),
    (3.3333, 1.6667),
    (1.6667, 3.3333),
    (3.3333, 3.3333),
]

function mesh_plate_with_hole(; ndiv=12, nome="plate_with_hole", show=false)
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(nome)
    lc = L / max(ndiv, 8)
    c = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, lc)
    ph1 = gmsh.model.geo.addPoint(R, 0.0, 0.0, lc)
    ph2 = gmsh.model.geo.addPoint(0.0, R, 0.0, lc)
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
    gmsh.model.addPhysicalGroup(1, [ll], -1, "0;0;1;0")
    gmsh.model.addPhysicalGroup(1, [lr], -1, "1;$P;1;0")
    gmsh.model.addPhysicalGroup(1, [lt, ah], -1, "1;0;1;0")
    gmsh.model.addPhysicalGroup(2, [s], -1, "Domain")
    gmsh.model.mesh.generate(2)
    out = datadir("elastico", "iso", nome * ".msh")
    mkpath(dirname(out))
    gmsh.write(out)
    show && gmsh.fltk.run()
    gmsh.finalize()
    return out
end
