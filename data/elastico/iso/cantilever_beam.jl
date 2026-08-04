# Cantilever beam, plane stress, parabolic end traction
# include(datadir("elastico", "iso", "cantilever_beam.jl"))

L = 48.0
D = 12.0
E = 3.0e7
ν = 0.3
P = 1000.0
plane_strain = false
I = D^3 / 12

# End traction ty(y) on x = L
ty_end(y) = -P / (2I) * (D^2 / 4 - y^2)

function u1_beam(x, y)
    return -(P * y) / (6 * E * I) * ((6L - 3x) * x + (2 + ν) * (y^2 - D^2 / 4))
end
function u2_beam(x, y)
    return P / (6 * E * I) * (
        3 * ν * y^2 * (L - x) + (4 + 5ν) * D^2 * x / 4 + (3L - x) * x^2
    )
end

σxx_beam(x, y) = -P * (L - x) * y / I
σxy_beam(x, y) = -P / (2I) * (D^2 / 4 - y^2)
σyy_beam(x, y) = 0.0

line_x_mid = [(L / 2, y) for y in range(-D / 2, D / 2; length=14)]
line_y0 = [(x, 0.0) for x in range(0, L; length=16)]

function mesh_cantilever_beam(; ndiv=16, nome="cantilever_beam", show=false)
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(nome)
    lc = min(L, D) / max(ndiv, 8)
    y0, y1 = -D / 2, D / 2
    p1 = gmsh.model.geo.addPoint(0.0, y0, 0.0, lc)
    p2 = gmsh.model.geo.addPoint(L, y0, 0.0, lc)
    p3 = gmsh.model.geo.addPoint(L, y1, 0.0, lc)
    p4 = gmsh.model.geo.addPoint(0.0, y1, 0.0, lc)
    lb = gmsh.model.geo.addLine(p1, p2)
    lr = gmsh.model.geo.addLine(p2, p3)
    lt = gmsh.model.geo.addLine(p3, p4)
    ll = gmsh.model.geo.addLine(p4, p1)
    cl = gmsh.model.geo.addCurveLoop([lb, lr, lt, ll])
    s = gmsh.model.geo.addPlaneSurface([cl])
    gmsh.model.geo.synchronize()
    nx, ny = max(ndiv, 8), max(ndiv ÷ 4, 3)
    gmsh.model.mesh.setTransfiniteCurve(lb, nx)
    gmsh.model.mesh.setTransfiniteCurve(lt, nx)
    gmsh.model.mesh.setTransfiniteCurve(lr, ny)
    gmsh.model.mesh.setTransfiniteCurve(ll, ny)
    gmsh.model.mesh.setTransfiniteSurface(s)
    gmsh.model.mesh.setRecombine(2, s)
    gmsh.model.addPhysicalGroup(1, [ll], -1, "0;0;0;0")
    gmsh.model.addPhysicalGroup(1, [lb, lt], -1, "1;0;1;0")
    gmsh.model.addPhysicalGroup(1, [lr], -1, "1;0;1;0")
    gmsh.model.addPhysicalGroup(2, [s], -1, "Domain")
    gmsh.model.mesh.generate(2)
    out = datadir("elastico", "iso", nome * ".msh")
    mkpath(dirname(out))
    gmsh.write(out)
    show && gmsh.fltk.run()
    gmsh.finalize()
    return out
end
