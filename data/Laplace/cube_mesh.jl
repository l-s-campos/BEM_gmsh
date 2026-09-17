using DrWatson: datadir

"""Build the six faces of `[0,L]³` as transfinite surfaces. Returns face tags.

`recombine=true` (default) emits quads; `false` leaves Gmsh triangles.
"""
function _cube_surfaces!(; L=1.0, ndiv=4, nome="cubo", recombine::Bool=true)
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(nome)
    lc = L / ndiv
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
    l1 = gmsh.model.geo.addLine(p[1], p[2])
    l2 = gmsh.model.geo.addLine(p[2], p[3])
    l3 = gmsh.model.geo.addLine(p[3], p[4])
    l4 = gmsh.model.geo.addLine(p[4], p[1])
    l5 = gmsh.model.geo.addLine(p[5], p[6])
    l6 = gmsh.model.geo.addLine(p[6], p[7])
    l7 = gmsh.model.geo.addLine(p[7], p[8])
    l8 = gmsh.model.geo.addLine(p[8], p[5])
    l9 = gmsh.model.geo.addLine(p[1], p[5])
    l10 = gmsh.model.geo.addLine(p[2], p[6])
    l11 = gmsh.model.geo.addLine(p[3], p[7])
    l12 = gmsh.model.geo.addLine(p[4], p[8])
    # Bottom: CW from +z so the surface normal is outward (−z). Top/sides
    # are already outward with the right-hand rule on these loops.
    cl_bot = gmsh.model.geo.addCurveLoop([-l4, -l3, -l2, -l1])
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
    faces = (s_bot, s_top, s_f, s_r, s_b, s_l)
    for s in faces
        gmsh.model.mesh.setTransfiniteSurface(s)
        recombine && gmsh.model.mesh.setRecombine(2, s)
    end
    return faces
end

function _write_cube_msh(nome)
    gmsh.model.mesh.generate(2)
    out = datadir("Laplace", nome * ".msh")
    mkpath(dirname(out))
    gmsh.write(out)
    gmsh.finalize()
    return out
end

"""
    mesh_cube(; L=1.0, ndiv=4, nome="cubo_bem", show=false) -> path

Unit cube surface mesh with physical BC names for Laplace ``T=z`` (``L=1``):
- bottom z=0: Dirichlet T=0
- top z=L: Dirichlet T=1
- sides: Neumann q=0
"""
function mesh_cube(; L=1.0, ndiv=4, nome="cubo_bem", show=false, recombine::Bool=true)
    s_bot, s_top, s_f, s_r, s_b, s_l = _cube_surfaces!(; L=L, ndiv=ndiv, nome=nome, recombine=recombine)
    gmsh.model.addPhysicalGroup(2, [s_bot], -1, "0;0")
    gmsh.model.addPhysicalGroup(2, [s_top], -1, "0;1")
    gmsh.model.addPhysicalGroup(2, [s_f, s_r, s_b, s_l], -1, "1;0")
    show && (gmsh.model.mesh.generate(2); gmsh.fltk.run())
    return _write_cube_msh(nome)
end

"""
    mesh_unit_cube(; L=1.0, ndiv=4, nome="cubo_unit", bc="0;0") -> path

Cube `[0,L]³` with the same physical BC string on every face. Use with
`apply_analytical_bc!` for 3-D patch tests. Elasticity (3 DOF):
`bc="0;0;0;0;0;0"`.
"""
function mesh_unit_cube(; L=1.0, ndiv=4, nome="cubo_unit", bc="0;0", show=false,
        recombine::Bool=true)
    faces = _cube_surfaces!(; L=L, ndiv=ndiv, nome=nome, recombine=recombine)
    gmsh.model.addPhysicalGroup(2, collect(faces), -1, bc)
    show && (gmsh.model.mesh.generate(2); gmsh.fltk.run())
    return _write_cube_msh(nome)
end

"""Uniform interior sample on `(0,L)³` (not on the boundary)."""
function cube_interior_grid(L=1.0, n::Integer=2)
    n >= 1 || throw(ArgumentError("n ≥ 1"))
    xs = n == 1 ? (L / 2,) : range(0.25L, 0.75L; length=n)
    return [Point3D(x, y, z) for x in xs, y in xs, z in xs]
end
