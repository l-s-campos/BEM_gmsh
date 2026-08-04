# Square plate with multiple circular holes (uniform hole pattern)
# Hole-area / plate-area ratio fixed at 12.47%. Left fixed, right remote σx.
# include(datadir("elastico", "aniso", "plate_multi_holes.jl"))

# Carbon fabric / phenolic
E1 = 20.0            # GPa
E2 = 19.0            # GPa
G12 = 6.8            # GPa
ν12 = 0.23
σ∞ = 1.0             # MPa remote tension in x
plane_strain = false
hole_area_fraction = 0.1247

# Plate side length a (set as needed); holes placed on a regular grid
a = 10.0
# Example 3×3 hole pattern with equal radii so Σ π r_i² / a² = hole_area_fraction
n_holes_side = 3
n_holes = n_holes_side^2
r_hole = sqrt(hole_area_fraction * a^2 / (n_holes * π))

# FMM / GMRES settings used in thesis (info)
fmm_expansion_terms = 8
fmm_max_elements_per_leaf = 5
gmres_tol = 1e-6

# BC: N1 bottom uy=0; N2 right tx=σ∞; N3 top free; N4 left ux=0; holes free
# Mesh: build plate with n×n holes via OCC cut
function mesh_plate_multi_holes(; nside=n_holes_side, a=a, nome="plate_multi_holes",
    ndiv_edge=20, ndiv_hole=12, show=false)
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(nome)
    lc = a / max(ndiv_edge, 10)
    plate = gmsh.model.occ.addRectangle(0.0, 0.0, 0.0, a, a)
    holes = Int[]
    # centers on a regular grid inset by margin
    margin = a / (2 * nside)
    xs = range(margin, a - margin; length=nside)
    ys = range(margin, a - margin; length=nside)
    r = sqrt(hole_area_fraction * a^2 / (nside^2 * π))
    for yi in ys, xi in xs
        push!(holes, gmsh.model.occ.addDisk(xi, yi, 0.0, r, r))
    end
    gmsh.model.occ.cut([(2, plate)], [(2, h) for h in holes])
    gmsh.model.occ.synchronize()
    # Physical groups: identify boundary curves by bbox
    curves = gmsh.model.getEntities(1)
    left, right, bottom, top, hole_c = Int[], Int[], Int[], Int[], Int[]
    tol = 1e-6 * a
    for (_, tag) in curves
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(1, tag)
        if abs(xmax - xmin) < tol && abs(xmin) < tol
            push!(left, tag)
        elseif abs(xmax - xmin) < tol && abs(xmax - a) < tol
            push!(right, tag)
        elseif abs(ymax - ymin) < tol && abs(ymin) < tol
            push!(bottom, tag)
        elseif abs(ymax - ymin) < tol && abs(ymax - a) < tol
            push!(top, tag)
        else
            push!(hole_c, tag)
        end
    end
    isempty(bottom) || gmsh.model.addPhysicalGroup(1, bottom, -1, "1;0;0;0")
    isempty(right) || gmsh.model.addPhysicalGroup(1, right, -1, "1;$σ∞;1;0")
    isempty(top) || gmsh.model.addPhysicalGroup(1, top, -1, "1;0;1;0")
    isempty(left) || gmsh.model.addPhysicalGroup(1, left, -1, "0;0;1;0")
    isempty(hole_c) || gmsh.model.addPhysicalGroup(1, hole_c, -1, "1;0;1;0")
    surfs = [t for (d, t) in gmsh.model.getEntities(2)]
    isempty(surfs) || gmsh.model.addPhysicalGroup(2, surfs, -1, "Domain")
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)
    gmsh.model.mesh.generate(2)
    out = datadir("elastico", "aniso", nome * ".msh")
    mkpath(dirname(out))
    gmsh.write(out)
    show && gmsh.fltk.run()
    gmsh.finalize()
    return out
end
