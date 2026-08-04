# Infinite plate with circular hole — orthotropic stress concentration (Lekhnitskii)
# Tension in y (or x). Quarter model of a large plate.
# include(datadir("elastico", "aniso", "plate_hole_orthotropic.jl"))

# Geometry (computational quarter plate)
r_hole = 1.0         # hole radius
L = 10.0             # outer half-width / height (≫ r for “infinite” plate)
σ∞ = 1.0             # remote uniaxial tension magnitude
tension_dir = :y     # :y as in FCT example; :x for other applications
r_over_l = 0.1       # thesis also studies 0.5

# --- Graphite / epoxy (Wang; Qin; Lei 2017) — Table 6.3 ---
graphite_epoxy = (
    name = "graphite_epoxy",
    E1 = 181.0,      # GPa
    E2 = 10.3,       # GPa
    G12 = 7.17,      # GPa
    ν12 = 0.28,
)

# Analytical FCT (Table 6.4) for graphite/epoxy, remote tension
# KT,max at point A; KT,min at B — fiber angles θ = 0°, 45°, 90°
KT_max_analytic = Dict(0 => 2.37177, 45 => 2.38043, 90 => 6.75048)
KT_min_analytic = Dict(0 => -4.19199, 45 => -0.52892, 90 => -0.23855)

# Fiber orientations studied
fiber_angles_deg = (0, 45, 90)

# BC (quarter, tension in y on top edge N3): 
# N1 bottom: tx free, uy=0; N2 right free; N3 top: ty=σ∞; N4 left ux=0; N5 hole free

function mesh_plate_hole_orthotropic(; ndiv=16, nome="plate_hole_orthotropic",
    r=r_hole, Lout=L, show=false)
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(nome)
    lc = Lout / max(ndiv, 8)
    c = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, lc)
    ph1 = gmsh.model.geo.addPoint(r, 0.0, 0.0, lc)
    ph2 = gmsh.model.geo.addPoint(0.0, r, 0.0, lc)
    po1 = gmsh.model.geo.addPoint(Lout, 0.0, 0.0, lc)
    po2 = gmsh.model.geo.addPoint(Lout, Lout, 0.0, lc)
    po3 = gmsh.model.geo.addPoint(0.0, Lout, 0.0, lc)
    lb = gmsh.model.geo.addLine(ph1, po1)          # N1 bottom
    lr = gmsh.model.geo.addLine(po1, po2)          # N2 right
    lt = gmsh.model.geo.addLine(po2, po3)          # N3 top (tension y)
    ll = gmsh.model.geo.addLine(po3, ph2)          # N4 left
    ah = gmsh.model.geo.addCircleArc(ph2, c, ph1)  # N5 hole
    cl = gmsh.model.geo.addCurveLoop([lb, lr, lt, ll, ah])
    s = gmsh.model.geo.addPlaneSurface([cl])
    gmsh.model.geo.synchronize()
    for crv in (lb, lr, lt, ll, ah)
        gmsh.model.mesh.setTransfiniteCurve(crv, max(ndiv, 4))
    end
    gmsh.model.addPhysicalGroup(1, [lb], -1, "1;0;0;0")
    gmsh.model.addPhysicalGroup(1, [lr], -1, "1;0;1;0")
    gmsh.model.addPhysicalGroup(1, [lt], -1, "1;0;1;$σ∞")
    gmsh.model.addPhysicalGroup(1, [ll], -1, "0;0;1;0")
    gmsh.model.addPhysicalGroup(1, [ah], -1, "1;0;1;0")
    gmsh.model.addPhysicalGroup(2, [s], -1, "Domain")
    gmsh.model.mesh.generate(2)
    out = datadir("elastico", "aniso", nome * ".msh")
    mkpath(dirname(out))
    gmsh.write(out)
    show && gmsh.fltk.run()
    gmsh.finalize()
    return out
end

# Lekhnitskii FCT (Sollero): KT,max ≈ ± Re[1 + i(μ1+μ2)/(μ1 μ2)] etc. — use table above
# Stress along ligament x>r (Nuismer–Whitney, orthotropic):
function σy_ligament(x, KT_max; r=r_hole, σ=σ∞)
    x <= r && return NaN
    ξ = r / x
    return (σ / 2) * (2 + ξ^2 + 3ξ^4 - (KT_max - 3) * (5ξ^6 - 7ξ^8))
end
