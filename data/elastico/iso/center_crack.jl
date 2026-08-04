# Center-cracked finite plate — Gmsh mesh + format2d + dual BEM
# Crack faces use physical BC type 5:
#   face A (disp BIE):  "5;2;5;2"
#   face B (trac BIE):  "5;3;5;3"

"""
    mesh_center_crack(; W=5, H=10, a=1, ndiv_b=10, ndiv_h=16, ndiv_crack=16,
                        σ=1.0, nome="center_crack", show=false) -> path

Build a Gmsh boundary mesh of a plate with a central crack and write `.msh`.
Physical groups encode outer BCs and crack type-5 dual faces.
"""
function mesh_center_crack(; W=5.0, H=10.0, a=1.0,
    ndiv_b=10, ndiv_h=16, ndiv_crack=16, σ=1.0,
    nome="center_crack", ordem=2, show=false)

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(nome)
    lc = min(W, H) / max(ndiv_b, 8)

    # outer corners
    p1 = gmsh.model.geo.addPoint(-W, -H, 0, lc)
    p2 = gmsh.model.geo.addPoint(W, -H, 0, lc)
    p3 = gmsh.model.geo.addPoint(W, H, 0, lc)
    p4 = gmsh.model.geo.addPoint(-W, H, 0, lc)
    # crack tips (shared geometry for both faces)
    ptL = gmsh.model.geo.addPoint(-a, 0, 0, lc / 2)
    ptR = gmsh.model.geo.addPoint(a, 0, 0, lc / 2)

    lb = gmsh.model.geo.addLine(p1, p2)   # bottom
    lr = gmsh.model.geo.addLine(p2, p3)   # right
    lt = gmsh.model.geo.addLine(p3, p4)   # top
    ll = gmsh.model.geo.addLine(p4, p1)   # left

    # crack: two coincident faces, opposite orientation → opposite normals
    c_low = gmsh.model.geo.addLine(ptL, ptR)   # face A: L→R  (eq=2)
    c_up = gmsh.model.geo.addLine(ptR, ptL)    # face B: R→L  (eq=3)

    cl = gmsh.model.geo.addCurveLoop([lb, lr, lt, ll])
    s = gmsh.model.geo.addPlaneSurface([cl])
    gmsh.model.geo.synchronize()

    # embed crack in surface (open slit, coincident)
    gmsh.model.mesh.embed(1, [c_low, c_up], 2, s)

    gmsh.model.mesh.setTransfiniteCurve(lb, ndiv_b)
    gmsh.model.mesh.setTransfiniteCurve(lt, ndiv_b)
    gmsh.model.mesh.setTransfiniteCurve(lr, ndiv_h)
    gmsh.model.mesh.setTransfiniteCurve(ll, ndiv_h)
    gmsh.model.mesh.setTransfiniteCurve(c_low, ndiv_crack)
    gmsh.model.mesh.setTransfiniteCurve(c_up, ndiv_crack)

    # BCs: "tx;vx;ty;vy"
    gmsh.model.addPhysicalGroup(1, [lb], -1, "1;0;1;$(-σ)")   # bottom ty=-σ
    gmsh.model.addPhysicalGroup(1, [lt], -1, "1;0;1;$σ")       # top ty=+σ
    gmsh.model.addPhysicalGroup(1, [ll, lr], -1, "1;0;1;0")    # sides free
    # crack type 5 — value carries dual equation id (2 or 3)
    gmsh.model.addPhysicalGroup(1, [c_low], -1, "5;2;5;2")     # disp BIE
    gmsh.model.addPhysicalGroup(1, [c_up], -1, "5;3;5;3")      # traction BIE
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

"""
    solve_center_crack_dual(; kwargs...) -> (mesh, KI_L, KII_L, KI_R, KII_R, KI_ana)

Gmsh → `format2d` (BC type 5) → dual BEM → COD SIFs.
"""
function solve_center_crack_dual(; W=5.0, H=10.0, a=1.0,
    n_bottom=6, n_right=12, n_top=6, n_left=12, n_crack=12,
    E=3000.0, ν=0.2, σ=1.0, plane_strain=true, npg=12, ordem=2)

    msh = mesh_center_crack(; W=W, H=H, a=a,
        ndiv_b=max(n_bottom, n_top), ndiv_h=max(n_right, n_left),
        ndiv_crack=n_crack, σ=σ, ordem=ordem, show=false)

    props = Elasticity(E, ν, 1.0; plane_strain=plane_strain)
    # tipo = ordem (linear=1 → 2 collocation pts, quadratic=2 → 3 pts)
    dad = format2d(msh, props; tipo=ordem, pontointerno=false)

    mesh = dual_mesh_from_bemdata(dad; plane_strain=plane_strain)
    # rigid-body pins on outer boundary
    _pin_plate_rbm!(mesh; W=W, H=H)

    assemble_dual!(mesh; npg=npg)
    solve_dual!(mesh)

    tL, tR = mesh.tip_nodes[1], mesh.tip_nodes[2]
    KI_L, KII_L = sif_cod_dual(mesh, tL; sample=2)
    KI_R, KII_R = sif_cod_dual(mesh, tR; sample=2)
    KI_ana = analytical_KI_center_crack(σ, a; W=W)
    return mesh, KI_L, KII_L, KI_R, KII_R, KI_ana
end
