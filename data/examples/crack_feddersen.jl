# Center crack — Gmsh (BC type 5) + format2d discontinuous + dual BEM
using DrWatson
@quickactivate :BEM
using .Crack
include(datadir("elastico", "iso", "center_crack.jl"))

println("="^60)
println(" Example: center crack Feddersen KI (Gmsh type-5 + dual BEM)")
println("="^60)

W, H, a, σ = 5.0, 10.0, 1.0, 1.0
E, ν = 3000.0, 0.2

msh = mesh_center_crack(; W=W, H=H, a=a, ndiv_b=8, ndiv_h=12, ndiv_crack=12,
    σ=σ, ordem=2, show=false)
dad = format2d(msh, Elasticity(E, ν, 1.0; plane_strain=true); tipo=2, pontointerno=false)
println(dad)
println("  crack DOFs (type 5): ", count(==(Crack.CRACK_BC), dad.BC))

mesh = dual_mesh_from_bemdata(dad)
# RBM pins
BEM.Crack._pin_plate_rbm!(mesh; W=W, H=H)
assemble_dual!(mesh; npg=10)
solve_dual!(mesh)

KI_L, KII_L = sif_cod_dual(mesh, mesh.tip_nodes[1])
KI_R, KII_R = sif_cod_dual(mesh, mesh.tip_nodes[2])
KIana = analytical_KI_center_crack(σ, a; W=W)
KIn = 0.5 * (abs(KI_L) + abs(KI_R))
err = abs(KIn - KIana) / KIana
println("  KI_num / KI_ana = ", KIn, " / ", KIana)
println("  |KII| mean      = ", 0.5 * (abs(KII_L) + abs(KII_R)))
println("  rel_error       = ", err)
println("  (coarse mesh — qualitative COD; refine ndiv_crack for <10%)")
@assert isfinite(KIn) && KIn > 0
@assert 0.5 * (abs(KII_L) + abs(KII_R)) < 0.5 * KIana
println("OK.")
