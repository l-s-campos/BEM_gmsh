# Center crack — Gmsh (BC type 5) + format2d discontinuous + dual BEM on BEMdata
using DrWatson
@quickactivate :BEM
using BEM.Crack
using .Crack

println("="^60)
println(" Example: center crack Feddersen KI (Gmsh type-5 + dual BEM)")
println("="^60)

W, H, a, σ = 5.0, 10.0, 1.0, 1.0
E, ν = 3000.0, 0.2

dad = dual_elasticity_problem(; W=W, H=H, a=a, E=E, ν=ν, σ=σ,
    ndiv_b=8, ndiv_h=12, ndiv_crack=12, ordem=2, nome="center_crack")
println(dad)
println("  crack DOFs (type 5 rewritten): ", count(i -> dad.eq_type[i] in (2, 3), 1:dad.n))

assemble_dual!(dad; npg=10, threaded=false)
solve_dual!(dad; threaded=false)

KI_L, KII_L = sif_cod_dual(dad, dad.tip_nodes[1])
KI_R, KII_R = sif_cod_dual(dad, dad.tip_nodes[2])
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
