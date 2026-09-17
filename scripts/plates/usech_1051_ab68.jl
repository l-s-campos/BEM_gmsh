# Useche 10.5.1 a/b = 0.6 and 0.8, book mesh (same as MATLAB driver).
# F = K1b / (Mo √(πa))
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf
using BEM.Plate

b = 1.0
h = b / 2
c = 2 * b
E, ν = 2.1e5, 0.3
Mo = 1.0
props = FSDTProps(; E=E, ν=ν, h=h, q_c=0.0)
W, Ht = b, c
ndiv_b, ndiv_h, ndiv_c = 9, 9, 17   # 8/edge, 16/face

println("="^72)
println(" Useche 10.5.1  a/b=0.6, 0.8  Julia book mesh")
println("="^72)
@printf("  plate [-%g,%g]×[-%g,%g]  h=%g\n", W, W, Ht, Ht, h)
@printf("  mesh  %d/edge outer  %d/crack-face\n", ndiv_b - 1, ndiv_c - 1)
@printf("  %6s %12s %12s %12s %12s %12s\n",
    "a/b", "K1b", "F", "book F", "K2b", "K3b")
flush(stdout)

for ab in (0.6, 0.8)
    a = ab * b
    kb = ab ≈ 0.6 ? 0.095 : 0.134
    mesh = build_rect_fsdt_crack(; W=W, H=Ht, a=a, props=props, Mo=Mo,
        ndiv_b=ndiv_b, ndiv_h=ndiv_h, ndiv_crack=ndiv_c,
        nome="usech1051_ab$(ab)")
    assemble_fsdt_dual!(mesh; npg=12, nsub=10)
    solve_fsdt!(mesh)
    K1b, K2b, K3b, rA, rB, Le = sif_ctod_fsdt(mesh; tip=:right)
    Fb = K1b / (Mo * sqrt(π * a))
    @printf("  %6.1f %12.6f %12.6f %12.3f %12.3e %12.3e  (n=%d rA/Le=%.3f rB/Le=%.3f Le=%.5f)\n",
        ab, K1b, Fb, kb, K2b, K3b, length(mesh.nodes), rA / Le, rB / Le, Le)
    flush(stdout)
end
println("Done.")
