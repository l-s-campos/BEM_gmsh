# Useche 10.5.1 a/b = 0.6, 0.8: Dual BEM CTOD vs Reissner XBEM (Dolbow/Hui–Zehnder).
# Book mesh: 8/edge, 16/face.  F_table = K1b/(Mo√(πa)) = K1_dolbow/(Mo√a).
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
ndiv_b, ndiv_h, ndiv_c = 9, 9, 17

println("="^72)
println(" Useche 10.5.1  Dual CTOD vs Reissner XBEM  (book mesh)")
println("="^72)
@printf("  plate [-%g,%g]×[-%g,%g]  h=%g  8/edge 16/face\n", W, W, Ht, Ht, h)
@printf("  %6s %12s %12s %12s %12s %12s\n",
    "a/b", "Dual F", "XBEM CTOD", "XBEM K", "MATLAB F", "book F")
flush(stdout)

matlab_F = Dict(0.6 => 1.102436, 0.8 => 1.581744)
book_F = Dict(0.6 => 0.095, 0.8 => 0.134)

for ab in (0.6, 0.8)
    a = ab * b
    mesh = build_rect_fsdt_crack(; W=W, H=Ht, a=a, props=props, Mo=Mo,
        ndiv_b=ndiv_b, ndiv_h=ndiv_h, ndiv_crack=ndiv_c,
        nome="usech1051_xbem_$(ab)")
    assemble_fsdt_dual!(mesh; npg=12, nsub=10)
    solve_fsdt!(mesh)
    K1b, _, _, _, _, _ = sif_ctod_fsdt(mesh; tip=:right)
    Fd = K1b / (Mo * sqrt(π * a))
    _, K1, K2, K3, tips, _ = solve_fsdt_xbem!(mesh; n_enr=3, npg=12, nsub=10, n_v=3)
    ir = argmax(p[1] for p in tips)
    K1b_x, _, _, _, _, _ = sif_ctod_fsdt(mesh; tip=:right)
    Fx_ctod = K1b_x / (Mo * sqrt(π * a))
    Fx_k = abs(K1[ir]) / (Mo * sqrt(a))
    @printf("  %6.1f %12.6f %12.6f %12.6f %12.6f %12.3f  (K2=%.2e K3=%.2e)\n",
        ab, Fd, Fx_ctod, Fx_k, matlab_F[ab], book_F[ab], K2[ir], K3[ir])
    flush(stdout)
end
println("  Dual/XBEM CTOD = K1b/(Mo√(πa));  XBEM K = extra-DOF K1_dolbow/(Mo√a).")
println("  book 0.095/0.134 is OCR.")
println("Done.")
