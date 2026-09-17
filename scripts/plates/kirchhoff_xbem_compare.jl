# Kirchhoff Dual vs Reissner Dual/XBEM.
# Centre crack, far-field Mo. Infinite Kirchhoff F = K1/(Mo √a) → 1.
# Reissner Sih (t/a=2) Φ≈0.82. Same Gmsh twins (`mesh_center_crack`).
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf
using BEM.Plate

a = 0.2
W, H = 1.0, 2.0
E, ν = 2.1e5, 0.3
Mo = 1.0
ndiv_b = ndiv_h = 5
ndiv_c = 7
den = Mo * sqrt(a)

println("="^78)
println(" Kirchhoff Dual vs Reissner Dual/XBEM   centre crack  a=$a")
println("="^78)

function run_reissner(h; α=0.0)
    props = FSDTProps(; E=E, ν=ν, h=h)
    mesh = build_rect_fsdt_crack(; W=W, H=H, a=a, α=α, props=props, Mo=Mo,
        ndiv_b=ndiv_b, ndiv_h=ndiv_h, ndiv_crack=ndiv_c,
        nome="cmp_fsdt_$(h)_$(α)")
    assemble_fsdt_dual!(mesh; npg=8, nsub=6)
    solve_fsdt!(mesh)
    K1b, K2b, K3b, _, _, _ = sif_ctod_fsdt(mesh; tip=:right, absK1=false)
    Fd = (abs(K1b), abs(K2b), abs(K3b)) ./ sqrt(π) ./ den
    _, K1, K2, K3, tips, _ = solve_fsdt_xbem!(mesh; n_enr=2, npg=8, nsub=6, n_v=3)
    ir = argmax(p[1] + 1e-9 * p[2] for p in tips)
    Fx = (abs(K1[ir]), abs(K2[ir]), abs(K3[ir])) ./ den
    return Fd, Fx
end

function run_kirchhoff(props; α=0.0, tag="iso")
    mesh = build_rect_plate_crack(; W=W, H=H, a=a, α=α, props=props, Mo=Mo,
        ndiv_b=ndiv_b, ndiv_h=ndiv_h, ndiv_crack=ndiv_c,
        nome="cmp_k_$(tag)_$(α)")
    assemble_plate_dual!(mesh; npg=8, nsub=6)
    solve_plate!(mesh)
    K1, K2, _, _, _ = sif_ctod_plate(mesh; tip=:right, absK1=true, method=:band)
    K1t, _, _, _, _ = sif_ctod_plate(mesh; tip=:right, absK1=true, method=:tip)
    Fd = (K1, K1t, abs(K2)) ./ den
    return Fd
end

@printf("  %s\n", "β=0  Fi=Ki/(Mo√a).  Kirchhoff inf. F=1 (Sih 1977).  Reissner t/a=2 Φ=0.82 (Sih/Dolbow).")
@printf("  %8s %9s %9s %9s %9s %9s %9s\n",
    "h", "K band", "K tip", "R Dual", "R XBEM", "Sih K", "Sih R")
flush(stdout)

for h in (0.05, 0.5)
    kprops = ThinPlateProps(; E=E, ν=ν, h=h)
    FdK = run_kirchhoff(kprops; tag="iso$h")
    FdR, FxR = run_reissner(h)
    @printf("  %8.3f %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f\n",
        h, FdK[1], FdK[2], FdR[1], FxR[1], 1.0, 0.82)
    flush(stdout)
end

println()
D22 = E * (0.5)^3 / (12 * (1 - ν^2))
props_a = aniso_thin_plate_props(; D11=2 * D22, D22=D22, D12=ν * D22,
    D66=(1 - ν) * D22 / 2 * 1.2, h=0.5)
println("  anisotropic Kirchhoff  D11/D22=2  h=0.5")
@printf("  %8s %10s %10s\n", "β°", "D band", "D tip")
flush(stdout)
for βdeg in (0, 45)
    β = deg2rad(βdeg)
    Fd = run_kirchhoff(props_a; α=β, tag="aniso")
    @printf("  %8.0f %10.4f %10.4f\n", βdeg, Fd[1], Fd[2])
    flush(stdout)
end
println("  K band = median K1(ρ) on 0.3a–0.85a.  K tip = two-point √r (Dirgantara).")
println("  R Dual/XBEM = Reissner.  Sih K=1, Sih R=0.82.")
println("Done.")
