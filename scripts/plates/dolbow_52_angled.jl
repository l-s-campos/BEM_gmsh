# Dolbow–Moës–Belytschko (2000) §5.2 — angled centre crack, far-field Mo.
# Geometry (Fig. 8 / §5.1): 2a=1, W_full=20a=10, t/a=2, E=200 GPa, ν=0.3.
# Sih (1977) infinite-plate (Dolbow eq. 65), t/a=2:
#   K1 = Φ Mo √a cos²β,  K2 = Ψ Mo √a cosβ sinβ,
#   K3 = −√10/((1+ν)t) Ω Mo √a cosβ sinβ
#   Φ≈0.82, Ψ≈0.68, Ω≈0.06.
# Dual CTOD is Dirgantara K1b=√π K1_dolbow, so Fi = Ki_dolbow/(Mo√a)
# = K1b/(Mo√(πa)). XBEM extra DOFs are already Dolbow K.
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf
using BEM.Plate

a = 0.5
W = 10 * a            # half-width 5 → full width 10 = 20a
H = W
h = 2 * a             # t/a = 2
E, ν = 2.0e5, 0.3
Mo = 1.0
props = FSDTProps(; E=E, ν=ν, h=h, q_c=0.0)
ndiv_b = ndiv_h = 11
ndiv_c = 13
Φ, Ψ, Ω = 0.82, 0.68, 0.06
sden = sqrt(10) / ((1 + ν) * h)

println("="^78)
println(" Dolbow 5.2  angled centre crack   t/a=2  W=20a  Dual BEM vs XBEM vs Sih")
println("="^78)
@printf("  plate [-%g,%g]²  a=%g  h=%g  mesh %d/edge  %d/face\n",
    W, W, a, h, ndiv_b - 1, ndiv_c - 1)
@printf("  %6s %10s %10s %10s %10s %10s %10s %10s %10s %10s\n",
    "β°", "D F1", "X F1", "Sih F1", "D F2", "X F2", "Sih F2", "D F3", "X F3", "Sih F3")
flush(stdout)

for βdeg in (0, 15, 30, 45, 60, 75, 90)
    β = deg2rad(βdeg)
    mesh = build_rect_fsdt_crack(; W=W, H=H, a=a, α=β, props=props, Mo=Mo,
        ndiv_b=ndiv_b, ndiv_h=ndiv_h, ndiv_crack=ndiv_c,
        nome="dolbow52_$(βdeg)")
    assemble_fsdt_dual!(mesh; npg=10, nsub=8)
    solve_fsdt!(mesh)
    K1b, K2b, K3b, _, _, _ = sif_ctod_fsdt(mesh; tip=:right, absK1=false)
    den = Mo * sqrt(π * a)
    Fd1, Fd2, Fd3 = K1b / den, K2b / den, K3b / den   # Dolbow Fi = K_dirg/√π /(Mo√a)

    _, K1, K2, K3, tips, _ = solve_fsdt_xbem!(mesh; n_enr=3, npg=10, nsub=8, n_v=3)
    ir = argmax(p[1] + 1e-9 * p[2] for p in tips)
    denx = Mo * sqrt(a)
    Fx1, Fx2, Fx3 = K1[ir] / denx, K2[ir] / denx, K3[ir] / denx

    Fs1 = Φ * cos(β)^2
    Fs2 = Ψ * cos(β) * sin(β)
    Fs3 = -sden * Ω * cos(β) * sin(β)
    @printf("  %6.0f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f\n",
        βdeg, Fd1, Fx1, Fs1, Fd2, Fx2, Fs2, Fd3, Fx3, Fs3)
    flush(stdout)
end
println("  Fi = Ki_dolbow / (Mo √a).  Dual uses Dirgantara/√π; XBEM extra DOFs.")
println("  Sih (65) at t/a=2: Φ=0.82 Ψ=0.68 Ω=0.06.")
println("Done.")
