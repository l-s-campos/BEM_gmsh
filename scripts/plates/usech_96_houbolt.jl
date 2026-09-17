# Useche Ch.9 dynamics — laminated shallow shell Houbolt + DIBEM mass.
# Geometry: 9.6.1 SS spherical [0/90]s. Step q(t)=1. Gold: 5-DOF Navier T11, 2×static peak.
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf, StaticArrays
using BEM.Plate

println("="^72)
println(" Useche Ch.9  laminated shell dynamics  (Houbolt + DIBEM)")
println("="^72)

E1, E2, ν12 = 25.0, 1.0, 0.25
G12 = 0.5 * E2
a, h, R, q, ρ = 1.0, 0.01, 1.0, 1.0, 1.0
κ = 1 / R
plies = [(E1, E2, ν12, G12, θ, h / 4) for θ in (0.0, 90.0, 90.0, 0.0)]
props = laminate_fsdt_props(plies; Ks=5 / 6, G13=G12, G23=0.2 * E2, q_c=q, ρ=ρ)
A, _, D, AT, _ = BEM.Plate._laminate_ABD_AT(plies; Ks=5 / 6, G13=G12, G23=0.2 * E2)
As = @SMatrix [AT[2, 2] AT[1, 2]; AT[1, 2] AT[1, 1]]
ctr = navier_ss_laminate_shell(a / 2, a / 2; a=a, q=q, κ1=κ, κ2=κ, A=A, D=D, As=As)
T11 = navier_ss_laminate_T11(; a=a, κ1=κ, κ2=κ, A=A, D=D, As=As, ρ=ρ, h=h)
@printf("  9.6.1 SS sphere  a=%.2f  h=%.3f  R=%.2f  ρ=%.2f\n", a, h, R, ρ)
@printf("  Navier 5-DOF w_c=%.6e  T11=%.4f s  T11/2=%.4f\n", ctr.w, T11, T11 / 2)

# 81 DIBEM centres match static 9.6.1 (~8% vs Navier) but the PHS mass
# is indefinite (hundreds of λ<0) and Houbolt diverges. 25 centres is
# the stable dynamics cloud (test/plates.jl); period tracks this mesh's
# static stiffness, peak ≈ 2×static.
n_el, n_int = 4, 25
mesh = build_square_fsdt(; a=a, n_el=n_el, bc="SSSS", props=props, n_internal=n_int)
shell = LaminatedShell(mesh, A, κ, κ; mem_bc=:navier_ss)
assemble_laminated_shell!(shell; npg=8, nsub=6)
solve_laminated_shell!(shell)
wstat = fsdt_w_int(mesh, 1)
@printf("  DIBEM static w_c=%.6e  rel Navier %.2f %%  (n_el=%d n_int=%d)\n",
    wstat, 100 * abs(wstat - ctr.w) / abs(ctr.w), n_el, n_int)

# T11/80 lets high-frequency DIBEM modes grow; T11/30 matches the test
# (Houbolt damps them). Peak/2stat ≈ 1, t_peak a bit early (mesh is stiff).
dt = T11 / 30
tmax = 0.65 * T11
println("\n  Houbolt step q  dt=$(round(dt; sigdigits=3))  tmax=$(round(tmax; sigdigits=3))  mass=:raw")
res = solve_laminated_shell_houbolt!(shell; dt=dt, tmax=tmax, mass=:raw)
imax = argmax(abs.(res.w_center))
T2mesh = T11 / 2 * sqrt(abs(wstat) / abs(ctr.w))
@printf("  peak=%.6e  t=%.4f  peak/2stat=%.3f  (Navier T11/2=%.4f, mesh T/2~%.4f)\n",
    res.w_center[imax], res.t[imax],
    abs(res.w_center[imax]) / (2 * abs(wstat) + eps()), T11 / 2, T2mesh)
println("Done.")
