# FSDT (Vander Weeën) vs Kirchhoff Navier and MATLAB thick-plate examples
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf
using BEM.Plate

println("="^64)
println(" FSDT Reissner BEM vs MATLAB / Navier")
println("="^64)

# --- static: MATLAB prueba03 geometry (thin a/h=100) ---
println("\n## Static SS square  (MATLAB Static_Thick_Plate/prueba03.m)")
E, ν, h, a, q0 = 200e9, 0.3, 0.02, 2.0, 1.0
Dk = bending_stiffness(ThinPlateProps(; E=E, ν=ν, h=h))
wK = analytical_wmax_ss_square(; a=a, q=q0, D=Dk)
κGh = shear_stiffness(FSDTProps(; E=E, ν=ν, h=h))
wF = navier_w_ss_fsdt(a / 2, a / 2; a=a, q=q0, D=Dk, ν=ν, κGh=κGh)
props = FSDTProps(; E=E, ν=ν, h=h, q_c=q0, ρ=1.0)
mesh = build_square_fsdt(; a=a, n_el=4, bc="SSSS", props=props, n_internal=9)
assemble_fsdt!(mesh; npg=8, nsub=6)
dibem_fsdt!(mesh; method=:drm)
solve_fsdt!(mesh)
wc_drm = fsdt_w_int(mesh, 1)
dibem_fsdt!(mesh; method=:dibem)
solve_fsdt!(mesh)
wc = fsdt_w_int(mesh, 1)
@printf("  Kirchhoff Navier w_c = %.6e\n", wK)
@printf("  FSDT Navier      w_c = %.6e  (rel Kirchhoff %.3f %%)\n",
    wF, 100 * abs(wF - wK) / abs(wK))
@printf("  BEM DRM  (MATLAB uqchp)  %.6e  rel Navier %.2f %%\n",
    wc_drm, 100 * abs(wc_drm - wF) / abs(wF))
@printf("  BEM DIBEM (Maxima RIM)   %.6e  rel Navier %.2f %%\n",
    wc, 100 * abs(wc - wF) / abs(wF))

# --- static moderate thickness a/h=10 ---
println("\n## Static SS square  a/h=10")
h10 = 0.1
a10 = 1.0
E10, q10 = 1e5, 1.0
Dk10 = bending_stiffness(ThinPlateProps(; E=E10, ν=ν, h=h10))
wK10 = analytical_wmax_ss_square(; a=a10, q=q10, D=Dk10)
κ10 = shear_stiffness(FSDTProps(; E=E10, ν=ν, h=h10))
wF10 = navier_w_ss_fsdt(a10 / 2, a10 / 2; a=a10, q=q10, D=Dk10, ν=ν, κGh=κ10)
p10 = FSDTProps(; E=E10, ν=ν, h=h10, q_c=q10, ρ=1.0)
m10 = build_square_fsdt(; a=a10, n_el=4, bc="SSSS", props=p10, n_internal=9)
assemble_fsdt!(m10; npg=8, nsub=6)
dibem_fsdt!(m10; method=:drm); solve_fsdt!(m10)
wc10d = fsdt_w_int(m10, 1)
dibem_fsdt!(m10; method=:dibem); solve_fsdt!(m10)
wc10 = fsdt_w_int(m10, 1)
@printf("  Kirchhoff Navier %.6e\n", wK10)
@printf("  FSDT Navier      %.6e  (%.1f %% above Kirchhoff)\n",
    wF10, 100 * (abs(wF10) - abs(wK10)) / abs(wK10))
@printf("  BEM DRM          %.6e  rel Navier %.2f %%\n",
    wc10d, 100 * abs(wc10d - wF10) / abs(wF10))
@printf("  BEM DIBEM        %.6e  rel Navier %.2f %%\n",
    wc10, 100 * abs(wc10 - wF10) / abs(wF10))

# --- dynamics: MATLAB Dynamic_Thick_Plate/example01.m ---
println("\n## Houbolt  MATLAB example01  SS [-1,1]²  q=1e3  ρ=0.7853")
# scale to [0,a] with a=2
Ed, hd, ad, qd, ρd = 200e3, 0.1, 2.0, 1e3, 0.7853
pd = FSDTProps(; E=Ed, ν=0.3, h=hd, q_c=qd, ρ=ρd)
md = build_square_fsdt(; a=ad, n_el=4, bc="SSSS", props=pd, n_internal=9)
assemble_fsdt!(md; npg=8, nsub=6)
dibem_fsdt!(md; method=:drm)
solve_fsdt!(md)
wstat = fsdt_w_int(md, 1)
res = solve_fsdt_houbolt!(md; dt=5e-3, tmax=0.15)
imax = argmax(abs.(res.w_center))
@printf("  DRM    static=%.6e  Houbolt peak=%.6e t=%.3f  ratio=%.3f\n",
    wstat, abs(res.w_center[imax]), res.t[imax],
    abs(res.w_center[imax]) / (2 * abs(wstat) + eps()))
md2 = build_square_fsdt(; a=ad, n_el=4, bc="SSSS", props=pd, n_internal=9)
assemble_fsdt!(md2; npg=8, nsub=6)
dibem_fsdt!(md2; method=:dibem)
solve_fsdt!(md2)
wstat2 = fsdt_w_int(md2, 1)
res2 = solve_fsdt_houbolt!(md2; dt=5e-3, tmax=0.15)
imax2 = argmax(abs.(res2.w_center))
@printf("  DIBEM  static=%.6e  Houbolt peak=%.6e t=%.3f  ratio=%.3f\n",
    wstat2, abs(res2.w_center[imax2]), res2.t[imax2],
    abs(res2.w_center[imax2]) / (2 * abs(wstat2) + eps()))
println("  MATLAB step-load peak ~ 2×static. Internals = 3×3 cell centroids.")

# --- Wang laminate kernels vs MATLAB KernelP (isotropic D, AT) ---
println("\n## Wang KernelP vs Octave (isotropic equivalent D, AT)")
pw = FSDTProps(; E=1e5, ν=0.3, h=0.05)
lw = LaminateFSDTProps(pw; nθ=12)
pg, pf, nh = SVector(0.4, 0.2), SVector(0.0, 0.0), SVector(1.0, 0.0)
Uw, Pw, _ = wang_kernels(pg, pf, nh, lw.D, lw.AT; nθ=12)
UM = [-0.020706 -0.027410 -0.013844; -0.027410 0.020409 -0.006922; 0.013844 0.006922 -0.006937]
PM = [-0.2518 0.0317 0.4777; -0.0161 -0.0665 0.6367; -0.0169 -0.0223 -0.3183]
@printf("  max |U-Umatlab|/|U| = %.2e   |P-Pmatlab|/|P| = %.2e\n",
    maximum(abs.(Uw .- UM)) / maximum(abs.(UM)),
    maximum(abs.(Pw .- PM)) / maximum(abs.(PM)))

println("\n## ABD ContsLam  MATLAB Material.m [0/90/90/0]")
plies = [(4e6, 2e6, 0.25, 1e6, θ, 0.025) for θ in (0.0, 90.0, 90.0, 0.0)]
lam = laminate_fsdt_props(plies; Ks=5 / 6, G13=1e6, G23=5e5, q_c=1.0)
@printf("  D11=%.4f  D22=%.4f  D12=%.4f  D66=%.4f\n",
    lam.D[1, 1], lam.D[2, 2], lam.D[1, 2], lam.D[3, 3])
@printf("  AT A44=%.1f  A55=%.1f  (Octave 62500)\n", lam.AT[1, 1], lam.AT[2, 2])
wF = navier_w_ss_fsdt(0.5, 0.5, lam; a=1.0, q=1.0)
@printf("  FSDT Navier w_c [0/90]s a=1 q=1  = %.6e\n", wF)
println("\n## Wang laminate DIBEM (Maxima Fρ) vs Navier  [0/90]s")
ml = build_square_fsdt(; a=1.0, n_el=3, bc="SSSS", props=lam, n_internal=1)
assemble_fsdt!(ml; npg=8, nsub=6)
dibem_fsdt!(ml; method=:dibem, npg=8)
solve_fsdt!(ml)
wcl = fsdt_w_int(ml, 1)
@printf("  DIBEM BEM w_c = %.6e  rel Navier %.2f %%\n",
    wcl, 100 * abs(wcl - wF) / abs(wF))
println("  MATLAB Dynamic_Composite_Plate is Wang + Gauss RIM of U* (int_carga).")
println("Done.")
