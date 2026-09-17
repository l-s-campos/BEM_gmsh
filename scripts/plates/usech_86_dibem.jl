# Useche 8.5 resultants + 8.6.2 impulsive SS [0/90]s Wang FSDT Houbolt+DIBEM.
# Figs 8.5–8.7: w(t) vs RBF, mesh, Δt. MATLAB: Dynamic_Composite_Plate/test/EjemploDin_10.m
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf, StaticArrays
using BEM.Plate

println("="^72)
println(" Useche 8.5 / 8.6.2  Wang FSDT resultants + Houbolt DIBEM")
println("="^72)

E1, E2 = 4e6, 2e6
G12, G13, G23 = 1e6, 1e6, 5e5
a, h, q, ρ = 1.0, 0.1, 1.0, 4000.0
ν_book, ν_m = 0.5, 0.25
plies(ν) = [(E1, E2, ν, G12, θ, h / 4) for θ in (0.0, 90.0, 90.0, 0.0)]

function period_T11(p; a, ρ)
    D11, D22, D12, D66 = p.D[1, 1], p.D[2, 2], p.D[1, 2], p.D[3, 3]
    A44, A55 = p.AT[1, 1], p.AT[2, 2]
    α = π / a
    K11 = D11 * α^2 + D66 * α^2 + A55
    K12 = (D12 + D66) * α * α
    K13 = A55 * α
    K22 = D66 * α^2 + D22 * α^2 + A44
    K23 = A44 * α
    K33 = A55 * α^2 + A44 * α^2
    KK = [K11 K12 K13; K12 K22 K23; K13 K23 K33]
    Kred = KK[3, 3] - dot(KK[1:2, 3], KK[1:2, 1:2] \ KK[1:2, 3])
    return 2π / sqrt(Kred / (ρ * p.h))
end

# ---------------------------------------------------------------------------
# 8.5  internal M, Q on SS square (static)
# ---------------------------------------------------------------------------
println("\n## 8.5  internal resultants  SS [0/90]s  a=1  h=0.1  q=1")
props = laminate_fsdt_props(plies(ν_m); Ks=5 / 6, G13=G13, G23=G23, q_c=q, ρ=ρ, nθ=8)
gold = navier_ss_fsdt_MQ(a / 2, a / 2, props; a=a, q=q)
@printf("  Navier  w=%.4e  Mx=%.4e  My=%.4e  Qx=%.4e\n", gold.w, gold.Mx, gold.My, gold.Qx)
mesh = build_square_fsdt(; a=a, n_el=4, bc="SSSS", props=props, n_internal=9)
assemble_fsdt!(mesh; npg=8, nsub=6)
dibem_fsdt!(mesh; npg=8)
solve_fsdt!(mesh)
nb = length(mesh.nodes)
ic = nb + 1
wc = fsdt_w_int(mesh, 1)
rq = fsdt_resultants(mesh)
@printf("  DIBEM   w=%.4e  (%.2f %% vs Navier)\n", wc, 100 * abs(wc - gold.w) / abs(gold.w))
@printf("  centre  Mx=%.4e  My=%.4e  Qx=%.4e  Qy=%.4e\n",
    rq.Mx[ic], rq.My[ic], rq.Qx[ic], rq.Qy[ic])
@printf("  rel Mx %.1f %%  My %.1f %%\n",
    100 * abs(rq.Mx[ic] - gold.Mx) / max(abs(gold.Mx), 1e-30),
    100 * abs(rq.My[ic] - gold.My) / max(abs(gold.My), 1e-30))

# ---------------------------------------------------------------------------
# 8.6.2  Houbolt  (Figs 8.5–8.7)
# ---------------------------------------------------------------------------
println("\n## 8.6.2  impulsive SS [0/90]s  q(t)=1  (Houbolt + DIBEM)")
for (tag, ν) in (("MATLAB ν=0.25", ν_m), ("book ν=0.5", ν_book))
    p = laminate_fsdt_props(plies(ν); Ks=5 / 6, G13=G13, G23=G23, q_c=q, ρ=ρ, nθ=8)
    wF = navier_w_ss_fsdt(a / 2, a / 2, p; a=a, q=q)
    T11 = period_T11(p; a=a, ρ=ρ)
    @printf("  %-16s  w_stat Navier=%.4e  T11=%.4f s\n", tag, wF, T11)
end

props2 = laminate_fsdt_props(plies(ν_m); Ks=5 / 6, G13=G13, G23=G23, q_c=q, ρ=ρ, nθ=8)
wF = navier_w_ss_fsdt(a / 2, a / 2, props2; a=a, q=q)
T11 = period_T11(props2; a=a, ρ=ρ)

function run_houbolt(; n_el, n_int=9, dt=5e-3, tmax=0.5, rbf=PHS(), mass=:raw)
    m = build_square_fsdt(; a=a, n_el=n_el, bc="SSSS", props=props2, n_internal=n_int)
    assemble_fsdt!(m; npg=8, nsub=6)
    dibem_fsdt!(m; npg=8, rbf=rbf)
    solve_fsdt!(m)
    wstat = fsdt_w_int(m, 1)
    res = solve_fsdt_houbolt!(m; dt=dt, tmax=tmax, mass=mass)
    imax = argmax(abs.(res.w_center))
    return (n_el=n_el, n_be=length(m.elements), wstat=wstat,
        peak=res.w_center[imax], tpeak=res.t[imax], t=res.t, w=res.w_center)
end

println("\n  Fig 8.5  RBF  (8 BE, n_el=2)")
@printf("  %-10s %12s %12s %10s %10s\n", "RBF", "w_stat", "peak", "t_peak", "peak/2stat")
for (name, rbf) in (("PHS3 r³", PHS(3; poly_deg=2)), ("PHS2 r²ln r", PHS(2; poly_deg=1)))
    r = run_houbolt(; n_el=2, rbf=rbf)
    @printf("  %-10s %12.4e %12.4e %10.3f %10.3f\n",
        name, r.wstat, r.peak, r.tpeak, abs(r.peak) / (2 * abs(r.wstat) + eps()))
end

println("\n  Fig 8.6  mesh  (PHS3, Δt=5e-3)")
@printf("  %6s %6s %12s %12s %10s %10s\n", "n_el", "n_BE", "w_stat", "peak", "t_peak", "rel Nav")
for nel in (2, 3, 4)
    r = run_houbolt(; n_el=nel)
    @printf("  %6d %6d %12.4e %12.4e %10.3f %10.2f %%\n",
        nel, r.n_be, r.wstat, r.peak, r.tpeak,
        100 * abs(r.wstat - wF) / abs(wF))
end

println("\n  Fig 8.7  Δt  (12 BE, PHS3)")
@printf("  %10s %12s %10s %10s\n", "Δt", "peak", "t_peak", "peak/2stat")
for dt in (1e-2, 5e-3, 2.5e-3)
    r = run_houbolt(; n_el=3, dt=dt, tmax=0.5)
    @printf("  %10.1e %12.4e %10.3f %10.3f\n",
        dt, r.peak, r.tpeak, abs(r.peak) / (2 * abs(r.wstat) + eps()))
end
@printf("  Navier static w_c=%.4e  T11=%.4f s  (first peak ~ T11/2=%.3f)\n",
    wF, T11, T11 / 2)
println("Done.")
