# Useche 8.3.1 — same [0/90] SS square: Hsu–Hwu CBIE vs a single HBIE.
# HBIE on every boundary node; interior w by Somigliana. Self-element
# Guiggiani: CBIE interpolant (0,-1); HBIE Richardson (-1,-2).
# S from Maxima F' (θ-HFP). Gold: 5-DOF Navier, B ≠ 0.
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf, StaticArrays
using BEM.Plate

println("="^72)
println(" Useche 8.3.1  unsymmetric FSDT  CBIE vs HBIE  (Guiggiani self)")
println("="^72)

plies = [(4e6, 2e6, 0.25, 1e6, θ, 0.05) for θ in (0.0, 90.0)]
props = laminate_unsym_props(plies; Ks=5 / 6, G13=1e6, G23=5e5, q_c=1.0, ρ=1.0, nθ=8)
wN = navier_w_ss_unsym(0.5, 0.5, props; a=1.0, q=1.0)
@printf("  h=%.2f  B11=%.4e  D11=%.4f\n", props.h, props.B[1, 1], props.D[1, 1])
@printf("  Navier 5-DOF w_c = %.6e\n", wN)

n_el, n_int = 3, 9
npg, nsub, ninterp = 6, 4, 16

function _run(bie, singular)
    mesh = build_square_fsdt(; a=1.0, n_el=n_el, bc="SSSS", props=props,
        n_internal=n_int)
    t0 = time()
    assemble_fsdt!(mesh; npg=npg, nsub=nsub, singular=singular, bie=bie,
        ninterp=ninterp)
    bie === :cbie && dibem_fsdt!(mesh; npg=npg)
    solve_fsdt!(mesh)
    dt = time() - t0
    wc = fsdt_w_int(mesh, 1)
    return mesh, wc, dt
end

mesh_c, wc_c, dt_c = _run(:cbie, :guiggiani)
@printf("  CBIE  Guiggiani  w_c = %.6e  rel Navier %6.2f %%  (%.1f s)\n",
    wc_c, 100 * abs(wc_c - wN) / abs(wN), dt_c)
@printf("    ||H||=%.3e  ||G||=%.3e  ||q||=%.3e\n",
    norm(mesh_c.H), norm(mesh_c.G), norm(mesh_c.q))

mesh_h, wc_h, dt_h = _run(:hbie, :guiggiani)
@printf("  HBIE  Guiggiani  w_c = %.6e  rel Navier %6.2f %%  (%.1f s)\n",
    wc_h, 100 * abs(wc_h - wN) / abs(wN), dt_h)
@printf("    ||H||=%.3e  ||G||=%.3e  ||q||=%.3e\n",
    norm(mesh_h.H), norm(mesh_h.G), norm(mesh_h.q))

nb = 5 * length(mesh_c.nodes)
du = norm(mesh_c.u[1:nb] .- mesh_h.u[1:nb]) / max(norm(mesh_c.u[1:nb]), 1e-30)
dtn = norm(mesh_c.t .- mesh_h.t) / max(norm(mesh_c.t), 1e-30)
@printf("  |w_c CBIE − HBIE| / |Navier| = %.2f %%\n",
    100 * abs(wc_c - wc_h) / abs(wN))
@printf("  boundary ||Δu||/||u|| = %.3e  ||Δt||/||t|| = %.3e\n", du, dtn)
println("Done.")
