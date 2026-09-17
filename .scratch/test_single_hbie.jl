using LinearAlgebra, Printf, StaticArrays, BEM, BEM.Plate

plies = [(4e6, 2e6, 0.25, 1e6, θ, 0.05) for θ in (0.0, 90.0)]
props = laminate_unsym_props(plies; Ks=5 / 6, G13=1e6, G23=5e5, q_c=1.0, nθ=6)
wN = navier_w_ss_unsym(0.5, 0.5, props; a=1.0, q=1.0)
@printf("Navier w_c = %.6e\n", wN)

meshC = build_square_fsdt(; a=1.0, n_el=2, bc="SSSS", props=props, n_internal=1)
assemble_fsdt!(meshC; npg=4, nsub=4, singular=:guiggiani, ninterp=12)
dibem_fsdt!(meshC; npg=4)
solve_fsdt!(meshC)
wcC = fsdt_w_int(meshC, 1)
@printf("CBIE  w_c=%.6e  rel=%.2f%%\n", wcC, 100 * abs(wcC - wN) / abs(wN))

meshH = build_square_fsdt(; a=1.0, n_el=2, bc="SSSS", props=props, n_internal=1)
t0 = time()
assemble_unsym_fsdt_hbie!(meshH; npg=4, nsub=4, ninterp=12)
@printf("  eq_type all 3: %s  H size %s  q_bd || ||=%.3e  q_int || ||=%.3e\n",
    all(==(3), meshH.eq_type), size(meshH.H),
    norm(meshH.q[1:5*length(meshH.nodes)]),
    norm(meshH.q[5*length(meshH.nodes)+1:end]))
solve_fsdt!(meshH)
dt = time() - t0
wcH = fsdt_w_int(meshH, 1)
@printf("HBIE  w_c=%.6e  rel=%.2f%%  finite=%s  (%.1fs)\n",
    wcH, 100 * abs(wcH - wN) / abs(wN), isfinite(wcH), dt)
@printf("|CBIE-HBIE|/|wN|=%.2f%%\n", 100 * abs(wcC - wcH) / abs(wN))
