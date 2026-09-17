using LinearAlgebra, Printf, StaticArrays, BEM, BEM.Plate

plies = [(4e6, 2e6, 0.25, 1e6, θ, 0.05) for θ in (0.0, 90.0)]
props = laminate_unsym_props(plies; Ks=5 / 6, G13=1e6, G23=5e5, q_c=1.0, nθ=6)
wN = navier_w_ss_unsym(0.5, 0.5, props; a=1.0, q=1.0)

meshC = build_square_fsdt(; a=1.0, n_el=2, bc="SSSS", props=props, n_internal=1)
assemble_fsdt!(meshC; npg=4, nsub=4, singular=:guiggiani, ninterp=12)
dibem_fsdt!(meshC; npg=4)
solve_fsdt!(meshC)
n = length(meshC.nodes)
nb = 5n
u, t = meshC.u, meshC.t
@printf("CBIE w_c=%.6e  rel=%.2f%%\n", fsdt_w_int(meshC, 1),
    100 * abs(fsdt_w_int(meshC, 1) - wN) / abs(wN))

meshH = build_square_fsdt(; a=1.0, n_el=2, bc="SSSS", props=props, n_internal=1)
assemble_fsdt!(meshH; npg=4, nsub=4, singular=:guiggiani, bie=:hbie, ninterp=12,
    scale_hbie=false)
H, G, q = meshH.H, meshH.G, meshH.q
r = H * u - G * t[1:nb]
@printf("||H u - G t||=%.3e  ||q||=%.3e  ||r-q||/||q||=%.3e\n",
    norm(r), norm(q), norm(r - q) / max(norm(q), 1e-30))
@printf("  ||r_bd||=%.3e  ||q_bd||=%.3e  ||r_int||=%.3e  ||q_int||=%.3e\n",
    norm(r[1:nb]), norm(q[1:nb]), norm(r[nb+1:end]),
    length(q) > nb ? norm(q[nb+1:end]) : 0)
# if we SET q = residual of CBIE solution, HBIE must recover it
meshH.q .= r
solve_fsdt!(meshH)
@printf("HBIE with q:=H u_c - G t_c  w_c=%.6e  rel=%.2f%%\n",
    fsdt_w_int(meshH, 1), 100 * abs(fsdt_w_int(meshH, 1) - wN) / abs(wN))
