using LinearAlgebra, Printf, StaticArrays, BEM, BEM.Plate

plies = [(4e6, 2e6, 0.25, 1e6, θ, 0.05) for θ in (0.0, 90.0)]
props = laminate_unsym_props(plies; Ks=5 / 6, G13=1e6, G23=5e5, q_c=1.0, nθ=6)
wN = navier_w_ss_unsym(0.5, 0.5, props; a=1.0, q=1.0)
@printf("Navier w_c = %.6e\n", wN)

function run(bie, singular)
    mesh = build_square_fsdt(; a=1.0, n_el=2, bc="SSSS", props=props, n_internal=1)
    t0 = time()
    assemble_fsdt!(mesh; npg=4, nsub=4, singular=singular, bie=bie, ninterp=12)
    bie === :cbie && dibem_fsdt!(mesh; npg=4)
    solve_fsdt!(mesh)
    dt = time() - t0
    wc = fsdt_w_int(mesh, 1)
    @printf("  %s/%s  w_c=%.6e  rel=%.2f%%  ||H||=%.3e  ||q||=%.3e  (%.1fs)\n",
        bie, singular, wc, 100 * abs(wc - wN) / abs(wN),
        norm(mesh.H), norm(mesh.q), dt)
    return wc
end

wcC = run(:cbie, :guiggiani)
wcH = run(:hbie, :guiggiani)
@printf("  |CBIE-HBIE|/|wN|=%.2f%%\n", 100 * abs(wcC - wcH) / abs(wN))
