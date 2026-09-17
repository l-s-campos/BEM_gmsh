# Useche 8.3.1 — unsymmetric FSDT Hsu–Hwu 5×5 BEM + DIBEM.
# Two-ply [0/90] SS square, uniform q. Gold: 5-DOF Navier with B ≠ 0.
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf, StaticArrays
using BEM.Plate

println("="^72)
println(" Useche 8.3.1  unsymmetric FSDT  [0/90]  Hsu–Hwu 5-DOF + DIBEM")
println("="^72)

plies = [(4e6, 2e6, 0.25, 1e6, θ, 0.05) for θ in (0.0, 90.0)]
props = laminate_unsym_props(plies; Ks=5 / 6, G13=1e6, G23=5e5, q_c=1.0, ρ=1.0, nθ=8)
@printf("  h=%.2f  B11=%.4e  B22=%.4e  D11=%.4f  A11=%.4e\n",
    props.h, props.B[1, 1], props.B[2, 2], props.D[1, 1], props.A[1, 1])
wN = navier_w_ss_unsym(0.5, 0.5, props; a=1.0, q=1.0)
@printf("  Navier 5-DOF w_c = %.6e\n", wN)

n_el, n_int = 3, 9
mesh = build_square_fsdt(; a=1.0, n_el=n_el, bc="SSSS", props=props, n_internal=n_int)
@printf("  mesh ndofn=%d  n=%d  ni=%d  ndof=%d\n",
    mesh.ndofn, length(mesh.nodes), length(mesh.internal), size(mesh.H, 1))
assemble_fsdt!(mesh; npg=6, nsub=4)
dibem_fsdt!(mesh; npg=6)
@printf("  ||H||=%.3e  ||G||=%.3e  ||M||=%.3e  ||q||=%.3e\n",
    norm(mesh.H), norm(mesh.G), norm(mesh.M), norm(mesh.q))
solve_fsdt!(mesh)
wc = fsdt_w_int(mesh, 1)
@printf("  DIBEM 5-DOF w_c = %.6e  rel Navier %.2f %%\n",
    wc, 100 * abs(wc - wN) / abs(wN))
println("Done.")
