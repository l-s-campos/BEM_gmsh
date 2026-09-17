# CBIE solve, then at an interior point: Hsu–Hwu EABE 156 complete-solution
# traction (differentiate Somigliana, then constitutive T*) vs FD of u.
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf, StaticArrays
using BEM.Plate

println("="^72)
println(" Unsym FSDT: CBIE solve → Hsu EABE 156 interior T* vs FD")
println("="^72)

plies = [(4e6, 2e6, 0.25, 1e6, θ, 0.05) for θ in (0.0, 90.0)]
props = laminate_unsym_props(plies; Ks=5 / 6, G13=1e6, G23=5e5, q_c=1.0, nθ=8)
a = 1.0
wN = navier_w_ss_unsym(0.5, 0.5, props; a=a, q=1.0)
mesh = build_square_fsdt(; a=a, n_el=3, bc="SSSS", props=props, n_internal=1)
assemble_fsdt!(mesh; npg=6, nsub=4, singular=:guiggiani, ninterp=12)
dibem_fsdt!(mesh; npg=6)
solve_fsdt!(mesh)
wc = fsdt_w_int(mesh, 1)
@printf("  Navier w_c = %.6e   CBIE w_c = %.6e  rel %.2f %%\n",
    wN, wc, 100 * abs(wc - wN) / abs(wN))

pf = SVector(0.5, 0.5)
nx, ny = SVector(1.0, 0.0), SVector(0.0, 1.0)
npg, nsub, hfd = 6, 4, 1e-3

tx_h = unsym_interior_t(mesh, pf, nx; npg=npg, nsub=nsub)
ty_h = unsym_interior_t(mesh, pf, ny; npg=npg, nsub=nsub)
tx_fd, aux = unsym_interior_t_fd(mesh, pf, nx; h=hfd, npg=npg, nsub=nsub)
ty_fd, _ = unsym_interior_t_fd(mesh, pf, ny; h=hfd, npg=npg, nsub=nsub)
# n=ex: t[3]=Mx; n=ey: t[4]=My
Mx_h, My_h = tx_h[3], ty_h[4]
Mx_fd, My_fd = tx_fd[3], ty_fd[4]
M = aux.M
@printf("\n  interior (%.2f, %.2f)  FD h=%.1e\n", pf[1], pf[2], hfd)
@printf("  constitutive FD   Mx=%.6e  My=%.6e  Mxy=%.6e\n", M[1], M[2], M[3])
@printf("  FD traction n=ex  Mx=%.6e   n=ey My=%.6e\n", Mx_fd, My_fd)
@printf("  Hsu T*      n=ex  Mx=%.6e   vs FD rel %.2f %%\n",
    Mx_h, 100 * abs(Mx_h - Mx_fd) / max(abs(Mx_fd), 1e-30))
@printf("  Hsu T*      n=ey  My=%.6e   vs FD rel %.2f %%\n",
    My_h, 100 * abs(My_h - My_fd) / max(abs(My_fd), 1e-30))
@printf("\n  full t (n=ex)\n")
@printf("    Hsu   N=%.4e %.4e  M=%.4e %.4e  Q=%.4e\n", tx_h...)
@printf("    FD    N=%.4e %.4e  M=%.4e %.4e  Q=%.4e\n", tx_fd...)
relt = maximum(abs.(tx_h .- tx_fd)) / max(maximum(abs, tx_fd), 1e-30)
@printf("    max|Δt|/max|t_FD| = %.3e\n", relt)
println("Done.")
