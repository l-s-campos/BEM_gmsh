# Useche 7.7.1 — SS cylindrical [90/0/0/90]s, Wang FSDT + anisotropic membrane.
# MATLAB/book: 16 BE + 72 RIM. DIBEM replaces RIM (same coupling as Ch.9).
# κ11=1/50, κ22=0, a/h=10, q=1. Gold: 5-DOF Navier (Reddy), not Donnell-on-w.
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf, StaticArrays
using BEM.Plate

function navier_fsdt_kmem(p, kmem; a, q, nterms=40)
    D11, D22, D12, D66 = p.D[1, 1], p.D[2, 2], p.D[1, 2], p.D[3, 3]
    A44, A55 = p.AT[1, 1], p.AT[2, 2]
    w = 0.0
    @inbounds for m in 1:2:nterms, n in 1:2:nterms
        α = m * π / a
        β = n * π / a
        K11 = D11 * α^2 + D66 * β^2 + A55
        K12 = (D12 + D66) * α * β
        K13 = A55 * α
        K22 = D66 * α^2 + D22 * β^2 + A44
        K23 = A44 * β
        K33 = A55 * α^2 + A44 * β^2 + kmem
        Δ = [K11 K12 K13; K12 K22 K23; K13 K23 K33] \ [0.0, 0.0, 16q / (π^2 * m * n)]
        w += Δ[3] * sin(α * a / 2) * sin(β * a / 2)
    end
    return w
end

println("="^72)
println(" Useche 7.7.1  SS cylindrical [90/0/0/90]s  — coupled DIBEM")
println("="^72)

E2, ν12 = 1.0e9, 0.25
E1 = 25 * E2
G12 = 0.5 * E2
G13, G23 = G12, 0.2 * E2
a, h, R, q = 10.0, 1.0, 50.0, 1.0
κ1, κ2 = 1 / R, 0.0
angs = [90.0, 0.0, 0.0, 90.0, 90.0, 0.0, 0.0, 90.0]
plies = [(E1, E2, ν12, G12, θ, h / 8) for θ in angs]
props = laminate_fsdt_props(plies; Ks=5 / 6, G13=G13, G23=G23, q_c=q, ρ=1.0)
A, _, D, AT, _ = BEM.Plate._laminate_ABD_AT(plies; Ks=5 / 6, G13=G13, G23=G23)
As = @SMatrix [AT[2, 2] AT[1, 2]; AT[1, 2] AT[1, 1]]
kmem = A[1, 1] * κ1^2
@printf("  a=%.1f  h=%.2f  R=%.1f  a/h=%.0f  a/R=%.2f  κ22=0\n", a, h, R, a / h, a / R)
@printf("  A11=%.4e  A22=%.4e  A12=%.4e  D11=%.4e  D22=%.4e  kmem=%.4e\n",
    A[1, 1], A[2, 2], A[1, 2], D[1, 1], D[2, 2], kmem)

ctr = navier_ss_laminate_shell(a / 2, a / 2; a=a, q=q, κ1=κ1, κ2=κ2, A=A, D=D, As=As)
flat = navier_ss_laminate_shell(a / 2, a / 2; a=a, q=q, κ1=0, κ2=0, A=A, D=D, As=As)
w3 = navier_fsdt_kmem(props, kmem; a=a, q=q)
@printf("  series 5-DOF Navier  flat w_c=%.6e\n", flat.w)
@printf("  series 5-DOF Navier  cyl  w_c=%.6e  flat/cyl=%.3f\n", ctr.w, flat.w / ctr.w)
@printf("  series 3-DOF+A11/R²      w_c=%.6e  (vs 5-DOF %.2f %%)\n",
    w3, 100 * abs(w3 - ctr.w) / abs(ctr.w))
@printf("  Nx=%.4e  Ny=%.4e  Mx=%.4e  My=%.4e\n", ctr.Nx, ctr.Ny, ctr.Mx, ctr.My)

n_el, n_int = 4, 81
println("\n  coupled Wang+membrane DIBEM  n_el=$n_el (16 BE, book)  n_int=$n_int (~book 72 RIM)")
mesh_f = build_square_fsdt(; a=a, n_el=n_el, bc="SSSS", props=props, n_internal=n_int)
assemble_fsdt!(mesh_f; npg=8, nsub=6)
dibem_fsdt!(mesh_f)
solve_fsdt!(mesh_f)
w_flat = fsdt_w_int(mesh_f, 1)
@printf("  DIBEM FSDT flat w_c=%.6e  rel 5-DOF flat %.2f %%\n",
    w_flat, 100 * abs(w_flat - flat.w) / abs(flat.w))

mesh = build_square_fsdt(; a=a, n_el=n_el, bc="SSSS", props=props, n_internal=n_int)
shell = LaminatedShell(mesh, A, κ1, κ2; mem_bc=:navier_ss)
assemble_laminated_shell!(shell; npg=8, nsub=6)
solve_laminated_shell!(shell)
w_cyl = fsdt_w_int(mesh, 1)
res = shell_resultants(shell)
nb = length(mesh.nodes)
ic = nb + 1
@printf("  DIBEM coupled 5-DOF w_c=%.6e  rel 5-DOF cyl %.2f %%  rel 3-DOF+kmem %.2f %%\n",
    w_cyl, 100 * abs(w_cyl - ctr.w) / abs(ctr.w), 100 * abs(w_cyl - w3) / abs(w3))
@printf("  BEM flat/cyl=%.3f  series 5-DOF flat/cyl=%.3f\n", w_flat / w_cyl, flat.w / ctr.w)
@printf("  BEM centre Nx=%.4e  Ny=%.4e  Mx=%.4e  My=%.4e\n",
    res.Nx[ic], res.Ny[ic], res.Mx[ic], res.My[ic])

# Figs 7.5–7.6: N, M along y = b/2 (internals, x2 axis through centre)
println("\n  along y=a/2 internals (book Figs 7.5–7.6 vs 5-DOF Navier)")
@printf("  %8s %12s %12s %12s %12s %12s %12s\n",
    "x", "Nx BEM", "Nx ser", "Mx BEM", "Mx ser", "w BEM", "w ser")
tol = 0.6 * a / sqrt(n_int)
keep = findall(i -> i > nb && abs(res.pts[i][2] - a / 2) < tol, eachindex(res.pts))
ord = sort(keep; by=i -> res.pts[i][1])
for i in ord
    p = res.pts[i]
    g = navier_ss_laminate_shell(p[1], p[2]; a=a, q=q, κ1=κ1, κ2=κ2, A=A, D=D, As=As)
    @printf("  %8.3f %12.4e %12.4e %12.4e %12.4e %12.4e %12.4e\n",
        p[1], res.Nx[i], g.Nx, res.Mx[i], g.Mx, res.w[i], g.w)
end
println("Done.")
