using BEM, BEM.Plate, LinearAlgebra, Printf, StaticArrays

E, ν, h = 1e5, 0.3, 0.05
D = E * h^3 / (12 * (1 - ν^2))
Gsh = E / (2 * (1 + ν))
A11 = E * h / (1 - ν^2)
A = @SMatrix [A11 ν*A11 0; ν*A11 A11 0; 0 0 Gsh*h]
Dmat = @SMatrix [D ν*D 0; ν*D D 0; 0 0 (1-ν)*D/2]
AT = @SMatrix [5/6*Gsh*h 0; 0 5/6*Gsh*h]
pH = UnsymFSDTProps(A, 1e-8*Dmat, Dmat, AT; h=h, nθ=12)
pf = SVector(0.0, 0.0)
nh = SVector(1.0, 0.0)

println("=== radius sweep (iso tiny-B, β=-3/2) ===")
@printf("%8s %12s %12s %10s %10s %10s %10s\n",
    "R", "UwwH", "UwwW", "Uww/W", "Pww/W", "Uββ/Wψψ", "U11")
for (x, y) in ((0.1, 0.0), (0.2, 0.1), (0.4, 0.2), (0.8, 0.4), (1.2, 0.3))
    pg = SVector(x, y)
    UW, PW, _ = wang_kernels(pg, pf, nh, Dmat, AT; nθ=12)
    UH, PH = unsym_fsdt_kernels(pg, pf, nh, pH)
    @printf("%8.3f %12.4e %12.4e %10.4f %10.4f %10.4f %10.3e\n",
        hypot(x, y), UH[5, 5], UW[3, 3], UH[5, 5]/UW[3, 3],
        PH[5, 5]/PW[3, 3], UH[3, 3]/UW[1, 1], UH[1, 1])
end

pg = SVector(0.4, 0.2)
UW, PW, _ = wang_kernels(pg, pf, nh, Dmat, AT; nθ=12)
UH, PH = unsym_fsdt_kernels(pg, pf, nh, pH)
println("\nWang U:"); display(UW)
println("Hsu U (β,w):"); display(UH[3:5, 3:5])
println("Wang P:"); display(PW)
println("Hsu P (β,w):"); display(PH[3:5, 3:5])

println("\n=== [0/90] 5-DOF BEM vs Navier ===")
plies = [(4e6, 2e6, 0.25, 1e6, θ, 0.05) for θ in (0.0, 90.0)]
props = laminate_unsym_props(plies; Ks=5 / 6, G13=1e6, G23=5e5, q_c=1.0, ρ=1.0, nθ=8)
wN = navier_w_ss_unsym(0.5, 0.5, props; a=1.0, q=1.0)
@printf("Navier w_c=%.6e\n", wN)
mesh = build_square_fsdt(; a=1.0, n_el=3, bc="SSSS", props=props, n_internal=9)
assemble_fsdt!(mesh; npg=6, nsub=4)
dibem_fsdt!(mesh; npg=6)
@printf("||H||=%.3e ||G||=%.3e ||q||=%.3e condH=%.3e\n",
    norm(mesh.H), norm(mesh.G), norm(mesh.q), cond(mesh.H))
solve_fsdt!(mesh)
wc = fsdt_w_int(mesh, 1)
@printf("BEM w_c=%.6e  rel Navier %.2f %%  sign %s\n",
    wc, 100 * abs(wc - wN) / abs(wN), wc * wN > 0 ? "same" : "FLIP")
