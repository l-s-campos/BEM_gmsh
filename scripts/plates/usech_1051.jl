# Useche 10.5.1 — Table 10.1  K1b / (Mo √(πa))
# Geometry: b/h=2, c/b=2, Mo=1, E=2.1e5, ν=0.3.
# Book mesh: 32 outer BE + 16 quadratic/face. Here: 16/edge (64 outer) + 32/face.
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf
using BEM.Plate

println("="^72)
println(" Useche 10.5.1  Table 10.1  centre crack, bending  (finer mesh)")
println("="^72)

b = 1.0
h = b / 2
c = 2 * b
E, ν = 2.1e5, 0.3
Mo = 1.0
props = FSDTProps(; E=E, ν=ν, h=h, q_c=0.0)
W, Ht = b, c
# transfinite nodes = n_el + 1  (quadratic 1-D after setOrder)
ndiv_b, ndiv_h, ndiv_c = 17, 17, 33

book = (
    (0.1, 0.993, 0.995),
    (0.2, 0.992, 0.990),
    (0.4, 0.845, 0.850),
    (0.6, 0.095, 0.100),
    (0.8, 0.134, 0.135),
)

@printf("  plate [-%.0f,%.0f]×[-%.0f,%.0f]  h=%.2f\n", W, W, Ht, Ht, h)
@printf("  mesh  %d/edge outer  %d/crack-face  (book 8/edge, 16/face)\n",
    ndiv_b - 1, ndiv_c - 1)
@printf("  %6s %12s %12s %12s %8s %8s\n",
    "a/b", "K1b F", "book BEM", "Dirg. [2]", "err book", "err ref")

for (ab, kb, kr) in book
    a = ab * b
    mesh = build_rect_fsdt_crack(; W=W, H=Ht, a=a, props=props, Mo=Mo,
        ndiv_b=ndiv_b, ndiv_h=ndiv_h, ndiv_crack=ndiv_c,
        nome="usech1051f_$(ab)")
    assemble_fsdt_dual!(mesh; npg=12, nsub=10)
    solve_fsdt!(mesh)
    K1b, K2b, K3b, rA, rB, Le = sif_ctod_fsdt(mesh; tip=:right)
    Fb = K1b / (Mo * sqrt(π * a))
    eb = 100 * abs(Fb - kb) / abs(kb)
    er = 100 * abs(Fb - kr) / abs(kr)
    @printf("  %6.1f %12.4f %12.3f %12.3f %7.2f%% %7.2f%%   (K2b=%.2e K3b=%.2e  n=%d  rA/Le=%.3f rB/Le=%.3f)\n",
        ab, Fb, kb, kr, eb, er, K2b, K3b, length(mesh.nodes), rA / Le, rB / Le)
    flush(stdout)
end
println("  a/b=0.6–0.8 book ~0.1 is likely OCR (infinite-plate F≈1).")
println("Done.")
