# Useche 10.5.1 centre crack: Julia Dual BEM vs MATLAB/Octave two-subregion.
# Plate [-1,1]×[-2,2], a=0.2, h=0.5, E=2.1e5, ν=0.3, Mo=1 on y=±2.
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf, DelimitedFiles
using BEM.Plate

W, Ht, a = 1.0, 2.0, 0.2
E, ν, h, Mo = 2.1e5, 0.3, 0.5, 1.0
props = FSDTProps(; E=E, ν=ν, h=h, q_c=0.0)
bookF = 0.992

println("="^72)
println(" Centre crack  Dual BEM (Julia) vs subregion (Octave)")
println("="^72)
@printf("  plate [-%.0f,%.0f]×[-%.0f,%.0f]  a=%.2f  h=%.2f  Mo=%.1f\n",
    W, W, Ht, Ht, a, h, Mo)
@printf("  book Dual F = K1b/(Mo √(πa)) ≈ %.3f\n", bookF)

mesh = build_rect_fsdt_crack(; W=W, H=Ht, a=a, props=props, Mo=Mo,
    ndiv_b=5, ndiv_h=5, ndiv_crack=7, nome="crack_dbem_cmp")
@printf("  Dual mesh  n_node=%d  eq2=%d  eq3=%d\n",
    length(mesh.nodes), count(==(2), mesh.eq_type), count(==(3), mesh.eq_type))
assemble_fsdt_dual!(mesh; npg=8, nsub=6)
solve_fsdt!(mesh)
K1b, K2b, K3b, rA, rB, Le = sif_ctod_fsdt(mesh; tip=:right)
F = K1b / (Mo * sqrt(π * a))
@printf("  Julia Dual  K1b=%.4e  F=%.4f  vs book %.1f %%\n",
    K1b, F, 100 * abs(F - bookF) / bookF)
@printf("  K2b=%.2e  K3b=%.2e  rA/Le=%.3f rB/Le=%.3f\n", K2b, K3b, rA / Le, rB / Le)

println("  Dual COD  (face A − face B)  along the crack:")
println("    x          Δψx         Δψy          Δw")
codJ = Tuple{Float64,Float64,Float64,Float64}[]
for i in eachindex(mesh.nodes)
    mesh.eq_type[i] == 2 || continue
    tw = mesh.twin[i]
    tw == 0 && continue
    p = mesh.nodes[i]
    Δ = BEM.Plate.crack_opening_fsdt(mesh, i)
    @printf("   %+6.3f  %11.4e  %11.4e  %11.4e\n", p[1], Δ[1], Δ[2], Δ[3])
    push!(codJ, (p[1], Δ[1], Δ[2], Δ[3]))
end

oct = joinpath(@__DIR__, "crack_multiregion_octave.csv")
if isfile(oct)
    A = readdlm(oct, ',')
    println("\n  Octave subregion COD:")
    println("    x          Δψx         Δψy          Δw")
    for k in 1:size(A, 1)
        @printf("   %+6.3f  %11.4e  %11.4e  %11.4e\n", A[k, 1], A[k, 2], A[k, 3], A[k, 4])
    end
    # match nearest x for Δψy (mode I)
    println("\n  Δψy (mode I)  nearest-x match:")
    println("    x_oct    Δψy_oct     Δψy_Dual    rel")
    for k in 1:size(A, 1)
        x = A[k, 1]
        j = argmin(abs(t[1] - x) for t in codJ)
        t = codJ[j]
        rel = abs(A[k, 3] - t[3]) / max(abs(t[3]), 1e-30)
        @printf("   %+6.3f  %11.4e  %11.4e  %6.1f %%\n",
            x, A[k, 3], t[3], 100 * rel)
    end
end
println("Done.")
