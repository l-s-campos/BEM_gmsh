# MATLAB Static_Thick_Cracked_Plate/test01.m — Dual BEM equivalent in Julia.
# SS square [-1,1]², centre crack a=0.5, q=1, E=1e9, ν=0.3, h=0.1667.
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf, DelimitedFiles
using BEM.Plate

W, Ht, a = 1.0, 1.0, 0.5
E, ν, hpl, q0 = 1.0e9, 0.3, 0.1667, 1.0
props = FSDTProps(; E=E, ν=ν, h=hpl, q_c=q0)

println("="^72)
println(" Static_Thick_Cracked_Plate / test01  —  Julia Dual BEM")
println("="^72)
@printf("  SS [-%.0f,%.0f]²  a=%.2f  h=%.4f  E=%.3e  q=%.1f\n", W, W, a, hpl, E, q0)

# MATLAB: 8 BE/outer edge, 16/crack face. Gmsh transfinite nodes = n_el+1.
mesh = build_rect_fsdt_crack(; W=W, H=Ht, a=a, props=props, Mo=0.0,
    ndiv_b=9, ndiv_h=9, ndiv_crack=17, nome="matlab_test01")
# Match MATLAB BCU/BCF (test01):
#   y=±H: ψx=0, w=0, My traction 0
#   x=±W: ψy=0, w=0, Mx traction 0
#   crack: traction-free
n = length(mesh.nodes)
for i in 1:n
    p = mesh.nodes[i]
    if mesh.eq_type[i] != 1
        mesh.BC[3i-2:3i] .= 1
        mesh.BV[3i-2:3i] .= 0.0
        continue
    end
    mesh.BC[3i-2:3i] .= 1
    mesh.BV[3i-2:3i] .= 0.0
    if abs(abs(p[2]) - Ht) < 1e-6
        mesh.BC[3i-2] = 0
        mesh.BC[3i] = 0
    elseif abs(abs(p[1]) - W) < 1e-6
        mesh.BC[3i-1] = 0
        mesh.BC[3i] = 0
    end
end
@printf("  Dual mesh n_node=%d  eq2=%d  eq3=%d\n",
    n, count(==(2), mesh.eq_type), count(==(3), mesh.eq_type))
assemble_fsdt_dual!(mesh; npg=10, nsub=8)
dibem_fsdt!(mesh; npg=8)
# MATLAB IntQeDBE on face B; our DIBEM is U-RIM on every row. Zero HBIE q
# for a second solve to isolate the domain-load kernel.
qU = copy(mesh.q)
solve_fsdt!(mesh)
K1b, K2b, K3b, rA, rB, Le = sif_ctod_fsdt(mesh; tip=:right)
@printf("  Julia Dual (U-RIM all rows)  K1b=%.4e  K2b=%.4e  K3b=%.4e\n", K1b, K2b, K3b)
@printf("  rA/Le=%.3f  rB/Le=%.3f\n", rA / Le, rB / Le)
for i in 1:n
    mesh.eq_type[i] == 3 || continue
    mesh.q[3i-2:3i] .= 0
end
solve_fsdt!(mesh)
K1b2, _, _, _, _, _ = sif_ctod_fsdt(mesh; tip=:right)
@printf("  Julia Dual (q=0 on HBIE rows) K1b=%.4e\n", K1b2)
mesh.q .= qU
println("  Dual COD (A−B), U-RIM load:")
println("    x          Δψx         Δψy          Δw")
codJ = Tuple{Float64,Float64,Float64,Float64}[]
for i in 1:n
    mesh.eq_type[i] == 2 || continue
    mesh.twin[i] == 0 && continue
    p = mesh.nodes[i]
    Δ = BEM.Plate.crack_opening_fsdt(mesh, i)
    @printf("   %+6.3f  %11.4e  %11.4e  %11.4e\n", p[1], Δ[1], Δ[2], Δ[3])
    push!(codJ, (p[1], Δ[1], Δ[2], Δ[3]))
end

sif = joinpath(@__DIR__, "usech_thick_crack_octave_sif.txt")
oct = joinpath(@__DIR__, "usech_thick_crack_octave.csv")
if isfile(sif)
    println("\n  Octave MATLAB Dual SIFs:")
    print(read(sif, String))
end
if isfile(oct)
    A = readdlm(oct, ',')
    println("  Octave COD vs Julia Dual Δψy (nearest x):")
    println("    x_oct    Δψy_oct     Δψy_Julia   rel")
    for k in 1:size(A, 1)
        x = A[k, 1]
        j = argmin(abs(t[1] - x) for t in codJ)
        t = codJ[j]
        rel = abs(A[k, 3] - t[3]) / max(abs(t[3]), 1e-30)
        @printf("   %+6.3f  %11.4e  %11.4e  %6.1f %%\n",
            x, A[k, 3], t[3], 100 * rel)
    end
    K1m = A[1, 5]
    @printf("  K1b  Octave=%.4e  Julia=%.4e  rel %.1f %%\n",
        K1m, K1b, 100 * abs(K1b - K1m) / max(abs(K1m), 1e-30))
end
println("Done.")
