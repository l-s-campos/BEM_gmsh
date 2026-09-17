using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf, StaticArrays
include(datadir("Laplace", "Laplace_dad.jl"))

MAT1 = lekhnitskii_engineering(124.04e3, 10.09e3, 6.03e3, 0.334; η12_1=1.255, η12_2=-0.031)
props = AnisotropicElasticity(MAT1)
r = SVector(0.3, 0.4)
n = SVector(0.0, 1.0)
U = BEM._to_smat(fundamental(props, r, zero(r), n).U)
T = BEM._to_smat(fundamental(props, r, zero(r), n).T)
@printf("||U-UT||/||U||=%.3e  ||T-TT||/||T||=%.3e\n",
    norm(U - transpose(U)) / norm(U), norm(T - transpose(T)) / norm(T))
println("T=\n", T)

# Y-tension patch on P1 rectangle
D = inv(props.params.C)
p = 100.0
Lx, Ly = 500.0, 200.0
ana_y = AnalyticalSolution("sy",
    (x; t=0.0) -> SVector(D[1, 2] * p * x[1] + D[2, 3] * p * x[2],
                          D[2, 2] * p * x[2]);
    q = (x, n; t=0.0) -> SVector(0.0, p * n[2]))
ana_x = AnalyticalSolution("sx",
    (x; t=0.0) -> SVector(D[1, 1] * p * x[1] + D[1, 3] * p * x[2],
                          D[1, 2] * p * x[2]);
    q = (x, n; t=0.0) -> SVector(p * n[1], 0.0))

function patch(ana, lab)
    dad = format2d(datadir("elastico", "cordeiro_p1.msh"), props; tipo=2, pontointerno=false)
    apply_analytical_bc!(dad, ana, Int[])
    assemble!(dad; npg=16, threaded=false); solve(dad)
    uana = analytical(dad)
    tana = [ana.q(dad.Nodes[i], dad.Normal[i])[k] for i in 1:dad.n for k in 1:2]
    @printf("  CBIE %-4s  rel(u)=%.3e  rel(t)=%.3e  max|u|=%.4f mm\n",
        lab, norm(dad.u .- uana) / (norm(uana) + 1e-30),
        norm(dad.traction .- tana) / (norm(tana) + 1e-30), maximum(abs, dad.u))
    H_G_hyper(dad; npg=16, threaded=false); solve(dad)
    @printf("  HBIE %-4s  rel(u)=%.3e  rel(t)=%.3e  max|u|=%.4f mm\n",
        lab, norm(dad.u .- uana) / (norm(uana) + 1e-30),
        norm(dad.traction .- tana) / (norm(tana) + 1e-30), maximum(abs, dad.u))
end
println("all-Dirichlet patches")
patch(ana_x, "σx")
patch(ana_y, "σy")
