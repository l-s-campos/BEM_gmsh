# Constant-stress patch test on the P3 quarter-annulus mesh.
# If HBIE fails here, the operator is wrong on curves; if it passes, P3 BCs/conditioning.
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf, StaticArrays
include(datadir("Laplace", "Laplace_dad.jl"))

MAT1 = (E1=124.04e3, E2=10.09e3, G12=6.03e3, ν12=0.334, η12_1=1.255, η12_2=-0.031)
props = AnisotropicElasticity(lekhnitskii_engineering(
    MAT1.E1, MAT1.E2, MAT1.G12, MAT1.ν12; η12_1=MAT1.η12_1, η12_2=MAT1.η12_2))
D = inv(props.params.C)
p = 100.0
ana = AnalyticalSolution("p3-patch",
    (x; t=0.0) -> SVector(D[1, 1] * p * x[1] + D[1, 3] * p * x[2],
                          D[1, 2] * p * x[2]);
    q = (x, n; t=0.0) -> SVector(p * n[1], 0.0),
    description="constant σx=p")

msh = datadir("elastico", "p3_exact.msh")
isfile(msh) || (msh = datadir("elastico", "cordeiro_p3.msh"))
dad = format2d(msh, props; tipo=2, pontointerno=false)

# all-Dirichlet patch
neu = Int[]
apply_analytical_bc!(dad, ana, neu)
println("all-Dirichlet patch, n=$(dad.n)")
assemble!(dad; npg=16, threaded=false); solve(dad)
uana = analytical(dad)
@printf("  CBIE  rel(u)=%.3e  rel(t)=%.3e  max|u|=%.4f  max|t|=%.4f\n",
    norm(dad.u .- uana)/(norm(uana)+1e-30),
    norm(dad.traction .- [ana.q(dad.Nodes[i], dad.Normal[i])[k] for i in 1:dad.n for k in 1:2]) /
        (norm([ana.q(dad.Nodes[i], dad.Normal[i])[k] for i in 1:dad.n for k in 1:2])+1e-30),
    maximum(abs, dad.u), maximum(abs, dad.traction))

H_G_hyper(dad; npg=50, threaded=false); solve(dad)
tana = [ana.q(dad.Nodes[i], dad.Normal[i])[k] for i in 1:dad.n for k in 1:2]
@printf("  HBIE  rel(u)=%.3e  rel(t)=%.3e  max|u|=%.4f  max|t|=%.4f  cond=%.2e\n",
    norm(dad.u .- uana)/(norm(uana)+1e-30),
    norm(dad.traction .- tana)/(norm(tana)+1e-30),
    maximum(abs, dad.u), maximum(abs, dad.traction), cond(Matrix(dad.A)))

# mixed: Dirichlet on vertical (x≈0), Neumann elsewhere — same field
dad = format2d(msh, props; tipo=2, pontointerno=false)
vert = Int[i for i in 1:dad.n if dad.Nodes[i][1] < 1.0]
neu = setdiff(1:dad.n, vert)
apply_analytical_bc!(dad, ana, neu)
println("\nmixed patch (Dirichlet on x=0), nDir=$(length(vert))")
assemble!(dad; npg=16, threaded=false); solve(dad)
uana = analytical(dad)
@printf("  CBIE  rel(u)=%.3e  max|u|=%.4f\n",
    norm(dad.u .- uana)/(norm(uana)+1e-30), maximum(abs, dad.u))
H_G_hyper(dad; npg=50, threaded=false); solve(dad)
@printf("  HBIE  rel(u)=%.3e  max|u|=%.4f  cond=%.2e\n",
    norm(dad.u .- uana)/(norm(uana)+1e-30), maximum(abs, dad.u), cond(Matrix(dad.A)))

# isotropic patch on the same mesh
iso = Elasticity(MAT1.E1, MAT1.ν12, 1.0; plane_stress=true)
Diso = (1 - iso.nu^2) / iso.E   # not needed
ana_iso = AnalyticalSolution("iso-patch",
    (x; t=0.0) -> SVector(p / iso.E * x[1], -iso.nu * p / iso.E * x[2]);
    q = (x, n; t=0.0) -> SVector(p * n[1], 0.0))
dad = format2d(msh, iso; tipo=2, pontointerno=false)
apply_analytical_bc!(dad, ana_iso, Int[])
println("\nisotropic all-Dirichlet patch")
assemble!(dad; npg=16, threaded=false); solve(dad)
uana = analytical(dad)
@printf("  CBIE  rel(u)=%.3e\n", norm(dad.u .- uana)/(norm(uana)+1e-30))
H_G_hyper(dad; npg=50, threaded=false); solve(dad)
@printf("  HBIE  rel(u)=%.3e  cond=%.2e\n",
    norm(dad.u .- uana)/(norm(uana)+1e-30), cond(Matrix(dad.A)))
println("done")
