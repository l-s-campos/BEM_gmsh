# Local BEM vs particular-solution RBF-BEM on the unit square.
#   ∇²u = 4,  u = x² + y² on Γ.
using BEM
using LinearAlgebra
using Statistics: mean
using Printf

include(joinpath(@__DIR__, "..", "..", "data", "Laplace", "Laplace_dad.jl"))

ufun(p) = p[1]^2 + p[2]^2

function rmse(dad)
    pts = all_points(dad)
    return sqrt(mean(abs2, dad.T .- ufun.(pts)))
end

msh = quadrado(ndiv=10, show=false, nome="lbem_demo")
dadL = format2d(msh, Laplace(1.0); pontointerno=true)
for i in 1:dadL.n
    dadL.BC[i] = 0
    dadL.BV[i] = ufun(dadL.Nodes[i])
end
dadR = deepcopy(dadL)

dadLg = deepcopy(dadL)
solve_local_bem!(dadL, 4.0; npg=16, source=:local)
solve_local_bem!(dadLg, 4.0; npg=16, source=:global)
eL = rmse(dadL)
eG = rmse(dadLg)
@printf("Local BEM M-local  RMSE = %.4e   n = %d (ni = %d)\n", eL, dadL.nt, dadL.ni)
@printf("Local BEM M-global RMSE = %.4e\n", eG)

uR = solve_poisson_rbf_bem!(dadR, 4.0; method=:global, basis=PHS(3; poly_deg=1), npg=14)
eR = sqrt(mean(abs2, uR .- ufun.(all_points(dadR))))
@printf("RBF-BEM     RMSE = %.4e\n", eR)
