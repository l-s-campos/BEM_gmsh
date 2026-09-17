# Ill-conditioned on-circle HBIE: truncated SVD / pinv vs default solve.
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf
include(datadir("Laplace", "Laplace_dad.jl"))
MAT1 = (E1=124.04e3, E2=10.09e3, G12=6.03e3, ν12=0.334, η12_1=1.255, η12_2=-0.031)
props = AnisotropicElasticity(lekhnitskii_engineering(
    MAT1.E1, MAT1.E2, MAT1.G12, MAT1.ν12; η12_1=MAT1.η12_1, η12_2=MAT1.η12_2))
msh = datadir("elastico", "p3_cmp_q2.msh")
dad = format2d(msh, props; tipo=2, pontointerno=false)
assemble!(dad; npg=16, threaded=false); solve(dad)
uc = copy(dad.u)
println("CBIE max|u|=$(maximum(abs, uc)) mm")
H_G_hyper(dad; npg=50, threaded=false)
applyBC(dad)
A, b = Matrix(dad.A), copy(dad.b)
println("cond(A)=$(cond(A))  size=$(size(A))")
S = svd(A)
println("σ max=$(S.S[1])  σ min=$(S.S[end])  σ[end-5:end]=$(S.S[end-5:end])")

function recover(x, dad)
    u = zeros(length(x)); t = zeros(length(x))
    BEM.split_sol!(dad, x, u, t)
    return u, t
end

x0 = A \ b
u0, _ = recover(x0, dad)
println("backslash  max|u|=$(maximum(abs, u0))  relCBIE=$(norm(u0.-uc)/(norm(uc)+1e-30))")

for rtol in (1e-4, 1e-6, 1e-8, 1e-10, 1e-12)
    x = pinv(A; rtol=rtol) * b
    u, _ = recover(x, dad)
    kkeep = count(>(rtol * S.S[1]), S.S)
    @printf("pinv rtol=%7.1e  keep=%3d  max|u|=%10.3f  relCBIE=%.3e\n",
        rtol, kkeep, maximum(abs, u), norm(u.-uc)/(norm(uc)+1e-30))
end
println("done")
