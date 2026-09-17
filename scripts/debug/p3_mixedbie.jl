# Mixed BIE: CBIE rows on Dirichlet collocation, HBIE rows on Neumann.
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf
include(datadir("Laplace", "Laplace_dad.jl"))
MAT1 = (E1=124.04e3, E2=10.09e3, G12=6.03e3, ν12=0.334, η12_1=1.255, η12_2=-0.031)
props = AnisotropicElasticity(lekhnitskii_engineering(
    MAT1.E1, MAT1.E2, MAT1.G12, MAT1.ν12; η12_1=MAT1.η12_1, η12_2=MAT1.η12_2))
msh = datadir("elastico", "p3_cmp_q2.msh")
dad = format2d(msh, props; tipo=2, pontointerno=false)
assemble!(dad; npg=16, threaded=false)
Hc, Gc = copy(dad.H), copy(dad.G)
solve(dad); uc = copy(dad.u)
H_G_hyper(dad; npg=50, threaded=false)
Hh, Gh = copy(dad.H), copy(dad.G)

function mix_and_scale(dad, Hc, Gc, Hh, Gh, uc)
n = dad.n; dim = 2
Hmix = copy(Hh); Gmix = copy(Gh)
ndir = 0
for i in 1:n
    d1 = dim*(i-1)+1; d2 = dim*i
    if dad.BC[d1] == 0 && dad.BC[d2] == 0
        ndir += 1
        Hmix[d1:d2, :] .= Hc[d1:d2, :]
        Gmix[d1:d2, :] .= Gc[d1:d2, :]
    end
end
println("mixed BIE: replaced $ndir Dirichlet-node rows with CBIE")
set_cache!(dad; H=Hmix, G=Gmix)
solve(dad)
rel = norm(dad.u .- uc)/(norm(uc)+1e-30)
println("CBIE max|u|=$(maximum(abs, uc))")
println("mixed max|u|=$(maximum(abs, dad.u))  rel=$(rel)  cond=$(cond(Matrix(dad.A)))")

# also: all HBIE with row+col scaling
set_cache!(dad; H=Hh, G=Gh)
applyBC(dad)
A = Matrix(dad.A); b = copy(dad.b)
rd = [max(maximum(abs, view(A, i, :)), 1e-30) for i in 1:size(A,1)]
A ./= rd; b ./= rd
cd = [max(maximum(abs, view(A, :, j)), 1e-30) for j in 1:size(A,2)]
for j in 1:length(cd); A[:, j] ./= cd[j]; end
println("scaled cond=$(cond(A))")
xsc = (A \ b) ./ cd
u = zeros(length(xsc)); t = zeros(length(xsc))
BEM.split_sol!(dad, xsc, u, t)
println("scaled HBIE max|u|=$(maximum(abs, u))  rel=$(norm(u.-uc)/(norm(uc)+1e-30))")
end
mix_and_scale(dad, Hc, Gc, Hh, Gh, uc)
println("done")
