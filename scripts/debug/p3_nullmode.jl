# Pure HBIE on on-circle P3: inspect the tiny SVD mode and try equilibrated solve.
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
A = Matrix(dad.A); b = copy(dad.b)
S = svd(A)
println("cond=$(cond(A))  σmin=$(S.S[end])  σ[end-2:end]=$(S.S[end-2:end])")

v = S.Vt[end, :]          # right singular vector
u = S.U[:, end]           # left
# where is the mode
n = dad.n
println("\nright singular vector (unknowns): largest 8 |entries|")
idx = sortperm(abs.(v); rev=true)
for k in 1:8
    j = idx[k]
    node = (j + 1) ÷ 2
    xy = dad.Nodes[node]
    dir = isodd(j) ? "x" : "y"
    bc = dad.BC[j] == 0 ? "Dir" : "Neu"
    @printf("  dof=%3d node=%2d %s %s  (%.1f,%.1f)  v=% .3e\n",
        j, node, dir, bc, xy[1], xy[2], v[j])
end
println("left singular vector: largest 8")
idx = sortperm(abs.(u); rev=true)
for k in 1:8
    j = idx[k]
    node = (j + 1) ÷ 2
    xy = dad.Nodes[node]
    dir = isodd(j) ? "x" : "y"
    @printf("  row=%3d node=%2d %s  (%.1f,%.1f)  u=% .3e\n",
        j, node, dir, xy[1], xy[2], u[j])
end

function recover(x)
    uu = zeros(length(x)); tt = zeros(length(x))
    BEM.split_sol!(dad, x, uu, tt)
    return uu
end

# row+column equilibration (pure HBIE, same BIE)
function equilibrate_solve(A, b)
    rd = [max(maximum(abs, view(A, i, :)), 1e-30) for i in axes(A, 1)]
    Ae = A ./ rd
    be = b ./ rd
    cd = [max(maximum(abs, view(Ae, :, j)), 1e-30) for j in axes(Ae, 2)]
    for j in axes(Ae, 2)
        Ae[:, j] ./= cd[j]
    end
    x = (Ae \ be) ./ cd
    return x, cond(Ae)
end

xeq, ceq = equilibrate_solve(A, b)
ueq = recover(xeq)
@printf("\nequilibrated  cond=%.2e  max|u|=%.3f  relCBIE=%.3e\n",
    ceq, maximum(abs, ueq), norm(ueq .- uc)/(norm(uc)+1e-30))

# drop only the last SVD mode
rtol = 10 * S.S[end] / S.S[1]
xpin = pinv(A; rtol=rtol) * b
upin = recover(xpin)
@printf("drop 1 mode   rtol=%.2e  max|u|=%.3f  relCBIE=%.3e\n",
    rtol, maximum(abs, upin), norm(upin .- uc)/(norm(uc)+1e-30))
println("done")
