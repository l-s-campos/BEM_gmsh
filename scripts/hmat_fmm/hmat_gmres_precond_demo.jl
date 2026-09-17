# GMRES on hierarchical operator with LU / Chol preconditioner
using LinearAlgebra
using BEM
using BEM.HMatrices

pts = [SVector(float(x), float(y)) for y in range(0, 1; length=10) for x in range(0, 1; length=10)]
n = length(pts)
K = KernelMatrix(pts, pts) do x, y
    d2 = sum(abs2, x - y)
    exp(-8 * d2) + (d2 < 1e-30 ? 2.0 : 0.0)
end
tree = ClusterTree(pts, PrincipalComponentSplitter(; nmax=16))
H = assemble_hmatrix(K, tree, tree;
    adm=StrongAdmissibilityStd(; eta=2.0),
    comp=PartialACA(; rtol=1e-6), threads=false)
b = randn(n)

x_u, st_u = gmres_h(H, b; rtol=1e-8, itmax=4n)
Pl = lu(deepcopy(H); rtol=1e-6)
x_p, st_p = gmres_h(H, b; Pl=Pl, rtol=1e-8, itmax=4n)
Fc = cholesky(H; ridge=1e-10, rtol=1e-6)
x_c, st_c = gmres_h(H, b; Pl=Fc, rtol=1e-8, itmax=4n)

println("n=$n")
println("  unprecond  iters=$(st_u.niter)  res=$(norm(H*x_u - b)/norm(b))")
println("  LU precond iters=$(st_p.niter)  res=$(norm(H*x_p - b)/norm(b))")
println("  Chol precond iters=$(st_c.niter) res=$(norm(H*x_c - b)/norm(b))")
