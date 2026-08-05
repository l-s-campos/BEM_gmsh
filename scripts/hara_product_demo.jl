# Demo: HARA build of C ≈ A*B from matvecs only (v ↦ A(Bv))
using LinearAlgebra
using BEM

pts = [SVector(float(x), float(y)) for y in range(0, 1; length=12) for x in range(0, 1; length=12)]
n = length(pts)
K = KernelMatrix(pts, pts) do x, y
    d2 = sum(abs2, x - y)
    exp(-8 * d2) + (d2 < 1e-30 ? 2.0 : 0.0)
end
tree = ClusterTree(pts, PrincipalComponentSplitter(; nmax=20))
adm = StrongAdmissibilityStd(; eta=2.0)
comp = PartialACA(; rtol=1e-6)

A = assemble_hmatrix(K, tree, tree; adm=adm, comp=comp, threads=false)
B = assemble_hmatrix(K, tree, tree; adm=adm, comp=comp, threads=false)
Ad, Bd = Matrix(A), Matrix(B)

S = FunctionSampler(
    (Y, X) -> mul!(Y, Ad, Bd * X), n;
    f_adj! = (Y, X) -> mul!(Y, Bd', Ad' * X))

C = hara(S, tree, tree; adm=adm, rtol=1e-3, batch=12, threads=false)
x = randn(n)
rel = norm(C * x - Ad * (Bd * x)) / (norm(Ad * (Bd * x)) + 1e-14)
println("HARA product demo: n=$n  rel_matvec_err=$rel  compression=$(compression_ratio(C))")
@assert rel < 0.1
