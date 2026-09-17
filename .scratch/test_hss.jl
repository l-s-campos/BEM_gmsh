using DrWatson
@quickactivate :BEM
using LinearAlgebra, Random, StaticArrays, BEM.HMatrices

Random.seed!(1)
xs = range(0.0, 1.0; length=8)
pts = [SVector{2, Float64}(float(x), float(y)) for y in xs for x in xs]
n = length(pts)
kf = function (a, b)
    r = hypot(a[1] - b[1], a[2] - b[2])
    return r < 1e-30 ? 16.0 : 2.0 * log(r)
end
K = KernelMatrix{typeof(kf), typeof(pts), typeof(pts), Float64}(kf, pts, pts)
tree = ClusterTree(pts, PrincipalComponentSplitter(; nmax=12))
for (label, kw) in (("id", (; rtol=1e-8)), ("svd", (; rtol=1e-8, method=:svd)))
    H = assemble_hss(K, tree; kw...)
    println(label, " ", H)
    x = randn(n)
    Kd = Matrix(K)
    y = H * x
    println("  matvec resid ", norm(y - Kd * x) / (norm(Kd * x) + 1e-14))
    F = ulv(H)
    b = randn(n)
    u = F \ copy(b)
    println("  ", F, " dense resid ", norm(Kd * u - b) / (norm(b) + 1e-14),
        " HSS resid ", norm(H * u - b) / (norm(b) + 1e-14))
end
