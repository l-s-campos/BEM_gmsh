using DrWatson
@quickactivate :BEM
using LinearAlgebra, Random, StaticArrays, BEM.HMatrices
Random.seed!(1)
n1d = 48
nmax = 16
xs = range(0.0, 1.0; length=n1d)
pts = [SVector(float(x), float(y)) for y in xs for x in xs]
K = KernelMatrix(pts, pts) do a, b
    r = hypot(a[1] - b[1], a[2] - b[2])
    return r < 1e-30 ? 4.0 : log(r)
end
Kd = Matrix(K)
Kd = Kd + Kd' + 8.0 * I
tree = ClusterTree(pts, DyadicSplitter(; nmax=nmax, tight=false); cube=true)
A = assemble_h2(Kd, tree; rtol=1e-8, threads=false)
b = randn(length(pts))
t = @elapsed F = lu(A; method=:nested, rtol=0.0)
u = F \ copy(b)
rel = norm(A * u - b) / (norm(b) + 1e-14)
println("N=$(length(pts)) rtol=0 lu=$(round(t; digits=3))s resid=$rel")
