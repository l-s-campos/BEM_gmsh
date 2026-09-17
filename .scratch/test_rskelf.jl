using DrWatson
@quickactivate :BEM
using LinearAlgebra, Random, StaticArrays, BEM.HMatrices

Random.seed!(1)
U = randn(20, 4)
V = randn(12, 4)
A = U * V'
sk, rd, T = interpolative_decomp(A; rtol=1e-12)
println("ID resid ", norm(A[:, rd] - A[:, sk] * T) / norm(A))

xs = range(0.0, 1.0; length=16)
pts = [SVector{2, Float64}(float(x), float(y)) for y in xs for x in xs]
n = length(pts)
kf = function (a, b)
    r = hypot(a[1] - b[1], a[2] - b[2])
    return r < 1e-30 ? 16.0 : 2.0 * log(r)
end
K = KernelMatrix{typeof(kf), typeof(pts), typeof(pts), Float64}(kf, pts, pts)
tree = ClusterTree(pts, DyadicSplitter(; nmax=16, tight=false); cube=true)
b = randn(n)
Kd = Matrix(K)

pxy = circle_proxy(kf, pts; npts=32)
Fr = rskelf(K, tree; rtol=1e-8, rank=24, pxyfun=pxy, Tmax=2, symm=:s)
ur = Fr \ copy(b)
println("rskelf ", Fr)
println("  solve resid ", norm(Kd * ur - b) / norm(b))
println("  mul resid   ", norm(Fr * ur - b) / norm(b))

Fp = rskelf(K, tree; rtol=1e-6, rank=16, pxyfun=pxy, Tmax=2, symm=:s)
up = Fp \ copy(b)
println("rskelf+proxy ", Fp, " resid ", norm(Kd * up - b) / norm(b))

Fh0 = hifie2(K, tree; rtol=1e-6, rank=16, pxyfun=pxy, Tmax=2, symm=:s, skip=100)
uh0 = Fh0 \ copy(b)
println("hifie2 skip ", Fh0, " resid ", norm(Kd * uh0 - b) / norm(b))
Fh = hifie2(K, tree; rtol=1e-6, rank=16, pxyfun=pxy, Tmax=2, symm=:s)
uh = Fh \ copy(b)
println("hifie2 ", Fh, " resid ", norm(Kd * uh - b) / norm(b))

Fs = srskelf(K, tree; rtol=1e-6, rank=16, pxyfun=pxy, Tmax=2, symm=:s)
us = Fs \ copy(b)
println("srskelf ", Fs, " resid ", norm(Kd * us - b) / norm(b))
