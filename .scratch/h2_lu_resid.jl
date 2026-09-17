using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf, Random, StaticArrays, BEM.HMatrices

function run(n1d; nmax=16, rtol=1e-6)
    Random.seed!(1)
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
    t = @elapsed F = lu(A; method=:nested, rtol=rtol)
    u = F \ copy(b)
    rel = norm(A * u - b) / (norm(b) + 1e-14)
    @printf("N=%4d rtol=%8.1e  lu=%7.3fs  resid=%.3e\n", length(pts), rtol, t, rel)
    return rel
end

run(16; rtol=0.0)
run(16; rtol=1e-6)
run(32; rtol=1e-6)
run(48; rtol=0.0)
run(48; rtol=1e-6)
