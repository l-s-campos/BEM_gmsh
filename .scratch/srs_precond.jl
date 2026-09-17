using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf, Random, StaticArrays, BEM.HMatrices

function run(n1d; nmax=32, rtol=1e-6)
    Random.seed!(1)
    xs = range(0.0, 1.0; length=n1d)
    pts = [SVector(float(x), float(y)) for y in xs for x in xs]
    n = length(pts)
    K = KernelMatrix(pts, pts) do a, b
        r = hypot(a[1] - b[1], a[2] - b[2])
        return r < 1e-30 ? 16.0 : 2.0 * log(r)
    end
    tree = ClusterTree(pts, DyadicSplitter(; nmax=nmax, tight=false); cube=true)
    b = randn(n)
    tH2 = @elapsed H2 = assemble_h2(K, tree; rtol=rtol, threads=true)
    tF = @elapsed F = srs_factor(K, tree; rtol=rtol, rank=40)
    u = F \ copy(b)
    rdir = norm(H2 * u - b) / (norm(b) + 1e-14)
    t0 = @elapsed _, st0 = gmres_h(H2, b; rtol=1e-8, itmax=400, history=true)
    t1 = @elapsed _, st1 = gmres_h(H2, b; Pl=F, rtol=1e-8, itmax=80, history=true)
    @printf("N=%5d  H2=%.3fs  SRS=%.3fs  dir_resid=%.2e  GMRES %s it / %.2fs  +SRS %s it / %.2fs  (%s)\n",
        n, tH2, tF, rdir, st0.niter, t0, st1.niter, t1, st1.status)
    flush(stdout)
end

println("threads=", Threads.nthreads())
run(8)
run(16)
run(32)
run(48)
