using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf, Random, StaticArrays, BEM.HMatrices

function run2d(n1d; nmax=32, rtol=1e-6, lfil=20, droptol=1e-4)
    Random.seed!(1)
    xs = range(0.0, 1.0; length=n1d)
    pts = [SVector{2, Float64}(float(x), float(y)) for y in xs for x in xs]
    n = length(pts)
    kf = function (a, b)
        r = hypot(a[1] - b[1], a[2] - b[2])
        return r < 1e-30 ? 16.0 : 2.0 * log(r)
    end
    K = KernelMatrix{typeof(kf), typeof(pts), typeof(pts), Float64}(kf, pts, pts)
    tree = ClusterTree(pts, DyadicSplitter(; nmax=nmax, tight=false); cube=true)
    b = randn(n)
    println("="^78)
    @printf("2D log  N=%d  nmax=%d  H2 rtol=%.0e  ILUT lfil=%d droptol=%.0e\n",
        n, nmax, rtol, lfil, droptol)
    tH = @elapsed H2 = assemble_h2(K, tree; rtol=rtol, threads=true)
    @printf("  H2 assemble %.3fs  %s\n", tH, H2)

    t0 = @elapsed x0, st0 = gmres_h(H2, b; rtol=1e-8, itmax=400, history=true)
    r0 = norm(H2 * x0 - b) / (norm(b) + 1e-14)
    @printf("  GMRES H2          %4s it  %.3fs  resid=%.2e  tot=%.3fs  %s\n",
        string(st0.niter), t0, r0, tH + t0, st0.status)

    tI = @elapsed Pl = ilut(H2; lfil=lfil, droptol=droptol)
    t1 = @elapsed x1, st1 = gmres_h(H2, b; Pl=Pl, rtol=1e-8, itmax=400, history=true)
    r1 = norm(H2 * x1 - b) / (norm(b) + 1e-14)
    @printf("  ILUT factor       %.3fs\n", tI)
    @printf("  GMRES H2+ILUT     %4s it  %.3fs  resid=%.2e  tot=%.3fs  %s\n",
        string(st1.niter), t1, r1, tH + tI + t1, st1.status)
    flush(stdout)
end

println("threads=", Threads.nthreads())
run2d(16; nmax=16, rtol=1e-6, lfil=16, droptol=1e-4)
run2d(32; nmax=32, rtol=1e-6, lfil=20, droptol=1e-4)
run2d(32; nmax=32, rtol=1e-6, lfil=40, droptol=1e-6)
run2d(64; nmax=32, rtol=1e-6, lfil=20, droptol=1e-4)
run2d(100; nmax=64, rtol=1e-6, lfil=32, droptol=1e-4)
