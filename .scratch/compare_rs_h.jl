using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf, Random, StaticArrays, BEM.HMatrices

function shifted_log(n1d; nmax=32)
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
    pxy = circle_proxy(kf, pts; npts=64)
    b = randn(n)
    return K, tree, pxy, b, n
end

function report(name, t_fac, F, A, b, t_gmres, x, st)
    r = norm(A * x - b) / (norm(b) + 1e-14)
    @printf("  %-18s fac=%7.3fs  GMRES %4s it  %.3fs  resid=%.2e  tot=%.3fs  %s\n",
        name, t_fac, string(st.niter), t_gmres, r, t_fac + t_gmres, st.status)
    return nothing
end

function run(n1d; nmax=32)
    K, tree, pxy, b, n = shifted_log(n1d; nmax=nmax)
    println("="^78)
    @printf("N=%d  nmax=%d  threads=%d\n", n, nmax, Threads.nthreads())

    tH2 = @elapsed H2 = assemble_h2(K, tree; rtol=1e-6, threads=true)
    tH = @elapsed H = assemble_hmatrix(K, tree, tree;
        adm=StrongAdmissibilityStd(2.0), comp=PartialACA(; rtol=1e-6), threads=true)
    @printf("  assemble H2=%.3fs  H=%.3fs\n", tH2, tH)

    t0 = @elapsed x0, st0 = gmres_h(H2, b; rtol=1e-8, itmax=200, history=true)
    r0 = norm(H2 * x0 - b) / (norm(b) + 1e-14)
    @printf("  %-18s fac=%7.3fs  GMRES %4s it  %.3fs  resid=%.2e  tot=%.3fs  %s\n",
        "H2 unprecond", tH2, string(st0.niter), t0, r0, tH2 + t0, st0.status)

    tHL = @elapsed FH = lu(H; rtol=1e-8)
    tHg = @elapsed xH, stH = gmres_h(H2, b; Pl=FH, rtol=1e-8, itmax=40, history=true)
    report("H2 + H-LU", tH + tHL, FH, H2, b, tHg, xH, stH)

    tS = @elapsed Fsrs = srs_factor(K, tree; rtol=1e-6, rank=40)
    tSg = @elapsed xs, sts = gmres_h(H2, b; Pl=Fsrs, rtol=1e-8, itmax=40, history=true)
    report("H2 + SRS(K)", tS, Fsrs, H2, b, tSg, xs, sts)

    tR = @elapsed Fr = rskelf(K, tree; rtol=1e-6, rank=40, pxyfun=pxy, Tmax=2, symm=:s)
    tRg = @elapsed xr, str = gmres_h(H2, b; Pl=Fr, rtol=1e-8, itmax=40, history=true)
    report("H2 + rskelf", tR, Fr, H2, b, tRg, xr, str)
    ur = Fr \ copy(b)
    @printf("           rskelf direct resid vs H2 = %.2e  %s\n",
        norm(H2 * ur - b) / (norm(b) + 1e-14), string(Fr))

    flush(stdout)
end

println("threads=", Threads.nthreads())
run(100; nmax=64)
