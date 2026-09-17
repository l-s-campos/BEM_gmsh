# HSS+ULV vs nested H² LU vs rskelf on the 2D log kernel.
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Random, StaticArrays, Printf, BEM.HMatrices

function log_kernel(pts)
    kf = function (a, b)
        r = hypot(a[1] - b[1], a[2] - b[2])
        return r < 1e-30 ? 16.0 : 2.0 * log(r)
    end
    return KernelMatrix{typeof(kf), typeof(pts), typeof(pts), Float64}(kf, pts, pts)
end

function grid2d(m)
    xs = range(0.0, 1.0; length=m)
    return [SVector{2, Float64}(float(x), float(y)) for y in xs for x in xs]
end

function run_one(m; rtol=1e-8, nmax=16)
    Random.seed!(1)
    pts = grid2d(m)
    n = length(pts)
    K = log_kernel(pts)
    Kd = Matrix(K)
    b = randn(n)
    println("="^72)
    @printf("N = %d  (grid %d×%d)  rtol=%.0e  nmax=%d\n", n, m, m, rtol, nmax)

    tree_pca = ClusterTree(pts, PrincipalComponentSplitter(; nmax=nmax))
    x = randn(n)
    for (lab, meth) in (("HSS ID ", :id), ("HSS ACA", :aca))
        t_hss = @elapsed Hss = assemble_hss(K, tree_pca; rtol=rtol, method=meth)
        t_ulv = @elapsed Fulv = ulv(Hss)
        t_ulv_s = @elapsed u_ulv = Fulv \ copy(b)
        r_ulv = norm(Kd * u_ulv - b) / (norm(b) + 1e-14)
        r_hss = norm(Hss * u_ulv - b) / (norm(b) + 1e-14)
        r_mv = norm(Hss * x - Kd * x) / (norm(Kd * x) + 1e-14)
        @printf("  %s assemble %8.3fs  factor %8.3fs  solve %8.3fs\n",
            lab, t_hss, t_ulv, t_ulv_s)
        @printf("            %s  matvec=%.2e  dense resid=%.2e  HSS resid=%.2e\n",
            Hss, r_mv, r_ulv, r_hss)
    end

    tree_h2 = ClusterTree(pts, DyadicSplitter(; nmax=nmax, tight=false); cube=true)
    t_h2 = @elapsed H2 = assemble_h2(K, tree_h2; rtol=rtol, threads=false)
    t_h2lu = @elapsed Fh2 = lu(H2; method=:nested, rtol=rtol)
    t_h2s = @elapsed u_h2 = Fh2 \ copy(b)
    r_h2 = norm(Kd * u_h2 - b) / (norm(b) + 1e-14)
    r_h2op = norm(H2 * u_h2 - b) / (norm(b) + 1e-14)
    @printf("  H² nested assemble %8.3fs  factor %8.3fs  solve %8.3fs\n",
        t_h2, t_h2lu, t_h2s)
    @printf("            maxrank=%d  dense resid=%.2e  H² resid=%.2e\n",
        maxrank(H2), r_h2, r_h2op)

    pxy = circle_proxy(K; npts=32)
    t_rs = @elapsed Frs = rskelf(K, tree_h2; rtol=rtol, rank=48, pxyfun=pxy,
        Tmax=2, symm=:s)
    t_rss = @elapsed u_rs = Frs \ copy(b)
    r_rs = norm(Kd * u_rs - b) / (norm(b) + 1e-14)
    @printf("  rskelf    factor   %8.3fs  (incl. samples)  solve %8.3fs\n",
        t_rs, t_rss)
    @printf("            dense resid=%.2e  nsteps=%d\n", r_rs, length(Frs.steps))
    return nothing
end

println("warmup N=64")
run_one(8; rtol=1e-6, nmax=12)
println()
println("timed")
run_one(16; rtol=1e-8, nmax=16)
run_one(32; rtol=1e-8, nmax=32)
