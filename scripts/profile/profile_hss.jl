# Profile HSS assembly / ULV vs nested H² LU vs rskelf.
#
#   julia --project=. scripts/profile/profile_hss.jl
#
# ENV: HSS_N=256,1024,4096  HSS_RTOL=1e-8  HSS_NMAX=32  HSS_NRUN=5

using DrWatson
@quickactivate :BEM
using LinearAlgebra
using Printf
using Random
using Statistics
using Profile
using StaticArrays
using BEM.HMatrices

const NS = parse.(Int, split(get(ENV, "HSS_N", "256,1024,4096"), ','; keepempty=false))
const RTOL = parse(Float64, get(ENV, "HSS_RTOL", "1e-8"))
const NMAX = parse(Int, get(ENV, "HSS_NMAX", "32"))
const NRUN = parse(Int, get(ENV, "HSS_NRUN", "5"))

function log_kernel(pts)
    kf = function (a, b)
        r = hypot(a[1] - b[1], a[2] - b[2])
        return r < 1e-30 ? 16.0 : 2.0 * log(r)
    end
    return KernelMatrix{typeof(kf), typeof(pts), typeof(pts), Float64}(kf, pts, pts)
end

function grid2d(n)
    m = Int(round(sqrt(n)))
    m * m == n || throw(ArgumentError("HSS_N must be a square; got $n"))
    xs = range(0.0, 1.0; length=m)
    return [SVector{2, Float64}(float(x), float(y)) for y in xs for x in xs]
end

fmtb(b) = b < 1024 ? @sprintf("%d B", b) :
          b < 1024^2 ? @sprintf("%.1f KiB", b / 1024) :
          b < 1024^3 ? @sprintf("%.2f MiB", b / 1024^2) :
          @sprintf("%.2f GiB", b / 1024^3)

function _med(f; n=NRUN, w=1)
    for _ in 1:w
        f()
    end
    ts = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        ts[i] = @elapsed f()
    end
    return median(ts), minimum(ts)
end

function _alloc(f)
    f()
    return @allocated f()
end

function setup(n)
    Random.seed!(1)
    pts = grid2d(n)
    K = log_kernel(pts)
    tree_pca = ClusterTree(pts, PrincipalComponentSplitter(; nmax=NMAX))
    tree_h2 = ClusterTree(pts, DyadicSplitter(; nmax=NMAX, tight=false); cube=true)
    b = randn(n)
    return pts, K, tree_pca, tree_h2, b
end

function print_stats(st::HMatrices.HSSBuildStats, wall)
    tot = st.t_kernel + st.t_svd + st.t_pinv + st.t_lsq
    @printf("    kernel fill  %8.3fs  (%5.1f%%)  %d blocks  %s entries\n",
        st.t_kernel, 100 * st.t_kernel / max(wall, 1e-12),
        st.n_kernel, fmtb(st.numel_kernel * sizeof(Float64)))
    @printf("    ID / SVD     %8.3fs  (%5.1f%%)  %d calls\n",
        st.t_svd, 100 * st.t_svd / max(wall, 1e-12), st.n_svd)
    @printf("    pinv (B)     %8.3fs  (%5.1f%%)\n",
        st.t_pinv, 100 * st.t_pinv / max(wall, 1e-12))
    @printf("    lsq (R,W)    %8.3fs  (%5.1f%%)\n",
        st.t_lsq, 100 * st.t_lsq / max(wall, 1e-12))
    @printf("    timed sum    %8.3fs  / wall %8.3fs\n", tot, wall)
    return
end

function profile_n(n; do_nested=true)
    pts, K, tree_pca, tree_h2, b = setup(n)
    println("="^72)
    @printf("N=%d  rtol=%.0e  nmax=%d  nrun=%d\n", n, RTOL, NMAX, NRUN)

    st = HMatrices.HSSBuildStats()
    t_hss_wall = @elapsed Hss = assemble_hss(K, tree_pca; rtol=RTOL, stats=st)
    print_stats(st, t_hss_wall)
    t_hss, t_hss_min = _med(() -> assemble_hss(K, tree_pca; rtol=RTOL))
    a_hss = _alloc(() -> assemble_hss(K, tree_pca; rtol=RTOL))
    t_ulv, t_ulv_min = _med(() -> ulv(Hss))
    F = ulv(Hss)
    t_sol, t_sol_min = _med(() -> F \ copy(b))
    a_ulv = _alloc(() -> ulv(Hss))
    x = randn(n)
    t_mv, t_mv_min = _med(() -> Hss * x)
    @printf("  assemble_hss  med %8.3fs  min %8.3fs  alloc %s  %s\n",
        t_hss, t_hss_min, fmtb(a_hss), Hss)
    @printf("  ulv           med %8.3fs  min %8.3fs  alloc %s\n",
        t_ulv, t_ulv_min, fmtb(a_ulv))
    @printf("  ULV \\         med %8.3fs  min %8.3fs\n", t_sol, t_sol_min)
    @printf("  HSS matvec    med %8.3fs  min %8.3fs\n", t_mv, t_mv_min)
    flush(stdout)

    t_h2, t_h2_min = _med(() -> assemble_h2(K, tree_h2; rtol=RTOL, threads=false))
    H2 = assemble_h2(K, tree_h2; rtol=RTOL, threads=false)
    a_h2 = _alloc(() -> assemble_h2(K, tree_h2; rtol=RTOL, threads=false))
    @printf("  assemble_h2   med %8.3fs  min %8.3fs  alloc %s  maxrank=%d\n",
        t_h2, t_h2_min, fmtb(a_h2), maxrank(H2))
    if do_nested
        t_lu, t_lu_min = _med(() -> lu(H2; method=:nested, rtol=RTOL); n=max(1, NRUN - 2))
        @printf("  nested H² LU  med %8.3fs  min %8.3fs\n", t_lu, t_lu_min)
    else
        println("  nested H² LU  skipped")
    end

    pxy = circle_proxy(K; npts=32)
    t_rs, t_rs_min = _med(() -> rskelf(K, tree_h2; rtol=RTOL, rank=48,
        pxyfun=pxy, Tmax=2, symm=:s))
    a_rs = _alloc(() -> rskelf(K, tree_h2; rtol=RTOL, rank=48,
        pxyfun=pxy, Tmax=2, symm=:s))
    Frs = rskelf(K, tree_h2; rtol=RTOL, rank=48, pxyfun=pxy, Tmax=2, symm=:s)
    t_rss, t_rss_min = _med(() -> Frs \ copy(b))
    @printf("  rskelf        med %8.3fs  min %8.3fs  alloc %s  steps=%d\n",
        t_rs, t_rs_min, fmtb(a_rs), length(Frs.steps))
    @printf("  rskelf \\      med %8.3fs  min %8.3fs\n", t_rss, t_rss_min)
    return Hss, K, tree_pca
end

function stack_sample(n)
    _, K, tree_pca, _, _ = setup(n)
    H = assemble_hss(K, tree_pca; rtol=RTOL)
    ulv(H)
    Profile.clear()
    Profile.init(; delay=0.0005)
    @profile for _ in 1:8
        assemble_hss(K, tree_pca; rtol=RTOL)
    end
    println()
    println("Profile sample (assemble_hss N=$n × 8, delay=0.5 ms)")
    Profile.print(IOContext(stdout, :displaysize => (28, 140));
        C=false, combine=true, mincount=6, noisefloor=2, maxdepth=14, format=:flat,
        sortedby=:count)
    Profile.clear()
    H = assemble_hss(K, tree_pca; rtol=RTOL)
    @profile for _ in 1:40
        ulv(H)
    end
    println()
    println("Profile sample (ulv N=$n × 40, delay=0.5 ms)")
    Profile.print(IOContext(stdout, :displaysize => (28, 140));
        C=false, combine=true, mincount=6, noisefloor=2, maxdepth=14, format=:flat,
        sortedby=:count)
    println()
    return
end

function main()
    println("HSS / ULV profile")
    println("rtol=", RTOL, "  nmax=", NMAX, "  nrun=", NRUN,
        "  threads=", Threads.nthreads())
    println()
    _, K, tree_pca, _, _ = setup(first(NS))
    assemble_hss(K, tree_pca; rtol=RTOL)
    ulv(assemble_hss(K, tree_pca; rtol=RTOL))
    for n in NS
        profile_n(n; do_nested = false)
    end
    stack_sample(maximum(ns for ns in NS if ns <= 4096))
    return
end

main()
