# Stage-level Laplace FMM profile (2D log / 3D 1/(4πr)).
#
#   julia --project=. -t auto scripts/profile/profile_fmm.jl
#
# ENV:
#   FMM_N2=2048,8192,32768
#   FMM_N3=1024,4096,16384
#   FMM_EPS=1e-8
#   FMM_NRUN=7

using DrWatson
@quickactivate :BEM
using LinearAlgebra
using Printf
using Random
using Statistics
using Profile
using BEM.FMM
using BEM.HMatrices: nnodes, isleaf, leaves

const N2 = parse.(Int, split(get(ENV, "FMM_N2", "2048,8192,32768"), ','; keepempty=false))
const N3 = parse.(Int, split(get(ENV, "FMM_N3", "1024,4096,16384"), ','; keepempty=false))
const EPS = parse(Float64, get(ENV, "FMM_EPS", "1e-8"))
const NRUN = parse(Int, get(ENV, "FMM_NRUN", "7"))

function _med(f; n=NRUN, w=2)
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

fmtb(b) = b < 1024 ? @sprintf("%d B", b) :
          b < 1024^2 ? @sprintf("%.1f KiB", b / 1024) :
          @sprintf("%.2f MiB", b / 1024^2)

function _zero3d!(plan)
    fill!(plan.pot_local, 0)
    for ed in plan.sdata
        fill!(ed.multipole, 0)
        fill!(ed.localexp, 0)
    end
    return nothing
end

function profile2d(n)
    rng = Random.default_rng()
    Random.seed!(rng, 1)
    P = rand(rng, 2, n)
    q = randn(rng, n)
    y = zeros(n)
    t_plan, _ = _med(() -> FMM.build_laplace2d_plan(P; eps=EPS); n=3, w=1)
    plan = FMM.build_laplace2d_plan(P; eps=EPS)
    t_apply, tmin = _med(() -> FMM.apply_laplace2d!(plan, y; charges=q))
    a_apply = _alloc(() -> FMM.apply_laplace2d!(plan, y; charges=q))

    FMM._permute_to_local!(plan.dens_loc, q, plan.sl2g)
    function far_up()
        fill!(plan.pot_local, 0)
        FMM._zero_multipoles!(plan.sdata)
        FMM._zero_locals!(plan.sdata)
        FMM._upward_plan!(plan, plan.dens_loc, nothing)
    end
    t_up, _ = _med(far_up)
    t_m2l, _ = _med(() -> FMM._m2l_plan_2d!(plan))
    t_p2p, _ = _med(() -> FMM._p2p_plan_2d!(plan, plan.dens_loc, nothing, nothing, nothing))
    t_dn, _ = _med(() -> FMM._downward_plan!(plan))
    a_m2l = _alloc(() -> FMM._m2l_plan_2d!(plan))
    a_p2p = _alloc(() -> FMM._p2p_plan_2d!(plan, plan.dens_loc, nothing, nothing, nothing))

    return (; dim=2, n, p=plan.nterms, nmax=nothing,
        nnodes=nnodes(plan.tree), nleaf=length(plan.leaf_nodes),
        n_m2l=length(plan.m2l_jobs), n_p2p=length(plan.p2p_jobs),
        t_plan, t_apply, tmin, a_apply, t_up, t_m2l, t_p2p, t_dn, a_m2l, a_p2p)
end

function profile3d(n)
    rng = Random.default_rng()
    Random.seed!(rng, 2)
    P = rand(rng, 3, n)
    q = randn(rng, n)
    y = zeros(n)
    t_plan, _ = _med(() -> FMM.build_laplace3d_plan(P; eps=EPS); n=3, w=1)
    plan = FMM.build_laplace3d_plan(P; eps=EPS)
    t_apply, tmin = _med(() -> FMM.apply_laplace3d!(plan, y; charges=q))
    a_apply = _alloc(() -> FMM.apply_laplace3d!(plan, y; charges=q))

    FMM._permute_to_local!(plan.dens_loc, q, plan.sl2g)
    ch = plan.dens_loc
    function up()
        _zero3d!(plan)
        FMM._upward_octree!(plan, ch)
    end
    t_up, _ = _med(up)
    t_m2l, _ = _med(() -> FMM._m2l_octree!(plan))
    t_dn, _ = _med(() -> FMM._l2p_octree!(plan, nothing))
    t_p2p, _ = _med(() -> FMM._p2p_plan!(plan, ch, nothing))
    a_m2l = _alloc(() -> FMM._m2l_octree!(plan))
    a_p2p = _alloc(() -> FMM._p2p_plan!(plan, ch, nothing))
    a_up = _alloc(up)

    K = (plan.p + 1)^2
    return (; dim=3, n, p=plan.p, nmax=plan.nmax, K,
        nnodes=nnodes(plan.stree), nleaf=length(plan.leaf_nodes),
        n_m2l=length(plan.m2l_jobs), n_p2p=length(plan.p2p_jobs),
        t_plan, t_apply, tmin, a_apply, t_up, t_m2l, t_p2p, t_dn, a_m2l, a_p2p, a_up)
end

function _row2(r)
    tot = r.t_up + r.t_m2l + r.t_dn + r.t_p2p
    @printf("2D %7d  p=%2d  nodes=%5d leaf=%4d  M2L=%6d P2P=%6d\n",
        r.n, r.p, r.nnodes, r.nleaf, r.n_m2l, r.n_p2p)
    @printf("    plan %7.2f ms   apply %7.2f ms  alloc %s\n",
        1e3 * r.t_plan, 1e3 * r.t_apply, fmtb(r.a_apply))
    @printf("    P2M/M2M %6.1f%%  %6.2f ms\n", 100 * r.t_up / tot, 1e3 * r.t_up)
    @printf("    M2L     %6.1f%%  %6.2f ms  (%d jobs, %.1f µs/job, alloc %s)\n",
        100 * r.t_m2l / tot, 1e3 * r.t_m2l, r.n_m2l,
        1e6 * r.t_m2l / max(r.n_m2l, 1), fmtb(r.a_m2l))
    @printf("    L2L/L2P %6.1f%%  %6.2f ms\n", 100 * r.t_dn / tot, 1e3 * r.t_dn)
    @printf("    P2P     %6.1f%%  %6.2f ms  (%d jobs, alloc %s)\n",
        100 * r.t_p2p / tot, 1e3 * r.t_p2p, r.n_p2p, fmtb(r.a_p2p))
    println()
end

function _row3(r)
    tot = r.t_up + r.t_m2l + r.t_dn + r.t_p2p
    @printf("3D %7d  p=%2d K=%3d  nodes=%5d leaf=%4d  M2L=%6d P2P=%6d\n",
        r.n, r.p, r.K, r.nnodes, r.nleaf, r.n_m2l, r.n_p2p)
    @printf("    plan %7.2f ms   apply %7.2f ms  alloc %s\n",
        1e3 * r.t_plan, 1e3 * r.t_apply, fmtb(r.a_apply))
    @printf("    P2M/M2M %6.1f%%  %6.2f ms  (alloc %s)\n",
        100 * r.t_up / tot, 1e3 * r.t_up, fmtb(r.a_up))
    @printf("    M2L     %6.1f%%  %6.2f ms  (%d jobs, %.1f µs/job, alloc %s)\n",
        100 * r.t_m2l / tot, 1e3 * r.t_m2l, r.n_m2l,
        1e6 * r.t_m2l / max(r.n_m2l, 1), fmtb(r.a_m2l))
    @printf("    L2L/L2P %6.1f%%  %6.2f ms\n", 100 * r.t_dn / tot, 1e3 * r.t_dn)
    @printf("    P2P     %6.1f%%  %6.2f ms  (%d jobs, alloc %s)\n",
        100 * r.t_p2p / tot, 1e3 * r.t_p2p, r.n_p2p, fmtb(r.a_p2p))
    println()
end

function stack_sample()
    n = 4096
    rng = Random.default_rng()
    Random.seed!(rng, 3)
    P = rand(rng, 3, n)
    q = randn(rng, n)
    y = zeros(n)
    plan = FMM.build_laplace3d_plan(P; eps=EPS)
    FMM.apply_laplace3d!(plan, y; charges=q)
    Profile.clear()
    Profile.init(; delay=0.001)
    @profile for _ in 1:8
        FMM.apply_laplace3d!(plan, y; charges=q)
    end
    println("Profile sample (3D N=$n apply × 8, delay=1 ms)")
    Profile.print(IOContext(stdout, :displaysize => (30, 120));
        C=false, combine=true, mincount=5, noisefloor=2, maxdepth=18)
    println()
end

function main()
    println("Laplace FMM stage profile")
    println("eps=", EPS, "  threads=", Threads.nthreads(), "  nrun=", NRUN)
    println()
    for n in N2
        _row2(profile2d(n))
    end
    for n in N3
        _row3(profile3d(n))
    end
    stack_sample()
    return
end

main()
