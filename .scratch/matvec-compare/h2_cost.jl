# Diagnose H2 (NNCA) matvec complexity: ranks, IL, flops, phase times.
using DrWatson
@quickactivate :BEM
using LinearAlgebra
using Printf
using Random
using Statistics
using StaticArrays
using BEM.HMatrices

const NMAX = 32
const RTOL = 1e-6
const NSIDES = [32, 64, 128, 256]  # N = 1k, 4k, 16k, 65k

function _pts(n1d)
    xs = range(0.0, 1.0; length=n1d)
    return [SVector(float(x), float(y)) for y in xs for x in xs]
end

function stats(A::NNCAMatrix)
    nbox = length(A.boxes)
    nm2l = 0
    m2l_ent = 0
    il_sum = 0
    il_max = 0
    r_sum = 0
    r_max = 0
    r_n = 0
    l2p_ent = 0
    near_ent = 0
    nnear = 0
    nb_sum = 0
    nb_max = 0
    for (id, b) in enumerate(A.boxes)
        nm2l += length(b.M2L)
        iln = length(A.il[id])
        il_sum += iln
        il_max = max(il_max, iln)
        nbn = length(A.neighbors[id])
        nb_sum += nbn
        nb_max = max(nb_max, nbn)
        r = length(b.check)
        if r > 0 && !isempty(A.il[id])
            r_sum += r
            r_n += 1
            r_max = max(r_max, r)
        end
        l2p_ent += length(b.L2P)
        for M in values(b.M2L)
            m2l_ent += length(M)
        end
        near_ent += length(b.self)
        for M in values(b.near)
            near_ent += length(M)
            nnear += 1
        end
    end
    nlev = count(!isempty, A.levels)
    return (; nbox, nleaf=length(A.leaf_ids), nlev, nm2l, m2l_ent, l2p_ent, near_ent,
        nnear, il_mean = il_sum / max(nbox, 1), il_max,
        nb_mean = nb_sum / max(nbox, 1), nb_max,
        avg_rank = A.avg_rank, r_mean = r_n == 0 ? 0.0 : r_sum / r_n, r_max,
        mem = Base.summarysize(A))
end

function time_phases(A, x)
    xt = HMatrices._nnca_permute_in(x, A.perm, A.p)
    # warmup
    HMatrices._nnca_apply(A, xt)
    nrun = 7
    function med(f)
        for _ in 1:2
            f()
        end
        ts = Vector{Float64}(undef, nrun)
        for i in 1:nrun
            t0 = time_ns()
            f()
            ts[i] = (time_ns() - t0) / 1e9
        end
        return median(ts)
    end
    t_perm = med(() -> HMatrices._nnca_permute_in(x, A.perm, A.p))
    t_m2m = med(() -> HMatrices._nnca_m2m!(A, xt))
    t_m2l = med(() -> HMatrices._nnca_m2l!(A))
    t_l2l = med(() -> HMatrices._nnca_l2l!(A))
    t_near = med(() -> HMatrices._nnca_near!(A, xt))
    t_all = med(() -> HMatrices._nnca_apply(A, xt))
    t_mul = med(() -> mul!(similar(x), A, x))
    return (; t_perm, t_m2m, t_m2l, t_l2l, t_near, t_all, t_mul)
end

function main()
    Random.seed!(1)
    println("H2 cost breakdown  nmax=$NMAX rtol=$RTOL threads=$(Threads.nthreads())")
    println()
    @printf("%6s %5s %5s %4s %6s %6s %6s %8s %8s %8s %7s %7s %8s\n",
        "N", "nbox", "nlev", "ilMx", "r_avg", "r_max", "nm2l",
        "m2l_ent", "near_ent", "l2p_ent", "m2l/N", "near/N", "memMB")
    rows = []
    for n1d in NSIDES
        pts = _pts(n1d)
        n = length(pts)
        K = KernelMatrix(pts, pts) do a, b
            r = hypot(a[1] - b[1], a[2] - b[2])
            return r < 1e-30 ? 0.0 : log(r)
        end
        tree = ClusterTree(pts, DyadicSplitter(; nmax=NMAX, tight=false); cube=true)
        A = assemble_h2(K, tree; rtol=RTOL, threads=true)
        s = stats(A)
        x = randn(n)
        t = time_phases(A, x)
        @printf("%6d %5d %5d %4d %6.1f %6d %6d %8.2e %8.2e %8.2e %7.1f %7.1f %8.1f\n",
            n, s.nbox, s.nlev, s.il_max, s.avg_rank, s.r_max, s.nm2l,
            float(s.m2l_ent), float(s.near_ent), float(s.l2p_ent),
            s.m2l_ent / n, s.near_ent / n, s.mem / 1024^2)
        push!(rows, (; n, s..., t...))
        flush(stdout)
    end
    println()
    println("phase times [ms] and per-N [ns]")
    @printf("%6s %8s %8s %8s %8s %8s %8s %8s\n",
        "N", "perm", "m2m", "m2l", "l2l", "near", "apply", "mul!")
    for r in rows
        @printf("%6d %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f\n",
            r.n, 1e3*r.t_perm, 1e3*r.t_m2m, 1e3*r.t_m2l, 1e3*r.t_l2l,
            1e3*r.t_near, 1e3*r.t_all, 1e3*r.t_mul)
    end
    println()
    @printf("%6s %8s %8s %8s %8s %8s %8s %8s\n",
        "N", "perm/N", "m2m/N", "m2l/N", "l2l/N", "near/N", "app/N", "mul/N")
    for r in rows
        @printf("%6d %8.1f %8.1f %8.1f %8.1f %8.1f %8.1f %8.1f\n",
            r.n, 1e9*r.t_perm/r.n, 1e9*r.t_m2m/r.n, 1e9*r.t_m2l/r.n,
            1e9*r.t_l2l/r.n, 1e9*r.t_near/r.n, 1e9*r.t_all/r.n, 1e9*r.t_mul/r.n)
    end
    println()
    println("flop-like entries / N  (O(1) => linear storage)")
    for r in rows
        tot = r.m2l_ent + r.near_ent + r.l2p_ent
        @printf("  N=%6d  tot/N=%6.1f  m2l/N=%5.1f  near/N=%5.1f  l2p/N=%5.1f  r_avg=%.1f\n",
            r.n, tot/r.n, r.m2l_ent/r.n, r.near_ent/r.n, r.l2p_ent/r.n, r.avg_rank)
    end
    if length(rows) >= 2
        a, b = rows[end-1], rows[end]
        αt = log(b.t_mul / a.t_mul) / log(b.n / a.n)
        αf = log((b.m2l_ent + b.near_ent) / (a.m2l_ent + a.near_ent)) / log(b.n / a.n)
        println()
        @printf("last-two-point  α_time=%.2f  α_flops=%.2f  (1 = linear)\n", αt, αf)
    end
    return rows
end

main()
