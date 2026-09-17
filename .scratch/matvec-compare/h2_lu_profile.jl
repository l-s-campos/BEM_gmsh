# Profile nested H² LR (assemble → h2node → lu → solve)
using DrWatson
@quickactivate :BEM
using LinearAlgebra
using Printf
using Random
using Profile
using StaticArrays
using Statistics
using BEM.HMatrices
import BEM.HMatrices: h2_foreach, isdense_h2, isuniform, h2node

const NMAX = 32
const RTOL = 1e-6
const NSIDES = [16, 32, 48]  # N = 256, 1024, 2304

function _pts(n1d)
    xs = range(0.0, 1.0; length=n1d)
    return [SVector(float(x), float(y)) for y in xs for x in xs]
end

function _kernel(pts)
    return KernelMatrix(pts, pts) do a, b
        r = hypot(a[1] - b[1], a[2] - b[2])
        return r < 1e-30 ? 4.0 : log(r)
    end
end

function _spd(pts)
    K = _kernel(pts)
    Kd = Matrix(K)
    return Kd + Kd' + 8.0 * I
end

function _count(N::H2Node)
    n_d = 0
    n_u = 0
    n_sf = 0
    n_s = 0
    HMatrices.h2_foreach(N) do node
        if isdense_h2(node)
            n_d += 1
        elseif isuniform(node)
            n_u += 1
            node.s_full && (n_sf += 1)
        else
            n_s += 1
        end
    end
    return (; n_d, n_u, n_sf, n_s)
end

function run_one(n1d; do_profile=false)
    Random.seed!(1)
    pts = _pts(n1d)
    n = length(pts)
    tree = ClusterTree(pts, DyadicSplitter(; nmax=NMAX, tight=false); cube=true)
    t_asm = @elapsed H2 = assemble_h2(_kernel(pts), tree; rtol=RTOL, threads=true)
    t_node = @elapsed root = h2node(H2)
    c0 = _count(root)
    x = randn(n)
    t_mv = median([(@elapsed mul!(similar(x), H2, x)) for _ in 1:7])
    t_mvn = median([(@elapsed mul!(similar(x), root, x)) for _ in 1:5])
    b = randn(n)
    Kd = _spd(pts)
    tree2 = ClusterTree(pts, DyadicSplitter(; nmax=NMAX, tight=false); cube=true)
    A2 = assemble_h2(Kd, tree2; rtol=1e-8, threads=true)
    t_lu = NaN
    t_sol = NaN
    rel = NaN
    c1 = (n_d=0, n_u=0, n_sf=0, n_s=0)
    try
        t_lu = @elapsed F = lu(A2; method=:nested, rtol=RTOL)
        c1 = _count(F.factors)
        t_sol = median([(@elapsed F \ copy(b)) for _ in 1:5])
        u = F \ copy(b)
        rel = norm(A2 * u - b) / (norm(b) + 1e-14)
        if do_profile
            Profile.clear()
            @profile lu(A2; method=:nested, rtol=RTOL)
            println("\n--- Profile.lu nested N=$n (C=true, mincount=20) ---")
            Profile.print(IOContext(stdout, :displaysize => (50, 160));
                C=true, mincount=20, noisefloor=2.0)
        end
    catch e
        @warn "lu failed at n=$n" exception = e
    end
    return (; n, t_asm, t_node, t_mv, t_mvn, t_lu, t_sol, rel, c0, c1)
end

function main()
    println("H² nested LR profile  nmax=$NMAX rtol=$RTOL threads=$(Threads.nthreads())")
    println()
    run_one(8)  # warmup
    @printf("%6s %8s %8s %8s %8s %8s %8s %8s  %s\n",
        "N", "asm", "h2node", "mv_H2", "mv_N", "lu", "solve", "resid",
        "after LU: dense/unif/s_full/split")
    rows = []
    for n1d in NSIDES
        r = run_one(n1d)
        @printf("%6d %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.1e  %d/%d/%d/%d\n",
            r.n, r.t_asm, r.t_node, r.t_mv, r.t_mvn, r.t_lu, r.t_sol, r.rel,
            r.c1.n_d, r.c1.n_u, r.c1.n_sf, r.c1.n_s)
        println("         before LU: dense=$(r.c0.n_d) unif=$(r.c0.n_u) s_full=$(r.c0.n_sf) split=$(r.c0.n_s)")
        push!(rows, r)
        flush(stdout)
    end
    if length(rows) >= 2
        a, b = rows[end-1], rows[end]
        α = log(b.t_lu / a.t_lu) / log(b.n / a.n)
        @printf("\nlast-two-point α_lu = %.2f  (1 = linear)\n", α)
    end
    println()
    run_one(NSIDES[end]; do_profile=true)
    return
end

main()
