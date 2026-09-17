# Profile NNCA H² assembly phases (tree, IL, ACA skeletons, M2L, near).
using DrWatson
@quickactivate :BEM
using LinearAlgebra
using Printf
using Random
using StaticArrays
using BEM.HMatrices

const NMAX = 32
const RTOL = 1e-6
const NSIDES = [64, 128, 256]  # 4k, 16k, 65k

function _pts(n1d)
    xs = range(0.0, 1.0; length=n1d)
    return [SVector(float(x), float(y)) for y in xs for x in xs]
end

function _kernel(pts)
    return KernelMatrix(pts, pts) do a, b
        r = hypot(a[1] - b[1], a[2] - b[2])
        return r < 1e-30 ? 0.0 : log(r)
    end
end

function assemble_timed(pts; threads=true)
    n = length(pts)
    K = _kernel(pts)
    t_tree = @elapsed tree = ClusterTree(pts, DyadicSplitter(; nmax=NMAX, tight=false); cube=true)
    HMatrices.node_id(tree) == 0 && HMatrices.assign_node_ids!(tree)
    t_il = @elapsed begin
        neighbors, il, id2node = HMatrices.neighbor_il_lists(tree)
    end
    nn = HMatrices.nnodes(tree)
    boxes = [HMatrices.NNCABox{Float64}() for _ in 1:nn]
    perm = copy(HMatrices.loc2glob(tree))
    Kg = HMatrices.PermutedMatrix(K, perm, perm)
    levels_nodes = HMatrices.nodes_by_depth(tree)
    dmax = length(levels_nodes) - 1
    t_sk_levels = Float64[]
    n_sk_levels = Int[]
    t_sk = @elapsed begin
        for d in dmax:-1:2
            lev = levels_nodes[d + 1]
            t = @elapsed begin
                if threads && Threads.nthreads() > 1 && length(lev) >= 8
                    Threads.@threads for k in eachindex(lev)
                        HMatrices._nnca_get_nodes!(boxes, lev[k], Kg, il, id2node,
                            Float64, 1; rtol=RTOL, rank=typemax(Int))
                    end
                else
                    for node in lev
                        HMatrices._nnca_get_nodes!(boxes, node, Kg, il, id2node,
                            Float64, 1; rtol=RTOL, rank=typemax(Int))
                    end
                end
            end
            push!(t_sk_levels, t)
            push!(n_sk_levels, length(lev))
        end
        if dmax >= 1
            for node in levels_nodes[2]
                HMatrices._nnca_shallow_basis!(boxes, node, Float64, 1)
            end
        end
    end
    t_m2l = @elapsed HMatrices._nnca_assemble_m2l!(boxes, il, Kg, Float64, 1; threads=threads)
    leaf_ids = [HMatrices.node_id(L) for L in HMatrices.leaves(tree)]
    t_near = @elapsed HMatrices._nnca_assemble_near!(boxes, neighbors, id2node, Kg, Float64, 1, leaf_ids; threads=threads)
    t_tot = t_tree + t_il + t_sk + t_m2l + t_near
    return (; n, t_tree, t_il, t_sk, t_m2l, t_near, t_tot, t_sk_levels, n_sk_levels, dmax, nn)
end

function main()
    Random.seed!(1)
    println("H2 assembly profile  nmax=$NMAX rtol=$RTOL threads=$(Threads.nthreads())")
    println()
    # warmup
    assemble_timed(_pts(32); threads=true)
    println("threaded ACA within each level (siblings independent)")
    @printf("%7s %8s %8s %8s %8s %8s %8s\n",
        "N", "tree", "IL", "ACA", "M2L", "near", "total")
    rows = []
    for n1d in NSIDES
        r = assemble_timed(_pts(n1d); threads=true)
        @printf("%7d %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f\n",
            r.n, r.t_tree, r.t_il, r.t_sk, r.t_m2l, r.t_near, r.t_tot)
        print("         ACA by level (fine→coarse):")
        for (t, n) in zip(r.t_sk_levels, r.n_sk_levels)
            @printf("  d n=%d %.3fs", n, t)
        end
        println()
        push!(rows, r)
        flush(stdout)
    end
    println()
    println("serial ACA (package default) vs threaded ACA, N=$(rows[end].n)")
    pts = _pts(NSIDES[end])
    # serial skeletons: call package assemble_h2
    t_pkg = @elapsed assemble_h2(_kernel(pts),
        ClusterTree(pts, DyadicSplitter(; nmax=NMAX, tight=false); cube=true);
        rtol=RTOL, threads=true)
    println("  assemble_h2 (serial ACA, threaded M2L/near)  $(round(t_pkg; digits=3)) s")
    rth = assemble_timed(pts; threads=true)
    println("  this script (threaded ACA + M2L/near)        $(round(rth.t_tot; digits=3)) s")
    rser = assemble_timed(pts; threads=false)
    println("  this script fully serial                     $(round(rser.t_tot; digits=3)) s")
    println()
    @printf("  serial  ACA=%.3f  M2L=%.3f  near=%.3f\n", rser.t_sk, rser.t_m2l, rser.t_near)
    @printf("  thread  ACA=%.3f  M2L=%.3f  near=%.3f\n", rth.t_sk, rth.t_m2l, rth.t_near)
    return
end

main()
