#!/usr/bin/env julia
# Diagnose why NNCA is slower than NCA: dual-column sizes and phase times.
using LinearAlgebra, Random, StaticArrays, Printf, Statistics
using BEM
using BEM.HMatrices
const BH = BEM.HMatrices
using BEM.HMatrices: index_range, container, loc2glob, ClusterTree, KernelMatrix,
    GeometricSplitter, DyadicSplitter, assemble_h2, PartialACA, PermutedMatrix

function volume_grid(dim, nside)
    xs = range(0.0, 1.0; length=nside)
    dim == 2 && return [SVector(float(x), float(y)) for y in xs for x in xs]
    return [SVector(float(x), float(y), float(z)) for z in xs for y in xs for x in xs]
end

function cube_surface(ndiv)
    xs = range(0.0, 1.0; length=ndiv + 1)
    seen = Set{NTuple{3,Float64}}()
    pts = SVector{3,Float64}[]
    function pushpt!(x, y, z)
        k = (x, y, z)
        k in seen && return
        push!(seen, k)
        push!(pts, SVector(x, y, z))
        return
    end
    for y in xs, x in xs
        pushpt!(x, y, 0.0); pushpt!(x, y, 1.0)
    end
    for z in xs, x in xs
        pushpt!(x, 0.0, z); pushpt!(x, 1.0, z)
    end
    for z in xs, y in xs
        pushpt!(0.0, y, z); pushpt!(1.0, y, z)
    end
    return pts
end

function kerfun(dim)
    dim == 2 && return (x, y) -> (r = norm(x - y); r < 1e-30 ? 0.0 : log(r))
    inv4π = 1 / (4π)
    return (x, y) -> (r = norm(x - y); r < 1e-30 ? 0.0 : inv4π / r)
end

function setup(pts; splitter)
    tree = ClusterTree(copy(pts), splitter)
    # ClusterTree permutes `_elements` into local order — no second permute.
    K = KernelMatrix(kerfun(length(pts[1])), tree._elements, tree._elements)
    tidx = BH.H2TreeIndex(tree)
    _, far = BH._h2_block_partition(tidx, tidx, BH.H2BoxAdmissibility(0.5); symmetric=true)
    return tree, K, tidx, far
end

function dual_stats(label, pts; splitter, nca_dual=128)
    tree, K, tidx, far = setup(pts; splitter)
    nnode = length(tidx.nodes)
    il = BH._nnca_il_lists(tidx, tidx, far; side=:row, same=true)
    far_of = [Int[] for _ in 1:nnode]
    for (i, j) in far
        push!(far_of[i], j); push!(far_of[j], i)
    end
    skeleton = [Int[] for _ in 1:nnode]
    for i in tidx.leafnodes
        skeleton[i] = collect(index_range(tidx.nodes[i]))
    end
    pch = tidx.children
    pnodes = tidx.nodes

    n_il_empty = 0
    n_fallback = 0
    rows = NamedTuple[]
    for lvl in length(tidx.levels):-1:1
        ndual_nca = Int[]
        ndual_il = Int[]
        ndual_nn = Int[]
        ncand = Int[]
        n_il = Int[]
        n_far = Int[]
        for node in tidx.levels[lvl]
            isleaf = isempty(tidx.children[node])
            cand = isleaf ? skeleton[node] :
                reduce(vcat, (skeleton[c] for c in tidx.children[node]); init=Int[])
            isempty(cand) && continue
            push!(ncand, length(cand))
            push!(n_il, length(il[node]))
            push!(n_far, length(far_of[node]))

            # NCA dual
            dual_n = Int[]
            for p in far_of[node]
                if p <= length(skeleton) && !isempty(skeleton[p])
                    append!(dual_n, skeleton[p])
                elseif p <= length(pch) && !isempty(pch[p])
                    for c in pch[p]
                        append!(dual_n, skeleton[c])
                    end
                else
                    append!(dual_n, collect(index_range(pnodes[p])))
                end
            end
            unique!(dual_n)
            if length(dual_n) < nca_dual
                extra = BH._nca_outside_from_box(container(tidx.nodes[node]), tree; maxn=nca_dual)
                append!(dual_n, extra); unique!(dual_n)
            elseif length(dual_n) > nca_dual
                step = max(1, length(dual_n) ÷ nca_dual)
                dual_n = dual_n[1:step:end]
                length(dual_n) > nca_dual && (dual_n = dual_n[1:nca_dual])
            end
            push!(ndual_nca, length(dual_n))

            # NNCA IL-only dual
            dual_il = Int[]
            if isleaf
                for p in il[node]
                    append!(dual_il, collect(index_range(pnodes[p])))
                end
            else
                for p in il[node]
                    ch = p <= length(pch) ? pch[p] : Int[]
                    if isempty(ch)
                        append!(dual_il, collect(index_range(pnodes[p])))
                    else
                        for c in ch
                            append!(dual_il, skeleton[c])
                        end
                    end
                end
            end
            unique!(dual_il)
            push!(ndual_il, length(dual_il))
            isempty(il[node]) && (n_il_empty += 1)

            # NNCA after fallback
            dual = copy(dual_il)
            need = max(64, 2 * length(cand))
            fb = false
            if length(dual) < need
                fb = true
                n_fallback += 1
                for p in far_of[node]
                    if p <= length(skeleton) && !isempty(skeleton[p])
                        append!(dual, skeleton[p])
                    else
                        append!(dual, collect(index_range(pnodes[p])))
                    end
                end
                extra = BH._nca_outside_from_box(container(tidx.nodes[node]), tree; maxn=need)
                append!(dual, extra); unique!(dual)
            end
            push!(ndual_nn, length(dual))

            # advance skeleton with a dummy ID so next level sees realistic skel size
            # (use min(cand, 24) as a typical rank)
            rkeep = min(length(cand), 24)
            skeleton[node] = cand[1:rkeep]
        end
        isempty(ncand) && continue
        push!(rows, (;
            lvl, n=length(ncand),
            cand=mean(ncand),
            il=mean(n_il), far=mean(n_far),
            nca=mean(ndual_nca), nca_max=maximum(ndual_nca),
            il_dual=mean(ndual_il), il_max=maximum(ndual_il),
            nnca=mean(ndual_nn), nnca_max=maximum(ndual_nn),
        ))
    end
    nnodes = nnode
    @printf("\n== %s  N=%d  nodes=%d  fallback=%d (%.0f%%)  IL-empty=%d\n",
        label, length(pts), nnodes, n_fallback, 100 * n_fallback / max(1, nnodes), n_il_empty)
    @printf("  %4s %5s %6s %6s %6s %8s %8s %8s %8s\n",
        "lvl", "n", "cand", "|IL|", "|far|", "NCA", "IL-dual", "NNCA", "NNCA max")
    for r in reverse(rows)
        @printf("  %4d %5d %6.1f %6.1f %6.1f %8.1f %8.1f %8.1f %8d\n",
            r.lvl, r.n, r.cand, r.il, r.far, r.nca, r.il_dual, r.nnca, r.nnca_max)
    end
end

function time_phases(label, pts; splitter)
    tree, K, tidx, far = setup(pts; splitter)
    nnode = length(tidx.nodes)
    function bases!(method)
        U = [zeros(Float64, 0, 0) for _ in 1:nnode]
        sk = [Int[] for _ in 1:nnode]
        for i in tidx.leafnodes
            sk[i] = collect(index_range(tidx.nodes[i]))
        end
        if method === :nca
            BH._h2_build_bases_nca!(U, sk, tidx, tree, K, Float64, far;
                rtol=1e-6, rank=typemax(Int), nca_dual=128, pblk=1,
                side=:row, other=tidx, partner_skel=sk, partner_tree=tree)
        else
            BH._h2_build_bases_nnca!(U, sk, tidx, tree, K, Float64, far;
                rtol=1e-6, rank=typemax(Int),
                side=:row, other=tidx, partner_skel=sk, partner_tree=tree)
        end
        return U, sk
    end
    bases!(:nca); bases!(:nnca)
    tn = @elapsed bases!(:nca)
    tnn = @elapsed bases!(:nnca)
    tfull_n = @elapsed assemble_h2(K, tree; rtol=1e-6, far_method=:aca,
        basis_method=:nca, alpha=0.5, global_index=false,
        comp=PartialACA(; rtol=1e-6))
    tfull_nn = @elapsed assemble_h2(K, tree; rtol=1e-6, far_method=:aca,
        basis_method=:nnca, alpha=0.5, global_index=false,
        comp=PartialACA(; rtol=1e-6))
    @printf("  times %-20s  basis NCA=%.3fs NNCA=%.3fs   full NCA=%.3fs NNCA=%.3fs\n",
        label, tn, tnn, tfull_n, tfull_nn)
end

Random.seed!(1)
geo = BH.GeometricSplitter(; nmax=32)
dya = BH.DyadicSplitter(; nmax=32)

println("=== dual sizes (GeometricSplitter, binary tree) ===")
dual_stats("2D vol 64²", volume_grid(2, 64); splitter=geo)
dual_stats("3D vol 16³", volume_grid(3, 16); splitter=geo)
dual_stats("3D surf ndiv=24", cube_surface(24); splitter=dya)

println("\n=== dual sizes (DyadicSplitter, 2^d tree) ===")
dual_stats("2D vol 64² dyadic", volume_grid(2, 64); splitter=dya)
dual_stats("3D vol 16³ dyadic", volume_grid(3, 16); splitter=dya)

println("\n=== phase times ===")
time_phases("2D vol 64² geo", volume_grid(2, 64); splitter=geo)
time_phases("3D vol 16³ geo", volume_grid(3, 16); splitter=geo)
time_phases("2D vol 64² dyadic", volume_grid(2, 64); splitter=dya)
time_phases("3D vol 16³ dyadic", volume_grid(3, 16); splitter=dya)
println("done")
