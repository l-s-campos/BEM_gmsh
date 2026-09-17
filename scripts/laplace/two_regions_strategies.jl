# Compare dense / Kane noncondensing / Kane condensation on two-region Laplace.
#
#   julia --project=. scripts/laplace/two_regions_strategies.jl
#
# Analytic 1-D conduction: T(0)=0, T(1)=1, kL=1, kR=4, interface x=0.5.

using DrWatson
@quickactivate :BEM
using LinearAlgebra
using Statistics
using Printf
using BEM.MultiRegion
include(datadir("Laplace", "two_regions.jl"))

const STRATS = (:dense, :noncondensing, :condense)

function _rel(dad, kL, kR)
    Tex = [ana_two_layer_T(dad.Nodes[i][1], kL, kR) for i in 1:dad.n]
    return norm(dad.T[1:dad.n] .- Tex) / max(norm(Tex), 1e-12)
end

function _jump(prob)
    j = 0.0
    for ip in prob.interfaces
        Ta = prob.regions[ip.reg_a].T[ip.node_a]
        Tb = prob.regions[ip.reg_b].T[ip.node_b]
        j = max(j, abs(Ta - Tb))
    end
    return j
end

function _flop_est(prob, strategy)
    nts = [dad.nt for dad in prob.regions]
    n_if = length(prob.interfaces)
    if strategy === :dense
        N = sum(nts) + n_if
        return N^3
    elseif strategy === :noncondensing
        return sum(n -> n^3, nts) + n_if^3
    else
        cost = (2 * n_if)^3
        for (r, dad) in enumerate(prob.regions)
            nloc = count(ip -> ip.reg_a == r || ip.reg_b == r, prob.interfaces)
            cost += (dad.nt - nloc)^3
        end
        return cost
    end
end

function run_mesh(ndiv; kL=1.0, kR=4.0, npg=12, nwarm=1, nrep=3)
    msh = mesh_two_regions(ndiv=ndiv, show=false, nome="two_reg_strat_$ndiv")
    prob = load_two_regions(msh, Laplace(kL), Laplace(kR))
    pair_interfaces!(prob)
    assemble_multiregion(prob; npg=npg)
    n_if = length(prob.interfaces)
    nts = [dad.nt for dad in prob.regions]
    println()
    println("ndiv=$ndiv  nL=$(nts[1])  nR=$(nts[2])  n_if=$n_if")
    @printf("  %-16s %8s %12s %10s %10s %10s %10s %12s\n",
        "strategy", "N", "flops~", "t[s]", "relL", "relR", "jump", "‖T−Tdens‖")
    Tref = nothing
    # mix BCs once; timed path is the linear algebra only
    solve_multiregion!(prob; strategy=:dense, apply_bc=true)
    for strat in STRATS
        for _ in 1:nwarm
            solve_multiregion!(prob; strategy=strat, apply_bc=false)
        end
        t = Inf
        for _ in 1:nrep
            t = min(t, @elapsed solve_multiregion!(prob; strategy=strat, apply_bc=false))
        end
        relL = _rel(prob.regions[1], kL, kR)
        relR = _rel(prob.regions[2], kL, kR)
        j = _jump(prob)
        Ts = vcat((dad.T[1:dad.n] for dad in prob.regions)...)
        if Tref === nothing
            Tref = Ts
            dref = 0.0
        else
            dref = norm(Ts .- Tref) / max(norm(Tref), 1e-16)
        end
        N = multiregion_ndof(prob; strategy=strat)
        @printf("  %-16s %8d %12.3e %10.4f %10.2e %10.2e %10.2e %12.2e\n",
            strat, N, float(_flop_est(prob, strat)), t, relL, relR, j, dref)
    end
end

println("="^72)
println(" Multi-region Laplace: dense vs Kane noncondensing vs condensation")
println(" kL=1, kR=4  (T_if exact = 0.8)")
println("="^72)
run_mesh(8; nwarm=1, nrep=2)
run_mesh(16; nwarm=1, nrep=3)
run_mesh(32; nwarm=1, nrep=3)
run_mesh(64; nwarm=1, nrep=3)
println("done")
