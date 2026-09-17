# Two-region Laplace: perfect interface, equal and contrasting k.
using Test
using LinearAlgebra
using Statistics
using DrWatson
@quickactivate :BEM
using BEM.MultiRegion

include(datadir("Laplace", "two_regions.jl"))

function _rel_T(dad, kL, kR)
    Tex = [ana_two_layer_T(dad.Nodes[i][1], kL, kR) for i in 1:dad.n]
    return norm(dad.T[1:dad.n] .- Tex) / max(norm(Tex), 1e-12)
end

function _max_interface_jump(prob)
    j = 0.0
    for ip in prob.interfaces
        Ta = prob.regions[ip.reg_a].T[ip.node_a]
        Tb = prob.regions[ip.reg_b].T[ip.node_b]
        j = max(j, abs(Ta - Tb))
    end
    return j
end

@testset "two-region Laplace T=x (k=1/k=1)" begin
    msh = mesh_two_regions(ndiv=6, show=false, nome="t_2reg")
    prob = load_two_regions(msh, Laplace(1.0))
    pair_interfaces!(prob)
    assemble_multiregion(prob; npg=10)
    solve_multiregion!(prob)
    dadL = prob.regions[1]
    err = norm([dadL.T[i] - dadL.Nodes[i][1] for i in 1:dadL.n]) /
          max(norm([dadL.Nodes[i][1] for i in 1:dadL.n]), 1e-12)
    @test err < 0.15
end

@testset "two-region Laplace contrast k=1 / k=4" begin
    kL, kR = 1.0, 4.0
    msh = mesh_two_regions(ndiv=10, show=false, nome="t_2reg_k14")
    prob = load_two_regions(msh, Laplace(kL), Laplace(kR))
    @test prob.regions[1].properties.k == kL
    @test prob.regions[2].properties.k == kR
    pair_interfaces!(prob)
    @test !isempty(prob.interfaces)
    assemble_multiregion(prob; npg=12)
    solve_multiregion!(prob)
    dadL, dadR = prob.regions
    @test all(isfinite, dadL.T)
    @test all(isfinite, dadR.T)
    @test _rel_T(dadL, kL, kR) < 0.08
    @test _rel_T(dadR, kL, kR) < 0.08
    @test _max_interface_jump(prob) < 0.05
    # interface T = kR/(kL+kR) = 0.8 (homogeneous would be 0.5)
    if_idx = findall(i -> abs(dadL.Nodes[i][1] - 0.5) < 1e-8, 1:dadL.n)
    Tif = median(dadL.T[if_idx])
    @test abs(Tif - kR / (kL + kR)) < 0.05
end

@testset "multi-region strategies agree (k=1/k=4)" begin
    kL, kR = 1.0, 4.0
    msh = mesh_two_regions(ndiv=8, show=false, nome="t_2reg_strat")
    prob = load_two_regions(msh, Laplace(kL), Laplace(kR))
    pair_interfaces!(prob)
    assemble_multiregion(prob; npg=12)
    @test multiregion_ndof(prob; strategy=:dense) ==
          multiregion_ndof(prob; strategy=:noncondensing)
    @test multiregion_ndof(prob; strategy=:condense) == 2 * length(prob.interfaces)
    @test multiregion_ndof(prob; strategy=:condense) <
          multiregion_ndof(prob; strategy=:dense)

    solve_multiregion!(prob; strategy=:dense)
    Td = [copy(dad.T) for dad in prob.regions]
    qd = [copy(dad.q) for dad in prob.regions]

    for strat in (:noncondensing, :condense, :blocked, :condensation)
        solve_multiregion!(prob; strategy=strat)
        for r in 1:2
            @test prob.regions[r].T ≈ Td[r] rtol=1e-8 atol=1e-8
            @test prob.regions[r].q ≈ qd[r] rtol=1e-7 atol=1e-8
        end
        @test _rel_T(prob.regions[1], kL, kR) < 0.08
        @test _rel_T(prob.regions[2], kL, kR) < 0.08
        @test _max_interface_jump(prob) < 1e-8
    end
end
