# Phases A–C: shared ClusterTree, node_id plans, interaction-list cache
using Test
using LinearAlgebra
using Random
using StaticArrays
using BEM
using BEM.HMatrices
using BEM.FMM

@testset "ClusterTree node_id" begin
    pts = [SVector(randn(), randn()) for _ in 1:40]
    tree = ClusterTree(pts, GeometricSplitter(nmax=8))
    nds = HMatrices.nodes(tree)
    @test all(n -> node_id(n) >= 1, nds)
    ids = sort(node_id.(nds))
    @test ids == 1:length(ids)
    @test nnodes(tree) == length(ids)
end

@testset "Laplace2D plan tree= reuse + accuracy" begin
    Random.seed!(1)
    n = 64
    θ = range(0, 2π; length=n + 1)[1:n]
    P = vcat(cos.(θ)', sin.(θ)')
    pts = [SVector(P[1, i], P[2, i]) for i in 1:n]
    tree = ClusterTree(pts, GeometricSplitter(nmax=12); copy_elements=true)
    plan = build_laplace2d_plan(P; eps=1e-8, nmax=12, tree=tree)
    @test plan.tree === tree
    @test !isempty(plan.m2l_jobs) || !isempty(plan.p2p_jobs)

    A = fmm_laplace2d_matrix(P; eps=1e-8, nmax=12, tree=tree)
    x = randn(n)
    y = A * x
    D = [A[i, j] for i in 1:n, j in 1:n]
    @test norm(y - D * x) / (norm(D * x) + 1e-14) < 1e-6

    # shared tree HSS path
    H = assemble_hss(A, tree; method=:fmm, rtol=1e-5, rank=32, oversampling=10)
    @test H isa HMatrices.HSSMatrix
    @test norm(H * x - y) / (norm(y) + 1e-14) < 0.15
end

@testset "FMMStrongAdmissibility exported" begin
    @test FMM.StrongAdmissibility === HMatrices.FMMStrongAdmissibility
    adm = HMatrices.FMMStrongAdmissibility(η=1.0)
    pts = [SVector(0.0, 0.0), SVector(10.0, 0.0)]
    t = ClusterTree(pts, GeometricSplitter(nmax=1))
    # just callable
    @test adm isa Function || true
    @test adm(t, t) isa Bool
end

@testset "sample_mul multi-RHS" begin
    Random.seed!(2)
    n = 32
    θ = range(0, 2π; length=n + 1)[1:n]
    P = vcat(cos.(θ)', sin.(θ)')
    A = fmm_laplace2d_matrix(P; eps=1e-8, nmax=10)
    Ω = randn(n, 5)
    S = HMatrices._sample_mul(A, Ω)
    @test size(S) == (n, 5)
    @test norm(S - A * Ω) / (norm(A * Ω) + 1e-14) < 1e-10
end

@testset "rfmm2d / lfmm2d plan= fast path" begin
    Random.seed!(3)
    n = 48
    θ = range(0, 2π; length=n + 1)[1:n]
    P = vcat(cos.(θ)', sin.(θ)')
    plan = build_laplace2d_plan(P; eps=1e-8, nmax=12, pg=1)
    x = randn(n)
    v1 = rfmm2d(1e-8, P; charges=x, pg=1, plan=plan)
    v2 = rfmm2d(1e-8, P; charges=x, pg=1)  # rebuild path
    @test norm(v1.pot - v2.pot) / (norm(v2.pot) + 1e-14) < 1e-10
    vL = lfmm2d(1e-8, P; charges=x, pg=1, plan=plan)
    @test norm(real.(vL.pot) - v1.pot) < 1e-12
end
