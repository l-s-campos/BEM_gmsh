# Nested H² assembled from FMM matvecs (HARA-H²)
using Test
using LinearAlgebra
using BEM

@testset "assemble_h2_fmm laplace2d" begin
    nside = 10
    xs = range(0.0, 1.0; length=nside)
    pts = [SVector(float(x), float(y)) for y in xs for x in xs]
    n = length(pts)
    Pmat = Matrix{Float64}(undef, 2, n)
    @inbounds for j in 1:n
        Pmat[1, j] = pts[j][1]
        Pmat[2, j] = pts[j][2]
    end
    tree = ClusterTree(pts, PrincipalComponentSplitter(; nmax=16))
    Afmm = FMM.fmm_laplace2d_matrix(Pmat; eps=1e-6, nmax=40)
    H2 = assemble_h2_fmm(Afmm, tree; rtol=1e-3, nsample=48, alpha=0.5, rank=32)
    @test H2 isa H2Matrix
    @test size(H2) == (n, n)
    x = randn(n)
    yH = H2 * x
    yF = Afmm * x
    rel = norm(yH - yF) / (norm(yF) + 1e-14)
    @test rel < 0.2
end

@testset "assemble_h2_fmm one-shot points API" begin
    nside = 8
    xs = range(0.0, 1.0; length=nside)
    P = zeros(2, nside * nside)
    k = 0
    for y in xs, x in xs
        k += 1
        P[1, k] = x
        P[2, k] = y
    end
    H2 = assemble_h2_fmm(P; kernel=:laplace2d, rtol=1e-3, nsample=40, nmax=24, rank=24)
    @test H2 isa H2Matrix
    @test size(H2, 1) == size(P, 2)
end

@testset "DIBEM H2 + hss_method=:fmm" begin
    include(joinpath(@__DIR__, "..", "data", "Laplace", "Laplace_dad.jl"))
    msh = Base.invokelatest(quadrado; ndiv=6, show=false, nome="test_h2fmm_dibem")
    dad = format2d(msh, Laplace(1.0); tipo=1, pontointerno=false)
    xs = range(0.3, 0.7; length=3)
    set_internal_nodes!(dad, [SVector(float(x), float(y)) for y in xs for x in xs])
    Md = DIBEM(dad; method=:dense)
    dad2 = format2d(msh, Laplace(1.0); tipo=1, pontointerno=false)
    set_internal_nodes!(dad2, [SVector(float(x), float(y)) for y in xs for x in xs])
    Mh = DIBEM(dad2; method=:h2, hss_method=:fmm, nmax=16, rtol=1e-4, f_method=:dense)
    @test dad2.dibem_D isa H2Matrix
    x = randn(size(Md, 1))
    @test norm(Md * x - Mh * x) / (norm(Md * x) + 1e-14) < 0.5
end
