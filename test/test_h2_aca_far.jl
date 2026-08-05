# H² far blocks via on-the-fly ACA (near always dense; never fill full far dense first)
using Test
using LinearAlgebra
using BEM

@testset "assemble_h2 far_method=:aca" begin
    pts = [SVector(float(x), float(y)) for y in range(0, 1; length=12) for x in range(0, 1; length=12)]
    tree = ClusterTree(pts, PrincipalComponentSplitter(; nmax=20))
    K = KernelMatrix((x, y) -> (x == y ? 0.0 : log(norm(x - y))), pts, pts)

    Haca = assemble_h2(K, tree; rtol=1e-6, far_method=:aca,
        comp=PartialACA(; rtol=1e-6), alpha=0.5)
    Hden = assemble_h2(K, tree; rtol=1e-6, far_method=:dense, alpha=0.5)

    @test Haca isa HMatrices.H2Matrix
    @test !isempty(Haca.Bfar)
    @test all(v -> v isa HMatrices.RkMatrix, values(Haca.Bfar))
    @test all(v -> v isa Matrix, values(Haca.Dnear))
    @test all(v -> v isa Matrix, values(Haca.Ddiag))
    @test all(v -> v isa Matrix, values(Hden.Bfar))

    x = randn(length(pts))
    rel = norm(Haca * x - Hden * x) / (norm(Hden * x) + 1e-14)
    @test rel < 1e-4
end

@testset "DIBEM H2 + hss_method=:aca" begin
    include(joinpath(@__DIR__, "..", "data", "Laplace", "Laplace_dad.jl"))
    msh = Base.invokelatest(quadrado; ndiv=8, show=false, nome="test_h2aca_dibem")
    dad = format2d(msh, Laplace(1.0); tipo=1, pontointerno=false)
    xs = range(0.25, 0.75; length=3)
    set_internal_nodes!(dad, [SVector(float(x), float(y)) for y in xs for x in xs])
    Md = DIBEM(dad; method=:dense)
    dad2 = format2d(msh, Laplace(1.0); tipo=1, pontointerno=false)
    set_internal_nodes!(dad2, [SVector(float(x), float(y)) for y in xs for x in xs])
    Mh = DIBEM(dad2; method=:h2, hss_method=:aca, nmax=16, rtol=1e-5, f_method=:dense)
    @test dad2.dibem_D isa HMatrices.H2Matrix
    @test all(v -> v isa Matrix, values(dad2.dibem_D.Dnear))
    x = randn(size(Md, 1))
    @test norm(Md * x - Mh * x) / (norm(Md * x) + 1e-14) < 0.35
end
