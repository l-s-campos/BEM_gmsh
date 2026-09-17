# DIBEM remainder identity M 1 = ID (PHS3 + poly).
using Test
using LinearAlgebra
using StaticArrays
using BEM

include(joinpath(@__DIR__, "..", "data", "Laplace", "Laplace_dad.jl"))

@testset "DIBEM Laplace M*1 = ID" begin
    msh = Base.invokelatest(quadrado; ndiv=6, show=false, nome="phs3_L")
    dad = format2d(msh, Laplace(1.0); tipo=1, pontointerno=true)
    M = DIBEM(dad; rbf=PHS(3; poly_deg=1), npg=12)
    @test size(M) == (dad.nt, dad.nt)
    @test length(dad.dibem_c) == dad.nt
    onesv = ones(dad.nt)
    @test norm(M * onesv - dad.dibem_ID) / (norm(dad.dibem_ID) + 1e-14) < 1e-10
end

@testset "DIBEM elasticity M*1 = ID" begin
    msh = Base.invokelatest(quadrado_elasticity; ndiv=6, show=false, nome="phs3_E")
    dad = format2d(msh, Elasticity(1.0, 0.3, 1.0; plane_strain=true);
        tipo=1, pontointerno=true)
    M = DIBEM(dad; rbf=PHS(3; poly_deg=1), npg=12)
    @test size(M) == (2dad.nt, 2dad.nt)
    b = zeros(2 * dad.nt)
    for i in 1:dad.nt
        b[2i-1] = 1.0
    end
    IDb = zeros(2 * dad.nt)
    ID = dad.dibem_ID
    for i in 1:dad.nt
        IDb[2i-1:2i] .= ID[2i-1:2i, :] * SVector(1.0, 0.0)
    end
    @test norm(M * b - IDb) / (norm(IDb) + 1e-14) < 1e-10
end
