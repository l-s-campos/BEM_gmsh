# Isoparametric Bézier IGA-BEM
using Test
using LinearAlgebra
using BEM

include(joinpath(@__DIR__, "..", "data", "Laplace", "Laplace_dad.jl"))

@testset "IGA isoparametric format + assemble" begin
    msh = Base.invokelatest(quadrado; ndiv=6, show=false, nome="iga_iso")
    dad = format2d(msh, Laplace(1.0);
        discretization = :iga, iga_mode = :bezier_mesh, iga_degree = 2,
        tipo = 2, pontointerno = false)
    @test dad.element_type isa Bernstein
    @test all(e -> e.controls !== nothing, dad.elements)
    @test all(e -> length(e) == length(e.controls), dad.elements)
    @test all(e -> length(e.Jacobian) == length(e), dad.elements)

    # isoparametric shape path
    el = dad.elements[1]
    x = el.controls
    eta = [-0.5, 0.0, 0.5]
    N, dN, pg, dx = BEM._elem_field_and_geom(dad, el, x, eta)
    @test size(N, 2) == length(el)
    @test length(pg) == length(eta)
    @test all(isfinite, N)
    @test all(isfinite, norm.(dx))

    H_G_full_direct(dad, 12)
    @test has_cache(dad, :H)
    @test size(dad.H, 1) == dad.n
    @test all(isfinite, dad.H)
end
