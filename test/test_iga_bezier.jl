# Isogeometric / Bézier extraction tests
using Test
using LinearAlgebra
using BEM
using BEM: shapefun, degree

@testset "Bernstein partition of unity" begin
    for p in 1:4
        poly = Bernstein(p)
        for ξ in range(-1, 1; length = 11)
            N, dN = shapefun(poly, ξ)
            @test sum(N) ≈ 1 atol = 1e-12
            @test sum(dN) ≈ 0 atol = 1e-10
        end
        @test degree(poly) == p
        @test bernstein_degree(poly) == p
    end
end

@testset "Bezier extraction open uniform" begin
    p = 2
    n_el = 4
    Ξ = open_knot_vector(n_el, p)
    Cs, spans = bezier_extraction(Ξ, p)
    @test length(Cs) == n_el
    @test length(spans) == n_el
    for C in Cs
        @test size(C) == (p + 1, p + 1)
        # rows of extraction should sum ~1 (partition of unity preserved)
        @test all(abs.(sum(C; dims = 2) .- 1) .< 1e-8)
    end
end

@testset "Lagrange → Bézier geometry exact for line" begin
    p = 2
    # straight segment: equispaced Lagrange nodes
    xL = [Point2D(0.0, 0.0), Point2D(0.5, 0.0), Point2D(1.0, 0.0)]
    Pb = bezier_controls_from_lagrange(xL, p)
    poly = Bernstein(p)
    # geometry at mid must match
    N, _ = shapefun(poly, 0.0)
    xmid = sum(N[1, a] * Pb[a] for a in 1:(p + 1))
    @test xmid[1] ≈ 0.5 atol = 1e-12
    @test xmid[2] ≈ 0.0 atol = 1e-12
end

@testset "Element extraction interface" begin
    p = 2
    poly = Bernstein(p)
    C = Matrix{Float64}(I, p + 1, p + 1)
    w = ones(p + 1)
    el = BezierElement(collect(1:(p + 1)), ones(3), 1.0, 1, C; weights = w)
    @test is_bezier_element(el)
    @test length(el) == p + 1
    N0, dN0 = shapefun(poly, 0.0)
    N1, dN1 = element_shapefun(poly, el, 0.0)
    @test N0 ≈ N1
    @test dN0 ≈ dN1
end

@testset "format2d discretization=:iga CAD" begin
    geo = joinpath(@__DIR__, "..", "data", "Laplace", "unit_square.geo")
    @test isfile(geo)
    dad = format2d(geo, Laplace(1.0);
        discretization = :iga,
        iga_mode = :cad,
        iga_degree = 2,
        iga_nel = 4,
        tipo = 2,
        pontointerno = false,
    )
    @test dad.element_type isa Bernstein
    @test bernstein_degree(dad.element_type) == 2
    @test !isempty(dad.elements)
    @test all(is_bezier_element, dad.elements)
    @test dad.n == sum(length, dad.elements)
    # perimeter of unit square ≈ 4
    L = sum(e.Length for e in dad.elements)
    @test L ≈ 4.0 rtol = 0.15
end

@testset "format2d IGA assemble smoke" begin
    geo = joinpath(@__DIR__, "..", "data", "Laplace", "unit_square.geo")
    dad = format2d(geo, Laplace(1.0);
        discretization = :iga, iga_mode = :cad, iga_degree = 2, iga_nel = 3,
        tipo = 1, pontointerno = false,
    )
    H_G_full_direct(dad, 12)
    @test has_cache(dad, :H)
    @test size(dad.H, 1) == dad.n
    solve(dad)
    @test has_cache(dad, :T)
    @test all(isfinite, dad.T[1:dad.n])
end
