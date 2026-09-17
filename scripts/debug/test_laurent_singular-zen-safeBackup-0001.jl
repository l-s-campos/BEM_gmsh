# Laurent / Guiggiani singular integration (Inti-inspired)
using Test
using DrWatson
@quickactivate :BEM
using LinearAlgebra
using StaticArrays
using FastGaussQuadrature

const Legendre = BEM.Legendre

@testset "laurent_coefficients (Richardson)" begin
    f = ρ -> ρ^2 + 2ρ + 1
    f2, f1, f0 = BEM.laurent_coefficients(f, 1e-2, Val(-2))
    @test norm((f2, f1, f0) .- (0, 0, 1)) < 1e-10
    f2, f1, f0 = BEM.laurent_coefficients(f, 1e-2, Val(0))
    @test norm((f2, f1, f0) .- (0, 0, 1)) < 1e-10

    f = ρ -> cos(ρ) / ρ^2 + exp(ρ) / ρ + exp(ρ)
    f2, f1, f0 = BEM.laurent_coefficients(f, 1.0, Val(-2); atol=1e-12, breaktol=2, contract=1 / 2)
    @test norm((f2, f1, f0) .- (1, 1, 1.5)) < 1e-9

    f = ρ -> SVector(cos(ρ), sin(ρ)) / ρ^2 + SVector(exp(ρ), 0.2) / ρ
    f2, f1, f0 = BEM.laurent_coefficients(f, 1e-1, Val(-2))
    @test f2 ≈ SVector(1.0, 0.0)
    @test f1 ≈ SVector(1.0, 1.2)
end

@testset "laurent_shape_coefficients analytic vs Richardson" begin
    poly = Legendre(2)  # 3 nodes
    a = 0.3
    for s in (-1.0, 1.0), order in (1, 2)
        Fa = BEM.laurent_shape_coefficients(poly, a, s, order; method=:analytic)
        Fr = BEM.laurent_shape_coefficients(poly, a, s, order; method=:richardson, h=1e-3)
        @test Fa[1] ≈ Fr[1] rtol = 1e-8 atol = 1e-10
        @test Fa[2] ≈ Fr[2] rtol = 1e-8 atol = 1e-10
        # F₀ is noisier under Richardson; looser tol
        @test Fa[3] ≈ Fr[3] rtol = 1e-5 atol = 1e-7
    end
end

@testset "singular_laurent matches singular (analytic)" begin
    for deg in (1, 2, 3)
        poly = Legendre(deg)
        qsi, w = gausslegendre(deg + 1)
        for a in (-0.4, 0.0, 0.55), order in (0, 1, 2)
            wn0 = BEM.singular(qsi, w, order, a; poly=poly)
            wnL = BEM.singular_laurent(qsi, w, order, a; poly=poly, method=:analytic)
            @test wn0 ≈ wnL rtol = 1e-10 atol = 1e-12
        end
    end
end

@testset "singular_laurent Richardson matches analytic" begin
    poly = Legendre(2)
    qsi, w = gausslegendre(3)
    a = 0.2
    for order in (1, 2)
        wnA = BEM.singular_laurent(qsi, w, order, a; poly=poly, method=:analytic)
        wnR = BEM.singular_laurent(qsi, w, order, a; poly=poly, method=:richardson, h=5e-4)
        @test wnA ≈ wnR rtol = 1e-6 atol = 1e-8
    end
end

@testset "guiggiani_integral CPV self-check" begin
    # ∫_{-1}^{1} 1/(ξ - s) dξ = log|(1-s)/(-1-s)|  (CPV, s real ∈ (-1,1))
    qsi, w = gausslegendre(24)
    for s in (-0.4, 0.1, 0.7)
        exact = log(abs((1 - s) / (-1 - s)))
        I = BEM.guiggiani_integral(ξ -> 1 / (ξ - s), s, -1; qsi=qsi, w=w, h=1e-3)
        @test abs(I - exact) < 1e-8
    end
end

@testset "guiggiani_integral HFP self-check" begin
    # ∫_{-1}^{1} 1/(ξ - a)² dξ  (HFP) = -2/(1-a²)
    qsi, w = gausslegendre(32)
    for a in (-0.3, 0.0, 0.5)
        exact = -2 / (1 - a^2)
        I = BEM.guiggiani_integral(ξ -> 1 / (ξ - a)^2, a, -2; qsi=qsi, w=w, h=1e-3)
        @test abs(I - exact) < 1e-7
    end
end
