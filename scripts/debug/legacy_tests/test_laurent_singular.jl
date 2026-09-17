# Laurent / Guiggiani singular integration
using Test
using DrWatson
@quickactivate :BEM
using LinearAlgebra
using StaticArrays
using FastGaussQuadrature

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

@testset "guiggiani_integral log self-check" begin
    # ∫_{-1}^{1} log|ξ - a| dξ = (1-a)log|1-a| + (1+a)log|1+a| - 2
    qsi, w = gausslegendre(32)
    for a in (-0.4, 0.0, 0.55)
        exact = (1 - a) * log(abs(1 - a)) + (1 + a) * log(abs(1 + a)) - 2
        I = BEM.guiggiani_integral(ξ -> log(abs(ξ - a)), a, 0; qsi=qsi, w=w, h=1e-3)
        @test abs(I - exact) < 1e-6
    end
end

@testset "singularity_orders on Problem" begin
    @test singularity_orders(Laplace(1.0)) === (0, -1)
    @test singularity_orders(Elasticity(1.0, 0.3, 1.0)) === (0, -1)
    @test singularity_order_G(Helmholtz()) == 0
    @test singularity_order_H(Helmholtz()) == -1
end

@testset "guiggiani_GH matches separate passes" begin
    qsi, w = gausslegendre(24)
    for a in (-0.4, 0.0, 0.55)
        # G ~ log, H ~ 1/(ξ-a); fused must match two guiggiani_integral calls
        fG = ξ -> log(abs(ξ - a))
        fH = ξ -> 1 / (ξ - a)
        IG = BEM.guiggiani_integral(fG, a, 0; qsi=qsi, w=w, h=1e-3)
        IH = BEM.guiggiani_integral(fH, a, -1; qsi=qsi, w=w, h=1e-3)
        IGf, IHf = BEM.guiggiani_GH(ξ -> (fG(ξ), fH(ξ)), a;
            order_G=0, order_H=-1, qsi=qsi, w=w, h=1e-3)
        @test IGf ≈ IG rtol = 1e-9 atol = 1e-12
        @test IHf ≈ IH rtol = 1e-9 atol = 1e-12
        # same via Problem defaults
        og, oh = singularity_orders(Laplace(1.0))
        IGp, IHp = BEM.guiggiani_GH(ξ -> (fG(ξ), fH(ξ)), a;
            order_G=og, order_H=oh, qsi=qsi, w=w, h=1e-3)
        @test IGp ≈ IG rtol = 1e-9 atol = 1e-12
        @test IHp ≈ IH rtol = 1e-9 atol = 1e-12
    end
    # matrix-valued pair (vectorial kernel layout)
    a = 0.2
    fG = ξ -> log(abs(ξ - a)) * @SMatrix [1.0 0.1; 0.1 1.0]
    fH = ξ -> (1 / (ξ - a)) * @SMatrix [2.0 0.0; 0.0 3.0]
    IG = BEM.guiggiani_integral(fG, a, 0; qsi=qsi, w=w, h=1e-3)
    IH = BEM.guiggiani_integral(fH, a, -1; qsi=qsi, w=w, h=1e-3)
    IGf, IHf = BEM.guiggiani_GH(ξ -> (fG(ξ), fH(ξ)), a;
        order_G=0, order_H=-1, qsi=qsi, w=w, h=1e-3)
    @test IGf ≈ IG rtol = 1e-9 atol = 1e-12
    @test IHf ≈ IH rtol = 1e-9 atol = 1e-12
end
