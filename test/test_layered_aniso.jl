# Layered / anisotropic half-space kernels (Bagault et al., IJSS 50, 2013).
using Test
using LinearAlgebra
using BEM
using BEM.Contact

@testset "Layered anisotropic half-space" begin
    G, ν = 82000.0, 0.28
    E = 2G * (1 + ν)

    @testset "Voigt C" begin
        C = isotropic_C(E, ν)
        @test C ≈ C'
        @test C[4, 4] ≈ G atol=1e-8 * G
        Cc = cubic_almost_isotropic(E, ν; δG=0.01)
        @test Cc[4, 4] ≈ 1.01 * G atol=1e-8 * G
        Co = orthotropic_C(E, E, 2E, ν, ν, ν, G, G, G)
        @test Co ≈ Co'
        @test Co[3, 3] > Co[1, 1]
        R = rotate_C_about_x(C, π / 5)
        @test R ≈ R'
    end

    @testset "homogeneous isotropic Hertz vs Pohrt" begin
        R, a = 10.0, 1.0
        δ = a^2 / R
        Estar = 2G / (1 - ν)
        p0 = 2Estar * a / (π * R)
        N = 24
        L = 2.5 * a
        hx = 2L / N
        x = collect(range(-L + hx / 2, L - hx / 2; length=N))
        gap = [(xi^2 + yj^2) / (2R) for xi in x, yj in x]
        hs = ElasticHalfSpace(G, ν; hx=hx, hy=hx)
        solP = solve_normal_contact(gap, δ, hs; tol=1e-5)
        lhs = isotropic_halfspace(E, ν; hx=hx, hy=hx)
        prep = precompute_kernels(N, N, lhs; components=(Kzz,), nq=128)
        solS = solve_normal_contact(gap, δ, lhs; tol=1e-5, prep=prep)
        @test maximum(solS.p) / p0 ≈ 1 atol=0.08
        @test maximum(solS.p) / maximum(solP.p) ≈ 1 atol=0.08
    end

    @testset "isotropic coating Ec/Es (Bagault Fig. 1 trend)" begin
        R, a = 10.0, 1.0
        δ = a^2 / R
        Estar = 2G / (1 - ν)
        p0 = 2Estar * a / (π * R)
        N = 24
        L = 2.5 * a
        hx = 2L / N
        x = collect(range(-L + hx / 2, L - hx / 2; length=N))
        gap = [(xi^2 + yj^2) / (2R) for xi in x, yj in x]
        Zc = a / 2
        function pmax_ratio(Ec_over_Es)
            coat = isotropic_coated(Ec_over_Es * E, ν, Zc, E, ν; hx=hx, hy=hx)
            prep = precompute_kernels(N, N, coat; components=(Kzz,), nq=128)
            sol = solve_normal_contact(gap, δ, coat; tol=1e-5, prep=prep)
            return maximum(sol.p) / p0
        end
        r1 = pmax_ratio(1.0)
        r4 = pmax_ratio(4.0)
        r025 = pmax_ratio(0.25)
        @test r1 ≈ 1 atol=0.08
        @test r4 > 1.2          # stiff coating raises pmax
        @test r025 < 0.85       # compliant coating lowers pmax
        @test r4 > r1 > r025
    end

    @testset "orthotropic E3=2E (Bagault Fig. 3 trend)" begin
        R, a = 10.0, 1.0
        δ = a^2 / R
        Estar = 2G / (1 - ν)
        p0 = 2Estar * a / (π * R)
        N = 24
        L = 2.5 * a
        hx = 2L / N
        x = collect(range(-L + hx / 2, L - hx / 2; length=N))
        gap = [(xi^2 + yj^2) / (2R) for xi in x, yj in x]
        Co = orthotropic_C(E, E, 2E, ν, ν, ν, G, G, G)
        lhs = homogeneous(Co; hx=hx, hy=hx)
        prep = precompute_kernels(N, N, lhs; components=(Kzz,), nq=128)
        sol = solve_normal_contact(gap, δ, lhs; tol=1e-5, prep=prep)
        @test maximum(sol.p) / p0 > 1.05
    end
end
