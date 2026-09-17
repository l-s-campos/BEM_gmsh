# Half-plane / half-space contact: Hertz line + Cattaneo identities.
using Test
using LinearAlgebra
using Statistics: mean
using BEM
using BEM.Contact
using BEM.MultiRegion

@testset "Hertz line pressure integrates to F" begin
    G, ν = 1.0, 0.3
    N = 64
    x = collect(range(-3.0, 3.0; length=N))
    h = x[2] - x[1]
    hp = ElasticHalfPlane2D(G, ν; h=h)
    F_hz = 0.05
    p_hz, _ = hertz_line_pressure(F_hz, 1.0, hp, x)
    @test abs(sum(p_hz) * h - F_hz) / F_hz < 0.05
    @test influence_coeff_2d(0, hp) > abs(influence_coeff_2d(5, hp))
end

@testset "Cattaneo |q| ≤ f p and stick width" begin
    par = loyola_cattaneo_params()
    x = collect(range(-1.5par.a, 1.5par.a; length=201))
    p = cattaneo_pressure(x, par.a, par.p0)
    Q = 0.5 * par.f * par.P
    q = cattaneo_shear(x, par.a, par.p0, Q, par.f, par.P)
    @test all(abs.(q) .<= par.f .* p .+ 1e-9 * par.p0)
    c = cattaneo_c(par.a, Q, par.f, par.P)
    @test c ≈ par.a * sqrt(1 - abs(Q) / (par.f * par.P)) atol=1e-12
end

@testset "Pohrt–Li Kzz > 0 and FFT = dense" begin
    hs = ElasticHalfSpace(1.0, 0.3; hx=0.25, hy=0.25)
    @test influence_coeff(Kzz, 0, 0, hs) > 0
    nx = ny = 8
    prep = precompute_kernels(nx, ny, hs; components=(Kzz,))
    p = zeros(nx, ny); p[4, 4] = 1.0
    u_fc = fc_forward(p, Kzz, prep)
    u_dir = zeros(nx, ny)
    for j in 1:ny, i in 1:nx
        s = 0.0
        for jj in 1:ny, ii in 1:nx
            s += influence_coeff(Kzz, i - ii, j - jj, hs) * p[ii, jj]
        end
        u_dir[i, j] = s
    end
    @test u_fc ≈ u_dir rtol=1e-10
end

@testset "Pohrt H-matrix / H² / FMM match FFT on Kzz" begin
    hs = ElasticHalfSpace(1.0, 0.3; hx=0.25, hy=0.25)
    nx = ny = 8
    p = zeros(nx, ny); p[4, 4] = 1.0
    u_fft = fc_forward(p, Kzz, precompute_kernels(nx, ny, hs; components=(Kzz,)))
    Kd = build_pohrt_operator(hs, nx, ny, Kzz; method=:dense)
    @test reshape(Kd * vec(p), nx, ny) ≈ u_fft rtol=1e-10
    prep = precompute_kernels(nx, ny, hs; components=(Kzz,), method=:hmatrix,
        nmax=8, atol=1e-10)
    @test fc_forward(p, Kzz, prep) ≈ u_fft rtol=1e-4 atol=1e-8
    prep2 = precompute_kernels(nx, ny, hs; components=(Kzz,), method=:h2,
        nmax=8, rtol=1e-8)
    @test fc_forward(p, Kzz, prep2) ≈ u_fft rtol=1e-3 atol=1e-8
    prepHss = precompute_kernels(nx, ny, hs; components=(Kzz,), method=:hss,
        nmax=8, rtol=1e-8)
    @test fc_forward(p, Kzz, prepHss) ≈ u_fft rtol=1e-3 atol=1e-8
    # FMM = Laplace 1/r far field + Love near stencil (same as HalfSpaceBEM).
    u_fmm = fc_forward(p, Kzz, precompute_kernels(nx, ny, hs; components=(Kzz,),
        method=:fmm, eps=1e-8))
    @test u_fmm ≈ u_fft rtol=1e-3 atol=1e-8
end

@testset "combined_halfspace and Kxz/Kyz odd (no atan2 leak)" begin
    G, ν = 80.769, 0.3
    hs1 = ElasticHalfSpace(G, ν)
    hs2 = combined_halfspace(G, ν, G, ν)
    @test hs2.K ≈ 0 atol=1e-14
    @test hs2.G ≈ G / 2
    @test influence_coeff(Kzz, 0, 0, hs2) ≈ 2 * influence_coeff(Kzz, 0, 0, hs1) rtol=1e-12
    hs = combined_halfspace(80.0, 0.3, 8000.0, 0.3; hx=0.1, hy=0.1)
    @test influence_coeff(Kxz, 0, 0, hs) ≈ 0 atol=1e-18
    @test influence_coeff(Kyz, 0, 0, hs) ≈ 0 atol=1e-18
    for di in 1:6
        @test influence_coeff(Kxz, di, 0, hs) ≈ -influence_coeff(Kxz, -di, 0, hs) rtol=1e-12
        @test influence_coeff(Kyz, 0, di, hs) ≈ -influence_coeff(Kyz, 0, -di, hs) rtol=1e-12
    end
end

@testset "common contact normal n_AB = (n_A E_A − n_B E_B)/‖·‖" begin
    nA = (0.0, -1.0)
    nB = (0.0, 1.0)
    n = contact_common_normal(nA, 1.0, nB, 1.0)
    @test n ≈ [0.0, -1.0]
    # stiffer B → n_AB ≈ −n_B
    n = contact_common_normal(nA, 1.0, nB, 1e6)
    @test n ≈ [0.0, -1.0]
    # tilted pad vs flat specimen, equal E
    θ = 0.2
    nA = (sin(θ), -cos(θ))
    nB = (0.0, 1.0)
    n = contact_common_normal(nA, 73.4e3, nB, 73.4e3)
    v = [nA[1] - nB[1], nA[2] - nB[2]]
    v ./= hypot(v[1], v[2])
    @test n ≈ v
    @test hypot(n[1], n[2]) ≈ 1
    # more vertical than the pad geometric normal
    @test abs(n[1]) < abs(nA[1])
end
