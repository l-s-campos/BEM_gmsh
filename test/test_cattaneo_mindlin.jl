# Cattaneo–Mindlin analytical + half-space + mortar smoke tests
using Test
using LinearAlgebra
using BEM

@testset "Loyola parameters" begin
    par = loyola_cattaneo_params()
    @test par.R == 70
    @test par.P == 100
    @test par.f == 0.3
    # Hertz-consistent from Table 9.19 (R,E,ν,P).
    # Note: thesis Tables 9.20–9.21 quote a=1.186 mm, p0=697.8 MPa which
    # imply P≈1301 N/mm — inconsistent with Table 9.19 P=100; we use P=100.
    @test par.a ≈ sqrt(4 * par.P * par.R_eq / (π * par.E_eq)) rtol = 1e-12
    @test par.p0 ≈ 2 * par.P / (π * par.a) rtol = 1e-12
    @test par.a > 0.2 && par.a < 0.5
    @test par.p0 > 100 && par.p0 < 300
end

@testset "Analytical Cattaneo shear" begin
    par = loyola_cattaneo_params()
    x = range(-1.5par.a, 1.5par.a; length=401) |> collect
    p = cattaneo_pressure(x, par.a, par.p0)
    @test maximum(p) ≈ par.p0 rtol = 1e-3
    @test sum(p) * (x[2] - x[1]) ≈ par.P rtol = 0.02

    Q = 0.5 * par.f * par.P
    q = cattaneo_shear(x, par.a, par.p0, Q, par.f, par.P)
    @test sum(q) * (x[2] - x[1]) ≈ Q rtol = 0.05
    # Coulomb: |q| ≤ f p
    @test all(abs.(q) .<= par.f .* p .+ 1e-9 * par.p0)
    # stick half-width
    c = cattaneo_c(par.a, Q, par.f, par.P)
    @test c ≈ par.a * sqrt(1 - abs(Q) / (par.f * par.P)) atol = 1e-12
end

@testset "Mindlin history A-E" begin
    par = loyola_cattaneo_params()
    x = range(-1.5par.a, 1.5par.a; length=201) |> collect
    Qpath = [st.Q for st in par.load_steps]
    # step B = monotonic
    qB = mindlin_shear_history(x, par.a, par.p0, par.f, par.P, Qpath[1:2])
    qB_c = cattaneo_shear(x, par.a, par.p0, Qpath[2], par.f, par.P)
    @test qB ≈ qB_c rtol = 1e-10
    # step D ≈ reverse of B after full cycle extremes
    qD = mindlin_shear_history(x, par.a, par.p0, par.f, par.P, Qpath[1:4])
    @test sum(qD) * (x[2] - x[1]) ≈ Qpath[4] rtol = 0.1
end

@testset "Half-space Cattaneo step B" begin
    par = loyola_cattaneo_params()
    ν = par.ν
    G = par.E_eq * (1 - ν) / 2
    N = 121
    L = 3par.a
    x = collect(range(-L, L; length=N))
    hp = ElasticHalfPlane2D(G, ν; h=x[2] - x[1])
    sol = solve_cattaneo_halfplane(x, par.R_eq, par.P, par.Qmax, par.f, hp; tol=1e-8)
    @test sol.force_n ≈ par.P rtol = 0.02
    @test sol.force_t ≈ par.Qmax rtol = 0.05
    @test sol.a ≈ par.a rtol = 0.15
    @test sol.p0 ≈ par.p0 rtol = 0.1
    @test count(sol.stick) > 0
    @test count(sol.slip) > 0
end

@testset "Cohesive Cattaneo normal" begin
    par = loyola_cattaneo_params()
    ν = par.ν
    G = par.E_eq * (1 - ν) / 2
    x = collect(range(-3par.a, 3par.a; length=81))
    h = x[2] - x[1]
    hp = ElasticHalfPlane2D(G, ν; h=h)
    kn = 30 * par.E_eq / h
    sol = solve_cattaneo_cohesive_halfplane(x, par.R_eq, par.P, 0.0, par.f, hp; kn=kn, kt=kn)
    @test sol.force_n ≈ par.P rtol = 0.05
    @test sol.a > 0
    @test sol.p0 > 0
end

@testset "Mortar transfer" begin
    par = loyola_cattaneo_params()
    ν = par.ν
    G = par.E_eq * (1 - ν) / 2
    xs = collect(range(-3par.a, 3par.a; length=101))
    xm = collect(range(-3par.a, 3par.a; length=41))
    sol = solve_cattaneo_mortar_halfplane(xs, xm, par.R_eq, par.P, par.Qmax, par.f, G, ν)
    @test sol.force_n_s ≈ par.P rtol = 0.05
    @test sol.a_s > 0
    @test sol.p0_m > 0
    @test size(sol.D, 1) == length(xs)
    @test size(sol.M, 2) == length(xm)
end
