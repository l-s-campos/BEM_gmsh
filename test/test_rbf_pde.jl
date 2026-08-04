# RBF module + particular-solution BEM
using Test
using LinearAlgebra
using Statistics: mean
using StaticArrays
using BEM

@testset "radial_integral PHS" begin
    # PHS3 φ=r³: ∫_0^1 r^4 dr = 1/5
    @test radial_integral(PHS(3), 1.0; dim = 2) ≈ 0.2 atol = 1e-12
    @test radial_integral(PHS(3), 1.0; dim = 3) ≈ 1 / 6 atol = 1e-12
    @test radial_integral(PHS(1), 2.0; dim = 2) ≈ 8 / 3 atol = 1e-12
    # IMQ numerical finite
    @test isfinite(radial_integral(IMQ(1.0), 1.0; dim = 2))
    @test isfinite(radial_integral(WendlandC2(2.0), 1.0; dim = 2))
end

@testset "backward compat RBF/PHS/cardinal" begin
    pts = Point2D[Point2D(0, 0), Point2D(1, 0), Point2D(0, 1), Point2D(1, 1), Point2D(0.5, 0.5)]
    y = [p[1] + 2p[2] for p in pts]
    rbf = RBF(pts, PHS(3; poly_deg = 1))
    ŷ = rbf_evaluate(rbf, pts, y)
    @test ŷ ≈ y atol = 1e-8
    w = rbf(pts[3])  # weights callable
    @test length(w) == 5
    @test sum(w) ≈ 1 atol = 1e-6
    w2 = rbf_cardinal(pts[1], pts; basis = PHS(3; poly_deg = 1))
    @test abs(sum(w2) - 1) < 1e-5 || abs(w2[1] - 1) < 1e-5
end

@testset "local / PU / rational interpolate" begin
    # franke-like smooth
    f(p) = sin(π * p[1]) * cos(π * p[2])
    n = 8
    xs = range(0, 1; length = n)
    pts = Point2D[Point2D(x, y) for x in xs for y in xs]
    y = f.(pts)
    xt = Point2D[Point2D(0.3, 0.4), Point2D(0.7, 0.2), Point2D(0.5, 0.5)]
    yt = f.(xt)

    lr = local_rbf_fit(pts, PHS(3; poly_deg = 1); k = 12)
    e_loc = maximum(abs(local_rbf_eval(lr, x, y) - f(x)) for x in xt)
    @test e_loc < 0.05

    pu = pu_rbf_fit(pts, PHS(3; poly_deg = 1); k_local = 12)
    e_pu = maximum(abs(pu_rbf_eval(pu, x, y) - f(x)) for x in xt)
    @test e_pu < 0.1

    rr = rational_rbf_fit(pts, y, IMQ(2.0; poly_deg = -1))
    e_rat = maximum(abs(rational_rbf_eval(rr, x) - f(x)) for x in xt)
    @test e_rat < 0.15
end

@testset "compare_rbf_methods" begin
    f(p) = exp(-((p[1] - 0.5)^2 + (p[2] - 0.5)^2) * 8)
    n = 7
    xs = range(0, 1; length = n)
    pts = Point2D[Point2D(x, y) for x in xs for y in xs]
    y = f.(pts)
    xt = Point2D[Point2D(rand(), rand()) for _ in 1:10]
    yt = f.(xt)
    res = compare_rbf_methods(pts, y, xt, yt; k_local = 10)
    @test length(res) >= 4
    # at least one method has finite rmse
    @test any(isfinite(r.rmse) && r.rmse < 1 for r in res)
    @info "RBF comparison" res
end

@testset "kansa poisson unit square" begin
    # ∇²u = -2π² sin(πx)sin(πy), u=0 on boundary
    uex(p) = sin(π * p[1]) * sin(π * p[2])
    fsrc(p) = -2π^2 * uex(p)
    n = 6
    xs = range(0.1, 0.9; length = n)
    interior = Point2D[Point2D(x, y) for x in xs for y in xs]
    # boundary samples
    nb = 16
    boundary = Point2D[]
    for t in range(0, 1; length = nb)
        push!(boundary, Point2D(t, 0))
        push!(boundary, Point2D(t, 1))
        push!(boundary, Point2D(0, t))
        push!(boundary, Point2D(1, t))
    end
    unique!(boundary)
    g(p) = 0.0
    sol = kansa_poisson(interior, boundary, fsrc, g; basis = PHS(5; poly_deg = 1))
    err = maximum(abs(kansa_eval(sol, p) - uex(p)) for p in interior)
    @test err < 0.15
end

@testset "poisson particular + BEM" begin
    # unit square, u=x²+y² ⇒ ∇²u=4, Dirichlet u on boundary
    include(joinpath(@__DIR__, "..", "data", "Laplace", "Laplace_dad.jl"))
    msh = quadrado(ndiv = 6, show = false, nome = "rbf_pois")
    dad = format2d(msh, Laplace(1.0); pontointerno = true)
    # set Dirichlet u = x²+y² everywhere on boundary
    for i in 1:dad.n
        dad.BC[i] = 0
        p = dad.Nodes[i]
        dad.BV[i] = p[1]^2 + p[2]^2
    end
    fsrc = 4.0
    u = solve_poisson_rbf_bem!(dad, fsrc; method = :global, basis = PHS(3; poly_deg = 1), npg = 12)
    pts = [dad.Nodes; dad.internalNodes]
    uex = [p[1]^2 + p[2]^2 for p in pts]
    rmse = sqrt(mean(abs2, u .- uex))
    @info "Poisson RBF-BEM RMSE" rmse
    @test rmse < 0.15
    @test has_cache(dad, :up)
end

@testset "laplace particular consistency" begin
    # ∇² Ψ ≈ φ for PHS3 at r=0.5
    r = 0.5
    φ = (r^3)  # PHS3
    # numerical Laplacian of Ψ in 2D via finite differences
    Ψ(rr) = laplace_particular(PHS(3), rr; dim = 2)
    ε = 1e-5
    # radial L = Ψ'' + (1/r)Ψ'
    Ψp = Ψ(r + ε); Ψ0 = Ψ(r); Ψm = Ψ(r - ε)
    Ψ′ = (Ψp - Ψm) / (2ε)
    Ψ′′ = (Ψp - 2Ψ0 + Ψm) / ε^2
    L = Ψ′′ + Ψ′ / r
    @test L ≈ φ rtol = 0.05
end
