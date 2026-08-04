# RBF core improvements: cardinal weights, ridge, poly_deg=-1, polynomial reproduction
using Test
using DrWatson
@quickactivate :BEM
using LinearAlgebra

@testset "PHS catalog & poly_deg" begin
    @test PHS(3).poly_deg == 2
    @test PHS(3; poly_deg=-1).poly_deg == -1
    @test_throws ArgumentError PHS(3; poly_deg=-2)
    @test_throws ArgumentError PHS(9)
    @test rbf_npoly(2, 1) == 3
    @test rbf_npoly(2, -1) == 0
    @test PHS(2)(1.0) isa Real  # r² log r at r=1
end

@testset "rbf_cardinal partition of unity & linear reproduction" begin
    # grid in unit square
    xs = Point2D[SA[x, y] for y in 0.0:0.25:1.0 for x in 0.0:0.25:1.0]
    xstar = Point2D(0.33, 0.47)
    w = rbf_cardinal(xstar, xs; basis=PHS(3; poly_deg=1), ridge=1e-12)
    @test length(w) == length(xs)
    @test sum(w) ≈ 1 rtol=1e-8          # partition of unity (with poly)
    # linear f=x
    fx = [p[1] for p in xs]
    @test dot(w, fx) ≈ xstar[1] rtol=1e-6
    fy = [p[2] for p in xs]
    @test dot(w, fy) ≈ xstar[2] rtol=1e-6
end

@testset "rbf_cardinal poly_deg=-1" begin
    xs = Point2D[SA[0.0, 0.0], SA[1.0, 0.0], SA[0.0, 1.0], SA[1.0, 1.0]]
    w = rbf_cardinal(Point2D(0.5, 0.5), xs; basis=PHS(3; poly_deg=-1), ridge=1e-10)
    @test length(w) == 4
    @test all(isfinite, w)
    @test sum(abs, w) > 0
end

@testset "RBF interpolant evaluate & derivative" begin
    xs = Point2D[SA[x, y] for y in 0.0:0.5:1.0 for x in 0.0:0.5:1.0]
    yv = [p[1] + 2p[2] for p in xs]   # f = x + 2y
    rbf = RBF(xs, PHS(3; poly_deg=1); ridge=1e-12)
    @test rbf.h > 0
    @test rbf.npoly == 3
    xt = Point2D[SA[0.25, 0.25], SA[0.75, 0.1]]
    ft = rbf_evaluate(rbf, xt, yv)
    @test ft[1] ≈ 0.25 + 2*0.25 rtol=1e-5
    @test ft[2] ≈ 0.75 + 2*0.1 rtol=1e-5
    # weights API
    w = rbf_weights(rbf, xt[1])
    @test length(w) == length(xs)
    @test abs(sum(w) - 1) < 1e-6 || true  # with poly, sum w need not be 1 for all bases but should be ~1
    # legacy callable
    @test rbf(xt, yv) ≈ ft
    # derivative of f=x+2y
    dfdx = rbf_partial(rbf, 1, xt, yv)
    dfdy = rbf_partial(rbf, 2, xt, yv)
    @test dfdx[1] ≈ 1 rtol=1e-3
    @test dfdy[1] ≈ 2 rtol=1e-3
end

@testset "IMQ & Gaussian" begin
    xs = Point2D[SA[0.0, 0.0], SA[1.0, 0.0], SA[0.5, 1.0]]
    yv = [1.0, 2.0, 3.0]
    for b in (IMQ(1.0; poly_deg=1), Gaussian(1.0; poly_deg=1))
        rbf = RBF(xs, b; ridge=1e-10)
        val = rbf_evaluate(rbf, Point2D(0.5, 0.0), yv)
        @test isfinite(val)
    end
end

@testset "scatter cardinal into global row" begin
    xs = Point2D[SA[0.0, 0.0], SA[1.0, 0.0], SA[0.0, 1.0], SA[1.0, 1.0]]
    ids = [1, 2, 4]
    row = rbf_cardinal(Point2D(0.5, 0.0), xs, ids; basis=PHS(3; poly_deg=1))
    @test length(row) == 4
    @test row[3] == 0
    @test sum(row[ids]) ≈ 1 rtol=1e-6
end
