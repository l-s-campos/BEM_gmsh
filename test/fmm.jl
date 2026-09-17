# FMM double-layer matvec vs dense Laplace kernel.
using Test
using LinearAlgebra
using Random
using StaticArrays
using BEM
using BEM.FMM

@testset "Laplace 2D SL FMM vs dense (threaded apply)" begin
    rng = Random.default_rng()
    Random.seed!(rng, 3)
    n = 512
    P = rand(rng, 2, n)
    x = randn(rng, n)
    yd = zeros(n)
    @inbounds for j in 1:n, i in 1:n
        i == j && continue
        r = hypot(P[1, j] - P[1, i], P[2, j] - P[2, i])
        r < 1e-30 && continue
        yd[j] += x[i] * log(r)
    end
    plan = FMM.build_laplace2d_plan(P; eps=1e-8)
    y = zeros(n)
    FMM.apply_laplace2d!(plan, y; charges=x)
    @test norm(y - yd) / (norm(yd) + 1e-14) < 1e-8
end

@testset "Laplace 2D double-layer FMM vs dense" begin
    dad = format2d(quadrado(ndiv=5, show=false, nome="t_fmm"), Laplace(1.0);
        tipo=1, pontointerno=false)
    pts = collect(all_points(dad))
    nt, n = length(pts), dad.n
    P = reduce(hcat, pts)
    N = zeros(2, nt)
    @inbounds for j in 1:n
        N[:, j] .= dad.Normal[j]
    end
    KF = FMM.fmm_laplace2d_double_layer_matrix(P, N; n_boundary=n, eps=1e-10, nmax=12)
    KD = BEM.LaplaceDqKernel(pts, dad.Normal, dad.properties, n)
    Dd = [KD[i, j] for i in 1:nt, j in 1:nt]
    x = randn(nt)
    @test norm(Dd * x - KF * x) / (norm(Dd * x) + 1e-14) < 1e-6
end

@testset "Laplace 3D SL octree FMM vs dense" begin
    rng = Random.default_rng()
    Random.seed!(rng, 7)
    n = 80
    P = rand(rng, 3, n)
    x = randn(rng, n)
    inv4π = 1 / (4π)
    yd = zeros(n)
    @inbounds for j in 1:n, i in 1:n
        i == j && continue
        r = hypot(P[1, j] - P[1, i], P[2, j] - P[2, i], P[3, j] - P[3, i])
        r < 1e-30 && continue
        yd[j] += x[i] * inv4π / r
    end
    KF = FMM.fmm_laplace3d_matrix(P; eps=1e-8, nmax=16, η=1.0, p=6)
    yf = KF * x
    @test norm(yf - yd) / (norm(yd) + 1e-14) < 5e-4
    # local-expansion gradient vs dense ∇(1/(4πr))
    gd = zeros(3, n)
    @inbounds for j in 1:n, i in 1:n
        i == j && continue
        rx = P[1, j] - P[1, i]
        ry = P[2, j] - P[2, i]
        rz = P[3, j] - P[3, i]
        r2 = rx * rx + ry * ry + rz * rz
        r2 < 1e-30 && continue
        c = -x[i] * inv4π / (r2 * sqrt(r2))
        gd[1, j] += c * rx
        gd[2, j] += c * ry
        gd[3, j] += c * rz
    end
    plan = FMM.build_laplace3d_plan(P; eps=1e-8, nmax=16, η=1.0, p=8)
    gf = zeros(3, n)
    yp = zeros(n)
    FMM.apply_laplace3d!(plan, yp; charges=x, grad=gf)
    @test norm(yp - yd) / (norm(yd) + 1e-14) < 5e-4
    @test norm(gf - gd) / (norm(gd) + 1e-14) < 5e-3
end

@testset "Laplace 3D SL octree FMM on a sphere" begin
    rng = Random.default_rng()
    Random.seed!(rng, 11)
    n = 400
    P = randn(rng, 3, n)
    @inbounds for j in 1:n
        r = hypot(P[1, j], P[2, j], P[3, j])
        s = r < 1e-30 ? 1.0 : 1 / r
        P[1, j] *= s
        P[2, j] *= s
        P[3, j] *= s
    end
    x = randn(rng, n)
    inv4π = 1 / (4π)
    yd = zeros(n)
    @inbounds for j in 1:n, i in 1:n
        i == j && continue
        r = hypot(P[1, j] - P[1, i], P[2, j] - P[2, i], P[3, j] - P[3, i])
        r < 1e-30 && continue
        yd[j] += x[i] * inv4π / r
    end
    plan = FMM.build_laplace3d_plan(P; eps=1e-8, nmax=12, p=8)
    y = zeros(n)
    FMM.apply_laplace3d!(plan, y; charges=x)
    @test norm(y - yd) / (norm(yd) + 1e-14) < 5e-4
end

@testset "Kelvin 3D FMM vs dense U*" begin
    rng = Random.default_rng()
    n = 24
    P = 0.3 .+ rand(rng, 3, n)
    μ, ν = 1.0, 0.3
    props = Elasticity(μ * 2 * (1 + ν), ν, 1.0; plane_strain=true)
    KF = FMM.fmm_kelvin3d_matrix(P; μ=μ, ν=ν, eps=1e-8, nmax=12, full_fmm=true, p=6)
    n0 = Point3D(0.0, 0.0, 1.0)
    D = zeros(3n, 3n)
    @inbounds for j in 1:n, i in 1:n
        i == j && continue
        r = Point3D(P[1, j] - P[1, i], P[2, j] - P[2, i], P[3, j] - P[3, i])
        U, _ = fundamental(props, r, n0)
        D[3*(i-1)+1:3i, 3*(j-1)+1:3j] .= U
    end
    x = randn(rng, 3n)
    @test norm(D * x - KF * x) / (norm(D * x) + 1e-14) < 5e-4
    @test KF[1, 4] ≈ D[1, 4] rtol=1e-10
end
