# H-matrix algebra: matvec, hmul, hlru, hadd, h2 orthog
using Test
using LinearAlgebra
using Statistics: mean
using BEM
using BEM.HMatrices

function _laplace_pts(nside)
    xs = range(0.0, 1.0; length=nside)
    return [SVector(float(x), float(y)) for y in xs for x in xs]
end

function _laplace_kernel(pts)
    return KernelMatrix(pts, pts) do x, y
        d = norm(x - y)
        return d < 1e-14 ? 0.0 : log(d)
    end
end

@testset "H matvec vs dense" begin
    pts = _laplace_pts(10)
    K = _laplace_kernel(pts)
    tree = ClusterTree(pts, PrincipalComponentSplitter(; nmax=16))
    H = assemble_hmatrix(K, tree, tree;
        adm=StrongAdmissibilityStd(2.0),
        comp=PartialACA(; rtol=1e-6),
        threads=false)
    Kd = Matrix(K)
    x = randn(length(pts))
    yH = H * x
    yD = Kd * x
    rel = norm(yH - yD) / (norm(yD) + 1e-14)
    @test rel < 5e-4
    @test compression_ratio(H) > 0   # small n may not beat dense storage
    # multi-RHS
    X = randn(length(pts), 3)
    YH = H * X
    YD = Kd * X
    @test norm(YH - YD) / (norm(YD) + 1e-14) < 5e-4
end

@testset "hlru! rank-k update" begin
    pts = _laplace_pts(8)
    K = _laplace_kernel(pts)
    tree = ClusterTree(pts, PrincipalComponentSplitter(; nmax=12))
    H = assemble_hmatrix(K, tree, tree;
        adm=StrongAdmissibilityStd(2.0),
        comp=PartialACA(; rtol=1e-5),
        threads=false)
    n = length(pts)
    k = 2
    X = randn(n, k)
    Y = randn(n, k)
    A0 = Matrix(H)
    hlru!(H, X, Y; rtol=1e-8)
    A1 = Matrix(H)
    Aref = A0 + X * Y'
    rel = norm(A1 - Aref) / (norm(Aref) + 1e-14)
    @test rel < 5e-3
end

@testset "hadd! compatible structure" begin
    pts = _laplace_pts(8)
    K = _laplace_kernel(pts)
    tree = ClusterTree(pts, PrincipalComponentSplitter(; nmax=12))
    kwargs = (adm=StrongAdmissibilityStd(2.0), comp=PartialACA(; rtol=1e-5), threads=false)
    A = assemble_hmatrix(K, tree, tree; kwargs...)
    B = assemble_hmatrix(K, tree, tree; kwargs...)
    C = assemble_hmatrix(K, tree, tree; kwargs...)  # same structure
    hadd!(C, A, B, 1.0, 1.0; rtol=1e-8)
    rel = norm(Matrix(C) - (Matrix(A) + Matrix(B))) / (norm(Matrix(A)) + 1e-14)
    @test rel < 5e-3
end


