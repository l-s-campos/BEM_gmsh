# Hierarchical Cholesky, ridge, GMRES+LU precond
using Test
using LinearAlgebra
using BEM
using BEM.HMatrices

function _gauss_pts(nside)
    xs = range(0.0, 1.0; length=nside)
    return [SVector(float(x), float(y)) for y in xs for x in xs]
end

function _spd_kernel(pts; σ=8.0, diag=2.0)
    return KernelMatrix(pts, pts) do x, y
        d2 = sum(abs2, x - y)
        return exp(-σ * d2) + (d2 < 1e-30 ? diag : 0.0)
    end
end

@testset "H Cholesky + ridge" begin
    pts = _gauss_pts(8)
    K = _spd_kernel(pts; diag=1.0)
    tree = ClusterTree(pts, PrincipalComponentSplitter(; nmax=12))
    H = assemble_hmatrix(K, tree, tree;
        adm=StrongAdmissibilityStd(; eta=2.0),
        comp=PartialACA(; rtol=1e-6), threads=false)
    n = length(pts)
    b = randn(n)
    F = cholesky(H; ridge=1e-8, rtol=1e-6)
    x = F \ copy(b)
    @test norm(H * x - b) / (norm(b) + 1e-14) < 5e-4
    xd = Matrix(K) \ b
    @test norm(x - xd) / (norm(xd) + 1e-14) < 1e-2
end

@testset "GMRES + H LU left precond" begin
    pts = _gauss_pts(8)
    K = _spd_kernel(pts)
    tree = ClusterTree(pts, PrincipalComponentSplitter(; nmax=12))
    H = assemble_hmatrix(K, tree, tree;
        adm=StrongAdmissibilityStd(; eta=2.0),
        comp=PartialACA(; rtol=1e-6), threads=false)
    n = length(pts)
    b = randn(n)
    Pl = lu(deepcopy(H); rtol=1e-6)
    x0, st0 = gmres_h(H, b; atol=1e-10, rtol=1e-8, itmax=5n)
    x1, st1 = gmres_h(H, b; Pl=Pl, atol=1e-10, rtol=1e-8, itmax=5n)
    @test st1.niter <= st0.niter
    @test norm(H * x1 - b) / (norm(b) + 1e-14) < 1e-6
    # precond should not need more iters than bare (usually fewer)
    @test st1.niter < 5n
end
