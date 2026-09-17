# H-matrix LU factor + multi-RHS
using Test
using LinearAlgebra
using BEM
using BEM.HMatrices

function _gauss_pts(nside)
    xs = range(0.0, 1.0; length=nside)
    return [SVector(float(x), float(y)) for y in xs for x in xs]
end

# SPD-ish smooth kernel + diagonal mass
function _spd_kernel(pts; σ=8.0, diag=2.0)
    return KernelMatrix(pts, pts) do x, y
        d2 = sum(abs2, x - y)
        return exp(-σ * d2) + (d2 < 1e-30 ? diag : 0.0)
    end
end

@testset "H LU solve residual" begin
    pts = _gauss_pts(8)
    K = _spd_kernel(pts)
    tree = ClusterTree(pts, PrincipalComponentSplitter(; nmax=12))
    H = assemble_hmatrix(K, tree, tree;
        adm=StrongAdmissibilityStd(; eta=2.0),
        comp=PartialACA(; rtol=1e-6),
        threads=false)
    n = length(pts)
    b = randn(n)
    F = lu(deepcopy(H); rtol=1e-6)
    x = F \ copy(b)
    # residual in dense sense of hierarchical A
    r = norm(H * x - b) / (norm(b) + 1e-14)
    @test r < 1e-4
    # vs dense
    xd = Matrix(K) \ b
    @test norm(x - xd) / (norm(xd) + 1e-14) < 5e-3
end

@testset "H multi-RHS blocked mul!" begin
    pts = _gauss_pts(10)
    K = _spd_kernel(pts)
    tree = ClusterTree(pts, PrincipalComponentSplitter(; nmax=16))
    H = assemble_hmatrix(K, tree, tree;
        adm=StrongAdmissibilityStd(; eta=2.0),
        comp=PartialACA(; rtol=1e-6),
        threads=false)
    X = randn(length(pts), 5)
    Y = H * X
    Yref = hcat((H * X[:, j] for j in 1:5)...)
    @test norm(Y - Yref) / (norm(Yref) + 1e-14) < 1e-10
    Yd = Matrix(K) * X
    @test norm(Y - Yd) / (norm(Yd) + 1e-14) < 5e-4
end
