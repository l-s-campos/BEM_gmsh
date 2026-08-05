# H-matrix LU factor + multi-RHS + HARA product
using Test
using LinearAlgebra
using BEM

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

@testset "H2 multi-RHS mul!" begin
    pts = _gauss_pts(10)
    K = KernelMatrix(pts, pts) do x, y
        d = norm(x - y)
        d < 1e-14 ? 0.0 : log(d)
    end
    tree = ClusterTree(pts, PrincipalComponentSplitter(; nmax=16))
    H2 = assemble_h2(K, tree; rtol=1e-5, far_method=:aca,
        comp=PartialACA(; rtol=1e-5), alpha=0.5)
    X = randn(length(pts), 4)
    Y = H2 * X
    Yref = hcat((H2 * X[:, j] for j in 1:4)...)
    @test norm(Y - Yref) / (norm(Yref) + 1e-14) < 1e-9
end

@testset "H2 lrdecomp_h2matrix (H2Lib port)" begin
    pts = _gauss_pts(8)
    K = _spd_kernel(pts; σ=6.0, diag=3.0)
    tree = ClusterTree(pts, PrincipalComponentSplitter(; nmax=12))
    H2 = assemble_h2(K, tree; rtol=1e-5, far_method=:aca,
        comp=PartialACA(; rtol=1e-5), alpha=0.5, symmetric=true)
    n = length(pts)
    b = randn(n)
    # H2Lib-named path
    out = lrdecomp_h2matrix(H2; rtol=1e-4, method=:block, threads=false)
    x = lrsolve_h2matrix(out, b)
    r = norm(H2 * x - b) / (norm(b) + 1e-14)
    @test r < 5e-3
    # LinearAlgebra.lu wrapper
    F = lu(deepcopy(H2); rtol=1e-4, threads=false)
    x2 = F \ copy(b)
    @test norm(H2 * x2 - b) / (norm(b) + 1e-14) < 5e-3
    # vs dense
    xd = Matrix(K) \ b
    @test norm(x - xd) / (norm(xd) + 1e-14) < 5e-2
    # conversion sanity
    Hh = h2_to_hmatrix(H2; rtol=1e-4, method=:block)
    v = randn(n)
    @test norm(Hh * v - H2 * v) / (norm(H2 * v) + 1e-14) < 5e-3
end

@testset "HARA product sampler A(Bv)" begin
    pts = _gauss_pts(8)
    Ka = _spd_kernel(pts; σ=6.0, diag=1.5)
    Kb = _spd_kernel(pts; σ=10.0, diag=1.2)
    tree = ClusterTree(pts, PrincipalComponentSplitter(; nmax=12))
    Ha = assemble_hmatrix(Ka, tree, tree;
        adm=StrongAdmissibilityStd(; eta=2.0),
        comp=PartialACA(; rtol=1e-5), threads=false)
    Hb = assemble_hmatrix(Kb, tree, tree;
        adm=StrongAdmissibilityStd(; eta=2.0),
        comp=PartialACA(; rtol=1e-5), threads=false)
    n = length(pts)
    Ad = Matrix(Ha)
    Bd = Matrix(Hb)
    # product sampler: Y = A*(B*X) without forming C = A*B
    Sp = FunctionSampler(
        (Y, X) -> mul!(Y, Ad, Bd * X), n;
        f_adj! = (Y, X) -> mul!(Y, Bd', Ad' * X))
    Hc = hara(Sp, tree, tree;
        adm=StrongAdmissibilityStd(; eta=2.0),
        rtol=1e-3, batch=8, threads=false)
    x = randn(n)
    yC = Hc * x
    yRef = Ad * (Bd * x)
    rel = norm(yC - yRef) / (norm(yRef) + 1e-14)
    @test rel < 0.08
end
