# Nested H² HARA vs proxy assemble_h2
using Test
using LinearAlgebra
using BEM

function _pts(nside)
    xs = range(0.0, 1.0; length=nside)
    return [SVector(float(x), float(y)) for y in xs for x in xs]
end

function _logK(pts)
    return KernelMatrix(pts, pts) do x, y
        d = norm(x - y)
        return d < 1e-14 ? 0.0 : log(d)
    end
end

@testset "hara_h2 vs assemble_h2 matvec" begin
    pts = _pts(10)
    K = _logK(pts)
    tree = ClusterTree(pts, PrincipalComponentSplitter(; nmax=16))
    H2e = assemble_h2(K, tree; rtol=1e-5, far_method=:aca,
        comp=PartialACA(; rtol=1e-5), alpha=0.5)
    S = KernelMatvecSampler(K)
    H2h = hara_h2(S, tree; rtol=1e-4, nsample=48, alpha=0.5,
        orthog=true, compress=true)
    @test H2h isa H2Matrix
    @test size(H2h) == size(H2e)
    x = randn(length(pts))
    ye = H2e * x
    yh = H2h * x
    yd = Matrix(K) * x
    # HARA H² should track dense better than random; allow looser than entry H²
    rel_h = norm(yh - yd) / (norm(yd) + 1e-14)
    rel_e = norm(ye - yd) / (norm(yd) + 1e-14)
    @test rel_h < 0.15
    @test rel_e < 0.05
    # format= dispatch
    H2b = hara(S, tree; format=:H2, rtol=1e-4, nsample=32, alpha=0.5)
    @test H2b isa H2Matrix
end

@testset "hara_h2 product sampler" begin
    pts = _pts(8)
    K = KernelMatrix(pts, pts) do x, y
        d2 = sum(abs2, x - y)
        exp(-8 * d2) + (d2 < 1e-30 ? 2.0 : 0.0)
    end
    tree = ClusterTree(pts, PrincipalComponentSplitter(; nmax=12))
    H = assemble_hmatrix(K, tree, tree;
        adm=StrongAdmissibilityStd(; eta=2.0),
        comp=PartialACA(; rtol=1e-5), threads=false)
    n = length(pts)
    # C ≈ H*H via nested H² HARA on sampler v ↦ H(Hv)
    S = FunctionSampler(
        (Y, X) -> begin
            Z = H * X
            mul!(Y, H, Z)
        end,
        n;
        f_adj! = (Y, X) -> begin
            Z = H' * X
            mul!(Y, H', Z)
        end,
    )
    H2c = hara_h2(S, tree; rtol=1e-3, nsample=40, alpha=0.5)
    x = randn(n)
    y = H2c * x
    yref = H * (H * x)
    @test norm(y - yref) / (norm(yref) + 1e-14) < 0.2
end
