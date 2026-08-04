# Verify CAS closed-form radial integrals against high-order Gauss
using Test
using LinearAlgebra
using BEM

function check_kernel(name, basis, Rs, dims; rtol = 1e-10)
    @testset "$name" begin
        for dim in dims, R in Rs
            a = radial_integral(basis, R; dim = dim)
            g = radial_integral_gauss(basis, R; dim = dim, n = 96)
            @test isfinite(a) && isfinite(g)
            # relative if g large, else absolute
            if abs(g) > 1e-8
                @test a ≈ g rtol = rtol
            else
                @test a ≈ g atol = 1e-12
            end
        end
    end
end

@testset "analytical radial_integral vs Gauss" begin
    Rs = [0.1, 0.5, 1.0, 2.0, 5.0]
    dims = (2, 3)

    for n in 1:7
        check_kernel("PHS$n", PHS(n), Rs, dims)
    end

    for ε in (0.5, 1.0, 2.0)
        check_kernel("IMQ(ε=$ε)", IMQ(ε), Rs, dims; rtol = 1e-9)
        check_kernel("MQ(ε=$ε)", MQ(ε), Rs, dims; rtol = 1e-9)
        check_kernel("Gaussian(ε=$ε)", Gaussian(ε), Rs, dims; rtol = 1e-9)
    end

    # Wendland: R inside and outside support.
    # Gauss is slightly less accurate at R=δ (integrand C^0 kink); use looser rtol.
    for s in (1.0, 2.5)
        Rs_w = [0.2 * s, 0.7 * s, s, 1.5 * s, 3s]
        check_kernel("WendlandC2(δ=$s)", WendlandC2(s), Rs_w, dims; rtol = 1e-7)
        check_kernel("WendlandC4(δ=$s)", WendlandC4(s), Rs_w, dims; rtol = 1e-7)
        check_kernel("WendlandC6(δ=$s)", WendlandC6(s), Rs_w, dims; rtol = 1e-7)
    end
end

@testset "NearestNeighbors kNN" begin
    pts = Point2D[Point2D(randn(), randn()) for _ in 1:200]
    x = Point2D(0.0, 0.0)
    k = 7
    ids = rbf_neighbors(x, pts, k)
    @test length(ids) == k
    # distances nondecreasing
    ds = [norm(pts[i] - x) for i in ids]
    @test issorted(ds)
    # tree form matches
    tree = BEM._rbf_kdtree(pts)
    ids2 = rbf_neighbors(tree, x, k)
    @test ids == ids2
    # local fit uses tree
    y = rand(length(pts))
    lr = local_rbf_fit(pts, PHS(3; poly_deg = 1); k = 10)
    v = local_rbf_eval(lr, x, y)
    @test isfinite(v)
end

@testset "production path has no gauss fallback for known kernels" begin
    # should not throw
    @test radial_integral(IMQ(1.0), 1.3; dim = 2) > 0
    @test radial_integral(MQ(1.0), 1.3; dim = 3) > 0
    @test radial_integral(Gaussian(1.0), 0.8; dim = 2) > 0
    @test radial_integral(WendlandC2(1.0), 0.5; dim = 2) > 0
    @test radial_integral(WendlandC2(1.0), 2.0; dim = 2) == radial_integral(WendlandC2(1.0), 1.0; dim = 2)
end
