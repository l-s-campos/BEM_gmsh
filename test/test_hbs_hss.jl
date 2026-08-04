# HBS ≡ HSS nested format + Chapter 18 scattering direct solver
using Test
using LinearAlgebra
using StaticArrays

using BEM  # reexports HMatrices (HSS/HBS, assemble_*, ...)

"""1D Laplace-like kernel on a sorted point set (SPD after shift)."""
function _kernel_matrix(pts::Vector{SVector{1,Float64}})
    n = length(pts)
    A = zeros(n, n)
    @inbounds for j in 1:n, i in 1:n
        A[i, j] = i == j ? 0.0 : log(norm(pts[i] - pts[j]) + 1e-15)
    end
    # shift to ensure invertibility
    A += 5.0 * I
    return A
end

function _points_1d(n::Int)
    return [SVector(i / (n + 1)) for i in 1:n]
end

@testset "HBS ≡ HSS aliases" begin
    @test BEM.HBSMatrix === BEM.HSSMatrix
    @test BEM.HBSBasisID === BEM.HSSBasisID
    @test BEM.assemble_hbs === BEM.assemble_hss
    @test HBSMatrix === HSSMatrix
    @test assemble_hbs === assemble_hss
end

@testset "HSS/HBS matvec + scattering solve" begin
    n = 128
    pts = _points_1d(n)
    A = _kernel_matrix(pts)
    tree = ClusterTree(pts, CardinalitySplitter(; nmax = 16))
    H = assemble_hbs(A, tree; rtol = 1e-8, method = :dense, global_index = true)

    # matvec accuracy
    x = randn(n)
    yH = H * x
    yA = A * x
    @test norm(yH - yA) / norm(yA) < 1e-6

    # Chapter 18 scattering factorization
    Fsc = lu(H; method = :scattering)
    b = A * ones(n)
    xsc = ldiv!(Fsc, copy(b))
    @test norm(A * xsc - b) / norm(b) < 1e-5

    # HODLR-ULV path (legacy)
    Fhod = lu(H; method = :hodlr, rtol = 1e-10)
    xhod = ldiv!(Fhod, copy(b))
    @test norm(A * xhod - b) / norm(b) < 1e-4

    # both solvers agree
    @test norm(xsc - xhod) / norm(xsc) < 1e-3
end

@testset "assemble_structured :HBS" begin
    n = 64
    pts = _points_1d(n)
    A = _kernel_matrix(pts)
    tree = ClusterTree(pts, CardinalitySplitter(; nmax = 12))
    H = assemble_structured(A, tree; format = :HBS, rtol = 1e-8, method = :dense)
    @test H isa HSSMatrix
    @test BEM.HMatrices.maxrank(H) >= 0
    x = randn(n)
    @test norm(H * x - A * x) / norm(A * x) < 1e-5
end
