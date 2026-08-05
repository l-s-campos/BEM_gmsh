# H_G_Hmat with HSS-from-FMM (single + double layer)
using Test
using LinearAlgebra
using BEM

include(joinpath(@__DIR__, "..", "data", "Laplace", "Laplace_dad.jl"))

@testset "double-layer FMM kernel" begin
    msh = Base.invokelatest(quadrado; ndiv=5, show=false, nome="test_dl_fmm")
    dad = format2d(msh, Laplace(1.0); tipo=1, pontointerno=false)
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
    @test norm(transpose(Dd) * x - adjoint(KF) * x) / (norm(transpose(Dd) * x) + 1e-14) < 1e-5
end

@testset "H_G_Hmat format=:hss hss_method=:fmm" begin
    msh = Base.invokelatest(quadrado; ndiv=6, show=false, nome="test_hg_hssfmm")
    dad = format2d(msh, Laplace(1.0); tipo=1, pontointerno=false)
    apply_analytical_bc!(dad, ana_laplace_linear())
    H_G_Hmat(dad; format=:hss, hss_method=:fmm, nmax=14, rtol=1e-5, rank=40, eps=1e-6)
    @test dad.H isa ColWeightedOp
    @test dad.H.K isa HMatrices.HSSMatrix
    @test dad.G_bare_square isa HMatrices.HSSMatrix
    solve(dad)
    @test rel_error(dad) < 0.1
    @test all(isfinite, dad.T)
end
