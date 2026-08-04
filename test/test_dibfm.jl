# DiBFM: HMLS vs 2D RBF vs Hermite-RBF second layers
using Test
using DrWatson
@quickactivate :BEM
include(datadir("Laplace", "Laplace_dad.jl"))

@testset "DiBFM-HMLS Laplace T=x" begin
    msh = quadrado(ndiv=10, show=false, nome="dibfm_Tx", ordem=1)
    dad = format2d(msh, Laplace(1.0); tipo=1, pontointerno=false)
    apply_analytical_bc!(dad, ana_laplace_linear(; direction=SA[1.0, 0.0], k=1.0))
    dib = solve_dibfm_laplace(dad; npg=12, method=:hmls)
    err = dibfm_rel_error(dib, (x, y) -> x)
    @info "DiBFM-HMLS" err second=dib.second_layer
    @test dib.second_layer == :hmls
    @test err < 0.08
end

@testset "DiBFM-RBF2d Laplace T=x" begin
    msh = quadrado(ndiv=10, show=false, nome="dibfm_rbf", ordem=1)
    dad = format2d(msh, Laplace(1.0); tipo=1, pontointerno=false)
    apply_analytical_bc!(dad, ana_laplace_linear(; direction=SA[1.0, 0.0], k=1.0))
    dib = solve_dibfm_laplace(dad; npg=12, method=:rbf)
    err = dibfm_rel_error(dib, (x, y) -> x)
    @info "DiBFM-RBF2d" err second=dib.second_layer
    @test dib.second_layer == :rbf
    @test err < 0.08
end

@testset "DiBFM-RBF-Hermite Laplace T=x" begin
    msh = quadrado(ndiv=10, show=false, nome="dibfm_rbfh", ordem=1)
    dad = format2d(msh, Laplace(1.0); tipo=1, pontointerno=false)
    apply_analytical_bc!(dad, ana_laplace_linear(; direction=SA[1.0, 0.0], k=1.0))
    dib = solve_dibfm_laplace(dad; npg=12, method=:rbf_hermite)
    err = dibfm_rel_error(dib, (x, y) -> x)
    @info "DiBFM-RBF-Hermite" err second=dib.second_layer
    @test dib.second_layer == :rbf_hermite
    @test err < 0.10
end

@testset "compare all second layers" begin
    msh = quadrado(ndiv=12, show=false, nome="dibfm_all", ordem=1)
    dad = format2d(msh, Laplace(1.0); tipo=1, pontointerno=false)
    apply_analytical_bc!(dad, ana_laplace_linear(; direction=SA[1.0, 0.0], k=1.0))
    cmp = compare_dlim_dibfm(dad, (x, y) -> x; npg=12)
    @info "compare" cmp.err_std cmp.err_dlim_mls cmp.err_dlim_rbf cmp.err_dibfm_hmls cmp.err_dibfm_rbf cmp.err_dibfm_rbfh
    @test cmp.err_std < 0.05
    @test cmp.err_dlim_mls < 0.05
    @test cmp.err_dibfm_hmls < 0.08
    @test cmp.err_dibfm_rbf < 0.08
    @test cmp.err_dibfm_rbfh < 0.12
end
