# Double-layer interpolation BEM — Laplace; MLS vs RBF second layer
using Test
using DrWatson
@quickactivate :BEM
include(datadir("Laplace", "Laplace_dad.jl"))

@testset "DLIM Laplace T=x (MLS default)" begin
    msh = quadrado(ndiv=10, show=false, nome="dlim_Tx", ordem=1)
    dad = format2d(msh, Laplace(1.0); tipo=1, pontointerno=false)
    ana = ana_laplace_linear(; direction=SA[1.0, 0.0], k=1.0)
    apply_analytical_bc!(dad, ana)
    H_G_full_direct(dad; npg=12, threaded=false)
    solve(dad)
    err_std = rel_error(dad)

    d = solve_dlim_laplace(dad; npg=12, method=:mls)
    err_dlim = dlim_rel_error(d, (x, y) -> x)
    @info "DLIM MLS" err_dlim err_std n_s=length(d.source_pos) n_v=length(d.virt_global)
    @test d.second_layer == :mls
    @test err_dlim < 0.08
end

@testset "DLIM Laplace T=x (RBF second layer)" begin
    msh = quadrado(ndiv=10, show=false, nome="dlim_Tx_rbf", ordem=1)
    dad = format2d(msh, Laplace(1.0); tipo=1, pontointerno=false)
    ana = ana_laplace_linear(; direction=SA[1.0, 0.0], k=1.0)
    apply_analytical_bc!(dad, ana)
    d = solve_dlim_laplace(dad; npg=12, method=:rbf, rbf=BEM.PHS(3; poly_deg=1))
    err = dlim_rel_error(d, (x, y) -> x)
    @info "DLIM RBF" err second=d.second_layer
    @test d.second_layer == :rbf
    @test err < 0.08
end

@testset "DLIM MLS vs RBF comparison" begin
    msh = quadrado(ndiv=12, show=false, nome="dlim_cmp", ordem=1)
    dad = format2d(msh, Laplace(1.0); tipo=1, pontointerno=false)
    ana = ana_laplace_linear(; direction=SA[1.0, 0.0], k=1.0)
    apply_analytical_bc!(dad, ana)

    cmp = compare_dlim_second_layer(dad, (x, y) -> x; npg=12)
    @info "DLIM compare" cmp.err_mls cmp.err_rbf cmp.diff_max cmp.time_mls cmp.time_rbf
    @test cmp.err_mls < 0.05
    @test cmp.err_rbf < 0.05
    @test isfinite(cmp.diff_max)

    # quadratic harmonic
    msh2 = quadrado(ndiv=14, show=false, nome="dlim_cmp_q", ordem=1)
    dad2 = format2d(msh2, Laplace(1.0); tipo=1, pontointerno=false)
    apply_analytical_bc!(dad2, ana_laplace_quadratic(; k=1.0))
    cmp2 = compare_dlim_second_layer(dad2, (x, y) -> x^2 - y^2; npg=14)
    @info "DLIM compare quadratic" cmp2.err_mls cmp2.err_rbf cmp2.diff_max
    @test cmp2.err_mls < 0.08
    @test cmp2.err_rbf < 0.08
end
