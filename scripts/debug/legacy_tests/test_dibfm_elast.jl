# DiBFM elasticity (Zhang et al. EJMS 2019)
using Test
using DrWatson
@quickactivate :BEM
include(datadir("Laplace", "Laplace_dad.jl"))

@testset "DiBFM elasticity patch MLS" begin
    E, ν, εxx = 1.0, 0.3, 0.01
    msh = quadrado_elasticity(ndiv=10, show=false, nome="dibfm_el_mls", ordem=1)
    dad = format2d(msh, Elasticity(E, ν, 1.0; plane_strain=true); tipo=1, pontointerno=false)
    ana = ana_elasticity_patch(; E=E, ν=ν, εxx=εxx)
    apply_analytical_bc!(dad, ana)

    # standard
    H_G_full_direct(dad; npg=12, threaded=false)
    solve(dad)
    err0 = rel_error(dad)

    d = solve_dibfm_elasticity(dad; npg=12, method=:mls)
    uana = (x, y) -> SA[εxx * x, -ν * εxx * y / (1 - ν)]  # plane strain uy
    # plane strain: εyy = 0 if constrained; patch ana uses plane strain kinematics
    # use numerical from standard solution mean scale — better: evaluate ana field
    function u_from_ana(x, y)
        # ana_elasticity_patch stores displacement function
        return SA[ana.u(Point2D(x, y))[1], ana.u(Point2D(x, y))[2]]
    end
    err = dibfm_elast_rel_error(d, u_from_ana)
    @info "DiBFM elast MLS" err err0 n_s=length(d.source_pos) n_v=length(d.virt_global)
    @test d.second_layer == :mls
    @test err < 0.15
end

@testset "DiBFM elasticity patch RBF" begin
    E, ν, εxx = 1.0, 0.3, 0.01
    msh = quadrado_elasticity(ndiv=10, show=false, nome="dibfm_el_rbf", ordem=1)
    dad = format2d(msh, Elasticity(E, ν, 1.0; plane_strain=true); tipo=1, pontointerno=false)
    ana = ana_elasticity_patch(; E=E, ν=ν, εxx=εxx)
    apply_analytical_bc!(dad, ana)
    d = solve_dibfm_elasticity(dad; npg=12, method=:rbf)
    u_from_ana = (x, y) -> SA[ana.u(Point2D(x, y))[1], ana.u(Point2D(x, y))[2]]
    err = dibfm_elast_rel_error(d, u_from_ana)
    @info "DiBFM elast RBF" err
    @test d.second_layer == :rbf
    @test err < 0.15
end

@testset "compare elasticity BEM / DiBFM" begin
    E, ν = 1.0, 0.3
    msh = quadrado_elasticity(ndiv=8, show=false, nome="dibfm_el_cmp", ordem=1)
    dad = format2d(msh, Elasticity(E, ν, 1.0; plane_strain=true); tipo=1, pontointerno=false)
    ana = ana_elasticity_patch(; E=E, ν=ν, εxx=0.01)
    apply_analytical_bc!(dad, ana)
    uana = (x, y) -> SA[ana.u(Point2D(x, y))[1], ana.u(Point2D(x, y))[2]]
    cmp = compare_dibfm_elasticity(dad, uana; npg=10)
    @info "elast compare" cmp.err_std cmp.err_dibfm_mls cmp.err_dibfm_rbf
    @test cmp.err_std < 0.15
    @test cmp.err_dibfm_mls < 0.15
    @test cmp.err_dibfm_rbf < 0.15
end
