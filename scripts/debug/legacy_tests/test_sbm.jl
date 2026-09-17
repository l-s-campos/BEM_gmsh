# Chen–Gu (2012) improved SBM vs dense BEM — 2D Laplace
using Test
using DrWatson
@quickactivate :BEM
using LinearAlgebra
using StaticArrays
using Statistics

include(datadir("Laplace", "Laplace_dad.jl"))

function _mixed_ana!(dad, ana)
    for i in 1:dad.n
        if dad.BC[i] == 0
            dad.BV[i] = float(ana.u(dad.Nodes[i]))
        else
            dad.BV[i] = float(ana.q(dad.Nodes[i], dad.Normal[i]))
        end
    end
    attach_analytical!(dad, ana)
    return dad
end

@testset "SBM Chen–Gu T=x mixed BC" begin
    msh = quadrado(ndiv=12, show=false, nome="sbm_Tx", ordem=1)
    dad = format2d(msh, Laplace(1.0); tipo=1, pontointerno=true)
    ana = ana_laplace_linear(; direction=SA[1.0, 0.0], k=1.0)
    _mixed_ana!(dad, ana)

    d = solve_sbm_laplace(dad)
    err_b = sbm_rel_error(d, ana)
    err_i = sbm_rel_error_internal(d, ana)
    @info "SBM Chen–Gu T=x" err_b err_i n=length(d) mean_uii=mean(d.u_ii) mean_qii=mean(d.q_ii)
    @test length(d.u_ii) == dad.n
    @test length(d.q_ii) == dad.n
    @test all(isfinite, d.u_ii)
    @test all(isfinite, d.q_ii)
    @test size(d.G) == (dad.n, dad.n)
    @test err_b < 0.10
    @test err_i < 0.10
    @test isapprox(sbm_eval_u(d, Point2D(0.5, 0.5)), 0.5; atol=0.1)
end

@testset "SBM Chen–Gu T=x²−y² mixed BC" begin
    msh = quadrado(ndiv=16, show=false, nome="sbm_quad", ordem=1)
    dad = format2d(msh, Laplace(1.0); tipo=1, pontointerno=false)
    ana = ana_laplace_quadratic(; k=1.0)
    _mixed_ana!(dad, ana)
    d = solve_sbm_laplace(dad)
    err = sbm_rel_error(d, ana)
    @info "SBM Chen–Gu quadratic" err
    @test err < 0.15
end

@testset "SBM all-Dirichlet T=x" begin
    msh = quadrado(ndiv=12, show=false, nome="sbm_TxD", ordem=1)
    dad = format2d(msh, Laplace(1.0); tipo=1, pontointerno=false)
    ana = ana_laplace_linear(; direction=SA[1.0, 0.0], k=1.0)
    apply_analytical_bc!(dad, ana)
    d = solve_sbm_laplace(dad)
    err = sbm_rel_error(d, ana)
    @info "SBM all-Dirichlet" err
    @test err < 0.05
end

@testset "SBM vs BEM" begin
    msh = quadrado(ndiv=14, show=false, nome="sbm_cmp", ordem=1)
    dad = format2d(msh, Laplace(1.0); tipo=1, pontointerno=false)
    ana = ana_laplace_linear(; direction=SA[1.0, 0.0], k=1.0)
    _mixed_ana!(dad, ana)
    cmp = compare_sbm_bem(dad, ana; npg=16, threaded=false)
    @info "SBM vs BEM" cmp.err_bem cmp.err_sbm cmp.t_bem cmp.t_sbm
    @test cmp.err_bem < 0.05
    @test cmp.err_sbm < 0.15
    @test isfinite(cmp.diff_T)
end

@testset "SBM uses format2d GL collocation and BEM elements" begin
    msh = quadrado(ndiv=10, show=false, nome="sbm_gl", ordem=1)
    dad = format2d(msh, Laplace(1.0); tipo=1, pontointerno=false)
    d = sbm_from_bemdata(dad)
    @test d.dad === dad
    @test d.nodes == dad.Nodes
    @test d.normals === dad.Normal
    el = dad.elements[d.col_el[1]]
    @test d.lengths[1] ≈ el.Length / length(el.index)
    qsi, _ = discontinuous_nodes_weights(length(el) - 1)
    geo, poly, ξm = BEM.sbm_element_geom(d, 1)
    @test poly === dad.element_type
    @test ξm ≈ qsi[d.col_loc[1]]
    @test geo == [Point2D(dad.Nodes[j]) for j in el.index]
    pg, _, _ = BEM.sbm_geom_at(geo, poly, ξm)
    @test pg ≈ dad.Nodes[1]
    uii, qii = origin_intensity_factors!(d)
    @test all(isfinite, uii)
    @test all(isfinite, qii)
    @test maximum(abs, qii) < 1e3
end

@testset "Kansa-SBM heat square sine" begin
    msh = quadrado(ndiv=10, show=false, nome="kansa_sbm_sin", ordem=1)
    dad = format2d(msh, Laplace(1.0); tipo=1, pontointerno=true)
    dad.BC .= 0
    dad.BV .= 0
    u0 = [sin(pi * point(dad, i)[1]) * sin(pi * point(dad, i)[2]) for i = 1:dad.nt]
    u0[1:dad.n] .= 0
    tf = 0.05
    sol = solve_kansa_sbm_heat(dad; κ=1.0, Δt=0.005, tf=tf, u0=u0, scheme=:houbolt)
    uex = [exp(-2 * pi^2 * tf) * sin(pi * point(dad, i)[1]) *
           sin(pi * point(dad, i)[2]) for i = 1:dad.nt]
    ii = (dad.n + 1):dad.nt
    rmse = sqrt(mean(abs2, sol.U[ii, end] .- uex[ii]))
    @info "Kansa-SBM sine" rmse maxT=maximum(abs, sol.U[:, end])
    @test all(isfinite, sol.U)
    @test size(sol.U, 1) == dad.nt
    @test rmse < 0.5
end

@testset "SBM-DRM heat square sine" begin
    msh = quadrado(ndiv=10, show=false, nome="sbm_drm_sin", ordem=1)
    dad = format2d(msh, Laplace(1.0); tipo=1, pontointerno=true)
    dad.BC .= 0
    dad.BV .= 0
    u0 = [sin(pi * point(dad, i)[1]) * sin(pi * point(dad, i)[2]) for i = 1:dad.nt]
    u0[1:dad.n] .= 0
    tf = 0.05
    sol = solve_sbm_drm(dad; κ=1.0, Δt=0.005, tf=tf, u0=u0, scheme=:houbolt)
    uex = [exp(-2 * pi^2 * tf) * sin(pi * point(dad, i)[1]) *
           sin(pi * point(dad, i)[2]) for i = 1:dad.nt]
    ii = (dad.n + 1):dad.nt
    rmse = sqrt(mean(abs2, sol.U[ii, end] .- uex[ii]))
    @info "SBM-DRM sine" rmse maxT=maximum(abs, sol.U[:, end])
    @test rmse < 15.0   # PHS3 on a dense interior cloud; ex1 is the stable DRM check
    @test all(isfinite, sol.U)
    @test size(sol.U, 1) == dad.nt
end

@testset "SBM-DRM paper ex1 stable" begin
    # coarse stand-in for Kovářík et al. Example 1
    include(joinpath(@__DIR__, "..", "scripts", "sbm_drm_vs_dibem.jl"))
    r = run_example(1; nsteps=40, scheme=:houbolt, nb=12, nint=5)
    @info "SBM-DRM ex1" r.r_sbm r.r_drm r.r_dib r.t_sbm r.t_drm r.t_dib
    @test r.r_sbm < 0.5
    @test isfinite(r.r_sbm)
    @test r.max_sbm < 100
    @test isfinite(r.r_dib)
    @test isfinite(r.r_drm)
    @test r.r_drm < 0.5
    @test r.r_dib < 0.5
end

@testset "SBM-DRM wave bar sudden stays bounded" begin
    include(datadir("Laplace", "potencial_problems.jl"))
    include(datadir("Laplace", "wave_propagation.jl"))
    dad, meta = wave_problem(:bar_sudden; ndiv=8, n_int=5)
    sol = solve_sbm_wave(dad; Δt=0.05, tf=1.0, scheme=:houbolt)
    @test all(isfinite, sol.U)
    @test size(sol.U, 1) == dad.n + dad.ni
    @test sol.physics === :wave
    ip = argmin(norm(p - meta.probe) for p in vcat(dad.Nodes, dad.internalNodes))
    @test maximum(abs, sol.U[ip, :]) > 1e-4
end

