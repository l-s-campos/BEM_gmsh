# Alternative domain/boundary treatments: one analytic each.
using Test
using LinearAlgebra
using Statistics: mean
using StaticArrays
using BEM

@testset "SBM Chen–Gu T=x" begin
    dad = format2d(quadrado(ndiv=10, show=false, nome="t_sbm", ordem=1), Laplace(1.0);
        tipo=1, pontointerno=false)
    ana = ana_laplace_linear(; direction=SA[1.0, 0.0], k=1.0)
    apply_analytical_bc!(dad, ana)
    d = solve_sbm_laplace(dad)
    @test sbm_rel_error(d, ana) < 0.10
end

@testset "local kernel vanishes at r_i" begin
    ri = 0.4
    @test local_u_star(ri, ri; dim=2) ≈ 0 atol=1e-14
    @test local_du_dr(ri, ri; dim=2) ≈ 0 atol=1e-14
    @test local_ball_volume(ri; dim=2) ≈ π * ri^2
    @test local_u_star(ri, ri; dim=3) ≈ 0 atol=1e-14
    @test local_du_dr(ri, ri; dim=3) ≈ 0 atol=1e-14
    @test local_ball_volume(ri; dim=3) ≈ 4π / 3 * ri^3
end

@testset "local BEM Poisson u=x²+y²" begin
    dad = format2d(quadrado(ndiv=8, show=false, nome="t_lbem"), Laplace(1.0);
        pontointerno=true)
    for i in 1:dad.n
        dad.BC[i] = 0
        dad.BV[i] = dad.Nodes[i][1]^2 + dad.Nodes[i][2]^2
    end
    solve_local_bem!(dad, 4.0; npg=12)
    uex = [p[1]^2 + p[2]^2 for p in dad.Nodes]
    @test sqrt(mean(abs2, dad.T[1:dad.n] .- uex)) < 0.25
end

@testset "cube 3D local BEM Poisson u=|x|²" begin
    include(joinpath(@__DIR__, "..", "data", "Laplace", "cube_mesh.jl"))
    msh = mesh_unit_cube(; L=1.0, ndiv=2, nome="t_lbem3d")
    dad = format3d(msh, Laplace(1.0); pontointerno=false)
    set_internal_nodes!(dad, vec(cube_interior_grid(1.0, 2)))
    ana = ana_poisson_r2(; k=1.0, dim=3)
    apply_analytical_bc!(dad, ana)
    solve_local_bem!(dad, 6.0; npg=8, source=:local)
    @test all(isfinite, dad.T)
    @test all(isfinite, dad.q)
    err = 0.0
    den = 0.0
    @inbounds for k in 1:dad.ni
        p = dad.internalNodes[k]
        ui = dad.T[dad.n + k]
        ue = sum(abs2, p)
        err += (ui - ue)^2
        den += ue^2
    end
    @test sqrt(err / max(den, eps())) < 0.05
    @test rel_error_flux(dad) < 0.08
end

@testset "DRM Poisson u=x²+y²" begin
    dad = format2d(quadrado(ndiv=8, show=false, nome="t_drm"), Laplace(1.0);
        pontointerno=true)
    for i in 1:dad.n
        dad.BC[i] = 0
        dad.BV[i] = dad.Nodes[i][1]^2 + dad.Nodes[i][2]^2
    end
    u = solve_poisson_rbf_bem!(dad, 4.0; method=:global, basis=PHS(3; poly_deg=1), npg=12)
    pts = [dad.Nodes; dad.internalNodes]
    uex = [p[1]^2 + p[2]^2 for p in pts]
    @test sqrt(mean(abs2, u .- uex)) < 0.15
end

@testset "Galerkin V,W symmetric" begin
    dad = format2d(quadrado(ndiv=6, show=false, nome="t_gal"), Laplace(1.0);
        tipo=1, pontointerno=false)
    ops = assemble_galerkin_calderon(dad; npg=8, threaded=false)
    @test norm(ops.V - ops.V') / (norm(ops.V) + 1e-14) < 1e-12
    @test norm(ops.W - ops.W') / (norm(ops.W) + 1e-14) < 1e-12
end

@testset "DLIM T=x" begin
    dad = format2d(quadrado(ndiv=8, show=false, nome="t_dlim", ordem=1), Laplace(1.0);
        tipo=1, pontointerno=false)
    apply_analytical_bc!(dad, ana_laplace_linear(; direction=SA[1.0, 0.0], k=1.0))
    d = solve_dlim_laplace(dad; npg=10, method=:mls)
    @test dlim_rel_error(d, (x, y) -> x) < 0.08
end

@testset "DiBFM T=x" begin
    dad = format2d(quadrado(ndiv=8, show=false, nome="t_dibfm", ordem=1), Laplace(1.0);
        tipo=1, pontointerno=false)
    apply_analytical_bc!(dad, ana_laplace_linear(; direction=SA[1.0, 0.0], k=1.0))
    dib = solve_dibfm_laplace(dad; npg=10, method=:hmls)
    @test dibfm_rel_error(dib, (x, y) -> x) < 0.08
end

@testset "diffuse–advective PDE identity" begin
    m = 2.0
    u = exp_mxy_solution(m)
    p = SVector(0.37, 0.61)
    uu = u(p)
    lap = m^2 * (p[1]^2 + p[2]^2) * uu
    adv = (m * p[2]) * (m * p[2] * uu) + (m * p[1]) * (m * p[1] * uu)
    @test lap ≈ adv rtol=1e-12
end
