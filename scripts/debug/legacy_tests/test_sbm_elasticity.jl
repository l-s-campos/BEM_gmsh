# 2D elastic SBM (Kelvin) — static patch test
using Test
using DrWatson
@quickactivate :BEM
using LinearAlgebra
using StaticArrays

include(datadir("Laplace", "Laplace_dad.jl"))

function _mixed_elast!(dad, ana)
    dim = dad.dimension
    for i in 1:dad.n
        ui = ana.u(dad.Nodes[i])
        ti = ana.q(dad.Nodes[i], dad.Normal[i])
        for d in 1:dim
            k = dim * (i - 1) + d
            dad.BV[k] = dad.BC[k] == 0 ? float(ui[d]) : float(ti[d])
        end
    end
    attach_analytical!(dad, ana)
    return dad
end

@testset "elastic SBM OIFs finite" begin
    msh = quadrado_elasticity(ndiv=8, show=false, nome="esbm_oif")
    dad = format2d(msh, Elasticity(1.0, 0.3, 1.0); pontointerno=false)
    d = sbm_from_bemdata(dad)
    Uii, Tii = origin_intensity_factors!(d)
    @test length(Uii) == dad.n
    @test length(Tii) == dad.n
    @test all(U -> all(isfinite, U), Uii)
    @test all(T -> all(isfinite, T), Tii)
end

@testset "elastic SBM all-Dirichlet patch" begin
    msh = quadrado_elasticity(ndiv=12, show=false, nome="esbm_dir")
    dad = format2d(msh, Elasticity(1.0, 0.3, 1.0); pontointerno=true)
    ana = ana_elasticity_patch(; E=1.0, ν=0.3, εxx=0.01)
    apply_analytical_bc!(dad, ana)
    d = solve_sbm_elasticity(dad)
    err_b = sbm_rel_error(d, ana)
    err_i = sbm_rel_error_internal(d, ana)
    @info "elastic SBM Dirichlet patch" err_b err_i n=length(d)
    @test size(d.G) == (2 * dad.n, 2 * dad.n)
    @test length(d.u) == 2 * dad.n
    @test all(isfinite, d.u)
    @test err_i < 0.15
    uc = sbm_eval_u(d, Point2D(0.5, 0.5))
    @test isapprox(uc[1], 0.005; atol=0.001)
    @test isapprox(uc[2], 0.0; atol=0.001)
end

@testset "elastic SBM mixed BC patch" begin
    msh = quadrado_elasticity(ndiv=12, show=false, nome="esbm_mix")
    dad = format2d(msh, Elasticity(1.0, 0.3, 1.0); pontointerno=false)
    ana = ana_elasticity_patch(; E=1.0, ν=0.3, εxx=0.01)
    _mixed_elast!(dad, ana)
    d = solve_sbm_elasticity(dad)
    err = sbm_rel_error(d, ana)
    @info "elastic SBM mixed patch" err
    @test all(isfinite, d.u)
    @test err < 0.45
end
