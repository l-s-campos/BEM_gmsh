# potencial_direto problems (ported from BEM.jl atual)
using Test
using DrWatson
@quickactivate :BEM
using LinearAlgebra

include(datadir("Laplace", "Laplace_dad.jl"))
include(datadir("Laplace", "potencial_problems.jl"))

function _solve_err(msh, ana; keep=false, tipo=1, npg=12)
    dad = format2d(msh, Laplace(1.0); tipo=tipo, pontointerno=false)
    keep ? apply_bc_keep_type!(dad, ana) : attach_analytical!(dad, ana)
    H_G_full_direct(dad; npg=npg, threaded=false)
    solve(dad)
    return rel_error(dad), dad
end

@testset "potencial1d T=x" begin
    err, dad = _solve_err(potencial1d_mesh(; ndiv=10, nome="t_p1d"), ana_potencial1d())
    @test err < 2e-2
    @test all(isfinite, dad.T)
end

@testset "laquini3 Dirichlet top" begin
    err, _ = _solve_err(laquini3_mesh(; ndiv=12, nome="t_laq3"), ana_laquini3())
    @test err < 8e-2
end

@testset "quarto_circ radial" begin
    ana = ana_quarto_circ()
    # point check of analytical
    @test ana.u(Point2D(1.0, 0.0)) ≈ 100.0
    r = 2.0
    T_out = 100.0 - (-200.0) * 2.0 * log(r / 1.0)
    @test ana.u(Point2D(r, 0.0)) ≈ T_out
    err, _ = _solve_err(quarto_circ_mesh(; ndiv=10, nome="t_qcirc"), ana)
    @test err < 8e-2
end

@testset "ana_moulton field" begin
    ana = ana_moulton()
    # on negative x-axis θ=π → T=0
    @test abs(ana.u(Point2D(-1.0, 0.0))) < 1e-12
    # on positive x-axis θ=0 → T=√r
    @test ana.u(Point2D(1.0, 0.0)) ≈ 1.0
    err, _ = _solve_err(
        placa_moulton_mesh(; ndiv=14, nome="t_moulton"),
        ana;
        keep=true,
    )
    @test err < 0.2   # singular tip — qualitative on coarse mesh
end

@testset "laquini series finite" begin
    for ana in (ana_laquini1(), ana_laquini2(), ana_laquini3())
        v = ana.u(Point2D(0.5, 0.5))
        @test isfinite(v)
    end
end
