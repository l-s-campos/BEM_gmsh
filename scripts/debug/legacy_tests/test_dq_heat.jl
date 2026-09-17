# Wang §1.9 DQ time (Grid V / Legendre roots) vs Houbolt on collocation+DIBEM heat.
using Test
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Statistics

include(datadir("Laplace", "Laplace_dad.jl"))

function _heat_dad(ndiv)
    msh = quadrado(ndiv=ndiv, show=false, nome="dq_heat")
    dad = format2d(msh, Laplace(1.0); tipo=1, pontointerno=true)
    for i in 1:dad.n
        dad.BC[i] = 0
        dad.BV[i] = 0.0
    end
    H_G_full_direct(dad; npg=10, threaded=false)
    DIBEM(dad; method=:dense, rbf=PHS(3; poly_deg=1))
    return dad
end

_heat_exact(p, t) = exp(-2 * π^2 * t) * sin(π * p[1]) * sin(π * p[2])

function _rmse_final(dad)
    pts = all_points(dad)
    uex = [_heat_exact(p, dad.t[end]) for p in pts]
    return sqrt(mean(abs2, dad.T[:, end] .- uex))
end

@testset "DQ Grid V nodes include ends and Legendre roots" begin
    ξ = BEM._gridV_nodes(5)
    @test ξ[1] == -1.0
    @test ξ[end] == 1.0
    gl, _ = gausslegendre(3)
    @test ξ[2:4] ≈ gl
end

@testset "DQ heat vs Houbolt collocation DIBEM" begin
    tf = 0.02
    dad = _heat_dad(6)
    u0 = [_heat_exact(p, 0.0) for p in all_points(dad)]

    dadH = _heat_dad(6)
    solve_Houbolt_heat(dadH, 0.002, tf; u0=u0)
    rmse_h = _rmse_final(dadH)

    dadQ = _heat_dad(6)
    solve_dq_heat(dadQ, tf; u0=u0, nτ=11, nblocks=1)
    rmse_q = _rmse_final(dadQ)

    @info "heat RMSE" houbolt=rmse_h dq=rmse_q
    @test all(isfinite, dadH.T)
    @test all(isfinite, dadQ.T)
    @test rmse_q < 0.05
    @test rmse_h < 0.4
    @test rmse_q < rmse_h
end
