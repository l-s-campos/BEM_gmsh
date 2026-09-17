# Wave-propagation data + full-system Houbolt / OrdinaryDiffEq
using Test
using DrWatson
@quickactivate :BEM
using LinearAlgebra
using Printf
using Statistics

include(datadir("Laplace", "Laplace_dad.jl"))
include(datadir("Laplace", "potencial_problems.jl"))
include(datadir("Laplace", "wave_propagation.jl"))

function _assemble_bar(; ndiv=10, n_int=5, npg=8)
    dad, meta = wave_problem(:bar_sudden; ndiv=ndiv, n_int=n_int)
    H_G_full_direct(dad; npg=npg, threaded=false)
    DIBEM(dad; rbf=PHS(3; poly_deg=0))
    return dad, meta
end

@testset "wave problem factory" begin
    @test length(wave_problem_names()) == 6
    for s in wave_problem_names()
        dad, meta = wave_problem(s; ndiv=6, n_int=3)
        @test dad.n > 0
        @test meta.Δt > 0
    end
end

@testset "ana_bar_sudden table points" begin
    ana = ana_bar_sudden(; N=2000)
    @test ana.u(Point2D(1.0, 0.5); t=0.5) ≈ 0.5 atol=0.02
    @test ana.u(Point2D(1.0, 0.5); t=1.0) ≈ 1.0 atol=0.02
    @test abs(ana.u(Point2D(0.0, 0.5); t=1.0)) < 1e-10
end

@testset "Houbolt does not corrupt H" begin
    dad, _ = _assemble_bar(; ndiv=8, n_int=4)
    H0 = copy(dad.H)
    solve_Houbolt(dad, 0.05, 0.5)
    @test dad.H ≈ H0 rtol=1e-12
    @test all(isfinite, dad.T[:, 1:3])
    @test maximum(abs, dad.T[:, 2]) > 1e-12  # implicit Euler startup
end

@testset "Newmark wave smoke" begin
    dad, _ = _assemble_bar(; ndiv=8, n_int=4)
    H0 = copy(dad.H)
    T = solve_Newmark(dad, 0.05, 0.5)
    @test all(isfinite, T)
    @test dad.H ≈ H0 rtol=1e-12
    @test size(T, 2) == length(0:0.05:0.5)
end

@testset "reduced build_wave_ode" begin
    dad, _ = _assemble_bar(; ndiv=8, n_int=4)
    prob, p = build_wave_ode(dad; tspan=(0.0, 0.2))
    @test hasproperty(p, :B) && hasproperty(p, :f)
    @test size(p.B, 1) == p.n
    @test length(prob.u0) == 2p.n
    ddu = zeros(p.n)
    wave_full_rhs!(ddu, zeros(p.n), zeros(p.n), p, 0.0)
    @test all(isfinite, ddu)
    J = zeros(2p.n, 2p.n)
    BEM.wave_fo_jac!(J, prob.u0, p, 0.0)
    @test J[(p.n + 1):(2p.n), 1:p.n] ≈ I(p.n)
    @test J[1:p.n, (p.n + 1):(2p.n)] ≈ p.B
end

@testset "Houbolt vs DiffEq finite response" begin
    Δt, tf = 0.05, 1.0
    dadH, meta = _assemble_bar(; ndiv=10, n_int=5)
    dadD, _ = _assemble_bar(; ndiv=10, n_int=5)

    solve_Houbolt(dadH, Δt, tf)
    sol = solve_transient_o2(dadD, Δt, tf; abstol=1e-4, reltol=1e-4, progress=false)

    @test all(isfinite, dadH.T)
    @test all(isfinite, dadD.T)
    @test size(dadH.T, 2) == length(0:Δt:tf)

    pts = vcat(dadH.Nodes, dadH.internalNodes)
    ip = argmin(norm(p - Point2D(1.0, 0.5)) for p in pts)
    uH = dadH.T[ip, :]
    uD = dadD.T[ip, :]
    @printf("  free-end max |Houbolt|=%.3e  |DiffEq|=%.3e\n", maximum(abs, uH), maximum(abs, uD))
    # both should produce *some* motion under end load (or stay near zero if M poor —
    # at least no NaN and same length)
    @test length(uH) == length(uD)
    # mutual correlation when both move
    if maximum(abs, uH) > 1e-8 && maximum(abs, uD) > 1e-8
        @test abs(cor(uH, uD)) > 0.2 || norm(uH - uD) / max(norm(uD), eps()) < 2
    end
end

@testset "membrane_v0 IC + DiffEq" begin
    dad, meta = wave_problem(:membrane_v0; ndiv=8, n_int=5, a_v0=0.25)
    H_G_full_direct(dad; npg=8, threaded=false)
    DIBEM(dad; rbf=PHS(3; poly_deg=0))
    du = wave_initial_velocity!(dad, meta)
    @test any(du .== 1) && any(du .== 0)
    sol = solve_transient_o2(dad, 0.02, 0.4; du0=du, abstol=1e-4, reltol=1e-4, progress=false)
    @test all(isfinite, dad.T)
end

@testset "wave_static_operators" begin
    dad, _ = _assemble_bar(; ndiv=8, n_int=4)
    H0 = copy(dad.H)
    ops = wave_static_operators(dad)
    @test dad.H ≈ H0
    @test size(ops.A) == size(dad.H)
    @test length(ops.b) == dad.nt
end
