# Phase 0: closed-form elastodynamics (no BEM)
using Test
using LinearAlgebra
using Statistics: mean
using BEM

include(joinpath(@__DIR__, "..", "data", "elastico", "iso", "analytical_elastodynamics.jl"))
include(joinpath(@__DIR__, "..", "data", "Laplace", "Laplace_dad.jl"))
include(joinpath(@__DIR__, "..", "data", "Laplace", "potencial_problems.jl"))
include(joinpath(@__DIR__, "..", "data", "Laplace", "wave_propagation.jl"))

@testset "euler_bernoulli_ss_step" begin
    x = 0.5
    @test euler_bernoulli_ss_step(x, 0.0) ≈ 0 atol=1e-14
    t = 0.3
    w = euler_bernoulli_ss_step(x, t; L=1.0, EI=1.0, ρA=1.0, F0=1.0)
    @test w > 0
    ts = range(0, 8; length=400)
    wbar = mean(euler_bernoulli_ss_step.(x, ts; L=1.0, EI=1.0, ρA=1.0, F0=1.0))
    wstat = 1.0^3 / 48
    @test wbar ≈ wstat rtol=0.05
end

@testset "EB SS uniform/sine 2×static and T1" begin
    L, EI, ρA, F0 = 1.0, 1.0, 1.0, 1.0
    T1 = 2L^2 / (π * sqrt(EI / ρA))
    ts = range(0, 4T1; length=800)
    # sine: exactly one mode
    q0 = F0 * π / (2L)
    δsine = q0 * L^4 / (EI * π^4)
    ws = collect(euler_bernoulli_ss_sine.(0.5, ts; L=L, EI=EI, ρA=ρA, F0=F0))
    @test maximum(ws) ≈ 2δsine rtol=0.02
    @test mean(ws) ≈ δsine rtol=0.03
    t1 = ts[argmax(ws[1:length(ts)÷2])]
    @test t1 ≈ T1 / 2 rtol=0.03
    # uniform: first peak near 2 × 5/384
    δu = 5 * (F0 / L) * L^4 / (384 * EI)
    wu = collect(euler_bernoulli_ss_uniform.(0.5, ts; L=L, EI=EI, ρA=ρA, F0=F0))
    @test maximum(wu) ≈ 2δu rtol=0.08
    t1u = ts[argmax(wu[1:length(ts)÷2])]
    @test t1u ≈ T1 / 2 rtol=0.08
end

@testset "EB cantilever point/uniform/sine 2×static and T1" begin
    L, EI, ρA, F0 = 1.0, 1.0, 1.0, 1.0
    T1 = 2π / (1.875104068^2 * sqrt(EI / (ρA * L^4)))
    ts = range(0, 4T1; length=800)
    δp = F0 * L^3 / (3 * EI)
    wp = collect(euler_bernoulli_cantilever.(L, ts; L=L, EI=EI, ρA=ρA, F0=F0, load=:point))
    @test wp[1] ≈ 0 atol=1e-12
    @test maximum(wp) ≈ 2δp rtol=0.08
    tpk(u) = begin
        um = maximum(u)
        for i in 2:length(u)-1
            u[i] >= u[i-1] && u[i] >= u[i+1] && u[i] > 0.5 * um && return ts[i]
        end
        return ts[argmax(u)]
    end
    @test tpk(wp) ≈ T1 / 2 rtol=0.12
    δu = (F0 / L) * L^4 / (8 * EI)
    wu = collect(euler_bernoulli_cantilever.(L, ts; L=L, EI=EI, ρA=ρA, F0=F0, load=:uniform))
    @test maximum(wu) ≈ 2δu rtol=0.08
    @test tpk(wu) ≈ T1 / 2 rtol=0.12
    ws = collect(euler_bernoulli_cantilever.(L, ts; L=L, EI=EI, ρA=ρA, F0=F0, load=:sine))
    @test mean(ws) > 0
    @test maximum(ws) ≈ 2 * mean(ws) rtol=0.08
    @test tpk(ws) ≈ T1 / 2 rtol=0.12
end

@testset "timoshenko_ss_step" begin
    @test timoshenko_ss_step(0.5, 0.0; L=1.0, F0=1e3) ≈ 0 atol=1e-12
    w = timoshenko_ss_step(0.5, 1e-4; L=1.0, F0=1e3)
    @test isfinite(w) && w > 0
end

@testset "timoshenko_cantilever_tip_step" begin
    w0 = timoshenko_cantilever_tip_step(0.0, 0.05)
    @test abs(w0) < 1e-4 * (1 + abs(timoshenko_cantilever_tip_step(1.0, 0.05)))
    wtip = timoshenko_cantilever_tip_step(1.0, 0.05)
    @test isfinite(wtip) && wtip > 0
end

@testset "cylinder_step_pressure" begin
    a, b = 1.0, 2.0
    ust(r) = cylinder_u_static(r; a=a, b=b, E=1.0, ν=0.3, p0=1.0)
    @test cylinder_step_pressure(1.5, 0.0; a=a, b=b) ≈ 0 atol=1e-10
    @test ust(a) > ust(b) > 0
    ts = range(0, 20; length=250)
    umid = mean(cylinder_step_pressure.(1.5, ts; a=a, b=b, N=20))
    @test umid ≈ ust(1.5) rtol=0.15
    @test cylinder_step_pressure(a, 0.2; a=a, b=b) > cylinder_step_pressure(b, 0.2; a=a, b=b)
end

@testset "plate_hole_pressure" begin
    @test plate_hole_pressure(1.5, 0.0) ≈ 0 atol=1e-14
    t = 0.5
    u1 = plate_hole_pressure(1.2, t)
    u2 = plate_hole_pressure(2.5, t)
    @test isfinite(u1) && isfinite(u2)
    @test abs(u1) > abs(u2)
    tg = range(0.05, 1.0; length=8)
    @test all(isfinite, plate_hole_pressure.(1.5, tg))
end

@testset "TOE uniform strip identities" begin
    l, c, q = 0.5, 0.125, 1.0
    E, ν = 1.0, 0.3
    I = toe_beam_I(c)
    @test I ≈ 2 * c^3 / 3
    _, σyc, τc = toe_beam_uniform_stress(0.0, c; l=l, c=c, q=q)
    @test σyc ≈ 0 atol=1e-14
    @test τc ≈ 0 atol=1e-14
    _, σym, τm = toe_beam_uniform_stress(0.0, -c; l=l, c=c, q=q)
    @test σym ≈ -q rtol=1e-12
    @test τm ≈ 0 atol=1e-14
    tx, ty = toe_beam_uniform_t(0.1, c, (0.0, 1.0); l=l, c=c, q=q)
    @test tx ≈ 0 atol=1e-14
    @test ty ≈ 0 atol=1e-14
    txb, tyb = toe_beam_uniform_t(0.1, -c, (0.0, -1.0); l=l, c=c, q=q)
    @test txb ≈ 0 atol=1e-14
    @test tyb ≈ q rtol=1e-12
    ys = range(-c, c; length=401)
    dy = 2c / 400
    accτ = accσ = accM = accτy = 0.0
    for y in ys[1:end-1]
        ym = y + dy / 2
        σx, _, τ = toe_beam_uniform_stress(l, ym; l=l, c=c, q=q)
        accτ += τ * dy
        accσ += σx * dy
        accM += σx * ym * dy
        accτy += τ * ym * dy
    end
    @test accτ ≈ -q * l rtol=1e-3
    @test accσ ≈ 0 atol=1e-6
    @test accM ≈ 0 atol=1e-6
    @test accτy ≈ 0 atol=1e-6
    u0 = toe_beam_uniform_u(0.0, 0.0; l=l, c=c, E=E, ν=ν, q=q)
    @test u0[1] ≈ 0 atol=1e-14
    @test u0[2] ≈ 0 atol=1e-14
    _, vl = toe_beam_uniform_u(l, 0.0; l=l, c=c, E=E, ν=ν, q=q)
    @test u0[2] - vl ≈ toe_beam_uniform_δ(; l=l, c=c, E=E, ν=ν, q=q) rtol=1e-12
    um = toe_beam_uniform_u_mesh(l, c; l=l, c=c, E=E, ν=ν, q=q)
    @test um[1] ≈ 0 atol=1e-14
    @test um[2] ≈ 0 atol=1e-14
    # plane-stress Hooke from centred differences on (u, v)
    h = 1e-6
    x, y = 0.17, 0.04
    ua, va = toe_beam_uniform_u(x, y; l=l, c=c, E=E, ν=ν, q=q)
    ux = (toe_beam_uniform_u(x + h, y; l=l, c=c, E=E, ν=ν, q=q)[1] -
          toe_beam_uniform_u(x - h, y; l=l, c=c, E=E, ν=ν, q=q)[1]) / (2h)
    uy = (toe_beam_uniform_u(x, y + h; l=l, c=c, E=E, ν=ν, q=q)[1] -
          toe_beam_uniform_u(x, y - h; l=l, c=c, E=E, ν=ν, q=q)[1]) / (2h)
    vx = (toe_beam_uniform_u(x + h, y; l=l, c=c, E=E, ν=ν, q=q)[2] -
          toe_beam_uniform_u(x - h, y; l=l, c=c, E=E, ν=ν, q=q)[2]) / (2h)
    vy = (toe_beam_uniform_u(x, y + h; l=l, c=c, E=E, ν=ν, q=q)[2] -
          toe_beam_uniform_u(x, y - h; l=l, c=c, E=E, ν=ν, q=q)[2]) / (2h)
    σx, σy, τ = toe_beam_uniform_stress(x, y; l=l, c=c, q=q)
    @test ux ≈ (σx - ν * σy) / E rtol=1e-6
    @test vy ≈ (σy - ν * σx) / E rtol=1e-6
    @test (uy + vx) ≈ 2 * (1 + ν) * τ / E rtol=1e-6
    # div σ = 0
    hs = 1e-7
    σxp, _, τp = toe_beam_uniform_stress(x + hs, y; l=l, c=c, q=q)
    σxm, _, τm2 = toe_beam_uniform_stress(x - hs, y; l=l, c=c, q=q)
    _, σyp, τyp = toe_beam_uniform_stress(x, y + hs; l=l, c=c, q=q)
    _, σym2, τym = toe_beam_uniform_stress(x, y - hs; l=l, c=c, q=q)
    @test (σxp - σxm) / (2hs) + (τyp - τym) / (2hs) ≈ 0 atol=1e-6
    @test (τp - τm2) / (2hs) + (σyp - σym2) / (2hs) ≈ 0 atol=1e-6
    @test ua ≈ -toe_beam_uniform_u(-x, y; l=l, c=c, E=E, ν=ν, q=q)[1]
    @test va ≈ toe_beam_uniform_u(-x, y; l=l, c=c, E=E, ν=ν, q=q)[2]
end

@testset "TOE cantilever identities" begin
    L, D, E, ν, P = 1.0, 0.25, 1.0, 0.3, 1.0
    u0 = timoshenko_elasticity_cantilever_u(0.0, D / 2; L=L, D=D, E=E, ν=ν, P=P)
    @test u0[1] ≈ 0 atol=1e-14
    @test u0[2] ≈ 0 atol=1e-14
    I = D^3 / 12
    c = D / 2
    tip = timoshenko_elasticity_cantilever_tip(; L=L, D=D, E=E, ν=ν, P=P)
    @test tip ≈ P * L^3 / (3 * E * I) + P * L * (4 + 5ν) * D^2 / (24 * E * I) rtol=1e-12
    # top face n=(0,1): traction 0
    tx, ty = timoshenko_elasticity_cantilever_t(L / 2, D, (0.0, 1.0); L=L, D=D, E=E, ν=ν, P=P)
    @test tx ≈ 0 atol=1e-12
    @test ty ≈ 0 atol=1e-12
    # end shear integrates to P
    ys = range(0, D; length=401)
    acc = 0.0
    dy = D / 400
    for y in ys[1:end-1]
        _, ty = timoshenko_elasticity_cantilever_t(L, y + dy / 2, (1.0, 0.0); L=L, D=D, E=E, ν=ν, P=P)
        acc += ty * dy
    end
    @test acc ≈ P rtol=1e-3
end

@testset "kirsch_static" begin
    σrr, σθθ, σrθ = kirsch_static(1.0, π / 2; a=1.0, σ∞=1.0)
    @test σθθ ≈ 3 rtol=1e-12
    @test σrr ≈ 0 atol=1e-12
    σrr0, _, _ = kirsch_static(1.0, 0.0; a=1.0, σ∞=1.0)
    @test σrr0 ≈ 0 atol=1e-12
end

@testset "transient_kirsch" begin
    @test transient_kirsch(1.0, π / 2, 0.0) ≈ 0 atol=1e-14
    σ = transient_kirsch(1.0, π / 2, 1.0)
    @test isfinite(σ)
    @info "transient_kirsch σθθ(a,π/2) at t=1" σ static=3.0
    if abs(σ) > 100
        @info "transient_kirsch is illustrative; Talbot invert is not a usable benchmark"
    end
end

@testset "ana_bar_sudden" begin
    ana = ana_bar_sudden(; N=400, c=1.0, L=1.0)
    pL = Point2D(1.0, 0.5)
    @test ana.u(pL; t=0.0) ≈ 0 atol=1e-3   # truncated modal series (Gibbs)
    @test ana.u(pL; t=0.5) ≈ 0.5 rtol=0.05
    @test ana.u(pL; t=1.5) ≈ 1.5 rtol=0.05
    ts = range(0, 8; length=400)
    um = mean(ana.u(Point2D(0.4, 0.5); t=t) for t in ts)
    @test um ≈ 0.4 rtol=0.08
end
