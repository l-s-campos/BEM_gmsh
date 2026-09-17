# Wheel–rail (CONTACT module 1 planar D=2) — geometry gates vs mbench_a22_left case 1.
using Test
using LinearAlgebra
using BEM
using BEM.Contact

const DATA = joinpath(@__DIR__, "..", "data", "contact", "vollebregt")

@testset "Wheel–rail profiles and Manchester A-2.2 case 1" begin
    @testset "SIMPACK profile load" begin
        rail = read_rail_profile(joinpath(DATA, "MBench_UIC60_v3.prr"))
        wheel = read_wheel_profile(joinpath(DATA, "MBench_S1002_v3.prw"))
        @test length(rail.spl) > 100
        @test length(wheel.spl) > 100
        @test !rail.is_wheel && wheel.is_wheel
        yg0, zmin, _ = gauge_meas_pt(rail.spl, 0.0, 14.0)
        @test zmin ≈ 0 atol=1e-3
        @test yg0 < -20          # inner face of the head
    end

    @testset "locate Y=0 at CONTACT z_ws" begin
        rail = read_rail_profile(joinpath(DATA, "MBench_UIC60_v3.prr"))
        wheel = read_wheel_profile(joinpath(DATA, "MBench_S1002_v3.prw"))
        trk = TrackGeom()
        ws = WheelsetGeom(z=0.1981, vs=2000.0, vpitch=-4.34811810)
        sgn = -1.0
        m_rail, _, _ = set_rail_marker(trk, rail, sgn)
        m_w, m_ws = set_wheel_markers(ws, sgn)
        cps = locate_patches(rail, wheel, m_rail, m_w, m_ws, ws, sgn; dx=0.2, ds=0.2)
        @test length(cps) == 1
        cp = cps[1]
        @test cp.gap_min ≈ -0.016 atol=0.008
        @test sgn * oy(cp.mref) ≈ -751.87 atol=4.0     # mm
        @test oz(cp.mref) ≈ 0.125 atol=0.08
        @test abs(sgn * cp.delttr) ≈ 0.029 atol=0.02
        V, ξx, ξy, φ = creepage_at_patch(cp, ws, m_w, m_ws, sgn)
        @test V ≈ 2000 atol=1
        @test abs(ξx) < 1e-4   # pure rolling at VS/|ω|
        @test abs(ξy) < 1e-4
        @test abs(φ) < 2e-4
    end
end
