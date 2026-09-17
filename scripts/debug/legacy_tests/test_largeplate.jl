# Fast large-plate (von Kármán) test
using Test
using DrWatson
@quickactivate :BEM
using BEM.Plate

@testset "large plate von Karman" begin
    E, ν, h, a = 1e5, 0.3, 0.05, 1.0
    q0 = 50.0
    props = ThinPlateProps(; E=E, ν=ν, h=h, q_c=q0)
    plate = build_square_plate(; a=a, n_el=3, bc="SSSS", props=props,
        corner_bc='F', n_internal=1)
    assemble_plate!(plate; npg=8)

    include(datadir("Laplace", "Laplace_dad.jl"))
    msh = quadrado_elasticity(ndiv=4, show=false, nome="test_large_pe", Lx=a, Ly=a)
    dad_pe = format2d(msh, Elasticity(E, ν, 1.0; plane_strain=false); pontointerno=false)
    fill!(dad_pe.BC, 0)
    fill!(dad_pe.BV, 0.0)
    H_G_full_direct(dad_pe; npg=8, threaded=false)

    prob = build_large_plate_problem(plate, dad_pe; npg_plate=8, npg_pe=8)
    @test membrane_stiffness_CB(E, ν, h) ≈ E * h / (1 - ν^2)

    w_lin = linear_wmax_reference(prob)
    @test isfinite(w_lin) && abs(w_lin) > 0

    res = solve_large_plate!(prob; nsteps=4, λ_max=1.0, e_relax=0.5,
        abstol=1e-6, reltol=1e-6, maxiters=30)
    @test length(res.λ) == 4
    @test length(res.w_center) == 4
    @test all(isfinite, res.w_center)
    ratio = abs(res.w_center[end]) / abs(w_lin)
    # finite solution; stiffening expected under larger loads
    @test isfinite(res.w_center[end])
    @test sign(res.w_center[end]) == sign(w_lin) || abs(res.w_center[end]) < abs(w_lin)
    @info "large plate" w_lin=w_lin w_nl=res.w_center[end] ratio=ratio
end
