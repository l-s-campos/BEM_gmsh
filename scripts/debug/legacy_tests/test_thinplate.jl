# Fast standalone thin-plate tests
using Test
using DrWatson
@quickactivate :BEM
using BEM.Plate

@testset "thin plate Kirchhoff" begin
    E, ν, h = 1e5, 0.3, 0.01
    a = 1.0
    q0 = 1.0
    props = ThinPlateProps(; E=E, ν=ν, h=h, q_c=q0)
    D = bending_stiffness(props)
    @test D ≈ E * h^3 / (12 * (1 - ν^2))

    w_ana = analytical_wmax_ss_square(; a=a, q=q0, D=D)
    @test w_ana > 0
    # classic coefficient ≈ 0.004062
    @test abs(w_ana * D / (q0 * a^4) - 0.004062) < 5e-5

    # SS plate: free corner forces (Rc=0); clamped corners under-stiffen
    mesh = build_square_plate(; a=a, n_el=6, bc="SSSS", props=props,
        corner_bc='F', n_internal=1)
    @test length(mesh.corners) == 4
    @test length(mesh.internal) == 1
    assemble_plate!(mesh; npg=10)
    @test size(mesh.H, 1) == 2 * length(mesh.nodes) + 1 + 4
    u, t = solve_plate!(mesh)
    w_c = plate_w_int(mesh, 1)
    rel = abs(w_c - w_ana) / w_ana
    @info "SS square plate" w_c w_ana rel
    @test rel < 0.10
    # boundary w ≈ 0 on simply supported edges
    w_b = [plate_w(mesh, i) for i in 1:length(mesh.nodes)]
    @test maximum(abs, w_b) < 0.05 * w_ana + 1e-10

    # clamped plate deflects less than SS
    meshC = build_square_plate(; a=a, n_el=4, bc="CCCC", props=props,
        corner_bc='C', n_internal=1)
    assemble_plate!(meshC; npg=8)
    solve_plate!(meshC)
    @test abs(plate_w_int(meshC)) < abs(w_c)
end
