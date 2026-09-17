# Cohesive-contact DBEM: bilinear law + mode-I smoke.
using Test
using LinearAlgebra
using Statistics: mean
using StaticArrays
using BEM
using BEM.Crack

@testset "bilinear contact vs tension" begin
    law = BilinearCZM(; σn=4e6, σt=3e6, Gn=100.0, Gt=200.0, δn0=1e-6, δt0=1e-6)
    hist = CohesiveHistory()
    tn_c, _, _, _, st_c = evaluate_surface!(law, -1e-8, 0.0, hist)
    @test st_c == STATE_CONTACT
    @test tn_c < 0
end

@testset "mode-I cohesive patch" begin
    mesh, top_elems = modeI_patch_mesh(; L=0.1, n_coh=3, n_side=2, E=32e9, ν=0.2)
    assemble_dual!(mesh; npg=8, threaded=false)
    load_dofs = Int[]
    load_u = Float64[]
    uy = 3e-5
    for e in top_elems
        for j in mesh.elements[e].index
            dof = 2j
            push!(load_dofs, dof)
            push!(load_u, uy)
            mesh.BC[dof] = 0
            mesh.BV[dof] = uy
        end
    end
    law = BilinearCZM(; σn=4e6, σt=3e6, Gn=100.0, Gt=200.0, δn0=5e-7, δt0=5e-7)
    prob = CohesiveDBEMProblem(mesh, law; kn_pen=1e13, tol=1e-5, maxiter=25)
    prob.load_dofs = load_dofs
    prob.load_ū = load_u
    u_hist, _ = solve_cohesive_dbem!(prob; nsteps=3, λ_end=1.0)
    @test all(isfinite, u_hist[end])
    @test mean(o -> o[1], cohesive_openings(prob)) ≥ -1e-12
end
