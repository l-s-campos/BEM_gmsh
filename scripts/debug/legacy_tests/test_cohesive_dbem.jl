# Cohesive-contact DBEM (Cordeiro 2024 surface states + PPR / bilinear / Alfano–Sacco)
using Test
using LinearAlgebra
using BEM
using BEM.Crack

@testset "cohesive surface states (bilinear)" begin
    law = BilinearCZM(; σn = 4e6, σt = 3e6, Gn = 100.0, Gt = 200.0,
        δn0 = 1e-6, δt0 = 1e-6)
    hist = CohesiveHistory()
    δnf, δtf = BEM.Crack._final_openings(law)

    # contact
    tn, tt, kn, kt, st = evaluate_surface!(law, -1e-7, 0.0, hist)
    @test st == STATE_CONTACT
    @test tn < 0
    @test kn > 0

    # elastic / softening
    hist2 = CohesiveHistory()
    tn, tt, kn, kt, st = evaluate_surface!(law, 0.5 * δnf, 0.0, hist2)
    @test st == STATE_SOFTENING
    @test tn > 0
    @test hist2.δn_max ≈ 0.5 * δnf

    # unload
    tn_u, _, kn_u, _, st_u = evaluate_surface!(law, 0.2 * δnf, 0.0, hist2)
    @test st_u == STATE_UNLOAD
    @test tn_u < tn
    @test kn_u > 0

    # failure
    hist3 = CohesiveHistory()
    _, _, _, _, st_f = evaluate_surface!(law, 1.1 * δnf, 0.0, hist3)
    @test st_f == STATE_FAILED
end

@testset "PPR and Alfano–Sacco evaluate" begin
    ppr = PPRLaw(; Γn = 100.0, Γt = 200.0, σn = 4e6, σt = 3e6, α = 5.0, β = 1.6)
    h = CohesiveHistory()
    tn, tt, kn, kt, st = evaluate_surface!(ppr, 1e-5, 0.0, h)
    @test isfinite(tn) && isfinite(kn)
    @test st in (STATE_SOFTENING, STATE_UNLOAD, STATE_CONTACT)

    as = AlfanoSaccoLaw(; kn = 1e12, kt = 1e12, σn = 3e6, μ = 0.4)
    h2 = CohesiveHistory()
    tn, tt, kn, kt, st = evaluate_surface!(as, 1e-5, 1e-6, h2)
    @test 0 <= h2.D <= 1
    tn_c, _, _, _, st_c = evaluate_surface!(as, -1e-7, 1e-6, h2)
    @test st_c == STATE_CONTACT
    @test tn_c < 0
end

@testset "local/global rotation" begin
    n̂ = SA[0.0, 1.0]
    R = local_to_global_R(n̂)
    Δu = SA[0.1, 0.3]
    δ = opening_local(R, Δu)
    @test δ[1] ≈ 0.3 atol = 1e-14   # normal = y
    @test δ[2] ≈ -0.1 atol = 1e-14  # t = (-ny, nx) = (-1,0) → δt = -Δux
end

@testset "mode-I cohesive patch (smoke)" begin
    mesh, top_elems = modeI_patch_mesh(; L = 0.1, n_coh = 3, n_side = 2,
        E = 32e9, ν = 0.2)
    @test !isempty(mesh.crack_face_a)
    assemble_dual!(mesh; npg = 10, threaded = false)

    # top uy load DOFs
    load_dofs = Int[]
    load_u = Float64[]
    uy = 3e-5
    for e in top_elems
        el = mesh.elements[e]
        for j in el.index
            dof = 2j   # uy
            push!(load_dofs, dof)
            push!(load_u, uy)
            mesh.BC[dof] = 0
            mesh.BV[dof] = uy
        end
    end

    law = BilinearCZM(; σn = 4e6, σt = 3e6, Gn = 100.0, Gt = 200.0,
        δn0 = 5e-7, δt0 = 5e-7)
    prob = CohesiveDBEMProblem(mesh, law; kn_pen = 1e13, tol = 1e-5, maxiter = 25)
    prob.load_dofs = load_dofs
    prob.load_ū = load_u

    @test length(prob.pairs) > 0

    u_hist, λ_hist = solve_cohesive_dbem!(prob; nsteps = 5, λ_end = 1.0)
    @test length(u_hist) == 5
    @test length(λ_hist) == 5
    @test all(isfinite, u_hist[end])

    opens = cohesive_openings(prob)
    @test length(opens) == length(prob.pairs)
    # under tension, average normal opening should be ≥ 0
    δn_mean = mean(o[1] for o in opens)
    @test δn_mean > -1e-6

    trs = cohesive_tractions(prob)
    @test all(isfinite(t[1]) && isfinite(t[2]) for t in trs)
end

@testset "unload/reload path on law only" begin
    law = BilinearCZM(; σn = 1e6, Gn = 50.0, δn0 = 1e-6, σt = 1e6, Gt = 50.0, δt0 = 1e-6)
    h = CohesiveHistory()
    δnf, _ = BEM.Crack._final_openings(law)
    path = Float64[]
    for δ in range(0, 0.6δnf; length = 8)
        tn, _, _, _, _ = evaluate_surface!(law, δ, 0.0, h)
        push!(path, tn)
    end
    # unload
    tn_peak = path[end]
    tn_u, _, _, _, st = evaluate_surface!(law, 0.2δnf, 0.0, h)
    @test st == STATE_UNLOAD
    @test tn_u < tn_peak
    # reload toward peak
    tn_r, _, _, _, st2 = evaluate_surface!(law, 0.55δnf, 0.0, h)
    @test st2 in (STATE_UNLOAD, STATE_SOFTENING)
    @test tn_r > tn_u
end
