# Contact + arc-length + fatigue + process-zone growth
using Test
using LinearAlgebra
using Statistics: mean
using BEM
using BEM.Crack

@testset "contact law: penalty + Coulomb" begin
    law = BilinearCZM(; σn = 1e6, σt = 1e6, Gn = 50.0, Gt = 50.0,
        δn0 = 1e-6, δt0 = 1e-6, μ = 0.3, kt_contact = 1e11)
    h = CohesiveHistory()
    # pure compression, no slip
    tn, tt, kn, kt, st = evaluate_surface!(law, -1e-6, 0.0, h; kn_pen = 1e12)
    @test st == STATE_CONTACT
    @test tn ≈ -1e6 atol = 1.0
    @test tt ≈ 0 atol = 1e-6
    # stick
    tn, tt, kn, kt, st = evaluate_surface!(law, -1e-6, 1e-9, h; kn_pen = 1e12)
    @test st == STATE_CONTACT
    @test abs(tt) <= 0.3 * abs(tn) + 1e-3
    # forced slip
    tn, tt, kn, kt, st = evaluate_surface!(law, -1e-6, 1e-3, h; kn_pen = 1e12)
    @test st == STATE_CONTACT
    @test abs(tt) ≈ 0.3 * abs(tn) rtol = 1e-5
    @test kt ≈ 0 atol = 1e-12
end

@testset "DBEM compression contact" begin
    mesh, top = contact_compression_mesh(; L = 0.1, n_coh = 3, n_side = 2,
        E = 32e9, ν = 0.2, gap0 = 0.0)
    assemble_dual!(mesh; npg = 10)

    # compress top uy < 0
    uy = -2e-5
    load_dofs = Int[]
    load_u = Float64[]
    for e in top
        el = mesh.elements[e]
        for loc in 1:3
            j = el.fis[loc]
            push!(load_dofs, 2j)
            push!(load_u, uy)
            el.bc_type[loc, 2] = 0
            el.bc_val[loc, 2] = uy
        end
    end

    law = BilinearCZM(; σn = 4e6, σt = 3e6, Gn = 100.0, Gt = 200.0,
        δn0 = 5e-7, μ = 0.2, kt_contact = 1e11)
    prob = CohesiveDBEMProblem(mesh, law; kn_pen = 1e13, tol = 1e-4, maxiter = 30)
    prob.load_dofs = load_dofs
    prob.load_ū = load_u

    u_hist, λ_hist = solve_cohesive_dbem!(prob; nsteps = 4, λ_end = 1.0)
    @test length(u_hist) == 4
    @test all(isfinite, u_hist[end])

    opens = cohesive_openings(prob)
    δn_mean = mean(o[1] for o in opens)
    # under compression, openings should be ≤ 0 (contact)
    @test δn_mean <= 1e-7

    states = [cp.hist.state for cp in prob.pairs]
    @test any(s -> s == STATE_CONTACT, states)

    trs = cohesive_tractions(prob)
    tn_mean = mean(t[1] for t in trs)
    # contact pressure (compression → tn ≤ 0 in our sign)
    @test tn_mean <= 1e3
end

@testset "arc-length method smoke" begin
    mesh, top = modeI_patch_mesh(; L = 0.1, n_coh = 2, n_side = 2, E = 32e9, ν = 0.2)
    assemble_dual!(mesh; npg = 8)
    uy = 2e-5
    load_dofs = Int[]; load_u = Float64[]
    for e in top, loc in 1:3
        j = mesh.elements[e].fis[loc]
        push!(load_dofs, 2j); push!(load_u, uy)
        mesh.elements[e].bc_type[loc, 2] = 0
        mesh.elements[e].bc_val[loc, 2] = uy
    end
    law = BilinearCZM(; σn = 4e6, Gn = 100.0, δn0 = 5e-7, σt = 3e6, Gt = 200.0, δt0 = 5e-7)
    prob = CohesiveDBEMProblem(mesh, law; kn_pen = 1e13, tol = 1e-3, maxiter = 15)
    prob.load_dofs = load_dofs
    prob.load_ū = load_u
    u_hist, λ_hist = solve_cohesive_dbem!(prob; nsteps = 3, method = :arclength,
        Δs = 1e-4, λ_end = 1.0)
    @test length(u_hist) >= 1
    @test all(isfinite, u_hist[end])
    @test all(isfinite, λ_hist)
end

@testset "fatigue damage accumulation" begin
    base = BilinearCZM(; σn = 1e6, Gn = 50.0, δn0 = 1e-6, σt = 1e6, Gt = 50.0, δt0 = 1e-6)
    law = FatigueCZM(; base = base, C = 0.05, m = 1.0, δf_ref = 1e-4)
    h = CohesiveHistory()
    δnf, _ = BEM.Crack._final_openings(base)
    for cyc in 1:20
        evaluate_surface!(law, 0.3 * δnf, 0.0, h)
        fatigue_cycle!(law, h, 0.3 * δnf, 0.0)
    end
    @test h.D > 0
    @test h.D <= 1
    tn1, _, _, _, _ = evaluate_surface!(law, 0.3 * δnf, 0.0, h)
    h0 = CohesiveHistory()
    tn0, _, _, _, _ = evaluate_surface!(base, 0.3 * δnf, 0.0, h0)
    @test abs(tn1) < abs(tn0)   # damaged → softer
end

@testset "process zone extension" begin
    mesh, _ = modeI_patch_mesh(; L = 0.1, n_coh = 2, n_side = 2)
    assemble_dual!(mesh; npg = 6)
    n0 = length(mesh.nodes)
    # pick a tip-ish node on face A
    tip = mesh.elements[mesh.crack_face_a[end]].fis[end]
    mesh.tip_nodes = [tip]
    new_els = extend_cohesive_process_zone!(mesh, tip, 0.02; n_new = 2)
    @test length(mesh.nodes) > n0
    @test !isempty(new_els)
    # twins set on new nodes
    n_twinned = count(nd -> nd.twin != 0, mesh.nodes)
    @test n_twinned >= 2
    assemble_dual!(mesh; npg = 6)
    @test size(mesh.H, 1) == 2 * length(mesh.nodes)
end
