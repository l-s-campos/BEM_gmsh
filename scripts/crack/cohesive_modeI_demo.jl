#!/usr/bin/env julia
# =============================================================================
# Cohesive DBEM — mode-I patch with unload/reload (Cordeiro et al. 2024 style)
# =============================================================================
# Run:  julia --project=. scripts/cohesive_modeI_demo.jl
# =============================================================================

using LinearAlgebra
using Printf
using BEM
using BEM.Crack

function main()
    println("="^70)
    println("Cohesive-contact DBEM — mode I patch (bilinear CZM)")
    println("Refs: Cordeiro 2024 (DBEM states); Alfano–Sacco 2006 (damage+friction)")
    println("="^70)

    mesh, top = modeI_patch_mesh(; L = 0.1, n_coh = 4, n_side = 3, E = 32e9, ν = 0.2)
    println("nodes=$(mesh.n)  elements=$(length(mesh.elements))  cohesive pairs pending")
    assemble_dual!(mesh; npg = 12, threaded = false)

    # prescribe top uy (loading will scale by λ)
    uy_max = 4e-5
    load_dofs = Int[]
    load_u = Float64[]
    for e in top
        el = mesh.elements[e]
        for j in el.index
            push!(load_dofs, 2j)
            push!(load_u, uy_max)
            mesh.BC[2j] = 0
            mesh.BV[2j] = uy_max
        end
    end

    law = BilinearCZM(;
        σn = 4e6, σt = 3e6,
        Gn = 100.0, Gt = 200.0,
        δn0 = 5e-7, δt0 = 5e-7,
    )
    prob = CohesiveDBEMProblem(mesh, law; kn_pen = 1e13, tol = 1e-5, maxiter = 30, verbose = true)
    prob.load_dofs = load_dofs
    prob.load_ū = load_u
    println("cohesive pairs = ", length(prob.pairs))

    # --- loading ---
    println("\n--- LOADING λ: 0 → 1 ---")
    u_hist, λ_hist = solve_cohesive_dbem!(prob; nsteps = 10, λ_end = 1.0)

    function report(tag)
        opens = cohesive_openings(prob)
        trs = cohesive_tractions(prob)
        δn = mean(o[1] for o in opens)
        tn = mean(t[1] for t in trs)
        states = [cp.hist.state for cp in prob.pairs]
        @printf("%s  ⟨δn⟩=%.3e  ⟨tn⟩=%.3e Pa  states=%s\n",
            tag, δn, tn, string(states))
    end
    report("after load")

    # --- unloading (reduce Dirichlet) ---
    println("\n--- UNLOADING λ: 1 → 0.2 ---")
    # continue from current free DOFs by re-solving smaller λ
    # reset free guess from last solution
    part_dofs = Int[]  # solve_cohesive continues with fresh newton from 1e-12;
    # better: set load to 0.2 * uy and resolve a few steps from current mesh.u
    for e in top
        el = mesh.elements[e]
        for loc in 1:3
            el.bc_val[loc, 2] = 0.2 * uy_max
        end
    end
    prob2 = CohesiveDBEMProblem(mesh, law; kn_pen = 1e13, tol = 1e-5, maxiter = 30)
    # keep history from previous pairs
    for (a, b) in zip(prob2.pairs, prob.pairs)
        a.hist = b.hist
    end
    prob2.load_dofs = load_dofs
    prob2.load_ū = 0.2 .* load_u
    # seed u from previous
    # (solver starts from ~0 free; for demo just re-load)
    solve_cohesive_dbem!(prob2; nsteps = 4, λ_end = 1.0)
    report("after unload")

    # --- reloading ---
    println("\n--- RELOADING λ → full ---")
    for e in top
        el = mesh.elements[e]
        for loc in 1:3
            el.bc_val[loc, 2] = uy_max
        end
    end
    prob3 = CohesiveDBEMProblem(mesh, law; kn_pen = 1e13, tol = 1e-5, maxiter = 30)
    for (a, b) in zip(prob3.pairs, prob2.pairs)
        a.hist = b.hist
    end
    prob3.load_dofs = load_dofs
    prob3.load_ū = load_u
    solve_cohesive_dbem!(prob3; nsteps = 6, λ_end = 1.0)
    report("after reload")

    println("\nDone.")
    return prob3
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
