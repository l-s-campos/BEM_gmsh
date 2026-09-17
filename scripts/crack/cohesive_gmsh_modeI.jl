#!/usr/bin/env julia
# Calibrated cohesive DBEM — Gmsh center-crack plate (load / unload / reload)
#   julia --project=. scripts/cohesive_gmsh_modeI.jl

using Printf
using LinearAlgebra
using Statistics: mean
using StaticArrays
using BEM
using BEM.Crack

function apply_farfield!(mesh, σ)
    eq = mesh.eq_type
    for i in 1:mesh.n
        if eq[i] == 1
            n = mesh.Normal[i]
            if abs(n[2]) > 0.7
                mesh.BC[2i - 1] = 1
                mesh.BC[2i] = 1
                mesh.BV[2i - 1] = 0.0
                mesh.BV[2i] = sign(n[2]) * σ
            end
        elseif eq[i] in (2, 3)
            mesh.BC[2i - 1] = 1
            mesh.BC[2i] = 1
            mesh.BV[2i - 1] = 0.0
            mesh.BV[2i] = 0.0
        end
    end
end

function report(prob, tag)
    opens = cohesive_openings(prob)
    trs = cohesive_tractions(prob)
    δn = mean(first, opens)
    tn = mean(t -> t[1], trs)
    st = [cp.hist.state for cp in prob.pairs]
    @printf("%-16s  ⟨δn⟩=%10.4e  ⟨tn⟩=%10.4e  contact=%d soft=%d unload=%d fail=%d\n",
        tag, δn, tn,
        count(==(STATE_CONTACT), st),
        count(==(STATE_SOFTENING), st),
        count(==(STATE_UNLOAD), st),
        count(==(STATE_FAILED), st),
    )
end

function main()
    println("="^72)
    println("Cohesive DBEM — calibrated center-crack plate")
    println("  E=3000, ν=0.2, a=1, far-field tension σ")
    println("="^72)

    E = 3000.0
    mesh = build_center_crack_mesh(;
        W = 5.0, H = 10.0, a = 1.0, σ = 0.0,
        n_crack = 6, n_bottom = 6, n_top = 6, n_left = 8, n_right = 8,
        E = E, ν = 0.2, plane_strain = true,
    )
    println("nodes=$(mesh.n)  elems=$(length(mesh.elements))")
    assemble_dual!(mesh; npg = 10, threaded = false)

    # linear reference (σ=1, traction-free crack)
    apply_farfield!(mesh, 1.0)
    solve_dual!(mesh; threaded = false)
    pairs0 = build_cohesive_pairs(mesh)
    δ_ref = mean(cp -> begin
        up = SVector(mesh.u[2cp.node_plus-1], mesh.u[2cp.node_plus])
        um = SVector(mesh.u[2cp.node_minus-1], mesh.u[2cp.node_minus])
        dot(cp.n̂, up - um)
    end, pairs0)
    @printf("linear dual reference  ⟨δn⟩=%10.4e  (expect O(σa/E) ~ %.2e)\n\n",
        δ_ref, 1.0 / E)

    law = BilinearCZM(;
        σn = 5.0, σt = 5.0,          # cohesive strength (same units as σ)
        Gn = 2.0, Gt = 2.0,
        δn0 = 5e-4, δt0 = 5e-4,
        μ = 0.15, kt_contact = 1e4,
    )

    function run_σ(σ, tag, prev_hist)
        apply_farfield!(mesh, σ)
        prob = CohesiveDBEMProblem(mesh, law; kn_pen = 1e5, tol = 1e-5, maxiter = 25)
        if prev_hist !== nothing
            for (a, b) in zip(prob.pairs, prev_hist)
                a.hist = deepcopy(b.hist)
            end
        end
        solve_cohesive_dbem!(prob; nsteps = 2, λ_end = 1.0)
        report(prob, tag)
        return prob.pairs
    end

    println("--- LOAD ---")
    h = run_σ(0.5, "load σ=0.5", nothing)
    h = run_σ(1.0, "load σ=1.0", h)
    h = run_σ(2.0, "load σ=2.0", h)
    h = run_σ(4.0, "load σ=4.0", h)   # may enter softening if σ ~ σn

    println("\n--- UNLOAD ---")
    h = run_σ(1.0, "unload σ=1.0", h)

    println("\n--- RELOAD ---")
    h = run_σ(4.0, "reload σ=4.0", h)

    println("\nDone.")
end

main()
