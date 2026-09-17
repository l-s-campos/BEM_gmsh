# Calibrate cohesive opening sign + magnitude on dual center-crack mesh
using Test
using LinearAlgebra
using Statistics: mean
using BEM
using BEM.Crack

@testset "opening normal sign (linear dual reference)" begin
    mesh = build_center_crack_mesh(;
        W = 5.0, H = 10.0, a = 1.0, σ = 0.0,
        n_crack = 4, n_bottom = 4, n_top = 4, n_left = 6, n_right = 6,
        E = 3000.0, ν = 0.2, plane_strain = true,
    )
    assemble_dual!(mesh; npg = 8, threaded = false)

    # far-field tension σ=1, crack faces traction-free
    σ = 1.0
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
    solve_dual!(mesh; threaded = false)

    pairs = build_cohesive_pairs(mesh)
    @test !isempty(pairs)
    δns = Float64[]
    for cp in pairs
        up = SVector(mesh.u[2cp.node_plus-1], mesh.u[2cp.node_plus])
        um = SVector(mesh.u[2cp.node_minus-1], mesh.u[2cp.node_minus])
        δn = dot(cp.n̂, up - um)
        push!(δns, δn)
    end
    @info "linear dual openings" mean=mean(δns) min=minimum(δns) max=maximum(δns)
    # under tension, cohesive opening must be positive
    @test mean(δns) > 0
    @test minimum(δns) > -1e-6
    # order of magnitude: COD ~ O(σ a / E) = O(1/3000)
    @test mean(δns) < 50 / 3000
    @test mean(δns) > 0.05 / 3000
end

@testset "cohesive solve magnitude under tension" begin
    E = 3000.0
    σ = 1.0
    mesh = build_center_crack_mesh(;
        W = 5.0, H = 10.0, a = 1.0, σ = 0.0,
        n_crack = 4, n_bottom = 4, n_top = 4, n_left = 6, n_right = 6,
        E = E, ν = 0.2, plane_strain = true,
    )
    assemble_dual!(mesh; npg = 8, threaded = false)

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

    # cohesive strength well above σ so response stays nearly elastic opening
    law = BilinearCZM(;
        σn = 100.0, σt = 100.0, Gn = 50.0, Gt = 50.0,
        δn0 = 1e-4, δt0 = 1e-4, μ = 0.0,
    )
    prob = CohesiveDBEMProblem(mesh, law; kn_pen = 1e6, tol = 1e-5, maxiter = 30)
    @test !isempty(prob.pairs)

    solve_cohesive_dbem!(prob; nsteps = 3, λ_end = 1.0)

    opens = cohesive_openings(prob)
    δn = mean(first, opens)
    tn = mean(t -> t[1], cohesive_tractions(prob))
    states = [cp.hist.state for cp in prob.pairs]
    @info "cohesive tension" δn tn states=count(==(STATE_SOFTENING), states) contact=count(==(STATE_CONTACT), states)

    @test δn > 0
    @test δn < 0.05          # not huge (was O(1e2) before sign fix)
    @test abs(tn) < 150.0    # O(σn), not 1e14
    @test abs(tn) > 1e-6
    # under tension: not locked in contact
    @test count(==(STATE_CONTACT), states) < length(states)
end

@testset "cohesive compression → contact" begin
    E = 3000.0
    mesh = build_center_crack_mesh(;
        W = 5.0, H = 10.0, a = 1.0, σ = 0.0,
        n_crack = 3, n_bottom = 3, n_top = 3, n_left = 4, n_right = 4,
        E = E, ν = 0.2, plane_strain = true,
    )
    assemble_dual!(mesh; npg = 6, threaded = false)
    # compression
    eq = mesh.eq_type
    for i in 1:mesh.n
        if eq[i] == 1
            n = mesh.Normal[i]
            if abs(n[2]) > 0.7
                mesh.BC[2i - 1] = 1
                mesh.BC[2i] = 1
                mesh.BV[2i - 1] = 0.0
                mesh.BV[2i] = -sign(n[2]) * 1.0
            end
        elseif eq[i] in (2, 3)
            mesh.BC[2i - 1] = 1
            mesh.BC[2i] = 1
            mesh.BV[2i - 1] = 0.0
            mesh.BV[2i] = 0.0
        end
    end
    law = BilinearCZM(; σn = 10.0, Gn = 1.0, δn0 = 1e-4, σt = 10.0, Gt = 1.0, δt0 = 1e-4, μ = 0.2)
    prob = CohesiveDBEMProblem(mesh, law; kn_pen = 1e5, tol = 1e-4, maxiter = 25)
    solve_cohesive_dbem!(prob; nsteps = 2, λ_end = 1.0)
    opens = cohesive_openings(prob)
    δn = mean(first, opens)
    tn = mean(t -> t[1], cohesive_tractions(prob))
    n_c = count(cp -> cp.hist.state == STATE_CONTACT, prob.pairs)
    @info "cohesive compression" δn tn n_contact=n_c
    @test δn <= 1e-3
    @test tn <= 1e-6   # compression tn ≤ 0
    @test n_c >= 1
    @test abs(tn) < 1e4
end
