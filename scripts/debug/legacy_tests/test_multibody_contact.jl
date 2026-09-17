# Multibody elasticity contact — NTN and NTS
using Test
using LinearAlgebra
using Statistics: mean
using BEM
using BEM.MultiRegion

include(joinpath(@__DIR__, "..", "data", "elastico", "two_blocks_contact.jl"))

@testset "project_point_to_segment2d" begin
    a = Point2D(0.0, 0.0)
    b = Point2D(2.0, 0.0)
    pr = project_point_to_segment2d(Point2D(0.5, 0.1), a, b)
    @test pr.N1 ≈ 0.75 atol = 1e-12
    @test pr.N2 ≈ 0.25 atol = 1e-12
    @test pr.p̄ ≈ Point2D(0.5, 0.0) atol = 1e-12
    # clamp past end
    pr2 = project_point_to_segment2d(Point2D(3.0, 1.0), a, b)
    @test pr2.N2 ≈ 1.0 atol = 1e-12
    @test pr2.p̄ ≈ b atol = 1e-12
end

@testset "NTN pairing + rigid approach" begin
    E, ν = 100.0, 0.3
    props = Elasticity(E, ν, 1.0; plane_strain=true)
    gap0 = 0.02
    prob = load_two_blocks_contact(props;
        W=1.0, H=0.4, gap=gap0, μ=0.25,
        ndiv_bot=5, ndiv_top=5, ndiv_y=3, nome="mb_ntn")

    pairs = pair_contacts!(prob; method=:ntn, slave_reg=1, master_reg=2)
    @test length(pairs) >= 3
    @test all(cp -> cp.method === :ntn, pairs)
    @test all(cp -> cp.gap0 > 0, pairs)
    @test mean(cp.gap0 for cp in pairs) ≈ gap0 rtol = 0.35

    kn = 20 * E / 0.4
    kt = 0.5 * kn
    δ = 0.035   # > gap0 → interference after approach
    solve_multibody_elasticity_contact!(prob; method=:ntn, kn=kn, kt=kt,
        δ=δ, tol=1e-6, maxiter=60, ω=0.45, pair=false, npg=10)

    n_closed = count(cp -> cp.state != 1, prob.contacts)
    @test n_closed >= 1
    tn_closed = [cp.tn for cp in prob.contacts if cp.state != 1]
    @test !isempty(tn_closed)
    @test all(<(0), tn_closed)          # compression
    @test all(isfinite, prob.regions[1].u)
    @test all(isfinite, prob.regions[2].u)
    @test maximum(abs, prob.regions[1].u) < 1.0
    @test maximum(abs, prob.regions[2].u) < 1.0
end

@testset "NTS pairing non-matching + rigid approach" begin
    E, ν = 100.0, 0.3
    props = Elasticity(E, ν, 1.0; plane_strain=true)
    gap0 = 0.02
    # non-matching contact discretizations
    prob = load_two_blocks_contact(props;
        W=1.0, H=0.4, gap=gap0, μ=0.3,
        ndiv_bot=8, ndiv_top=5, ndiv_y=3, nome="mb_nts")

    pairs = pair_contacts!(prob; method=:nts, slave_reg=1, master_reg=2)
    @test length(pairs) >= 4
    @test all(cp -> cp.method === :nts, pairs)
    @test any(cp -> length(cp.master_nodes) == 2, pairs)
    for cp in pairs
        if length(cp.master_nodes) == 2
            @test cp.N1 + cp.N2 ≈ 1.0 atol = 1e-12
            @test -1.0 - 1e-12 <= cp.ξ <= 1.0 + 1e-12
        end
    end

    kn = 20 * E / 0.4
    solve_multibody_elasticity_contact!(prob; method=:nts, kn=kn, kt=0.5*kn,
        δ=0.035, tol=1e-6, maxiter=60, ω=0.45, pair=false, npg=10,
        slave_reg=1, master_reg=2)

    n_closed = count(cp -> cp.state != 1, prob.contacts)
    @test n_closed >= 1
    @test all(isfinite, prob.regions[1].u)
    @test all(isfinite, prob.regions[2].u)
    @test maximum(abs, prob.regions[1].u) < 1.0
    tn_abs = mean(abs(cp.tn) for cp in prob.contacts if cp.state != 1)
    @test tn_abs > 0
end

@testset "NTN vs NTS both run on matching mesh" begin
    E, ν = 80.0, 0.25
    props = Elasticity(E, ν, 1.0; plane_strain=true)
    kn = 15 * E / 0.4
    kwargs = (W=1.0, H=0.4, gap=0.015, μ=0.2,
              ndiv_bot=6, ndiv_top=6, ndiv_y=3)
    δ = 0.03

    p_ntn = load_two_blocks_contact(props; kwargs..., nome="mb_cmp_ntn")
    solve_multibody_elasticity_contact!(p_ntn; method=:ntn, frame=:local, kn=kn, kt=0.5*kn,
        δ=δ, maxiter=50, ω=0.4, npg=10)
    c_ntn = count(cp -> cp.state != 1, p_ntn.contacts)

    p_nts = load_two_blocks_contact(props; kwargs..., nome="mb_cmp_nts")
    solve_multibody_elasticity_contact!(p_nts; method=:nts, frame=:local, kn=kn, kt=0.5*kn,
        δ=δ, maxiter=50, ω=0.4, npg=10, slave_reg=1, master_reg=2)
    c_nts = count(cp -> cp.state != 1, p_nts.contacts)

    @test c_ntn >= 1
    @test c_nts >= 1
    pn = mean(abs(cp.tn) for cp in p_ntn.contacts)
    ps = mean(abs(cp.tn) for cp in p_nts.contacts)
    @test pn > 0 && ps > 0
    @test pn / ps > 0.05 && ps / pn > 0.05
end

@testset "Multi-body fretting AS path A–B–C" begin
    # fretting=true (free top sides), bulk u_x, Contato verify/assemble, ut_lock
    E, ν, μ = 100.0, 0.3, 0.3
    props = Elasticity(E, ν, 1.0; plane_strain=true)
    prob = load_two_blocks_contact(props;
        W=4.0, H=2.0, gap=0.03, μ=μ,
        ndiv_bot=16, ndiv_top=16, ndiv_y=4, nome="mb_fret", fretting=true)
    apply_parabolic_contact_gap!(prob; R=10.0, gap_min=0.0, method=:ntn)
    δn, u_max = 0.04, 0.03

    function Q_of(prob)
        fr = contact_interface_xyτ(prob)
        h = abs(fr.x[2] - fr.x[1])
        cl = abs.(fr.state) .!= 1
        any(cl) || return 0.0
        return sum(.-fr.tt[cl]) * h
    end
    function qrms_of(prob)
        fr = contact_interface_xyτ(prob)
        cl = abs.(fr.state) .!= 1
        any(cl) || return 0.0
        return sqrt(mean(abs2, (.-fr.tt)[cl]))
    end
    function max_util(prob)
        μc = μ
        u = 0.0
        for cp in prob.contacts
            abs(cp.state) == 1 && continue
            u = max(u, abs(cp.tt) / (μc * abs(cp.tn) + 1e-15))
        end
        return u
    end

    # A: normal ramp
    set_farfield_displacement!(prob.regions[2]; ux=0.0, uy=0.0, face=:top)
    xw = nothing
    for δ in range(δn / 4, δn; length=4)
        _, xw = solve_contact_friction!(prob; δ=δ, solver=:activeset, method=:ntn,
            tol=1e-7, maxiter=60, npg=8, x0=xw,
            reset_states=(xw === nothing), return_x=true)
    end
    QA = Q_of(prob)
    @test abs(QA) < 1e-2 * μ * 10 + 1.0   # essentially unloaded tangentially
    @test count(cp -> abs(cp.state) != 1, prob.contacts) >= 2

    # B: bulk +u_x
    for s in 1:4
        ux = u_max * s / 4
        set_farfield_displacement!(prob.regions[2]; ux=ux, uy=0.0, face=:top)
        _, xw = solve_contact_friction!(prob; δ=δn, solver=:activeset, method=:ntn,
            tol=1e-7, maxiter=60, npg=8, x0=xw, reset_states=false, return_x=true)
    end
    QB = Q_of(prob)
    qB = qrms_of(prob)
    @test abs(QB) > 10 * max(abs(QA), 1e-9)
    @test max_util(prob) <= 1.05 + 1e-6

    # C: unload u_x → 0 (residual shear)
    for s in 1:4
        ux = u_max * (1 - s / 4)
        set_farfield_displacement!(prob.regions[2]; ux=ux, uy=0.0, face=:top)
        _, xw = solve_contact_friction!(prob; δ=δn, solver=:activeset, method=:ntn,
            tol=1e-7, maxiter=60, npg=8, x0=xw, reset_states=false, return_x=true)
    end
    QC = Q_of(prob)
    qC = qrms_of(prob)
    @test abs(QC) < 0.5 * abs(QB) + 1e-6
    @test qC > 0.15 * qB          # Mindlin residual
    @test max_util(prob) <= 1.05 + 1e-6
    @test all(isfinite, prob.regions[1].u)
end

@testset "Contato load-stepped active-set" begin
    E, ν = 100.0, 0.3
    props = Elasticity(E, ν, 1.0; plane_strain=true)
    gap0 = 0.02
    δ_end = 0.035
    kwargs = (W=1.0, H=0.4, gap=gap0, μ=0.25, ndiv_bot=4, ndiv_top=4, ndiv_y=3)

    # --- Contato stepped ---
    pC = load_two_blocks_contact(props; kwargs..., nome="mb_contato")
    solve_contact_friction_stepped!(pC; δ_end=δ_end, nsteps=8, tol=1e-6,
        maxiter=40, npg=8, method=:ntn, verbose=false)
    nC = count(cp -> abs(cp.state) != 1, pC.contacts)
    @test nC >= 1
    tnC = [cp.tn for cp in pC.contacts if abs(cp.state) != 1]
    @test !isempty(tnC)
    @test all(<(0), tnC)   # compression
    @test all(isfinite, pC.regions[1].u)
    @test all(isfinite, pC.regions[2].u)
    @test has_cache(pC.regions[1], :contact_δ_hist)
    @test length(pC.regions[1].contact_δ_hist) >= 8

    # one-shot Contato at full δ should also run (may be less robust)
    p1 = load_two_blocks_contact(props; kwargs..., nome="mb_contato1")
    solve_contact_friction!(p1; δ=δ_end, tol=1e-6, maxiter=50, npg=8)
    @test count(cp -> abs(cp.state) != 1, p1.contacts) >= 1

    # --- Penalty at same δ ---
    pP = load_two_blocks_contact(props; kwargs..., nome="mb_pen")
    kn = 30 * E / 0.4
    solve_multibody_elasticity_contact!(pP; method=:ntn, frame=:local,
        kn=kn, kt=0.5*kn, δ=δ_end, tol=1e-6, maxiter=60, ω=0.45, npg=8)
    tnP_list = [abs(cp.tn) for cp in pP.contacts if cp.state != 1]
    @test !isempty(tnP_list)
    tnP = mean(tnP_list)
    tnCs = mean(abs.(tnC))
    @test tnP > 0 && tnCs > 0
    # same order of magnitude (penalty softer → often smaller |tn| for same δ)
    @test tnCs / max(tnP, 1e-12) > 0.05
    @test tnP / max(tnCs, 1e-12) > 0.05
end

@testset "SSN Alart–Curnier unit + FD Jacobian" begin
    # NCF zeros on analytic regimes
    Cn, Ct, reg, _, _, _, _ = alart_curnier(0.1, 0.0, 0.0, 0.0, 0.3, 1e3, 1e3)
    @test reg === :open
    @test Cn ≈ 0 atol = 1e-14
    @test Ct ≈ 0 atol = 1e-14
    # stick: gn=gt=0, compression tn=-2, |tt|<μ|tn|
    Cn, Ct, reg, _, _, _, _ = alart_curnier(0.0, 0.0, -2.0, 0.4, 0.3, 1e3, 1e3)
    @test reg === :stick
    @test Cn ≈ 0 atol = 1e-12
    @test Ct ≈ 0 atol = 1e-12
    # slip piece residual zero when |tt|=μ|tn|, gn=0, λt = -s μ λn
    μ = 0.3
    tn = -2.0
    tt = μ * (-tn)   # positive tt with compression
    # choose gt so τt has sign consistent with slip outward of disk
    gn, gt = 0.0, -1e-3
    rn = rt = 1e3
    Cn, Ct, reg, s, _, _, _ = alart_curnier(gn, gt, tn, tt, μ, rn, rt)
    # may be stick or slip depending on trial; force slip with large gt
    Cn, Ct, reg, s, _, _, _ = alart_curnier(0.0, -0.1, tn, -μ * (-tn), μ, rn, rt)
    @test reg === :slip
    @test abs(Cn) < 1e-10
    @test abs(Ct) < 1e-8

    # FD Jacobian on assembled SSN residual (two blocks)
    E, ν = 100.0, 0.3
    props = Elasticity(E, ν, 1.0; plane_strain=true)
    gap0 = 0.02
    δ = 0.03
    prob = load_two_blocks_contact(props;
        W=1.0, H=0.4, gap=gap0, μ=0.25,
        ndiv_bot=3, ndiv_top=3, ndiv_y=2, nome="mb_ssn_fd")
    ctx = BEM._contact_friction_setup(prob; method=:ntn, npg=8)
    @test ctx !== nothing
    prep, pairs = ctx.prep, ctx.pairs
    h = [cp.gap0 - δ for cp in pairs]
    N = ctx.N
    x = 1e-3 .* randn(N)
    rn, rt = BEM._default_contact_r(prep)
    R, J = BEM._assemble_contact_R_J(prep, pairs, h, x; rn=rn, rt=rt)
    ε = 1e-7
    err = 0.0
    ncheck = min(N, 40)
    for j in 1:ncheck
        x2 = copy(x)
        x2[j] += ε
        R2 = BEM._assemble_contact_R(prep, pairs, h, x2; rn=rn, rt=rt)
        Jcol = (R2 .- R) ./ ε
        err = max(err, norm(Jcol - J[:, j]) / max(1.0, norm(J[:, j])))
    end
    @test err < 5e-4
end

@testset "SSN load-stepped vs Contato active-set" begin
    E, ν = 100.0, 0.3
    props = Elasticity(E, ν, 1.0; plane_strain=true)
    gap0 = 0.02
    δ_end = 0.035
    kwargs = (W=1.0, H=0.4, gap=gap0, μ=0.25, ndiv_bot=4, ndiv_top=4, ndiv_y=3)

    pA = load_two_blocks_contact(props; kwargs..., nome="mb_as")
    solve_contact_friction_stepped!(pA; δ_end=δ_end, nsteps=8, tol=1e-6,
        maxiter=40, npg=8, method=:ntn, solver=:activeset)
    nA = count(cp -> abs(cp.state) != 1, pA.contacts)
    @test nA >= 1
    tnA = mean(abs(cp.tn) for cp in pA.contacts if abs(cp.state) != 1)

    pS = load_two_blocks_contact(props; kwargs..., nome="mb_ssn")
    solve_contact_friction_stepped!(pS; δ_end=δ_end, nsteps=8, tol=1e-6,
        maxiter=40, npg=8, method=:ntn, solver=:ssn)
    nS = count(cp -> abs(cp.state) != 1, pS.contacts)
    @test nS >= 1
    tnS_list = [cp.tn for cp in pS.contacts if abs(cp.state) != 1]
    @test all(<(0), tnS_list)
    tnS = mean(abs.(tnS_list))
    @test all(isfinite, pS.regions[1].u)
    @test all(isfinite, pS.regions[2].u)
    @test tnA > 0 && tnS > 0
    # same order of magnitude on identical mesh/load path
    @test tnA / tnS > 0.2 && tnS / tnA > 0.2
end

@testset "projected Coulomb projections" begin
    # open: positive augmented normal traction
    tn, tt, reg = project_contact_traction(0.0, 0.0, 0.1, 0.0, 0.3, 1e3, 1e3)
    @test reg === :open && tn == 0 && tt == 0
    # stick: closed, small shear
    tn, tt, reg = project_contact_traction(-2.0, 0.1, 0.0, 0.0, 0.3, 1e3, 1e3)
    @test reg === :stick && tn ≈ -2.0 && tt ≈ 0.1
    # slip: large shear projects to Coulomb bound
    tn, tt, reg = project_contact_traction(-2.0, 5.0, 0.0, 0.0, 0.3, 1e3, 1e3)
    @test reg === :slip && tn ≈ -2.0 && abs(tt) ≈ 0.3 * 2.0
    # dual multiplier form
    λn, λt, reg = project_contact_multipliers(0.1, 0.0, 0.0, 0.0, 0.3, 1e3, 1e3)
    @test reg === :open && λn ≈ 0 && λt ≈ 0
    λn, λt, reg = project_contact_multipliers(0.0, 0.0, 2.0, 0.1, 0.3, 1e3, 1e3)
    @test reg === :stick && λn ≈ 2.0 && λt ≈ 0.1
end

@testset "projected GNM / accelerated Newton" begin
    E, ν = 100.0, 0.3
    props = Elasticity(E, ν, 1.0; plane_strain=true)
    gap0 = 0.02
    δ_end = 0.035
    kwargs = (W=1.0, H=0.4, gap=gap0, μ=0.25, ndiv_bot=4, ndiv_top=4, ndiv_y=3)

    pA = load_two_blocks_contact(props; kwargs..., nome="mb_as_rt")
    solve_contact_friction_stepped!(pA; δ_end=δ_end, nsteps=8, tol=1e-6,
        maxiter=40, npg=8, method=:ntn, solver=:activeset)
    tnA = mean(abs(cp.tn) for cp in pA.contacts if abs(cp.state) != 1)
    @test tnA > 0

    pG = load_two_blocks_contact(props; kwargs..., nome="mb_rt_g")
    solve_contact_friction_stepped!(pG; δ_end=δ_end, nsteps=8, tol=1e-6,
        maxiter=40, npg=8, method=:ntn, solver=:proj_gnm)
    nG = count(cp -> abs(cp.state) != 1, pG.contacts)
    @test nG >= 1
    tnG = mean(abs(cp.tn) for cp in pG.contacts if abs(cp.state) != 1)
    @test tnA / tnG > 0.15 && tnG / tnA > 0.15

    # Accelerated Newton (paper §6.1)
    pN = load_two_blocks_contact(props; kwargs..., nome="mb_rt_n")
    solve_contact_friction_stepped!(pN; δ_end=δ_end, nsteps=8, tol=1e-6,
        maxiter=40, npg=8, method=:ntn, solver=:proj_newton)
    nN = count(cp -> abs(cp.state) != 1, pN.contacts)
    @test nN >= 1
    tnN = mean(abs(cp.tn) for cp in pN.contacts if abs(cp.state) != 1)
    @test all(<(0), (cp.tn for cp in pN.contacts if abs(cp.state) != 1))
    @test tnA / tnN > 0.15 && tnN / tnA > 0.15
    @test all(isfinite, pN.regions[1].u)
    @test all(isfinite, pN.regions[2].u)

    pL = load_two_blocks_contact(props; kwargs..., nome="mb_rt_gnmls")
    solve_contact_friction_stepped!(pL; δ_end=δ_end, nsteps=8, tol=1e-6,
        maxiter=40, npg=8, method=:ntn, solver=:gnmls)
    nL = count(cp -> abs(cp.state) != 1, pL.contacts)
    @test nL >= 1
    tnL = mean(abs(cp.tn) for cp in pL.contacts if abs(cp.state) != 1)
    @test all(<(0), (cp.tn for cp in pL.contacts if abs(cp.state) != 1))
    @test tnA / tnL > 0.15 && tnL / tnA > 0.15
    @test all(isfinite, pL.regions[1].u)
    @test all(isfinite, pL.regions[2].u)

    pBad = load_two_blocks_contact(props; kwargs..., nome="mb_bad_solver")
    @test_throws ArgumentError solve_contact_friction!(pBad; δ=δ_end, solver=:alm)


    # RT Newton without quasi-complementarity reduction (open pins t1 and t2)
    pNr = load_two_blocks_contact(props; kwargs..., nome="mb_rt_n_noreduce")
    ctx = BEM._contact_friction_setup(pNr; method=:ntn, npg=8)
    @test ctx !== nothing
    prep, pairs = ctx.prep, ctx.pairs
    h = [cp.gap0 - δ_end for cp in pairs]
    rn0, rt0 = BEM._default_contact_r(prep)
    xR, okR = BEM._contact_proj_newton!(prep, pairs, h, zeros(ctx.N);
        tol=1e-6, maxiter=40, rn=rn0, rt=rt0, reduce=false)
    nR = norm(BEM._assemble_proj_gnm_R(prep, pairs, h, xR; rn=rn0, rt=rt0))
    @test okR || nR < 1e-4
    @test all(isfinite, xR)
end

@testset "local vs global frame multibody" begin
    E, ν = 100.0, 0.3
    props = Elasticity(E, ν, 1.0; plane_strain=true)
    kn = 20 * E / 0.4
    δ = 0.035
    kwargs = (W=1.0, H=0.4, gap=0.02, μ=0.25, ndiv_bot=5, ndiv_top=5, ndiv_y=3)

    pL = load_two_blocks_contact(props; kwargs..., nome="mb_frame_L")
    solve_multibody_elasticity_contact!(pL; method=:ntn, frame=:local, kn=kn, kt=0.5*kn,
        δ=δ, maxiter=60, ω=0.45, npg=10)
    pG = load_two_blocks_contact(props; kwargs..., nome="mb_frame_G")
    solve_multibody_elasticity_contact!(pG; method=:ntn, frame=:global, kn=kn, kt=0.5*kn,
        δ=δ, maxiter=60, ω=0.45, npg=10)

    cL = count(cp -> cp.state != 1, pL.contacts)
    cG = count(cp -> cp.state != 1, pG.contacts)
    @test cL >= 1 && cG >= 1
    tnL = mean(abs(cp.tn) for cp in pL.contacts)
    tnG = mean(abs(cp.tn) for cp in pG.contacts)
    @test tnL > 0 && tnG > 0
    # same order of magnitude (staggered penalty is not bit-identical)
    @test tnL / tnG > 0.2 && tnG / tnL > 0.2
    @test has_cache(pL.regions[1], :u_local)   # local path stores (n,t) fields
    @test maximum(abs, pL.regions[1].u) < 1.0
    @test maximum(abs, pG.regions[1].u) < 1.0
end
