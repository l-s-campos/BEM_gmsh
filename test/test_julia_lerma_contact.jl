# Juliá Lerma (2025) half-space contact — coarse-mesh gates
using Test
using LinearAlgebra
using BEM
using BEM.Contact

@testset "Juliá Lerma half-space" begin

    @testset "combined moduli" begin
        G, ν = 80.769, 0.3          # steel G ≈ E/2(1+ν) with E=210
        hs1 = ElasticHalfSpace(G, ν)
        hs2 = combined_halfspace(G, ν, G, ν)
        @test hs2.K ≈ 0 atol=1e-14
        @test hs2.G ≈ G / 2
        @test hs2.ν ≈ ν
        # rigid second body → single-body Pohrt
        hsR = combined_halfspace(G, ν, 1e12 * G, ν)
        @test hsR.G ≈ G rtol=1e-10
        @test hsR.K ≈ (1 - 2ν) / (4G) rtol=1e-8
        @test influence_coeff(Kzz, 0, 0, hs1) ≈ influence_coeff(Kzz, 0, 0, hsR) rtol=1e-8
        # identical pair has twice the deflection of one body
        @test influence_coeff(Kzz, 0, 0, hs2) ≈ 2 * influence_coeff(Kzz, 0, 0, hs1) rtol=1e-12
    end

    @testset "Kxz/Kyz are odd (no atan2 branch-cut leak)" begin
        hs = combined_halfspace(80.0, 0.3, 8000.0, 0.3; hx=0.1, hy=0.1)
        @test influence_coeff(Kxz, 0, 0, hs) ≈ 0 atol=1e-18
        @test influence_coeff(Kyz, 0, 0, hs) ≈ 0 atol=1e-18
        for di in 1:6
            @test influence_coeff(Kxz, di, 0, hs) ≈ -influence_coeff(Kxz, -di, 0, hs) rtol=1e-12
            @test influence_coeff(Kyz, 0, di, hs) ≈ -influence_coeff(Kyz, 0, -di, hs) rtol=1e-12
        end
    end

    @testset "coupled FFT u = A p" begin
        G, ν = 1.0, 0.3
        nx = ny = 8
        hs = ElasticHalfSpace(G, ν; hx=0.25, hy=0.25)
        prep = precompute_kernels(nx, ny, hs)
        px = zeros(nx, ny); py = zeros(nx, ny); pn = zeros(nx, ny)
        pn[4, 4] = 1.0
        ux, uy, uz = fc_displacements(px, py, pn, prep)
        @test uz ≈ fc_forward(pn, Kzz, prep) rtol=1e-12
        @test ux ≈ fc_forward(pn, Kxz, prep) rtol=1e-12
        # identical bodies: coupling K=0 ⇒ ux from pn vanishes
        hs2 = combined_halfspace(G, ν, G, ν; hx=0.25, hy=0.25)
        prep2 = precompute_kernels(nx, ny, hs2; components=(Kzz, Kxz, Kzx))
        ux2, _, uz2 = fc_displacements(px, py, pn, prep2)
        @test maximum(abs, ux2) < 1e-14
        @test uz2 ≈ 2 .* uz rtol=1e-12
    end

    @testset "Hertz sphere Uzawa + Johnson VM" begin
        E, ν = PIN.E, PIN.ν
        G = G_from_E(E, ν)
        R, δ = PIN.R, PIN.δ
        N = 21
        L = 0.6
        hx = L / N
        hs = combined_halfspace(G, ν, G, ν; hx=hx, hy=hx)
        x = collect(range(-L / 2 + hx / 2, stop=L / 2 - hx / 2, length=N))
        y = copy(x)
        gap = sphere_gap(x, y, R)
        grid = make_grid(x, y, hs, gap)
        prep = precompute_kernels(N, N, hs)
        st = init_state(grid)
        law = isotropic_law(0.0, 0.0)
        niter, Ψ = solve_contact_step!(st, grid, prep, law, δ, 0.0, 0.0; tol=1e-8)
        Estar = contact_modulus(hs)
        hz = hertz_sphere(R, δ, Estar)
        P, _, _ = contact_resultants(st, hs)
        @test P > 0
        @test abs(P - hz.P) / hz.P < 0.2
        @test abs(maximum(st.pn) - hz.p0) / hz.p0 < 0.25
        # Johnson on-axis VM at z = 0.48 a
        z = 0.48 * hz.a
        _, _, σVM_ana = hertz_axis_stress(z, hz.a, hz.p0, ν)
        @test σVM_ana / hz.p0 ≈ 0.62 atol=0.02
        σ = subsurface_stress(0.0, 0.0, z, st.ptx, st.pty, st.pn, x, y, hs, ν)
        @test σ.VM / hz.p0 ≈ σVM_ana / hz.p0 rtol=0.25
        @info "Hertz Uzawa" niter Ψ P_ratio=P/hz.P pmax_ratio=maximum(st.pn)/hz.p0 VM_ratio=σ.VM/hz.p0
    end

    @testset "Argatov sliding wear (few steps)" begin
        E, ν = 2.10e5, 0.3
        G = E / (2(1 + ν))
        R, δ = 50.0, 4.5e-4
        i_w = 1.33e-7
        N = 17
        L = 0.8
        hx = L / N
        hs = combined_halfspace(G, ν, G, ν; hx=hx, hy=hx)
        x = collect(range(-L / 2 + hx / 2, stop=L / 2 - hx / 2, length=N))
        grid = make_grid(x, x, hs, sphere_gap(x, x, R))
        prep = precompute_kernels(N, N, hs)
        st = init_state(grid)
        law = isotropic_law(0.0, i_w)
        hz = hertz_sphere(R, δ, contact_modulus(hs))
        Δs = 5.0
        nsteps = 4
        hist = sliding_wear_steps!(st, grid, prep, law, δ, Δs, nsteps; tol=1e-6, maxiter=200)
        s = nsteps * Δs
        a_arg = argatov_worn_radius(hz.a, i_w, hz.P, s, R)
        w_arg = a_arg^2 / (2R)
        @test hist.wmax[end] > 0
        @test hist.wmax[end] / w_arg ≈ 1 rtol=0.5   # coarse mesh, few steps
        @info "Argatov" w_num=hist.wmax[end] w_arg a_num=hist.a[end] a_arg
    end

    @testset "force-controlled Argatov wear (no crater)" begin
        E, ν = 2.10e5, 0.3
        G = E / (2(1 + ν))
        R, δ = 50.0, 4.5e-4
        i_w = 1.33e-7
        N = 17
        L = 0.8
        hx = L / N
        hs = combined_halfspace(G, ν, G, ν; hx=hx, hy=hx)
        x = collect(range(-L / 2 + hx / 2, stop=L / 2 - hx / 2, length=N))
        grid = make_grid(x, x, hs, sphere_gap(x, x, R))
        prep = precompute_kernels(N, N, hs)
        st = init_state(grid)
        law = isotropic_law(0.0, i_w)
        hz = hertz_sphere(R, δ, contact_modulus(hs))
        nsteps = 6
        Δs = 8.0
        hist = sliding_wear_force_steps!(st, grid, prep, law, hz.P, Δs, nsteps;
                                         δ0=δ, tol=1e-6, maxiter=200, rtol=3e-2,
                                         maxouter=8)
        @test length(hist.s) == nsteps + 1
        @test hist.s[1] == 0
        @test hist.wmax[1] == 0
        @test all(p -> abs(p - hz.P) / hz.P < 0.2, hist.P)
        @test all(diff(hist.wmax) .> 0)          # centre depth keeps growing
        @test hist.a[end] > hist.a[1]
        @test hist.pmean[end] < hist.pmean[1]    # flattening / area growth
        ag = argatov2011(hist.s[end], hz.a, R, i_w, hz.P)
        @test hist.wmax[end] / ag.H ≈ 1 rtol=0.55
        @info "force Argatov" w=hist.wmax[end] H=ag.H a=hist.a[end] a_arg=ag.a[end] P=hist.P[end]
    end

    @testset "Sneddon flat punch (frictionless)" begin
        E, ν = PUNCH.E_A, PUNCH.ν
        G = G_from_E(E, ν)
        a0, P = PUNCH.a0, PUNCH.P
        N = 25
        L = PUNCH.L
        hx = L / N
        hs = combined_halfspace(G, ν, 100G, ν; hx=hx, hy=hx)
        x = collect(range(-L / 2 + hx / 2, stop=L / 2 - hx / 2, length=N))
        grid = make_grid(x, x, hs, flat_punch_gap(x, x, a0; out=10.0))
        prep = precompute_kernels(N, N, hs)
        st = init_state(grid)
        law = isotropic_law(0.0, 0.0)
        Estar = contact_modulus(hs)
        # Sneddon: P = 2 E* a0 δ  (E* = combined)
        δ = P / (2 * Estar * a0)
        set_approach_for_load!(st, grid, prep, law, P, 0.0, 0.0; δ0=δ, tol=1e-6, rtol=5e-3)
        Pn, _, _ = contact_resultants(st, hs)
        @test abs(Pn - P) / P < 0.08
        # peak at the edge, centre finite
        jc = (N + 1) ÷ 2
        @test st.pn[jc, jc] > 0
        @test maximum(st.pn) > 1.5 * st.pn[jc, jc]
    end

    @testset "rolling spheres isotropic smoke" begin
        N = 17
        L = 2 * ROLL.L
        hx = L / N
        hs = combined_halfspace(ROLL.G, ROLL.ν, ROLL.G, ROLL.ν; hx=hx, hy=hx)
        x = collect(range(-L / 2 + hx / 2, stop=L / 2 - hx / 2, length=N))
        grid = make_grid(x, x, hs, sphere_gap(x, x, ROLL.R / 2))
        prep = precompute_kernels(N, N, hs)
        hz = hertz_sphere_load(ROLL.R / 2, ROLL.P, contact_modulus(hs))
        law = isotropic_law(ROLL.μ, 0.0)
        st = init_state(grid)
        δ, _, _ = set_approach_for_load!(st, grid, prep, law, ROLL.P, 0.0, 0.0;
                                         δ0=hz.δ, tol=1e-6, wear_jump=0, maxouter=25)
        kin = RollingKinematics(1.0, ROLL.ξx, 0.0, 0.0)
        niter, Ψ = solve_rolling_step!(st, grid, prep, law, kin, δ; wear=false, tol=1e-6)
        P, Qx, Qy = contact_resultants(st, hs)
        @test P > 0
        @test abs(Qy) < 0.05 * max(abs(Qx), 1e-12)
        @test Qx * (-kin.ξx) > 0   # traction opposes creepage
        rn, rt = default_penalties(hs)
        @test rn > 0 && rt > 0
        @info "rolling smoke" niter Ψ Qx_over_μP=Qx/(ROLL.μ * P) Qy
    end
end
