using Test
using DrWatson
@quickactivate :BEM

# geometry helpers
include(datadir("Laplace", "Laplace_dad.jl"))

println("Starting BEM.jl tests")
ti = time()

@testset "BEM.jl" begin

    @testset "Newton closest-point projection" begin
        poly = BEM.Equispaced(1)   # nodes at ξ=±1
        nodes = [Point2D(0.0, 0.0), Point2D(1.0, 0.0)]
        # point above midpoint → ξ≈0, dist≈0.1
        ξ, x, d = closest_point_1d(poly, nodes, Point2D(0.5, 0.1))
        @test abs(x[1] - 0.5) < 1e-6
        @test d ≈ 0.1 atol=1e-6
        # outside segment → endpoint
        ξ2, x2, d2 = closest_point_1d(poly, nodes, Point2D(-1.0, 0.0))
        @test ξ2 ≈ -1.0 atol=1e-8
        @test d2 ≈ 1.0 atol=1e-8
    end

    @testset "AD-compatible heat RHS" begin
        include(datadir("Laplace", "Laplace_dad.jl"))
        msh = quadrado(ndiv=6, show=false, nome="ad_quad")
        dad = format2d(msh, Laplace(1.0); pontointerno=true)
        H_G_full_direct(dad; npg=8, threaded=false)
        DIBEM(dad)
        prob, sys = build_heat_ode(dad; tspan=(0.0, 0.05))
        u = prob.u0 .+ 0.01
        du = heat_rhs(u, prob.p, 0.0)
        @test length(du) == length(u)
        @test all(isfinite, du)
        # ForwardDiff gradient of ‖rhs‖²
        using ForwardDiff
        g = u -> sum(abs2, heat_rhs(u, prob.p, 0.0))
        ∇g = ForwardDiff.gradient(g, u)
        @test all(isfinite, ∇g)
        @test length(∇g) == length(u)
    end

    @testset "crack SIF criteria & Paris" begin
        # pure mode I → θ=0, KIeq=KI
        θ, KIeq = max_tens_circ(2.0, 0.0)
        @test abs(θ) < 1e-12
        @test KIeq ≈ 2.0
        # pure mode II
        θ2, _ = max_tens_circ(0.0, -1.0)
        @test θ2 ≈ acos(1/3) atol=1e-10
        # Tanaka + Paris positive
        dK = tanaka_deltaK(10.0, 0.0, 0.0)
        @test dK ≈ 10.0
        dN = paris_cycles(1e-12, 3.0, 10.0, 11.0, 0.1)
        @test dN > 0
        # analytical center crack
        KI = analytical_KI_center_crack(1.0, 1.0)
        @test KI ≈ sqrt(π)
        # extend tip geometry
        tip = CrackTip(1, 1, 1, Point2D(1.0, 0.0), Point2D(-1.0, 0.0))
        path = CrackPath([Point2D(-1.0, 0.0), Point2D(1.0, 0.0)], [tip])
        extend_crack_tip!(path, 1, 0.0, 0.5)
        @test norm(path.tips[1].pos - Point2D(1.5, 0.0)) < 1e-12
    end

    @testset "dual BEM center crack" begin
        mesh = build_center_crack_mesh(; W=5.0, H=10.0, a=1.0, σ=1.0,
            E=3000.0, ν=0.2, n_bottom=4, n_right=6, n_top=4, n_left=6, n_crack=6)
        @test all(e -> true, mesh.elements)  # discontinuous by construction
        @test any(e -> e.eq_type == 2, mesh.elements)
        @test any(e -> e.eq_type == 3, mesh.elements)
        # coincident twins (no face offset)
        nA = mesh.nodes[mesh.elements[mesh.crack_face_a[1]].fis[2]]
        nB = mesh.nodes[nA.twin]
        @test norm(nA.pos - nB.pos) < 1e-14
        assemble_dual!(mesh; npg=8)
        solve_dual!(mesh)
        KI_L, KII_L = sif_cod_dual(mesh, mesh.tip_nodes[1])
        KI_R, KII_R = sif_cod_dual(mesh, mesh.tip_nodes[2])
        KIana = analytical_KI_center_crack(1.0, 1.0; W=5.0)
        KIn = 0.5 * (abs(KI_L) + abs(KI_R))
        @test abs(KII_L) < 0.05 * KIana   # pure mode I
        @test abs(KIn - KIana) / KIana < 0.15  # ≤15% on coarse mesh
        @info "dual BEM center crack" KIn KIana rel=abs(KIn-KIana)/KIana
    end

    @testset "thermoelasticity constrained Δθ" begin
        E, ν, α, Δθ = 1000.0, 0.3, 1e-5, 50.0
        props = Elasticity(E, ν, 1.0; plane_strain=true, α=α)
        k̂ = thermal_modulus(props)
        @test k̂ ≈ E * α / (1 - 2ν)
        @test analytical_constrained_thermal_stress(props, Δθ) ≈ -k̂ * Δθ

        msh = quadrado_elasticity(ndiv=6, show=false, nome="test_thermo")
        dad = format2d(msh, props; pontointerno=false)
        # fully constrained (Dirichlet u=0)
        fill!(dad.BC, 0)
        fill!(dad.BV, 0.0)
        H_G_full_direct(dad; npg=10, threaded=false)
        u = solve_thermoelastic!(dad; θ=Δθ)
        @test maximum(abs, u) < 1e-8
        # recovered traction ≈ -k̂ θ n
        t = dad.traction
        err = 0.0
        for i in 1:dad.n
            t_ana = -k̂ * Δθ * dad.Normal[i]
            err = max(err, abs(t[2i-1] - t_ana[1]), abs(t[2i] - t_ana[2]))
        end
        @test err / (abs(k̂ * Δθ) + eps()) < 0.15
        @info "thermoelasticity constrained" k̂ err_rel=err/(abs(k̂*Δθ)+eps())
    end

    @testset "structures & analytical" begin
        # q = -k ∂T/∂n ; T=x ⇒ q(n=êx) = -1
        ana = ana_laplace_linear(; direction=SA[1.0, 0.0], k=1.0)
        @test ana(Point2D(0.5, 0.3)) ≈ 0.5
        @test ana.q(Point2D(1, 0), Point2D(1, 0)) ≈ -1.0
        @test occursin("laplace_linear", sprint(show, ana))
    end

    @testset "fundamental solutions" begin
        r = Point2D(0.3, 0.4)
        n = Point2D(1.0, 0.0)
        R = norm(r)

        # Laplace 2D — match closed form
        lap = Laplace(1.0)
        kp = fundamental(lap, r, n)
        @test kp.U ≈ -log(R) / (2π)
        @test kp.T ≈ dot(r, n) / (R^2 * 2π)
        # BEMdata-style tuple
        dadL = format2d(quadrado(ndiv=4, show=false, nome="fund_lap"), lap; pontointerno=false)
        G, H = fundamental(dadL, r, n)
        @test G ≈ kp.U
        @test H ≈ kp.T

        # Laplace hypersingular finite
        kh = fundamental_hyper(lap, r, n, Point2D(0.0, 1.0))
        @test isfinite(kh.U) && isfinite(kh.T)

        # Laplace 3D
        kp3 = fundamental(lap, Point3D(0.3, 0.4, 0.5), Point3D(0, 0, 1))
        @test kp3.U > 0 && isfinite(kp3.T)

        # Helmholtz complex, Sommerfeld (Im G ≥ 0 for exp(-iωt) / H¹ convention)
        helm = Helmholtz(; ω=2.0, c=1.0)
        kph = fundamental(helm, r, n)
        @test kph.U isa Complex
        @test isfinite(real(kph.U)) && isfinite(imag(kph.U))

        # Kelvin 2D — symmetry of U, finite T
        el = Elasticity(1.0, 0.3, 1.0)
        # Lamé cached at construction (plane strain default)
        λ0, μ0 = lame_constants(1.0, 0.3, true)
        @test el.mu ≈ μ0
        @test el.lambda ≈ λ0
        @test shear_modulus(el) === el.mu
        @test lame_λ(el) === el.lambda
        @test el.plane_strain && !plane_stress(el)
        el_ps = Elasticity(1.0, 0.3, 1.0; plane_stress=true)
        @test plane_stress(el_ps) && !el_ps.plane_strain
        @test el_ps.mu ≈ el.mu          # μ from material ν
        @test el_ps.lambda != el.lambda # λ uses effective ν
        @test effective_nu(el_ps) ≈ 0.3 / 1.3
        el.E = 2.0
        @test el.mu ≈ 2.0 / (2 * 1.3)
        kpe = fundamental(el, r, n)
        @test kpe.U isa AbstractMatrix
        @test kpe.U ≈ kpe.U' atol=1e-14
        @test isfinite(sum(abs, kpe.T))
        # stress kernels
        sk = fundamental_stress(el, r, n)
        @test size(sk.D) == (2, 2, 2)
        @test sk.D[1, 1, 2] ≈ sk.D[1, 2, 1]   # minor symmetry
        # gradients
        Ux, Tx, Uy, Ty = fundamental_grad(el, r, n)
        @test size(Ux) == (2, 2)

        # Kelvin 3D
        kpe3 = fundamental(el, Point3D(1, 0, 0), Point3D(1, 0, 0))
        @test size(kpe3.U) == (3, 3)
        @test kpe3.U ≈ kpe3.U' atol=1e-12

        # Lekhnitskii isotropic limit roughly matches order of magnitude
        # (orthotropic with E1=E2, ν12=ν, G12=E/2(1+ν))
        E, ν = 1.0, 0.3
        G12 = E / (2(1 + ν))
        pars = lekhnitskii_params(E, E, G12, ν)
        @test imag(pars.mi[1]) > 0 && imag(pars.mi[2]) > 0
        an = AnisotropicElasticity(pars)
        kpa = fundamental(an, r, zero(r), n)
        @test size(kpa.U) == (2, 2)
        @test isfinite(sum(abs, kpa.U)) && isfinite(sum(abs, kpa.T))
    end

    @testset "square Laplace steady (dense)" begin
        msh = quadrado(ndiv=12, show=false, nome="test_quad_dense")
        props = Laplace(1.0)
        dad = format2d(msh, props; pontointerno=true)
        @test dad.n > 0
        @test dad.dimension == 2
        @test any(==(0), dad.BC)   # has Dirichlet
        @test any(==(1), dad.BC)   # has Neumann

        ana = ana_laplace_linear(; direction=SA[1.0, 0.0])
        attach_analytical!(dad, ana)

        H_G_full_direct(dad, 16)
        @test has_cache(dad, :H)
        @test size(dad.H, 1) == dad.nt
        @test size(dad.G, 2) == dad.n

        solve(dad)
        @test length(dad.T) == dad.nt

        err = rel_error(dad)
        @test err < 0.05   # < 5 % on coarse mesh
        @info "dense Laplace rel_error" err
    end

    @testset "square Laplace steady (H-matrix)" begin
        msh = quadrado(ndiv=16, show=false, nome="test_quad_hmat")
        props = Laplace(1.0)
        dad = format2d(msh, props; pontointerno=false)
        ana = ana_laplace_linear(; direction=SA[1.0, 0.0])
        attach_analytical!(dad, ana)

        H_G_Hmat(dad; atol=1e-5, nmax=16, threads=false)
        @test dad.H isa HMatrix
        @test dad.G isa HMatrix

        solve(dad)
        err = rel_error(dad)
        @test err < 0.08
        @info "H-matrix Laplace rel_error" err compressionH = compression_ratio(dad.H)
    end

    @testset "elasticity BC path" begin
        msh = quadrado_elasticity(ndiv=8, show=false, nome="test_quad_elast")
        props = Elasticity(1.0, 0.3, 1.0)
        dad = format2d(msh, props; pontointerno=false)
        @test length(dad.BC) == 2 * dad.n
        @test length(dad.BV) == 2 * dad.n
        @test any(==(0), dad.BC)

        # full Dirichlet patch from analytical field
        ana = ana_elasticity_patch(; E=1.0, ν=0.3, εxx=0.01)
        apply_analytical_bc!(dad, ana)

        H_G_full_direct(dad, 12)
        @test size(dad.H, 1) == 2 * dad.nt
        @test size(dad.G, 2) == 2 * dad.n

        u = solve(dad)
        @test length(u) == 2 * dad.n
        err = rel_error(dad)
        @test err < 0.15
        @info "elasticity patch rel_error" err
    end

    @testset "2D line contact vs Hertz" begin
        G, ν = 1.0, 0.3
        N = 256
        L = 3.0
        x = collect(range(-L, L; length=N))
        h = x[2] - x[1]
        hp = ElasticHalfPlane2D(G, ν; h=h)
        @test influence_coeff_2d(0, hp) > 0
        @test abs(influence_coeff_2d(0, hp)) > abs(influence_coeff_2d(5, hp))

        # Forward: discrete Hertz pressure integrates to F and yields parabolic u
        R = 1.0
        F_hz = 0.05
        p_hz, hz = hertz_line_pressure(F_hz, R, hp, x)
        @test abs(sum(p_hz) * h - F_hz) / F_hz < 0.05
        prep = precompute_kernel_2d(N, hp)
        u = fc_forward_2d(p_hz, prep)
        idx = findall(abs.(x) .<= hz.a * 0.8)  # interior of contact
        # relative deflection should correlate with -x²
        urel = u[idx] .- mean(u[idx])
        pref = .-x[idx] .^ 2
        pref .-= mean(pref)
        corr = dot(urel, pref) / (norm(urel) * norm(pref))
        @test corr > 0.98
        @info "Hertz line forward" F_ratio=sum(p_hz)*h/F_hz shape_corr=corr

        # HalfSpaceBEM backends: dense vs FFT matvec agreement
        dad = HalfSpace2D(-2.0, 2.0, 64; E=1.0)
        Kd = build_operator(dad, :dense)
        Kf = build_operator(dad, :fft)
        p = rand(64)
        @test Kd * p ≈ Kf * p rtol=1e-9
        Kh = build_operator(dad, :hmatrix; nmax=16, atol=1e-6)
        yh = zeros(64); mul!(yh, Kh, p)
        @test norm(Kd * p - yh) / norm(Kd * p) < 0.05
        @info "half-space backends" dense_fft_ok=true hmat_rel=norm(Kd*p-yh)/norm(Kd*p)
    end

    @testset "multi-region interface (type 3)" begin
        include(datadir("Laplace", "two_regions.jl"))
        msh = mesh_two_regions(ndiv=6, show=false, nome="test_two_reg")
        prob = load_two_regions(msh, Laplace(1.0))
        @test length(prob.regions) == 2
        pair_interfaces!(prob)
        @test length(prob.interfaces) >= 1
        assemble_multiregion(prob; npg=10)
        solve_multiregion!(prob)
        # T ≈ x
        dadL = prob.regions[1]
        err = norm([dadL.T[i] - dadL.Nodes[i][1] for i in 1:dadL.n]) /
              max(norm([dadL.Nodes[i][1] for i in 1:dadL.n]), 1e-12)
        @test err < 0.15
        @info "two-region Laplace err" err n_if=length(prob.interfaces)
    end

    @testset "Pohrt–Li half-space contact" begin
        G, ν = 1.0, 0.3
        hs = ElasticHalfSpace(G, ν; hx=0.1, hy=0.1)
        # Kzz self-influence positive
        @test influence_coeff(Kzz, 0, 0, hs) > 0
        # Kxx self-influence positive
        @test influence_coeff(Kxx, 0, 0, hs) > 0
        # symmetries
        @test influence_coeff(Kzx, 1, 0, hs) ≈ -influence_coeff(Kxz, 1, 0, hs) atol=1e-12
        @test influence_coeff(Kxy, 1, 2, hs) ≈ influence_coeff(Kyx, 1, 2, hs) atol=1e-12

        # FFT convolution agrees with direct sum on tiny grid
        nx = ny = 8
        hs2 = ElasticHalfSpace(G, ν; hx=0.25, hy=0.25)
        prep = precompute_kernels(nx, ny, hs2; components=(Kzz,))
        p = zeros(nx, ny); p[4, 4] = 1.0
        u_fc = fc_forward(p, Kzz, prep)
        u_dir = zeros(nx, ny)
        for j in 1:ny, i in 1:nx
            s = 0.0
            for jj in 1:ny, ii in 1:nx
                s += influence_coeff(Kzz, i - ii, j - jj, hs2) * p[ii, jj]
            end
            u_dir[i, j] = s
        end
        @test u_fc ≈ u_dir rtol=1e-10

        # CG inverse recovers the point load on its support
        mask = falses(nx, ny); mask[4, 4] = true
        p_rec = fc_inverse(u_dir, mask, Kzz, prep; tol=1e-12)
        @test p_rec[4, 4] ≈ 1.0 rtol=1e-6

        # Hertz-like sphere: force scales correctly order of magnitude
        N = 32
        L = 1.5
        xv = range(-L, L; length=N)
        hx = xv[2] - xv[1]
        hs3 = ElasticHalfSpace(G, ν; hx=hx, hy=hx)
        R = 1.0; δ = 0.02
        gap0 = [(xv[i]^2 + xv[j]^2) / (2R) for i in 1:N, j in 1:N]
        sol = solve_normal_contact(gap0, δ, hs3; tol=1e-6)
        Estar = contact_modulus(hs3)
        F_hz = 4/3 * Estar * sqrt(R) * δ^(3/2)
        @test sol.force > 0
        @test abs(sol.force - F_hz) / F_hz < 0.15   # ≤15 % on 32² grid
        @info "Hertz force error" rel = abs(sol.force - F_hz) / F_hz
    end

    @testset "solve_transient_o2 API" begin
        # Smoke test: builds second-order ODE without error and returns a solution
        msh = quadrado(ndiv=8, show=false, nome="test_quad_o2")
        props = Laplace(1.0)
        dad = format2d(msh, props; pontointerno=true)
        H_G_full_direct(dad, 10)
        DIBEM(dad)
        @test haskey(dad.cache, :M)

        sol = solve_transient_o2(dad, 0.05, 0.2; abstol=1e-4, reltol=1e-4)
        @test sol !== nothing
        @test haskey(dad.cache, :T)
        @test size(dad.T, 2) == length(0:0.05:0.2)
        # position extraction must be finite
        @test all(isfinite, dad.T)
    end


    @testset "Diffuse-advective DIBEM smoke (exp mxy)" begin
        include(joinpath(@__DIR__, "test_diffuse_advective.jl"))
    end

    @testset "DIBEM Hmat/FMM" begin
        include(joinpath(@__DIR__, "test_dibem_fast.jl"))
    end

    @testset "DIBEM elasticity" begin
        include(joinpath(@__DIR__, "test_dibem_elasticity.jl"))
    end

    @testset "HSS-FMM H/G" begin
        include(joinpath(@__DIR__, "test_hss_fmm_hg.jl"))
    end

    @testset "H2 far ACA" begin
        include(joinpath(@__DIR__, "test_h2_aca_far.jl"))
    end

    @testset "H2 recursive node repackage" begin
        include(joinpath(@__DIR__, "test_h2_node.jl"))
    end

    @testset "H-matrix algebra (hlru/hara/h2)" begin
        include(joinpath(@__DIR__, "test_hmat_algebra.jl"))
    end

    @testset "H-matrix factor / multi-RHS / HARA product" begin
        include(joinpath(@__DIR__, "test_hmat_factor.jl"))
    end

    @testset "H-matrix precond / Chol / hara_product" begin
        include(joinpath(@__DIR__, "test_hmat_precond.jl"))
    end

    @testset "Nested H² HARA" begin
        include(joinpath(@__DIR__, "test_hara_h2.jl"))
    end

    @testset "H² from FMM" begin
        include(joinpath(@__DIR__, "test_h2_fmm.jl"))
    end

    @testset "MMM modal smoke" begin
        include(joinpath(@__DIR__, "test_mmm.jl"))
    end

    @testset "Cattaneo–Mindlin" begin
        include(joinpath(@__DIR__, "test_cattaneo_mindlin.jl"))
    end

    @testset "Multibody elasticity NTN/NTS" begin
        include(joinpath(@__DIR__, "test_multibody_contact.jl"))
    end

    @testset "Elasticity local (n,t) frame §4.7" begin
        include(joinpath(@__DIR__, "test_local_frame_elasticity.jl"))
    end

end

println("\nTests finished in ", round((time() - ti) / 60; digits=3), " minutes")
