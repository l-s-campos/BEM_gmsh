# DIBEM mass: remainder identity M 1 = ID.
using Test
using LinearAlgebra
using StaticArrays
using BEM

const HAVE_CUDA = try
    using CUDA
    CUDA.functional()
catch
    false
end

@testset "DIBEM Laplace M*1 = ID" begin
    dad = format2d(quadrado(ndiv=6, show=false, nome="t_dibem_id"), Laplace(1.0);
        tipo=1, pontointerno=true)
    M = DIBEM(dad; rbf=PHS(3; poly_deg=1), npg=12)
    onesv = ones(dad.nt)
    @test norm(M * onesv - dad.dibem_ID) / (norm(dad.dibem_ID) + 1e-14) < 1e-10
end

@testset "DIBEM dense mass finite" begin
    dad = format2d(quadrado(ndiv=6, show=false, nome="t_dibem"), Laplace(1.0);
        pontointerno=true)
    assemble!(dad, 10)
    DIBEM(dad)
    @test has_cache(dad, :M)
    @test all(isfinite, dad.M)
    @test size(dad.M) == (dad.nt, dad.nt)
end

# Barcelos–Loeffler DIBEM for ∇·(K ∇u)=0
@testset "heterogeneous K=1 matches homogeneous BEM" begin
    dad = format2d(quadrado(ndiv=8, show=false, nome="t_het_k1"), Laplace(1.0);
        pontointerno=true, tipo=1)
    assemble!(dad; npg=10, threaded=false)
    solve(dad)
    T0 = copy(dad.T)
    q0 = copy(dad.q)
    dad2 = format2d(quadrado(ndiv=8, show=false, nome="t_het_k1b"), Laplace(1.0);
        pontointerno=true, tipo=1)
    assemble!(dad2; npg=10, threaded=false)
    solve_heterogeneous!(dad2, p -> 1.0; rbf=PHS(1; poly_deg=-1))
    @test all(isfinite, dad2.T)
    @test median(abs.(dad2.T[1:dad2.n] .- T0[1:dad.n])) < 0.08
    @test median(abs.(dad2.q .- q0)) < 0.15
end

@testset "heterogeneous A from cached dM" begin
    dad = format2d(quadrado(ndiv=6, show=false, nome="t_het_dM"), Laplace(1.0);
        pontointerno=true, tipo=1)
    assemble!(dad; npg=8, threaded=false)
    rbf = PHS(1; poly_deg=-1)
    A1, _ = BEM.heterogeneous_K_operator(dad, p -> 1 + p[2]; rbf=rbf)
    @test has_cache(dad, :het_dM)
    @test has_cache(dad, :het_Ggrad)
    key = dad.het_geom_key
    A2, _ = BEM.heterogeneous_K_operator(dad, p -> 1 + p[2]; rbf=PHS(1; poly_deg=-1))
    @test dad.het_geom_key == key
    @test A1 ≈ A2
    @test maximum(abs.(A1 * ones(dad.nt))) < 1e-10
    Aconst, _ = BEM.heterogeneous_K_operator(dad, p -> 1.0; rbf=rbf)
    @test dad.het_geom_key == key
    # PHS(1) with poly_deg=-1 does not reproduce ∇(const)=0 exactly
    @test maximum(abs.(Aconst)) < 0.05
    @test maximum(abs.(Aconst)) < 0.2 * maximum(abs.(A1))
end

@testset "heterogeneous K=1+y, u=x" begin
    dad = format2d(quadrado(ndiv=10, show=false, nome="t_het_ky"), Laplace(1.0);
        pontointerno=true, tipo=1)
    # left T=0, right T=1, top/bottom q=0  (u=x, q_phys=0 on horizontal)
    @inbounds for i in 1:dad.n
        p = dad.Nodes[i]
        if p[1] < 1e-8
            dad.BC[i] = 0
            dad.BV[i] = 0.0
        elseif p[1] > 1 - 1e-8
            dad.BC[i] = 0
            dad.BV[i] = 1.0
        else
            dad.BC[i] = 1
            dad.BV[i] = 0.0
        end
    end
    assemble!(dad; npg=12, threaded=false)
    solve_heterogeneous!(dad, p -> 1 + p[2]; rbf=PHS(1; poly_deg=-1))
    ux = [p[1] for p in all_points(dad)]
    @test median(abs.(dad.T .- ux)) < 0.08
    @test maximum(abs.(dad.T[1:dad.n] .- ux[1:dad.n])) < 0.2
end

@testset "heterogeneous K=1, source=4 matches Poisson DIBEM" begin
    msh = quadrado(ndiv=8, show=false, nome="t_het_src")
    dad = format2d(msh, Laplace(1.0); pontointerno=true, tipo=1)
    dadp = format2d(msh, Laplace(1.0); pontointerno=true, tipo=1)
    uex(p) = p[1]^2 + p[2]^2
    for d in (dad, dadp)
        @inbounds for i in 1:d.n
            d.BC[i] = 0
            d.BV[i] = uex(d.Nodes[i])
        end
        assemble!(d; npg=10, threaded=false)
    end
    solve_poisson_dibem!(dadp, 4.0; rbf=PHS(3; poly_deg=1), npg=10)
    solve_heterogeneous!(dad, p -> 1.0; source=4.0, rbf=PHS(1; poly_deg=-1),
        source_rbf=PHS(3; poly_deg=1))
    @test median(abs.(dad.T .- dadp.T)) < 0.12
end

@testset "Reynolds heterogeneous DIBEM vs 1-D wedge" begin
    film = film_linear(; a=2.0, hi=2.0, L=1.0)
    msh = quadrado(ndiv=10, Lx=1.0, Ly=0.25, show=false, nome="t_rey_het")
    dad = format2d(msh, Laplace(1.0); pontointerno=true, tipo=1)
    @inbounds for i in 1:dad.n
        x, y = dad.Nodes[i]
        if x < 1e-8 || x > 1 - 1e-8
            dad.BC[i] = 0
            dad.BV[i] = 0.0
        else
            dad.BC[i] = 1
            dad.BV[i] = 0.0
        end
    end
    assemble!(dad; npg=10, threaded=false)
    solve_reynolds_het!(dad, film; μ=1.0, U=1.0)
    pint = dad.T[(dad.n + 1):end]
    @test !isempty(pint)
    @test maximum(pint) > 0.01
    p1d = maximum(infinite_bearing_pressure(film, x) for x in 0.1:0.1:0.9)
    @test maximum(pint) < p1d * 1.3
end

@testset "periodic x: T=y on the unit square" begin
    msh = quadrado(ndiv=8, show=false, nome="t_per_y")
    dad = format2d(msh, Laplace(1.0); pontointerno=true, tipo=1)
    @inbounds for i in 1:dad.n
        y = dad.Nodes[i][2]
        x = dad.Nodes[i][1]
        if abs(x) < 1e-8 || abs(x - 1) < 1e-8
            dad.BC[i] = 1
            dad.BV[i] = 0.0
        elseif y < 1e-8
            dad.BC[i] = 0
            dad.BV[i] = 0.0
        elseif y > 1 - 1e-8
            dad.BC[i] = 0
            dad.BV[i] = 1.0
        else
            dad.BC[i] = 1
            dad.BV[i] = 0.0
        end
    end
    pairs = mark_periodic_x!(dad; x0=0.0, x1=1.0, tol=1e-6)
    @test !isempty(pairs)
    # corners belong to y-Dirichlet; keep them
    @inbounds for i in 1:dad.n
        y = dad.Nodes[i][2]
        if y < 1e-8
            dad.BC[i] = 0; dad.BV[i] = 0.0
        elseif y > 1 - 1e-8
            dad.BC[i] = 0; dad.BV[i] = 1.0
        end
    end
    dad.periodic_pairs = [(i, j) for (i, j) in pairs if dad.BC[i] == 4 && dad.BC[j] == 4]
    assemble!(dad; npg=10, threaded=false)
    L, Kv, _ = BEM.heterogeneous_L(dad, p -> 1.0; rbf=PHS(1; poly_deg=-1))
    Asys, b = BEM._het_mixed_from_L(dad, L, Kv)
    xsol = bem_linsolve(Asys, b)
    T = BEM._split_het_sol!(dad, xsol, Kv)
    uy = [p[2] for p in all_points(dad)]
    @test median(abs.(T .- uy)) < 0.08
end

@testset "CFP characteristic θ is 1 in full film" begin
    film = film_parabolic(; L=1.0, hmax=2.0, hmin=1.0)
    xs = collect(range(0.0, 1.0; length=11))
    pts = [Point2D(x, 0.0) for x in xs]
    p = [x <= 0.6 + 1e-12 ? 1.0 : -0.1 for x in xs]
    θ = characteristic_theta(pts, p, film; pc=0.0)
    @test all(θ[i] == 1.0 for i in eachindex(p) if p[i] > 0)
    @test θ[end] < 1.0
    @test θ[end] ≈ film.h(0.5) / film.h(1.0) atol=0.05
end

@testset "misaligned journal film h(x,y)" begin
    film = film_journal_misaligned(; L=1.0, W=0.25, c=1.0, ε=0.8)
    @test film_h(film, 0.0, 0.0) ≈ 1.8
    @test film_h(film, 0.0, 0.125) ≈ 1.0
    @test film_h(film, 0.0, 0.25) ≈ 0.2
    @test film_h(film, 0.5, 0.0) ≈ 0.2
    pts = [Point2D(x, 0.0) for x in range(0.0, 1.0; length=5)]
    p = [1.0, 1.0, -0.1, -0.1, -0.1]
    θ = characteristic_theta(pts, p, film; pc=0.0)
    @test θ[1] == 1.0
    @test θ[end] < 1.0
end

@testset "semi-system EHL smoke (tiny grid)" begin
    using BEM.Contact
    E = 200e9; ν = 0.3; R = 0.01905; F = 8.0
    G = G_from_E(E, ν)
    Estar = contact_modulus(combined_halfspace(G, ν, G, ν))
    hz = hertz_sphere_load(R, F, Estar)
    n_int = 7
    msh = mesh_ehl_square(; L=4.0, nside=n_int + 1, nome="t_ehl_smoke")
    dad = format2d(msh, Laplace(1.0); pontointerno=false, tipo=1)
    internal_grid!(dad, n_int, n_int; d_min=0.0, layout=:cell)
    gmap = interior_rect_map(dad)
    @test gmap.nix == n_int && gmap.niy == n_int
    a = hz.a; pH = hz.p0
    hs = combined_halfspace(G, ν, G, ν; hx=gmap.hx * a, hy=gmap.hy * a)
    prep = precompute_kernels(gmap.nix, gmap.niy, hs; components=(Kzz,))
    h0 = [(point(dad, i)[1]^2 + point(dad, i)[2]^2) / 2 for i in 1:dad.nt]
    p0 = zeros(dad.nt)
    @inbounds for i in 1:dad.nt
        pt = point(dad, i)
        r2 = pt[1]^2 + pt[2]^2
        r2 < 1 && (p0[i] = 0.3 * sqrt(1 - r2))
    end
    λ = 12 * 0.096 * 4 * R^2 / (pH * a^3)
    sol = solve_semi_system!(dad, gmap, hs, prep; h0=h0, W=F / (pH * a^2),
        ū=λ / 12, η0=1.0, R=R, δ0=0.4, p0=p0, μfun=(_ -> 1.0),
        p_fft_scale=pH, u_scale=R / a^2, maxiter=3, verbose=false)
    @test sol.pmax > 0
    @test sol.hmin >= 0
    @test isfinite(sol.load)
end

@testset "cube 3D Poisson DIBEM u=|x|²" begin
    include(joinpath(@__DIR__, "..", "data", "Laplace", "cube_mesh.jl"))
    msh = mesh_unit_cube(; L=1.0, ndiv=2, nome="t_poiss3d")
    dad = format3d(msh, Laplace(1.0); pontointerno=false)
    set_internal_nodes!(dad, vec(cube_interior_grid(1.0, 2)))
    ana = ana_poisson_r2(; k=1.0, dim=3)
    apply_analytical_bc!(dad, ana)
    assemble!(dad; npg=8, threaded=false)
    solve_poisson_dibem!(dad, 6.0; rbf=PHS(3; poly_deg=1), npg=8)
    @test rel_error(dad) < 0.08
    @test rel_error_flux(dad) < 0.20
end

@testset "cube 3D DIBEM hmatrix vs dense" begin
    include(joinpath(@__DIR__, "..", "data", "Laplace", "cube_mesh.jl"))
    msh = mesh_unit_cube(; L=1.0, ndiv=2, nome="t_dibem3h")
    dad = format3d(msh, Laplace(1.0); pontointerno=false)
    set_internal_nodes!(dad, vec(cube_interior_grid(1.0, 2)))
    assemble!(dad; npg=8, threaded=false)
    rbf = PHS(3; poly_deg=1)
    Md = DIBEM(deepcopy(dad); method=:dense, rbf=rbf)
    Mh = DIBEM(dad; method=:hmatrix, rbf=rbf, nmax=16, atol=1e-5, rtol=1e-5, threads=false)
    @test Mh isa DibemFactoredOperator
    x = randn(dad.nt)
    @test norm(Mh * x - Md * x) / (norm(Md * x) + 1e-14) < 0.05
    apply_analytical_bc!(dad, ana_poisson_r2(; dim=3))
    dadh = deepcopy(dad)
    solve_poisson_dibem!(dadh, 6.0; rbf=rbf, method=:hmatrix, npg=8)
    @test rel_error(dadh) < 0.10
end

@testset "DIBEM KA CPU vs dense" begin
    rbf = PHS(3; poly_deg=1)
    dad = format2d(quadrado(ndiv=6, show=false, nome="t_dibem_gpu_cpu"), Laplace(1.0);
        pontointerno=true)
    dad_ref = format2d(quadrado(ndiv=6, show=false, nome="t_dibem_gpu_ref"), Laplace(1.0);
        pontointerno=true)
    Md = DIBEM(dad_ref; rbf=rbf, npg=12)
    Mg = DIBEM_gpu(dad; rbf=rbf, T=Float64, npg=12, device=:cpu, threaded=false)
    @test norm(Mg - Md) / norm(Md) < 1e-8
    onesv = ones(dad.nt)
    @test norm(Mg * onesv - dad.dibem_ID) / (norm(dad.dibem_ID) + 1e-14) < 1e-10
end

@testset "DIBEM CUDA vs dense" begin
    if !HAVE_CUDA
        @info "CUDA not functional; skipping GPU DIBEM test"
    else
        rbf = PHS(3; poly_deg=1)
        dad = format2d(quadrado(ndiv=6, show=false, nome="t_dibem_gpu"), Laplace(1.0);
            pontointerno=true)
        dad_ref = format2d(quadrado(ndiv=6, show=false, nome="t_dibem_gpu_d"), Laplace(1.0);
            pontointerno=true)
        Md = DIBEM(dad_ref; rbf=rbf, npg=12)
        Mg = DIBEM(dad; method=:gpu, rbf=rbf, T=Float64, npg=12, threaded=false)
        @test norm(Mg - Md) / norm(Md) < 1e-8
        solve_poisson_dibem!(dad, 4.0; rbf=rbf, method=:gpu, npg=12)
        @test all(isfinite, dad.T)
    end
end
