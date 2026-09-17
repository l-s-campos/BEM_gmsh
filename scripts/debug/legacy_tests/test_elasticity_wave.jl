# Plane-stress sudden bar: Houbolt + MMM stay bounded and track the 1D series.
using Test
using DrWatson
@quickactivate :BEM
using LinearAlgebra

include(datadir("Laplace", "Laplace_dad.jl"))
include(datadir("Laplace", "potencial_problems.jl"))
include(datadir("Laplace", "wave_propagation.jl"))
include(datadir("elastico", "iso", "bar_sudden.jl"))

@testset "analytical u,ü residual DIBEM vs cells" begin
    dad, meta = elasticity_bar_sudden(; ndiv=8, n_int=4, ν=0.0)
    H_G_full_direct(dad; npg=8, threaded=false)
    H, G = Matrix(dad.H), Matrix(dad.G)
    Md = DIBEM(deepcopy(dad); method=:dense, rbf=PHS(3; poly_deg=1))
    Mc = build_cell_mass(deepcopy(dad); npg=8).M
    t = 0.5
    u = zeros(2 * dad.nt)
    tv = zeros(2 * dad.n)
    ddu = zeros(2 * dad.nt)
    pts = all_points(dad)
    @inbounds for i in 1:dad.nt
        f = bar_sudden_fields(pts[i], t; N=400, c=meta.c, L=meta.L)
        u[2i-1] = f.u
        ddu[2i-1] = f.ddu
    end
    @inbounds for i in 1:dad.n
        f = bar_sudden_fields(dad.Nodes[i], t; N=400, c=meta.c, L=meta.L)
        tv[2i-1] = meta.E * f.dudx * dad.Normal[i][1]
    end
    rd = H * u + Md * ddu - G * tv
    rc = H * u + Mc * ddu - G * tv
    den = norm(H * u) + norm(G * tv) + norm(Mc * ddu) + 1e-30
    @test all(isfinite, rd) && all(isfinite, rc)
    @info "ana residual t=0.5" dibem=norm(rd)/den cells=norm(rc)/den
    @test norm(rc) < 0.5 * norm(rd)   # cells closer once inertia is active
end

@testset "elasticity bar sudden Houbolt and MMM" begin
    dad, meta = elasticity_bar_sudden(; ndiv=8, n_int=4, ν=0.0)
    H_G_full_direct(dad; npg=8, threaded=false)
    DIBEM(dad; method=:dense, rbf=PHS(1; poly_deg=-1))
    @test size(dad.H, 1) == 2 * dad.nt
    @test size(dad.M, 1) == 2 * dad.nt

    sys = build_modal_system(dad)
    @test length(sys.free) == 2 * dad.nt - count(==(0), dad.BC)
    b = modal_analysis_mmm(sys; nmodes=8)
    @test b.ω[1] > 0.5
    @test b.ω[1] < 4.0

    dadH = deepcopy(dad)
    Uh = solve_Houbolt(dadH, 0.05, 1.0)
    @test all(isfinite, Uh)
    @test size(Uh, 1) == 2 * dad.nt

    dadN = deepcopy(dad)
    Un = solve_Newmark(dadN, 0.05, 1.0)
    @test all(isfinite, Un)
    @test size(Un) == size(Uh)

    dadM = deepcopy(dad)
    Um, t, bm = solve_mmm!(dadM, 0.05, 1.0; alg=:houbolt)
    @test all(isfinite, Um)
    @test maximum(abs, Um) < 5
    @test length(bm.ω) ≥ 4
    @test bm.ω[1] > 0.2
end

@testset "elasticity bar sudden DRM MMM" begin
    dad, meta = elasticity_bar_sudden(; ndiv=8, n_int=4, ν=0.0)
    H_G_full_direct(dad; npg=8, threaded=false)
    drm = build_drm_matrices(dad; npg=8)
    @test size(drm.M) == (2 * dad.nt, 2 * dad.nt)
    @test tr(drm.M) > 0

    sys = build_modal_system(dad)
    b = modal_analysis_mmm(sys; nmodes=8)
    @info "DRM bar sudden" ω1=b.ω[1] nmodes=length(b.ω)
    @test b.ω[1] > 0.5
    @test b.ω[1] < 4.0

    Um, t, bm = solve_mmm!(dad, 0.05, 1.0; alg=:houbolt)
    @test all(isfinite, Um)
    @test maximum(abs, Um) < 5
    @test bm.ω[1] > 0.2
end

@testset "elasticity bar sudden quadratic 12/side ni=9" begin
    # 12 quadratic elements/side: transfinite 13 nodes + order 2; 3×3 internals.
    dad, meta = elasticity_bar_sudden(; ndiv=13, n_int=3, tipo=2, ordem=2,
        ν=0.0, pad=0.15)
    @test length(dad.elements) == 48
    @test dad.n == 144
    @test dad.ni == 9
    H_G_full_direct(dad; npg=12, threaded=false)
    build_drm_matrices(dad; npg=12, kernel=:r)
    @test size(dad.M, 1) == 2 * dad.nt
    @test count(<( -1e-8), real.(eigvals(Matrix(dad.M)))) == 0
    Um, t, bm = solve_mmm!(dad, 0.05, 1.0; alg=:houbolt)
    @test all(isfinite, Um)
    @test bm.ω[1] > 0.5
    @test bm.ω[1] < 4.0
    pts = vcat(dad.Nodes, dad.internalNodes)
    ip = argmin(norm(p - meta.probe) for p in pts)
    ux = Um[2 * (ip - 1) + 1, :]
    ua = [meta.ana.u(meta.probe; t=ti) for ti in t]
    @test maximum(abs, ux) < 3
    @test norm(ux - ua) / (norm(ua) + eps()) < 0.25
end

