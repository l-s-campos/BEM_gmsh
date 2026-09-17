# Time-domain Burton–Miller correlato (Laplace FS + DRM / DIBEM / cells)
using Test
using DrWatson
@quickactivate :BEM
using LinearAlgebra

include(datadir("Laplace", "Laplace_dad.jl"))
include(datadir("Laplace", "potencial_problems.jl"))
include(datadir("Laplace", "wave_propagation.jl"))

function _relres(r, parts...)
    den = sum(norm, parts) + 1e-30
    return norm(r) / den
end

@testset "Laplace hyper closed form" begin
    r = Point2D(0.3, 0.4)
    n = Point2D(1.0, 0.0)
    nf = Point2D(0.0, 1.0)
    lap = Laplace(1.0)
    R = norm(r)
    e = r / R
    kh = fundamental_hyper(lap, r, n, nf)
    @test kh.U ≈ dot(e, nf) / (2π * R) atol=1e-14
    @test kh.T ≈ -(dot(nf, n) - 2 * dot(e, n) * dot(e, nf)) / (2π * R^2) atol=1e-14
    ε = 1e-6
    Gp = fundamental(lap, r - ε * nf, n).U
    Gm = fundamental(lap, r + ε * nf, n).U
    @test kh.U ≈ (Gp - Gm) / (2ε) rtol=1e-4
end

@testset "α=0 copies CBIE" begin
    msh = quadrado(ndiv=4, show=false, nome="bm_a0")
    dad = format2d(msh, Laplace(1.0); pontointerno=true, tipo=1)
    H_G_full_direct(dad; npg=8, threaded=false)
    H0, G0 = copy(dad.H), copy(dad.G)
    DIBEM(dad; method=:dense, rbf=PHS(3; poly_deg=1))
    M0 = copy(dad.M)
    Hp, Gp = H_G_hyper(dad; npg=8, threaded=false)
    Mp = dibem_hyper_mass(dad; rbf=PHS(3; poly_deg=1))
    Hc, Gc, Mc = combine_burton_miller(H0, G0, M0, Hp, Gp, Mp, 0.0; n=dad.n)
    @test Hc ≈ H0
    @test Gc ≈ G0
    @test Mc ≈ M0
    @test size(Hp) == (dad.n, dad.nt)
    @test size(Gp) == (dad.n, dad.n)
    @test all(isfinite, Hp) && all(isfinite, Gp)
end

@testset "harmonic HBIE residual (ü=0)" begin
    msh = quadrado(ndiv=6, show=false, nome="bm_harm")
    dad = format2d(msh, Laplace(1.0); pontointerno=true, tipo=1)
    ana = ana_laplace_linear(; direction=SA[1.0, 0.0], k=1.0)
    attach_analytical!(dad, ana)
    H_G_full_direct(dad; npg=12, threaded=false)
    Hp, Gp = H_G_hyper(dad; npg=12, threaded=false)
    u = [float(ana.u(p)) for p in all_points(dad)]
    q = [float(ana.q(dad.Nodes[i], dad.Normal[i])) for i in 1:dad.n]
    r = dad.H * u - dad.G * q
    rp = Hp * u - Gp * q
    @test _relres(r, dad.H * u, dad.G * q) < 0.05
    @test all(isfinite, rp)
    @test _relres(rp, Hp * u, Gp * q) < 0.05
end

@testset "wave residual DRM/DIBEM/cells" begin
    dad0, _ = wave_problem(:bar_sudden; ndiv=6, n_int=3)
    t = 0.5
    for mass in (:drm, :dibem, :cells)
        dad = deepcopy(dad0)
        assemble_wave_burton_miller!(dad; mass=mass, α=0.0, npg=8,
            rbf=PHS(3; poly_deg=1), threaded=false)
        u = zeros(dad.nt)
        q = zeros(dad.n)
        ddu = zeros(dad.nt)
        pts = all_points(dad)
        @inbounds for i in eachindex(pts)
            f = bar_sudden_fields(pts[i], t; N=200, c=1.0, L=1.0)
            u[i] = f.u
            ddu[i] = f.ddu
        end
        @inbounds for i in 1:dad.n
            f = bar_sudden_fields(dad.Nodes[i], t; N=200, c=1.0, L=1.0)
            q[i] = -f.dudx * dad.Normal[i][1]
        end
        H, G, M = dad.H_cbie, dad.G_cbie, dad.M_cbie
        Hp, Gp, Mp = dad.H_hyper, dad.G_hyper, dad.M_hyper
        r = H * u - G * q - M * ddu
        rp = Hp * u - Gp * q - view(Mp, 1:dad.n, :) * ddu
        @test all(isfinite, r) && all(isfinite, rp)
        dad1 = deepcopy(dad0)
        assemble_wave_burton_miller!(dad1; mass=mass, α=1.0, npg=8,
            rbf=PHS(3; poly_deg=1), threaded=false)
        rc = dad1.H * u - dad1.G * q - dad1.M * ddu
        @test all(isfinite, rc)
    end
end

@testset "Houbolt α=0 matches CBIE DIBEM" begin
    dad, _ = wave_problem(:bar_sudden; ndiv=6, n_int=3)
    dadA = deepcopy(dad)
    dadB = deepcopy(dad)
    H_G_full_direct(dadA; npg=8, threaded=false)
    DIBEM(dadA; rbf=PHS(3; poly_deg=1))
    assemble_wave_burton_miller!(dadB; mass=:dibem, α=0.0, npg=8,
        rbf=PHS(3; poly_deg=1), threaded=false)
    @test dadB.H ≈ dadA.H rtol=1e-12
    @test dadB.G ≈ dadA.G rtol=1e-12
    @test dadB.M ≈ dadA.M rtol=1e-12
    @test dadB.H ≈ dadB.H_cbie rtol=1e-15
    Δt, tf = 0.05, 0.3
    solve_Houbolt(dadA, Δt, tf)
    solve_Houbolt(dadB, Δt, tf)
    @test dadA.T ≈ dadB.T rtol=1e-10
    @test all(isfinite, dadB.T)
    @test maximum(abs, dadB.T[:, 2]) > 1e-12
end
