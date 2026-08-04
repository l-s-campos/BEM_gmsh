# Método Modal Modificado (MMM) — thesis Áquila Santos §4.5
using Test
using LinearAlgebra
using Statistics: mean
using BEM

include(joinpath(@__DIR__, "..", "data", "Laplace", "Laplace_dad.jl"))

function _fixed_square_dad(ndiv; n_int=4, nome="mmm_sq")
    # unit square, all edges Dirichlet u=0 (membrane)
    msh = Base.invokelatest(quadrado; ndiv=ndiv, show=false, nome=nome)
    dad = format2d(msh, Laplace(1.0); tipo=1, pontointerno=false)
    # force all Dirichlet 0
    dad.BC .= 0
    dad.BV .= 0.0
    # internal collocation grid for DIBEM inertia
    xs = range(0.15, 0.85; length=n_int)
    internals = [SVector(x, y) for y in xs for x in xs]
    empty!(dad.internalNodes)
    append!(dad.internalNodes, internals)
    dad.ni = length(dad.internalNodes)
    dad.nt = dad.n + dad.ni
    H_G_full_direct(dad, 12)
    DIBEM(dad)
    return dad
end

@testset "build_modal_system structure" begin
    dad = _fixed_square_dad(6; n_int=3, nome="mmm_struct")
    sys = build_modal_system(dad)
    nf = length(sys.free)
    @test size(sys.M) == (nf, nf)
    @test size(sys.K) == (nf, nf)
    @test length(sys.f0) == nf
    @test nf == dad.ni   # all boundary Dirichlet → free = internals only
    @test all(isfinite, sys.M)
    @test all(isfinite, sys.K)
end

@testset "MMM bi-orthogonality" begin
    dad = _fixed_square_dad(8; n_int=4, nome="mmm_bi")
    sys = build_modal_system(dad)
    basis = modal_analysis_mmm(sys; nmodes=min(8, size(sys.M, 1)))
    nm = size(basis.Φ, 2)
    @test nm >= 1
    Gbi = basis.Φ̃' * basis.Φ
    @test norm(Gbi - I(nm)) / nm < 0.15
    # Φ̃ᵀ D Φ ≈ Λ
    Dproj = basis.Φ̃' * basis.D * basis.Φ
    @test norm(Dproj - Diagonal(basis.ω²)) / (norm(basis.ω²) + 1) < 0.2
    @info "MMM freqs" ω=basis.ω[1:min(4, end)]
end

@testset "MMC vs MMM eigenvalues" begin
    dad = _fixed_square_dad(8; n_int=4, nome="mmm_cmp")
    sys = build_modal_system(dad)
    b1 = modal_analysis_mmm(sys; nmodes=5)
    b2 = modal_analysis_mmc(sys; nmodes=5)
    @test length(b1.ω) == length(b2.ω)
    # same spectrum (order may match after sort)
    @test norm(b1.ω² .- b2.ω²) / (norm(b1.ω²) + 1) < 1e-6
end

@testset "membrane fundamental frequency order" begin
    # analytical ω11 = π√2 ≈ 4.442 for unit square, c=1
    dad = _fixed_square_dad(10; n_int=5, nome="mmm_freq")
    sys = build_modal_system(dad)
    basis = modal_analysis_mmm(sys; nmodes=3)
    ω_ana = π * sqrt(2)
    @info "ω1 num vs ana" num=basis.ω[1] ana=ω_ana rel=abs(basis.ω[1]-ω_ana)/ω_ana
    # DIBEM on coarse internals is approximate — allow generous band
    @test basis.ω[1] > 1.0
    @test basis.ω[1] < 15.0
end

@testset "MMM transient runs + amplitude selection" begin
    dad = _fixed_square_dad(6; n_int=3, nome="mmm_tr")
    # non-zero IC on internals (pluck)
    u0 = zeros(dad.nt)
    for (k, p) in enumerate(dad.internalNodes)
        u0[dad.n + k] = sin(π * p[1]) * sin(π * p[2])
    end
    U, t, basis = solve_mmm!(dad, 0.02, 0.4; nmodes=6, u0=u0, select=:freq)
    @test size(U, 2) == length(t)
    @test all(isfinite, U)
    @test maximum(abs, U) > 1e-8

    A = mode_amplitudes(basis)
    @test all(isfinite, A)
    idx = select_modes_amplitude(basis; nkeep=3)
    @test length(idx) == 3

    U2, t2, b2 = solve_mmm!(dad, 0.02, 0.2; nmodes=6, select=:amplitude, nkeep=3, u0=u0)
    @test all(isfinite, U2)
    @test size(b2.Φ, 2) == 3
end

@testset "MMC alias" begin
    dad = _fixed_square_dad(6; n_int=3, nome="mmc_tr")
    u0 = zeros(dad.nt)
    for (k, p) in enumerate(dad.internalNodes)
        u0[dad.n + k] = sin(π * p[1]) * sin(π * p[2])
    end
    U, t, b = solve_mmc!(dad, 0.025, 0.25; nmodes=4, u0=u0)
    @test b.method === :mmc
    @test all(isfinite, U)
end
