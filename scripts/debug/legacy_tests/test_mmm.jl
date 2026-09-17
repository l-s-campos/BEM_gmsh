# Método Modal Modificado (MMM) — thesis Áquila Santos §4.5
using Test
using LinearAlgebra
using Statistics: mean, median
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
    set_internal_nodes!(dad, internals)
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
    D = basis.M \ basis.K

    Dproj = basis.Φ̃' * D * basis.Φ
    @test norm(Dproj - Diagonal(basis.ω²)) / (norm(basis.ω²) + 1) < 0.2
    @info "MMM freqs" ω=basis.ω[1:min(4, end)]
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
    U, t, basis = solve_mmm!(dad, 0.02, 0.4; nmodes=6, u0=u0, select=:freq, alg=:houbolt)

    @test size(U, 2) == length(t)
    @test all(isfinite, U)
    @test maximum(abs, U) > 1e-8

    A = mode_amplitudes(basis)
    @test all(isfinite, A)
    idx = select_modes_amplitude(basis; nkeep=3)
    @test length(idx) == 3

    U2, t2, b2 = solve_mmm!(dad, 0.02, 0.2; nmodes=6, select=:amplitude, nkeep=3, u0=u0, alg=:houbolt)

    @test all(isfinite, U2)
    @test size(b2.Φ, 2) == 3
end

@testset "Houbolt SDOF alias" begin
    dad = _fixed_square_dad(6; n_int=3, nome="mmm_hb")
    u0 = zeros(dad.nt)
    for (k, p) in enumerate(dad.internalNodes)
        u0[dad.n + k] = sin(π * p[1]) * sin(π * p[2])
    end
    U, t, b = solve_mmm!(dad, 0.025, 0.25; nmodes=4, u0=u0, alg=:houbolt)
    @test all(isfinite, U)
    @test size(b.Φ, 2) == 4
end

@testset "time-varying Neumann load" begin
    dad = _fixed_square_dad(6; n_int=3, nome="mmm_tv")
    for (i, p) in enumerate(dad.Nodes)
        if p[1] > 0.99
            dad.BC[i] = 1
            dad.BV[i] = 1.0
        end
    end
    H_G_full_direct(dad, 12)
    DIBEM(dad)
    U, t, b = solve_mmm!(dad, 0.05, 0.5; nmodes=4, alg=:houbolt, load=sin)
    @test all(isfinite, U)
    @test size(b.Φ, 2) == 4
end

@testset "forced membrane ωmax keeps field bounded" begin
    include(joinpath(@__DIR__, "..", "data", "Laplace", "potencial_problems.jl"))
    include(joinpath(@__DIR__, "..", "data", "Laplace", "wave_propagation.jl"))
    dad, _ = wave_problem(:membrane_forced; ndiv=10, n_int=8)
    H_G_full_direct(dad; npg=8, threaded=false)
    DIBEM(dad; method=:dense, rbf=PHS(1; poly_deg=-1))
    U, _, b = solve_mmm!(dad, 0.04, 2.0; alg=:houbolt)
    @test all(isfinite, U)
    @test maximum(abs, U) ≤ 5.0

    @test length(b.ω) ≥ 4
    A = mode_amplitudes(b)
    A0 = mode_amplitudes(b; ω=0)
    @test A ≈ A0
    @test maximum(A) ≤ 1e3 * maximum(A[1:min(8, end)]) + 1e-12


end





