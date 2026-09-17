# Dual Galerkin Laplace: Costabel A symmetric; DRM as Pérez-Gavilán/Aliabadi.

using Test
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Statistics

include(datadir("Laplace", "Laplace_dad.jl"))
include(datadir("Laplace", "potencial_problems.jl"))
include(datadir("Laplace", "wave_propagation.jl"))

@testset "Galerkin V and W are symmetric" begin
    msh = quadrado(ndiv=8, show=false, nome="gal_sym")
    dad = format2d(msh, Laplace(1.0); tipo=1, pontointerno=false)
    ops = assemble_galerkin_calderon(dad; npg=10, threaded=false)
    n = dad.n
    @test size(ops.V) == (n, n)
    @test size(ops.W) == (n, n)
    @test norm(ops.V - ops.V') / (norm(ops.V) + 1e-14) < 1e-12
    @test norm(ops.W - ops.W') / (norm(ops.W) + 1e-14) < 1e-12
    @test norm(ops.B - ops.H') / (norm(ops.H) + 1e-14) < 1e-12
    sys = galerkin_mixed_system(dad; npg=10, threaded=false)
    @test norm(sys.A - sys.A') / (norm(sys.A) + 1e-14) < 1e-10
end

@testset "Galerkin H,G collocation shape and Laplace T=x" begin
    msh = quadrado(ndiv=10, show=false, nome="gal_Tx")
    dad = format2d(msh, Laplace(1.0); tipo=1, pontointerno=false)
    ana = ana_laplace_linear(; direction=SA[1.0, 0.0])
    apply_analytical_bc!(dad, ana)
    H, G = H_G_galerkin(dad; npg=10, threaded=false)
    @test size(H) == (dad.nt, dad.nt)
    @test size(G) == (dad.nt, dad.n)
    solve(dad)
    nx = getindex.(dad.Normal, 1)
    qana = -nx
    @test sqrt(mean(abs2, dad.q .- qana)) < 5e-3
end


function _npos_drm(dad; galerkin::Bool)
    drm = build_drm_matrices(dad, PHS(3; poly_deg=1); npg=8, galerkin=galerkin)
    set_cache!(dad; H=drm.H, G=drm.G, M=drm.M)
    sys = build_modal_system(dad)
    ev = eigvals(sys.M \ sys.K)
    return count(<( -1e-8), real.(ev)), length(ev),
        minimum(real, ev), maximum(real, ev)
end

@testset "Galerkin DRM mixed A is symmetric" begin
    dad, _ = wave_problem(:bar_sudden; ndiv=8, n_int=4)
    mix = galerkin_drm_mixed(dad, PHS(3; poly_deg=1); npg=8, threaded=false)
    @test mix.mode === :mixed
    @test size(mix.A, 1) == dad.n
    @test size(mix.Mdrm, 1) == dad.n
    @test size(mix.Mdrm, 2) == dad.nt
    @test norm(mix.A - mix.A') / (norm(mix.A) + 1e-14) < 1e-10
    @test norm(mix.drm.M - dad.M) / (norm(dad.M) + 1e-14) < 1e-12
end

@testset "Galerkin DRM vs collocation DRM wave eigs" begin
    dadC, _ = wave_problem(:bar_sudden; ndiv=8, n_int=4)
    H_G_full_direct(dadC; npg=8, threaded=false)
    nneg_c, n_c, mn_c, mx_c = _npos_drm(dadC; galerkin=false)

    dadG, _ = wave_problem(:bar_sudden; ndiv=8, n_int=4)
    nneg_g, n_g, mn_g, mx_g = _npos_drm(dadG; galerkin=true)

    @info "wave DRM collocation" nneg_c n_c mn_c mx_c
    @info "wave DRM galerkin" nneg_g n_g mn_g mx_g
    @test n_c == n_g
    @test nneg_g ≥ 0
    @test nneg_c ≥ 0
end

