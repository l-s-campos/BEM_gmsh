# Diffuse–advective DIBEM — variable velocity (Pinheiro Ch.8 §8.2.1)
using Test
using LinearAlgebra
using StaticArrays
using BEM

include(joinpath(@__DIR__, "..", "data", "Laplace", "Laplace_dad.jl"))

@testset "Diffuse-advective operators assemble" begin
    msh = Base.invokelatest(quadrado; ndiv=6, show=false, nome="da_op")
    dad = setup_da_c8e1(msh; m=1.0, n_int=3)
    H_G_full_direct(dad, 10)
    S = build_da_S_matrix(dad)
    M′ = build_da_Mprime(dad, da_c8e1_velocity(1.0))
    @test size(S) == (dad.nt, dad.nt)
    @test size(M′) == (dad.nt, dad.nt)
    @test all(isfinite, S) && all(isfinite, M′)
    M_DA = dibem_diffuse_advective!(dad, da_c8e1_velocity(1.0))
    @test size(M_DA) == (dad.nt, dad.nt)
    @test all(isfinite, M_DA)
end

@testset "C8E1 PDE residual α∇²u = v·∇u" begin
    m = 2.0
    u = da_c8e1_analytic(m)
    p = SVector(0.37, 0.61)
    uu = u(p)
    gx = m * p[2] * uu
    gy = m * p[1] * uu
    lap = m^2 * (p[1]^2 + p[2]^2) * uu
    adv = (m * p[2]) * gx + (m * p[1]) * gy
    @test lap ≈ adv rtol = 1e-12
end

@testset "C8E1 diffuse-advective m=1 flux error" begin
    msh = Base.invokelatest(quadrado; ndiv=12, show=false, nome="da_c8e1_m1")
    dad = setup_da_c8e1(msh; m=1.0, n_int=5)
    res = test_da_c8e1(dad; m=1.0, npg=12)
    @info "C8E1" res.flux_err_pct res.err_u
    @test res.n_flux > 0
    @test isfinite(res.flux_err_pct)
    @test res.flux_err_pct < 25.0
    @test res.err_u < 0.15
end

@testset "C8E1 parametric m=1,2,3" begin
    errs = Float64[]
    for m in (1.0, 2.0, 3.0)
        msh = Base.invokelatest(quadrado; ndiv=14, show=false, nome="da_c8e1_m$m")
        dad = setup_da_c8e1(msh; m=m, n_int=6)
        res = test_da_c8e1(dad; m=m, npg=12, verbose=true)
        push!(errs, res.flux_err_pct)
        @test isfinite(res.flux_err_pct)
        @test res.flux_err_pct < 40.0
    end
    @info "parametric flux %" errs
end
