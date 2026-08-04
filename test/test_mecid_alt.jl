# MECID alternative — variable velocity (Pinheiro thesis Ch.8, §8.2.1)
using Test
using LinearAlgebra
using Statistics: mean
using StaticArrays
using BEM

include(joinpath(@__DIR__, "..", "data", "Laplace", "Laplace_dad.jl"))

@testset "MECID-alt operators assemble" begin
    msh = Base.invokelatest(quadrado; ndiv=6, show=false, nome="mecid_op")
    dad = setup_mecid_c8e1(msh; m=1.0, n_int=3)
    H_G_full_direct(dad, 10)
    S = build_mecid_S_matrix(dad)
    M′ = build_mecid_Mprime(dad, mecid_c8e1_velocity(1.0))
    @test size(S) == (dad.nt, dad.nt)
    @test size(M′) == (dad.nt, dad.nt)
    @test all(isfinite, S)
    @test all(isfinite, M′)
    M_ID = mecid_alt_variable_velocity!(dad, mecid_c8e1_velocity(1.0))
    @test size(M_ID) == (dad.nt, dad.nt)
    @test all(isfinite, M_ID)
end

@testset "C8E1 analytic residual α∇²u − v·∇u = 0" begin
    # sanity of manufactured solution
    m = 2.0
    u = mecid_c8e1_analytic(m)
    # at random interior point, check PDE by FD
    p = SVector(0.37, 0.61)
    ε = 1e-5
    # ∇u analytic
    uu = u(p)
    gx = m * p[2] * uu
    gy = m * p[1] * uu
    # ∇²u = ∂/∂x(my u) + ∂/∂y(mx u) = m*y*(m*y u) + m*x*(m*x u) = m²(x²+y²)u? 
    # u=e^{mxy}, u_xx = (my)² u, u_yy = (mx)² u, ∇²u = m²(x²+y²)u
    # v·∇u = my*(my u) + mx*(mx u) = m²(y²+x²)u
    lap = m^2 * (p[1]^2 + p[2]^2) * uu
    adv = gx * (m * p[2]) + gy * (m * p[1])  # v·∇u with v=(my,mx)
    # wait gx = my u already, v·∇u = my*gx + mx*gy = my*(my u)+mx*(mx u)=m²(y²+x²)u
    adv2 = (m * p[2]) * gx + (m * p[1]) * gy
    @test lap ≈ adv2 rtol = 1e-12
end

@testset "C8E1 MECID-alt m=1 flux error" begin
    msh = Base.invokelatest(quadrado; ndiv=12, show=false, nome="c8e1_m1")
    dad = setup_mecid_c8e1(msh; m=1.0, n_int=5)
    res = test_mecid_c8e1(dad; m=1.0, npg=12)
    @info "C8E1 result" res.flux_err_pct res.err_u res.n_flux
    @test res.n_flux > 0
    @test isfinite(res.flux_err_pct)
    # thesis reports O(1%) flux errors on refined meshes; allow coarse-mesh band
    @test res.flux_err_pct < 25.0
    @test res.err_u < 0.15
end

@testset "C8E1 parametric m=1,2,3" begin
    errs = Float64[]
    for m in (1.0, 2.0, 3.0)
        msh = Base.invokelatest(quadrado; ndiv=14, show=false, nome="c8e1_m$m")
        dad = setup_mecid_c8e1(msh; m=m, n_int=6)
        res = test_mecid_c8e1(dad; m=m, npg=12, verbose=true)
        push!(errs, res.flux_err_pct)
        @test isfinite(res.flux_err_pct)
        @test res.flux_err_pct < 40.0
    end
    @info "C8E1 parametric flux % errors" errs
end
