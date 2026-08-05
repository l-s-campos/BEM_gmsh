# Elasticity DIBEM (Domain.jl / Domain_fast.jl) — dense + H-matrix backends
using Test
using LinearAlgebra
using Statistics: mean
using BEM

include(joinpath(@__DIR__, "..", "data", "Laplace", "Laplace_dad.jl"))

function _dad_elast(ndiv; nome="dibem_el", n_int=0)
    msh = Base.invokelatest(quadrado_elasticity; ndiv=ndiv, show=false, nome=nome)
    dad = format2d(msh, Elasticity(1.0, 0.3, 1.0; plane_strain=true);
        tipo=1, pontointerno=false)
    if n_int > 0
        xs = range(0.25, 0.75; length=n_int)
        set_internal_nodes!(dad, [SVector(float(x), float(y)) for y in xs for x in xs])
    end
    return dad
end

@testset "DIBEM elasticity dense" begin
    dad = _dad_elast(6; nome="dibem_el_d", n_int=2)
    M = DIBEM(dad; method=:dense)
    @test size(M) == (2dad.nt, 2dad.nt)
    @test all(isfinite, M)
    @test M ≈ DIBEM_dense(dad)
    @test has_cache(dad, :M)
    # alias
    dad2 = _dad_elast(6; nome="dibem_el_alias", n_int=2)
    M2 = dibem_elasticity!(dad2)
    @test size(M2) == size(M)
end

@testset "DIBEM elasticity Hmat ≈ dense matvec" begin
    dad = _dad_elast(6; nome="dibem_el_h0", n_int=2)
    Md = DIBEM(dad; method=:dense)
    dad2 = _dad_elast(6; nome="dibem_el_h1", n_int=2)
    Mh = DIBEM(dad2; method=:hmatrix, atol=1e-5, nmax=16)
    @test Mh isa DibemElastFactoredOperator
    @test size(Mh) == size(Md)
    rels = Float64[]
    for _ in 1:4
        x = randn(size(Md, 1))
        yd = Md * x
        yh = Mh * x
        push!(rels, norm(yd - yh) / (norm(yd) + 1e-14))
    end
    @info "elast Hmat vs dense matvec" mean=mean(rels) max=maximum(rels)
    @test mean(rels) < 0.35
    @test maximum(rels) < 0.55
end

@testset "DIBEM elasticity FMM ≈ dense matvec" begin
    dad = _dad_elast(6; nome="dibem_el_f0", n_int=2)
    Md = DIBEM(dad; method=:dense)
    dad2 = _dad_elast(6; nome="dibem_el_f1", n_int=2)
    Mf = DIBEM(dad2; method=:fmm, eps=1e-6, nmax=16, f_method=:dense)
    @test Mf isa DibemElastFactoredOperator
    @test dad2.dibem_D isa FMM.KelvinFMMMatrix
    @test size(Mf) == size(Md)
    for _ in 1:3
        x = randn(size(Md, 1))
        rel = norm(Md * x - Mf * x) / (norm(Md * x) + 1e-14)
        @test rel < 1e-6
    end
end

@testset "DIBEM elasticity HSS-FMM (block sampler)" begin
    dad = _dad_elast(5; nome="dibem_el_hssf0", n_int=0)
    Md = DIBEM(dad; method=:dense)
    dad2 = _dad_elast(5; nome="dibem_el_hssf1", n_int=0)
    Mh = DIBEM(dad2; method=:hss, hss_method=:fmm, rtol=1e-5, nmax=14,
        rank=40, f_method=:dense)
    @test dad2.dibem_D isa HMatrices.HSSMatrix
    @test size(dad2.dibem_D, 1) == 2 * dad2.nt
    x = randn(size(Md, 1))
    rel = norm(Md * x - Mh * x) / (norm(Md * x) + 1e-14)
    @test rel < 0.2
end

@testset "DIBEM elasticity H2 (expand_tree + proxies)" begin
    dad = _dad_elast(6; nome="dibem_el_h2_0", n_int=0)
    Md = DIBEM(dad; method=:dense)
    dad2 = _dad_elast(6; nome="dibem_el_h2_1", n_int=0)
    Mh = DIBEM(dad2; method=:h2, nmax=14, rtol=1e-4, alpha=1.0, f_method=:dense)
    @test dad2.dibem_D isa HMatrices.H2Matrix
    @test size(dad2.dibem_D) == (2 * dad2.nt, 2 * dad2.nt)
    rels = Float64[]
    for _ in 1:3
        x = randn(size(Md, 1))
        push!(rels, norm(Md * x - Mh * x) / (norm(Md * x) + 1e-14))
    end
    @test mean(rels) < 0.15
    @test maximum(rels) < 0.3
end

@testset "DIBEM elasticity in thermo path" begin
    E, ν, α, Δθ = 1000.0, 0.3, 1e-5, 50.0
    props = Elasticity(E, ν, 1.0; plane_strain=true, α=α)
    msh = Base.invokelatest(quadrado_elasticity; ndiv=5, show=false, nome="dibem_el_th")
    dad = format2d(msh, props; pontointerno=false)
    fill!(dad.BC, 0)
    fill!(dad.BV, 0.0)
    H_G_full_direct(dad; npg=8, threaded=false)
    θfun = (x, y) -> Δθ * (1 + 0.05 * x)
    u = solve_thermoelastic!(dad; θ=θfun)
    @test length(u) == 2 * dad.n
    @test has_cache(dad, :M)
    @test size(dad.M, 1) == 2 * dad.nt
    @test all(isfinite, u)
end
