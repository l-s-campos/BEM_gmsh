# Fast DIBEM: H-matrix and FMM backends vs dense
using Test
using LinearAlgebra
using Statistics: mean
using BEM
using BEM.HMatrices

include(joinpath(@__DIR__, "..", "data", "Laplace", "Laplace_dad.jl"))

function _dad_square(ndiv, n_int; nome="dibem_fast")
    msh = Base.invokelatest(quadrado; ndiv=ndiv, show=false, nome=nome)
    dad = format2d(msh, Laplace(1.0); tipo=1, pontointerno=false)
    if n_int > 0
        xs = range(0.2, 0.8; length=n_int)
        set_internal_nodes!(dad, [SVector(float(x), float(y)) for y in xs for x in xs])
    end
    H_G_full_direct(dad, 10)
    return dad
end

@testset "DIBEM dense baseline" begin
    dad = _dad_square(6, 3; nome="dibem_d")
    M = DIBEM(dad; method=:dense)
    @test size(M) == (dad.nt, dad.nt)
    @test all(isfinite, M)
    @test M ≈ DIBEM_dense(dad)
end

@testset "Laplace DIBEM volume centers" begin
    msh = Base.invokelatest(quadrado; ndiv=6, show=false, nome="dibem_vol_L")
    dad = format2d(msh, Laplace(1.0); tipo=1, pontointerno=true)
    H_G_full_direct(dad; npg=8, threaded=false)
    M = DIBEM(dad; method=:dense, rbf=PHS(3; poly_deg=1), centers=:cells)
    @test dad.dibem_centers === :cells
    @test size(M) == (dad.nt, dad.nt)
    @test length(dad.dibem_c) == length(dad.cells)
    @test count(<(0), dad.dibem_c) == 0
    ID = dad.dibem_ID
    @test norm(M * ones(dad.nt) - ID) / (norm(ID) + 1e-14) < 1e-6
end

@testset "Laplace cells and CPD poly" begin
    msh = Base.invokelatest(quadrado; ndiv=6, show=false, nome="dibem_cells_L")
    dad = format2d(msh, Laplace(1.0); tipo=1, pontointerno=true)
    @test !isempty(extract_domain_cells(dad))
    H_G_full_direct(dad; npg=8, threaded=false)
    ID = zeros(dad.nt)
    BEM._dibem_accumulate_IF_ID!(zeros(dad.nt), ID, dad, PHS(3; poly_deg=-1))
    cm = build_cell_mass(dad; npg=8)
    @test norm(vec(sum(cm.M_cell; dims=2)) - ID) / (norm(ID) + 1e-14) < 1e-4
    onesv = ones(dad.nt)
    M0 = DIBEM(deepcopy(dad); method=:dense, rbf=PHS(3; poly_deg=-1))
    M1 = DIBEM(deepcopy(dad); method=:dense, rbf=PHS(3; poly_deg=1))
    @test norm(M0 * onesv - ID) / (norm(ID) + 1e-14) < 1e-10
    @test norm(M1 * onesv - ID) / (norm(ID) + 1e-14) < 1e-10
    drm = build_drm_matrices(deepcopy(dad), PHS(3; poly_deg=1); npg=8)
    @test all(isfinite, drm.M)
    @test drm.npoly == 3
end

@testset "DIBEM_Hmat ≈ dense" begin
    dad = _dad_square(8, 4; nome="dibem_h")
    Md = DIBEM(dad; method=:dense)
    # fresh dad for H-path
    dad2 = _dad_square(8, 4; nome="dibem_h2")
    Mh = DIBEM(dad2; method=:hmatrix, atol=1e-6, nmax=20)
    @test size(Mh) == size(Md)
    # ACA approx — compare matvecs (relative; not bit-identical to dense)
    rels = Float64[]
    for trial in 1:5
        x = randn(size(Md, 1))
        yd = Md * x
        yh = Mh * x
        push!(rels, norm(yd - yh) / (norm(yd) + 1e-14))
    end
    @info "Hmat vs dense matvec rel" mean=mean(rels) max=maximum(rels)
    @test mean(rels) < 0.25
    @test maximum(rels) < 0.4
end

@testset "DIBEM_FMM ≈ dense" begin
    dad = _dad_square(8, 4; nome="dibem_f")
    Md = DIBEM(dad; method=:dense)
    dad2 = _dad_square(8, 4; nome="dibem_f2")
    Mf = DIBEM(dad2; method=:fmm, eps=1e-5, f_method=:dense)
    @test Mf isa DibemFactoredOperator
    @test size(Mf) == size(Md)
    for trial in 1:3
        x = randn(size(Md, 1))
        yd = Md * x
        yf = Mf * x
        rel = norm(yd - yf) / (norm(yd) + 1e-14)
        @test rel < 0.35
    end
end

@testset "DIBEM_FMM works in Houbolt smoke" begin
    dad = _dad_square(6, 3; nome="dibem_hou")
    DIBEM(dad; method=:fmm, eps=1e-4, f_method=:dense)
    @test has_cache(dad, :M)
    # one small wave step path: operators exist
    T = solve_Houbolt(dad, 0.05, 0.15)
    @test all(isfinite, T)
end

@testset "DIBEM HODLR / HSS ≈ dense matvec" begin
    dad = _dad_square(6, 3; nome="dibem_st0")
    Md = DIBEM(dad; method=:dense)
    n = size(Md, 1)
    x = randn(n)
    yd = Md * x

    for (meth, kw) in (
            (:hodlr, (; nmax=12, atol=1e-5)),
            (:hss, (; nmax=12, rtol=1e-4, hss_method=:dense)),
        )
        dadm = _dad_square(6, 3; nome="dibem_st_$meth")
        Mm = DIBEM(dadm; method=meth, kw...)
        @test size(Mm, 1) == n
        ym = Mm * x
        rel = norm(yd - ym) / (norm(yd) + 1e-14)
        @info "DIBEM $meth vs dense" rel T=typeof(Mm)
        @test rel < 0.55
        @test size(Mm) == size(Md)
        @test all(isfinite, ym)
    end
    # HBS alias of HSS
    dadb = _dad_square(6, 3; nome="dibem_hbs")
    Mb = DIBEM(dadb; method=:hbs, nmax=12, rtol=1e-4)
    @test size(Mb, 1) == n
    yb = Mb * x
    @test all(isfinite, yb)
    @test norm(yd - yb) / (norm(yd) + 1e-14) < 0.5
end
