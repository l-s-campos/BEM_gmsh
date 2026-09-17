# Elasticity DIBEM (Domain.jl / Domain_fast.jl) — dense + H-matrix backends
using Test
using LinearAlgebra
using Statistics: mean
using BEM
using BEM.HMatrices

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

# FD only in tests: check book û against Navier and p̂ = σ(û)·n.
function _navier_col(Ufun, x, k; ν=0.3, μ=0.5, ε=1e-5)
    λ = 2μ * ν / (1 - 2ν)
    u(y) = Ufun(y)[:, k]
    function fd(f, y)
        dx, dy = SVector(ε, 0.0), SVector(0.0, ε)
        return (f(y + dx) - f(y - dx)) / (2ε), (f(y + dy) - f(y - dy)) / (2ε)
    end
    function divu(y)
        ux, uy = fd(u, y)
        return ux[1] + uy[2]
    end
    uxx, _ = fd(y -> fd(u, y)[1], x)
    _, uyy = fd(y -> fd(u, y)[2], x)
    ddx, ddy = fd(divu, x)
    Lu = μ * (uxx + uyy) + (λ + μ) * SVector(ddx, ddy)
    return Lu
end

function _traction_fd(Ufun, x, k, nrm; ν=0.3, μ=0.5, ε=1e-6)
    λ = 2μ * ν / (1 - 2ν)
    u(y) = Ufun(y)[:, k]
    dx, dy = SVector(ε, 0.0), SVector(0.0, ε)
    ux = (u(x + dx) - u(x - dx)) / (2ε)
    uy = (u(x + dy) - u(x - dy)) / (2ε)
    exx, eyy, gxy = ux[1], uy[2], ux[2] + uy[1]
    sxx = λ * (exx + eyy) + 2μ * exx
    syy = λ * (exx + eyy) + 2μ * eyy
    sxy = μ * gxy
    return SVector(sxx * nrm[1] + sxy * nrm[2], sxy * nrm[1] + syy * nrm[2])
end

@testset "DRM elasticity f=1+r uses analytic û, t̂" begin
    ν, μ = 0.3, 0.5
    @test BEM._drm_u(SVector(0.0, 0.0), ν, μ) == zeros(2, 2)
    @test BEM._drm_t(SVector(0.0, 0.0), SVector(1.0, 0.0), ν, μ) == zeros(2, 2)
    T05 = BEM._drm_t(SVector(0.3, 0.4), SVector(0.6, 0.8), 0.5, μ)
    @test all(isfinite, T05)

    dad = _dad_elast(6; nome="drm_el", n_int=2)
    H_G_full_direct(dad; npg=8, threaded=false)
    drm = build_drm_matrices(dad; npg=8, kernel=:one_plus_r)
    @test size(drm.M) == (2dad.nt, 2dad.nt)
    @test all(isfinite, drm.M)
    @test tr(drm.M) > 0
    @test has_cache(dad, :M)

    nu = effective_nu(dad.properties)
    Gmod = shear_modulus(dad.properties)
    pts = all_points(dad)
    i, j = 1, min(3, dad.nt)
    rvec = dad.Nodes[i] - pts[j]
    T = BEM._drm_t(rvec, dad.Normal[i], nu, Gmod, Val(:one_plus_r))
    @test drm.η[2i-1:2i, 2j-1:2j] ≈ -T
    U = BEM._drm_u(pts[i] - pts[j], nu, Gmod, Val(:one_plus_r))
    @test drm.Ψ[2i-1:2i, 2j-1:2j] ≈ -U
end

@testset "DRM elasticity f=r particular solutions" begin
    ν, μ = 0.3, 0.5
    ker = Val(:r)
    x = SVector(0.3, 0.4)
    nrm = SVector(0.6, 0.8)
    @test BEM._drm_u(SVector(0.0, 0.0), ν, μ, ker) == zeros(2, 2)
    @test BEM._drm_t(SVector(0.0, 0.0), nrm, ν, μ, ker) == zeros(2, 2)
    T = BEM._drm_t(x, nrm, ν, μ, ker)
    @test all(isfinite, T)
    Ufun = y -> BEM._drm_u(SVector{2}(y), ν, μ, ker)
    f = norm(x)  # f = r
    for k in 1:2
        Lu = _navier_col(Ufun, x, k; ν=ν, μ=μ)
        b = f * (k == 1 ? SVector(1.0, 0.0) : SVector(0.0, 1.0))
        # book L(û) = +f e_k
        @test norm(Lu - b) / (norm(b) + 1e-14) < 5e-3
        tfd = _traction_fd(Ufun, x, k, nrm; ν=ν, μ=μ)
        @test T[:, k] ≈ tfd rtol=1e-4 atol=1e-6
    end

    dad = _dad_elast(6; nome="drm_el_r", n_int=2)
    H_G_full_direct(dad; npg=8, threaded=false)
    drm = build_drm_matrices(dad; npg=8, kernel=:r)
    @test drm.kernel === :r
    @test size(drm.M) == (2dad.nt, 2dad.nt)
    @test all(isfinite, drm.M)
    @test has_cache(dad, :M)
    nu = effective_nu(dad.properties)
    Gmod = shear_modulus(dad.properties)
    pts = all_points(dad)
    i, j = 1, min(3, dad.nt)
    rvec = dad.Nodes[i] - pts[j]
    @test drm.η[2i-1:2i, 2j-1:2j] ≈
        -BEM._drm_t(rvec, dad.Normal[i], nu, Gmod, ker)
    @test drm.Ψ[2i-1:2i, 2j-1:2j] ≈
        -BEM._drm_u(pts[i] - pts[j], nu, Gmod, ker)
end

@testset "DRM elasticity f=MQ particular solutions" begin
    ν, μ = 0.3, 0.5
    C = 0.4
    ker = Val(:mq)
    x = SVector(0.3, 0.4)
    nrm = SVector(0.6, 0.8)
    @test all(isfinite, BEM._drm_u(SVector(0.0, 0.0), ν, μ, ker, C))
    @test BEM._drm_t(SVector(0.0, 0.0), nrm, ν, μ, ker, C) == zeros(2, 2)
    T = BEM._drm_t(x, nrm, ν, μ, ker, C)
    @test all(isfinite, T)
    Ufun = y -> BEM._drm_u(SVector{2}(y), ν, μ, ker, C)
    f = hypot(norm(x), C)
    for k in 1:2
        Lu = _navier_col(Ufun, x, k; ν=ν, μ=μ)
        b = f * (k == 1 ? SVector(1.0, 0.0) : SVector(0.0, 1.0))
        @test norm(Lu - b) / (norm(b) + 1e-14) < 5e-3
        tfd = _traction_fd(Ufun, x, k, nrm; ν=ν, μ=μ)
        @test T[:, k] ≈ tfd rtol=1e-4 atol=1e-6
    end

    dad = _dad_elast(6; nome="drm_el_mq", n_int=2)
    H_G_full_direct(dad; npg=8, threaded=false)
    drm = build_drm_matrices(dad; npg=8, kernel=:mq, C=C)
    @test drm.kernel === :mq
    @test drm.C ≈ C
    @test size(drm.M) == (2dad.nt, 2dad.nt)
    @test all(isfinite, drm.M)
    nu = effective_nu(dad.properties)
    Gmod = shear_modulus(dad.properties)
    pts = all_points(dad)
    i, j = 1, min(3, dad.nt)
    rvec = dad.Nodes[i] - pts[j]
    @test drm.η[2i-1:2i, 2j-1:2j] ≈
        -BEM._drm_t(rvec, dad.Normal[i], nu, Gmod, ker, C)
    @test drm.Ψ[2i-1:2i, 2j-1:2j] ≈
        -BEM._drm_u(pts[i] - pts[j], nu, Gmod, ker, C)
end

@testset "DIBEM elasticity MQ Gram" begin
    dad = _dad_elast(6; nome="dibem_el_mq", n_int=2)
    M = DIBEM(dad; method=:dense, rbf=MQ(; C=0.01, poly_deg=-1))
    @test size(M) == (2dad.nt, 2dad.nt)
    @test all(isfinite, M)
    @test dad.dibem_rbf.paper
end

@testset "constant cells RIM matches ID" begin
    msh = Base.invokelatest(quadrado_elasticity; ndiv=6, show=false, nome="cell_el_t")
    dad = format2d(msh, Elasticity(1.0, 0.3, 1.0; plane_strain=true);
        tipo=1, pontointerno=true)
    @test !isempty(extract_domain_cells(dad))
    @test length(dad.internalNodes) == length(dad.cells)
    IF, ID, pts = BEM._dibem_elast_IF_ID(dad, PHS(3; poly_deg=-1); npg=8)
    cm = build_cell_mass(dad; npg=8)
    @test size(cm.M_cell, 1) == 2 * dad.nt
    @test size(cm.M, 1) == 2 * dad.nt
    n2 = 2 * dad.nt
    b = zeros(n2)
    for i in 1:dad.nt
        b[2i-1] = 1.0
    end
    IDb = zeros(n2)
    for i in 1:dad.nt
        IDb[2i-1:2i] .= ID[2i-1:2i, :] * SVector(1.0, 0.0)
    end
    colsum = zeros(n2)
    for k in 1:length(cm.cells)
        colsum .+= cm.M_cell[:, 2k-1]
    end
    @test norm(colsum - IDb) / (norm(IDb) + 1e-14) < 1e-10
    @test norm(cm.M * b - IDb) / (norm(IDb) + 1e-14) < 1e-8
end

@testset "DIBEM poly_deg=-1 regression and CPD constants" begin
    dad = _dad_elast(6; nome="dibem_el_poly", n_int=2)
    M0 = DIBEM(dad; method=:dense, rbf=PHS(3; poly_deg=-1))
    @test size(M0) == (2dad.nt, 2dad.nt)
    @test all(isfinite, M0)
    dad1 = _dad_elast(6; nome="dibem_el_poly1", n_int=2)
    M1 = DIBEM(dad1; method=:dense, rbf=PHS(3; poly_deg=1))
    @test all(isfinite, M1)
    ID = dad1.dibem_ID
    b = zeros(2 * dad1.nt)
    for i in 1:dad1.nt
        b[2i-1] = 1.0
    end
    IDb = zeros(2 * dad1.nt)
    for i in 1:dad1.nt
        IDb[2i-1:2i] .= ID[2i-1:2i, :] * SVector(1.0, 0.0)
    end
    @test norm(M1 * b - IDb) / (norm(IDb) + 1e-14) < 1e-8
    @test norm(M0 * b - IDb) / (norm(IDb) + 1e-14) < 1e-6
end

@testset "DIBEM volume centers (cell centroids)" begin
    msh = Base.invokelatest(quadrado_elasticity; ndiv=6, show=false, nome="dibem_vol")
    dad = format2d(msh, Elasticity(1.0, 0.3, 1.0; plane_strain=true);
        tipo=1, pontointerno=true)
    H_G_full_direct(dad; npg=8, threaded=false)
    M = DIBEM(dad; method=:dense, rbf=PHS(3; poly_deg=1), centers=:cells)
    @test dad.dibem_centers === :cells
    @test size(M) == (2dad.nt, 2dad.nt)
    @test length(dad.dibem_c) == length(dad.cells)
    @test all(isfinite, M)
    @test count(<(0), dad.dibem_c) == 0
    ID = dad.dibem_ID
    b = zeros(2 * dad.nt)
    for i in 1:dad.nt
        b[2i-1] = 1.0
    end
    IDb = zeros(2 * dad.nt)
    for i in 1:dad.nt
        IDb[2i-1:2i] .= ID[2i-1:2i, :] * SVector(1.0, 0.0)
    end
    @test norm(M * b - IDb) / (norm(IDb) + 1e-14) < 1e-6
end

@testset "DRM elasticity poly_deg=1 finite" begin
    dad = _dad_elast(6; nome="drm_el_poly", n_int=2)
    H_G_full_direct(dad; npg=8, threaded=false)
    drm = build_drm_matrices(dad; npg=8, kernel=:r, poly_deg=1)
    @test size(drm.M) == (2dad.nt, 2dad.nt)
    @test all(isfinite, drm.M)
    @test drm.poly_deg == 1
end
