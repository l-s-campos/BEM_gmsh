# Compact-kernel local BEM with DIBEM domain integrals
using Test
using LinearAlgebra
using Statistics: mean
using BEM

include(joinpath(@__DIR__, "..", "data", "Laplace", "Laplace_dad.jl"))

function _lbem_gauss_ustar(R, ri; dim=2, n=96, power::Union{Nothing,Int}=nothing)
    s = min(float(R), float(ri))
    s <= 0 && return 0.0
    ξ, w = BEM.gausslegendre(n)
    acc = 0.0
    pwr = power === nothing ? (dim - 1) : power
    @inbounds for i in 1:n
        ρ = (ξ[i] + 1) / 2 * s
        acc += local_u_star(ρ, ri; dim=dim) * ρ^pwr * w[i] * (s / 2)
    end
    return acc
end

function _square_dad_lbem(ndiv; nome="lbem")
    msh = Base.invokelatest(quadrado; ndiv=ndiv, show=false, nome=nome)
    return format2d(msh, Laplace(1.0); pontointerno=true)
end

function _set_dirichlet_u!(dad, ufun)
    for i in 1:dad.n
        dad.BC[i] = 0
        dad.BV[i] = ufun(dad.Nodes[i])
    end
end

function _set_mixed_square!(dad, ufun, qfun)
    for i in 1:dad.n
        p = dad.Nodes[i]
        n = dad.Normal[i]
        if abs(abs(n[1]) - 1) < 0.5
            dad.BC[i] = 0
            dad.BV[i] = ufun(p)
        else
            dad.BC[i] = 1
            dad.BV[i] = qfun(p, n)
        end
    end
end

@testset "local kernel C¹ at r = r_i" begin
    for (dim, ri) in ((2, 0.4), (3, 0.7))
        @test local_u_star(ri, ri; dim=dim) ≈ 0 atol=1e-14
        @test local_du_dr(ri, ri; dim=dim) ≈ 0 atol=1e-14
        @test local_u_star(ri * 1.01, ri; dim=dim) == 0
        @test local_du_dr(ri * 1.01, ri; dim=dim) == 0
        @test local_du_dn(ri * [1.0, 0.0, 0.0][1:dim], [1.0, 0.0, 0.0][1:dim], ri; dim=dim) ≈ 0 atol=1e-14
        # inside: 2D kernel is positive
        dim == 2 && @test local_u_star(ri / 2, ri; dim=2) > 0
        v0 = local_ball_volume(ri; dim=dim)
        if dim == 2
            @test v0 ≈ π * ri^2
            # Δ(b r²) = 4b = 1/(π r_i²)
            b = 1 / (4π * ri^2)
            @test 4b ≈ 1 / v0
        else
            @test v0 ≈ 4π * ri^3 / 3
            b = 1 / (8π * ri^3)
            @test 6b ≈ 1 / v0
        end
    end
end

@testset "compact RIM primitives vs Gauss" begin
    ri = 0.35
    for R in (0.1, 0.35, 0.9, 2.0)
        # 2D integrand ~ ρ log ρ near 0; plain Gauss is only ~1e-8 relative
        @test radial_integral_local_ustar(R, ri; dim=2) ≈ _lbem_gauss_ustar(R, ri; dim=2) rtol=1e-6
        @test radial_integral_local_ustar(R, ri; dim=3) ≈ _lbem_gauss_ustar(R, ri; dim=3) rtol=1e-6
        s = min(R, ri)
        @test radial_integral_local_one(R, ri; dim=2) ≈ s^2 / 2
        @test radial_integral_local_one(R, ri; dim=3) ≈ s^3 / 3
    end
    # full-disk ∫ u_i* dA = r_i² / 8
    Ψ = radial_integral_local_ustar(ri, ri; dim=2)
    @test 2π * Ψ ≈ ri^2 / 8 rtol=1e-12
    for pwr in (2, 3)
        for R in (0.1, 0.35, 0.9)
            @test radial_integral_local_ustar(R, ri; dim=2, power=pwr) ≈
                _lbem_gauss_ustar(R, ri; dim=2, power=pwr) rtol=1e-6
        end
    end
    Ψ3 = radial_integral_local_ustar(ri, ri; dim=2, power=3)
    @test π * Ψ3 ≈ ri^4 / 96 rtol=1e-12
end

@testset "clip element to disk Γ ∩ B" begin
    poly = BEM.Equispaced(1)   # nodes at ξ = ±1
    # chord on x-axis: x(ξ) = (ξ, 0)
    nodes = [Point2D(-1.0, 0.0), Point2D(1.0, 0.0)]
    c = Point2D(0.0, 0.0)
    @test clip_element_to_ball(poly, nodes, c, 0.5) == [(-0.5, 0.5)]
    @test clip_element_to_ball(poly, nodes, c, 2.0) == [(-1.0, 1.0)]
    @test isempty(clip_element_to_ball(poly, nodes, Point2D(0.0, 2.0), 0.5))
    # offset disk covering only the right half of the chord
    segs = clip_element_to_ball(poly, nodes, Point2D(1.0, 0.0), 1.0)
    @test length(segs) == 1
    @test segs[1][1] ≈ 0.0 atol=1e-12
    @test segs[1][2] ≈ 1.0 atol=1e-12
    # tangent
    segs_t = clip_element_to_ball(poly, nodes, Point2D(0.0, 0.5), 0.5)
    @test length(segs_t) <= 1
    if !isempty(segs_t)
        @test segs_t[1][2] - segs_t[1][1] < 0.05
    end
end

@testset "local BEM requires internals" begin
    msh = quadrado(ndiv=4, show=false, nome="lbem_no_int")
    dad = format2d(msh, Laplace(1.0); pontointerno=false)
    @test_throws ErrorException assemble_local_bem!(dad)
end

@testset "DIBEM disk identities (interior collocation)" begin
    dad = _square_dad_lbem(8; nome="lbem_disk")
    # radius small enough that a centre node is a full disk inside (0,1)²
    assemble_local_bem!(dad; radius=0.2, npg=20, rbf=PHS(3; poly_deg=-1))
    pts = all_points(dad)
    # pick an interior node near (0.5, 0.5)
    i0 = dad.n + argmin([norm(p - Point2D(0.5, 0.5)) for p in dad.internalNodes])
    p = pts[i0]
    @test 0.2 < p[1] < 0.8 && 0.2 < p[2] < 0.8
    onesv = ones(dad.nt)
    ID1 = dad.lbem_ID1
    IDu = dad.dibem_ID
    @test ID1[i0] ≈ π * 0.2^2 rtol=1e-10
    @test IDu[i0] ≈ (0.2^2) / 8 rtol=1e-10
    M1 = dad.lbem_M1
    Mf = dad.M
    @test abs((M1 * onesv)[i0] - ID1[i0]) / (abs(ID1[i0]) + eps()) < 0.2
    @test abs((Mf * onesv)[i0] - IDu[i0]) / (abs(IDu[i0]) + eps()) < 0.2
    # disk of radius 0.2 about the centre misses Γ
    nclip = 0
    for el in dad.elements
        nclip += length(clip_element_to_ball(dad.element_type, dad.Nodes[el.index], p, 0.2))
    end
    @test nclip == 0
end

@testset "local CPD M1 reproduces polynomials" begin
    dad = _square_dad_lbem(8; nome="lbem_cpd")
    assemble_local_bem!(dad; npg=16)
    pts = all_points(dad)
    M1 = dad.lbem_M1
    ID1, Mx, My = dad.lbem_ID1, dad.lbem_Mx, dad.lbem_My
    Mxx, Mxy, Myy = dad.lbem_Mxx, dad.lbem_Mxy, dad.lbem_Myy
    onesv = ones(dad.nt)
    @test maximum(abs.(M1 * onesv .- ID1)) / maximum(abs.(ID1)) < 1e-8
    x = [p[1] for p in pts]
    y = [p[2] for p in pts]
    Ix = Mx .+ [pts[i][1] for i in 1:dad.nt] .* ID1
    Iy = My .+ [pts[i][2] for i in 1:dad.nt] .* ID1
    @test maximum(abs.(M1 * x .- Ix)) / (maximum(abs.(Ix)) + eps()) < 1e-6
    @test maximum(abs.(M1 * y .- Iy)) / (maximum(abs.(Iy)) + eps()) < 1e-6
    xi = [p[1] for p in pts]
    yi = [p[2] for p in pts]
    Ixx = Mxx .+ 2 .* xi .* Mx .+ xi.^2 .* ID1
    Ixy = Mxy .+ yi .* Mx .+ xi .* My .+ xi .* yi .* ID1
    Iyy = Myy .+ 2 .* yi .* My .+ yi.^2 .* ID1
    @test maximum(abs.(M1 * (x .* x) .- Ixx)) / (maximum(abs.(Ixx)) + eps()) < 1e-5
    @test maximum(abs.(M1 * (x .* y) .- Ixy)) / (maximum(abs.(Ixy)) + eps()) < 1e-5
    @test maximum(abs.(M1 * (y .* y) .- Iyy)) / (maximum(abs.(Iyy)) + eps()) < 1e-5
end

function _lbem_rmse_pair(dad, f, ufun, qfun)
    solve_local_bem!(dad, f; npg=16, source=:local)
    pts = all_points(dad)
    uex = ufun.(pts)
    qex = [qfun(p, n) for (p, n) in zip(dad.Nodes, dad.Normal)]
    el = sqrt(mean(abs2, dad.T .- uex))
    ql = sqrt(mean(abs2, dad.q .- qex))
    Tl = copy(dad.T)
    solve_local_bem!(dad, f; npg=16, source=:global)
    eg = sqrt(mean(abs2, dad.T .- uex))
    qg = sqrt(mean(abs2, dad.q .- qex))
    return el, eg, ql, qg, Tl
end

@testset "local vs global M" begin
    dad = _square_dad_lbem(8; nome="lbem_Mcmp")
    assemble_local_bem!(dad; radius=0.2, npg=16)
    pts = all_points(dad)
    i0 = dad.n + argmin([norm(p - Point2D(0.5, 0.5)) for p in dad.internalNodes])
    p = pts[i0]
    @test 0.2 < p[1] < 0.8 && 0.2 < p[2] < 0.8
    IDu = dad.dibem_ID
    Ml, Mg = dad.lbem_M_local, dad.lbem_M_global
    onesv = ones(dad.nt)
    x = [q[1] for q in pts]
    @test (Ml * onesv)[i0] ≈ IDu[i0] rtol=1e-6
    @test (Mg * onesv)[i0] ≈ IDu[i0] rtol=1e-6
    exact_x = p[1] * IDu[i0]
    err_l = abs((Ml * x)[i0] - exact_x)
    err_g = abs((Mg * x)[i0] - exact_x)
    @info "M ∫ x u* on interior disk" err_local=err_l err_global=err_g
    @test err_l / (abs(exact_x) + eps()) < 1e-4
    @test err_l < err_g || err_l < 1e-10

    u4(q) = q[1]^2 + q[2]^2
    dad4 = _square_dad_lbem(8; nome="lbem_Mcmp4")
    _set_dirichlet_u!(dad4, u4)
    e4l, e4g, q4l, q4g, _ = _lbem_rmse_pair(dad4, 4.0, u4,
        (pp, n) -> -2 * (pp[1] * n[1] + pp[2] * n[2]))
    @info "Poisson f=4 local vs global" u_local=e4l u_global=e4g q_local=q4l q_global=q4g
    @test e4l < 0.05 && e4g < 0.05

    ux3(q) = q[1]^3 / 6
    dadx = _square_dad_lbem(8; nome="lbem_Mcmpx")
    _set_dirichlet_u!(dadx, ux3)
    exl, exg, qxl, qxg, _ = _lbem_rmse_pair(dadx, q -> q[1], ux3,
        (pp, n) -> -(pp[1]^2 / 2) * n[1])
    @info "Poisson f=x local vs global" u_local=exl u_global=exg q_local=qxl q_global=qxg
    @test exl < 0.05
    @test exl <= exg * 1.1 + 1e-12
end

@testset "Laplace f=0, u=x" begin
    ufun(p) = p[1]
    dad = _square_dad_lbem(8; nome="lbem_ux")
    _set_dirichlet_u!(dad, ufun)
    solve_local_bem!(dad, 0; npg=16)
    pts = all_points(dad)
    uex = ufun.(pts)
    @test all(isfinite, dad.T)
    rmse = sqrt(mean(abs2, dad.T .- uex))
    @info "local BEM u=x RMSE" rmse
    @test rmse < 1e-5
    qex = [-dad.Normal[i][1] for i in 1:dad.n]
    qerr = sqrt(mean(abs2, dad.q .- qex))
    @info "local BEM u=x flux RMSE" qerr
    @test qerr < 1e-4
end

@testset "Poisson u=x²+y² (f=4) Dirichlet" begin
    ufun(p) = p[1]^2 + p[2]^2
    dad = _square_dad_lbem(8; nome="lbem_xy2")
    _set_dirichlet_u!(dad, ufun)
    solve_local_bem!(dad, 4.0; npg=16)
    pts = all_points(dad)
    uex = ufun.(pts)
    @test all(isfinite, dad.T)
    rmse = sqrt(mean(abs2, dad.T .- uex))
    @info "local BEM u=x²+y² RMSE" rmse
    @test rmse < 0.05
    qex = [-2 * (p[1] * n[1] + p[2] * n[2]) for (p, n) in zip(dad.Nodes, dad.Normal)]
    qerr = sqrt(mean(abs2, dad.q .- qex))
    @info "local BEM u=x²+y² flux RMSE" qerr
    @test qerr < 0.05
end

@testset "elastic compact Kelvin C¹ on circle" begin
    props = Elasticity(1.0, 0.3, 1.0)
    ri = 0.35
    a, b, c, d = local_kelvin_abcd(props, ri)
    λ, μ, ν = props.lambda, props.mu, effective_nu(props)
    for θ in range(0, 2π; length=9)[1:8]
        ê = Point2D(cos(θ), sin(θ))
        r = ri * ê
        U, T = local_kelvin(props, r, ê, ri)
        @test norm(U) < 1e-12
        @test norm(T) < 1e-12
    end
    U0, T0 = local_kelvin(props, Point2D(ri * 1.01, 0.0), Point2D(1.0, 0.0), ri)
    @test U0 == zero(U0) && T0 == zero(T0)
end

@testset "elastic local BEM patch test" begin
    msh = quadrado_elasticity(ndiv=8, show=false, nome="lbem_el_patch")
    props = Elasticity(1.0, 0.3, 1.0)
    dad = format2d(msh, props; pontointerno=true)
    ana = ana_elasticity_patch(; E=1.0, ν=0.3, εxx=0.01)
    apply_analytical_bc!(dad, ana)
    solve_local_bem!(dad; npg=12)
    @test all(isfinite, dad.u)
    err = rel_error(dad)
    @info "elastic local BEM patch rel_error" err
    @test err < 0.05
end

@testset "Poisson mixed BC" begin
    ufun(p) = p[1]^2 + p[2]^2
    qfun(p, n) = -2 * (p[1] * n[1] + p[2] * n[2])
    dad = _square_dad_lbem(8; nome="lbem_mix")
    _set_mixed_square!(dad, ufun, qfun)
    solve_local_bem!(dad, 4.0; npg=16)
    pts = all_points(dad)
    uex = ufun.(pts)
    @test all(isfinite, dad.T)
    rmse = sqrt(mean(abs2, dad.T .- uex))
    @info "local BEM mixed RMSE" rmse
    @test rmse < 0.05
    qerr = sqrt(mean(abs2, dad.q .- qfun.(dad.Nodes, dad.Normal)))
    @info "local BEM mixed flux RMSE" qerr
    @test qerr < 0.05
end
