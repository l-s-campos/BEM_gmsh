# Helmholtz / Burton–Miller: hypersingular identity + α=0 copies CBIE.
using Test
using LinearAlgebra
using StaticArrays
using BEM

@testset "Laplace hypersingular closed form" begin
    r = Point2D(0.3, 0.4)
    n = Point2D(1.0, 0.0)
    nf = Point2D(0.0, 1.0)
    lap = Laplace(1.0)
    R = norm(r)
    e = r / R
    kh = fundamental_hyper(lap, r, n, nf)
    @test kh.U ≈ dot(e, nf) / (2π * R) atol=1e-14
    @test kh.T ≈ -(dot(nf, n) - 2 * dot(e, n) * dot(e, nf)) / (2π * R^2) atol=1e-14
    r3 = Point3D(0.3, 0.4, 0.5)
    n3 = Point3D(0.0, 0.0, 1.0)
    nf3 = Point3D(0.0, 1.0, 0.0)
    R3 = norm(r3)
    e3 = r3 / R3
    kh3 = fundamental_hyper(lap, r3, n3, nf3)
    @test kh3.U ≈ dot(e3, nf3) / (4π * R3^2) atol=1e-14
    @test kh3.T ≈ -(dot(nf3, n3) - 3 * dot(e3, n3) * dot(e3, nf3)) / (4π * R3^3) atol=1e-14
end

@testset "Laplace HBIE interp vs Richardson H" begin
    mk(name) = format2d(quadrado(ndiv=4, show=false, nome=name), Laplace(1.0);
        pontointerno=false, tipo=1)
    Hi, Gi = H_G_hyper(mk("t_hbie_i"); npg=8, threaded=false, laurent=:interp)
    Hr, Gr = H_G_hyper(mk("t_hbie_r"); npg=8, threaded=false, laurent=:richardson)
    Ha, Ga = H_G_hyper(mk("t_hbie_a"); npg=8, threaded=false, laurent=:auto)
    @test norm(Hi - Hr) / max(norm(Hr), 1e-16) < 5e-4
    @test norm(Hi - Ha) / max(norm(Ha), 1e-16) < 5e-4
    @test norm(Gi - Gr) / max(norm(Gr), 1e-16) < 5e-4
    @test all(isfinite, Hi) && all(isfinite, Hr)
end

@testset "dibem_hyper_mass analytic ID′ vs FD" begin
    dad = format2d(quadrado(ndiv=4, show=false, nome="t_idp"), Laplace(1.0);
        pontointerno=true, tipo=1)
    assemble!(dad; npg=8, threaded=false)
    DIBEM(dad; method=:dense, rbf=PHS(3; poly_deg=1))
    geos = BEM._rim_build_elements(dad)
    pts = all_points(dad)
    ε = 1e-6
    for i in (1, dad.n ÷ 2, dad.n)
        x = pts[i]
        nf = dad.Normal[i]
        IDfd = (BEM._dibem_ID_at(dad, x + ε * nf; geos=geos) -
                BEM._dibem_ID_at(dad, x - ε * nf; geos=geos)) / (2ε)
        IDan = BEM._dibem_IDp_at(dad, x, nf; geos=geos)
        @test IDan ≈ IDfd rtol=5e-4 atol=1e-8
    end
    Mp = dibem_hyper_mass(dad; rbf=PHS(3; poly_deg=1), threaded=false)
    @test size(Mp) == (dad.nt, dad.nt)
    @test all(isfinite, Mp)
    @test dad.ni > 0
    M = Matrix(dad.M)
    @test Mp[dad.n + 1:dad.nt, :] ≈ M[dad.n + 1:dad.nt, :]
    @test any(!iszero, Mp[1, dad.n + 1:dad.nt])   # internals in c / columns
    d0 = format2d(quadrado(ndiv=4, show=false, nome="t_idp0"), Laplace(1.0);
        pontointerno=false, tipo=1)
    assemble!(d0; npg=8, threaded=false)
    @test_throws ArgumentError dibem_hyper_mass(d0; rbf=PHS(3; poly_deg=1))
end

@testset "Burton–Miller α=0 copies CBIE" begin
    dad = format2d(quadrado(ndiv=4, show=false, nome="t_bm"), Laplace(1.0);
        pontointerno=true, tipo=1)
    assemble!(dad; npg=8, threaded=false)
    H0, G0 = copy(dad.H), copy(dad.G)
    DIBEM(dad; method=:dense, rbf=PHS(3; poly_deg=1))
    M0 = copy(dad.M)
    Hp, Gp = H_G_hyper(dad; npg=8, threaded=false)
    Mp = dibem_hyper_mass(dad; rbf=PHS(3; poly_deg=1))
    Hc, Gc, Mc = combine_burton_miller(H0, G0, M0, Hp, Gp, Mp, 0.0; n=dad.n)
    @test Hc ≈ H0
    @test Gc ≈ G0
    @test Mc ≈ M0
end

@testset "Helmholtz G is complex" begin
    kph = fundamental(Helmholtz(; ω=2.0, c=1.0), Point2D(0.3, 0.4), Point2D(1.0, 0.0))
    @test kph.U isa Complex
    @test isfinite(imag(kph.U))
    kph3 = fundamental(Helmholtz(; ω=2.0, c=1.0), Point3D(0.3, 0.4, 0.5), Point3D(0.0, 0.0, 1.0))
    @test kph3.U isa Complex
    @test isfinite(real(kph3.U)) && isfinite(imag(kph3.U))
end

@testset "Helmholtz 3D FS closed form" begin
    r = Point3D(0.3, 0.4, 0.5)
    n = Point3D(0.0, 0.0, 1.0)
    κ = 2.0
    kp = fundamental(Helmholtz(; ω=κ, c=1.0), r, n)
    R = norm(r)
    e = cis(κ * R)
    @test kp.U ≈ e / (4π * R)
    @test kp.T ≈ e * (im * κ * R - 1) * dot(r, n) / (4π * R^3)
end

@testset "Helmholtz κ→0 matches Laplace (k=1, H flipped)" begin
    lap = Laplace(1.0)
    h0 = Helmholtz(; ω=1e-5, c=1.0)
    r2 = Point2D(0.3, 0.4)
    n2 = Point2D(1.0, 0.0)
    nf2 = Point2D(0.0, 1.0)
    kph = fundamental(h0, r2, n2)
    kpl = fundamental(lap, r2, n2)
    # 2-D G keeps the Hankel log-κ constant; H and hyper drop it.
    @test real(kph.T) ≈ -kpl.T rtol=1e-5
    hh = fundamental_hyper(h0, r2, n2, nf2)
    hl = fundamental_hyper(lap, r2, n2, nf2)
    @test real(hh.U) ≈ hl.U rtol=1e-4
    @test real(hh.T) ≈ -hl.T rtol=1e-4
    r3 = Point3D(0.3, 0.4, 0.5)
    n3 = Point3D(0.0, 0.0, 1.0)
    nf3 = Point3D(0.0, 1.0, 0.0)
    kph3 = fundamental(h0, r3, n3)
    kpl3 = fundamental(lap, r3, n3)
    @test real(kph3.U) ≈ kpl3.U rtol=1e-8
    @test real(kph3.T) ≈ -kpl3.T rtol=1e-8
    hh3 = fundamental_hyper(h0, r3, n3, nf3)
    hl3 = fundamental_hyper(lap, r3, n3, nf3)
    @test real(hh3.U) ≈ hl3.U rtol=1e-8
    @test real(hh3.T) ≈ -hl3.T rtol=1e-8
end

@testset "Helmholtz 3D hyper vs finite difference" begin
    props = Helmholtz(; ω=1.5, c=1.0)
    r = Point3D(0.3, 0.4, 0.5)
    n = Point3D(0.0, 0.0, 1.0)
    nf = normalize(Point3D(0.2, 0.8, 0.1))
    kh = fundamental_hyper(props, r, n, nf)
    ε = 1e-6
    Gp, Gm = fundamental(props, r - ε * nf, n), fundamental(props, r + ε * nf, n)
    @test kh.U ≈ (Gp.U - Gm.U) / (2ε) rtol=1e-5
    @test kh.T ≈ (Gp.T - Gm.T) / (2ε) rtol=1e-5
end

@testset "Helmholtz assemble+solve" begin
    dad = format2d(quadrado(ndiv=4, show=false, nome="t_helm"), Helmholtz(; ω=1.0, c=1.0);
        pontointerno=false, tipo=1)
    assemble!(dad; npg=8, threaded=false)
    @test eltype(dad.H) <: Complex
    solve(dad)
    @test all(isfinite, real.(dad.T))
    @test all(isfinite, imag.(dad.T))
end

@testset "scalar assemble! near_factor" begin
    function mk(name; ω=8.0)
        format2d(quadrado(ndiv=8, show=false, nome=name), Helmholtz(; ω=ω, c=1.0);
            pontointerno=false, tipo=1)
    end
    d1 = mk("t_nf_15"); assemble!(d1; npg=8, threaded=false, near_factor=1.5)
    dinf = mk("t_nf_inf"); assemble!(dinf; npg=8, threaded=false, near_factor=Inf)
    @test eltype(d1.H) <: Complex
    @test norm(d1.H - dinf.H) / max(norm(dinf.H), 1e-16) > 1e-6
    @test norm(d1.G - dinf.G) / max(norm(dinf.G), 1e-16) > 1e-6
    d2 = mk("t_nf_pos"); H_G_full_direct(d2, 8; threaded=false, near_factor=1.5)
    @test d2.H ≈ d1.H
    lap = format2d(quadrado(ndiv=6, show=false, nome="t_nf_lap"), Laplace(1.0);
        pontointerno=false, tipo=1)
    @test auto_near_factor(lap) == 1.5
    assemble!(lap; npg=8, threaded=false, near_factor=Inf)
    @test all(isfinite, lap.H) && all(isfinite, lap.G)

    dlo = mk("t_nf_auto_lo"; ω=1.0)
    @test auto_near_factor(dlo) == 1.5
    assemble!(dlo; npg=8, threaded=false)   # :auto → 1.5
    @test dlo.near_factor == 1.5
    dhi = mk("t_nf_auto_hi"; ω=8.0)
    @test auto_near_factor(dhi) == Inf
    assemble!(dhi; npg=8, threaded=false)   # :auto → Inf
    @test dhi.near_factor == Inf
    @test dhi.H ≈ dinf.H
end

@testset "Helmholtz H_G_hyper 2D" begin
    dad = format2d(quadrado(ndiv=4, show=false, nome="t_helm_hbie"), Helmholtz(; ω=1.0, c=1.0);
        pontointerno=false, tipo=1)
    Hp, Gp = H_G_hyper(dad; npg=8, threaded=false)
    @test eltype(Hp) <: Complex
    @test size(Hp) == (dad.n, dad.nt)
    @test size(Gp) == (dad.n, dad.n)
    @test all(isfinite, Hp) && all(isfinite, Gp)
end

@testset "Helmholtz 3D assemble + H_G_hyper" begin
    include(joinpath(@__DIR__, "..", "data", "Laplace", "cube_mesh.jl"))
    msh = mesh_unit_cube(; L=1.0, ndiv=1, nome="t_helm3d")
    dad = format3d(msh, Helmholtz(; ω=1.0, c=1.0); pontointerno=false)
    assemble!(dad; npg=8, threaded=false)
    @test eltype(dad.H) <: Complex
    @test all(isfinite, dad.H) && all(isfinite, dad.G)
    Hp, Gp = H_G_hyper(dad; npg=8, threaded=false)
    @test eltype(Hp) <: Complex
    @test size(Hp, 1) == dad.n
    @test all(isfinite, Hp) && all(isfinite, Gp)
end

# ---------------------------------------------------------------------------
# Guiggiani :interp vs :richardson on every Helmholtz kernel
# (2-D/3-D CBIE G,H and HBIE G′,H′)
# ---------------------------------------------------------------------------

function _relerr(A, B)
    return norm(A - B) / max(norm(B), 1e-16)
end

function _helm2d_pair(props, nodes, pf, nf; hyper::Bool)
    poly = BEM.Equispaced(1)
    nN = length(nodes)
    Z = zeros(ComplexF64, nN)
    return ξ -> begin
        N, dN = BEM.shapefun(poly, ξ)
        pg = N[1, 1] * nodes[1] + N[1, 2] * nodes[2]
        dx = dN[1, 1] * nodes[1] + dN[1, 2] * nodes[2]
        J = norm(dx)
        r = pg - pf
        norm(r) < 1e-30 && return Z, Z
        nrm = Point2D(dx[2], -dx[1]) / J
        kp = hyper ? fundamental_hyper(props, r, nrm, nf) : fundamental(props, r, nrm)
        Fg = zeros(ComplexF64, nN)
        Fh = zeros(ComplexF64, nN)
        @inbounds for j in 1:nN
            NjJ = N[1, j] * J
            Fg[j] = kp.U * NjJ
            Fh[j] = kp.T * NjJ
        end
        return Fg, Fh
    end
end

function _helm3d_pair(props, corners, pf, nf; hyper::Bool)
    poly = BEM.Equispaced(1)
    nN = 4
    Z = zeros(ComplexF64, nN)
    return (ξ, η) -> begin
        L, Lξ, Lη = BEM.shapefun2D(poly, poly, ξ, η)
        pg = zero(Point3D)
        tξ = zero(Point3D)
        tη = zero(Point3D)
        @inbounds for k in 1:nN
            pg += L[1, k] * corners[k]
            tξ += Lξ[1, k] * corners[k]
            tη += Lη[1, k] * corners[k]
        end
        Jv = cross(tξ, tη)
        J = norm(Jv)
        J < 1e-30 && return Z, Z
        r = pg - pf
        norm(r) < 1e-30 && return Z, Z
        nrm = Jv / J
        kp = hyper ? fundamental_hyper(props, r, nrm, nf) : fundamental(props, r, nrm)
        Fg = zeros(ComplexF64, nN)
        Fh = zeros(ComplexF64, nN)
        @inbounds for j in 1:nN
            NjJ = L[1, j] * J
            Fg[j] = kp.U * NjJ
            Fh[j] = kp.T * NjJ
        end
        return Fg, Fh
    end
end

@testset "Helmholtz Guiggiani interp vs Richardson" begin
    props = Helmholtz(; ω=1.5, c=1.0)
    qsi, w = BEM.gausslegendre(16)

    # 2-D line element, collocation at mid-node image.
    nodes2 = [Point2D(0.0, 0.0), Point2D(1.0, 0.0)]
    pf2 = Point2D(0.5, 0.0)
    nf2 = Point2D(0.0, -1.0)
    a = 0.0

    f2c = _helm2d_pair(props, nodes2, pf2, nf2; hyper=false)
    IgI, IhI = guiggiani_GH(f2c, a; order_G=0, order_H=-1, qsi=qsi, w=w, laurent=:interp)
    IgR, IhR = guiggiani_GH(f2c, a; order_G=0, order_H=-1, qsi=qsi, w=w, laurent=:richardson)
    IgA, IhA = guiggiani_GH(f2c, a; order_G=0, order_H=-1, qsi=qsi, w=w,
        props=props, poly=BEM.Equispaced(1), nodes=nodes2, laurent=:auto)
    @test all(isfinite, IgI) && all(isfinite, IhI)
    @test all(isfinite, IgR) && all(isfinite, IhR)
    @test all(isfinite, IgA) && all(isfinite, IhA)
    @test _relerr(IhI, IhR) < 5e-4          # CPV H (0 on a flat element)
    @test _relerr(IhA, IhR) < 5e-4
    @test _relerr(IgI, IgR) < 5e-3          # log G (Richardson slow)
    @test _relerr(IgI, IgA) < 5e-3

    f2h = _helm2d_pair(props, nodes2, pf2, nf2; hyper=true)
    HgI, HhI = guiggiani_GH(f2h, a; order_G=-1, order_H=-2, qsi=qsi, w=w, laurent=:interp)
    HgR, HhR = guiggiani_GH(f2h, a; order_G=-1, order_H=-2, qsi=qsi, w=w, laurent=:richardson)
    HgA, HhA = guiggiani_GH(f2h, a; order_G=-1, order_H=-2, qsi=qsi, w=w,
        props=props, poly=BEM.Equispaced(1), nodes=nodes2, laurent=:auto)
    @test all(isfinite, HgI) && all(isfinite, HhI)
    @test all(isfinite, HgR) && all(isfinite, HhR)
    @test all(isfinite, HgA) && all(isfinite, HhA)
    @test _relerr(HgI, HgR) < 5e-4          # CPV G′
    @test _relerr(HgA, HgR) < 5e-4
    # HFP H′: imag (regular Hankel) matches; real 1/ρ² remainder is slow
    # under interpolant F₀. Closed-form F₋₂ (`:auto`) is the reference.
    @test _relerr(imag.(HhI), imag.(HhR)) < 1e-10
    @test _relerr(imag.(HhA), imag.(HhR)) < 1e-10
    @test _relerr(HhR, HhA) < 5e-3
    @test _relerr(HhI, HhA) < 0.1

    # 3-D parent square, collocation at centre.
    corners = [Point3D(-1, -1, 0), Point3D(1, -1, 0),
               Point3D(-1, 1, 0), Point3D(1, 1, 0)]
    pf3 = Point3D(0, 0, 0)
    nf3 = Point3D(0, 0, 1)

    f3c = _helm3d_pair(props, corners, pf3, nf3; hyper=false)
    SgI, ShI = guiggiani_GH_surface(f3c, 0.0, 0.0; order_G=0, order_H=-1,
        qsi=qsi, w=w, nθ=12, laurent=:interp)
    SgR, ShR = guiggiani_GH_surface(f3c, 0.0, 0.0; order_G=0, order_H=-1,
        qsi=qsi, w=w, nθ=12, laurent=:richardson)
    @test all(isfinite, SgI) && all(isfinite, ShI)
    @test all(isfinite, SgR) && all(isfinite, ShR)
    @test _relerr(SgI, SgR) < 5e-4          # bounded polar G
    @test _relerr(ShI, ShR) < 5e-4          # CPV H

    f3h = _helm3d_pair(props, corners, pf3, nf3; hyper=true)
    TgI, ThI = guiggiani_GH_surface(f3h, 0.0, 0.0; order_G=-1, order_H=-2,
        qsi=qsi, w=w, nθ=12, laurent=:interp)
    TgR, ThR = guiggiani_GH_surface(f3h, 0.0, 0.0; order_G=-1, order_H=-2,
        qsi=qsi, w=w, nθ=12, laurent=:richardson)
    @test all(isfinite, TgI) && all(isfinite, ThI)
    @test all(isfinite, TgR) && all(isfinite, ThR)
    @test _relerr(TgI, TgR) < 5e-4          # CPV G′
    @test _relerr(ThI, ThR) < 5e-4          # HFP H′
end

@testset "Helmholtz assemble interp vs Richardson" begin
    mk2(name) = format2d(quadrado(ndiv=4, show=false, nome=name), Helmholtz(; ω=1.0, c=1.0);
        pontointerno=false, tipo=1)
    function cbie2(name, laurent)
        dad = mk2(name)
        set_cache!(dad; laurent=laurent)
        assemble!(dad; npg=8, threaded=false)
        return dad.H, dad.G
    end
    Hi, Gi = cbie2("t_helm_ci", :interp)
    Hr, Gr = cbie2("t_helm_cr", :richardson)
    @test all(isfinite, Hi) && all(isfinite, Gi)
    @test all(isfinite, Hr) && all(isfinite, Gr)
    @test _relerr(Hi, Hr) < 5e-4
    @test _relerr(Gi, Gr) < 5e-2   # log G, Richardson slow on Hankel

    Hpi, Gpi = H_G_hyper(mk2("t_helm_hi"); npg=8, threaded=false, laurent=:interp)
    Hpr, Gpr = H_G_hyper(mk2("t_helm_hr"); npg=8, threaded=false, laurent=:richardson)
    @test all(isfinite, Hpi) && all(isfinite, Gpi)
    @test all(isfinite, Hpr) && all(isfinite, Gpr)
    @test _relerr(Hpi, Hpr) < 2e-3  # HFP H′ real part (interp F₀)
    @test _relerr(Gpi, Gpr) < 5e-4

    include(joinpath(@__DIR__, "..", "data", "Laplace", "cube_mesh.jl"))
    function cube(name, laurent; hyper::Bool)
        msh = mesh_unit_cube(; L=1.0, ndiv=1, nome=name)
        dad = format3d(msh, Helmholtz(; ω=1.0, c=1.0); pontointerno=false)
        if hyper
            return H_G_hyper(dad; npg=8, threaded=false, laurent=laurent)
        end
        set_cache!(dad; laurent=laurent)
        assemble!(dad; npg=8, threaded=false)
        return dad.H, dad.G
    end
    H3i, G3i = cube("t_helm3ci", :interp; hyper=false)
    H3r, G3r = cube("t_helm3cr", :richardson; hyper=false)
    @test all(isfinite, H3i) && all(isfinite, G3i)
    @test all(isfinite, H3r) && all(isfinite, G3r)
    @test _relerr(H3i, H3r) < 5e-4
    @test _relerr(G3i, G3r) < 5e-4
    Hp3i, Gp3i = cube("t_helm3hi", :interp; hyper=true)
    Hp3r, Gp3r = cube("t_helm3hr", :richardson; hyper=true)
    @test all(isfinite, Hp3i) && all(isfinite, Gp3i)
    @test all(isfinite, Hp3r) && all(isfinite, Gp3r)
    @test _relerr(Hp3i, Hp3r) < 5e-4
    @test _relerr(Gp3i, Gp3r) < 5e-4
end
