# Guiggiani singular + sinh nearly-singular integration (2D curve elements)
using Test
using DrWatson
@quickactivate :BEM
using LinearAlgebra
using StaticArrays
using FastGaussQuadrature

const Legendre = BEM.Legendre
const Point2D = BEM.Point2D
const Element = BEM.Element
const BEMdata = BEM.BEMdata
const Laplace = BEM.Laplace
const Elasticity = BEM.Elasticity

function _make_laplace_elem()
    poly = Legendre(1)
    nodes = [Point2D((ξ + 1) / 2, 0.0) for ξ in poly.nodes]
    L = 1.0
    elem = Element(index=collect(1:2), Jacobian=ones(2), Length=L, Region=1)
    dad = BEMdata(
        name="sing_lap",
        dimension=2,
        elements=[elem],
        element_type=poly,
        elem_weight=SVector{2}(1.0, 1.0),
        collocation=nodes,
        Normal=[Point2D(0.0, -1.0), Point2D(0.0, -1.0)],
        properties=Laplace(1.0),
        BC=zeros(Int, 2),
        BV=zeros(2),
        n=2,
        ni=0,
        nt=2,
    )
    return dad, elem, nodes, poly
end

function _make_kelvin_elem()
    poly = Legendre(1)
    nodes = [Point2D((ξ + 1) / 2, 0.0) for ξ in poly.nodes]
    L = 1.0
    elem = Element(index=collect(1:2), Jacobian=ones(2), Length=L, Region=1)
    dad = BEMdata(
        name="sing_kel",
        dimension=2,
        elements=[elem],
        element_type=poly,
        elem_weight=SVector{2}(1.0, 1.0),
        collocation=nodes,
        Normal=[Point2D(0.0, -1.0), Point2D(0.0, -1.0)],
        properties=Elasticity(1.0, 0.3, 1.0; plane_strain=true),
        BC=zeros(Int, 4),
        BV=zeros(4),
        n=2,
        ni=0,
        nt=2,
    )
    return dad, elem, nodes, poly
end

function _sinh_ref_scalar(dad, poly, nodes, pf; npg=400, L=1.0)
    a, _, dist = BEM.closest_point_1d(poly, nodes, pf)
    b = max(dist / L, 1e-14)
    qsi, w = gausslegendre(npg)
    eta, ww = BEM.nearfield_1d(a, b; qsi=qsi, w=w)
    N, dN = BEM.shapefun(poly, eta)
    h = zeros(length(nodes)); g = zeros(length(nodes))
    for i in eachindex(eta)
        pg = zero(Point2D); dx = zero(Point2D)
        for j in eachindex(nodes)
            pg += N[i, j] * nodes[j]
            dx += dN[i, j] * nodes[j]
        end
        J = norm(dx)
        J < 1e-30 && continue
        r = pg - pf
        norm(r) < 1e-30 && continue
        nrm = BEM.tan2normal(dx / J)
        U, T = BEM.fundamental(dad, r, nrm)
        wi = J * ww[i]
        for j in eachindex(nodes)
            h[j] += N[i, j] * T * wi
            g[j] += N[i, j] * U * wi
        end
    end
    return h, g
end

function _sinh_ref_vec(dad, poly, nodes, pf; npg=400, L=1.0)
    a, _, dist = BEM.closest_point_1d(poly, nodes, pf)
    b = max(dist / L, 1e-14)
    qsi, w = gausslegendre(npg)
    eta, ww = BEM.nearfield_1d(a, b; qsi=qsi, w=w)
    N, dN = BEM.shapefun(poly, eta)
    nN = length(nodes)
    h = zeros(2, 2nN); g = zeros(2, 2nN)
    for i in eachindex(eta)
        pg = zero(Point2D); dx = zero(Point2D)
        for j in eachindex(nodes)
            pg += N[i, j] * nodes[j]
            dx += dN[i, j] * nodes[j]
        end
        J = norm(dx)
        J < 1e-30 && continue
        r = pg - pf
        norm(r) < 1e-30 && continue
        nrm = BEM.tan2normal(dx / J)
        U, T = BEM.fundamental(dad, r, nrm)
        wi = J * ww[i]
        for j in 1:nN
            cols = (2(j - 1) + 1):(2j)
            Nj = N[i, j] * wi
            for β in 1:2, α in 1:2
                g[α, cols[β]] += U[α, β] * Nj
                h[α, cols[β]] += T[α, β] * Nj
            end
        end
    end
    return h, g
end

@testset "sinh nearly-singular Laplace" begin
    dad, elem, nodes, poly = _make_laplace_elem()
    qsi, w = gausslegendre(24)
    BEM.set_cache!(dad; qsi=qsi, w=w)
    for d in (1e-2, 1e-3, 1e-4)
        pf = Point2D(0.5, -d)
        h = zeros(2); g = zeros(2)
        BEM.integrate_element(dad, elem, nodes, pf, h, g)
        href, gref = _sinh_ref_scalar(dad, poly, nodes, pf; npg=400, L=elem.Length)
        @test norm(g - gref) / max(norm(gref), 1e-30) < 5e-3
        @test norm(h - href) / max(norm(href), 1e-30) < 5e-3
    end
end

@testset "Guiggiani on-element Laplace finite" begin
    dad, elem, nodes, poly = _make_laplace_elem()
    qsi, w = gausslegendre(24)
    BEM.set_cache!(dad; qsi=qsi, w=w)
    pf = nodes[1]
    h = zeros(2); g = zeros(2)
    BEM.integrate_element(dad, elem, nodes, pf, h, g)
    @test all(isfinite, h) && all(isfinite, g)
    @test norm(g) > 0
    href, gref = _sinh_ref_scalar(dad, poly, nodes, Point2D(pf[1], -1e-10);
        npg=600, L=elem.Length)
    @test norm(g - gref) / max(norm(gref), 1e-30) < 0.05
end

@testset "sinh nearly-singular Kelvin" begin
    dad, elem, nodes, poly = _make_kelvin_elem()
    qsi, w = gausslegendre(24)
    BEM.set_cache!(dad; qsi=qsi, w=w)
    pf = Point2D(0.5, -1e-3)
    H = zeros(2, 4); G = zeros(2, 4)
    BEM.integrate_element(dad, elem, nodes, pf, H, G)
    Href, Gref = _sinh_ref_vec(dad, poly, nodes, pf; npg=400, L=elem.Length)
    @test norm(G - Gref) / max(norm(Gref), 1e-30) < 5e-3
    @test norm(H - Href) / max(norm(Href), 1e-30) < 5e-3
end

@testset "Guiggiani on-element Kelvin finite" begin
    dad, elem, nodes, _ = _make_kelvin_elem()
    qsi, w = gausslegendre(24)
    BEM.set_cache!(dad; qsi=qsi, w=w)
    pf = nodes[1]
    H = zeros(2, 4); G = zeros(2, 4)
    BEM.integrate_element(dad, elem, nodes, pf, H, G)
    @test all(isfinite, H) && all(isfinite, G)
    @test norm(G) > 0
end
