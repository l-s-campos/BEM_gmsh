# Dumont near-field path vs high-order sinh reference (2D Laplace element)
using Test
using DrWatson
@quickactivate :BEM
# BEM defines types inside the module without re-exporting all of them
const Legendre = BEM.Legendre
const Point2D = BEM.Point2D
const Element = BEM.Element
const BEMdata = BEM.BEMdata
const Laplace = BEM.Laplace
const Elasticity = BEM.Elasticity

@testset "Dumont C1 self-check" begin
    qsi, w = gausslegendre(16)
    for s in (0.3 + 0.1im, 1.5 + 0.0im, 0.2 + 1e-4im)
        exact = log((1 - s) / (-1 - s))
        Igl = sum(w[i] / (qsi[i] - s) for i in eachindex(qsi))
        C1 = BEM.correction_C1(s, qsi, w)
        @test abs(Igl + C1 - exact) < 1e-12
    end
end

@testset "Dumont vs sinh Laplace element" begin
    poly = Legendre(1)
    x0 = poly.nodes
    nodes = [Point2D((ξ + 1) / 2, 0.0) for ξ in x0]
    L = 1.0
    elem = Element(index=collect(1:2), Jacobian=ones(2), Length=L, Region=1)
    dad = BEMdata(
        name="test",
        dimension=2,
        elements=[elem],
        element_type=poly,
        elem_weight=SVector{2}(1.0, 1.0),
        Nodes=nodes,
        Normal=[Point2D(0.0, -1.0), Point2D(0.0, -1.0)],
        internalNodes=Point2D[],
        properties=Laplace(1.0),
        BC=zeros(Int, 2),
        BV=zeros(2),
        n=2,
        ni=0,
        nt=2,
    )
    qsi, w = gausslegendre(16)
    BEM.set_cache!(dad; qsi=qsi, w=w)

    function sinh_ref(pf; npg=200)
        a, _, dist = BEM.closest_point_1d(poly, nodes, pf)
        b = dist / L
        u, wu = gausslegendre(npg)
        eta, ww = BEM.sinhtrans(u, wu, a, max(b, 1e-14))
        N, dN = BEM.shapefun(poly, eta)
        h = zeros(2); g = zeros(2)
        for i in eachindex(eta)
            pg = zero(Point2D); dx = zero(Point2D)
            for j in 1:2
                pg += N[i, j] * nodes[j]
                dx += dN[i, j] * nodes[j]
            end
            J = norm(dx); nrm = BEM.tan2normal(dx / J)
            r = pg - pf; norm(r) < 1e-30 && continue
            U, T = BEM.fundamental(dad, r, nrm)
            wi = J * ww[i]
            for j in 1:2
                h[j] += N[i, j] * T * wi
                g[j] += N[i, j] * U * wi
            end
        end
        return h, g
    end

    for d in (1e-1, 1e-2, 1e-3, 1e-4)
        pf = Point2D(0.5, -d)
        hd = zeros(2); gd = zeros(2)
        BEM.integraelem_dumont!(hd, gd, dad, elem, nodes, pf, qsi, w)
        href, gref = sinh_ref(pf)
        eG = norm(gd - gref) / max(norm(gref), 1e-30)
        eH = norm(hd - href) / max(norm(href), 1e-30)
        @info "d=$d" eG eH gd gref hd href
        @test eG < 1e-5
        @test eH < 1e-4
    end
end

@testset "supports_dumont flags" begin
    dL = BEMdata(
        name="x", dimension=2, elements=Element[], element_type=Legendre(0),
        elem_weight=SVector{1}(1.0), Nodes=Point2D[], Normal=Point2D[],
        internalNodes=Point2D[], properties=Laplace(1.0),
        BC=Int[], BV=Float64[], n=0, ni=0, nt=0,
    )
    dE = BEMdata(
        name="x", dimension=2, elements=Element[], element_type=Legendre(0),
        elem_weight=SVector{1}(1.0), Nodes=Point2D[], Normal=Point2D[],
        internalNodes=Point2D[], properties=Elasticity(1.0, 0.3, 1.0),
        BC=Int[], BV=Float64[], n=0, ni=0, nt=0,
    )
    @test BEM.supports_dumont(dL)
    @test BEM.supports_dumont(dE)   # 2D Kelvin Dumont enabled
end
