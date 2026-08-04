# 2D Kelvin Dumont vs high-order sinh (single element)
using Test
using DrWatson
using LinearAlgebra
@quickactivate :BEM

const Legendre = BEM.Legendre
const Point2D = BEM.Point2D
const Element = BEM.Element
const BEMdata = BEM.BEMdata
const Elasticity = BEM.Elasticity

@testset "Kelvin Dumont vs sinh" begin
    poly = Legendre(1)
    nodes = [Point2D((ξ + 1) / 2, 0.0) for ξ in poly.nodes]
    L = 1.0
    elem = Element(index=[1, 2], Jacobian=ones(2), Length=L, Region=1)
    props = Elasticity(1.0, 0.3, 1.0; plane_strain=true)
    dad = BEMdata(
        name="kelvin", dimension=2, elements=[elem], element_type=poly,
        elem_weight=SVector(1.0, 1.0), Nodes=nodes,
        Normal=[Point2D(0.0, -1.0), Point2D(0.0, -1.0)],
        internalNodes=Point2D[], properties=props,
        BC=zeros(Int, 2), BV=zeros(2), n=2, ni=0, nt=2,
    )
    @test BEM.supports_dumont(dad)

    qsi, w = gausslegendre(16)
    BEM.set_cache!(dad; qsi=qsi, w=w)

    function sinh_ref(pf; npg=200)
        a, _, dist = BEM.closest_point_1d(poly, nodes, pf)
        b = max(dist / L, 1e-14)
        u, wu = gausslegendre(npg)
        eta, ww = BEM.sinhtrans(u, wu, a, b)
        N, dN = BEM.shapefun(poly, eta)
        Hm = zeros(2, 4); Gm = zeros(2, 4)
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
                cols = (2(j - 1) + 1):(2j)
                Nj = N[i, j] * wi
                for β in 1:2, α in 1:2
                    Gm[α, cols[β]] += U[α, β] * Nj
                    Hm[α, cols[β]] += T[α, β] * Nj
                end
            end
        end
        return Hm, Gm
    end

    for d in (1e-1, 1e-2, 1e-3, 1e-4)
        pf = Point2D(0.5, -d)
        Hd = zeros(2, 4); Gd = zeros(2, 4)
        BEM.integraelem_dumont!(Hd, Gd, dad, elem, nodes, pf, qsi, w)
        # also through integrate_element dispatch
        Hi = zeros(2, 4); Gi = zeros(2, 4)
        BEM.integrate_element(dad, elem, nodes, pf, Hi, Gi)
        @test Hi ≈ Hd
        @test Gi ≈ Gd

        Href, Gref = sinh_ref(pf)
        eG = norm(Gd - Gref) / max(norm(Gref), 1e-30)
        eH = norm(Hd - Href) / max(norm(Href), 1e-30)
        @info "Kelvin d=$d" eG eH
        @test eG < 1e-6
        @test eH < 1e-5
    end
end
