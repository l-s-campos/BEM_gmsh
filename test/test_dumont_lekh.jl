# Lekhnitskii: SST (on-element, Cordeiro & Leonel 2020) + sinh (nearly singular)
using Test
using DrWatson
using LinearAlgebra
@quickactivate :BEM

const Legendre = BEM.Legendre
const Point2D = BEM.Point2D
const Element = BEM.Element
const BEMdata = BEM.BEMdata
const AnisotropicElasticity = BEM.AnisotropicElasticity

function _make_dad()
    poly = Legendre(1)
    nodes = [Point2D((ξ + 1) / 2, 0.0) for ξ in poly.nodes]
    elem = Element(index=[1, 2], Jacobian=ones(2), Length=1.0, Region=1)
    params = BEM.lekhnitskii_params(2.0, 1.0, 0.5, 0.25; θ_deg=0.0)
    props = AnisotropicElasticity(params)
    dad = BEMdata(
        name="lekh", dimension=2, elements=[elem], element_type=poly,
        elem_weight=SVector(1.0, 1.0), Nodes=nodes,
        Normal=[Point2D(0.0, -1.0), Point2D(0.0, -1.0)],
        internalNodes=Point2D[], properties=props,
        BC=zeros(Int, 2), BV=zeros(2), n=2, ni=0, nt=2,
    )
    return dad, elem, nodes, poly
end

function _sinh_ref(dad, poly, nodes, pf; npg=250, L=1.0)
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

@testset "Lekhnitskii nearly-sing → sinh (SST off-element)" begin
    dad, elem, nodes, poly = _make_dad()
    @test BEM.supports_dumont(dad)
    qsi, w = gausslegendre(16)
    BEM.set_cache!(dad; qsi=qsi, w=w)

    for d in (1e-1, 1e-2, 1e-3, 1e-4)
        pf = Point2D(0.5, -d)
        Hd = zeros(2, 4); Gd = zeros(2, 4)
        BEM.integraelem_dumont!(Hd, Gd, dad, elem, nodes, pf, qsi, w)
        Hi = zeros(2, 4); Gi = zeros(2, 4)
        BEM.integraelem(dad, elem, nodes, pf, Hi, Gi)
        @test Hi ≈ Hd atol=1e-14
        @test Gi ≈ Gd atol=1e-14

        Href, Gref = _sinh_ref(dad, poly, nodes, pf; L=elem.Length)
        eG = norm(Gd - Gref) / max(norm(Gref), 1e-30)
        eH = norm(Hd - Href) / max(norm(Href), 1e-30)
        @info "Lekh near d=$d" eG eH
        @test eG < 1e-7
        @test eH < 1e-5
    end
end

@testset "Lekhnitskii on-element SST vs tiny-offset sinh" begin
    dad, elem, nodes, poly = _make_dad()
    qsi, w = gausslegendre(20)
    BEM.set_cache!(dad; qsi=qsi, w=w)

    # collocation on element (mid) — true singular; reference = sinh at d=1e-10
    pf_on = Point2D(0.5, 0.0)
    Hd = zeros(2, 4); Gd = zeros(2, 4)
    BEM.integraelem_sst!(Hd, Gd, dad, elem, nodes, pf_on, qsi, w)

    pf_eps = Point2D(0.5, -1e-10)
    Href, Gref = _sinh_ref(dad, poly, nodes, pf_eps; npg=400, L=elem.Length)

    eG = norm(Gd - Gref) / max(norm(Gref), 1e-30)
    eH = norm(Hd - Href) / max(norm(Href), 1e-30)
    @info "Lekh SST on-element" eG eH Gd Gref Hd Href
    # on-element SST should stay finite and track the near-limit
    @test all(isfinite, Gd) && all(isfinite, Hd)
    @test eG < 0.15   # log singularity: offset reference is imperfect
    @test eH < 0.25
end
