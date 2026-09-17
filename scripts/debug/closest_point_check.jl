# Brute-force vs closest_point_1d on P3 circular / chordal meshes.
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf
include(datadir("Laplace", "Laplace_dad.jl"))

MAT1 = (E1=124.04e3, E2=10.09e3, G12=6.03e3, ν12=0.334, η12_1=1.255, η12_2=-0.031)
props = AnisotropicElasticity(lekhnitskii_engineering(
    MAT1.E1, MAT1.E2, MAT1.G12, MAT1.ν12; η12_1=MAT1.η12_1, η12_2=MAT1.η12_2))

function brute(poly, nodes, pf; n=401)
    ξb = 0.0; db = Inf; xb = nodes[1]
    @inbounds for ξ in range(-1.0, 1.0; length=n)
        N, _ = BEM.shapefun(poly, ξ)
        x = (N * nodes)[1]
        d = norm(x - pf)
        if d < db
            db = d; ξb = ξ; xb = x
        end
    end
    return ξb, xb, db
end

function proj_quality(poly, nodes, pf, ξ)
    res, J, x = BEM._proj1d_rj(poly, nodes, pf, clamp(ξ, -1.0, 1.0))
    return res, J, norm(x - pf)
end

function check_mesh(msh, label)
    dad = format2d(msh, props; tipo=2, pontointerno=false)
    poly = dad.element_type
    n = dad.n
    nbad = 0; nmax = 0; nend = 0; nfar = 0
    worst = (ratio=1.0, i=0, el=0, dN=0.0, dB=0.0, ξN=0.0, ξB=0.0, J=0.0)
    println("\n== $label  n=$(n) elems=$(length(dad.elements)) poly=$(typeof(poly).name.wrapper) nodes=$(poly.nodes)")
    for i in 1:n
        pf = dad.Nodes[i]
        seed = BEM._seed_1d(poly, dad.Nodes[dad.elements[1].index], pf)  # dummy
        for (ie, el) in enumerate(dad.elements)
            xj = dad.Nodes[el.index]
            ξ0 = BEM._seed_1d(poly, xj, pf)
            ξN, xN, dN = closest_point_1d(poly, xj, pf; ξ0=ξ0)
            ξB, xB, dB = brute(poly, xj, pf)
            res, J, _ = proj_quality(poly, xj, pf, ξN)
            onel = i in el.index
            # Newton farther than brute by more than scan spacing
            if dN > dB + 1e-4 * max(dB, 1.0)
                nbad += 1
                ratio = dN / max(dB, 1e-30)
                if ratio > worst.ratio
                    worst = (ratio=ratio, i=i, el=ie, dN=dN, dB=dB, ξN=ξN, ξB=ξB, J=J)
                end
            end
            if abs(ξN) < 0.999 && J < 0
                nmax += 1
            end
            if onel && dN > 1e-8
                nfar += 1
                @printf("  ON-ELEMENT miss i=%d el=%d  dN=%.3e ξN=%.4f ξB=%.4f dB=%.3e\n",
                    i, ie, dN, ξN, ξB, dB)
            end
            if !onel && (abs(ξN + 1) < 1e-12 || abs(ξN - 1) < 1e-12)
                nend += 1
            end
        end
    end
    npair = n * length(dad.elements)
    println("  pairs=$npair  Newton-worse-than-brute=$nbad  interior-max(J<0)=$nmax  snapped-to-end=$nend  on-el dist>1e-8=$nfar")
    if nbad > 0
        w = worst
        pf = dad.Nodes[w.i]
        @printf("  worst: i=%d pf=(%.1f,%.1f) el=%d  dNewton=%.4f dBrute=%.4f ratio=%.3f  ξN=%.4f ξB=%.4f J=%.3e\n",
            w.i, pf[1], pf[2], w.el, w.dN, w.dB, w.ratio, w.ξN, w.ξB, w.J)
    end
    return nbad
end

# unit: straight segment (must be exact)
function unit_straight()
    poly = BEM.Equispaced(1)
    nodes = [Point2D(0.0, 0.0), Point2D(1.0, 0.0)]
    ξ, x, d = closest_point_1d(poly, nodes, Point2D(0.5, 0.1); ξ0=BEM._seed_1d(poly, nodes, Point2D(0.5, 0.1)))
    println("straight mid: ξ=$ξ x=$x d=$d  (expect 0.5, (0.5,0), 0.1)")
    ξ, x, d = closest_point_1d(poly, nodes, Point2D(-1.0, 0.0); ξ0=BEM._seed_1d(poly, nodes, Point2D(-1.0, 0.0)))
    println("straight past end: ξ=$ξ d=$d  (expect -1, 1)")
end

# unit: circular quadratic, pf at origin — closest is unique, any ξ (constant r)
# pf at a point inside near the arc
function unit_arc()
    poly = BEM.Legendre(2)
    θ = (0.0, π/10, π/5)
    nodes = [Point2D(600*cos(t), 600*sin(t)) for t in θ]
    # collocation-like: evaluate Equispaced geometry at GL? use these 3 points as interpolant
    pf = Point2D(300.0, 0.0)  # on +x, inside
    ξ0 = BEM._seed_1d(poly, nodes, pf)
    ξN, xN, dN = closest_point_1d(poly, nodes, pf; ξ0=ξ0)
    ξB, xB, dB = brute(poly, nodes, pf)
    res, J, _ = proj_quality(poly, nodes, pf, ξN)
    println("arc pf=$(pf)")
    @printf("  seed=%.4f  Newton ξ=%.4f d=%.4f  brute ξ=%.4f d=%.4f  (x-pf)·x'=%.3e J=%.3e\n",
        ξ0, ξN, dN, ξB, dB, res, J)
    # pf on the far side of the bulge (should prefer an endpoint)
    pf2 = Point2D(600*cos(π/10) + 50*cos(π/10), 600*sin(π/10) + 50*sin(π/10))
    ξ0 = BEM._seed_1d(poly, nodes, pf2)
    ξN, _, dN = closest_point_1d(poly, nodes, pf2; ξ0=ξ0)
    ξB, _, dB = brute(poly, nodes, pf2)
    _, J, _ = proj_quality(poly, nodes, pf2, ξN)
    println("arc pf outside bulge $(pf2)")
    @printf("  seed=%.4f  Newton ξ=%.4f d=%.4f  brute ξ=%.4f d=%.4f J=%.3e\n",
        ξ0, ξN, dN, ξB, dB, J)
end

println("=== unit ===")
unit_straight()
unit_arc()

q2 = datadir("elastico", "p3_cmp_q2.msh")
lin = datadir("elastico", "p3_cmp_lin.msh")
isfile(q2) && check_mesh(q2, "circular CAD-order-2")
isfile(lin) && check_mesh(lin, "chordal linear-then-setOrder")
println("done")
