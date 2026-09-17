# Diagnose anisotropic HBIE Laurent (Richardson vs Cordeiro SST) and free terms.
using DrWatson
@quickactivate :BEM
using LinearAlgebra, StaticArrays
include(datadir("Laplace", "Laplace_dad.jl"))

const MAT1 = (E1=124.04e3, E2=10.09e3, G12=6.03e3, ν12=0.334, η12_1=1.255, η12_2=-0.031)
props = AnisotropicElasticity(lekhnitskii_engineering(MAT1.E1, MAT1.E2, MAT1.G12, MAT1.ν12;
    η12_1=MAT1.η12_1, η12_2=MAT1.η12_2))

function _fhyp(props, poly, nodes, pf, nf)
    return ξ -> begin
        N, dN = BEM.shapefun(poly, ξ)
        pg = zero(eltype(nodes)); dx = zero(eltype(nodes))
        for k in eachindex(nodes)
            pg += N[1, k] * nodes[k]
            dx += dN[1, k] * nodes[k]
        end
        J = norm(dx)
        nrm = BEM.tan2normal(dx / J)
        Uh, Th = fundamental_hyper(props, pg, pf, nrm, nf)
        Uh = BEM._to_smat(Uh); Th = BEM._to_smat(Th)
        nN = length(nodes)
        Fg = zeros(2, 2nN); Fh = zeros(2, 2nN)
        for j in 1:nN
            cols = (2j - 1):(2j)
            Fg[:, cols] .= Uh .* (N[1, j] * J)
            Fh[:, cols] .= Th .* (N[1, j] * J)
        end
        return Fg, Fh
    end
end

function compare_laurent(label, poly, nodes, a; hR=1e-3)
    g = BEM._geom_1d(poly, nodes, a)
    Nrow, J, t, n = g
    pf = (BEM.shapefun(poly, a)[1] * nodes)[1]
    nf = n
    fhyp = _fhyp(props, poly, nodes, pf, nf)
    println("\n== $label  a=$a  J=$J  n=$n  |nodes|=$(length(nodes))")
    for s in (1.0, -1.0)
        abs(a + s) < 1e-14 && continue
        CA = laurent_coefficients(props, poly, nodes, a, s, :H, -2)
        CG = laurent_coefficients(props, poly, nodes, a, s, :G, -1)
        CA === nothing && (println("  s=$s analytic nothing"); continue)
        Fm2A, Fm1A, _ = CA
        _, Fm1G, _ = CG
        Fr = ρ -> fhyp(a + s * ρ)
        # default Richardson
        Fm2R, Fm1R, _ = BEM.laurent_coefficients(ρ -> Fr(ρ)[2], hR, Val(-2))
        _, Fm1GR, _ = BEM.laurent_coefficients(ρ -> Fr(ρ)[1], hR, Val(-1))
        ρ = 1e-8
        Fg, Fh = Fr(ρ)
        println("  s=$s")
        println("    ||Fm2A||=$(norm(Fm2A))  ||ρ²Fh||=$(norm(ρ^2 .* Fh))  rel(Fm2A,ρ²Fh)=$(norm(Fm2A .- ρ^2 .* Fh)/(norm(Fm2A)+1e-30))")
        println("    ||Fm2R||=$(norm(Fm2R))  rel(Fm2A,Fm2R)=$(norm(Fm2A .- Fm2R)/(norm(Fm2A)+1e-30))")
        println("    ||Fm1A||=$(norm(Fm1A))  ||Fm1R||=$(norm(Fm1R))  rel(Fm1A,Fm1R)=$(norm(Fm1A .- Fm1R)/(norm(Fm1A)+1e-30))")
        println("    Uh: ||Fm1G||=$(norm(Fm1G))  rel(A,R)=$(norm(Fm1G .- Fm1GR)/(norm(Fm1G)+1e-30))")
        println("    ||ρ Fh - Fm2A/ρ||=$(norm(ρ .* Fh .- Fm2A ./ ρ))  (should → ||Fm1A||=$(norm(Fm1A)))")
    end
end

# short straight (unit)
polyL = BEM.Equispaced(1)
compare_laurent("short linear L=1", polyL, [Point2D(0.0, 0.0), Point2D(1.0, 0.0)], 0.0; hR=1e-3)

# long straight (P2 scale)
compare_laurent("long linear L=2000", polyL, [Point2D(0.0, 0.0), Point2D(2000.0, 0.0)], 0.0; hR=1e-3)
compare_laurent("long linear L=2000 h=1e-6", polyL, [Point2D(0.0, 0.0), Point2D(2000.0, 0.0)], 0.0; hR=1e-6)

# quadratic Legendre on a long element, collocation at first GL node
polyQ = BEM.Legendre(2)
ξq, _ = BEM.discontinuous_nodes_weights(2)
Xgeo = [Point2D(0.0, 0.0), Point2D(1000.0, 0.0), Point2D(2000.0, 0.0)]
Ngeo, _ = BEM.shapefun(BEM.Equispaced(2), ξq)
Xcol = Ngeo * Xgeo
compare_laurent("P2-like disc. quad L=2000", polyQ, collect(Xcol), ξq[1]; hR=1e-3)
compare_laurent("P2-like disc. quad L=2000 h=1e-6", polyQ, collect(Xcol), ξq[1]; hR=1e-6)

# curved quarter-circle quadratic (geometric)
θ = [0.0, π/20, π/10]
Xarc = [Point2D(600*cos(t), 600*sin(t)) for t in θ]
Ngeo2, _ = BEM.shapefun(BEM.Equispaced(2), ξq)
Xarc_col = Ngeo2 * Xarc
compare_laurent("curved R=600 Δθ=18°", polyQ, collect(Xarc_col), ξq[1]; hR=1e-3)
compare_laurent("curved R=600 Δθ=18° h=1e-6", polyQ, collect(Xarc_col), ξq[1]; hR=1e-6)

# --- mesh-level: nf vs n_geom, H' row-sum ---
function _rect_mesh(; Lx, Ly, nlong, nshort, nome)
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(nome)
    lc = min(Lx, Ly) / 2
    p1 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, lc)
    p2 = gmsh.model.geo.addPoint(Lx, 0.0, 0.0, lc)
    p3 = gmsh.model.geo.addPoint(Lx, Ly, 0.0, lc)
    p4 = gmsh.model.geo.addPoint(0.0, Ly, 0.0, lc)
    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p4)
    l4 = gmsh.model.geo.addLine(p4, p1)
    cl = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    s1 = gmsh.model.geo.addPlaneSurface([cl])
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.setTransfiniteCurve(l1, nlong)
    gmsh.model.mesh.setTransfiniteCurve(l3, nlong)
    gmsh.model.mesh.setTransfiniteCurve(l2, nshort)
    gmsh.model.mesh.setTransfiniteCurve(l4, nshort)
    gmsh.model.mesh.setTransfiniteSurface(s1)
    gmsh.model.mesh.setRecombine(2, s1)
    gmsh.model.addPhysicalGroup(1, [l1, l2, l3, l4], -1, "1;0;1;0")
    gmsh.model.addPhysicalGroup(2, [s1], -1, "Domain")
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(2)
    out = datadir("elastico", nome * ".msh")
    mkpath(dirname(out))
    gmsh.write(out)
    gmsh.finalize()
    return out
end

function nf_vs_geom(dad)
    poly = dad.element_type
    maxdot = -Inf; mindot = Inf; nflip = 0
    for el in dad.elements
        x = dad.Nodes[el.index]
        for (k, i) in enumerate(el.index)
            a = poly.nodes[k]
            g = BEM._geom_1d(poly, x, a)
            g === nothing && continue
            _, _, _, n = g
            nf = dad.Normal[i]
            d = dot(n, nf)
            maxdot = max(maxdot, d); mindot = min(mindot, d)
            d < 0 && (nflip += 1)
        end
    end
    return (; maxdot, mindot, nflip, n=dad.n)
end

function rowsum_H(H, n)
    # rigid (1,0) and (0,1)
    e1 = zeros(2n); e2 = zeros(2n)
    for i in 1:n
        e1[2i-1] = 1; e2[2i] = 1
    end
    r1 = H * e1; r2 = H * e2
    return norm(r1), norm(r2), maximum(abs, r1), maximum(abs, r2)
end

println("\n======== P1 mesh nf vs geom / row-sum ========")
msh = _rect_mesh(; Lx=500.0, Ly=200.0, nlong=3, nshort=2, nome="diag_p1")
dad = format2d(msh, props; tipo=2, pontointerno=false)
println("  nf·n_geom: ", nf_vs_geom(dad))
H, G = H_G_hyper(dad; npg=24, threaded=false)
println("  HBIE npg=24 row-sum |H*e1|,|H*e2|,max: ", rowsum_H(H, dad.n))
H, G = H_G_full_direct(dad; npg=16, threaded=false)
println("  CBIE npg=16 row-sum (after rigid corr) |H*e1|,|H*e2|: ", rowsum_H(H, dad.n))

println("\n======== P3 quarter annulus ========")
gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)
gmsh.model.add("diag_p3")
lc = 80.0
c = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, lc)
p1 = gmsh.model.geo.addPoint(300.0, 0.0, 0.0, lc)
p2 = gmsh.model.geo.addPoint(600.0, 0.0, 0.0, lc)
p3 = gmsh.model.geo.addPoint(0.0, 600.0, 0.0, lc)
p4 = gmsh.model.geo.addPoint(0.0, 300.0, 0.0, lc)
bottom = gmsh.model.geo.addLine(p1, p2)
outer = gmsh.model.geo.addCircleArc(p2, c, p3)
top = gmsh.model.geo.addLine(p3, p4)
inner = gmsh.model.geo.addCircleArc(p4, c, p1)
cl = gmsh.model.geo.addCurveLoop([bottom, outer, top, inner])
s1 = gmsh.model.geo.addPlaneSurface([cl])
gmsh.model.geo.synchronize()
gmsh.model.mesh.setTransfiniteCurve(bottom, 3)
gmsh.model.mesh.setTransfiniteCurve(top, 3)
gmsh.model.mesh.setTransfiniteCurve(outer, 11)
gmsh.model.mesh.setTransfiniteCurve(inner, 11)
gmsh.model.addPhysicalGroup(1, [bottom], -1, "1;1000;1;0")
gmsh.model.addPhysicalGroup(1, [top], -1, "0;0;0;0")
gmsh.model.addPhysicalGroup(1, [inner, outer], -1, "1;0;1;0")
gmsh.model.addPhysicalGroup(2, [s1], -1, "Domain")
gmsh.model.mesh.generate(2)
gmsh.model.mesh.setOrder(2)
outp3 = datadir("elastico", "diag_p3.msh")
mkpath(dirname(outp3))
gmsh.write(outp3)
gmsh.finalize()
dad3 = format2d(outp3, props; tipo=2, pontointerno=false)
println("  nodes=$(dad3.n) elems=$(length(dad3.elements))")
println("  nf·n_geom: ", nf_vs_geom(dad3))
# sample one curved element Laurent vs kernel with dad.Normal
el = dad3.elements[findfirst(e -> e.Length > 80, dad3.elements)]
x = dad3.Nodes[el.index]
poly = dad3.element_type
a = poly.nodes[1]
g = BEM._geom_1d(poly, x, a)
_, J, _, ngeom = g
i0 = el.index[1]
nf = dad3.Normal[i0]
println("  sample el L=$(el.Length) J=$J ngeom=$ngeom nf=$nf dot=$(dot(ngeom,nf))")
fhyp = _fhyp(props, poly, x, dad3.Nodes[i0], nf)
ρ = 1e-8; s = 1.0
Fm2A, Fm1A, _ = laurent_coefficients(props, poly, x, a, s, :H, -2)
_, Fh = fhyp(a + s * ρ)
println("  with nf=dad.Normal: rel(Fm2A,ρ²Fh)=$(norm(Fm2A .- ρ^2 .* Fh)/(norm(Fm2A)+1e-30))")
fhyp2 = _fhyp(props, poly, x, dad3.Nodes[i0], ngeom)
_, Fh2 = fhyp2(a + s * ρ)
println("  with nf=n_geom:     rel(Fm2A,ρ²Fh)=$(norm(Fm2A .- ρ^2 .* Fh2)/(norm(Fm2A)+1e-30))")

for npg in (24, 50)
    H, G = H_G_hyper(dad3; npg=npg, threaded=false)
    println("  HBIE npg=$npg row-sum |H*e1|,|H*e2|,max: ", rowsum_H(H, dad3.n))
end
H, G = H_G_full_direct(dad3; npg=16, threaded=false)
println("  CBIE npg=16 row-sum |H*e1|,|H*e2|: ", rowsum_H(H, dad3.n))
println("done")
