# Top-5 on-element sums vs Tables 2–5; identify the highlighted element.
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf, StaticArrays
include(datadir("Laplace", "Laplace_dad.jl"))

MAT1 = (E1=124.04e3, E2=10.09e3, G12=6.03e3, ν12=0.334, η12_1=1.255, η12_2=-0.031)
props = AnisotropicElasticity(lekhnitskii_engineering(
    MAT1.E1, MAT1.E2, MAT1.G12, MAT1.ν12; η12_1=MAT1.η12_1, η12_2=MAT1.η12_2))
dad = format2d(datadir("elastico", "cordeiro_p3_repro.msh"), props; tipo=2, pontointerno=false)
BEM._init_quadrature!(dad, 50)
dim = 2

function mats(dad, el, i; hyper)
    nf = dad.Normal[i]; pf = dad.Nodes[i]; xj = dad.Nodes[el.index]
    hloc = zeros(dim, dim * length(el.index))
    gloc = zeros(dim, dim * length(el.index))
    f = hyper ? ((d, r, nrm) -> fundamental_hyper(d, r, nrm, nf)) : fundamental
    orders = hyper ? (-1, -2) : nothing
    BEM.integrate_element(dad, el, xj, pf, hloc, gloc, f; orders=orders, source=i)
    return gloc, hloc
end

function top5(vals, paper, name)
    sort!(vals; by=t -> abs(t.s - paper))
    println("\n$name  paper=$paper")
    for k in 1:min(8, length(vals))
        t = vals[k]
        r = hypot(t.p[1], t.p[2])
        @printf("  el=%2d src=%2d  sum=%12.5f  δ=%9.2e  r=%6.1f  pos=(%6.1f,%6.1f) n=(%6.3f,%6.3f)\n",
            t.ie, t.i, t.s, abs(t.s - paper), r, t.p[1], t.p[2], t.n[1], t.n[2])
    end
end

cbieG = []; cbieH = []; hbieG = []; hbieH = []
for (ie, el) in enumerate(dad.elements)
    for i in el.index
        g, h = mats(dad, el, i; hyper=false)
        push!(cbieG, (ie=ie, i=i, s=sum(g), p=dad.Nodes[i], n=dad.Normal[i]))
        push!(cbieH, (ie=ie, i=i, s=sum(h), p=dad.Nodes[i], n=dad.Normal[i]))
        g, h = mats(dad, el, i; hyper=true)
        push!(hbieG, (ie=ie, i=i, s=sum(g), p=dad.Nodes[i], n=dad.Normal[i]))
        push!(hbieH, (ie=ie, i=i, s=sum(h), p=dad.Nodes[i], n=dad.Normal[i]))
    end
end
top5(cbieG, -0.00348, "Table2 DGsing")
top5(cbieH,  0.20936, "Table3 DHsing")
top5(hbieG, -0.06112, "Table4 DGhyp")
top5(hbieH, -381.81087, "Table5 DHhyp")

println("\n--- element map (midpoint) ---")
for (ie, el) in enumerate(dad.elements)
    p = dad.Nodes[el.index[2]]
    @printf("  el=%2d  mid=(%7.1f,%7.1f) r=%6.1f\n", ie, p[1], p[2], hypot(p[1], p[2]))
end

# dump the Table2-matching element in full
println("\n--- el=21 CBIE/HBIE full ---")
el = dad.elements[21]
for i in el.index
    g, h = mats(dad, el, i; hyper=false)
    gh, hh = mats(dad, el, i; hyper=true)
    @printf("src=%d pf=(%.1f,%.1f)  CBIEsumG=%.6f CBIEsumH=%.6f  HBIEsumG=%.6f HBIEsumH=%.6f\n",
        i, dad.Nodes[i][1], dad.Nodes[i][2], sum(g), sum(h), sum(gh), sum(hh))
end
