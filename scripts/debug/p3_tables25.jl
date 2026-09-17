# Scan all on-element DG/DH sums vs Cordeiro Tables 2–5.
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf, StaticArrays
include(datadir("Laplace", "Laplace_dad.jl"))

MAT1 = (E1=124.04e3, E2=10.09e3, G12=6.03e3, ν12=0.334, η12_1=1.255, η12_2=-0.031)
props = AnisotropicElasticity(lekhnitskii_engineering(
    MAT1.E1, MAT1.E2, MAT1.G12, MAT1.ν12; η12_1=MAT1.η12_1, η12_2=MAT1.η12_2))

msh = datadir("elastico", "cordeiro_p3_repro.msh")
dad = format2d(msh, props; tipo=2, pontointerno=false)
BEM._init_quadrature!(dad, 50)
dim = 2

function mats(dad, el, i; hyper)
    nf = dad.Normal[i]
    pf = dad.Nodes[i]
    xj = dad.Nodes[el.index]
    hloc = zeros(dim, dim * length(el.index))
    gloc = zeros(dim, dim * length(el.index))
    f = hyper ? ((d, r, nrm) -> fundamental_hyper(d, r, nrm, nf)) : fundamental
    orders = hyper ? (-1, -2) : nothing
    BEM.integrate_element(dad, el, xj, pf, hloc, gloc, f; orders=orders, source=i)
    return gloc, hloc
end

targets = Dict(
    :DGs => -0.00348,
    :DHs => 0.20936,
    :DGh => -0.06112,
    :DHh => -381.81087,
)

println("n_elem=$(length(dad.elements)) n=$(dad.n)")
println("searching sums / 1-norms / (1,1)-sums closest to paper Tables 2–5")

best = Dict(k => (δ=Inf, ie=0, i=0, val=0.0, how="") for k in keys(targets))

function consider!(key, val, ie, i, how)
    δ = abs(val - targets[key])
    if δ < best[key].δ
        best[key] = (δ=δ, ie=ie, i=i, val=val, how=how)
    end
end

for (ie, el) in enumerate(dad.elements)
    for i in el.index
        g, h = mats(dad, el, i; hyper=false)
        consider!(:DGs, sum(g), ie, i, "sum")
        consider!(:DGs, sum(abs, g), ie, i, "sumabs")
        consider!(:DGs, sum(g[1, 1:2:end]), ie, i, "row1odd")
        consider!(:DGs, tr(g[:, 1:2]), ie, i, "tr_node1")
        consider!(:DHs, sum(h), ie, i, "sum")
        consider!(:DHs, sum(abs, h), ie, i, "sumabs")
        consider!(:DHs, sum(h[1, 1:2:end]), ie, i, "row1odd")
        consider!(:DHs, h[1, 1], ie, i, "h11")
        g, h = mats(dad, el, i; hyper=true)
        consider!(:DGh, sum(g), ie, i, "sum")
        consider!(:DGh, sum(abs, g), ie, i, "sumabs")
        consider!(:DGh, g[1, 1], ie, i, "g11")
        consider!(:DHh, sum(h), ie, i, "sum")
        consider!(:DHh, sum(abs, h), ie, i, "sumabs")
        consider!(:DHh, h[1, 1], ie, i, "h11")
        consider!(:DHh, sum(h[1, :]), ie, i, "row1")
        consider!(:DHh, sum(h[:, 1]), ie, i, "col1")
        consider!(:DHh, tr(h[:, 1:2]), ie, i, "tr_n1")
        # mean of nodal traces
        s = 0.0
        for j in 1:3
            s += h[1, 2j-1] + h[2, 2j]
        end
        consider!(:DHh, s, ie, i, "sum_nodal_tr")
    end
end

for k in (:DGs, :DHs, :DGh, :DHh)
    b = best[k]
    @printf("%s  paper=%12.5f  best=%12.5f  δ=%.3e  el=%d src=%d how=%s  pos=(%.1f,%.1f)\n",
        k, targets[k], b.val, b.δ, b.ie, b.i, b.how, dad.Nodes[b.i][1], dad.Nodes[b.i][2])
end

println("\n--- full HBIE DH/DG for outer-bottom elem 3, each src ---")
el = dad.elements[3]
for i in el.index
    g, h = mats(dad, el, i; hyper=true)
    println("src $i  n=$(dad.Normal[i])  pf=$(dad.Nodes[i])")
    println("  DG =\n", g)
    println("  DH =\n", h)
    println("  sumDG=$(sum(g)) sumDH=$(sum(h))  ||DH||=$(norm(h))  tr_n1=$(tr(h[:,1:2]))")
end

println("\n--- full CBIE for elem 3 ---")
for i in el.index
    g, h = mats(dad, el, i; hyper=false)
    println("src $i  sumDG=$(sum(g)) sumDH=$(sum(h))")
    println("  DH =\n", h)
end
