# Identify the Table 7 source (plain Gauss, npg=50 → 1.356e-7) and sweep it.
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf
include(datadir("Laplace", "Laplace_dad.jl"))

MAT1 = (E1=124.04e3, E2=10.09e3, G12=6.03e3, ν12=0.334, η12_1=1.255, η12_2=-0.031)
props = AnisotropicElasticity(lekhnitskii_engineering(
    MAT1.E1, MAT1.E2, MAT1.G12, MAT1.ν12; η12_1=MAT1.η12_1, η12_2=MAT1.η12_2))

function free_term(H, i, α)
    n = size(H, 1) ÷ 2
    row = 2 * (i - 1) + α
    s = 0.0
    @inbounds for j in 1:n
        s += H[row, 2 * (j - 1) + α]
    end
    return s
end

function assemble_H!(dad; npg)
    set_cache!(dad; nearfield=:plain)
    BEM._init_quadrature!(dad, npg)
    dim = 2; n = dad.n
    H = zeros(dim * n, dim * n)
    G = zeros(dim * n, dim * n)
    for i in 1:n
        pf = dad.Nodes[i]; nf = dad.Normal[i]
        ii = BEM.expand(i, dim)
        f = (d, r, nrm) -> fundamental_hyper(d, r, nrm, nf)
        for el in dad.elements
            xj = dad.Nodes[el.index]
            jj = BEM.expand(el.index, dim)
            hloc = zeros(dim, length(jj)); gloc = zeros(dim, length(jj))
            BEM.integrate_element(dad, el, xj, pf, hloc, gloc, f;
                orders=(-1, -2), source=i)
            H[ii, jj] .+= hloc
        end
    end
    return H
end

function run()
    dad = format2d(datadir("elastico", "cordeiro_table67.msh"), props;
        tipo=2, pontointerno=false)
    H50 = assemble_H!(dad; npg=50)
    println("npg=50  sources with |ft_xx| < 1e-5:")
    hits = Int[]
    for i in 1:dad.n
        fx = free_term(H50, i, 1)
        fy = free_term(H50, i, 2)
        if abs(fx) < 1e-5 || abs(fy) < 1e-5
            p = dad.Nodes[i]
            @printf("  src=%2d  (%.1f, %.1f) r=%.1f  xx=%12.5e  yy=%12.5e\n",
                i, p[1], p[2], hypot(p[1], p[2]), fx, fy)
            abs(fx - 1.3562e-7) < 1e-8 && push!(hits, i)
        end
    end
    println("exact Table 7 hits: $hits")
    src = isempty(hits) ? 60 : hits[1]
    println("\nTable 7 at src=$src  $(dad.Nodes[src])")
    println("npg     xx            yy            paper")
    for (npg, paper) in ((20, -0.31124), (25, 0.01284), (30, 5.03902e-4),
                         (35, -1.80607e-4), (40, 2.44274e-5),
                         (45, -2.45174e-6), (50, 1.35620e-7))
        H = npg == 50 ? H50 : assemble_H!(dad; npg=npg)
        @printf("%3d  %12.5e  %12.5e  %g\n",
            npg, free_term(H, src, 1), free_term(H, src, 2), paper)
    end
    println("done")
end
run()
