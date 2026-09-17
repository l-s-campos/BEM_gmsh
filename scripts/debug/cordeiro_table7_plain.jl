# Table 7, plain Gauss off-element (no sinh).
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
    msh = datadir("elastico", "cordeiro_table67.msh")
    dad = format2d(msh, props; tipo=2, pontointerno=false)
    println("plain Gauss  n=$(dad.n)")
    println("npg   mean xx        |max|          closest-to-paper   paper")
    for (npg, paper) in ((20, -0.31124), (25, 0.01284), (30, 5.039e-4),
                         (40, 2.443e-5), (50, 1.356e-7))
        H = assemble_H!(dad; npg=npg)
        n = dad.n
        fxx = [free_term(H, i, 1) for i in 1:n]
        k = argmin(i -> abs(fxx[i] - paper), 1:n)
        p = dad.Nodes[k]
        @printf("%3d  %12.5e  %12.5e  src=%2d xx=%12.5e (r=%.1f)  paper=%g\n",
            npg, sum(fxx)/n, maximum(abs, fxx), k, fxx[k], hypot(p[1], p[2]), paper)
    end
    println("done")
end
run()
