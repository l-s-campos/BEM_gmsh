# Cordeiro & Leonel 2020 Tables 6–7: free term from row-sum of H
# *before* rigid-body replacement.
#
# Table 6 (CBIE): row-sum → −cij = −1/2
# Table 7 (HBIE): row-sum → cij = 0
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf
include(datadir("Laplace", "Laplace_dad.jl"))

MAT1 = (E1=124.04e3, E2=10.09e3, G12=6.03e3, ν12=0.334, η12_1=1.255, η12_2=-0.031)
props = AnisotropicElasticity(lekhnitskii_engineering(
    MAT1.E1, MAT1.E2, MAT1.G12, MAT1.ν12; η12_1=MAT1.η12_1, η12_2=MAT1.η12_2))

function p3mesh()
    gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("t67"); lc = 80.0
    c = gmsh.model.geo.addPoint(0, 0, 0, lc)
    p1 = gmsh.model.geo.addPoint(300, 0, 0, lc)
    p2 = gmsh.model.geo.addPoint(600, 0, 0, lc)
    p3 = gmsh.model.geo.addPoint(0, 600, 0, lc)
    p4 = gmsh.model.geo.addPoint(0, 300, 0, lc)
    b = gmsh.model.geo.addLine(p1, p2)
    o = gmsh.model.geo.addCircleArc(p2, c, p3)
    t = gmsh.model.geo.addLine(p3, p4)
    inn = gmsh.model.geo.addCircleArc(p4, c, p1)
    cl = gmsh.model.geo.addCurveLoop([b, o, t, inn])
    s1 = gmsh.model.geo.addPlaneSurface([cl])
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.setTransfiniteCurve(b, 3)
    gmsh.model.mesh.setTransfiniteCurve(t, 3)
    gmsh.model.mesh.setTransfiniteCurve(o, 11)
    gmsh.model.mesh.setTransfiniteCurve(inn, 11)
    gmsh.model.addPhysicalGroup(1, [b], -1, "1;0;1;0")
    gmsh.model.addPhysicalGroup(1, [t], -1, "1;0;1;0")
    gmsh.model.addPhysicalGroup(1, [inn, o], -1, "1;0;1;0")
    gmsh.model.addPhysicalGroup(2, [s1], -1, "Domain")
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(2)
    out = datadir("elastico", "cordeiro_table67.msh")
    mkpath(dirname(out)); gmsh.write(out); gmsh.finalize()
    return out
end

function assemble_H_integral!(dad; hyper::Bool, npg::Int)
    BEM._init_quadrature!(dad, npg)
    dim = dad.dimension
    n = dad.n
    H = zeros(dim * n, dim * n)
    G = zeros(dim * n, dim * n)
    orders = hyper ? (-1, -2) : nothing
    f = if hyper
        (i) -> ((d, r, nrm) -> fundamental_hyper(d, r, nrm, dad.Normal[i]))
    else
        (i) -> fundamental
    end
    for i in 1:n
        pf = dad.Nodes[i]
        ii = BEM.expand(i, dim)
        fi = f(i)
        for el in dad.elements
            xj = dad.Nodes[el.index]
            jj = BEM.expand(el.index, dim)
            hloc = zeros(dim, length(jj))
            gloc = zeros(dim, length(jj))
            BEM.integrate_element(dad, el, xj, pf, hloc, gloc, fi;
                orders=orders, source=i)
            H[ii, jj] .+= hloc
            G[ii, jj] .+= gloc
        end
    end
    return H, G
end

# Free term at source i, component α: sum of H-row over translations in α
function free_term(H, i, α)
    n = size(H, 1) ÷ 2
    row = 2 * (i - 1) + α
    s = 0.0
    @inbounds for j in 1:n
        s += H[row, 2 * (j - 1) + α]
    end
    return s
end

function report(H, dad, paper, label)
    n = dad.n
    fxx = [free_term(H, i, 1) for i in 1:n]
    fyy = [free_term(H, i, 2) for i in 1:n]
    # source on outer arc, away from corners (elem 7 mid ≈ 45°)
    i45 = argmin(i -> begin
        p = dad.Nodes[i]
        abs(hypot(p[1], p[2]) - 600) + abs(atan(p[2], p[1]) - π/4)
    end, 1:n)
    @printf("  %-6s  mean xx=%11.5e  mean yy=%11.5e  |max|=%11.5e  outer45 xx=%11.5e yy=%11.5e  paper=%s\n",
        label, sum(fxx)/n, sum(fyy)/n, max(maximum(abs, fxx), maximum(abs, fyy)),
        fxx[i45], fyy[i45], paper)
    return (; fxx, fyy, i45)
end

function main()
    msh = p3mesh()
    dad = format2d(msh, props; tipo=2, pontointerno=false)
    println("n=$(dad.n)  elems=$(length(dad.elements))  (Fig. 3 / P3 24 quads)")
    println("\nTable 6  CBIE  paper → −0.50   (npg=2…20)")
    println("  npg     paper")
    for (npg, paper) in ((2, -0.31211), (4, -0.46364), (6, -0.53046),
                         (8, -0.50625), (10, -0.49683), (12, -0.49888),
                         (14, -0.50029), (16, -0.50017), (18, -0.49998),
                         (20, -0.49998))
        H, _ = assemble_H_integral!(dad; hyper=false, npg=npg)
        r = report(H, dad, paper, string(npg))
    end
    println("\nTable 7  HBIE  paper → 0.00   (npg=20…50)")
    for (npg, paper) in ((20, -0.31124), (25, 0.01284), (30, 5.03902e-4),
                         (35, -1.80607e-4), (40, 2.44274e-5),
                         (45, -2.45174e-6), (50, 1.35620e-7))
        H, _ = assemble_H_integral!(dad; hyper=true, npg=npg)
        r = report(H, dad, paper, string(npg))
    end
    println("done")
end
main()
