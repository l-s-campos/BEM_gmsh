# Compare dad.Normal vs geometry tan2normal; rigid rotation residual.
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf, StaticArrays
include(datadir("Laplace", "Laplace_dad.jl"))

MAT1 = (E1=124.04e3, E2=10.09e3, G12=6.03e3, ν12=0.334, η12_1=1.255, η12_2=-0.031)
props = AnisotropicElasticity(lekhnitskii_engineering(
    MAT1.E1, MAT1.E2, MAT1.G12, MAT1.ν12; η12_1=MAT1.η12_1, η12_2=MAT1.η12_2))
msh = datadir("elastico", "p3_exact.msh")
isfile(msh) || (msh = datadir("elastico", "cordeiro_p3.msh"))
dad = format2d(msh, props; tipo=2, pontointerno=false)
poly = dad.element_type

function scan_normals(dad, poly)
nmax = 0.0
nn = 0
for el in dad.elements
    x = dad.Nodes[el.index]
    for (k, i) in enumerate(el.index)
        a = poly.nodes[k]
        g = BEM._geom_1d(poly, x, a)
        g === nothing && continue
        _, _, _, n = g
        d = norm(n - dad.Normal[i])
        nmax = max(nmax, d)
        if d > 1e-8
            nn += 1
            @printf("  i=%2d el-node=%d  geo n=(%7.4f,%7.4f)  dad n=(%7.4f,%7.4f)  |Δ|=%.2e  pf=(%.1f,%.1f)\n",
                i, k, n[1], n[2], dad.Normal[i][1], dad.Normal[i][2], d,
                dad.Nodes[i][1], dad.Nodes[i][2])
        end
    end
end
println("max |n_geo - n_dad|=$nmax  mismatches=$nn / $(dad.n)")
return nmax, nn
end
scan_normals(dad, poly)

function rigid_and_rowsum(dad, msh, props)
assemble!(dad; npg=16, threaded=false); solve(dad)
uc, tc = copy(dad.u), copy(dad.traction)
H_G_hyper(dad; npg=50, threaded=false)
H, G = dad.H, dad.G
n = dad.n
e1 = zeros(2n); e2 = zeros(2n); rot = zeros(2n)
for i in 1:n
    e1[2i-1] = 1
    e2[2i] = 1
    rot[2i-1] = -dad.Nodes[i][2]
    rot[2i] = dad.Nodes[i][1]
end
println("||H e1||=$(norm(H*e1))  ||H e2||=$(norm(H*e2))  ||H rot||=$(norm(H*rot))  ||rot||=$(norm(rot))")
println("||H rot||/||rot||=$(norm(H*rot)/norm(rot))")
res = H * uc - G * tc
println("||H uC - G tC|| / ||G tC|| = $(norm(res)/(norm(G*tc)+1e-30))")
no_rowsum(msh, props, uc, tc, e1, e2, rot)
end

function no_rowsum(msh, props, uc, tc, e1, e2, rot)
dad2 = format2d(msh, props; tipo=2, pontointerno=false)
BEM._init_quadrature!(dad2, 50)
n = dad2.n; dim=2
Hh = zeros(2n, 2n); Gh = zeros(2n, 2n)
for i in 1:n
    pf = dad2.Nodes[i]; nf = dad2.Normal[i]
    ii = BEM.expand(i, dim)
    ff = (d, r, nrm) -> fundamental_hyper(d, r, nrm, nf)
    for el in dad2.elements
        xj = dad2.Nodes[el.index]
        jj = BEM.expand(el.index, dim)
        hloc = zeros(dim, length(jj)); gloc = zeros(dim, length(jj))
        BEM.integrate_element(dad2, el, xj, pf, hloc, gloc, ff; orders=(-1,-2), source=i)
        Hh[ii, jj] .+= hloc
        Gh[ii, jj] .+= gloc
    end
end
for i in 1:n
    Gh[2i-1, 2i-1] -= 0.5
    Gh[2i, 2i] -= 0.5
end
println("\nwithout row-sum: ||H e1||=$(norm(Hh*e1))  ||H e2||=$(norm(Hh*e2))  ||H rot||=$(norm(Hh*rot))")
res2 = Hh * uc - Gh * tc
println("  ||H uC - G tC|| / ||G tC|| = $(norm(res2)/(norm(Gh*tc)+1e-30))")
set_cache!(dad2; H=Hh, G=Gh, H_hyper=Hh, G_hyper=Gh)
solve(dad2)
println("  HBIE no-rowsum max|u|=$(maximum(abs, dad2.u)) mm  rel vs CBIE=$(norm(dad2.u.-uc)/(norm(uc)+1e-30))")
end
rigid_and_rowsum(dad, msh, props)
println("done")
