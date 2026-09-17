# Off-element |z| vs |r|, and HBIE residual on the CBIE solution (aniso P3).
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf
include(datadir("Laplace", "Laplace_dad.jl"))

MAT1 = (E1=124.04e3, E2=10.09e3, G12=6.03e3, ν12=0.334, η12_1=1.255, η12_2=-0.031)
props = AnisotropicElasticity(lekhnitskii_engineering(
    MAT1.E1, MAT1.E2, MAT1.G12, MAT1.ν12; η12_1=MAT1.η12_1, η12_2=MAT1.η12_2))
μ = props.params.mi

msh = datadir("elastico", "sst_inv_p3.msh")
if !isfile(msh)
    msh = datadir("elastico", "cordeiro_p3.msh")
end
dad = format2d(msh, props; tipo=2, pontointerno=false)
poly = dad.element_type
qsi, _ = BEM.gausslegendre(16)

println("nodes=$(dad.n)  μ=$(μ)")
function scan_z(dad, μ, poly, qsi)
min_r = Inf; min_z1 = Inf; min_z2 = Inf
n_near_z = 0
worst = (d=Inf, i=0, el=0, R=0.0, z1=0.0, z2=0.0)
for i in 1:dad.n
    pf = dad.Nodes[i]
    for (ie, el) in enumerate(dad.elements)
        i in el.index && continue
        x = dad.Nodes[el.index]
        for ξ in qsi
            N, dN = BEM.shapefun(poly, ξ)
            pg = (N * x)[1]
            r = pg - pf
            R = norm(r)
            z1 = abs(r[1] + μ[1] * r[2])
            z2 = abs(r[1] + μ[2] * r[2])
            min_r = min(min_r, R)
            min_z1 = min(min_z1, z1)
            min_z2 = min(min_z2, z2)
            mz = min(z1, z2)
            if mz < 5.0
                n_near_z += 1
            end
            if mz < worst.d
                worst = (d=mz, i=i, el=ie, R=R, z1=z1, z2=z2)
            end
        end
    end
end
return (; min_r, min_z1, min_z2, n_near_z, worst)
end
st = scan_z(dad, μ, poly, qsi)
println("off-element min |r|=$(st.min_r)  min |z1|=$(st.min_z1)  min |z2|=$(st.min_z2)")
println("  samples with min(|z1|,|z2|)<5 mm: $(st.n_near_z)")
w = st.worst
println("  worst: collocation $(w.i)  elem $(w.el)  |z|=$(w.d)  |r|=$(w.R)")

assemble!(dad; npg=16, threaded=false); solve(dad)
uc = copy(dad.u)
tc = copy(dad.traction)
Hc, Gc = copy(dad.H), copy(dad.G)
println("\nCBIE  max|u|=$(maximum(abs, uc))  ||Hc uc - Gc tc||=$(norm(Hc*uc-Gc*tc)/(norm(Gc*tc)+1e-30))")

H_G_hyper(dad; npg=16, threaded=false)
Hh, Gh = copy(dad.H), copy(dad.G)
res = Hh * uc - Gh * tc
println("HBIE op on CBIE sol: ||H' uc - G' tc|| / ||G' tc|| = $(norm(res)/(norm(Gh*tc)+1e-30))")
println("  max |res|=$(maximum(abs, res))  ||H'||=$(norm(Hh))  ||G'||=$(norm(Gh))  ||H_c||=$(norm(Hc))  ||G_c||=$(norm(Gc))")

# split residual: Dirichlet vs Neumann rows
nd = 0; nn = 0; ed = 0.0; en = 0.0
for i in 1:2*dad.n
    if dad.BC[i] == 0
        nd += 1; ed = max(ed, abs(res[i]))
    else
        nn += 1; en = max(en, abs(res[i]))
    end
end
println("  max |res| on Dirichlet dofs (n=$nd): $ed")
println("  max |res| on Neumann dofs    (n=$nn): $en")

# on-element-only H' (zero off-element) row-sum
println("\ndone")
