# Almost-isotropic Lekhnitskii vs Kelvin on P3; HBIE residual.
using DrWatson
@quickactivate :BEM
using LinearAlgebra, StaticArrays
include(datadir("Laplace", "Laplace_dad.jl"))

E, ν = 124.04e3, 0.334
G = E / (2 * (1 + ν))
kel = Elasticity(E, ν, 1.0; plane_stress=true)
lek = AnisotropicElasticity(lekhnitskii_engineering(E, E, G, ν))

r = Point2D(0.3, 0.4); n = Point2D(0.0, 1.0); nf = Point2D(1.0, 0.0)
Uk, Tk = fundamental(kel, r, n)
Ul, Tl = fundamental(lek, r, zero(r), n)
Ul = BEM._to_smat(Ul); Tl = BEM._to_smat(Tl)
Uk = BEM._to_smat(Uk); Tk = BEM._to_smat(Tk)
println("CBIE kernel  ||U_lek-U_kel||/||U_kel||=$(norm(Ul-Uk)/norm(Uk))  ||T||=$(norm(Tl-Tk)/norm(Tk))")
hk = fundamental_hyper(kel, r, n, nf)
hl = fundamental_hyper(lek, r, zero(r), n, nf)
Uh_k, Th_k = BEM._to_smat(hk.U), BEM._to_smat(hk.T)
Uh_l, Th_l = BEM._to_smat(hl.U), BEM._to_smat(hl.T)
println("HBIE kernel  ||Uh_lek-Uh_kel||/||=$(norm(Uh_l-Uh_k)/norm(Uh_k))  ||Th||=$(norm(Th_l-Th_k)/norm(Th_k))")
println("  Uh kel=\n$Uh_k\n  Uh lek=\n$Uh_l")
println("  Th kel=\n$Th_k\n  Th lek=\n$Th_l")

msh = datadir("elastico", "cordeiro_p3.msh")
function run(lab, props)
    dad = format2d(msh, props; tipo=2, pontointerno=false)
    assemble!(dad; npg=16, threaded=false); solve(dad)
    uc = copy(dad.u)
    H_G_hyper(dad; npg=16, threaded=false); solve(dad)
    uh = dad.u
    println("$lab  CBIE max|u|=$(maximum(abs,uc))  HBIE max|u|=$(maximum(abs,uh))  rel=$(norm(uh.-uc)/(norm(uc)+1e-30))")
end
run("Kelvin          ", kel)
run("Lekhnitskii iso ", lek)
run("Lekhnitskii MAT1", AnisotropicElasticity(lekhnitskii_engineering(
    124.04e3, 10.09e3, 6.03e3, 0.334; η12_1=1.255, η12_2=-0.031)))
println("done")
