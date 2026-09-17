# Paper split: tangent on H (1/r²), complex-pole sinh on G (1/r).
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf
include(datadir("Laplace", "Laplace_dad.jl"))
MAT1 = (E1=124.04e3, E2=10.09e3, G12=6.03e3, ν12=0.334, η12_1=1.255, η12_2=-0.031)
props = AnisotropicElasticity(lekhnitskii_engineering(
    MAT1.E1, MAT1.E2, MAT1.G12, MAT1.ν12; η12_1=MAT1.η12_1, η12_2=MAT1.η12_2))
msh = datadir("elastico", "p3_cmp_q2.msh")
dad = format2d(msh, props; tipo=2, pontointerno=false)
assemble!(dad; npg=16, threaded=false); solve(dad)
uc = copy(dad.u)
println("CBIE max|u|=$(maximum(abs, uc))")

function hg(near, npg)
    d = format2d(msh, props; tipo=2, pontointerno=false)
    set_cache!(d; nearfield=near)
    H_G_hyper(d; npg=npg, threaded=false)
    return copy(d.H), copy(d.G)
end

Ht, Gt = hg(:tangent, 50)
Hs, Gs = hg(:csinh, 50)
Hss, Gss = hg(:sinhsinh, 50)
Hp, Gp = hg(:p3c, 50)
Htp, Gtp = hg(:tanp3c, 50)

function report(lab, H, G)
    d = format2d(msh, props; tipo=2, pontointerno=false)
    set_cache!(d; H=H, G=G, H_hyper=H, G_hyper=G)
    solve(d)
    rel = norm(d.u .- uc)/(norm(uc)+1e-30)
    @printf("%-28s max|u|=%10.3f  rel=%.3e  cond=%.2e\n",
        lab, maximum(abs, d.u), rel, cond(Matrix(d.A)))
end
report("H,G tangent", Ht, Gt)
report("H,G csinh", Hs, Gs)
report("H,G p3c", Hp, Gp)
report("H,G tanp3c", Htp, Gtp)
report("H tangent, G csinh", Ht, Gs)
report("H tangent, G sinhsinh", Ht, Gss)
report("H tanp3c, G csinh", Htp, Gs)
report("H tanp3c, G p3c", Htp, Gp)
report("H sinhsinh, G csinh", Hss, Gs)
println("done")
