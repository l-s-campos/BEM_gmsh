# P3 CBIE vs HBIE after z-peak sinh; also ty=-P, η=0, P1 regression.
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf
include(datadir("Laplace", "Laplace_dad.jl"))

MAT1 = (E1=124.04e3, E2=10.09e3, G12=6.03e3, ν12=0.334, η12_1=1.255, η12_2=-0.031)

function p3mesh(; load=:tx, P=1000.0)
    gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("p3z"); lc=80.0
    c=gmsh.model.geo.addPoint(0,0,0,lc)
    p1=gmsh.model.geo.addPoint(300,0,0,lc); p2=gmsh.model.geo.addPoint(600,0,0,lc)
    p3=gmsh.model.geo.addPoint(0,600,0,lc); p4=gmsh.model.geo.addPoint(0,300,0,lc)
    b=gmsh.model.geo.addLine(p1,p2); o=gmsh.model.geo.addCircleArc(p2,c,p3)
    t=gmsh.model.geo.addLine(p3,p4); inn=gmsh.model.geo.addCircleArc(p4,c,p1)
    cl=gmsh.model.geo.addCurveLoop([b,o,t,inn]); s1=gmsh.model.geo.addPlaneSurface([cl])
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.setTransfiniteCurve(b,3); gmsh.model.mesh.setTransfiniteCurve(t,3)
    gmsh.model.mesh.setTransfiniteCurve(o,11); gmsh.model.mesh.setTransfiniteCurve(inn,11)
    bc = load===:ty ? "1;0;1;$(-P)" : "1;$P;1;0"
    gmsh.model.addPhysicalGroup(1,[b],-1,bc)
    gmsh.model.addPhysicalGroup(1,[t],-1,"0;0;0;0")
    gmsh.model.addPhysicalGroup(1,[inn,o],-1,"1;0;1;0")
    gmsh.model.addPhysicalGroup(2,[s1],-1,"Domain")
    gmsh.model.mesh.generate(2); gmsh.model.mesh.setOrder(2)
    out=datadir("elastico","p3_zsinh.msh"); mkpath(dirname(out)); gmsh.write(out)
    gmsh.finalize(); return out
end

function props(; η=true)
    AnisotropicElasticity(lekhnitskii_engineering(
        MAT1.E1, MAT1.E2, MAT1.G12, MAT1.ν12;
        η12_1=η ? MAT1.η12_1 : 0.0, η12_2=η ? MAT1.η12_2 : 0.0))
end

function runpair(msh, pr; npg_h=50, near=:auto, label="")
    dad = format2d(msh, pr; tipo=2, pontointerno=false)
    set_cache!(dad; nearfield=near)
    assemble!(dad; npg=16, threaded=false); solve(dad)
    uc = copy(dad.u)
    umaxc = maximum(abs, uc)
    set_cache!(dad; nearfield=near)
    H_G_hyper(dad; npg=npg_h, threaded=false); solve(dad)
    uh = dad.u
    rel = norm(uh .- uc) / (norm(uc) + 1e-30)
    @printf("%-36s CBIE=%10.3f mm (%6.2f cm)  HBIE=%10.3f mm (%6.2f cm)  rel=%.3e  cond=%.2e\n",
        label, umaxc, umaxc/10, maximum(abs, uh), maximum(abs, uh)/10, rel, cond(Matrix(dad.A)))
    return rel
end

println("μ = ", props().params.mi)
println("\n--- P3 ---")
msh_tx = p3mesh(; load=:tx)
msh_ty = p3mesh(; load=:ty)
pr = props()
pr0 = props(; η=false)

runpair(msh_tx, pr; npg_h=16, near=:euclid, label="tx P=1GPa euclid npg=16")
runpair(msh_tx, pr; npg_h=50, near=:euclid, label="tx P=1GPa euclid npg=50")
runpair(msh_tx, pr; npg_h=16, near=:auto,   label="tx P=1GPa z-sinh npg=16")
runpair(msh_tx, pr; npg_h=50, near=:auto,   label="tx P=1GPa z-sinh npg=50")
runpair(msh_tx, pr; npg_h=50, near=:plain,  label="tx P=1GPa plain npg=50")

runpair(msh_ty, pr; npg_h=50, near=:euclid, label="ty -P euclid npg=50")
runpair(msh_ty, pr; npg_h=50, near=:auto,   label="ty -P z-sinh npg=50")
runpair(msh_ty, pr0; npg_h=50, near=:auto,  label="ty -P η=0 z-sinh npg=50")

msh_ty100 = p3mesh(; load=:ty, P=100.0)
runpair(msh_ty100, pr; npg_h=50, near=:auto, label="ty -100MPa z-sinh npg=50")

println("done")
