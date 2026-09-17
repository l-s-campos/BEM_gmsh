# Compare P3 HBIE for quadratic msh written as order-2 vs linear-then-setOrder.
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf
include(datadir("Laplace", "Laplace_dad.jl"))
MAT1 = (E1=124.04e3, E2=10.09e3, G12=6.03e3, ν12=0.334, η12_1=1.255, η12_2=-0.031)
props = AnisotropicElasticity(lekhnitskii_engineering(
    MAT1.E1, MAT1.E2, MAT1.G12, MAT1.ν12; η12_1=MAT1.η12_1, η12_2=MAT1.η12_2))

function build(; order_before_write)
    gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("p3m"); lc=80.0
    c=gmsh.model.geo.addPoint(0,0,0,lc)
    p1=gmsh.model.geo.addPoint(300,0,0,lc); p2=gmsh.model.geo.addPoint(600,0,0,lc)
    p3=gmsh.model.geo.addPoint(0,600,0,lc); p4=gmsh.model.geo.addPoint(0,300,0,lc)
    b=gmsh.model.geo.addLine(p1,p2); o=gmsh.model.geo.addCircleArc(p2,c,p3)
    t=gmsh.model.geo.addLine(p3,p4); inn=gmsh.model.geo.addCircleArc(p4,c,p1)
    cl=gmsh.model.geo.addCurveLoop([b,o,t,inn]); s1=gmsh.model.geo.addPlaneSurface([cl])
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.setTransfiniteCurve(b,3); gmsh.model.mesh.setTransfiniteCurve(t,3)
    gmsh.model.mesh.setTransfiniteCurve(o,11); gmsh.model.mesh.setTransfiniteCurve(inn,11)
    gmsh.model.addPhysicalGroup(1,[b],-1,"1;1000;1;0")
    gmsh.model.addPhysicalGroup(1,[t],-1,"0;0;0;0")
    gmsh.model.addPhysicalGroup(1,[inn,o],-1,"1;0;1;0")
    gmsh.model.addPhysicalGroup(2,[s1],-1,"Domain")
    gmsh.model.mesh.generate(2)
    order_before_write && gmsh.model.mesh.setOrder(2)
    tag = order_before_write ? "q2" : "lin"
    out=datadir("elastico","p3_cmp_$(tag).msh"); mkpath(dirname(out)); gmsh.write(out)
    gmsh.finalize(); return out
end

function run(msh, label)
    dad = format2d(msh, props; tipo=2, pontointerno=false)
    # geometry check: first outer element mid radius
    el = dad.elements[3]
    rs = [hypot(dad.Nodes[i][1], dad.Nodes[i][2]) for i in el.index]
    assemble!(dad; npg=16, threaded=false); solve(dad)
    uc = copy(dad.u)
    H_G_hyper(dad; npg=50, threaded=false); solve(dad)
    rel = norm(dad.u .- uc)/(norm(uc)+1e-30)
    @printf("%-16s n=%d  r_col=%s  CBIE=%8.3f  HBIE=%8.3f  rel=%.3e  cond=%.2e\n",
        label, dad.n, string(round.(rs; digits=2)),
        maximum(abs, uc), maximum(abs, dad.u), rel, cond(Matrix(dad.A)))
    return dad, uc
end

println("tx=P  tipo=2")
run(build(; order_before_write=true), "write order-2")
run(build(; order_before_write=false), "write linear")
println("done")
