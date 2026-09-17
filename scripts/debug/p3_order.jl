# P3 CBIE/HBIE vs element order (same 24 elements).
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf
include(datadir("Laplace", "Laplace_dad.jl"))
MAT1 = (E1=124.04e3, E2=10.09e3, G12=6.03e3, ν12=0.334, η12_1=1.255, η12_2=-0.031)
props = AnisotropicElasticity(lekhnitskii_engineering(
    MAT1.E1, MAT1.E2, MAT1.G12, MAT1.ν12; η12_1=MAT1.η12_1, η12_2=MAT1.η12_2))

function mesh()
    gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("p3o"); lc=80.0
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
    out=datadir("elastico","p3_order.msh"); mkpath(dirname(out)); gmsh.write(out)
    gmsh.finalize(); return out
end
msh = mesh()
println("tipo  n     CBIE mm     HBIE mm     rel        cond")
for tipo in (2, 3, 4)
    dad = format2d(msh, props; tipo=tipo, pontointerno=false)
    assemble!(dad; npg=16, threaded=false); solve(dad)
    uc = copy(dad.u)
    H_G_hyper(dad; npg=50, threaded=false); solve(dad)
    rel = norm(dad.u .- uc) / (norm(uc)+1e-30)
    @printf("  %d  %4d  %10.3f  %10.3f  %8.3e  %8.2e\n",
        tipo, dad.n, maximum(abs, uc), maximum(abs, dad.u), rel, cond(Matrix(dad.A)))
end
println("done")
