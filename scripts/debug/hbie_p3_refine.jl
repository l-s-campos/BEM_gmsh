using DrWatson
@quickactivate :BEM
using LinearAlgebra
include(datadir("Laplace", "Laplace_dad.jl"))
MAT1 = (E1=124.04e3, E2=10.09e3, G12=6.03e3, ν12=0.334, η12_1=1.255, η12_2=-0.031)
props = AnisotropicElasticity(lekhnitskii_engineering(MAT1.E1, MAT1.E2, MAT1.G12, MAT1.ν12;
    η12_1=MAT1.η12_1, η12_2=MAT1.η12_2))

function mesh(narc)
    gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("p3r$narc")
    c = gmsh.model.geo.addPoint(0.0,0.0,0.0,80.0)
    p1 = gmsh.model.geo.addPoint(300.0,0.0,0.0,80.0)
    p2 = gmsh.model.geo.addPoint(600.0,0.0,0.0,80.0)
    p3 = gmsh.model.geo.addPoint(0.0,600.0,0.0,80.0)
    p4 = gmsh.model.geo.addPoint(0.0,300.0,0.0,80.0)
    bottom = gmsh.model.geo.addLine(p1,p2)
    outer = gmsh.model.geo.addCircleArc(p2,c,p3)
    top = gmsh.model.geo.addLine(p3,p4)
    inner = gmsh.model.geo.addCircleArc(p4,c,p1)
    cl = gmsh.model.geo.addCurveLoop([bottom,outer,top,inner])
    s1 = gmsh.model.geo.addPlaneSurface([cl])
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.setTransfiniteCurve(bottom,5); gmsh.model.mesh.setTransfiniteCurve(top,5)
    gmsh.model.mesh.setTransfiniteCurve(outer,narc); gmsh.model.mesh.setTransfiniteCurve(inner,narc)
    gmsh.model.addPhysicalGroup(1,[bottom],-1,"1;0;1;-1000")
    gmsh.model.addPhysicalGroup(1,[top],-1,"0;0;0;0")
    gmsh.model.addPhysicalGroup(1,[inner,outer],-1,"1;0;1;0")
    gmsh.model.addPhysicalGroup(2,[s1],-1,"Domain")
    gmsh.model.mesh.generate(2); gmsh.model.mesh.setOrder(2)
    out = datadir("elastico","p3r$narc.msh"); mkpath(dirname(out)); gmsh.write(out)
    gmsh.finalize(); return out
end

for narc in (11, 21)
    dad = format2d(mesh(narc), props; tipo=2, pontointerno=false)
    assemble!(dad; npg=16, threaded=false); solve(dad)
    uc = maximum(abs, dad.u)
    H_G_hyper(dad; npg=50, threaded=false); solve(dad)
    uh = maximum(abs, dad.u)
    println("narc=$narc n=$(dad.n)  cbie max|u|=$uc  hbie max|u|=$uh  ratio=$(uh/uc)")
end
