using DrWatson
@quickactivate :BEM
using LinearAlgebra, StaticArrays
include(datadir("Laplace", "Laplace_dad.jl"))

MAT1 = (E1=124.04e3, E2=10.09e3, G12=6.03e3, ν12=0.334, η12_1=1.255, η12_2=-0.031)
props = AnisotropicElasticity(lekhnitskii_engineering(MAT1.E1, MAT1.E2, MAT1.G12, MAT1.ν12;
    η12_1=MAT1.η12_1, η12_2=MAT1.η12_2))

function p3_mesh()
    gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("p3s"); lc=80.0
    c = gmsh.model.geo.addPoint(0.0,0.0,0.0,lc)
    p1 = gmsh.model.geo.addPoint(300.0,0.0,0.0,lc)
    p2 = gmsh.model.geo.addPoint(600.0,0.0,0.0,lc)
    p3 = gmsh.model.geo.addPoint(0.0,600.0,0.0,lc)
    p4 = gmsh.model.geo.addPoint(0.0,300.0,0.0,lc)
    bottom = gmsh.model.geo.addLine(p1,p2)
    outer = gmsh.model.geo.addCircleArc(p2,c,p3)
    top = gmsh.model.geo.addLine(p3,p4)
    inner = gmsh.model.geo.addCircleArc(p4,c,p1)
    cl = gmsh.model.geo.addCurveLoop([bottom,outer,top,inner])
    s1 = gmsh.model.geo.addPlaneSurface([cl])
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.setTransfiniteCurve(bottom,3); gmsh.model.mesh.setTransfiniteCurve(top,3)
    gmsh.model.mesh.setTransfiniteCurve(outer,11); gmsh.model.mesh.setTransfiniteCurve(inner,11)
    gmsh.model.addPhysicalGroup(1,[bottom],-1,"1;0;1;-1000")
    gmsh.model.addPhysicalGroup(1,[top],-1,"0;0;0;0")
    gmsh.model.addPhysicalGroup(1,[inner,outer],-1,"1;0;1;0")
    gmsh.model.addPhysicalGroup(2,[s1],-1,"Domain")
    gmsh.model.mesh.generate(2); gmsh.model.mesh.setOrder(2)
    out = datadir("elastico","p3_scale.msh"); mkpath(dirname(out)); gmsh.write(out)
    gmsh.finalize(); return out
end

function report(lab, dad)
    println("  $lab  max|u|=$(maximum(abs,dad.u))  cond(A)=$(cond(Matrix(dad.A)))  |b|=$(norm(dad.b))")
end

msh = p3_mesh()
dad = format2d(msh, props; tipo=2, pontointerno=false)

assemble!(dad; npg=16, threaded=false); solve(dad); report("cbie", dad)
uc = copy(dad.u)

H_G_hyper(dad; npg=50, threaded=false); solve(dad); report("hbie", dad)
uh = copy(dad.u)

# row-equilibrate mixed system and resolve
applyBC(dad)
A = Matrix(dad.A); b = copy(dad.b)
rs = [norm(view(A, i, :)) for i in 1:size(A,1)]
rs = map(x -> x < 1e-30 ? 1.0 : x, rs)
Ae = A ./ rs; be = b ./ rs
println("  hbie cond raw=$(cond(A))  equilibrated=$(cond(Ae))")
x = Ae \ be
u = zeros(length(x)); t = zeros(length(x))
BEM.split_sol!(dad, x, u, t)
println("  hbie-eq  max|u|=$(maximum(abs,u))  rel vs cbie=$(norm(u.-uc)/(norm(uc)+1e-30))")

# column scale by typical C/L vs 1
# Neumann cols are u (~mm), Dirichlet cols are t (~MPa)
println("  |H|fro=$(norm(dad.H))  |G|fro=$(norm(dad.G))")
println("done")
