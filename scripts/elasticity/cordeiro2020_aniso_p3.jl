# Redo Cordeiro & Leonel 2020 Problem 3 (quarter annulus, anisotropic).
#
#   julia --project=. scripts/elasticity/cordeiro2020_aniso_p3.jl
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf
include(datadir("Laplace", "Laplace_dad.jl"))

const MAT1 = (E1=124.04e3, E2=10.09e3, G12=6.03e3, ν12=0.344,
              η12_1=1.255, η12_2=-0.031)
const Ri, Ro, P = 600.0, 900.0, 1000.0   # mm = 0.6 m / 0.9 m (Fig. 7.8)

function p3_mesh()
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("cordeiro_p3")
    lc = 80.0
    c = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, lc)
    p1 = gmsh.model.geo.addPoint(Ri, 0.0, 0.0, lc)
    p2 = gmsh.model.geo.addPoint(Ro, 0.0, 0.0, lc)
    p3 = gmsh.model.geo.addPoint(0.0, Ro, 0.0, lc)
    p4 = gmsh.model.geo.addPoint(0.0, Ri, 0.0, lc)
    bottom = gmsh.model.geo.addLine(p1, p2)
    outer = gmsh.model.geo.addCircleArc(p2, c, p3)
    top = gmsh.model.geo.addLine(p3, p4)
    inner = gmsh.model.geo.addCircleArc(p4, c, p1)
    cl = gmsh.model.geo.addCurveLoop([bottom, outer, top, inner])
    s1 = gmsh.model.geo.addPlaneSurface([cl])
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.setTransfiniteCurve(bottom, 3)
    gmsh.model.mesh.setTransfiniteCurve(top, 3)
    gmsh.model.mesh.setTransfiniteCurve(outer, 11)
    gmsh.model.mesh.setTransfiniteCurve(inner, 11)
    gmsh.model.addPhysicalGroup(1, [bottom], -1, "1;0;1;-$P")  # Fig. 13a: ty = −P
    gmsh.model.addPhysicalGroup(1, [top], -1, "0;0;0;0")
    gmsh.model.addPhysicalGroup(1, [inner, outer], -1, "1;0;1;0")
    gmsh.model.addPhysicalGroup(2, [s1], -1, "Domain")
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(2)
    out = datadir("elastico", "cordeiro_p3.msh")
    mkpath(dirname(out))
    gmsh.write(out)
    gmsh.finalize()
    return out
end

function solve_case(msh, props; bie, npg, nearfield=:sinh)
    dad = format2d(msh, props; tipo=2, pontointerno=false)
    set_cache!(dad; nearfield=nearfield)
    if bie === :hbie
        H_G_hyper(dad; npg=npg, threaded=false)
    else
        assemble!(dad; npg=npg, threaded=false)
    end
    solve(dad)
    A = Matrix(dad.A)
    e1 = zeros(2 * dad.n); e2 = zeros(2 * dad.n)
    for i in 1:dad.n
        e1[2i - 1] = 1
        e2[2i] = 1
    end
    return (; u=copy(dad.u), umax=maximum(abs, dad.u),
        cond=cond(A), n=dad.n,
        He1=norm(dad.H * e1), He2=norm(dad.H * e2))
end

println("="^64)
println(" Cordeiro P3 anisotropic — SST eqs. 34–37")
println(" MAT1 plane stress, 24 disc. quads, P=$(P) MPa on bottom (tx)")
println("="^64)

msh = p3_mesh()
props = AnisotropicElasticity(lekhnitskii_engineering(
    MAT1.E1, MAT1.E2, MAT1.G12, MAT1.ν12;
    η12_1=MAT1.η12_1, η12_2=MAT1.η12_2))

cbie = solve_case(msh, props; bie=:cbie, npg=16, nearfield=:sinh)
@printf("CBIE npg=16 sinh   max|u|=%.6f mm  cond=%.3e\n", cbie.umax, cbie.cond)

println("\nHBIE  npg   near     max|u| mm      rel vs CBIE     cond(A)      ||H e1||")
uref = cbie.u
for npg in (8, 12, 16, 24, 50)
    for nf in (:sinh, :plain)
        r = solve_case(msh, props; bie=:hbie, npg=npg, nearfield=nf)
        rel = norm(r.u .- uref) / (norm(uref) + 1e-30)
        @printf("      %4d  %-6s  %12.4f   %12.3e   %9.3e   %9.2e\n",
            npg, nf, r.umax, rel, r.cond, r.He1)
    end
end

println("done")
