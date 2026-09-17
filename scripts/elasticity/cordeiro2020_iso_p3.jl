# Isotropic Kelvin CBIE / HBIE on Cordeiro & Leonel 2020 Problem 3
# (quarter annulus, same mesh and BCs as the anisotropic run).
#
#   julia --project=. scripts/elasticity/cordeiro2020_iso_p3.jl
using DrWatson
@quickactivate :BEM
using LinearAlgebra, StaticArrays
include(datadir("Laplace", "Laplace_dad.jl"))

# Paper material 1 Young's / Poisson, used as isotropic constants (MPa, mm)
const E = 124.04e3
const ν = 0.334
const Ri, Ro, P = 600.0, 900.0, 1000.0

function p3_mesh()
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("cordeiro_p3_iso")
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
    gmsh.model.addPhysicalGroup(1, [bottom], -1, "1;$P;1;0")
    gmsh.model.addPhysicalGroup(1, [top], -1, "0;0;0;0")
    gmsh.model.addPhysicalGroup(1, [inner, outer], -1, "1;0;1;0")
    gmsh.model.addPhysicalGroup(2, [s1], -1, "Domain")
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(2)
    out = datadir("elastico", "cordeiro_p3_iso.msh")
    mkpath(dirname(out))
    gmsh.write(out)
    gmsh.finalize()
    return out
end

function run_case(label, props)
    dad = format2d(msh, props; tipo=2, pontointerno=false)
    assemble!(dad; npg=16, threaded=false)
    solve(dad)
    uc = copy(dad.u)
    Hc, Gc = copy(dad.H), copy(dad.G)
    cc = cond(Matrix(dad.A))
    H_G_hyper(dad; npg=50, threaded=false)
    solve(dad)
    uh = copy(dad.u)
    ch = cond(Matrix(dad.A))
    rel = norm(uh .- uc) / (norm(uc) + 1e-30)
    println("\n$label  nodes=$(dad.n)  elems=$(length(dad.elements))")
    println("  CBIE  max|u|=$(maximum(abs, uc)) mm   cond(A)=$(cc)")
    println("  HBIE  max|u|=$(maximum(abs, uh)) mm   cond(A)=$(ch)")
    println("  ||u_HBIE − u_CBIE|| / ||u_CBIE|| = $rel")
    e1 = zeros(2 * dad.n); e2 = zeros(2 * dad.n)
    for i in 1:dad.n
        e1[2i - 1] = 1
        e2[2i] = 1
    end
    println("  HBIE ||H*(1,0)||=$(norm(dad.H * e1))  ||H*(0,1)||=$(norm(dad.H * e2))")
    return (; uc, uh, rel)
end

println("="^64)
println(" Cordeiro P3 — isotropic Kelvin, E=$(E) MPa, ν=$(ν)")
println("="^64)
msh = p3_mesh()
run_case("plane stress", Elasticity(E, ν, 1.0; plane_stress=true))
run_case("plane strain", Elasticity(E, ν, 1.0; plane_strain=true))
println("\ndone")
