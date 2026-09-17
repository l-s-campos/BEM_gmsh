# Gauss-point / sinh sweep on isotropic Cordeiro P3 (quarter annulus).
#
#   julia --project=. scripts/elasticity/cordeiro2020_iso_p3_npg.jl
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf
include(datadir("Laplace", "Laplace_dad.jl"))

const E, ν = 124.04e3, 0.344
const Ri, Ro, P = 600.0, 900.0, 1000.0
const NPGS = (2, 4, 6, 8, 10, 12, 16, 24, 32, 50)

function p3_mesh()
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("iso_p3_npg")
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

# CBIE: integrate every element (no 1-point far lump), then rigid-body H.
function assemble_cbie_quad!(dad; npg, nearfield::Symbol=:sinh)
    BEM._init_quadrature!(dad, npg)
    set_cache!(dad; nearfield=nearfield)
    dim = dad.dimension
    n = dad.n
    H = zeros(dim * n, dim * n)
    G = zeros(dim * n, dim * n)
    elems = dad.elements
    for i in 1:n
        pf = point(dad, i)
        ii = BEM.expand(i, dim)
        for el in elems
            xj = dad.Nodes[el.index]
            jj = BEM.expand(el.index, dim)
            hloc = zeros(eltype(H), dim, length(jj))
            gloc = zeros(eltype(G), dim, length(jj))
            integrate_element(dad, el, xj, pf, hloc, gloc; source=i)
            H[ii, jj] .+= hloc
            G[ii, jj] .+= gloc
        end
    end
    @views for i in 1:n
        ii = BEM.expand(i, dim)
        H[ii, ii] .= 0.0
        for j in 1:dim
            H[ii, ii[j]] .= -sum(H[ii, j:dim:end]; dims=2)
        end
    end
    set_cache!(dad; H, G)
    return H, G
end

function solve_umax(dad)
    solve(dad)
    return maximum(abs, dad.u), copy(dad.u)
end

function rel_to(u, uref)
    return norm(u .- uref) / (norm(uref) + 1e-30)
end

println("="^72)
println(" isotropic P3 — npg sweep, sinh vs plain Gauss")
println(" E=$(E) MPa  ν=$(ν)  plane stress  24 disc. quads")
println("="^72)

msh = p3_mesh()
props = Elasticity(E, ν, 1.0; plane_stress=true)
dad0 = format2d(msh, props; tipo=2, pontointerno=false)
nnode = dad0.n

# references: 50-pt always-quad + sinh
dad = format2d(msh, props; tipo=2, pontointerno=false)
assemble_cbie_quad!(dad; npg=50, nearfield=:sinh)
_, uref_c = solve_umax(dad)
umax_c_ref = maximum(abs, uref_c)
dad = format2d(msh, props; tipo=2, pontointerno=false)
set_cache!(dad; nearfield=:sinh)
H_G_hyper(dad; npg=50, threaded=false)
_, uref_h = solve_umax(dad)
umax_h_ref = maximum(abs, uref_h)
println("\nreference (npg=50, sinh, always-quad):")
println("  CBIE max|u|=$(umax_c_ref) mm")
println("  HBIE max|u|=$(umax_h_ref) mm")
println("  ||uH−uC||/||uC|| = $(rel_to(uref_h, uref_c))")

# CBIE default (sinh-near + far lump) at npg=50 as a second baseline
dad = format2d(msh, props; tipo=2, pontointerno=false)
assemble!(dad; npg=50, threaded=false)
_, uref_c_def = solve_umax(dad)

@printf("\n%-6s  %-22s  %-22s  %-22s  %-22s  %-22s\n",
    "npg", "CBIE default", "CBIE sinh", "CBIE plain", "HBIE sinh", "HBIE plain")
@printf("%-6s  %-22s  %-22s  %-22s  %-22s  %-22s\n",
    "", "max|u|  rel", "max|u|  rel", "max|u|  rel", "max|u|  rel", "max|u|  rel")

for npg in NPGS
    row = String[]
    # CBIE default: near sinh, far 1-point
    dad = format2d(msh, props; tipo=2, pontointerno=false)
    assemble!(dad; npg=npg, threaded=false)
    um, u = solve_umax(dad)
    push!(row, @sprintf("%.4f  %.1e", um, rel_to(u, uref_c)))

    # CBIE always-quad sinh
    dad = format2d(msh, props; tipo=2, pontointerno=false)
    assemble_cbie_quad!(dad; npg=npg, nearfield=:sinh)
    um, u = solve_umax(dad)
    push!(row, @sprintf("%.4f  %.1e", um, rel_to(u, uref_c)))

    # CBIE always-quad plain
    dad = format2d(msh, props; tipo=2, pontointerno=false)
    assemble_cbie_quad!(dad; npg=npg, nearfield=:plain)
    um, u = solve_umax(dad)
    push!(row, @sprintf("%.4f  %.1e", um, rel_to(u, uref_c)))

    # HBIE sinh
    dad = format2d(msh, props; tipo=2, pontointerno=false)
    set_cache!(dad; nearfield=:sinh)
    H_G_hyper(dad; npg=npg, threaded=false)
    um, u = solve_umax(dad)
    push!(row, @sprintf("%.4f  %.1e", um, rel_to(u, uref_h)))

    # HBIE plain
    dad = format2d(msh, props; tipo=2, pontointerno=false)
    set_cache!(dad; nearfield=:plain)
    H_G_hyper(dad; npg=npg, threaded=false)
    um, u = solve_umax(dad)
    push!(row, @sprintf("%.4f  %.1e", um, rel_to(u, uref_h)))

    @printf("%-6d  %-22s  %-22s  %-22s  %-22s  %-22s\n", npg, row...)
end

println("\nrel = ||u − u_ref|| / ||u_ref||  with u_ref = npg=50 sinh always-quad")
println("CBIE default uses sinh only on near elements and 1-point far lumping.")
println("done")
