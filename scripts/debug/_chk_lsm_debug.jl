using DrWatson
@quickactivate :BEM
using BEM.Topology

d, opt = pacheco_problem(1; ne=10, nint=10, degree=1)
g = LevelSetGrid(d; ngrid=40)
println("grid ", length(g.xs), "x", length(g.ys), " dx=", g.dx)
println("φ min/max ", extrema(g.φ), " area_φ=", BEM._area_from_phi(g))
dad = bemdata_from_loops(d)
H_G_full_direct(dad; npg=12, threaded=false)
solve(dad)
DTb, DTi = topological_derivative(dad)
println("DTb ", extrema(DTb), " DTi ", extrema(DTi), " nint=", length(DTi))
DT = BEM._dt_field(g, dad, DTb, DTi)
println("DT grid finite ", count(isfinite, DT), "/", length(DT), " extrema ", extrema(filter(isfinite, DT)))
amstutz_step!(g, dad, DTb, DTi, 0.8)
println("after Amstutz φ extrema ", extrema(g.φ), " area_φ=", BEM._area_from_phi(g))
BEM._protect_dirichlet!(g, d)
lines = marching_squares(g.xs, g.ys, g.φ, 0.0)
println("nlines=", length(lines))
for (i, ln) in enumerate(lines)
    println("  line $i n=$(length(ln)) closed=$(norm(ln[1]-ln[end]))")
end
d2 = phi_to_design(g, d; min_area=1e-3, nel=20)
println("extracted A=", design_area(d2), " nloops=", length(d2.loops), " nseg=", length(d2.loops[1]))
