# Profile assembly + allocation report + threading speedup
using DrWatson
@quickactivate :BEM
using BenchmarkTools
using Printf

include(datadir("Laplace", "Laplace_dad.jl"))

println("="^60)
println(" Assembly performance  (threads=$(Threads.nthreads()))")
println("="^60)

msh = quadrado(ndiv=30, show=false, nome="prof_quad")
dad = format2d(msh, Laplace(1.0); pontointerno=false)
println("n = ", dad.n, "  elements = ", length(dad.elements))

# --- allocations / time ---
print("threaded=false: ")
@btime H_G_full_direct($dad; npg=12, threaded=false) setup = (dad2 = deepcopy($dad); dad = dad2)

dad = format2d(msh, Laplace(1.0); pontointerno=false)
print("threaded=true:  ")
@btime H_G_full_direct($dad; npg=12, threaded=true) setup = (dad2 = deepcopy($dad); dad = dad2)

# --- Newton projection cost ---
println("\nClosest-point Newton vs linear guess")
elem = dad.elements[1]
nodes = dad.Nodes[elem.index]
pf = dad.Nodes[1] + Point2D(0.01, 0.02)
@btime closest_point_1d($(dad.element_type), $nodes, $pf)

# --- AD smoke test ---
println("\nAD-compatible heat RHS")
H_G_full_direct(dad; npg=10, threaded=true)
DIBEM(dad)
prob, sys = build_heat_ode(dad; tspan=(0.0, 0.1))
using ForwardDiff
u0 = prob.u0
g = u -> sum(abs2, heat_rhs(u, prob.p, 0.0))
cfg = ForwardDiff.GradientConfig(g, u0)
grad = ForwardDiff.gradient(g, u0, cfg)
println("  ‖∇g‖ = ", norm(grad), "  (finite)")

println("\nTips to reduce allocations / time:")
println("  • Prefer threaded=true (set JULIA_NUM_THREADS)")
println("  • Far-field path already avoids quadrature (point collocation)")
println("  • Cache element node arrays (done in H_G_full_direct)")
println("  • For large N use H_G_Hmat or HalfSpaceBEM :fft/:hmatrix")
println("  • GPU far field: assemble!(dad; method=:gpu)  (KernelAbstractions)")
println("    near-field Newton+quad remains CPU-bound (`near=:cpu`)")
println("Done.")
