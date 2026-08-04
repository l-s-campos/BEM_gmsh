# Double-layer interpolation BEM (Zhang et al. 2017) — Laplace T=x
using DrWatson
@quickactivate :BEM
include(datadir("Laplace", "Laplace_dad.jl"))

println("="^60)
println(" Example: DLIM Laplace (AMM 2017 Zhang et al.)")
println("="^60)

msh = quadrado(ndiv=12, show=false, nome="ex_dlim", ordem=1)
dad = format2d(msh, Laplace(1.0); tipo=1, pontointerno=false)
ana = ana_laplace_linear(; direction=SA[1.0, 0.0], k=1.0)
apply_analytical_bc!(dad, ana)

# standard discontinuous BEM
H_G_full_direct(dad; npg=12, threaded=false)
solve(dad)
err0 = rel_error(dad)

# DLIM
d = solve_dlim_laplace(dad; npg=12)
err1 = dlim_rel_error(d, (x, y) -> x)

println("  sources / virtuals = ", length(d.source_pos), " / ", length(d.virt_global))
println("  standard BEM rel_error = ", err0)
println("  DLIM        rel_error = ", err1)
@assert err1 < 0.05
println("OK.")
