# DiBFM elasticity (Zhang et al. EJMS 2019) — patch test
using DrWatson
@quickactivate :BEM
include(datadir("Laplace", "Laplace_dad.jl"))

println("="^60)
println(" DiBFM elasticity (EJMS 2019) — constant strain patch")
println("="^60)

E, ν, εxx = 1.0, 0.3, 0.01
msh = quadrado_elasticity(ndiv=10, show=false, nome="ex_dibfm_el", ordem=1)
dad = format2d(msh, Elasticity(E, ν, 1.0; plane_strain=true); tipo=1, pontointerno=false)
ana = ana_elasticity_patch(; E=E, ν=ν, εxx=εxx)
apply_analytical_bc!(dad, ana)
uana = (x, y) -> SA[ana.u(Point2D(x, y))[1], ana.u(Point2D(x, y))[2]]

cmp = compare_dibfm_elasticity(dad, uana; npg=12)
println("  sources / virtuals = $(cmp.n_s) / $(cmp.n_v)")
println()
println("  Method           rel_error(u)   time (s)")
println("  standard BEM     $(round(cmp.err_std; sigdigits=4))         $(round(cmp.time_std; digits=3))")
println("  DiBFM-MLS        $(round(cmp.err_dibfm_mls; sigdigits=4))         $(round(cmp.time_mls; digits=3))")
println("  DiBFM-RBF2d      $(round(cmp.err_dibfm_rbf; sigdigits=4))         $(round(cmp.time_rbf; digits=3))")

@assert cmp.err_dibfm_mls < 0.15
@assert cmp.err_dibfm_rbf < 0.15
println("OK.")
