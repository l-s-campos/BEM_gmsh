# DiBFM second-layer comparison: HMLS vs RBF2d vs RBF-Hermite
using DrWatson
@quickactivate :BEM
include(datadir("Laplace", "Laplace_dad.jl"))

println("="^60)
println(" DiBFM second layer: HMLS / RBF2d / RBF-Hermite")
println("="^60)

msh = quadrado(ndiv=12, show=false, nome="ex_dibfm", ordem=1)
dad = format2d(msh, Laplace(1.0); tipo=1, pontointerno=false)
apply_analytical_bc!(dad, ana_laplace_linear(; direction=SA[1.0, 0.0], k=1.0))

cmp = compare_dlim_dibfm(dad, (x, y) -> x; npg=12)
println("  sources / virtuals = $(cmp.n_s) / $(cmp.n_v)")
println()
println("  Method              rel_error      time (s)")
println("  standard BEM        $(round(cmp.err_std; sigdigits=4))        $(round(cmp.time_std; digits=3))")
println("  DLIM-MLS            $(round(cmp.err_dlim_mls; sigdigits=4))        $(round(cmp.time_mls; digits=3))")
println("  DLIM-RBF (1D)       $(round(cmp.err_dlim_rbf; sigdigits=4))        $(round(cmp.time_rbf; digits=3))")
println("  DiBFM-HMLS          $(round(cmp.err_dibfm_hmls; sigdigits=4))        $(round(cmp.time_dibfm; digits=3))")
println("  DiBFM-RBF2d         $(round(cmp.err_dibfm_rbf; sigdigits=4))        $(round(cmp.time_dibfm_rbf; digits=3))")
println("  DiBFM-RBF-Hermite   $(round(cmp.err_dibfm_rbfh; sigdigits=4))        $(round(cmp.time_dibfm_rbfh; digits=3))")

@assert cmp.err_dibfm_hmls < 0.08
@assert cmp.err_dibfm_rbf < 0.08
println("OK.")
