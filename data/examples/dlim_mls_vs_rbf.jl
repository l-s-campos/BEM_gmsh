# DLIM second-layer comparison: MLS/Shepard vs RBF (Zhang et al. 2017)
using DrWatson
@quickactivate :BEM
include(datadir("Laplace", "Laplace_dad.jl"))

println("="^60)
println(" DLIM second layer: MLS vs RBF")
println("="^60)

function run_case(name, ana, Tana; ndiv=12)
    msh = quadrado(ndiv=ndiv, show=false, nome="dlim_" * name, ordem=1)
    dad = format2d(msh, Laplace(1.0); tipo=1, pontointerno=false)
    apply_analytical_bc!(dad, ana)
    cmp = compare_dlim_second_layer(dad, Tana; npg=12)
    println("\n[$name]")
    println("  sources/virtuals = $(cmp.n_s)/$(cmp.n_v)")
    println("  MLS  rel_error   = $(round(cmp.err_mls; sigdigits=4))  ($(round(cmp.time_mls; digits=3)) s)")
    println("  RBF  rel_error   = $(round(cmp.err_rbf; sigdigits=4))  ($(round(cmp.time_rbf; digits=3)) s)")
    println("  max |T_mls-T_rbf| = $(round(cmp.diff_max; sigdigits=4))")
    return cmp
end

c1 = run_case("T=x", ana_laplace_linear(; direction=SA[1.0, 0.0], k=1.0), (x, y) -> x)
c2 = run_case("T=x²-y²", ana_laplace_quadratic(; k=1.0), (x, y) -> x^2 - y^2; ndiv=14)

println("\nSummary: both second-layer options recover the analytical field;")
println("  choose :mls (local Shepard) for speed, :rbf for smoother condensation.")
@assert c1.err_mls < 0.05 && c1.err_rbf < 0.05
@assert c2.err_mls < 0.08 && c2.err_rbf < 0.08
println("OK.")
