# Gmsh square + discontinuous format2d — analytical T = x
using DrWatson
@quickactivate :BEM
include(datadir("Laplace", "Laplace_dad.jl"))

println("="^60)
println(" Example: Laplace T=x (Gmsh + format2d, discontinuous)")
println("="^60)

msh = quadrado(ndiv=12, show=false, nome="ex_laplace_Tx", ordem=2)
dad = format2d(msh, Laplace(1.0); tipo=2, pontointerno=false)  # 3 Gauss nodes/elem
println(dad)
println("  Gauss nodes/element = ", length(dad.elements[1].index))

ana = ana_laplace_linear(; direction=SA[1.0, 0.0], k=1.0)
apply_analytical_bc!(dad, ana)
H_G_full_direct(dad; npg=12, threaded=false)
solve(dad)
err = rel_error(dad)
println("  rel_error(T) = ", err)
println("  expected     < 5e-2 on this mesh")
@assert err < 0.05
println("OK.")
