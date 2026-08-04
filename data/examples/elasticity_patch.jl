# Gmsh square + discontinuous format2d — constant strain patch test
using DrWatson
@quickactivate :BEM
include(datadir("Laplace", "Laplace_dad.jl"))

println("="^60)
println(" Example: elasticity patch (Gmsh + format2d, discontinuous)")
println("="^60)

E, ν = 1.0, 0.3
εxx = 0.01
msh = quadrado_elasticity(ndiv=10, show=false, nome="ex_elast_patch", ordem=2)
dad = format2d(msh, Elasticity(E, ν, 1.0; plane_strain=true); tipo=2, pontointerno=false)
println(dad)

ana = ana_elasticity_patch(; E=E, ν=ν, εxx=εxx)
apply_analytical_bc!(dad, ana)
H_G_full_direct(dad; npg=12, threaded=false)
solve(dad)
err = rel_error(dad)
println("  rel_error(u) = ", err)
println("  expected     < 0.15 on this mesh")
@assert err < 0.15
println("OK.")
