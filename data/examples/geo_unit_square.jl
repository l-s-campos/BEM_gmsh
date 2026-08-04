# Unit square geometric properties via format2d discontinuous mesh
using DrWatson
@quickactivate :BEM
include(datadir("Laplace", "Laplace_dad.jl"))

println("="^60)
println(" Example: geometric props unit square (Gmsh + format2d)")
println("="^60)

msh = quadrado(ndiv=10, show=false, nome="ex_geo_sq", ordem=2)
dad = format2d(msh, Laplace(1.0); tipo=2, pontointerno=false)
g = geometric_props(dad)
println("  perimeter = ", g.perimeter, "  (exact 4)")
println("  area      = ", g.area, "  (exact 1)")
println("  centroid  = ", g.centroid, "  (exact 0.5,0.5)")
@assert g.perimeter ≈ 4 rtol=0.05
@assert g.area ≈ 1 rtol=0.05
@assert g.centroid ≈ SA[0.5, 0.5] atol=0.05
println("OK.")
