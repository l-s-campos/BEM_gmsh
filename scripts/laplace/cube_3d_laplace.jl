# 3D Laplace on the unit cube — mixed BCs for T=z (see cube_3d_convergence.jl)
using DrWatson
@quickactivate :BEM
include(datadir("Laplace", "cube_mesh.jl"))

println("="^60)
println(" 3D Laplace BEM — unit cube (T = z)")
println("="^60)

msh = mesh_cube(L=1.0, ndiv=3, show=false)
dad = format3d(msh, Laplace(1.0); pontointerno=false)
ana = ana_laplace_linear(; direction=SA[0.0, 0.0, 1.0], k=1.0)
attach_analytical!(dad, ana)
println(dad)

assemble!(dad; npg=8, threaded=true)
solve(dad)
println("rel T vs T=z:  ", rel_error(dad))
println("rel q vs -n_z: ", rel_error_flux(dad))
println("T range: ", extrema(dad.T[1:dad.n]))
println("Done.")
