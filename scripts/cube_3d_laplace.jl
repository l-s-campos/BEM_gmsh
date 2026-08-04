# 3D Laplace on the unit cube surface (Dirichlet top/bottom)
using DrWatson
@quickactivate :BEM
include(datadir("Laplace", "cube_mesh.jl"))

println("="^60)
println(" 3D Laplace BEM — unit cube")
println("="^60)

msh = mesh_cube(L=1.0, ndiv=3, show=false)
props = Laplace(1.0)
dad = format3d(msh, props; pontointerno=false)
println(dad)

H_G_full_direct(dad; npg=8, threaded=true)
solve(dad)

# exact T = z
Terr = [dad.T[i] - dad.Nodes[i][3] for i in 1:dad.n]
err = norm(Terr) / max(norm(getindex.(dad.Nodes, 3)), eps())
println("relative L2 error vs T=z: ", err)
println("T range: ", extrema(dad.T))
println("Done.")
