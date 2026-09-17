# Two-region Laplace with type-3 interface
using DrWatson
@quickactivate :BEM
using BEM.MultiRegion
include(datadir("Laplace", "two_regions.jl"))

println("="^60)
println(" Multi-region Laplace + interface (BC type 3)")
println("="^60)

msh = mesh_two_regions(ndiv=10, show=false)
props = Laplace(1.0)
prob = load_two_regions(msh, props)

println("region L: n=$(prob.regions[1].n), R: n=$(prob.regions[2].n)")
pair_interfaces!(prob)
println("interface pairs: ", length(prob.interfaces))

assemble_multiregion(prob; npg=12)
solve_multiregion!(prob)

# Exact solution T=x on the unit square
errL = norm([prob.regions[1].T[i] - prob.regions[1].Nodes[i][1] for i in 1:prob.regions[1].n]) /
       norm([prob.regions[1].Nodes[i][1] for i in 1:prob.regions[1].n])
errR = norm([prob.regions[2].T[i] - prob.regions[2].Nodes[i][1] for i in 1:prob.regions[2].n]) /
       norm([prob.regions[2].Nodes[i][1] for i in 1:prob.regions[2].n])
println("rel error left  = ", errL)
println("rel error right = ", errR)

# continuity check
max_jump = 0.0
for ip in prob.interfaces
    Ta = prob.regions[ip.reg_a].T[ip.node_a]
    Tb = prob.regions[ip.reg_b].T[ip.node_b]
    max_jump = max(max_jump, abs(Ta - Tb))
end
println("max interface jump |Ta-Tb| = ", max_jump)
println("Done.")
