using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf
include(datadir("elastico", "two_blocks_contact.jl"))

props = Elasticity(100.0, 0.3, 1.0; plane_strain=true)
prob = load_two_blocks_contact(props; W=4.0, H=1.5, gap=0.04, μ=0.3,
    ndiv_bot=12, ndiv_top=12, ndiv_y=3, nome="ncheck")
apply_parabolic_contact_gap!(prob; R=10.0, method=:ntn)
pair_contacts!(prob; method=:ntn)
da, db = prob.regions
for (i, cp) in enumerate(prob.contacts[1:min(4, end)])
    n1 = da.Normal[cp.node_a]
    n2 = db.Normal[cp.node_b]
    R1 = Matrix(node_rotation2d(n1))
    R2 = Matrix(node_rotation2d(n2))
    R = R2 * R1'
    println("pair $i")
    println("  n1=$n1 n2=$n2")
    println("  R1=$R1")
    println("  R2=$R2")
    println("  R=R2*R1' = $R")
    println("  gap0=$(cp.gap0) x=$(da.Nodes[cp.node_a][1])")
end
