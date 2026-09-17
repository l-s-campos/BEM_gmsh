# Center crack — dual BEM (displacement + traction BIE, coincident faces)
using DrWatson
@quickactivate :BEM
using BEM.Crack
using .Crack
include(datadir("elastico", "iso", "center_crack.jl"))

println("="^60)
println(" Center crack — Dual BEM")
println("="^60)

W, H, a = 5.0, 10.0, 1.0
σ, E, ν = 1.0, 3000.0, 0.2

mesh, KI_L, KII_L, KI_R, KII_R, KI_ana = solve_center_crack_dual(;
    W=W, H=H, a=a, σ=σ, E=E, ν=ν,
    n_bottom=6, n_right=12, n_top=6, n_left=12, n_crack=12, npg=12)

println("nodes = ", mesh.n, "  elements = ", length(mesh.elements))
println("  face A (disp BIE) elems: ", length(mesh.crack_face_a))
println("  face B (trac BIE) elems: ", length(mesh.crack_face_b))
println()
println("  tip L: KI=$(round(KI_L,sigdigits=5))  KII=$(round(KII_L,sigdigits=5))")
println("  tip R: KI=$(round(KI_R,sigdigits=5))  KII=$(round(KII_R,sigdigits=5))")
println("  analytical KI (Feddersen) = ", round(KI_ana; sigdigits=6))
KIn = 0.5 * (abs(KI_L) + abs(KI_R))
println("  numerical |KI| mean       = ", round(KIn; sigdigits=6))
println("  relative error            = ", round(abs(KIn - KI_ana) / KI_ana; sigdigits=4))

# Propagation bookkeeping (MTS + Paris) on dual SIFs
function _tip_from_dual(mesh, tip_node, id)
    faceA = crack_face_nodes(mesh; face=2)
    tip = mesh.Nodes[tip_node]
    sort!(faceA; by=i -> norm(mesh.Nodes[i] - tip))
    samp = faceA[min(3, length(faceA))]
    tw = mesh.twin[samp]
    t̂ = mesh.Nodes[samp] - tip
    t̂ = t̂ / (norm(t̂) + eps())
    return CrackTip(id, samp, tw, tip, t̂)
end
tL = _tip_from_dual(mesh, mesh.tip_nodes[1], 1)
tR = _tip_from_dual(mesh, mesh.tip_nodes[2], 2)
path = CrackPath([tL.pos, tR.pos], [tL, tR])
prob = CrackProblem(path; E=E, ν=ν, plane_strain=true,
    C=4.624e-12, m=3.3, R_ratio=2/3)
sifs = [(abs(KI_L), KII_L), (abs(KI_R), KII_R)]
θs, dN = propagate!(prob, sifs; da=0.25 * a, criterion=:MTS)
println()
println("One MTS step da=$(0.25a):")
println("  θ = ", θs)
println("  ΔN (Paris) = ", dN)
println("  new tips = ", [t.pos for t in path.tips])
println("Done.")
