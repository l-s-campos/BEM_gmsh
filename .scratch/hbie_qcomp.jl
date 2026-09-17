using LinearAlgebra, Printf, StaticArrays, BEM, BEM.Plate

plies = [(4e6, 2e6, 0.25, 1e6, θ, 0.05) for θ in (0.0, 90.0)]
props = laminate_unsym_props(plies; Ks=5 / 6, G13=1e6, G23=5e5, q_c=1.0, nθ=6)
meshC = build_square_fsdt(; a=1.0, n_el=2, bc="SSSS", props=props, n_internal=1)
assemble_fsdt!(meshC; npg=4, nsub=4, singular=:guiggiani, ninterp=12)
dibem_fsdt!(meshC; npg=4)
solve_fsdt!(meshC)
n = length(meshC.nodes)
nb = 5n
u, t = meshC.u, meshC.t
meshH = build_square_fsdt(; a=1.0, n_el=2, bc="SSSS", props=props, n_internal=1)
assemble_fsdt!(meshH; npg=4, nsub=4, singular=:guiggiani, bie=:hbie, ninterp=12,
    scale_hbie=false)
r = (meshH.H * u - meshH.G * t[1:nb])[1:nb]
q = meshH.q[1:nb]
names = ("N1", "N2", "M1", "M2", "Q")
for k in 1:5
    rk = r[k:5:nb]
    qk = q[k:5:nb]
    @printf("  %s  ||r||=%.3e  ||q||=%.3e  cos=%.3f\n", names[k],
        norm(rk), norm(qk), dot(rk, qk) / (norm(rk) * norm(qk) + 1e-30))
end
