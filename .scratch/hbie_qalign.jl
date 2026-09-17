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
H, G = meshH.H, meshH.G
r = (H * u - G * t[1:nb])[1:nb]
IDW = BEM.Plate._unsym_ID_W(meshH; npg=4)
qcol = zeros(nb)
qrow = zeros(nb)
@inbounds for i in 1:n
    qcol[5i-4:5i] .= IDW[5i-4:5i, 5]
    qrow[5i-4:5i] .= IDW[5i, :]
end
function report(name, q)
    nq, nr = norm(q), norm(r)
    @printf("%s  ||q||=%.3e  cos=%.4f  ||r-q||/||r||=%.3f\n",
        name, nq, dot(r, q) / (nr * nq + 1e-30), norm(r - q) / nr)
end
report("col5", qcol)
report("row5", qrow)
report("-col5", -qcol)
report("-row5", -qrow)
# try IDW as if physics: vec of each 5x5'
qT = zeros(nb)
@inbounds for i in 1:n
    B = IDW[5i-4:5i, :]
    qT[5i-4:5i] .= B'[:, 5]  # same as row5
end
report("B' col5", qT)
