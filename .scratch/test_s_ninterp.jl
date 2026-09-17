using LinearAlgebra, Printf, StaticArrays, BEM, BEM.Plate

plies = [(4e6, 2e6, 0.25, 1e6, θ, 0.05) for θ in (0.0, 90.0)]
p = laminate_unsym_props(plies; Ks=5 / 6, G13=1e6, G23=5e5, nθ=10)
pg, pf = SVector(0.4, 0.2), SVector(0.0, 0.0)
nh, nξ = SVector(1.0, 0.0), SVector(0.0, 1.0)
RX, RY = pg[1] - pf[1], pg[2] - pf[2]
hfd = 1e-5
_, Pp = unsym_fsdt_kernels(pg, pf + hfd * nξ, nh, p)
_, Pm = unsym_fsdt_kernels(pg, pf - hfd * nξ, nh, p)
Sfd = (Pp - Pm) / (2 * hfd)
θ0 = atan(-RX, RY)
n1, n2, nξ1, nξ2 = nh[1], nh[2], nξ[1], nξ[2]
for ni in (16, 20, 32, 48)
    S = BEM.Plate._unsym_S_pole(θ0, RX, RY, n1, n2, nξ1, nξ2, p, ni) .+
        BEM.Plate._unsym_S_pole(θ0 + π, RX, RY, n1, n2, nξ1, nξ2, p, ni)
    S .*= 1 / (4 * π^2)
    rel = maximum(abs.(S .- Sfd)) / maximum(abs, Sfd)
    @printf("ninterp=%2d  max|S|=%.3e  rel=%.3e\n", ni, maximum(abs, S), rel)
end
