using LinearAlgebra, Printf, StaticArrays, FastGaussQuadrature, BEM, BEM.Plate

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
At = BEM.Plate._hsu_At(p.AT)
F = zeros(10, 10)
Fp = zeros(10, 10)
Jθ = π / 2
qs, ws = gausslegendre(24)

function Spole(θc)
    return BEM.guiggiani_integral(0.0, -2; laurent=:richardson, qsi=qs, w=ws, h=1e-4) do ξ
        θ = θc + ξ * Jθ
        return BEM.Plate._unsym_dPρ(θ, RX, RY, n1, n2, nξ1, nξ2, p, At, F, Fp) .* Jθ
    end
end

S = (Spole(θ0) .+ Spole(θ0 + π)) ./ (4 * π^2)
rel = maximum(abs.(S .- Sfd)) / maximum(abs, Sfd)
@printf("rich24  max|S|=%.4e  max|Sfd|=%.4e  rel=%.4e\n", maximum(abs, S), maximum(abs, Sfd), rel)
display(S)
println()
display(Sfd)
println()
display(S .- Sfd)
