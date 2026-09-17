using LinearAlgebra, Printf, StaticArrays, BEM, BEM.Plate

plies = [(4e6, 2e6, 0.25, 1e6, θ, 0.05) for θ in (0.0, 90.0)]
p = laminate_unsym_props(plies; Ks=5 / 6, G13=1e6, G23=5e5, nθ=10)
pg, pf = SVector(0.4, 0.2), SVector(0.0, 0.0)
nh, nξ = SVector(1.0, 0.0), SVector(0.0, 1.0)
U, P = unsym_fsdt_kernels(pg, pf, nh, p)
W, S = unsym_hbie_kernels(pg, pf, nh, nξ, p)
hfd = 1e-5
_, Pp = unsym_fsdt_kernels(pg, pf + hfd * nξ, nh, p)
_, Pm = unsym_fsdt_kernels(pg, pf - hfd * nξ, nh, p)
Sfd = (Pp - Pm) / (2 * hfd)
rel = maximum(abs.(S .- Sfd)) / maximum(abs, Sfd)
@printf("max|S|=%.4e  max|Sfd|=%.4e  rel=%.4e\n", maximum(abs, S), maximum(abs, Sfd), rel)
@printf("max|W|=%.4e  finite S=%s\n", maximum(abs, W), all(isfinite, S))
display(S)
println()
display(Sfd)
