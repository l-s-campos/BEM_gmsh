using LinearAlgebra, Printf, StaticArrays, BEM, BEM.Plate

plies = [(4e6, 2e6, 0.25, 1e6, θ, 0.05) for θ in (0.0, 90.0)]
p = laminate_unsym_props(plies; Ks=5 / 6, G13=1e6, G23=5e5, nθ=8)
nh, nξ = SVector(0.0, 1.0), SVector(0.0, 1.0)  # on a horizontal element, n = nξ
pf = SVector(0.5, 0.0)
for R in (0.4, 0.1, 0.01, 1e-3, 1e-4)
    pg = pf + SVector(R, 0.0)  # along the element
    W, S = unsym_hbie_kernels(pg, pf, nh, nξ, p)
    _, P = unsym_fsdt_kernels(pg, pf, nh, p)
    @printf("R=%.1e  max|S|=%.3e  max|P|=%.3e  S*R²=%.3e  P*R=%.3e  finite=%s\n",
        R, maximum(abs, S), maximum(abs, P), maximum(abs, S) * R^2,
        maximum(abs, P) * R, all(isfinite, S))
end
