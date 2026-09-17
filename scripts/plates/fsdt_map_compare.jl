# Compare FSDT singularity maps: Gauss, Telles, sinh, sinh-sinh, power.
# 1) Wang θ-integral (ρ=0 at quadrant ends)
# 2) On/near-element ξ assembly (isotropic Reissner, cheap kernels)
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf
using BEM.Plate

relmax(A, G) = maximum(abs.(A .- G)) / max(maximum(abs.(G)), eps())

# ---------------------------------------------------------------------------
println("="^68)
println(" 1. Wang θ-integral  (isotropic D, AT)")
println("="^68)
p = FSDTProps(; E=1e5, ν=0.3, h=0.05)
lp = LaminateFSDTProps(p)
pts = [
    (SVector(0.4, 0.2), "R≈0.45"),
    (SVector(0.08, 0.04), "R≈0.09"),
    (SVector(1.0, 0.0), "R=1"),
]
pf, nh = SVector(0.0, 0.0), SVector(1.0, 0.0)

let (pg, lab) = pts[1]
    Ut, Pt, _ = wang_kernels(pg, pf, nh, lp.D, lp.AT; nθ=40, map=:telles)
    Us, Ps, _ = wang_kernels(pg, pf, nh, lp.D, lp.AT; nθ=40, map=:sinh, sinh_b=1e-4)
    @printf("  gold Telles40 vs sinh40  relU=%.2e  relP=%.2e  (%s)\n",
        relmax(Ut, Us), relmax(Pt, Ps), lab)
end

maps = (
    (:gauss, 1e-3),
    (:telles, 1e-3),
    (:power, 1e-3),
    (:sinh, 1e-2),
    (:sinh, 1e-3),
    (:sinh, 1e-4),
    (:sinhsinh, 1e-3),
)
ns = (6, 8, 10, 12, 16)
golds = [wang_kernels(pg, pf, nh, lp.D, lp.AT; nθ=40, map=:sinh, sinh_b=1e-4)
         for (pg, _) in pts]
println("  rel max |U-Ugold|  (P in parentheses)")
@printf("  %-18s", "map")
for nθ in ns
    @printf("  nθ=%-4d", nθ)
end
println()
best = Dict{Int,Tuple{Float64,String}}()
for nθ in ns
    best[nθ] = (Inf, "")
end
for (mp, b) in maps
    @printf("  %-10s b=%-7.0e", mp, b)
    for nθ in ns
        accU, accP = 0.0, 0.0
        for (k, (pg, _)) in enumerate(pts)
            Ug, Pg, _ = golds[k]
            U, P, _ = wang_kernels(pg, pf, nh, lp.D, lp.AT; nθ=nθ, map=mp, sinh_b=b)
            accU = max(accU, relmax(U, Ug))
            accP = max(accP, relmax(P, Pg))
        end
        @printf("  %.1e(%.1e)", accU, accP)
        tag = "$mp b=$b"
        if accU + accP < best[nθ][1]
            best[nθ] = (accU + accP, tag)
        end
    end
    println()
end
println("  best per nθ:")
for nθ in ns
    @printf("    nθ=%2d  %s  (U+P rel=%.2e)\n", nθ, best[nθ][2], best[nθ][1])
end

# ---------------------------------------------------------------------------
println("\n" * "="^68)
println(" 2. Near-element ξ assembly  (isotropic Reissner, SS square a=1 n_el=3)")
println("="^68)

function _ξ_compare()
    props = FSDTProps(; E=1e5, ν=0.3, h=0.05, q_c=1.0)
    gold = build_square_fsdt(; a=1.0, n_el=3, bc="SSSS", props=props, n_internal=1)
    assemble_fsdt!(gold; npg=20, nsub=32, map=:gauss)
    Hg, Gg = gold.H, gold.G
    println("  gold: plain Gauss  npg=20 nsub=32  (no clustering)")
    amap = (
        (:gauss, 1e-3, 8, 10),
        (:telles, 1e-3, 4, 10),
        (:telles, 1e-3, 8, 10),
        (:sinh, 1e-2, 8, 10),
        (:sinh, 1e-3, 8, 10),
        (:sinh, 1e-4, 8, 10),
        (:sinhsinh, 1e-3, 8, 10),
        (:sinh, 1e-3, 4, 8),
        (:telles, 1e-3, 8, 8),
    )
    @printf("  %-12s %-8s %4s %4s  %10s %10s\n", "map", "b", "nsub", "npg", "relH", "relG")
    bestξ = (Inf, "")
    for (mp, b, nsub, npg) in amap
        m = build_square_fsdt(; a=1.0, n_el=3, bc="SSSS", props=props, n_internal=1)
        assemble_fsdt!(m; npg=npg, nsub=nsub, map=mp, sinh_b=b)
        rH, rG = relmax(m.H, Hg), relmax(m.G, Gg)
        @printf("  %-12s %-8.0e %4d %4d  %10.2e %10.2e\n", mp, b, nsub, npg, rH, rG)
        if rH + rG < bestξ[1]
            bestξ = (rH + rG, "$mp b=$b nsub=$nsub npg=$npg")
        end
    end
    println("  best ξ: ", bestξ[2], "  (H+G rel=", @sprintf("%.2e", bestξ[1]), ")")
end
_ξ_compare()
println("Done.")
