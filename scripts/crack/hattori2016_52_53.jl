# Hattori, Alatawi & Trevelyan, IJNME 109:965–981 (2016) §§5.2–5.3
# Anisotropic dual BEM (COD) and XBEM (Sih–Paris–Irwin) vs the paper figures.
using DrWatson
@quickactivate :BEM
using BEM.Crack
using LinearAlgebra
using Printf
using Plots

const σ = 1.0
# Off-axis dual HBIE (SST remainder) needs ≳16 Gauss points; 10 is enough
# only when the material axes line up with the crack.
const NPG = 20
const NENR = 3
const NV = 9

"""Hattori §5.2: E₁ = G₁₂(φ + 2ν₁₂ + 1), E₂ = E₁/φ (plane stress)."""
function orthotropic_phi(φ; G12=6.0, ν12=0.03)
    E1 = G12 * (φ + 2ν12 + 1)
    E2 = E1 / φ
    return AnisotropicElasticity(lekhnitskii_params(E1, E2, G12, ν12))
end

function graphite_epoxy(β_deg)
    return AnisotropicElasticity(lekhnitskii_params(144.8, 11.7, 9.66, 0.21; θ_deg=β_deg))
end

function nearest_faceA(dad, tip)
    faceA = crack_face_nodes(dad; face=2)
    return faceA[argmin(norm(dad.Nodes[i] - tip) for i in faceA)]
end

function mean_KI(dad; sample=2)
    tips = williams_tip_positions(dad)
    isempty(tips) && error("no interior Williams tips")
    KIs = Float64[]
    KIIs = Float64[]
    for tip in tips
        KI, KII = sif_cod_dual(dad, nearest_faceA(dad, tip); sample=sample)
        push!(KIs, KI)
        push!(KIIs, KII)
    end
    return 0.5 * (abs(KIs[1]) + abs(KIs[end])), 0.5 * (abs(KIIs[1]) + abs(KIIs[end]))
end

function run_dual_xbem(dad; a, npg=NPG)
    assemble_dual!(dad; npg=npg, threaded=false)
    solve_dual!(dad; threaded=false)
    KIc, KIIc = mean_KI(dad)
    _, KIx, KIIx = solve_xbem!(dad; n_enr=NENR, npg=npg, n_v=NV, threaded=false)
    KIxn = 0.5 * (abs(KIx[1]) + abs(KIx[end]))
    KIIxn = 0.5 * (abs(KIIx[1]) + abs(KIIx[end]))
    s = σ * sqrt(π * a)
    return (cod=KIc / s, xbem=KIxn / s, kii_cod=KIIc / s, kii_xbem=KIIxn / s)
end

# Digitized Hattori 2012 Fig. 8 (geometrical anisotropic XFEM = 2016 [12])
const PHI = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.5, 2.5, 3.5, 4.5]
const KI_XFEM12 = [1.175, 1.118, 1.088, 1.072, 1.060, 1.052, 1.042, 1.032, 1.026, 1.022]

# Digitized Hattori 2016 Fig. 7 Direct SIF / XFEM [34]
const BETA = collect(0.0:10.0:90.0)
const KI_DIRECT53_ = [1.18, 1.19, 1.22, 1.32, 1.46, 1.64, 1.84, 2.04, 2.16, 2.18]
const KI_BEM24 = [1.18, 1.16, 1.18, 1.22, 1.30, 1.50, 1.76, 2.06, 2.18, 2.22]
const KI_XFEM34 = [1.18, 1.19, 1.25, 1.38, 1.52, 1.70, 1.88, 2.06, 2.12, 2.12]

println("="^72)
println(" Hattori 2016 §5.2 — orthotropic centred crack, a/w = 0.2, h/w = 1")
println("="^72)
w52, a52 = 1.0, 0.2
KIfed = analytical_KI_center_crack(σ, a52; W=w52) / (σ * sqrt(π * a52))
println(@sprintf("  Feddersen (φ=1 isotropic)  F = %.4f", KIfed))
cod52 = Float64[]
xbem52 = Float64[]
for φ in PHI
    dad = dual_elasticity_problem(; W=w52, H=w52, a=a52, σ=σ,
        ndiv_b=6, ndiv_h=6, ndiv_crack=5, ordem=2,
        nome="h16_52_$(replace(string(φ), '.' => 'p'))",
        props=orthotropic_phi(φ))
    r = run_dual_xbem(dad; a=a52)
    push!(cod52, r.cod)
    push!(xbem52, r.xbem)
    @printf("  φ=%4.1f  COD=%6.3f  XBEM=%6.3f  XFEM[12]≈%6.3f  KII_cod=%6.3f\n",
        φ, r.cod, r.xbem, KI_XFEM12[findfirst(==(φ), PHI)], r.kii_cod)
end
i1 = argmin(abs.(PHI .- 1.1))
@printf("  φ=1.1 vs Feddersen: COD rel=%.2f%%  XBEM rel=%.2f%%\n",
    100 * abs(cod52[i1] - KIfed) / KIfed,
    100 * abs(xbem52[i1] - KIfed) / KIfed)

println()
println("="^72)
println(" Hattori 2016 §5.3 — graphite-epoxy double edge, a/w = 0.5, h/w = 1")
println("="^72)
w53, a53 = 1.0, 0.5
cod53 = Float64[]
xbem53 = Float64[]
for β in BETA
    dad = double_edge_crack_problem(; W=w53, H=w53, a=a53, σ=σ,
        ndiv_b=6, ndiv_side=3, ndiv_crack=4, ordem=2,
        nome="h16_53_$(Int(β))",
        props=graphite_epoxy(β))
    nt = williams_tip_positions(dad)
    length(nt) == 2 || @warn "expected 2 interior tips, got $(length(nt))"
    r = run_dual_xbem(dad; a=a53)
    push!(cod53, r.cod)
    push!(xbem53, r.xbem)
    k = findfirst(==(β), BETA)
    @printf("  β=%4.0f°  COD=%6.3f  XBEM=%6.3f  Direct≈%6.3f  BEM[24]≈%6.3f  XFEM[34]≈%6.3f  (KII_xbem=%6.3f)\n",
        β, r.cod, r.xbem, KI_DIRECT53_[k], KI_BEM24[k], KI_XFEM34[k], r.kii_xbem)
end

outdir = joinpath(@__DIR__, "..", "..", "scripts", "debug")
mkpath(outdir)

p52 = plot(PHI, KI_XFEM12; label="XFEM [12] (digitized)", lw=2, ls=:dash,
    xlabel="φ = E₁/E₂", ylabel="KI / σ√(πa)",
    title="Hattori 2016 §5.2  centred crack  a/w=0.2",
    legend=:topright, ylims=(1.0, 1.25))
plot!(p52, PHI, cod52; label="dual COD", marker=:circle, lw=2)
plot!(p52, PHI, xbem52; label="XBEM", marker=:square, lw=2)
hline!(p52, [KIfed]; label="Feddersen φ=1", ls=:dot, color=:gray)
savefig(p52, joinpath(outdir, "hattori2016_52.png"))

p53 = plot(BETA, KI_DIRECT53_; label="Direct SIF (digitized)", lw=2, ls=:dash,
    xlabel="β (°)", ylabel="KI / σ√(πa)",
    title="Hattori 2016 §5.3  double edge  a/w=0.5",
    legend=:topleft, ylims=(1.0, 2.6))
plot!(p53, BETA, KI_BEM24; label="BEM [24] (digitized)", lw=1.5, ls=:dot)
plot!(p53, BETA, KI_XFEM34; label="XFEM [34] (digitized)", lw=1.5, ls=:dashdot)
plot!(p53, BETA, cod53; label="dual COD", marker=:circle, lw=2)
plot!(p53, BETA, xbem53; label="XBEM", marker=:square, lw=2)
savefig(p53, joinpath(outdir, "hattori2016_53.png"))

println()
println("figures → scripts/debug/hattori2016_52.png , hattori2016_53.png")
