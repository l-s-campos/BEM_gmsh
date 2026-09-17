# Ke, Chen, Ku & Chen, Int. J. Numer. Anal. Meth. Geomech. 33:1227–1253 (2009)
# [online 2008]. Anisotropic MTS (Sih–Paris–Irwin hoop) + dual BEM.
#
# Example 4 / Table V: 45° centre crack in a glass-epoxy plate
# (a/w=0.2, h/w=2), fibre angle ψ. Gandhi / Sollero & Aliabadi SIFs.
# Initiation θ from Ke's maximum circumferential stress on those SIFs.
using DrWatson
@quickactivate :BEM
using BEM
using BEM.Crack
using LinearAlgebra
using Printf
using Plots
using Statistics: mean

const σ = 1.0
const W, H, a = 1.0, 2.0, 0.2
const α = π / 4
const NPG = 12
const NCR = 10
const NB = 12
const NH = 20
const S = σ * sqrt(π * a)

# Gandhi [48] as in Ke Table V (KI, KII) / σ√(πa)
const PSI = [0.0, 45.0, 90.0, 105.0, 120.0, 135.0, 180.0]
const GANDHI = [
    (0.522, 0.507),
    (0.515, 0.505),
    (0.513, 0.509),
    (0.517, 0.510),
    (0.524, 0.512),
    (0.532, 0.511),
    (0.522, 0.507),
]
const SOLLERO = [
    (0.517, 0.506),
    (0.513, 0.502),
    (0.515, 0.510),
    (0.518, 0.512),
    (0.526, 0.513),
    (0.535, 0.514),
    (0.517, 0.506),
]

glass_epoxy(ψ) = AnisotropicElasticity(lekhnitskii_params(48.26, 17.24, 6.89, 0.29; θ_deg=ψ))

function plate(ψ; nome="ke2008")
    return dual_elasticity_problem(; W=W, H=H, a=a, α=α, σ=σ,
        ndiv_b=NB, ndiv_h=NH, ndiv_crack=NCR, ordem=2,
        nome="$(nome)_psi$(Int(ψ))", props=glass_epoxy(ψ))
end

function tip_sifs(dad)
    tips = williams_tip_positions(dad)
    nodesA = crack_face_nodes(dad; face=2)
    KIs = Float64[]
    KIIs = Float64[]
    for geo in tips
        inode = nodesA[argmin(norm(dad.Nodes[i] - geo) for i in nodesA)]
        KI, KII = sif_cod_dual(dad, inode; sample=2)
        push!(KIs, KI)
        push!(KIIs, KII)
    end
    return mean(KIs), mean(KIIs), tips
end

println("="^72)
println(" Ke 2008 Example 4 — 45° centre crack, glass-epoxy, a/w=0.2, h/w=2")
println("="^72)
F1 = Float64[]
F2 = Float64[]
θs = Float64[]
println()
@printf("%6s %8s %8s %8s %8s %8s %8s %8s\n",
    "ψ°", "KI_G", "KI_COD", "KI_Sol", "KII_G", "KII_COD", "KII_Sol", "θ_MTS°")
for (i, ψ) in enumerate(PSI)
    dad = plate(ψ)
    assemble_dual!(dad; npg=NPG, threaded=false)
    solve_dual!(dad; threaded=false)
    KI, KII, tips = tip_sifs(dad)
    e1 = BEM.Crack._crack_ahead(dad, tips[argmax(p[1] for p in tips)])
    θ, _ = max_tens_circ(KI, KII, dad.properties, e1)
    push!(F1, KI / S)
    push!(F2, KII / S)
    push!(θs, rad2deg(θ))
    g, so = GANDHI[i], SOLLERO[i]
    @printf("%6.0f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.2f\n",
        ψ, g[1], KI / S, so[1], g[2], KII / S, so[2], rad2deg(θ))
end

plt1 = plot(PSI, [g[1] for g in GANDHI]; lw=2, color=:black, label="Gandhi KI",
    xlabel="fibre angle ψ (deg)", ylabel="K / σ√(πa)",
    title="Ke 2008 Table V — 45° crack", legend=:topleft, grid=true)
plot!(plt1, PSI, [g[2] for g in GANDHI]; lw=2, ls=:dash, color=:black, label="Gandhi KII")
plot!(plt1, PSI, F1; marker=:circle, lw=2, color=:dodgerblue, label="dual COD KI")
plot!(plt1, PSI, F2; marker=:square, lw=2, color=:orange, label="dual COD KII")

plt2 = plot(PSI, -θs; marker=:circle, lw=2, color=:dodgerblue,
    xlabel="fibre angle ψ (deg)", ylabel="-θ₀ (deg)",
    title="Anisotropic MTS initiation (Ke)", legend=false, grid=true)
hline!(plt2, [53.13]; ls=:dot, color=:gray, label="isotropic 45°")

# A few growth steps at ψ=0 and ψ=90 (fibre along x vs y).
println()
println("MTS growth, 45° crack, da=0.15 a, 4 increments")
hists = Dict{Float64,Vector}()
for ψ in (0.0, 90.0)
    dad = plate(ψ; nome="ke2008_grow")
    hist = propagate_dual_mts!(dad; nsteps=5, da=0.15 * a, npg=NPG, sample=2)
    hists[ψ] = hist
    println("  ψ=$ψ")
    for (k, h) in enumerate(hist)
        @printf("    step %d  KI=%.3f  KII=%.3f  θ°=%.1f\n",
            k - 1, mean(h.KI), mean(h.KII), rad2deg(mean(h.θ)))
    end
end

plt3 = plot(; xlabel="x / a", ylabel="y / a", aspect_ratio=:equal,
    title="45° crack MTS path", grid=true, legend=:topleft)
plot!(plt3, [-cos(α), cos(α)] .* (1), [-sin(α), sin(α)]; lw=2, color=:gray,
    label="initial")
for (ψ, col) in ((0.0, :dodgerblue), (90.0, :orange))
    hist = hists[ψ]
    for j in 1:2
        xs = [hist[k].tips[j][1] / a for k in eachindex(hist)]
        ys = [hist[k].tips[j][2] / a for k in eachindex(hist)]
        plot!(plt3, xs, ys; lw=2, color=col, label=j == 1 ? "ψ=$(Int(ψ))°" : false)
        scatter!(plt3, xs, ys; ms=3, color=col, label=false)
    end
end

outdir = joinpath(@__DIR__, "..", "debug")
mkpath(outdir)
out = joinpath(outdir, "ke2008_aniso.png")
plt = plot(plt1, plt2, plt3; layout=(1, 3), size=(1400, 420), margin=5Plots.mm)
savefig(plt, out)
println()
println("figure → scripts/debug/ke2008_aniso.png")
println("Done.")
