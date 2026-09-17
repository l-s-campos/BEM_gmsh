# Erdogan & Sih, J. Basic Eng. 85:519–525 (1963) — MTS kink of an inclined
# centre crack under uniaxial tension. Dual COD and XBEM SIFs through
# max_tens_circ vs the infinite-plate MTS curve and the plexiglass
# initiation angles (Sih, Int. J. Fract. 10:305–321, 1974, Table 1).
#
# β = angle between the crack and the tensile axis (90° = mode I).
# Specimen: 229 × 457 mm sheet, 2a = 50.8 mm (Erdogan–Sih plexiglass).
using DrWatson
@quickactivate :BEM
using BEM
using BEM.Crack
using LinearAlgebra
using Printf
using Plots
using Statistics: mean

const σ = 1.0
const a = 1.0
# 229/50.8 ≈ 4.508, 457/50.8 ≈ 9.0
const W = 229 / 50.8
const H = 457 / 50.8
const NPG = 12
const NCR = 10
const NB = 8
const NH = 14
const NENR = 3
const NV = 9

# Sih (1974) Table 1, |θ₀| as negative kink from the crack plane.
const ERDOGAN_SIH_β = [30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
const ERDOGAN_SIH_θ = [-62.4, -55.6, -51.1, -43.1, -30.7, -17.3]

erdogan_α(β_deg) = deg2rad(90 - β_deg)

function mesh_inclined(β_deg; nome="erdogan_sih")
    α = erdogan_α(β_deg)
    return dual_elasticity_problem(; W=W, H=H, a=a, α=α, E=1.0, ν=0.33, σ=σ,
        ndiv_b=NB, ndiv_h=NH, ndiv_crack=NCR, plane_strain=false, ordem=2,
        nome="$(nome)_b$(Int(round(β_deg)))")
end

function mean_pair(KIs, KIIs)
    θs = [max_tens_circ(KIs[k], KIIs[k])[1] for k in eachindex(KIs)]
    return mean(KIs), mean(KIIs), mean(θs)
end

function cod_sifs(dad)
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
    return mean_pair(KIs, KIIs)
end

function xbem_sifs(dad)
    _, KI, KII = solve_xbem!(dad; n_enr=NENR, npg=NPG, n_v=NV, threaded=false)
    return mean_pair(KI, KII)
end

println("="^72)
println(" Erdogan–Sih 1963 — MTS initiation (dual COD vs XBEM)")
println(" plate 2W=$(round(2W; digits=2))  2H=$(round(2H; digits=2))  2a=$(2a)")
println("="^72)

β_sweep = collect(30.0:10.0:90.0)
θ_cod = Float64[]
θ_xbem = Float64[]
println()
@printf("%6s %7s %7s %7s %7s %7s %8s %8s %8s %8s\n",
    "β°", "KI_ana", "KI_COD", "KI_XBEM", "KII_ana", "KII_COD", "KII_XBEM",
    "θ_ana°", "θ_COD°", "θ_XBEM°")
for β in β_sweep
    dad = mesh_inclined(β)
    assemble_dual!(dad; npg=NPG, threaded=false)
    solve_dual!(dad; threaded=false)
    KIc, KIIc, θc = cod_sifs(dad)
    KIx, KIIx, θx = xbem_sifs(dad)
    α = erdogan_α(β)
    KIa, KIIa = analytical_sif_inclined_center(σ, a, α)
    θa, _ = max_tens_circ(KIa, KIIa)
    push!(θ_cod, rad2deg(θc))
    push!(θ_xbem, rad2deg(θx))
    @printf("%6.0f %7.3f %7.3f %7.3f %7.3f %7.3f %8.3f %8.2f %8.2f %8.2f\n",
        β, KIa, KIc, KIx, KIIa, KIIc, KIIx, rad2deg(θa), rad2deg(θc), rad2deg(θx))
end

β_th = range(20.0, 90.0; length=71)
θ_th = [rad2deg(max_tens_circ(analytical_sif_inclined_center(σ, a, erdogan_α(β))...)[1])
        for β in β_th]

plt1 = plot(β_th, -θ_th; lw=2, color=:black, label="MTS (infinite plate)",
    xlabel="crack angle β (deg)", ylabel="-θ₀ (deg)",
    title="Erdogan–Sih 1963 — MTS initiation",
    legend=:topright, grid=true)
scatter!(plt1, ERDOGAN_SIH_β, -ERDOGAN_SIH_θ; ms=6, color=:red, marker=:diamond,
    label="Plexiglass (Erdogan–Sih)")
scatter!(plt1, β_sweep, -θ_cod; ms=7, color=:dodgerblue, marker=:circle,
    label="dual COD + MTS")
scatter!(plt1, β_sweep, -θ_xbem; ms=7, color=:orange, marker=:square,
    label="XBEM + MTS")
scatter!(plt1, [90.0], [0.0]; ms=5, color=:red, marker=:diamond, label=false)

println()
println("MTS growth from dual COD, β=45°, da=0.2 a (XBEM only on the initial crack:")
println("  Williams columns after a kinked increment are not used)")
dad_cod = mesh_inclined(45.0; nome="erdogan_sih_grow_cod")
hist_cod = propagate_dual_mts!(dad_cod; nsteps=6, da=0.2 * a, npg=NPG, sample=2)
for (k, h) in enumerate(hist_cod)
    @printf("  step %d  KI=%.3f  KII=%.3f  θ°=%.1f\n",
        k - 1, mean(h.KI), mean(h.KII), rad2deg(mean(h.θ)))
end

plt2 = plot(; xlabel="x / a", ylabel="y / a", aspect_ratio=:equal,
    title="β=45° MTS path (dual COD)", grid=true, legend=:topleft)
α45 = erdogan_α(45)
plot!(plt2, [-cos(α45), cos(α45)], [-sin(α45), sin(α45)]; lw=2, color=:gray,
    label="initial crack")
for j in 1:2
    xs = [hist_cod[k].tips[j][1] / a for k in eachindex(hist_cod)]
    ys = [hist_cod[k].tips[j][2] / a for k in eachindex(hist_cod)]
    plot!(plt2, xs, ys; lw=2, color=:dodgerblue, label=j == 1 ? "dual COD" : false)
    scatter!(plt2, xs, ys; ms=4, color=:dodgerblue, label=false)
end

outdir = joinpath(@__DIR__, "..", "debug")
mkpath(outdir)
out = joinpath(outdir, "erdogan_sih_mts.png")
plt = plot(plt1, plt2; layout=(1, 2), size=(1100, 480), margin=5Plots.mm)
savefig(plt, out)
println()
println("figure → scripts/debug/erdogan_sih_mts.png")
println("Done.")
