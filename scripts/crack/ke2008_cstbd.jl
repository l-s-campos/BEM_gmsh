# Ke, Chen, Ku & Chen, IJNAMG 33:1227–1253 (2009) §4.2 / Figs. 18–24
# CSTBD marble: D=7.4 cm, 2a=2.2 cm, diametral compression, β=45° to the
# load, isotropy-plane ψ = 0, 30, 45, 60° (AM-4, CM-4, DM-4, EM-4).
using DrWatson
@quickactivate :BEM
using BEM
using BEM.Crack
using LinearAlgebra
using Printf
using Plots
using Statistics: mean

const R = 3.7
const a = 1.1
const β = π / 4
const PLOAD = 1.0
const NPG = 12
const NCR = 12
const NOUT = 28
const DA = 0.06 * R
const RMAX = 0.88 * R
const PSI = (0.0, 30.0, 45.0, 60.0)
const NAMES = ("AM-4", "CM-4", "DM-4", "EM-4")

function run_one(ψ; nome="cstbd")
    dad = cstbd_problem(; R=R, a=a, β=β, ψ=ψ, p=PLOAD,
        ndiv_outer=NOUT, ndiv_crack=NCR, ndiv_load=2, ordem=2,
        nome="$(nome)_psi$(Int(ψ))")
    hist = propagate_dual_mts!(dad; nsteps=14, da=DA, npg=NPG, sample=2,
        rmax=RMAX, threaded=false)
    return dad, hist
end

println("="^72)
println(" Ke 2008 CSTBD marble  D=$(2R) cm  2a=$(2a) cm  β=45°")
println("="^72)

# Mesh / SIF sanity: isotropic β=0 (crack // load → mode I).
println()
println("isotropic β=0 (mode I check)")
props_iso = Elasticity(78.3, 0.267, 1.0; plane_strain=false)
dad0 = cstbd_problem(; R=R, a=a, β=0.0, p=PLOAD, ndiv_outer=NOUT,
    ndiv_crack=NCR, ordem=2, nome="cstbd_iso0", props=props_iso)
assemble_dual!(dad0; npg=NPG, threaded=false)
solve_dual!(dad0; threaded=false)
tips0 = williams_tip_positions(dad0)
nodesA = crack_face_nodes(dad0; face=2)
KI0, KII0 = sif_cod_dual(dad0, nodesA[argmin(norm(dad0.Nodes[i] - tips0[1]) for i in nodesA)]; sample=2)
@printf("  KI=%.4f  KII=%.4f  KII/KI=%.3f\n", KI0, KII0, KII0 / (abs(KI0) + eps()))

hists = Dict{Float64,Vector}()
for (ψ, name) in zip(PSI, NAMES)
    println()
    println("$name  ψ=$(ψ)°  β=45°")
    _, hist = run_one(ψ; nome="ke_cstbd")
    hists[ψ] = hist
    h0 = hist[1]
    @printf("  nsteps=%d  KI=%.3f  KII=%.3f  θ°=%.1f  r_tip/R=%.3f → %.3f\n",
        length(hist), mean(h0.KI), mean(h0.KII), rad2deg(mean(h0.θ)),
        mean(norm(p) for p in h0.tips) / R,
        mean(norm(p) for p in hist[end].tips) / R)
end

θc = range(0, 2π; length=181)
circx, circy = R .* cos.(θc), R .* sin.(θc)
sβ, cβ = sin(β), cos(β)
plt = plot(layout=(2, 2), size=(900, 900), margin=4Plots.mm)
for (k, ψ) in enumerate(PSI)
    hist = hists[ψ]
    plot!(plt, circx, circy; subplot=k, lw=1, color=:black, label=false,
        aspect_ratio=:equal, xlims=(-1.15R, 1.15R), ylims=(-1.15R, 1.15R),
        xlabel="x (cm)", ylabel="y (cm)",
        title="$(NAMES[k])  ψ=$(Int(ψ))°  β=45°", grid=false)
    plot!(plt, [-a * sβ, a * sβ], [-a * cβ, a * cβ]; subplot=k, lw=2,
        color=:gray, label=k == 1 ? "initial" : false)
    for j in 1:2
        xs = [hist[i].tips[j][1] for i in eachindex(hist)]
        ys = [hist[i].tips[j][2] for i in eachindex(hist)]
        plot!(plt, xs, ys; subplot=k, lw=2, color=:red,
            label=(k == 1 && j == 1) ? "MTS dual BEM" : false)
        scatter!(plt, xs, ys; subplot=k, ms=3, color=:red, label=false)
    end
end

outdir = joinpath(@__DIR__, "..", "debug")
mkpath(outdir)
out = joinpath(outdir, "ke2008_cstbd.png")
savefig(plt, out)
println()
println("figure → scripts/debug/ke2008_cstbd.png")
println("Done.")
