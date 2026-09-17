# Paper 1 pin-on-disc worn profiles and pressure (Fig. 7-style).
# Centreline z/R = (sphere gap + Archard wear)/R and p_n/p0 at the paper's
# s/a0 stations. Disc stays at z = 0 (combined wear shown on the pin).
# Replot without resolving:  $env:REPLOT='true'
using DrWatson
@quickactivate :BEM
using BEM.Contact
using Plots
using Printf
using LinearAlgebra
using DelimitedFiles

const FIG = plotsdir("julia_lerma")
mkpath(FIG)
get(ENV, "WEAR_VERBOSE", nothing) === nothing && (ENV["WEAR_VERBOSE"] = "true")
const REPLOT = get(ENV, "REPLOT", "false") == "true"

N = parse(Int, get(ENV, "N", "41"))
nsteps = parse(Int, get(ENV, "NSTEPS", "80"))
s_end = 640.0
Δs = s_end / nsteps

# Paper stations: s/a0 = 0, 0.13, 0.26, 0.53, 1.06, 2.13, 4.26 × 10³
const SA0 = [0.0, 0.13e3, 0.26e3, 0.53e3, 1.06e3, 2.13e3, 4.26e3]
const SA0_LBL = ["0", "0.13", "0.26", "0.53", "1.06", "2.13", "4.26"]
const COL = ["#0072BD", "#D95319", "#EDB120", "#7E2F8E", "#77AC30", "#4DBEEE", "#A2142F", "#0072BD"]
const MK = [:none, :rect, :circle, :circle, :diamond, :plus, :x, :none]
const LS = [:solid, :dot, :solid, :dot, :solid, :solid, :solid, :dash]

function _savefig(plt, path)
    try
        isfile(path) && rm(path; force=true)
        savefig(plt, path)
        println("wrote ", path)
    catch e
        alt = replace(path, r"(\.[^.]+)$" => s"_new\1")
        @warn "savefig failed" path exception=e alt
        savefig(plt, alt)
        println("wrote ", alt)
    end
end

function load_snaps_tsv(path)
    data = readdlm(path, '\t'; header=true)
    M, hdr = data
    id = Int.(M[:, 1])
    snaps = NamedTuple[]
    for k in unique(id)
        rows = findall(==(k), id)
        push!(snaps, (; s=M[rows[1], 2], xa=M[rows, 4], zR=M[rows, 5], pp=M[rows, 6]))
    end
    return snaps
end

if REPLOT
    tsv = joinpath(FIG, "argatov_profiles.tsv")
    snaps = load_snaps_tsv(tsv)
    @printf "replot %d snapshots from %s\n" length(snaps) tsv
else
    E, ν, R, k = PIN.E, PIN.ν, PIN.R, PIN.i
    G = E / (2(1 + ν))
    x, hs = square_mesh(N, PIN.L, G, ν, G, ν)
    Estar = contact_modulus(hs)
    hz = hertz_sphere(R, PIN.δ, Estar)
    F, a0, δ0, p0 = hz.P, hz.a, PIN.δ, hz.p0
    @printf "Hertz a0=%.4f mm  F=%.3f N  p0=%.1f MPa  N=%d  nsteps=%d  Δs=%.2f mm\n" a0 F p0 N nsteps Δs

    grid = make_grid(x, x, hs, sphere_gap(x, x, R))
    prep = precompute_kernels(N, N, hs)
    st = init_state(grid)
    law = isotropic_law(0.0, k)
    hist = sliding_wear_force_steps!(
        st, grid, prep, law, F, Δs, nsteps;
        δ0=δ0, tol=3e-6, maxiter=300, rtol=2e-3, maxouter=10, relax=0.45,
        snapshot_s=SA0 .* a0,
    )
    @printf "captured %d/%d centreline snapshots\n" length(hist.snaps) length(SA0)
    for (i, sn) in enumerate(hist.snaps)
        @printf "  snap %d  s=%.1f mm  s/a0=%.0f  (target %.0f)  wmax/R=%.3e  pmax/p0=%.3f\n" i sn.s sn.s / a0 SA0[min(i, end)] maximum(sn.w) / R maximum(sn.pn) / p0
    end
    snaps = [(; s=sn.s, xa=sn.x ./ a0, zR=(sn.gap .+ sn.w) ./ R, pp=sn.pn ./ p0)
             for sn in hist.snaps]
    open(joinpath(FIG, "argatov_profiles.tsv"), "w") do io
        println(io, "snap\ts_mm\ts_over_a0\tx_over_a0\tz_over_R\tp_over_p0")
        for (i, sn) in enumerate(snaps)
            for j in eachindex(sn.xa)
                @printf io "%d\t%.6f\t%.6f\t%.8e\t%.8e\t%.8e\n" i sn.s sn.s / a0 sn.xa[j] sn.zR[j] sn.pp[j]
            end
        end
    end
    println("wrote ", joinpath(FIG, "argatov_profiles.tsv"))
end

default(linewidth=1.8, legendfontsize=8, tickfontsize=10,
        guidefontsize=12, grid=false, markerstrokewidth=0.5,
        markersize=4.5, legendfonthalign=:left)

# scale z/R by 1e5 so the axis matches the paper's ×10^{-5} panel
xa_disc = [-4.6, 4.6]
plt_a = plot(xa_disc, [0.0, 0.0];
             color=COL[1], ls=:solid, label="Disc surface",
             xlabel="x / a0", ylabel="z / R", title="(a)",
             xlims=(-4.5, 4.5), ylims=(0, 12.8),
             yticks=0:2:12, xticks=-4:2:4,
             legend=:top, foreground_color_legend=nothing,
             background_color_legend=:white, legendfontsize=7)
annotate!(plt_a, -4.45, 12.45, text("×10⁻⁵", :left, 9))
for (i, sn) in enumerate(snaps)
    lbl = i == 1 ?
          "Pin surface: s / a0 = 0" :
          "Pin surface: s / a0 = $(SA0_LBL[i])·10³"
    plot!(plt_a, sn.xa, 1e5 .* sn.zR;
          color=COL[i + 1], ls=LS[i + 1], marker=MK[i + 1],
          markeralpha=0.95, label=lbl)
end

x_hz = range(-1.001, 1.001; length=400)
p_hz = sqrt.(max.(1 .- collect(x_hz) .^ 2, 0.0))
plt_b = plot(x_hz, p_hz;
             color=COL[1], ls=:solid, label="Hertz",
             xlabel="x / a0", ylabel="pn / p0", title="(b)",
             xlims=(-4.5, 4.5), ylims=(0, 1.2),
             xticks=-4:2:4, yticks=0:0.25:1.0,
             legend=:top, foreground_color_legend=nothing,
             background_color_legend=:white, legendfontsize=8)
for (i, sn) in enumerate(snaps)
    lbl = i == 1 ? "s / a0 = 0" : "s / a0 = $(SA0_LBL[i])·10³"
    plot!(plt_b, sn.xa, sn.pp;
          color=COL[i + 1], ls=LS[i + 1], marker=MK[i + 1],
          markeralpha=0.95, label=lbl)
end

plt = plot(plt_a, plt_b; layout=(1, 2), size=(1150, 520), dpi=170)
_savefig(plt, joinpath(FIG, "argatov_profiles.pdf"))
_savefig(plt, joinpath(FIG, "argatov_profiles.png"))
