# Paper 1 Fig. 8: on-axis von Mises vs depth during pin-on-disc wear,
# for isotropic Coulomb μ ∈ {0, 0.15, 0.25, 0.35, 0.50, 0.65}.
#
# Identical bodies ⇒ K = 0 so p is independent of μ; isotropic Archard uses
# |p_n| Δs, so one force-controlled wear run is enough. Full sliding is then
# applied as q_x = −μ p_n (disc travel in +x) and σ_VM(0,0,z) is superposed.
# Replot:  $env:REPLOT='true'
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

const SA0 = [0.0, 0.13e3, 0.26e3, 0.53e3, 1.06e3, 2.13e3, 4.26e3]
const SA0_LBL = ["0", "0.13", "0.26", "0.53", "1.06", "2.13", "4.26"]
const MU = [0.0, 0.15, 0.25, 0.35, 0.50, 0.65]
const MU_LBL = ["0.0", "0.15", "0.25", "0.35", "0.50", "0.65"]
const PANEL = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]

# Fig. 8 series style
const COL = ["#000000", "#007E33", "#0072BD", "#EDB120", "#4DBEEE", "#A2142F", "#0072BD"]
const MK = [:rect, :circle, :none, :plus, :x, :star5, :none]
const LS = [:solid, :solid, :dot, :solid, :solid, :solid, :dash]

const TSV = joinpath(FIG, "argatov_vm.tsv")

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

function axis_vm(zs, pn, μ, x, y, hs, ν)
    px = μ == 0 ? fill!(similar(pn), 0) : (-μ) .* pn
    py = fill!(similar(pn), 0)
    σ = zeros(length(zs))
    @inbounds for k in eachindex(zs)
        s = subsurface_stress(0.0, 0.0, zs[k], px, py, pn, x, y, hs, ν)
        σ[k] = s.VM
    end
    return σ
end

if REPLOT
    data = readdlm(TSV, '\t'; header=true)
    M, _ = data
    @printf "replot from %s  (%d rows)\n" TSV size(M, 1)
else
    E, ν, R, k = PIN.E, PIN.ν, PIN.R, PIN.i
    G = E / (2(1 + ν))
    x, hs = square_mesh(N, PIN.L, G, ν, G, ν)
    Estar = contact_modulus(hs)
    hz = hertz_sphere(R, PIN.δ, Estar)
    F, a0, δ0, p0 = hz.P, hz.a, PIN.δ, hz.p0
    @printf "Hertz a0=%.4f mm  F=%.3f N  p0=%.1f MPa  N=%d  nsteps=%d\n" a0 F p0 N nsteps

    grid = make_grid(x, x, hs, sphere_gap(x, x, R))
    prep = precompute_kernels(N, N, hs)
    st = init_state(grid)
    hist = sliding_wear_force_steps!(
        st, grid, prep, isotropic_law(0.0, k), F, Δs, nsteps;
        δ0=δ0, tol=3e-6, maxiter=300, rtol=2e-3, maxouter=10, relax=0.45,
        snapshot_s=SA0 .* a0,
    )
    @printf "captured %d/%d snapshots\n" length(hist.snaps) length(SA0)

    zs = a0 .* vcat(0.005, collect(range(0.04, 3.0; length=60)))
    open(TSV, "w") do io
        println(io, "mu\tsnap\ts_mm\tz_over_a0\tVM_over_p0")
        for (im, μ) in enumerate(MU)
            @printf "  μ = %.2f\n" μ
            for (is, sn) in enumerate(hist.snaps)
                σ = axis_vm(zs, sn.pn2d, μ, sn.x, sn.y, hs, ν)
                for k in eachindex(zs)
                    @printf io "%.4f\t%d\t%.6f\t%.8e\t%.8e\n" μ is sn.s zs[k] / a0 σ[k] / p0
                end
                @printf "    snap %d  s/a0=%.0f  σVM(z=0.48a)/p0=%.3f  σVM_max/p0=%.3f\n" is sn.s / a0 σ[argmin(abs.(zs ./ a0 .- 0.48))] / p0 maximum(σ) / p0
            end
        end
    end
    println("wrote ", TSV)
    # Hertz on-axis check at s=0, μ=0
    _, _, σVM_ana = hertz_axis_stress(0.48 * a0, a0, p0, ν)
    @printf "Hertz ana σVM(0.48 a0)/p0 = %.3f\n" σVM_ana / p0
    data = readdlm(TSV, '\t'; header=true)
    M, _ = data
end

default(linewidth=1.7, legendfontsize=6, tickfontsize=9,
        guidefontsize=11, grid=false, markerstrokewidth=0.4,
        markersize=3.2, foreground_color_legend=nothing,
        background_color_legend=:white)

plts = Vector{Any}(undef, length(MU))
for (im, μ) in enumerate(MU)
    rows_μ = findall(r -> M[r, 1] ≈ μ, 1:size(M, 1))
    p = plot(title="μ = $(MU_LBL[im])",
             xlabel="z / a0    $(PANEL[im])", ylabel="σVM / p0",
             xlims=(0, 3), ylims=(1e-2, 1.0),
             yscale=:log10, yticks=([0.01, 0.1, 1.0], ["0.01", "0.1", "1"]),
             xticks=0:0.5:3,
             legend=:topright)
    for is in 1:length(SA0)
        rows = filter(r -> Int(M[r, 2]) == is, rows_μ)
        isempty(rows) && continue
        z = M[rows, 4]
        vm = M[rows, 5]
        lbl = is == 1 ? "s / a0 = 0" : "s / a0 = $(SA0_LBL[is])·10³"
        plot!(p, z, vm;
              color=COL[is], ls=LS[is], marker=MK[is],
              markeralpha=0.9, label=lbl)
    end
    plts[im] = p
end

plt = plot(plts...; layout=(3, 2), size=(900, 1050), dpi=160)
_savefig(plt, joinpath(FIG, "argatov_vm.pdf"))
_savefig(plt, joinpath(FIG, "argatov_vm.png"))
