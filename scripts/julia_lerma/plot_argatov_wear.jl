# Paper 1 Fig. 6: pin-on-disc wear vs Argatov, Wear 271 (2011) 1147–1155
# and Hegadekatte et al. (2006). Constant load; units N, mm, MPa.
# Paper §4.1.1: 1.6 mm × 1.6 mm, 81×81, s = 640 mm, Δs = 1 mm.
using DrWatson
@quickactivate :BEM
using BEM.Contact
using Plots
using Printf
using LinearAlgebra

const FIG = plotsdir("julia_lerma")
mkpath(FIG)
get(ENV, "WEAR_VERBOSE", nothing) === nothing && (ENV["WEAR_VERBOSE"] = "true")

N = parse(Int, get(ENV, "N", "41"))
nsteps = parse(Int, get(ENV, "NSTEPS", "80"))
s_end = 640.0                          # mm
Δs = s_end / nsteps

E, ν, R, k = PIN.E, PIN.ν, PIN.R, PIN.i
G = E / (2(1 + ν))
x, hs = square_mesh(N, PIN.L, G, ν, G, ν)
Estar = contact_modulus(hs)
hz = hertz_sphere(R, PIN.δ, Estar)
# Paper quotes P = 10.2 N; Hertz from δ = 0.45 μm is ~10.39 N. Use Hertz so
# a0 = √(Rδ) = 0.15 mm matches the Fig. 6 normalisation.
F = hz.P
a0, δ0, p0 = hz.a, PIN.δ, hz.p0
@printf "Hertz a0=%.4f mm  F=%.3f N  δ0=%.4f μm  p0=%.1f MPa  E*=%.0f MPa\n" a0 F 1e3 * δ0 p0 Estar
@printf "mesh N=%d  hx=%.4f mm  nsteps=%d  Δs=%.2f mm  (paper 81×81, Δs=1 mm)\n" N hs.hx nsteps Δs

grid = make_grid(x, x, hs, sphere_gap(x, x, R))
prep = precompute_kernels(N, N, hs)
st = init_state(grid)
law = isotropic_law(0.0, k)
hist = sliding_wear_force_steps!(
    st, grid, prep, law, F, Δs, nsteps;
    δ0=δ0, tol=3e-6, maxiter=300, rtol=2e-3, maxouter=10, relax=0.45,
)

s_num = hist.s
w_num = hist.wmax
a_num = hist.a
pmean_num = hist.pmean
pmax_num = hist.pmax
s_ana = range(0.0, s_end; length=400)
ag = argatov2011(collect(s_ana), a0, R, k, F)
w_heg = [hegadekatte_wear_depth(si, R, k, F) for si in s_ana]
ag_n = argatov2011(s_num, a0, R, k, F)
p_arg = @. (F / (π * ag.a^2)) / p0          # uniform P/(πa²) / Hertz p0
p_arg_n = @. (F / (π * ag_n.a^2)) / p0

@printf "\n--- Paper 1 Fig. 6 end point s=%.0f mm ---\n" s_num[end]
@printf "  w/R     num=%.4e  Argatov H/R=%.4e  Hegadekatte=%.4e\n" w_num[end] / R ag_n.H[end] / R w_heg[end] / R
@printf "  a/a0    num=%.3f  Argatov=%.3f\n" a_num[end] / a0 ag_n.a[end] / a0
@printf "  pmean/p0 num=%.3f  Argatov P/(πa²)/p0=%.3f  pmax/p0=%.3f\n" pmean_num[end] / p0 p_arg_n[end] pmax_num[end] / p0
@printf "  w       num=%.3f μm  Argatov H=%.3f μm  Heg=%.3f μm\n" 1e3 * w_num[end] 1e3 * ag_n.H[end] 1e3 * w_heg[end]
@printf "  load    P=%.3f N  (target %.3f, drift %.2f%%)\n" hist.P[end] F 100 * abs(hist.P[end] - F) / F

tsv = joinpath(FIG, "argatov_wear_depth.tsv")
open(tsv, "w") do io
    println(io, "s_mm\tw_num_mm\tH_arg_mm\tH0_arg_mm\ta_num_mm\ta_arg_mm\tpmean_MPa\tpmax_MPa\tp_arg_MPa\tP_N")
    for i in eachindex(s_num)
        @printf io "%.6f\t%.8e\t%.8e\t%.8e\t%.8e\t%.8e\t%.8e\t%.8e\t%.8e\t%.8e\n" s_num[i] w_num[i] ag_n.H[i] ag_n.H0[i] a_num[i] ag_n.a[i] pmean_num[i] pmax_num[i] F / (π * ag_n.a[i]^2) hist.P[i]
    end
end
println("wrote ", tsv)

default(linewidth=2, legendfontsize=9, tickfontsize=10,
        guidefontsize=11, grid=true, gridalpha=0.25,
        markerstrokewidth=0)

# Paper Fig. 6 uses ~20 markers; keep a thin line of every step.
mark = max(1, length(s_num) ÷ 20)

plt = plot(layout=(3, 1), size=(620, 900), dpi=150, link=:x)

plot!(plt[1], s_ana ./ a0, ag.H ./ R;
      color=:black, ls=:solid, label="Argatov (2011)")
plot!(plt[1], s_ana ./ a0, w_heg ./ R;
      color=:gray, ls=:dash, label="Hegadekatte et al. (2006)")
plot!(plt[1], s_num ./ a0, w_num ./ R;
      color=:crimson, ls=:solid, lw=1.2, label="computed")
scatter!(plt[1], s_num[1:mark:end] ./ a0, w_num[1:mark:end] ./ R;
         color=:crimson, ms=5, label="")
ylabel!(plt[1], "w / R")
title!(plt[1], "(a)")
ylims!(plt[1], 0, 5.5e-5)

plot!(plt[2], s_ana ./ a0, ag.a ./ a0;
      color=:black, ls=:solid, label="Argatov (2011)")
plot!(plt[2], s_num ./ a0, a_num ./ a0;
      color=:crimson, ls=:solid, lw=1.2, label="computed")
scatter!(plt[2], s_num[1:mark:end] ./ a0, a_num[1:mark:end] ./ a0;
         color=:crimson, ms=5, label="")
ylabel!(plt[2], "a / a0")
title!(plt[2], "(b)")
ylims!(plt[2], 1, 3.6)

plot!(plt[3], s_ana ./ a0, p_arg;
      color=:black, ls=:solid, label="Argatov (2011)")
plot!(plt[3], s_num ./ a0, pmean_num ./ p0;
      color=:crimson, ls=:solid, lw=1.2, label="computed")
scatter!(plt[3], s_num[1:mark:end] ./ a0, pmean_num[1:mark:end] ./ p0;
         color=:crimson, ms=5, label="")
ylabel!(plt[3], "p / p0")
xlabel!(plt[3], "s / a0")
title!(plt[3], "(c)")
ylims!(plt[3], 0, 1.05)

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

_savefig(plt, joinpath(FIG, "argatov_fig6.pdf"))
_savefig(plt, joinpath(FIG, "argatov_fig6.png"))

plt_w = plot(s_ana, 1e3 .* ag.H;
             color=:black, ls=:solid, label="Argatov (2011) H = a²/(2R)",
             xlabel="sliding distance s (mm)", ylabel="wear depth (μm)",
             title="Pin-on-disc wear depth — Argatov, Wear 271 (2011)",
             size=(720, 480), dpi=150)
plot!(plt_w, s_ana, 1e3 .* w_heg;
      color=:gray, ls=:dash, label="Hegadekatte spherical cap")
plot!(plt_w, s_num, 1e3 .* w_num;
      color=:crimson, ls=:solid, lw=1.2, label="numerical (this work)")
scatter!(plt_w, s_num[1:mark:end], 1e3 .* w_num[1:mark:end];
         color=:crimson, ms=5, label="")
_savefig(plt_w, joinpath(FIG, "argatov_wear_depth.pdf"))
_savefig(plt_w, joinpath(FIG, "argatov_wear_depth.png"))
