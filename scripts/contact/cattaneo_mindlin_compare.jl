# =============================================================================
# Cattaneo–Mindlin (Loyola 2022 §9.3.1) — model comparison
#   1. Analytical (Hertz + Cattaneo / Mindlin–Deresiewicz)
#   2. Half-space BEM (Flamant, node-to-node / regular NTS)
#   3. Cohesive penalty + Coulomb
#   4. Mortar (segment-to-segment) transfer on non-matching grids
# =============================================================================
using DrWatson
@quickactivate :BEM
using BEM.Contact

using LinearAlgebra
using Statistics
using Printf
using Dates

using Plots

const OUTDIR = joinpath(projectdir(), "plots", "cattaneo_mindlin")
mkpath(OUTDIR)

println("="^64)
println(" Cattaneo–Mindlin comparison (Loyola 2022 Table 9.19)")
println(" ", Dates.now())
println("="^64)

# match_thesis_ap0=true → P rescaled so a,p0 match thesis Tables 9.20–9.21
par = loyola_cattaneo_params(; Q_over_fP=0.5, match_thesis_ap0=true)
@printf("R=%.1f mm  E=%.1f MPa  ν=%.2f  P=%.1f N/mm  f=%.2f\n", par.R, par.E, par.ν, par.P, par.f)
@printf("E_eq=%.4f MPa  R_eq=%.2f mm\n", par.E_eq, par.R_eq)
@printf("Analytical: a=%.6f mm  p0=%.4f MPa  Qmax=%.4f N/mm\n", par.a, par.p0, par.Qmax)
@printf("Thesis Tables 9.20–9.21 targets: a=1.1860 mm  p0=697.8025 MPa\n")
@printf("(Table 9.19 lists P=100; matching a,p0 requires P≈%.1f — see loyola_cattaneo_params)\n", par.P)

# material for half-plane: single equivalent body with E* = E_eq
# ContactHalfPlane2D uses G, ν with Estar = 2G/(1-ν)
# We want Estar = E_eq, so set G_eff, ν_eff with 2G/(1-ν) = E_eq
ν = par.ν
G = par.E_eq * (1 - ν) / 2

# grid: cover a few contact widths
N = 201
L = 3.5 * par.a
x = collect(range(-L, L; length=N))
h = x[2] - x[1]
hp = ElasticHalfPlane2D(G, ν; h=h)

# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------
relerr(num, ref) = abs(num - ref) / max(abs(ref), eps()) * 100

function l2_err(x, y_num, y_ref; mask=nothing)
    w = ones(length(x))
    if mask !== nothing
        w .= mask
    end
    num = sqrt(sum(w .* abs2.(y_num .- y_ref)))
    den = sqrt(sum(w .* abs2.(y_ref))) + eps()
    return num / den
end

# -----------------------------------------------------------------------------
# 1) Analytical fields for all steps
# -----------------------------------------------------------------------------
println("\n[1] Analytical Cattaneo / Mindlin–Deresiewicz")
Q_path = Float64[]
ana = Dict{Symbol,Any}()
for st in par.load_steps
    push!(Q_path, st.Q)
    pA = cattaneo_pressure(x, par.a, par.p0)
    qA = mindlin_shear_history(x, par.a, par.p0, par.f, par.P, Q_path)
    c = cattaneo_c(par.a, st.Q, par.f, par.P)
    ana[st.name] = (; p=pA, q=qA, c, Q=st.Q, P=st.P)
    @printf("  step %s  Q=%7.3f  c/a=%.4f  max|q|/p0=%.4f\n",
        st.name, st.Q, (st.Q == 0 && st.name == :A) ? 1.0 : c / par.a,
        maximum(abs.(qA)) / par.p0)
end

# -----------------------------------------------------------------------------
# 2) Half-space regular (NTS) BEM
# -----------------------------------------------------------------------------
println("\n[2] Half-space regular contact (NTS + Mindlin residual)")
hs_res = Dict{Symbol,Any}()
t0 = time()
hs_hist = solve_cattaneo_history_halfplane(x, par.R_eq, par.f, hp, par.load_steps; tol=1e-9)
dt_all = time() - t0
for sol in hs_hist
    hs_res[sol.name] = sol
    @printf("  step %s  a=%.4f (err %.2f%%)  p0=%.2f (err %.2f%%)  Ft=%.3f  stick=%d slip=%d\n",
        sol.name, sol.a, relerr(sol.a, par.a), sol.p0, relerr(sol.p0, par.p0),
        sol.force_t, count(sol.stick), count(sol.slip))
end
@printf("  (history wall time %.2fs)\n", dt_all)

# -----------------------------------------------------------------------------
# 3) Cohesive with Mindlin history
# -----------------------------------------------------------------------------
println("\n[3] Cohesive (history + η-blend friction)")
coh_hist = solve_cattaneo_cohesive_history_halfplane(
    x, par.R_eq, par.f, hp, par.load_steps; η_coh=0.05, tol=1e-9)
coh_res = Dict(s.name => s for s in coh_hist)
for sol in coh_hist
    @printf("  step %s  a=%.4f (err %.2f%%)  p0=%.2f (err %.2f%%)  Ft=%.3f\n",
        sol.name, sol.a, relerr(sol.a, par.a), sol.p0, relerr(sol.p0, par.p0),
        sol.force_t)
end

# -----------------------------------------------------------------------------
# 4) Mortar STS with Mindlin history (non-matching master/slave)
# -----------------------------------------------------------------------------
println("\n[4] Mortar segment-to-segment history (refined slave → coarse master)")
N_m = 61
x_m = collect(range(-L, L; length=N_m))
mort_hist = solve_cattaneo_mortar_history_halfplane(
    x, x_m, par.R_eq, par.f, G, ν, par.load_steps; tol=1e-9)
mort_res = Dict(s.name => s for s in mort_hist)
for sol in mort_hist
    @printf("  step %s  a_s=%.4f a_m=%.4f  p0_s=%.2f p0_m=%.2f  Fn_m=%.2f Ft_m=%.3f\n",
        sol.name, sol.a_s, sol.a_m, sol.p0_s, sol.p0_m, sol.force_n_m, sol.force_t_m)
end

# -----------------------------------------------------------------------------
# Error tables (thesis style Tables 9.20–9.21)
# -----------------------------------------------------------------------------
println("\n" * "="^64)
println(" Summary vs analytical (step B unless noted)")
println("="^64)
@printf("%-12s %10s %10s %10s %10s %10s\n", "Model", "a", "err_a%", "p0", "err_p0%", "L2(q)_B")
function row(name, a, p0, q; qref=ana[:B].q)
    e_a = relerr(a, par.a)
    e_p = relerr(p0, par.p0)
    # interpolate q onto same x if needed
    l2q = length(q) == length(qref) ? l2_err(x, q, qref) : NaN
    @printf("%-12s %10.4f %10.3f %10.2f %10.3f %10.4f\n", name, a, e_a, p0, e_p, l2q)
end
row("Analytical", par.a, par.p0, ana[:B].q)
row("Half-space", hs_res[:B].a, hs_res[:B].p0, hs_res[:B].τ)
row("Cohesive", coh_res[:B].a, coh_res[:B].p0, coh_res[:B].τ)
row("Mortar-s", mort_res[:B].a_s, mort_res[:B].p0_s, mort_res[:B].p_s .* 0 .+ mort_res[:B].τ_s)
row("Mortar-m", mort_res[:B].a_m, mort_res[:B].p0_m, zeros(length(x)))  # master grid differs

# mesh convergence of half-space a, p0 (thesis Table 9.20/9.21 style)
println("\nHalf-space mesh study (step A, normal only):")
@printf("%6s %10s %10s %10s %10s\n", "N", "a", "err_a%", "p0", "err_p0%")
for Npc in (21, 41, 61, 81, 101, 201)
    xx = collect(range(-L, L; length=Npc))
    hh = xx[2] - xx[1]
    hpp = ElasticHalfPlane2D(G, ν; h=hh)
    sol = solve_cattaneo_halfplane(xx, par.R_eq, par.P, 0.0, par.f, hpp; tol=1e-9)
    @printf("%6d %10.4f %10.3f %10.2f %10.3f\n", Npc, sol.a, relerr(sol.a, par.a),
        sol.p0, relerr(sol.p0, par.p0))
end

# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------
println("\n[plots] writing to $OUTDIR")

function plot_step(step::Symbol; title="")
    xa = x ./ par.a
    ax = plot(xa, ana[step].p ./ par.p0; color=:black, linestyle=:dash, label="p ana",
        lw=2, xlabel="x/a", ylabel="traction / p0", title=title, legend=:topright,
        legendfontsize=9, xlim=(-1.5, 1.5), framestyle=:box)
    plot!(ax, xa, ana[step].q ./ par.p0; color=:black, label="q ana", lw=2)
    plot!(ax, xa, hs_res[step].p ./ par.p0; color=:dodgerblue, label="p HS")
    plot!(ax, xa, hs_res[step].τ ./ par.p0; color=:crimson, label="q HS")
    plot!(ax, xa, coh_res[step].p ./ par.p0; color=:forestgreen, linestyle=:dot, label="p coh")
    plot!(ax, xa, coh_res[step].τ ./ par.p0; color=:orange, linestyle=:dot, label="q coh")
    return ax
end

panels = [plot_step(st; title="Load step $st") for st in (:B, :C, :D, :E)]
fig1 = plot(panels...; layout=(2, 2), size=(1100, 800))
savefig(fig1, joinpath(OUTDIR, "tractions_BCDE.pdf"))
savefig(fig1, joinpath(OUTDIR, "tractions_BCDE.png"))

# mortar comparison at B
xa = x ./ par.a
xm = mort_res[:B].x_m ./ par.a
fig2 = plot(xa, ana[:B].p ./ par.p0; color=:black, linestyle=:dash, label="p ana", lw=2,
    xlabel="x/a", ylabel="p/p0, q/p0", title="Mortar STS vs analytical — step B",
    legend=:topright, legendfontsize=9, xlim=(-1.5, 1.5), size=(900, 400), framestyle=:box)
plot!(fig2, xa, ana[:B].q ./ par.p0; color=:black, label="q ana", lw=2)
plot!(fig2, xa, mort_res[:B].p_s ./ par.p0; color=:dodgerblue, label="p slave")
plot!(fig2, xa, mort_res[:B].τ_s ./ par.p0; color=:crimson, label="q slave")
scatter!(fig2, xm, mort_res[:B].p_m ./ par.p0; color=:dodgerblue, marker=:circle, markersize=6, label="p master")
scatter!(fig2, xm, mort_res[:B].τ_m ./ par.p0; color=:crimson, marker=:rect, markersize=6, label="q master")
savefig(fig2, joinpath(OUTDIR, "mortar_stepB.pdf"))
savefig(fig2, joinpath(OUTDIR, "mortar_stepB.png"))

# normal pressure only step A
fig3 = plot(xa, ana[:A].p ./ par.p0; color=:black, lw=2, label="Hertz",
    xlabel="x/a", ylabel="p/p0", title="Normal pressure — step A",
    legend=:topright, xlim=(-1.5, 1.5), size=(700, 400), framestyle=:box)
plot!(fig3, xa, hs_res[:A].p ./ par.p0; color=:dodgerblue, label="Half-space")
plot!(fig3, xa, coh_res[:A].p ./ par.p0; color=:forestgreen, linestyle=:dot, label="Cohesive")
savefig(fig3, joinpath(OUTDIR, "pressure_stepA.pdf"))

# stick/slip map step B
zone = zeros(length(x))
zone[hs_res[:B].slip] .= 2
zone[hs_res[:B].stick] .= 1
cB = ana[:B].c / par.a
fig4 = bar(xa, zone; xlabel="x/a", ylabel="zone", title="Stick / slip — step B (half-space)",
    legend=:topright, fill_z=reshape(zone, 1, :),
    color=cgrad([:white, :dodgerblue, :crimson]), linecolor=:match,
    size=(700, 350), framestyle=:box)
vline!(fig4, [-cB, cB]; color=:black, linestyle=:dash, label="±c/a ana")
vline!(fig4, [-1.0, 1.0]; color=:gray, linestyle=:dot, label="±a")
savefig(fig4, joinpath(OUTDIR, "stick_slip_B.pdf"))

println("\nDone. Figures in: $OUTDIR")
println("Models compared: analytical | half-space NTS | cohesive | mortar STS")
