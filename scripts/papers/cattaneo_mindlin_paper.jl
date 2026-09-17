# =============================================================================
# Cattaneo–Mindlin paper figures & tables
#   Formulations: analytical | half-space NTS | cohesive | mortar STS
#   Three discretisations N ∈ {61, 121, 241}
# Output → artigos/escritos/2026/BEM_contact
# =============================================================================
using DrWatson
@quickactivate :BEM
using BEM.Contact

using LinearAlgebra
using Statistics
using Printf
using Dates
using Plots
using LaTeXStrings

const OUTDIR = raw"C:\Users\lucas.s.campos\OneDrive\artigos\escritos\2026\BEM_contact"
const FIGDIR = joinpath(OUTDIR, "figures")
mkpath(FIGDIR)

println("="^72)
println(" Cattaneo–Mindlin paper benchmark")
println(" ", Dates.now())
println("="^72)

# Thesis-matching Hertz peak (Loyola Tables 9.20–9.21)
par = loyola_cattaneo_params(; Q_over_fP = 0.5, match_thesis_ap0 = true)
ν = par.ν
G = par.E_eq * (1 - ν) / 2
L = 3.5 * par.a

@printf("R=%.1f mm  E=%.1f MPa  ν=%.2f  P=%.1f N/mm  f=%.2f\n", par.R, par.E, par.ν, par.P, par.f)
@printf("E_eq=%.4f MPa  R_eq=%.2f mm  a=%.6f mm  p0=%.4f MPa\n", par.E_eq, par.R_eq, par.a, par.p0)

relerr(num, ref) = abs(num - ref) / max(abs(ref), eps()) * 100

function l2_rel(y_num, y_ref)
    den = sqrt(sum(abs2, y_ref)) + eps()
    return sqrt(sum(abs2, y_num .- y_ref)) / den
end

function interp_to(x_src, y_src, x_tgt)
    # piecewise-linear interpolation (grids are sorted)
    n = length(x_tgt)
    out = similar(y_src, n)
    j = 1
    @inbounds for i in 1:n
        xi = x_tgt[i]
        while j < length(x_src) - 1 && x_src[j + 1] < xi
            j += 1
        end
        x0, x1 = x_src[j], x_src[j + 1]
        t = (xi - x0) / max(x1 - x0, eps())
        out[i] = (1 - t) * y_src[j] + t * y_src[j + 1]
    end
    return out
end

# -----------------------------------------------------------------------------
# Fine reference grid for plots / L2 (N_ref)
# -----------------------------------------------------------------------------
const N_REF = 401
const N_MESHES = (61, 121, 241)   # three discretisations
x_ref = collect(range(-L, L; length = N_REF))

# Analytical fields on reference grid for every load step
Q_path = Float64[]
ana = Dict{Symbol,Any}()
for st in par.load_steps
    push!(Q_path, st.Q)
    pA = cattaneo_pressure(x_ref, par.a, par.p0)
    qA = mindlin_shear_history(x_ref, par.a, par.p0, par.f, par.P, Q_path)
    c = cattaneo_c(par.a, st.Q, par.f, par.P)
    ana[st.name] = (; p = pA, q = qA, c, Q = st.Q, P = st.P)
end

# -----------------------------------------------------------------------------
# Run all formulations × 3 meshes (report on load history, focus step B)
# -----------------------------------------------------------------------------
rows = NamedTuple[]
sols = Dict{Tuple{Symbol,Int},Any}()   # (method, N) → solution at B (+ history if any)

function push_row!(method, N, a, p0, q_on_ref, t_build; step = :B)
    e_a = relerr(a, par.a)
    e_p = relerr(p0, par.p0)
    l2p = l2_rel(interp_to(x_ref, ana[step].p, x_ref), ana[step].p)  # 0 on ana
    # q may be defined on a different grid — interpolate to x_ref
    q_ref = ana[step].q
    if length(q_on_ref) == length(x_ref)
        l2q = l2_rel(q_on_ref, q_ref)
        l2p = l2_rel(q_on_ref .* 0 .+ (method == :analytical ?
              ana[step].p : q_on_ref), ana[step].p)  # overwritten below when p given
    else
        l2q = NaN
    end
    # recompute properly if caller passed both via sols later
    push!(rows, (; method = String(method), N, step = String(step),
        a, err_a_pct = e_a, p0, err_p0_pct = e_p, L2_q = l2q, time_s = t_build))
end

# --- Analytical (mesh-independent, store once) ---
println("\n[1] Analytical")
for st in (:A, :B, :C, :D, :E)
    push!(rows, (; method = "analytical", N = N_REF, step = String(st),
        a = par.a, err_a_pct = 0.0, p0 = par.p0, err_p0_pct = 0.0,
        L2_q = 0.0, time_s = 0.0))
end

# --- Half-space NTS + Mindlin residual ---
println("\n[2] Half-space NTS")
hs_by_N = Dict{Int,Any}()
for N in N_MESHES
    x = collect(range(-L, L; length = N))
    h = x[2] - x[1]
    hp = ElasticHalfPlane2D(G, ν; h = h)
    t0 = time()
    hist = solve_cattaneo_history_halfplane(x, par.R_eq, par.f, hp, par.load_steps; tol = 1e-9)
    dt = time() - t0
    hs_by_N[N] = (; x, hist = Dict(s.name => s for s in hist), dt)
    for sol in hist
        q_refg = interp_to(x, sol.τ, x_ref)
        p_refg = interp_to(x, sol.p, x_ref)
        l2q = l2_rel(q_refg, ana[sol.name].q)
        l2p = l2_rel(p_refg, ana[sol.name].p)
        push!(rows, (; method = "halfspace", N, step = String(sol.name),
            a = sol.a, err_a_pct = relerr(sol.a, par.a),
            p0 = sol.p0, err_p0_pct = relerr(sol.p0, par.p0),
            L2_q = l2q, L2_p = l2p, time_s = dt / length(hist)))
        @printf("  N=%3d  %s  a=%.4f (%.2f%%)  p0=%.2f (%.2f%%)  L2(q)=%.3e\n",
            N, sol.name, sol.a, relerr(sol.a, par.a), sol.p0, relerr(sol.p0, par.p0), l2q)
    end
    sols[(:halfspace, N)] = hs_by_N[N]
end

# --- Cohesive (penalty / CZM limit) with Mindlin history ---
println("\n[3] Cohesive penalty (history)")
coh_by_N = Dict{Int,Any}()
for N in N_MESHES
    x = collect(range(-L, L; length = N))
    h = x[2] - x[1]
    hp = ElasticHalfPlane2D(G, ν; h = h)
    t0 = time()
    hist = solve_cattaneo_cohesive_history_halfplane(
        x, par.R_eq, par.f, hp, par.load_steps; η_coh = 0.05, tol = 1e-9)
    dt = time() - t0
    step_sols = Dict(s.name => s for s in hist)
    coh_by_N[N] = (; x, hist = step_sols, dt)
    for sol in hist
        q_refg = interp_to(x, sol.τ, x_ref)
        p_refg = interp_to(x, sol.p, x_ref)
        l2q = l2_rel(q_refg, ana[sol.name].q)
        l2p = l2_rel(p_refg, ana[sol.name].p)
        push!(rows, (; method = "cohesive", N, step = String(sol.name),
            a = sol.a, err_a_pct = relerr(sol.a, par.a),
            p0 = sol.p0, err_p0_pct = relerr(sol.p0, par.p0),
            L2_q = l2q, L2_p = l2p, time_s = dt / length(hist)))
        @printf("  N=%3d  %s  a=%.4f (%.2f%%)  p0=%.2f (%.2f%%)  L2(q)=%.3e\n",
            N, sol.name, sol.a, relerr(sol.a, par.a), sol.p0, relerr(sol.p0, par.p0), l2q)
    end
    sols[(:cohesive, N)] = coh_by_N[N]
end

# --- Mortar STS (non-matching) with Mindlin history ---
println("\n[4] Mortar STS (history)")
mort_by_N = Dict{Int,Any}()
for N in N_MESHES
    x_s = collect(range(-L, L; length = N))
    N_m = max(21, (N ÷ 3) | 1)          # odd, ~N/3
    x_m = collect(range(-L, L; length = N_m))
    t0 = time()
    hist = solve_cattaneo_mortar_history_halfplane(
        x_s, x_m, par.R_eq, par.f, G, ν, par.load_steps; tol = 1e-9)
    dt = time() - t0
    step_sols = Dict(s.name => s for s in hist)
    mort_by_N[N] = (; x_s, x_m, hist = step_sols, dt)
    for sol in hist
        q_refg = interp_to(x_s, sol.τ_s, x_ref)
        p_refg = interp_to(x_s, sol.p_s, x_ref)
        l2q = l2_rel(q_refg, ana[sol.name].q)
        l2p = l2_rel(p_refg, ana[sol.name].p)
        push!(rows, (; method = "mortar", N, step = String(sol.name),
            a = sol.a_s, err_a_pct = relerr(sol.a_s, par.a),
            p0 = sol.p0_s, err_p0_pct = relerr(sol.p0_s, par.p0),
            L2_q = l2q, L2_p = l2p, time_s = dt / length(hist)))
        @printf("  N=%3d (Nm=%d)  %s  a_s=%.4f (%.2f%%)  p0_s=%.2f (%.2f%%)  L2(q)=%.3e\n",
            N, N_m, sol.name, sol.a_s, relerr(sol.a_s, par.a),
            sol.p0_s, relerr(sol.p0_s, par.p0), l2q)
    end
    sols[(:mortar, N)] = mort_by_N[N]
end

# -----------------------------------------------------------------------------
# CSV tables
# -----------------------------------------------------------------------------
function _l2p(r)
    hasproperty(r, :L2_p) && return float(r.L2_p)
    return r.method == "analytical" ? 0.0 : NaN
end
function _l2q(r)
    hasproperty(r, :L2_q) && return float(r.L2_q)
    return NaN
end

open(joinpath(OUTDIR, "results_all.csv"), "w") do io
    println(io, "method,N,step,a,err_a_pct,p0,err_p0_pct,L2_q,L2_p,time_s")
    for r in rows
        @printf(io, "%s,%d,%s,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e\n",
            r.method, r.N, r.step, r.a, r.err_a_pct, r.p0, r.err_p0_pct,
            _l2q(r), _l2p(r), r.time_s)
    end
end
rowsB = filter(r -> r.step == "B", rows)
open(joinpath(OUTDIR, "results_stepB.csv"), "w") do io
    println(io, "method,N,step,a,err_a_pct,p0,err_p0_pct,L2_q,L2_p,time_s")
    for r in rowsB
        @printf(io, "%s,%d,%s,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e\n",
            r.method, r.N, r.step, r.a, r.err_a_pct, r.p0, r.err_p0_pct,
            _l2q(r), _l2p(r), r.time_s)
    end
end
println("\nWrote results_all.csv and results_stepB.csv")

println("\n=== Step B convergence ===")
@printf("%-12s %6s %10s %10s %10s %10s %12s\n",
    "method", "N", "a", "err_a%", "p0", "err_p0%", "L2(q)")
for r in rowsB
    @printf("%-12s %6d %10.4f %10.3f %10.2f %10.3f %12.4e\n",
        r.method, r.N, r.a, r.err_a_pct, r.p0, r.err_p0_pct, _l2q(r))
end

# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------
println("\n[plots]")
cols = Dict(
    :ana => :black,
    :halfspace => :dodgerblue,
    :cohesive => :forestgreen,
    :mortar => :crimson,
)
Nplot = N_MESHES[end]   # finest mesh for profile plots
xa = x_ref ./ par.a

# Fig 1: normal pressure step A — three formulations on finest mesh
fig = plot(xa, ana[:A].p ./ par.p0; color = cols[:ana], linestyle = :dash, label = "Hertz",
    lw = 2.2, xlabel = L"x/a", ylabel = L"p/p_0", title = "Normal pressure — step A",
    legend = :topright, legendfontsize = 9, xlim = (-1.5, 1.5), size = (520, 360),
    framestyle = :box, background_color = :white)
let s = hs_by_N[Nplot]
    plot!(fig, s.x ./ par.a, s.hist[:A].p ./ par.p0; color = cols[:halfspace], label = "Half-space")
end
let s = coh_by_N[Nplot]
    plot!(fig, s.x ./ par.a, s.hist[:A].p ./ par.p0; color = cols[:cohesive], linestyle = :dot, label = "Cohesive")
end
let s = mort_by_N[Nplot]
    plot!(fig, s.x_s ./ par.a, s.hist[:A].p_s ./ par.p0; color = cols[:mortar], linestyle = :dashdot, label = "Mortar")
end
savefig(fig, joinpath(FIGDIR, "fig_pressure_A.pdf"))
savefig(fig, joinpath(FIGDIR, "fig_pressure_A.png"))

# Fig 2: tractions BCDE on finest mesh (half-space + cohesive + ana)
panels = Plots.Plot[]
for (k, st) in enumerate((:B, :C, :D, :E))
    ax = plot(xa, ana[st].p ./ par.p0; color = :gray, linestyle = :dash,
        label = L"p_\mathrm{ana}", lw = 2, xlabel = L"x/a", ylabel = L"t/p_0",
        title = "Load step $st", xlim = (-1.5, 1.5), legend = k == 1 ? :topright : false,
        legendfontsize = 8, framestyle = :box, background_color = :white)
    plot!(ax, xa, ana[st].q ./ par.p0; color = :black, label = L"q_\mathrm{ana}", lw = 2)
    let s = hs_by_N[Nplot]
        plot!(ax, s.x ./ par.a, s.hist[st].p ./ par.p0; color = cols[:halfspace], label = L"p_\mathrm{HS}")
        plot!(ax, s.x ./ par.a, s.hist[st].τ ./ par.p0; color = :navy, label = L"q_\mathrm{HS}")
    end
    let s = coh_by_N[Nplot]
        plot!(ax, s.x ./ par.a, s.hist[st].p ./ par.p0; color = cols[:cohesive], linestyle = :dot, label = L"p_\mathrm{coh}")
        plot!(ax, s.x ./ par.a, s.hist[st].τ ./ par.p0; color = :darkorange, linestyle = :dot, label = L"q_\mathrm{coh}")
    end
    push!(panels, ax)
end
fig = plot(panels...; layout = (2, 2), size = (980, 720))
savefig(fig, joinpath(FIGDIR, "fig_tractions_BCDE.pdf"))
savefig(fig, joinpath(FIGDIR, "fig_tractions_BCDE.png"))

# Fig 3: mortar vs analytical at B (slave + master)
fig = plot(; xlabel = L"x/a", ylabel = L"t/p_0",
    title = "Mortar STS vs analytical — step B (N=$Nplot)",
    legend = :topright, legendfontsize = 9, xlim = (-1.5, 1.5), size = (560, 380),
    framestyle = :box, background_color = :white)
let s = mort_by_N[Nplot]
    plot!(fig, xa, ana[:B].p ./ par.p0; color = :gray, linestyle = :dash, label = L"p_\mathrm{ana}", lw = 2)
    plot!(fig, xa, ana[:B].q ./ par.p0; color = :black, label = L"q_\mathrm{ana}", lw = 2)
    plot!(fig, s.x_s ./ par.a, s.hist[:B].p_s ./ par.p0; color = cols[:halfspace], label = L"p_\mathrm{slave}")
    plot!(fig, s.x_s ./ par.a, s.hist[:B].τ_s ./ par.p0; color = cols[:mortar], label = L"q_\mathrm{slave}")
    scatter!(fig, s.x_m ./ par.a, s.hist[:B].p_m ./ par.p0; color = cols[:halfspace],
        marker = :circle, markersize = 5, label = L"p_\mathrm{master}")
    scatter!(fig, s.x_m ./ par.a, s.hist[:B].τ_m ./ par.p0; color = cols[:mortar],
        marker = :rect, markersize = 5, label = L"q_\mathrm{master}")
end
savefig(fig, joinpath(FIGDIR, "fig_mortar_B.pdf"))
savefig(fig, joinpath(FIGDIR, "fig_mortar_B.png"))

# Fig 4: stick/slip map step B (half-space finest)
let s = hs_by_N[Nplot]
    zone = zeros(length(s.x))
    zone[s.hist[:B].slip] .= 2
    zone[s.hist[:B].stick] .= 1
    cB = ana[:B].c / par.a
    fig = bar(s.x ./ par.a, zone; xlabel = L"x/a", ylabel = "zone",
        title = "Stick / slip — step B (half-space, N=$Nplot)",
        yticks = ([0, 1, 2], ["open", "stick", "slip"]),
        fill_z = reshape(zone, 1, :), color = cgrad([:white, :dodgerblue, :crimson]),
        linecolor = :match, legend = :topright, legendfontsize = 9,
        size = (560, 300), framestyle = :box, background_color = :white)
    vline!(fig, [-cB, cB]; color = :black, linestyle = :dash, label = L"\pm c/a")
    vline!(fig, [-1.0, 1.0]; color = :gray, linestyle = :dot, label = L"\pm a")
    savefig(fig, joinpath(FIGDIR, "fig_stick_slip_B.pdf"))
    savefig(fig, joinpath(FIGDIR, "fig_stick_slip_B.png"))
end

# Fig 5: mesh convergence of a and p0 (step A and B)
conv_panels = Plots.Plot[]
for (col, st, ttl) in ((1, :A, "Step A (normal)"), (2, :B, "Step B (partial slip)"))
    axa = plot(; xlabel = L"N", ylabel = "relative error [%]", title = ttl,
        yscale = :log10, xscale = :log2, xticks = (collect(N_MESHES), string.(collect(N_MESHES))),
        legend = col == 1 ? :topright : false, legendfontsize = 8,
        framestyle = :box, background_color = :white)
    for (meth, d) in ((:halfspace, hs_by_N), (:cohesive, coh_by_N), (:mortar, mort_by_N))
        Ns = Int[]; ea = Float64[]; ep = Float64[]
        for N in N_MESHES
            sol = d[N].hist[st]
            a_num = meth === :mortar ? sol.a_s : sol.a
            p0_num = meth === :mortar ? sol.p0_s : sol.p0
            push!(Ns, N)
            push!(ea, max(relerr(a_num, par.a), 1e-4))
            push!(ep, max(relerr(p0_num, par.p0), 1e-4))
        end
        plot!(axa, Ns, ea; color = cols[meth], marker = :circle, label = "$(meth) \$a\$")
        plot!(axa, Ns, ep; color = cols[meth], linestyle = :dash, marker = :utriangle,
            label = "$(meth) \$p_0\$")
    end
    push!(conv_panels, axa)
end
fig = plot(conv_panels...; layout = (1, 2), size = (900, 360))
savefig(fig, joinpath(FIGDIR, "fig_convergence.pdf"))
savefig(fig, joinpath(FIGDIR, "fig_convergence.png"))

# Fig 6: L2(q) at step B vs N
fig = plot(; xlabel = L"N", ylabel = L"L_2(q)/\|q_\mathrm{ana}\|_2",
    title = "Shear traction error — step B", yscale = :log10, xscale = :log2,
    xticks = (collect(N_MESHES), string.(collect(N_MESHES))),
    legend = :topright, legendfontsize = 9, size = (480, 360),
    framestyle = :box, background_color = :white)
for meth in ("halfspace", "cohesive", "mortar")
    sub = filter(r -> r.method == meth && r.step == "B", rows)
    isempty(sub) && continue
    Ns = [r.N for r in sub]
    y = [max(_l2q(r), 1e-6) for r in sub]
    plot!(fig, Ns, y; color = cols[Symbol(meth)], marker = :circle, label = meth)
end
savefig(fig, joinpath(FIGDIR, "fig_L2q_B.pdf"))
savefig(fig, joinpath(FIGDIR, "fig_L2q_B.png"))

# Fig 7: wall time vs N (full history)
fig = plot(; xlabel = L"N", ylabel = "wall time [s]",
    title = "Cost of full A–E history", yscale = :log10, xscale = :log2,
    xticks = (collect(N_MESHES), string.(collect(N_MESHES))),
    legend = :topleft, legendfontsize = 9, size = (480, 360),
    framestyle = :box, background_color = :white)
for (meth, d) in ((:halfspace, hs_by_N), (:cohesive, coh_by_N), (:mortar, mort_by_N))
    Ns = collect(N_MESHES)
    ts = [d[N].dt for N in Ns]
    plot!(fig, Ns, ts; color = cols[meth], marker = :circle, label = String(meth))
end
savefig(fig, joinpath(FIGDIR, "fig_timing.pdf"))
savefig(fig, joinpath(FIGDIR, "fig_timing.png"))

# Fig 8: overlay all formulations at step B on finest mesh
fig = plot(xa, ana[:B].p ./ par.p0; color = :gray55, linestyle = :dash, lw = 2.2,
    label = L"p_\mathrm{ana}", xlabel = L"x/a", ylabel = L"t/p_0",
    title = "All formulations — step B (N=$Nplot)",
    legend = :topright, legendfontsize = 9, xlim = (-1.5, 1.5), size = (560, 400),
    framestyle = :box, background_color = :white)
plot!(fig, xa, ana[:B].q ./ par.p0; color = :black, lw = 2.2, label = L"q_\mathrm{ana}")
let s = hs_by_N[Nplot]
    plot!(fig, s.x ./ par.a, s.hist[:B].τ ./ par.p0; color = cols[:halfspace], label = "HS \$q\$")
end
let s = coh_by_N[Nplot]
    plot!(fig, s.x ./ par.a, s.hist[:B].τ ./ par.p0; color = cols[:cohesive], linestyle = :dot, label = "coh \$q\$")
end
let s = mort_by_N[Nplot]
    plot!(fig, s.x_s ./ par.a, s.hist[:B].τ_s ./ par.p0; color = cols[:mortar], linestyle = :dashdot, label = "mortar \$q\$")
end
cB = ana[:B].c / par.a
vline!(fig, [-cB, cB]; color = :gray, linestyle = :dash, label = L"\pm c/a")
savefig(fig, joinpath(FIGDIR, "fig_all_q_B.pdf"))
savefig(fig, joinpath(FIGDIR, "fig_all_q_B.png"))

println("\nDone. Figures → $FIGDIR")
println("CSV      → $OUTDIR")
