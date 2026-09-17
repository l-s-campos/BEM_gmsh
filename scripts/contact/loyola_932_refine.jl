# =============================================================================
# Loyola §9.3.2 Step A — h-refinement of discontinuous quadratic GL
#
# If the q scatter / p0 offset is just unresolved disc. collocation, p0 should
# approach Hertz and Coulomb utilisation / |q| should drop as ndiv_c grows.
#
#   julia --project=. scripts/contact/loyola_932_refine.jl
# =============================================================================
ENV["GKSwstype"] = "100"

using DrWatson
@quickactivate :BEM
using BEM.Contact
using BEM.MultiRegion

using LinearAlgebra
using Statistics
using Printf
using Dates
using Plots

include(datadir("elastico", "two_blocks_contact.jl"))
include(datadir("elastico", "loyola_bulk_contact.jl"))

const OUTDIR = joinpath(projectdir(), "plots", "cattaneo_mindlin", "loyola932_refine")
mkpath(OUTDIR)
const MR = BEM.MultiRegion

relerr(num, ref) = abs(num - ref) / max(abs(ref), eps()) * 100

function pin_top_center_ux!(prob)
    dad = prob.regions[1]
    ymax = maximum(pt[2] for pt in dad.Nodes)
    thr = 1e-9 * max(abs(ymax), 1.0)
    best, bx = 0, Inf
    for i in 1:dad.n
        abs(dad.Nodes[i][2] - ymax) <= thr || continue
        ax = abs(dad.Nodes[i][1])
        ax < bx && (bx = ax; best = i)
    end
    best == 0 && error("pin_top_center_ux!: no top node")
    dad.BC[2best - 1] = 0
    dad.BV[2best - 1] = 0.0
    return best
end

function contato_newton!(prob; tol=1e-9, maxiter=80, npg=10, verbose=false)
    ctx = MR._contact_friction_setup(prob; method=:ntn, npg=npg)
    prep, pairs = ctx.prep, ctx.pairs
    h = [cp.gap0 for cp in pairs]
    x0 = zeros(ctx.N)
    for cp in pairs
        cp.state = 3
        cp.ut_lock = 0.0
    end
    ok = false
    nit = 0
    prev = [cp.state for cp in pairs]
    for it in 1:maxiter
        nit = it
        it > 1 && MR._verify_contact_states!(pairs, prep, h, x0; epsc=1e-7)
        st = [cp.state for cp in pairs]
        n_flip = count(i -> st[i] != prev[i], eachindex(st))
        A, b = MR._assemble_contact_system(prep, pairs, h, x0)
        x = A \ b
        dist = norm(x - x0) / max(1.0, norm(x))
        verbose && @printf("    it=%2d  rel=%.3e  nflips=%d  o/s/l=%d/%d/%d\n", it, dist, n_flip,
            count(==(1), st), count(==(3), st), count(s -> abs(s) == 2, st))
        x0 = x
        prev = st
        if dist < tol && n_flip == 0
            ok = true
            break
        elseif it >= 12 && n_flip <= 2 && dist < 0.2
            ok = true
            break
        end
    end
    MR._verify_contact_states!(pairs, prep, h, x0; epsc=1e-7)
    MR._update_contact_ut_locks!(pairs, prep, h, x0)
    MR._scatter_contact_solution!(prob, prep, pairs, x0)
    set_cache!(prob.regions[1]; contact_x=copy(x0))
    return (; ok, nit, x=x0)
end

function node_weights(xs)
    n = length(xs)
    w = ones(n)
    n < 2 && return w
    w[1] = abs(xs[2] - xs[1]); w[end] = abs(xs[end] - xs[end - 1])
    for i in 2:n-1
        w[i] = 0.5 * abs(xs[i + 1] - xs[i - 1])
    end
    return w
end

"""Mean traction per 3-node disc. quadratic contact element (sorted by x)."""
function element_means(m)
    n = length(m.x)
    n3 = n ÷ 3
    xe = zeros(n3); pe = zeros(n3); qe = zeros(n3)
    k = 0
    i = 1
    while i + 2 <= n
        k += 1
        xe[k] = mean(m.x[i:i+2])
        pe[k] = mean(m.p[i:i+2])
        qe[k] = mean(m.q[i:i+2])
        i += 3
    end
    return (; x=xe[1:k], p=pe[1:k], q=qe[1:k])
end

function metrics(prob, par)
    fr = contact_interface_xyτ(prob)
    xs, wnode = fr.x, node_weights(fr.x)
    p = .-fr.tn
    q = .-fr.tt
    cl = abs.(fr.state) .!= 1
    P = any(cl) ? sum(p[cl] .* wnode[cl]) : 0.0
    Q = any(cl) ? sum(q[cl] .* wnode[cl]) : 0.0
    p0 = any(cl) ? maximum(p[cl]) : 0.0
    a = p0 > 0 ? 2P / (π * p0) : 0.0
    a_sup = any(cl) ? 0.5 * (maximum(xs[cl]) - minimum(xs[cl])) : 0.0
    n_stick = count(==(3), fr.state)
    n_slip = count(s -> abs(s) == 2, fr.state)
    n_open = count(==(1), fr.state)
    util = Float64[]
    n_coul = 0
    for i in eachindex(fr.state)
        abs(fr.state[i]) == 1 && continue
        bound = par.μ * abs(fr.tn[i])
        bound < 1e-12 && continue
        u = abs(fr.tt[i]) / bound
        push!(util, u)
        u > 0.98 && (n_coul += 1)
    end
    q_signflips = 0
    for i in 2:length(q)
        cl[i] && cl[i - 1] || continue
        sign(q[i]) != sign(q[i - 1]) && abs(q[i]) > 1e-6 && abs(q[i - 1]) > 1e-6 &&
            (q_signflips += 1)
    end
    pH = cattaneo_pressure(xs, par.a_H, par.p0_H)
    denp = sqrt(sum(wnode .* abs2.(pH))) + eps()
    L2p = sqrt(sum(wnode .* abs2.(p .- pH))) / denp
    denq = par.μ * par.p0_H * sqrt(sum(wnode[cl])) + eps()
    L2q = any(cl) ? sqrt(sum(wnode[cl] .* abs2.(q[cl]))) / denq : 0.0
    qmax = any(cl) ? maximum(abs, q[cl]) : 0.0
    em = element_means((; x=xs, p, q))
    p0_el = isempty(em.p) ? 0.0 : maximum(em.p)
    return (; fr..., p, q, P, Q, p0, a, a_sup, n_stick, n_slip, n_open, n_coul,
        util_mean=isempty(util) ? 0.0 : mean(util),
        util_med=isempty(util) ? 0.0 : median(util),
        qrms=sqrt(mean(abs2, q)), qmax, q_signflips, L2p, L2q, p0_el, em,
        n_closed=count(cl), npc=length(fr.x))
end

# transfinite nPoints on contact: ndiv_c centre + 2× ndiv_f shoulders
# disc. quadratic ⇒ NPc = 3*((ndiv_c-1) + 2*(ndiv_f-1))
const LEVELS = (
    (ndiv_c=12, ndiv_f=6,  ndiv_s=4, ndiv_top=5,  ndiv_out=6,  ndiv_bot=12, ndiv_bulk=10),
    (ndiv_c=24, ndiv_f=12, ndiv_s=6, ndiv_top=8,  ndiv_out=8,  ndiv_bot=16, ndiv_bulk=12),
    (ndiv_c=48, ndiv_f=24, ndiv_s=8, ndiv_top=10, ndiv_out=10, ndiv_bot=20, ndiv_bulk=14),
)

par = loyola_bulk_params()
println("="^72)
println(" Loyola §9.3.2 Step A  — disc. quadratic h-refinement")
println(" ", Dates.now())
println("="^72)
@printf("Hertz  a=%.4f  p0=%.2f   thesis p0=%.2f  P=%.1f  μ=%.2f\n",
    par.a_H, par.p0_H, par.p0_th, par.P, par.μ)

rows = []
for (k, lv) in enumerate(LEVELS)
    npc_expect = 3 * ((lv.ndiv_c - 1) + 2 * (lv.ndiv_f - 1))
    println("\n[$k] ndiv_c=$(lv.ndiv_c)  ndiv_f=$(lv.ndiv_f)  expected NPc=$npc_expect")
    t0 = time()
    prob, _ = load_loyola_bulk_contact(; lv..., tipo=2, nome="loyola932_h$(lv.ndiv_c)")
    pin_top_center_ux!(prob)
    @printf("    pad n=%d  spec n=%d  NPc=%d\n",
        prob.regions[1].n, prob.regions[2].n, length(prob.contacts))
    sol = contato_newton!(prob; tol=1e-9, maxiter=100, npg=10, verbose=true)
    dt = time() - t0
    m = metrics(prob, par)
    @printf("    ok=%s it=%d  P=%.2f (%.2f%%)  Q=%+.2f  p0=%.2f (H %.2f%%  th %.2f%%)  p0_el=%.2f\n",
        sol.ok, sol.nit, m.P, relerr(m.P, par.P), m.Q, m.p0,
        relerr(m.p0, par.p0_H), relerr(m.p0, par.p0_th), m.p0_el)
    @printf("    a=%.4f a_sup=%.4f  st/sl/op=%d/%d/%d  coul=%d/%d  util=%.3f  |q|_max/(f p0)=%.3f\n",
        m.a, m.a_sup, m.n_stick, m.n_slip, m.n_open, m.n_coul, m.n_closed,
        m.util_mean, m.qmax / max(par.μ * par.p0_H, eps()))
    @printf("    L2(p-pH)/L2(pH)=%.4f  L2(q)/(f p0 √A)=%.4f  flips=%d  t=%.1fs\n",
        m.L2p, m.L2q, m.q_signflips, dt)
    push!(rows, (; lv..., m, dt, ok=sol.ok, nit=sol.nit, n_pad=prob.regions[1].n,
        n_spec=prob.regions[2].n))
end

println("\n" * "="^72)
println(" Summary")
println("="^72)
@printf("%-7s %5s %6s %8s %8s %8s %8s %7s %6s %6s %6s %6s\n",
    "ndiv_c", "NPc", "it", "P", "Q", "p0", "p0_el", "errH%", "coul%", "util", "L2p", "L2q")
for r in rows
    m = r.m
    @printf("%-7d %5d %6d %8.2f %+8.2f %8.2f %8.2f %7.2f %6.1f %6.3f %6.4f %6.4f\n",
        r.ndiv_c, m.npc, r.nit, m.P, m.Q, m.p0, m.p0_el, relerr(m.p0, par.p0_H),
        100 * m.n_coul / max(m.n_closed, 1), m.util_mean, m.L2p, m.L2q)
end
println("If disc. GL is only a resolution issue: errH%, util, L2q should fall with ndiv_c.")

# -----------------------------------------------------------------------------
println("\n[plots] $OUTDIR")
default(linewidth=1.6, legendfontsize=8, tickfontsize=9, guidefontsize=10,
    titlefontsize=11, grid=true, framestyle=:box, legend_foreground_color=:black)

cols = [:steelblue, :darkorange, :seagreen]
aH, p0H, μ = par.a_H, par.p0_H, par.μ
xa = collect(range(-1.2aH, 1.2aH; length=400))
pAna = cattaneo_pressure(xa, aH, p0H)

plt_p = plot(xa ./ aH, pAna ./ p0H; color=:black, label="Hertz",
    xlabel="x/a", ylabel="p / p0", title="§9.3.2 Step A  pressure vs h",
    ylims=(-0.05, 1.25))
plt_q = plot(; xlabel="x/a", ylabel="q / (f p0)", title="§9.3.2 Step A  shear vs h",
    ylims=(-0.7, 0.7))
hline!(plt_q, [0.0]; color=:gray, linestyle=:dot, label=false)
plt_pe = plot(xa ./ aH, pAna ./ p0H; color=:black, label="Hertz",
    xlabel="x/a", ylabel="⟨p⟩_elem / p0", title="element-mean pressure",
    ylims=(-0.05, 1.25))

for (k, r) in enumerate(rows)
    m = r.m
    lab = "ndiv_c=$(r.ndiv_c)  NPc=$(m.npc)"
    scatter!(plt_p, m.x ./ aH, m.p ./ p0H; color=cols[k], label=lab,
        markersize=3, markerstrokewidth=0)
    scatter!(plt_q, m.x ./ aH, m.q ./ (μ * p0H); color=cols[k], label=lab,
        markersize=3, markerstrokewidth=0)
    scatter!(plt_pe, m.em.x ./ aH, m.em.p ./ p0H; color=cols[k], label=lab,
        markersize=5, markerstrokewidth=0)
end
savefig(plt_p, joinpath(OUTDIR, "p_vs_h.png"))
savefig(plt_q, joinpath(OUTDIR, "q_vs_h.png"))
savefig(plt_pe, joinpath(OUTDIR, "p_elem_vs_h.png"))

plt_c = plot([r.ndiv_c for r in rows], [relerr(r.m.p0, p0H) for r in rows];
    marker=:circle, label="p0 vs Hertz %", xlabel="ndiv_c", ylabel="error / util",
    title="§9.3.2 Step A  refinement")
plot!(plt_c, [r.ndiv_c for r in rows], [100 * r.m.n_coul / max(r.m.n_closed, 1) for r in rows];
    marker=:square, label="% closed at Coulomb")
plot!(plt_c, [r.ndiv_c for r in rows], [100 * r.m.util_mean for r in rows];
    marker=:diamond, label="100 × mean |q|/(μ|p|)")
savefig(plt_c, joinpath(OUTDIR, "convergence.png"))

open(joinpath(OUTDIR, "refine.txt"), "w") do io
    println(io, "ndiv_c NPc n_pad n_spec it P Q p0 p0_el errH_pct coul_pct util L2p L2q t")
    for r in rows
        m = r.m
        @printf(io, "%d %d %d %d %d %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.6f %.6f %.2f\n",
            r.ndiv_c, m.npc, r.n_pad, r.n_spec, r.nit, m.P, m.Q, m.p0, m.p0_el,
            relerr(m.p0, p0H), 100 * m.n_coul / max(m.n_closed, 1), m.util_mean,
            m.L2p, m.L2q, r.dt)
    end
end
println("Done.  plots in $OUTDIR")
