# =============================================================================
# Loyola §9.3.2 Step A — standard collocation H,G vs DiBFM (Zhang 2019)
#
# DiBFM: continuous Lagrange on source+virtual endpoints, virtual DOFs
# condensed (MLS / RBF). Operators Hs, Gs replace dad.H, dad.G in the
# existing NTN contact solver.
#
#   julia --project=. scripts/contact/loyola_932_dibfm.jl
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

const OUTDIR = joinpath(projectdir(), "plots", "cattaneo_mindlin", "loyola932_dibfm")
mkpath(OUTDIR)
const MR = BEM.MultiRegion
relerr(a, b) = abs(a - b) / max(abs(b), eps()) * 100

function pin_top_center_ux!(prob)
    dad = prob.regions[1]
    ymax = maximum(pt[2] for pt in dad.Nodes)
    best, bx = 0, Inf
    for i in 1:dad.n
        abs(dad.Nodes[i][2] - ymax) <= 1e-9 * max(abs(ymax), 1.0) || continue
        ax = abs(dad.Nodes[i][1])
        ax < bx && (bx = ax; best = i)
    end
    dad.BC[2best-1] = 0; dad.BV[2best-1] = 0.0
    return best
end

"""Swap `dad.H, dad.G` for DiBFM condensed operators (source-node size)."""
function attach_dibfm_HG!(dad; npg=12, method=:mls)
    d = dibfm_elast_from_bemdata(dad)
    assemble_dibfm_elast!(d; npg=npg, method=method)
    ns = length(d.source_pos)
    ns == dad.n || error("DiBFM n_s=$ns ≠ dad.n=$(dad.n)")
    size(d.Hs) == (2ns, 2ns) || error("Hs size $(size(d.Hs))")
    set_cache!(dad; H=d.Hs, G=d.Gs)
    return d
end

function contato_newton!(prob; tol=1e-9, maxiter=80, npg=10, verbose=true)
    ctx = MR._contact_friction_setup(prob; method=:ntn, npg=npg)
    prep, pairs = ctx.prep, ctx.pairs
    h = [cp.gap0 for cp in pairs]
    x0 = zeros(ctx.N)
    for cp in pairs
        cp.state = 3; cp.ut_lock = 0.0
    end
    ok, nit, prev = false, 0, [cp.state for cp in pairs]
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
        x0 = x; prev = st
        if dist < tol && n_flip == 0
            ok = true; break
        elseif it >= 12 && n_flip <= 2 && dist < 0.2
            ok = true; break
        end
    end
    MR._verify_contact_states!(pairs, prep, h, x0; epsc=1e-7)
    MR._update_contact_ut_locks!(pairs, prep, h, x0)
    MR._scatter_contact_solution!(prob, prep, pairs, x0)
    set_cache!(prob.regions[1]; contact_x=copy(x0))
    return (; ok, nit)
end

function node_weights(xs)
    n = length(xs); w = ones(n); n < 2 && return w
    w[1] = abs(xs[2]-xs[1]); w[end] = abs(xs[end]-xs[end-1])
    for i in 2:n-1; w[i] = 0.5*abs(xs[i+1]-xs[i-1]); end
    return w
end

function metrics(prob, par)
    fr = contact_interface_xyτ(prob)
    xs, w = fr.x, node_weights(fr.x)
    p = .-fr.tn; q = .-fr.tt
    cl = abs.(fr.state) .!= 1
    P = any(cl) ? sum(p[cl].*w[cl]) : 0.0
    Q = any(cl) ? sum(q[cl].*w[cl]) : 0.0
    p0 = any(cl) ? maximum(p[cl]) : 0.0
    a_sup = any(cl) ? 0.5*(maximum(xs[cl])-minimum(xs[cl])) : 0.0
    n_stick = count(==(3), fr.state)
    n_slip = count(s -> abs(s)==2, fr.state)
    n_open = count(==(1), fr.state)
    n_closed = count(cl)
    util = Float64[]; n_coul = 0
    for i in eachindex(fr.state)
        abs(fr.state[i]) == 1 && continue
        bound = par.μ * abs(fr.tn[i]); bound < 1e-12 && continue
        u = abs(fr.tt[i]) / bound
        push!(util, u); u > 0.98 && (n_coul += 1)
    end
    pH = cattaneo_pressure(xs, par.a_H, par.p0_H)
    L2p = sqrt(sum(w .* abs2.(p .- pH))) / (sqrt(sum(w .* abs2.(pH))) + eps())
    qmax = any(cl) ? maximum(abs, q[cl]) : 0.0
    return (; fr..., p, q, P, Q, p0, a_sup, n_stick, n_slip, n_open, n_closed,
        n_coul, util_mean=isempty(util) ? 0.0 : mean(util), L2p, qmax)
end

function loadA(nome)
    prob, par = load_loyola_bulk_contact(; ndiv_c=12, ndiv_f=6, ndiv_s=4, ndiv_top=5,
        ndiv_out=6, ndiv_bot=12, ndiv_bulk=10, tipo=2, nome=nome)
    pin_top_center_ux!(prob)
    return prob, par
end

println("="^72)
println(" §9.3.2 Step A  standard BEM vs DiBFM  ", Dates.now())
println("="^72)
par = loyola_bulk_params()
@printf("Hertz p0=%.2f  a=%.4f  thesis p0=%.2f  P=%.0f\n", par.p0_H, par.a_H, par.p0_th, par.P)

cases = []

# --- standard ---
println("\n[1] standard collocation H,G")
let
    prob, _ = loadA("932_std")
    t0 = time()
    sol = contato_newton!(prob)
    dt = time() - t0
    m = metrics(prob, par)
    @printf("  ok=%s it=%d  P=%.2f  Q=%+.2f  p0=%.2f (H %.2f%%)  L2p=%.4f\n",
        sol.ok, sol.nit, m.P, m.Q, m.p0, relerr(m.p0, par.p0_H), m.L2p)
    @printf("  st/sl/op=%d/%d/%d  coul=%d/%d  util=%.3f  |q|max/(f p0)=%.3f  t=%.1fs\n",
        m.n_stick, m.n_slip, m.n_open, m.n_coul, m.n_closed, m.util_mean,
        m.qmax / (par.μ * par.p0_H), dt)
    push!(cases, (; name=:std, m, dt, sol))
end

# --- DiBFM MLS ---
println("\n[2] DiBFM-MLS Hs, Gs")
let
    prob, _ = loadA("932_dibfm_mls")
    t0 = time()
    info = nothing
    for (k, dad) in enumerate(prob.regions)
        println("  assemble DiBFM region $k  n=$(dad.n)  ne=$(length(dad.elements))")
        d = attach_dibfm_HG!(dad; npg=12, method=:mls)
        @printf("    n_s=%d  n_v=%d  Φt nnz virt=%.0f\n",
            length(d.source_pos), length(d.virt_global),
            count(!=(0), d.Φt[length(d.source_pos)+1:end, :]))
        info = d
    end
    sol = contato_newton!(prob)
    dt = time() - t0
    m = metrics(prob, par)
    @printf("  ok=%s it=%d  P=%.2f  Q=%+.2f  p0=%.2f (H %.2f%%)  L2p=%.4f\n",
        sol.ok, sol.nit, m.P, m.Q, m.p0, relerr(m.p0, par.p0_H), m.L2p)
    @printf("  st/sl/op=%d/%d/%d  coul=%d/%d  util=%.3f  |q|max/(f p0)=%.3f  t=%.1fs\n",
        m.n_stick, m.n_slip, m.n_open, m.n_coul, m.n_closed, m.util_mean,
        m.qmax / (par.μ * par.p0_H), dt)
    push!(cases, (; name=:dibfm_mls, m, dt, sol))
end

# --- DiBFM RBF ---
println("\n[3] DiBFM-RBF Hs, Gs")
let
    prob, _ = loadA("932_dibfm_rbf")
    t0 = time()
    for (k, dad) in enumerate(prob.regions)
        println("  assemble DiBFM-RBF region $k  n=$(dad.n)")
        attach_dibfm_HG!(dad; npg=12, method=:rbf)
    end
    sol = contato_newton!(prob)
    dt = time() - t0
    m = metrics(prob, par)
    @printf("  ok=%s it=%d  P=%.2f  Q=%+.2f  p0=%.2f (H %.2f%%)  L2p=%.4f\n",
        sol.ok, sol.nit, m.P, m.Q, m.p0, relerr(m.p0, par.p0_H), m.L2p)
    @printf("  st/sl/op=%d/%d/%d  coul=%d/%d  util=%.3f  |q|max/(f p0)=%.3f  t=%.1fs\n",
        m.n_stick, m.n_slip, m.n_open, m.n_coul, m.n_closed, m.util_mean,
        m.qmax / (par.μ * par.p0_H), dt)
    push!(cases, (; name=:dibfm_rbf, m, dt, sol))
end

println("\n" * "="^72)
println(" Summary  (thesis p0=486.92  Hertz p0=$(round(par.p0_H; digits=2)))")
println("="^72)
@printf("%-12s %8s %8s %8s %7s %6s %6s %6s %6s\n",
    "ops", "P", "Q", "p0", "errH%", "coul%", "util", "L2p", "t[s]")
for c in cases
    m = c.m
    @printf("%-12s %8.2f %+8.2f %8.2f %7.2f %6.1f %6.3f %6.4f %6.1f\n",
        c.name, m.P, m.Q, m.p0, relerr(m.p0, par.p0_H),
        100 * m.n_coul / max(m.n_closed, 1), m.util_mean, m.L2p, c.dt)
end

default(linewidth=1.8, legendfontsize=8, grid=true, framestyle=:box)
aH, p0H, μ = par.a_H, par.p0_H, par.μ
xa = collect(range(-1.15aH, 1.15aH; length=400))
pAna = cattaneo_pressure(xa, aH, p0H)
cols = Dict(:std=>:steelblue, :dibfm_mls=>:darkorange, :dibfm_rbf=>:seagreen)
labs = Dict(:std=>"collocation H,G", :dibfm_mls=>"DiBFM-MLS", :dibfm_rbf=>"DiBFM-RBF")
plt_p = plot(xa./aH, pAna./p0H; color=:black, label="Hertz",
    xlabel="x/a", ylabel="p/p0", title="§9.3.2 A  DiBFM vs collocation", ylims=(-0.05, 1.25))
plt_q = plot(xlabel="x/a", ylabel="q/(f p0)", title="§9.3.2 A  shear", ylims=(-0.7, 0.7))
hline!(plt_q, [0.0]; color=:gray, ls=:dot, label=false)
for c in cases
    scatter!(plt_p, c.m.x./aH, c.m.p./p0H; color=cols[c.name], label=labs[c.name],
        ms=4, markerstrokewidth=0)
    scatter!(plt_q, c.m.x./aH, c.m.q./(μ*p0H); color=cols[c.name], label=labs[c.name],
        ms=4, markerstrokewidth=0)
end
savefig(plt_p, joinpath(OUTDIR, "p.png"))
savefig(plt_q, joinpath(OUTDIR, "q.png"))
println("\nplots in $OUTDIR")
println("Done.")
