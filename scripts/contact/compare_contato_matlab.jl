# =============================================================================
# Julia contact vs Contato MATLAB (Área de Trabalho/Contato)
#
# MATLAB: quadratic discontinuous at ξ=±2/3, Euclidean gap, geometric n,
#   aplica_contato_com_atrito_multicorpos (same constraint rows as Julia).
#
#   julia --project=. scripts/contact/compare_contato_matlab.jl
# =============================================================================
ENV["GKSwstype"] = "100"
using DrWatson
@quickactivate :BEM
using BEM.Contact
using BEM.MultiRegion
using LinearAlgebra, Statistics, Printf, Dates, Plots

include(datadir("elastico", "two_blocks_contact.jl"))
include(datadir("elastico", "dad_5d_contact.jl"))
include(datadir("elastico", "loyola_bulk_contact.jl"))

const OUT = joinpath(projectdir(), "plots", "cattaneo_mindlin", "compare_matlab")
mkpath(OUT)
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

function contato_newton!(prob; tol=1e-9, maxiter=80, npg=10, verbose=true,
        common_normal=true)
    ctx = MR._contact_friction_setup(prob; method=:ntn, npg=npg,
        common_normal=common_normal)
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
    return (; fr..., p, q, P, Q, p0, n_stick, n_slip, n_open, n_closed, n_coul,
        util_mean=isempty(util) ? 0.0 : mean(util), L2p, qmax)
end

function run_case(tag, loadfun, par; matlab=false)
    println("\n  — $tag")
    t0 = time()
    if matlab
        prob, _ = loadfun(; gap=:euclidean)
        pin_top_center_ux!(prob)
        sol = contato_newton!(prob; common_normal=false)
    else
        prob, _ = loadfun()
        pin_top_center_ux!(prob)
        sol = contato_newton!(prob; common_normal=true)
    end
    dt = time() - t0
    m = metrics(prob, par)
    @printf("    ok=%s it=%d  P=%.2f  Q=%+.2f  p0=%.2f (H %.2f%%)  L2p=%.4f\n",
        sol.ok, sol.nit, m.P, m.Q, m.p0, relerr(m.p0, par.p0_H), m.L2p)
    @printf("    st/sl/op=%d/%d/%d  coul=%d/%d  util=%.3f  |q|max/(f p0)=%.3f  t=%.1fs\n",
        m.n_stick, m.n_slip, m.n_open, m.n_coul, m.n_closed, m.util_mean,
        m.qmax / (par.μ * par.p0_H), dt)
    ξ = extrema(prob.regions[1].element_type.nodes)
    @printf("    collocation ξ ∈ [%.4f, %.4f]  gap0 ∈ [%.5f, %.5f]\n",
        ξ..., extrema(cp.gap0 for cp in prob.contacts)...)
    return (; m, dt, sol, prob)
end

println("="^72)
println(" Contato MATLAB vs Julia  ", Dates.now())
println("="^72)
println("MATLAB: ξ=±2/3 disc. Lagrange, h=‖xB−xA‖, geometric n")
println("Julia default: Gauss–Legendre, h=n·Δx, n_AB")

p931 = loyola_dad5d_params()
p932 = loyola_bulk_params()

println("\n[1] §9.3.1 two cylinders  (dad_sapatas)")
r931j = run_case("Julia default",
    (; kwargs...) -> load_dad_5d_contact(; ndiv_c=12, ndiv_f=6, ndiv_s=6, ndiv_top=12,
        tipo=2, nome="ml_931_j", kwargs...), p931)
r931m = run_case("MATLAB-faithful",
    (; kwargs...) -> load_dad_5d_contact(; ndiv_c=12, ndiv_f=6, ndiv_s=6, ndiv_top=12,
        tipo=2, nome="ml_931_m", kwargs...), p931; matlab=true)

println("\n[2] §9.3.2 pad on specimen")
r932j = run_case("Julia default",
    (; kwargs...) -> load_loyola_bulk_contact(; ndiv_c=12, ndiv_f=6, ndiv_s=4, ndiv_top=5,
        tipo=2, nome="ml_932_j", kwargs...), p932)
r932m = run_case("MATLAB-faithful",
    (; kwargs...) -> load_loyola_bulk_contact(; ndiv_c=12, ndiv_f=6, ndiv_s=4, ndiv_top=5,
        tipo=2, nome="ml_932_m", kwargs...), p932; matlab=true)

println("\n" * "="^72)
println(" Summary")
println("="^72)
@printf("%-8s %-18s %8s %8s %8s %7s %6s %6s\n",
    "case", "path", "P", "Q", "p0", "errH%", "util", "L2p")
for (lab, rj, rm, pr) in (("9.3.1", r931j, r931m, p931), ("9.3.2", r932j, r932m, p932))
    for (nm, r) in (("Julia default", rj), ("MATLAB-faithful", rm))
        m = r.m
        @printf("%-8s %-18s %8.2f %+8.2f %8.2f %7.2f %6.3f %6.4f\n",
            lab, nm, m.P, m.Q, m.p0, relerr(m.p0, pr.p0_H), m.util_mean, m.L2p)
    end
end

default(linewidth=1.8, legendfontsize=8, grid=true, framestyle=:box)
function overlay(tag, rj, rm, par, titlep)
    aH, p0H, μ = par.a_H, par.p0_H, par.μ
    xa = collect(range(-1.15aH, 1.15aH; length=400))
    pH = cattaneo_pressure(xa, aH, p0H)
    plt_p = plot(xa./aH, pH./p0H; color=:black, label="Hertz", xlabel="x/a",
        ylabel="p/p0", title=titlep, ylims=(-0.05, 1.2))
    scatter!(plt_p, rj.m.x./aH, rj.m.p./p0H; ms=4, markerstrokewidth=0, color=:steelblue,
        label="Julia GL")
    scatter!(plt_p, rm.m.x./aH, rm.m.p./p0H; ms=4, markerstrokewidth=0, color=:darkorange,
        label="Contato ξ=±2/3")
    plt_q = plot(xlabel="x/a", ylabel="q/(f p0)", title="shear", ylims=(-0.7, 0.7))
    hline!(plt_q, [0.0]; color=:gray, ls=:dot, label=false)
    scatter!(plt_q, rj.m.x./aH, rj.m.q./(μ*p0H); ms=4, markerstrokewidth=0,
        color=:steelblue, label="Julia GL")
    scatter!(plt_q, rm.m.x./aH, rm.m.q./(μ*p0H); ms=4, markerstrokewidth=0,
        color=:darkorange, label="Contato ξ=±2/3")
    savefig(plt_p, joinpath(OUT, tag * "_p.png"))
    savefig(plt_q, joinpath(OUT, tag * "_q.png"))
end
overlay("931", r931j, r931m, p931, "§9.3.1 Step A")
overlay("932", r932j, r932m, p932, "§9.3.2 Step A")
println("\nplots in $OUT")
println("Done.")
