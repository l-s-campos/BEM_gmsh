# =============================================================================
# Rodríguez-Tembleque & Abascal, Comput. Struct. 88 (2010) 924–937
#   :gnmls  — GNM-ls, one contact traction Λ per pair (p¹=Λ, p²=−R⁻¹Λ)
# versus the current Contato active-set (`:activeset` / contato_newton!).
#
#   julia --project=. scripts/contact/compare_rt2010.jl
#
# Sections:
#   [1] two elastic blocks, displacement-controlled (all public solvers)
#   [2] Loyola 2022 §9.3.1 Step A  (two cylinders, P only)
#   [3] Loyola 2022 §9.3.2 Step A  (pad on specimen, P only)
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
include(datadir("elastico", "dad_5d_contact.jl"))
include(datadir("elastico", "loyola_bulk_contact.jl"))

const OUTDIR = joinpath(projectdir(), "plots", "cattaneo_mindlin", "compare_rt2010")
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

"""Current Loyola driver: first iteration all-stick, then Contato verify."""
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
    hist = Float64[]
    prev = [cp.state for cp in pairs]
    for it in 1:maxiter
        nit = it
        it > 1 && MR._verify_contact_states!(pairs, prep, h, x0; epsc=1e-7)
        st = [cp.state for cp in pairs]
        n_flip = count(i -> st[i] != prev[i], eachindex(st))
        A, b = MR._assemble_contact_system(prep, pairs, h, x0)
        x = A \ b
        dist = norm(x - x0) / max(1.0, norm(x))
        push!(hist, dist)
        verbose && @printf("    it=%2d  rel=%.3e  nflips=%d  o/s/l=%d/%d/%d\n", it, dist, n_flip,
            count(==(1), st), count(==(3), st), count(s -> abs(s) == 2, st))
        x0 = x
        prev = st
        if dist < tol && n_flip == 0
            ok = true
            break
        elseif it >= 10 && n_flip <= 2 && dist < 0.2
            ok = true
            break
        end
    end
    MR._verify_contact_states!(pairs, prep, h, x0; epsc=1e-7)
    MR._update_contact_ut_locks!(pairs, prep, h, x0)
    MR._scatter_contact_solution!(prob, prep, pairs, x0)
    set_cache!(prob.regions[1]; contact_x=copy(x0))
    return (; ok, nit, x=x0, hist)
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

function metrics(prob; μ=0.3)
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
        bound = μ * abs(fr.tn[i])
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
    return (; fr..., p, q, P, Q, p0, a, a_sup,
        n_stick, n_slip, n_open, n_coul,
        util_mean=isempty(util) ? 0.0 : mean(util),
        qrms=sqrt(mean(abs2, q)),
        q_signflips)
end

function print_metrics(label, m, t; Pref=nothing, p0ref=nothing)
    @printf("  %-14s  ok-fields  P=%8.2f  Q=%+8.2f  p0=%8.2f  a=%7.4f  st/sl/op=%d/%d/%d  coul=%d  |q|rms=%.2f  util=%.3f  flips=%d  t=%.2fs\n",
        label, m.P, m.Q, m.p0, m.a, m.n_stick, m.n_slip, m.n_open, m.n_coul,
        m.qrms, m.util_mean, m.q_signflips, t)
    Pref !== nothing && @printf("                 vs P_ref  %6.2f%%     vs p0_ref %6.2f%%\n",
        relerr(m.P, Pref), relerr(m.p0, p0ref))
end

# -----------------------------------------------------------------------------
println("="^72)
println(" RT & Abascal 2010 GNM-ls  vs  current active-set")
println(" ", Dates.now())
println("="^72)

# =============================================================================
println("\n[1] two elastic blocks  (δ-controlled, public solvers)")
# =============================================================================
let
    E, ν = 100.0, 0.3
    props = Elasticity(E, ν, 1.0; plane_strain=true)
    gap0 = 0.02
    δ_end = 0.04
    kwargs = (W=1.0, H=0.4, gap=gap0, μ=0.3, ndiv_bot=6, ndiv_top=6, ndiv_y=3)
    rows = []
    for sol in (:activeset, :proj_newton, :gnmls)
        prob = load_two_blocks_contact(props; kwargs..., nome="rt2010_$sol")
        t0 = time()
        solve_contact_friction_stepped!(prob; δ_end=δ_end, nsteps=8, tol=1e-7,
            maxiter=80, npg=10, method=:ntn, solver=sol)
        dt = time() - t0
        closed = filter(cp -> abs(cp.state) != 1, prob.contacts)
        tn = isempty(closed) ? 0.0 : mean(abs(cp.tn) for cp in closed)
        tt = isempty(closed) ? 0.0 : mean(abs(cp.tt) for cp in closed)
        @printf("  %-14s  closed=%2d/%2d  mean|tn|=%8.4f  mean|tt|=%8.4f  t=%.2fs\n",
            String(sol), length(closed), length(prob.contacts), tn, tt, dt)
        push!(rows, (; sol, n=length(closed), tn, tt, dt, prob))
    end
    tn0 = rows[1].tn
    println("  mean|tn| / activeset:")
    for r in rows
        @printf("    %-14s  %.4f\n", r.sol, r.tn / max(tn0, eps()))
    end
end

# =============================================================================
function run_stepA(name, loadfun, p0ref, Pref, a_ref; ndiv_c=12, tipo=2, npg=10)
    println("\n[$name] Step A  (P only, Q=0)")
    results = Dict{Symbol,Any}()
    for (lab, solver) in ((:activeset, :activeset), (:gnmls, :gnmls),
                          (:proj_newton, :proj_newton))
        println("  — $lab")
        try
            prob, par = loadfun()
            pin_top_center_ux!(prob)
            t0 = time()
            if solver === :activeset
                sol = contato_newton!(prob; tol=1e-9, maxiter=80, npg=npg, verbose=true)
                ok, nit = sol.ok, sol.nit
            else
                (prob, _) = solve_contact_friction!(prob; δ=0.0, solver=solver,
                    method=:ntn, tol=1e-8, maxiter=80, npg=npg, verbose=true,
                    return_x=true)
                ok, nit = true, -1
            end
            dt = time() - t0
            m = metrics(prob; μ=par.μ)
            print_metrics(String(lab), m, dt; Pref=Pref, p0ref=p0ref)
            @printf("                 ok=%s  nit=%s  NPc=%d  a_ref=%.4f  a_sup=%.4f\n",
                ok, nit, length(prob.contacts), a_ref, m.a_sup)
            results[lab] = (; m, dt, ok, nit, prob, par)
        catch err
            @error "solver failed" lab=lab exception=(err, catch_backtrace())
        end
    end
    return results
end

# =============================================================================
println("\n[2] Loyola §9.3.1  two cylinders  Step A")
# =============================================================================
r931 = run_stepA("9.3.1",
    () -> load_dad_5d_contact(; ndiv_c=12, ndiv_f=6, ndiv_s=6, ndiv_top=12,
        tipo=2, nome="rt2010_931"),
    let p = loyola_dad5d_params(); p.p0_H; end,
    let p = loyola_dad5d_params(); p.P; end,
    let p = loyola_dad5d_params(); p.a_H; end)

# =============================================================================
println("\n[3] Loyola §9.3.2  pad on specimen  Step A")
# =============================================================================
r932 = run_stepA("9.3.2",
    () -> load_loyola_bulk_contact(; ndiv_c=12, ndiv_f=6, ndiv_s=4, ndiv_top=5,
        ndiv_out=6, ndiv_bot=12, ndiv_bulk=10, tipo=2, nome="rt2010_932"),
    let p = loyola_bulk_params(); p.p0_H; end,
    let p = loyola_bulk_params(); p.P; end,
    let p = loyola_bulk_params(); p.a_H; end)

# =============================================================================
# Plots
# =============================================================================
println("\n[plots] $OUTDIR")
default(linewidth=1.8, legendfontsize=8, tickfontsize=9, guidefontsize=10,
    titlefontsize=11, grid=true, framestyle=:box, legend_foreground_color=:black)

function overlay!(figtag, results, p0ref, a_ref, Pref; title_p, title_q)
    cols = Dict(:activeset => :steelblue, :gnmls => :darkorange, :proj_newton => :seagreen)
    labs = Dict(:activeset => "active-set (current)", :gnmls => "GNM-ls 2010",
        :proj_newton => "proj-Newton 2013")
    xa = collect(range(-1.15 * a_ref, 1.15 * a_ref; length=400))
    pAna = cattaneo_pressure(xa, a_ref, p0ref)

    plt_p = plot(xa ./ a_ref, pAna ./ p0ref; color=:black, label="Hertz",
        xlabel="x/a", ylabel="p / p0", title=title_p, ylims=(-0.05, 1.15))
    plt_q = plot(; xlabel="x/a", ylabel="q / (f p0)", title=title_q,
        ylims=(-0.55, 0.55))
    hline!(plt_q, [0.0]; color=:gray, linestyle=:dot, label=false)
    μ = first(values(results)).par.μ
    for lab in (:activeset, :gnmls, :proj_newton)
        haskey(results, lab) || continue
        m = results[lab].m
        ξ = m.x ./ a_ref
        scatter!(plt_p, ξ, m.p ./ p0ref; color=cols[lab], label=labs[lab],
            markersize=4, markerstrokewidth=0)
        scatter!(plt_q, ξ, m.q ./ (μ * p0ref); color=cols[lab], label=labs[lab],
            markersize=4, markerstrokewidth=0)
    end
    savefig(plt_p, joinpath(OUTDIR, figtag * "_p.png"))
    savefig(plt_q, joinpath(OUTDIR, figtag * "_q.png"))
    return nothing
end

p931 = loyola_dad5d_params()
overlay!("loyola931_A", r931, p931.p0_H, p931.a_H, p931.P;
    title_p="§9.3.1 Step A  pressure", title_q="§9.3.1 Step A  shear")
p932 = loyola_bulk_params()
overlay!("loyola932_A", r932, p932.p0_H, p932.a_H, p932.P;
    title_p="§9.3.2 Step A  pressure", title_q="§9.3.2 Step A  shear")

println("\n" * "="^72)
println(" Summary  (Step A)")
println("="^72)
for (tag, res, pref, p0r) in (("9.3.1", r931, p931.P, p931.p0_H),
                              ("9.3.2", r932, p932.P, p932.p0_H))
    println("  $tag   P_ref=$(round(pref, digits=1))  p0_ref=$(round(p0r, digits=2))")
    @printf("    %-14s %8s %9s %8s %7s %6s %5s %6s %6s\n",
        "solver", "P", "Q", "p0", "a", "st/sl", "coul", "util", "t[s]")
    for lab in (:activeset, :gnmls, :proj_newton)
        haskey(res, lab) || continue
        m, dt = res[lab].m, res[lab].dt
        @printf("    %-14s %8.2f %+9.2f %8.2f %7.4f %3d/%-2d %5d %6.3f %6.2f\n",
            lab, m.P, m.Q, m.p0, m.a, m.n_stick, m.n_slip, m.n_coul, m.util_mean, dt)
    end
end
println("\nDone.  plots in $OUTDIR")
