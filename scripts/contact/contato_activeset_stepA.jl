# =============================================================================
# Contato-faithful active-set Step A on dad_5d geometry (Loyola)
#   force ty=-cargav on upper top, bottom fixed, circular arcs, δ=0
# =============================================================================
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Statistics, Printf
using Plots

include(datadir("elastico", "two_blocks_contact.jl"))  # contact_interface_xyτ
include(datadir("elastico", "dad_5d_contact.jl"))

const OUTDIR = raw"C:\Users\lucas.s.campos\OneDrive\artigos\escritos\2026\BEM_contact"
const FIGDIR = joinpath(OUTDIR, "figures")
mkpath(FIGDIR)

function contato_newton!(prob; δ=0.0, tol=2.5e-11, maxiter=100, npg=12, verbose=true)
    ctx = BEM._contact_friction_setup(prob; method=:ntn, npg=npg)
    prep, pairs = ctx.prep, ctx.pairs
    h = [cp.gap0 - δ for cp in pairs]
    x0 = zeros(ctx.N)
    # Contato verifica initialises contato=3 (all stick) before first overwrite;
    # starting open leaves the force-loaded upper body with RBMs (singular A).
    for cp in pairs
        cp.state = 3
        cp.ut_lock = 0.0
    end
    ok = false
    for it in 1:maxiter
        # Contato: first pass uses all-stick (verifica default); verifying from
        # x=0 forces all-open and a singular free upper body under cargav.
        if it > 1
            BEM._verify_contact_states!(pairs, prep, h, x0; epsc=1e-7)
        end
        A, b = BEM._assemble_contact_system(prep, pairs, h, x0)
        x = A \ b
        dist = norm(x - x0)
        verbose && @printf("  it=%3d  err=%.4e  o/s/l=%d/%d/%d\n", it, dist,
            count(cp -> cp.state == 1, pairs),
            count(cp -> cp.state == 3, pairs),
            count(cp -> abs(cp.state) == 2, pairs))
        x0 = x
        dist < tol && (ok = true; break)
    end
    BEM._verify_contact_states!(pairs, prep, h, x0; epsc=1e-7)
    BEM._scatter_contact_solution!(prob, prep, pairs, x0)
    set_cache!(prob.regions[1]; contact_x=copy(x0))
    return (; ok, x=x0, prep, pairs, h)
end

function metrics(prob)
    fr = contact_interface_xyτ(prob)
    hp = length(fr.x) > 1 ? abs(fr.x[2] - fr.x[1]) : 1.0
    # variable panel sizes on arcs — use per-node half-widths
    xs = fr.x
    n = length(xs)
    wnode = zeros(n)
    if n >= 2
        wnode[1] = abs(xs[2] - xs[1])
        wnode[end] = abs(xs[end] - xs[end-1])
        for i in 2:n-1
            wnode[i] = 0.5 * abs(xs[i+1] - xs[i-1])
        end
    end
    p = .-fr.tn; q = .-fr.tt
    cl = abs.(fr.state) .!= 1
    P = any(cl) ? sum(p[cl] .* wnode[cl]) : 0.0
    Q = any(cl) ? sum(q[cl] .* wnode[cl]) : 0.0
    p0 = any(cl) ? maximum(p[cl]) : 0.0
    a_p0 = p0 > 0 ? 2P / (π * p0) : 0.0
    a_sup = any(cl) ? 0.5 * (maximum(xs[cl]) - minimum(xs[cl])) : 0.0
    xc = any(cl) ? mean(xs[cl]) : 0.0
    return (; fr..., p, q, P, Q, p0, a_p0, a_sup, xc, wnode,
        n_stick=count(==(3), fr.state), n_slip=count(s -> abs(s) == 2, fr.state),
        n_open=count(==(1), fr.state))
end

println("="^70)
println(" dad_5d Step A — Contato-faithful active-set")
par = loyola_dad5d_params()
@printf("P_target=%.0f  a_H=%.4f  p0_H=%.2f  R=%g  w=%g\n",
    par.P, par.a_H, par.p0_H, par.R, par.w)
println("="^70)

for (tag, μ) in (("mu0", 0.0), ("mu03", 0.3))
    println("\n--- μ = $μ ---")
    prob, _ = load_dad_5d_contact(; μ=μ, ndiv_c=16, ndiv_f=8, ndiv_s=6, ndiv_top=12,
        nome="stepA_$tag")
    @printf("pairs=%d  gap0∈[%.5f, %.5f]\n", length(prob.contacts),
        extrema(cp.gap0 for cp in prob.contacts)...)
    # dad_5d: force load already on mesh; geometric gap, δ=0
    sol = contato_newton!(prob; δ=0.0, tol=2.5e-11, maxiter=80, verbose=true)
    println(sol.ok ? "  CONVERGED" : "  NOT CONVERGED")
    m = metrics(prob)
    @printf("  P=%.2f (target %.0f, %.1f%%)\n", m.P, par.P, 100*m.P/par.P)
    @printf("  p0=%.2f  p0_H=%.2f  err=%.2f%%\n", m.p0, par.p0_H, 100*abs(m.p0-par.p0_H)/par.p0_H)
    @printf("  a_p0=%.4f  a_H=%.4f  err=%.2f%%  a_sup=%.4f\n",
        m.a_p0, par.a_H, 100*abs(m.a_p0-par.a_H)/par.a_H, m.a_sup)
    @printf("  stick/slip/open=%d/%d/%d  Q=%.3e\n", m.n_stick, m.n_slip, m.n_open, m.Q)

    ξ = (m.x .- m.xc) ./ par.a_H
    xa = collect(range(-1.3par.a_H, 1.3par.a_H; length=400))
    fig = plot(ξ, m.p ./ par.p0_H;
        color=:steelblue, linewidth=2.2, label="Julia AS",
        xlabel="x/a_H", ylabel="p/p_{0H}",
        title="dad_5d Step A  Contato-AS μ=$μ vs Hertz",
        size=(640, 360), framestyle=:box, legend=:topright,
        xlims=(-1.5, 1.5), ylims=(-0.05, 1.25))
    plot!(fig, xa ./ par.a_H, cattaneo_pressure(xa, par.a_H, par.p0_H) ./ par.p0_H;
        color=:black, linestyle=:dash, label="Hertz (dad_5d P,R,E)", linewidth=1.5)
    savefig(fig, joinpath(FIGDIR, "fig_dad5d_stepA_$tag.pdf"))
    savefig(fig, joinpath(FIGDIR, "fig_dad5d_stepA_$tag.png"))
    println("  → fig_dad5d_stepA_$tag.pdf")
end

println("\nDone.")
