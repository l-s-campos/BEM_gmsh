# =============================================================================
# Contato-AS fretting B–C–D–E on dad_5d — IGABEM-bulk protocol
#
# Source: Desktop/IGABEM-bulk/Elastico_bulk.jl + Contato.jl
#   • Normal steps (A): milocal = 0  (frictionless Hertz, symmetric q≡0)
#   • Fretting steps:   milocal = mi, ramp cargah / bulk slip
#   • Stick freezes relative tangential gap between steps (ut_lock)
#   • Far-field fretting: frozen uy from A + ramp ux  (stable, no tip-over)
#     (IGABEM applies cargah on side curves; equivalent bulk shear drive)
# =============================================================================
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Statistics, Printf
using Plots

include(datadir("elastico", "two_blocks_contact.jl"))
include(datadir("elastico", "dad_5d_contact.jl"))

const OUTDIR = raw"C:\Users\lucas.s.campos\OneDrive\artigos\escritos\2026\BEM_contact"
const FIGDIR = joinpath(OUTDIR, "figures")
mkpath(FIGDIR)

par = loyola_dad5d_params()
μ = par.μ
n_leg = 12                 # fretting substeps per leg (IGABEM npassos-style)
u_max = 0.05               # far-field ux amplitude
tipo = 2                   # quadratic discontinuous (Contato / IGABEM order-ish)
ndiv_c, ndiv_f = 24, 10

println("="^70)
println(" dad_5d Contato-AS fretting — IGABEM-bulk protocol")
@printf("P=%.0f  a_H=%.4f  p0_H=%.1f  μ=%.2f  u_max=%.4f  n_leg=%d  tipo=%d\n",
    par.P, par.a_H, par.p0_H, μ, u_max, n_leg, tipo)
println(" A: milocal=0 (frictionless) → B–E: milocal=μ, ux ramp, uy frozen")
println("="^70)

function contato_newton!(prob; tol=1e-9, maxiter=100, npg=12, epsc=1e-7)
    ctx = BEM._contact_friction_setup(prob; method=:ntn, npg=npg)
    prep, pairs = ctx.prep, ctx.pairs
    h = [cp.gap0 for cp in pairs]
    x0 = has_cache(prob.regions[1], :contact_x) &&
         length(prob.regions[1].contact_x) == ctx.N ?
         collect(Float64, prob.regions[1].contact_x) : zeros(ctx.N)
    if !has_cache(prob.regions[1], :contact_x) || length(prob.regions[1].contact_x) != ctx.N
        # IGABEM newtonSubReg starts from ones; we use all-stick zeros (better conditioned)
        for cp in pairs
            cp.state = 3
            # keep existing ut_lock (Mindlin residual); only wipe if brand new
        end
        x0 = zeros(ctx.N)
    end
    ok = false
    nit = 0
    for it in 1:maxiter
        nit = it
        it > 1 && BEM._verify_contact_states!(pairs, prep, h, x0; epsc=epsc)
        A, b = BEM._assemble_contact_system(prep, pairs, h, x0)
        x = A \ b
        dist = norm(x - x0)
        x0 = x
        dist < tol && (ok = true; break)
    end
    BEM._verify_contact_states!(pairs, prep, h, x0; epsc=epsc)
    BEM._update_contact_ut_locks!(pairs, prep, h, x0)
    BEM._scatter_contact_solution!(prob, prep, pairs, x0)
    set_cache!(prob.regions[1]; contact_x=copy(x0))
    return ok, nit
end

function metrics(prob; μref=μ)
    fr = contact_interface_xyτ(prob)
    xs = fr.x; n = length(xs)
    wnode = ones(n)
    if n >= 2
        wnode[1] = abs(xs[2] - xs[1]); wnode[end] = abs(xs[end] - xs[end - 1])
        for i in 2:n-1
            wnode[i] = 0.5 * abs(xs[i + 1] - xs[i - 1])
        end
    end
    p = .-fr.tn; q = .-fr.tt
    cl = abs.(fr.state) .!= 1
    P = any(cl) ? sum(p[cl] .* wnode[cl]) : 0.0
    Q = any(cl) ? sum(q[cl] .* wnode[cl]) : 0.0
    p0 = any(cl) ? maximum(p[cl]) : 0.0
    a = p0 > 0 ? 2P / (π * p0) : 0.0
    xc = any(cl) ? sum(xs[cl] .* p[cl]) / max(sum(p[cl]), eps()) : 0.0
    qrms = any(cl) ? sqrt(mean(abs2, q[cl])) : 0.0
    # even / odd decomposition of q (symmetry diagnostic)
    odd = even = 0.0; nsym = 0
    for i in findall(cl)
        xs[i] < -1e-9 || continue
        j = argmin(abs.(xs .+ xs[i]))
        abs(fr.state[j]) == 1 && continue
        qi, qj = q[i], q[j]
        even += ((qi + qj) / 2)^2
        odd  += ((qi - qj) / 2)^2
        nsym += 1
    end
    util = 0.0
    for i in findall(cl)
        util = max(util, abs(fr.tt[i]) / (μref * abs(fr.tn[i]) + 1e-15))
    end
    return (; fr..., p, q, P, Q, p0, a, xc, qrms, max_util=util,
        n_stick=count(==(3), fr.state),
        n_slip=count(s -> abs(s) == 2, fr.state),
        n_open=count(==(1), fr.state),
        q_odd=sqrt(odd / max(nsym, 1)),
        q_even=sqrt(even / max(nsym, 1)))
end

function main()
    prob, _ = load_dad_5d_contact(; μ=μ, ndiv_c=ndiv_c, ndiv_f=ndiv_f, ndiv_s=6,
        ndiv_top=12, tipo=tipo, nome="bcde_igabem")
    corners = Dict{Symbol,Any}()
    ux_hist = Float64[]; Q_hist = Float64[]

    # ── A: frictionless normal load (IGABEM milocal = 0) ─────────────────
    println("\n[A] normal load, milocal = 0 (frictionless Hertz)")
    set_contact_mu!(prob, 0.0)
    reset_contact_ut_locks!(prob)
    ok, nit = contato_newton!(prob)
    mA = metrics(prob; μref=μ)   # util vs design μ
    corners[:A] = mA
    _, uyA = dad5d_top_u_mean(prob)
    push!(ux_hist, 0.0); push!(Q_hist, mA.Q)
    @printf("A  it=%d P=%.2f (%.1f%%) Q=%+.3f p0=%.1f a=%.4f xc=%.4f st/sl=%d/%d\n",
        nit, mA.P, 100 * mA.P / par.P, mA.Q, mA.p0, mA.a, mA.xc, mA.n_stick, mA.n_slip)
    @printf("   q_odd=%.3f q_even=%.3f  (expect ~0 after μ=0)\n", mA.q_odd, mA.q_even)
    @printf("   freeze uyA = %.6f\n", uyA)

    # IGABEM: do NOT resolve at μ>0 with zero shear — that reintroduces stick–Poisson
    # odd shear. Switch milocal=μ and jump straight into the fretting ramp.
    set_contact_mu!(prob, μ)
    reset_contact_ut_locks!(prob)
    # pin top uy; first fretting increment sets ux
    apply_dad5d_bulk_ux!(prob, 0.0; uy=uyA)

    # ── B–E fretting cycle (IGABEM: milocal=mi after normal steps) ────────
    legs = ((:B, 0.0, +u_max), (:C, +u_max, 0.0),
            (:D, 0.0, -u_max), (:E, -u_max, 0.0))
    for (name, u0, u1) in legs
        println("\n[$name] ux: $u0 → $u1  (n_leg=$n_leg, milocal=μ)")
        local m
        for s in 1:n_leg
            t = s / n_leg
            ux = (1 - t) * u0 + t * u1
            apply_dad5d_bulk_ux!(prob, ux; uy=uyA)
            ok_s, nit = contato_newton!(prob)
            m = metrics(prob)
            push!(ux_hist, ux); push!(Q_hist, m.Q)
            if s == n_leg || s == 1 || s % 4 == 0
                @printf("  s=%02d ux=%+.4f it=%d P=%.1f Q=%+.1f xc=%.3f st/sl=%d/%d odd=%.1f even=%.1f util=%.2f\n",
                    s, ux, nit, m.P, m.Q, m.xc, m.n_stick, m.n_slip, m.q_odd, m.q_even, m.max_util)
            end
        end
        corners[name] = m
    end

    # ── summary ──────────────────────────────────────────────────────────
    println("\n" * "="^70)
    @printf("%-5s %10s %10s %8s %8s %8s %8s %6s %6s\n",
        "step", "P", "Q", "|Q|/fP", "qrms", "odd", "xc", "stick", "slip")
    for s in (:A, :B, :C, :D, :E)
        m = corners[s]
        @printf("%-5s %10.2f %10.2f %8.3f %8.1f %8.1f %8.3f %6d %6d\n",
            s, m.P, m.Q, abs(m.Q) / max(μ * abs(m.P), eps()), m.qrms,
            m.q_odd, m.xc, m.n_stick, m.n_slip)
    end

    println("\nCHECKS (IGABEM fretting topology)")
    okA = abs(corners[:A].Q) < 1.0 && corners[:A].q_odd < 1.0
    okB = abs(corners[:B].Q) > 10
    okCq = abs(corners[:C].Q) < 0.5 * abs(corners[:B].Q)
    okCr = corners[:C].qrms > 0.15 * corners[:B].qrms
    okD = sign(corners[:D].Q) != sign(corners[:B].Q)
    okXc = abs(corners[:B].xc) < 0.25 * par.a_H
    okSym = corners[:B].q_odd < 0.35 * max(corners[:B].q_even, 1.0)
    println("  A Q≈0 & q_odd small: ", okA ? "PASS" : "FAIL")
    println("  B |Q| grows:         ", okB ? "PASS" : "FAIL")
    println("  C |Q| drops:         ", okCq ? "PASS" : "FAIL")
    println("  C residual qrms:     ", okCr ? "PASS" : "FAIL")
    println("  D Q reverses:        ", okD ? "PASS" : "FAIL")
    println("  B patch centered:    ", okXc ? "PASS" : "FAIL")
    println("  B even-dominated q:  ", okSym ? "PASS" : "FAIL")

    println("\nvs Hertz/Cattaneo/Mindlin at numerical P:")
    East = par.East; R = par.R
    Qpath = Float64[]
    for st in (:A, :B, :C, :D, :E)
        m = corners[st]
        push!(Qpath, m.Q)
        aH = sqrt(4 * R * abs(m.P) / (π * East))
        p0H = 2 * abs(m.P) / (π * aH)
        x = m.x .- m.xc
        pA = cattaneo_pressure(x, aH, p0H)
        qA = st === :A ? zeros(length(x)) :
             st === :B ? cattaneo_shear(x, aH, p0H, m.Q, μ, abs(m.P)) :
             mindlin_shear_history(x, aH, p0H, μ, abs(m.P), Qpath)
        cl = abs.(m.state) .!= 1
        L2p = any(cl) ? sqrt(sum(abs2, m.p[cl] .- pA[cl])) / (sqrt(sum(abs2, pA[cl])) + eps()) : NaN
        # guard: residual Mindlin at Q≈0 can have tiny ||qA|| → report abs L2
    L2q = st === :A ? 0.0 : begin
        if !any(cl)
            NaN
        else
            num = sqrt(sum(abs2, m.q[cl] .- qA[cl]))
            den = sqrt(sum(abs2, qA[cl]))
            den > 1e-6 * max(p0H, 1.0) ? num / den : num / max(p0H, 1.0)
        end
    end
        @printf("  %s  err_p0=%.1f%%  L2(p)=%.3f  L2(q)=%.3f  xc=%.4f  odd=%.1f\n",
            st, 100 * abs(m.p0 - p0H) / max(p0H, eps()), L2p, L2q, m.xc, m.q_odd)
    end

    # figures — raw nodal (no smoothing)
    fig = plot(ux_hist, Q_hist;
        color=:steelblue, linewidth=2, label="",
        xlabel="u_x^{top}", ylabel="Q",
        title="dad_5d AS hysteresis (IGABEM path)",
        size=(600, 360), framestyle=:box, legend=false)
    scatter!(fig, ux_hist, Q_hist; color=:steelblue, markersize=4, label="")
    hline!(fig, [0.0]; color=:gray70, linestyle=:dot, label="")
    vline!(fig, [0.0]; color=:gray70, linestyle=:dot, label="")
    savefig(fig, joinpath(FIGDIR, "fig_dad5d_hysteresis.pdf"))
    savefig(fig, joinpath(FIGDIR, "fig_dad5d_hysteresis.png"))

    plts = Plots.Plot[]
    for (j, st) in enumerate((:B, :C))
        m = corners[st]; a = max(m.a, eps()); p0 = max(m.p0, eps())
        ξ = (m.x .- m.xc) ./ a
        p = plot(;
            xlabel="x/a", ylabel="q/p_0", title="dad_5d AS shear $st",
            size=(360, 340), framestyle=:box,
            legend=(j == 1 ? :topright : false))
        if st === :B
            qref = cattaneo_shear(ξ .* a, a, p0, m.Q, μ, abs(m.P))
            plot!(p, ξ, qref ./ p0; color=:black, linestyle=:dash, label="Cattaneo", linewidth=1.5)
        else
            mB = corners[:B]
            xr = collect(range(-1.2mB.a, 1.2mB.a; length=300))
            qref = mindlin_shear_history(xr, mB.a, mB.p0, μ, abs(mB.P), [0.0, mB.Q, 0.0])
            plot!(p, xr ./ max(mB.a, eps()), qref ./ max(mB.p0, eps());
                color=:black, linestyle=:dash, label="Mindlin", linewidth=1.5)
        end
        plot!(p, ξ, m.q ./ p0; color=:steelblue, linewidth=1.5, label="AS raw")
        scatter!(p, ξ, m.q ./ p0; color=:steelblue, markersize=3, label="")
        push!(plts, p)
    end
    fig = plot(plts...; layout=(1, 2), size=(720, 340))
    savefig(fig, joinpath(FIGDIR, "fig_dad5d_shear_BC.pdf"))
    savefig(fig, joinpath(FIGDIR, "fig_dad5d_shear_BC.png"))

    println("\nDone → ", FIGDIR)
end

main()
