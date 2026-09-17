# =============================================================================
# Active-set only — Contato-faithful Step A vs Hertz
#
# Geometry inspired by Contato dad_contato_plano / sapatas:
#   large foundation (bottom) + smaller punch (top), parabolic gap
# Formulas from:
#   aplica_contato_sem_atrito_multicorpos.m   (μ=0 closed)
#   aplica_contato_com_atrito_multicorpos.m   (μ>0 stick/slip)
#   verfica_contato_com_atrito_multicorpos.m
# =============================================================================
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Statistics, Printf
using Plots

include(datadir("elastico", "two_blocks_contact.jl"))

const OUTDIR = raw"C:\Users\lucas.s.campos\OneDrive\artigos\escritos\2026\BEM_contact"
const FIGDIR = joinpath(OUTDIR, "figures")
mkpath(FIGDIR)

"""Contato-style active-set: verify → assemble → solve until ||Δx|| small."""
function contato_activeset!(prob; δ=0.0, tol=1e-10, maxiter=80, npg=12, verbose=true)
    ctx = BEM._contact_friction_setup(prob; method=:ntn, npg=npg)
    prep, pairs = ctx.prep, ctx.pairs
    h = [cp.gap0 - δ for cp in pairs]
    x = zeros(ctx.N)
    # Contato often starts with all stick
    for cp in pairs
        cp.state = 3
        cp.ut_lock = 0.0
    end
    ok = false
    hist = NamedTuple[]
    for it in 1:maxiter
        BEM._verify_contact_states!(pairs, prep, h, x; epsc=1e-9)
        states = [cp.state for cp in pairs]
        A, b = BEM._assemble_contact_system(prep, pairs, h, x)
        x_new = A \ b
        dist = norm(x_new - x)
        x .= x_new
        n_o = count(==(1), states)
        n_s = count(==(3), states)
        n_l = count(s -> abs(s) == 2, states)
        push!(hist, (; it, dist, n_o, n_s, n_l))
        verbose && it <= 3 || (verbose && (it % 5 == 0 || dist < tol)) ?
            @printf("  it=%3d  ||Δx||=%.3e  open/stick/slip=%d/%d/%d\n", it, dist, n_o, n_s, n_l) : nothing
        if dist < tol
            ok = true
            break
        end
    end
    BEM._verify_contact_states!(pairs, prep, h, x; epsc=1e-9)
    BEM._update_contact_ut_locks!(pairs, prep, h, x)
    BEM._scatter_contact_solution!(prob, prep, pairs, x)
    return (; ok, x, prep, pairs, h, hist)
end

function iface(prob)
    fr = contact_interface_xyτ(prob)
    hp = abs(fr.x[2] - fr.x[1])
    p = .-fr.tn; q = .-fr.tt
    cl = abs.(fr.state) .!= 1
    P = any(cl) ? sum(p[cl]) * hp : 0.0
    Q = any(cl) ? sum(q[cl]) * hp : 0.0
    p0 = any(cl) ? maximum(p[cl]) : 0.0
    a_p0 = p0 > 0 ? 2P / (π * p0) : 0.0
    a_sup = any(cl) ? 0.5 * (maximum(fr.x[cl]) - minimum(fr.x[cl])) : 0.0
    xc = any(cl) ? mean(fr.x[cl]) : 0.0
    return (; fr..., p, q, P, Q, p0, a_p0, a_sup, xc, hp,
        n_stick=count(==(3), fr.state), n_slip=count(s -> abs(s) == 2, fr.state),
        n_open=count(==(1), fr.state))
end

# ---- Contato-like geometry: wide foundation + punch ----
E, ν = 100.0, 0.3
props = Elasticity(E, ν, 1.0; plane_strain=true)
R = 10.0
# foundation much wider/deeper than punch (half-space-ish)
W_punch, H_punch = 4.0, 2.0
W_found, H_found = 16.0, 8.0
ndiv_p, ndiv_f = 32, 48   # punch contact denser relative to width
δn = 0.05
gap_flat = 0.03

println("="^70)
println(" Contato-faithful AS Step A  (foundation + punch, parabolic gap)")
println("="^70)

# Build manually: foundation bottom, punch top, aligned centres
function load_punch_foundation(; μ=0.0)
    # foundation: wide, contact on top
    msh_f = mesh_elastic_block(;
        x0=-(W_found - W_punch) / 2, y0=0.0, W=W_found, H=H_found,
        ndiv_x=ndiv_f, ndiv_y=max(6, ndiv_f ÷ 6), μ=μ,
        bottom_bc="0;0;0;0;1;1",
        top_bc="4;$μ;4;$μ;1;2",
        left_bc="1;0;1;0;1;3",
        right_bc="1;0;1;0;1;4",
        nome="found",
    )
    # punch: sits above with gap, load later via top-face uy (approach by δ on gap)
    msh_p = mesh_elastic_block(;
        x0=0.0, y0=H_found + gap_flat, W=W_punch, H=H_punch,
        ndiv_x=ndiv_p, ndiv_y=max(4, ndiv_p ÷ 6), μ=μ,
        bottom_bc="4;$μ;4;$μ;2;1",
        top_bc="0;0;0;0;2;2",          # fixed top of punch (δ closes gap)
        left_bc="1;0;1;0;2;3",
        right_bc="1;0;1;0;2;4",
        nome="punch",
    )
    dad_f = format2d(msh_f, props; pontointerno=false)
    dad_p = format2d(msh_p, props; pontointerno=false)
    dad_f.name = "foundation"; dad_p.name = "punch"
    prob = MultiRegionProblem([dad_f, dad_p]; name="punch_found")
    # pair: slave=punch bottom (reg 2), master=foundation top (reg 1)
    # pair_contacts uses nearest — punch nodes to foundation
    pair_contacts!(prob; method=:ntn, slave_reg=2, master_reg=1)
    apply_parabolic_contact_gap!(prob; R=R, method=:ntn, slave_reg=2, master_reg=1)
    # force pair μ
    for cp in prob.contacts
        cp.μ = μ
    end
    return prob
end

for (label, μ) in (("frictionless μ=0", 0.0), ("frictional μ=0.3", 0.3))
    println("\n--- $label ---")
    prob = load_punch_foundation(; μ=μ)
    @printf("pairs=%d  gap0∈[%.4f, %.4f]  h=gap0-δ∈[%.4f, %.4f]\n",
        length(prob.contacts),
        extrema(cp.gap0 for cp in prob.contacts)...,
        extrema(cp.gap0 - δn for cp in prob.contacts)...)
    sol = contato_activeset!(prob; δ=δn, tol=1e-10, maxiter=80, verbose=true)
    m = iface(prob)
    Eeq = E / (2 * (1 - ν^2))
    aH = sqrt(4 * m.P * R / (π * Eeq))
    p0H = 2 * m.P / (π * aH)
    @printf("P=%.4f  p0=%.4f p0H=%.4f (err %.2f%%)\n", m.P, m.p0, p0H, 100*abs(m.p0-p0H)/p0H)
    @printf("a_p0=%.4f a_sup=%.4f aH=%.4f (a_p0 err %.2f%%)\n",
        m.a_p0, m.a_sup, aH, 100*abs(m.a_p0-aH)/aH)
    @printf("closed stick/slip/open=%d/%d/%d  Q=%.4e\n",
        m.n_stick, m.n_slip, m.n_open, m.Q)

    # plot
    ξ = (m.x .- m.xc) ./ aH
    xa = collect(range(-1.25aH, 1.25aH; length=400))
    fig = plot(ξ, m.p ./ p0H;
        color=:steelblue, linewidth=2, label="Julia AS (Contato formulas)",
        xlabel="(x-x_c)/a_H", ylabel="p/p_{0H}",
        title="Contato-faithful AS Step A ($label)",
        size=(640, 360), framestyle=:box, legend=:topright,
        xlims=(-1.5, 1.5), ylims=(-0.05, 1.2))
    plot!(fig, xa ./ aH, cattaneo_pressure(xa, aH, p0H) ./ p0H;
        color=:black, linestyle=:dash, label="Hertz", linewidth=1.5)
    tag = μ == 0 ? "fricless" : "mu03"
    savefig(fig, joinpath(FIGDIR, "fig_mb_stepA_$(tag).pdf"))
    savefig(fig, joinpath(FIGDIR, "fig_mb_stepA_$(tag).png"))
    println("  wrote fig_mb_stepA_$(tag).pdf")
end

println("\nDone.")
