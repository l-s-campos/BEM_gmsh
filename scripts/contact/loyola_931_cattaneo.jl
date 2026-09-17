# =============================================================================
# Loyola 2022 §9.3.1 — Cattaneo–Mindlin two-cylinder frictional contact
#
# Geometry / materials: Table 9.19 + dad_5d (cargav=100 on top of width 2w
#   ⇒ line load P = 2 w cargav = 1300 N/mm, matching a=1.186 mm, p0=697.8 MPa
#   of Tables 9.20–9.21).
# Pin: Body 1 top-centre u_x = 0 (Fig. 9.22). Body 2 base fully fixed.
# History A→B→C→D→E at constant P, Q ∈ [−Qmax, Qmax] with Qmax = 0.5 f P.
#
#   julia --project=. scripts/contact/loyola_931_cattaneo.jl
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

const OUTDIR = joinpath(projectdir(), "plots", "cattaneo_mindlin", "loyola931")
mkpath(OUTDIR)

relerr(num, ref) = abs(num - ref) / max(abs(ref), eps()) * 100

function l2_rel(y, yref)
    den = sqrt(sum(abs2, yref)) + eps()
    return sqrt(sum(abs2, y .- yref)) / den
end

# -----------------------------------------------------------------------------
# Two-body helpers
# -----------------------------------------------------------------------------

"""Pin Body 1 top-centre in u_x (Fig. 9.22 roller). Keeps ty as Neumann."""
function pin_upper_top_center_ux!(prob)
    dad = prob.regions[1]
    ymax = maximum(pt[2] for pt in dad.Nodes)
    thr = 1e-9 * max(abs(ymax), 1.0)
    best = 0
    bx = Inf
    for i in 1:dad.n
        abs(dad.Nodes[i][2] - ymax) <= thr || continue
        ax = abs(dad.Nodes[i][1])
        if ax < bx
            bx = ax
            best = i
        end
    end
    best == 0 && error("pin_upper_top_center_ux!: no top node")
    dad.BC[2best - 1] = 0
    dad.BV[2best - 1] = 0.0
    return best
end

function apply_PQ!(prob, tx, ty; pin=true)
    apply_dad5d_bulk_tx!(prob, tx; cargav=-ty)   # ty already signed (e.g. -100)
    pin && pin_upper_top_center_ux!(prob)
    return nothing
end

const MR = BEM.MultiRegion

function contato_newton!(prob; tol=1e-9, maxiter=80, npg=10, verbose=false)
    ctx = MR._contact_friction_setup(prob; method=:ntn, npg=npg)
    prep, pairs = ctx.prep, ctx.pairs
    h = [cp.gap0 for cp in pairs]
    x0 = has_cache(prob.regions[1], :contact_x) &&
         length(prob.regions[1].contact_x) == ctx.N ?
         collect(Float64, prob.regions[1].contact_x) : zeros(ctx.N)
    if !has_cache(prob.regions[1], :contact_x) || length(prob.regions[1].contact_x) != ctx.N
        for cp in pairs
            cp.state = 3
            cp.ut_lock = 0.0
        end
        x0 = zeros(ctx.N)
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
        dist < tol && n_flip == 0 && (ok = true; break)
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
    w[1] = abs(xs[2] - xs[1])
    w[end] = abs(xs[end] - xs[end - 1])
    for i in 2:n-1
        w[i] = 0.5 * abs(xs[i + 1] - xs[i - 1])
    end
    return w
end

function metrics(prob; μref=0.3)
    fr = contact_interface_xyτ(prob)
    xs = fr.x
    wnode = node_weights(xs)
    p = .-fr.tn
    q = .-fr.tt
    cl = abs.(fr.state) .!= 1
    P = any(cl) ? sum(p[cl] .* wnode[cl]) : 0.0
    Q = any(cl) ? sum(q[cl] .* wnode[cl]) : 0.0
    p0 = any(cl) ? maximum(p[cl]) : 0.0
    a = p0 > 0 ? 2P / (π * p0) : 0.0
    a_sup = any(cl) ? 0.5 * (maximum(xs[cl]) - minimum(xs[cl])) : 0.0
    xc = any(cl) ? sum(xs[cl] .* p[cl]) / max(sum(p[cl]), eps()) : 0.0
    dad = prob.regions[1]
    un = similar(xs)
    ut = similar(xs)
    for (k, cp) in enumerate(prob.contacts)
        i = cp.node_a
        n̂ = dad.Normal[i]
        t̂ = (-n̂[2], n̂[1])
        ux = dad.u[2i - 1]
        uy = dad.u[2i]
        # store later via perm of contact_interface
    end
    # displacements on the same order as fr.x
    un = zeros(length(xs))
    ut = zeros(length(xs))
    ux_g = zeros(length(xs))
    uy_g = zeros(length(xs))
    pairs = prob.contacts
    dadu = has_cache(dad, :u)
    rawx = [dad.Nodes[cp.node_a][1] for cp in pairs]
    perm = sortperm(rawx)
    for (kk, k) in enumerate(perm)
        cp = pairs[k]
        i = cp.node_a
        ux = dadu ? dad.u[2i - 1] : 0.0
        uy = dadu ? dad.u[2i] : 0.0
        ux_g[kk] = ux
        uy_g[kk] = uy
        # thesis un, ut: Body-1 contact in global (n ≈ −e_y, t ≈ e_x)
        un[kk] = uy
        ut[kk] = ux
    end
    return (; fr..., p, q, P, Q, p0, a, a_sup, xc, un, ut, ux_g, uy_g, wnode,
        n_stick=count(==(3), fr.state),
        n_slip=count(s -> abs(s) == 2, fr.state),
        n_open=count(==(1), fr.state))
end

# =============================================================================
println("="^72)
println(" Loyola 2022 §9.3.1  Cattaneo–Mindlin")
println(" ", Dates.now())
println("="^72)

par5 = loyola_dad5d_params()
par = loyola_cattaneo_params(; Q_over_fP=0.5, match_thesis_ap0=true)
μ = par.f
tyA = -par5.cargav          # −100 N/mm² on top
txB = par5.cargah           # +15 N/mm²  → Q/fP = 0.5
Qmax = 0.5 * μ * par5.P

@printf("Table 9.19: R=%.1f mm  w=%.1f mm  E=%.1f MPa  ν=%.2f  f=%.2f\n",
    par5.R, par5.w, par5.E, par5.ν, μ)
@printf("cargav=%.0f  ⇒ P=2w·cargav=%.1f N/mm   (Tables 9.20–9.21)\n", par5.cargav, par5.P)
@printf("Hertz: a=%.6f mm  p0=%.4f MPa   (thesis a=1.1860  p0=697.8025)\n", par5.a_H, par5.p0_H)
@printf("Qmax=0.5 f P = %.3f N/mm   tx=±%.1f N/mm²\n", Qmax, txB)

# -----------------------------------------------------------------------------
# [1] Analytical + half-plane BEM (fast reference)
# -----------------------------------------------------------------------------
println("\n[1] Analytical Cattaneo / Mindlin–Deresiewicz + half-plane BEM")
Nhp = 201
Lhp = 3.5 * par.a
xhp = collect(range(-Lhp, Lhp; length=Nhp))
ν = par.ν
G = par.E_eq * (1 - ν) / 2
hp = ElasticHalfPlane2D(G, ν; h=xhp[2] - xhp[1])

Q_path = Float64[]
ana = Dict{Symbol,Any}()
for st in par.load_steps
    push!(Q_path, st.Q)
    pA = cattaneo_pressure(xhp, par.a, par.p0)
    qA = mindlin_shear_history(xhp, par.a, par.p0, par.f, par.P, Q_path)
    c = cattaneo_c(par.a, st.Q, par.f, par.P)
    ana[st.name] = (; p=pA, q=qA, c, Q=st.Q, P=st.P)
    @printf("  ana %s  Q=%8.3f  c/a=%.4f  max|q|/p0=%.4f\n",
        st.name, st.Q, (st.name === :A ? 1.0 : c / par.a), maximum(abs, qA) / par.p0)
end

t0 = time()
hs_hist = solve_cattaneo_history_halfplane(xhp, par.R_eq, par.f, hp, par.load_steps; tol=1e-9)
dt_hs = time() - t0
hs = Dict(s.name => s for s in hs_hist)
for sol in hs_hist
    @printf("  HS  %s  a=%.4f (%.2f%%)  p0=%.2f (%.2f%%)  Ft=%.3f  stick/slip=%d/%d\n",
        sol.name, sol.a, relerr(sol.a, par.a), sol.p0, relerr(sol.p0, par.p0),
        sol.force_t, count(sol.stick), count(sol.slip))
end
@printf("  half-plane wall time %.2fs\n", dt_hs)

# -----------------------------------------------------------------------------
# [2] Two-body BEM (dad_5d cylinders, NPc ≈ 61)
# -----------------------------------------------------------------------------
println("\n[2] Two-body BEM  (quadratic, ~61 contact node-pairs)")
ndiv_c, ndiv_f, ndiv_s, ndiv_top = 12, 6, 4, 5
tipo = 2
n_leg = 6

prob, _ = load_dad_5d_contact(; μ=μ, ndiv_c=ndiv_c, ndiv_f=ndiv_f,
    ndiv_s=ndiv_s, ndiv_top=ndiv_top, tipo=tipo, nome="loyola931")
npc = length(prob.contacts)
@printf("  nodes/body = %d / %d   contact pairs NPc=%d\n",
    prob.regions[1].n, prob.regions[2].n, npc)
@printf("  gap0 ∈ [%.5f, %.5f]\n", extrema(cp.gap0 for cp in prob.contacts)...)

apply_PQ!(prob, 0.0, tyA)
corners = Dict{Symbol,Any}()
residues = Dict{Symbol,Vector{Float64}}()
t_bem = time()

println("\n  [A] P only, Q=0  (force ty, pin u_x)")
solA = contato_newton!(prob; tol=1e-9, maxiter=80, verbose=true)
mA = metrics(prob; μref=μ)
corners[:A] = mA
residues[:A] = solA.hist
_, uyA = dad5d_top_u_mean(prob)
@printf("  A  ok=%s it=%d  P=%.2f (%.1f%% of %.0f)  Q=%+.3f  p0=%.2f  a=%.4f  a_sup=%.4f  xc=%.4f  st/sl/op=%d/%d/%d\n",
    solA.ok, solA.nit, mA.P, 100 * mA.P / par5.P, par5.P, mA.Q, mA.p0, mA.a, mA.a_sup, mA.xc,
    mA.n_stick, mA.n_slip, mA.n_open)
@printf("  freeze uyA=%.6f mm\n", uyA)

# B–E: Dirichlet far-field u_x with frozen u_y. A single top-centre u_x pin
# absorbs distributed t_x (Q stays ~0); pure Neumann t_x tips the cylinder
# and collapses the patch. Rigid-top u_x is the stable equivalent of Q.
u_max = 0.06            # mm; |Q|/fP ≈ 0.5 at B (probed)
println("\n  B–E  far-field u_x ∈ [−$(u_max), $(u_max)]  uy frozen  n_leg=$n_leg")
legs = ((:B, 0.0, +u_max), (:C, +u_max, 0.0), (:D, 0.0, -u_max), (:E, -u_max, 0.0))
for (name, u0, u1) in legs
    println("\n  [$name] ux: $u0 → $u1")
    local m, sol
    for s in 1:n_leg
        t = s / n_leg
        ux = (1 - t) * u0 + t * u1
        apply_dad5d_bulk_ux!(prob, ux; uy=uyA)
        sol = contato_newton!(prob; tol=1e-9, maxiter=80, verbose=(s == n_leg))
        m = metrics(prob; μref=μ)
        if s == n_leg || s == 1
            @printf("    s=%d ux=%+.4f it=%d P=%.1f Q=%+.1f |Q|/fP=%.3f st/sl=%d/%d xc=%.3f ok=%s\n",
                s, ux, sol.nit, m.P, m.Q, abs(m.Q) / max(μ * abs(m.P), eps()),
                m.n_stick, m.n_slip, m.xc, sol.ok)
        end
    end
    corners[name] = m
    residues[name] = sol.hist
end
dt_bem = time() - t_bem
@printf("\n  two-body wall time %.1fs\n", dt_bem)

# -----------------------------------------------------------------------------
# Summary vs Tables 9.20–9.21 and analytical
# -----------------------------------------------------------------------------
println("\n" * "="^72)
println(" Summary  (analytical a=$(round(par5.a_H, digits=4))  p0=$(round(par5.p0_H, digits=2)))")
println("="^72)
@printf("%-5s %10s %10s %8s %8s %8s %8s %6s %6s %6s\n",
    "step", "P", "Q", "|Q|/fP", "p0", "a", "xc", "stick", "slip", "open")
for s in (:A, :B, :C, :D, :E)
    m = corners[s]
    @printf("%-5s %10.2f %10.2f %8.3f %8.1f %8.4f %8.4f %6d %6d %6d\n",
        s, m.P, m.Q, abs(m.Q) / max(μ * abs(m.P), eps()),
        m.p0, m.a, m.xc, m.n_stick, m.n_slip, m.n_open)
end

println("\nStep A vs thesis Tables 9.20–9.21 (analytical p0=697.8025  a=1.1860)")
mA = corners[:A]
@printf("  BEM     p0=%.3f  err=%.3f%%     a=%.4f  err=%.3f%%   (a_sup=%.4f)  NPc=%d  it=%d\n",
    mA.p0, relerr(mA.p0, 697.8025), mA.a, relerr(mA.a, 1.1860), mA.a_sup, npc, solA.nit)
@printf("  HS      p0=%.3f  err=%.3f%%     a=%.4f  err=%.3f%%\n",
    hs[:A].p0, relerr(hs[:A].p0, par.p0), hs[:A].a, relerr(hs[:A].a, par.a))

println("\nTraction L2 vs analytical (numerical P,Q; Mindlin history of Q):")
Qpath = Float64[]
for st in (:A, :B, :C, :D, :E)
    m = corners[st]
    push!(Qpath, m.Q)
    aH = sqrt(4 * par5.R * abs(m.P) / (π * par5.East))
    p0H = 2 * abs(m.P) / (π * aH)
    x = m.x .- m.xc
    pA = cattaneo_pressure(x, aH, p0H)
    qA = st === :A ? zeros(length(x)) : mindlin_shear_history(x, aH, p0H, μ, abs(m.P), Qpath)
    cl = abs.(m.state) .!= 1
    L2p = any(cl) ? l2_rel(m.p[cl], pA[cl]) : NaN
    L2q = if st === :A
        0.0
    elseif !any(cl)
        NaN
    else
        den = sqrt(sum(abs2, qA[cl]))
        num = sqrt(sum(abs2, m.q[cl] .- qA[cl]))
        den > 1e-6 * max(p0H, 1.0) ? num / den : num / max(p0H, 1.0)
    end
    @printf("  %s  err_p0=%.2f%%  L2(p)=%.4f  L2(q)=%.4f  xc=%.4f\n",
        st, relerr(m.p0, p0H), L2p, L2q, m.xc)
end

# -----------------------------------------------------------------------------
# Figures 9.25 and 9.26 (thesis layout)
#   9.25: p/p0 and q/(f p0) on one axis — that is how Loyola Fig. 9.25 is drawn
#         (inner Cattaneo plateau ≈ 1 − c/a ≈ 0.29, reverse peaks ≈ 0.5).
#   9.26: Body-1 contact u_y (un) and u_x (ut), closed pairs only.
# -----------------------------------------------------------------------------
println("\n[plots] $OUTDIR")
default(linewidth=1.8, legendfontsize=7, tickfontsize=9, guidefontsize=10,
    titlefontsize=11, grid=true, framestyle=:box, legend_foreground_color=:black)

aH = par5.a_H
p0H = par5.p0_H
fp0 = μ * p0H
xa = collect(range(-1.05aH, 1.05aH; length=500))
# prescribed A–E path (not the numerical Q, which is what the thesis analytical uses)
Qana = [st.Q for st in par.load_steps]
qAna = Dict{Symbol,Vector{Float64}}()
pAna = cattaneo_pressure(xa, aH, p0H)
let seq = Float64[]
    for (st, Q) in zip((:A, :B, :C, :D, :E), Qana)
        push!(seq, Q)
        qAna[st] = mindlin_shear_history(xa, aH, p0H, μ, par.P, seq)
    end
end

yl_925 = Dict(:B => (-0.05, 1.08), :C => (-0.62, 1.08),
              :D => (-0.72, 1.08), :E => (-0.42, 1.08))
ttl_925 = Dict(:B => "a) Load step B.", :C => "b) Load step C.",
               :D => "c) Load step D.", :E => "d) Load step E.")

function _hs_markers(st; n=33)
    s = hs[st]
    ξ = s.x ./ aH
    keep = findall(x -> abs(x) <= 1.12, ξ)
    if length(keep) > n
        keep = keep[round.(Int, range(1, length(keep); length=n))]
    end
    return ξ[keep], s.p[keep] ./ p0H, s.τ[keep] ./ fp0
end

plts = Plots.Plot[]
for st in (:B, :C, :D, :E)
    m = corners[st]
    ξ = (m.x .- m.xc) ./ aH
    cl = abs.(m.state) .!= 1 .&& abs.(ξ) .<= 1.15
    p = plot(xlabel="x/a (mm)", ylabel="t / p0", title=ttl_925[st],
        xlims=(-1.05, 1.05), ylims=yl_925[st],
        legend=false, size=(420, 360))
    plot!(p, xa ./ aH, pAna ./ p0H; color=:black, linestyle=:solid,
        marker=:none, linewidth=1.6, label="tnA")
    plot!(p, xa ./ aH, qAna[st] ./ fp0; color=:black, linestyle=:dash,
        marker=:none, linewidth=1.6, label="ttA")
    ξh, ph, qh = _hs_markers(st)
    scatter!(p, ξh, ph; marker=:star, markersize=5, color=:royalblue,
        markerstrokewidth=0.15, label="tnHS")
    scatter!(p, ξh, qh; marker=:utriangle, markersize=5, color=:royalblue,
        markerstrokecolor=:royalblue, label="ttHS")
    scatter!(p, ξ[cl], m.p[cl] ./ p0H; marker=:star, markersize=6, color=:firebrick,
        markerstrokewidth=0.15, label="tnBEM")
    scatter!(p, ξ[cl], m.q[cl] ./ fp0; marker=:utriangle, markersize=6, color=:firebrick,
        markerstrokecolor=:firebrick, label="ttBEM")
    if st === :B
        plot!(p; legend=:top, legendfontsize=6,
            background_color_legend=:white, foreground_color_legend=:black)
    end
    push!(plts, p)
end
fig925 = plot(plts...; layout=(2, 2), size=(920, 760),
    plot_title="Figure 9.25  Cattaneo–Mindlin tractions",
    plot_titlefontsize=12)
savefig(fig925, joinpath(OUTDIR, "fig_9_25.png"))
savefig(fig925, joinpath(OUTDIR, "fig_9_25.pdf"))
savefig(fig925, joinpath(OUTDIR, "tractions_BCDE.png"))
savefig(fig925, joinpath(OUTDIR, "tractions_BCDE.pdf"))

# --- Figure 9.26 ---
ttl_926 = Dict(:B => "a) Step 2.", :C => "b) Step 3.",
               :D => "c) Step 4.", :E => "d) Step 5.")
yl_926 = Dict(:B => (-0.050, 0.045), :C => (-0.055, 0.015),
              :D => (-0.055, -0.022), :E => (-0.055, 0.015))

plts = Plots.Plot[]
for st in (:B, :C, :D, :E)
    m = corners[st]
    ξ = (m.x .- m.xc) ./ aH
    cl = abs.(m.state) .!= 1 .&& abs.(ξ) .<= 1.05
    p = plot(xlabel="x/a (mm)", ylabel="u (mm)", title=ttl_926[st],
        xlims=(-1.05, 1.05), ylims=yl_926[st],
        legend=false, size=(420, 360))
    scatter!(p, ξ[cl], m.un[cl]; marker=:star, markersize=6, color=:crimson,
        markerstrokewidth=0.2, label="un BEM")
    scatter!(p, ξ[cl], m.ut[cl]; marker=:utriangle, markersize=6, color=:darkorange,
        markerstrokecolor=:darkorange, label="ut BEM")
    if st === :B
        plot!(p; legend=:top, legendfontsize=7)
    end
    push!(plts, p)
end
fig926 = plot(plts...; layout=(2, 2), size=(920, 760),
    plot_title="Figure 9.26  Cattaneo–Mindlin displacements",
    plot_titlefontsize=12)
savefig(fig926, joinpath(OUTDIR, "fig_9_26.png"))
savefig(fig926, joinpath(OUTDIR, "fig_9_26.pdf"))
savefig(fig926, joinpath(OUTDIR, "displacements_BCDE.png"))
savefig(fig926, joinpath(OUTDIR, "displacements_BCDE.pdf"))

# Step A pressure (Hertz check)
let
    m = corners[:A]
    ξ = (m.x .- m.xc) ./ aH
    pAplt = plot(xlabel="x/a", ylabel="p / p0", title="Step A — normal pressure",
        xlims=(-1.5, 1.5), ylims=(-0.02, 1.12), size=(640, 420), legend=:bottom)
    xh = collect(range(-1.5aH, 1.5aH; length=400))
    plot!(pAplt, xh ./ aH, cattaneo_pressure(xh, aH, p0H) ./ p0H;
        color=:black, linestyle=:dash, linewidth=2.2, label="Hertz")
    plot!(pAplt, hs[:A].x ./ aH, hs[:A].p ./ p0H;
        color=:forestgreen, linewidth=1.8, label="half-plane")
    plot!(pAplt, ξ, m.p ./ p0H; color=:steelblue, linewidth=1.2, marker=:circle,
        markersize=4, label="two-body BEM")
    savefig(pAplt, joinpath(OUTDIR, "pressure_A.png"))
    savefig(pAplt, joinpath(OUTDIR, "pressure_A.pdf"))
end

# keep the supporting figures
figm = plot(aspect_ratio=1, xlabel="x (mm)", ylabel="y (mm)",
    title="dad_5d mesh  NPc=$npc", size=(520, 640), legend=:topright,
    framestyle=:box, xlims=(-8, 8), ylims=(-18, 18), grid=false)
for (dad, col, lab) in ((prob.regions[1], :crimson, "Body 1"),
                        (prob.regions[2], :dodgerblue, "Body 2"))
    xs = [pt[1] for pt in dad.Nodes]
    ys = [pt[2] for pt in dad.Nodes]
    scatter!(figm, xs, ys; color=col, markersize=3, markerstrokewidth=0, label=lab)
end
savefig(figm, joinpath(OUTDIR, "mesh.png"))
savefig(figm, joinpath(OUTDIR, "mesh.pdf"))

figN = plot(xlabel="iteration", ylabel="||Δx||", yscale=:log10,
    title="Newton residue", size=(520, 360), legend=:topright)
for st in (:A, :B, :C, :D, :E)
    h = residues[st]
    isempty(h) && continue
    plot!(figN, 1:length(h), max.(h, 1e-16); label=String(st), marker=:circle, markersize=4)
end
hline!(figN, [1e-9]; color=:gray, linestyle=:dash, label="ε=1e-9")
savefig(figN, joinpath(OUTDIR, "newton_residue.png"))
savefig(figN, joinpath(OUTDIR, "newton_residue.pdf"))

open(joinpath(OUTDIR, "summary.txt"), "w") do io
    println(io, "Loyola 2022 §9.3.1 Cattaneo–Mindlin")
    println(io, Dates.now())
    @printf(io, "NPc=%d  tipo=%d  ndiv_c=%d ndiv_f=%d  two-body %.1fs  half-plane %.2fs\n",
        npc, tipo, ndiv_c, ndiv_f, dt_bem, dt_hs)
    @printf(io, "P=%.4f  a_H=%.6f  p0_H=%.4f  Qmax=%.4f\n", par5.P, par5.a_H, par5.p0_H, Qmax)
    for s in (:A, :B, :C, :D, :E)
        m = corners[s]
        @printf(io, "%s P=%.4f Q=%.4f p0=%.4f a=%.6f a_sup=%.6f stick=%d slip=%d open=%d it=%d\n",
            s, m.P, m.Q, m.p0, m.a, m.a_sup, m.n_stick, m.n_slip, m.n_open,
            length(residues[s]))
    end
end

println("\nDone. Figures → $OUTDIR")
