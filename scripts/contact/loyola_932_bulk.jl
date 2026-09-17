# =============================================================================
# Loyola 2022 §9.3.2 — bulk-stress fretting (cylindrical pad on flat specimen)
#
# Fig. 9.31 / Table 9.23.  Q and B in phase.  Recreates Figs. 9.34 and 9.36.
#
#   julia --project=. scripts/contact/loyola_932_bulk.jl
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

const OUTDIR = joinpath(projectdir(), "plots", "cattaneo_mindlin", "loyola932")
mkpath(OUTDIR)

relerr(num, ref) = abs(num - ref) / max(abs(ref), eps()) * 100

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
        if dist < tol && n_flip == 0
            ok = true
            break
        elseif it >= 10 && n_flip <= 2 && dist < 0.2
            # two edge pairs open↔slip; P/Q already stable
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

function metrics(prob)
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
    dadu = has_cache(dad, :u)
    un = zeros(length(xs)); ut = zeros(length(xs))
    rawx = [dad.Nodes[cp.node_a][1] for cp in prob.contacts]
    perm = sortperm(rawx)
    for (kk, k) in enumerate(perm)
        i = prob.contacts[k].node_a
        ux = dadu ? dad.u[2i - 1] : 0.0
        uy = dadu ? dad.u[2i] : 0.0
        un[kk] = uy
        ut[kk] = ux
    end
    return (; fr..., p, q, P, Q, p0, a, a_sup, xc, un, ut, wnode,
        n_stick=count(==(3), fr.state),
        n_slip=count(s -> abs(s) == 2, fr.state),
        n_open=count(==(1), fr.state))
end

# =============================================================================
println("="^72)
println(" Loyola 2022 §9.3.2  bulk-stress fretting")
println(" ", Dates.now())
println("="^72)

par = loyola_bulk_params()
μ = par.μ
@printf("R=%.1f mm  w=%.1f mm  E=%.1f MPa  ν=%.2f  f=%.2f\n", par.R, par.w, par.E, par.ν, μ)
@printf("cargav=%.0f  ⇒ P=2w·cargav=%.1f N/mm\n", par.cargav, par.P)
@printf("Hertz cylinder-on-flat: a=%.4f mm  p0=%.2f MPa\n", par.a_H, par.p0_H)
@printf("Thesis Tables 9.24–9.25: a=%.4f mm  p0=%.2f MPa\n", par.a_th, par.p0_th)
@printf("Qmax=0.5 f P=%.2f N/mm   σB=%.1f MPa   e/a=%.4f (Nowell)\n",
    par.Qmax, par.σB, nowell_e(par.a_H, par.σB, μ, par.p0_H) / par.a_H)

# -----------------------------------------------------------------------------
# [1] Analytical + half-plane (cylinder on flat, R*=R)
# -----------------------------------------------------------------------------
println("\n[1] Analytical Nowell / Mindlin + half-plane BEM")
Nhp = 201
Lhp = 3.5 * par.a_H
xhp = collect(range(-Lhp, Lhp; length=Nhp))
G = par.E_eq * (1 - par.ν) / 2
hp = ElasticHalfPlane2D(G, par.ν; h=xhp[2] - xhp[1])

load_steps = (
    (name=:A, P=par.P, Q=0.0, σ=0.0),
    (name=:B, P=par.P, Q=+par.Qmax, σ=+par.σB),
    (name=:C, P=par.P, Q=0.0, σ=0.0),
    (name=:D, P=par.P, Q=-par.Qmax, σ=-par.σB),
    (name=:E, P=par.P, Q=0.0, σ=0.0),
)
Q_path = Float64[]
ana = Dict{Symbol,Any}()
for st in load_steps
    push!(Q_path, st.Q)
    pA = cattaneo_pressure(xhp, par.a_H, par.p0_H)
    e = nowell_e(par.a_H, st.σ, μ, par.p0_H)
    qA = if st.name === :A
        zeros(length(xhp))
    elseif st.name === :B || st.name === :D
        nowell_shear(xhp, par.a_H, par.p0_H, st.Q, μ, par.P, e)
    else
        mindlin_shear_history(xhp, par.a_H, par.p0_H, μ, par.P, Q_path)
    end
    c = cattaneo_c(par.a_H, st.Q, μ, par.P)
    ana[st.name] = (; p=pA, q=qA, c, e, Q=st.Q, P=st.P, σ=st.σ)
    @printf("  ana %s  Q=%8.2f  σ=%+5.1f  c/a=%.3f  e/a=%.4f  max|q|/(f p0)=%.3f\n",
        st.name, st.Q, st.σ, (st.name === :A ? 1.0 : c / par.a_H), e / par.a_H,
        maximum(abs, qA) / max(μ * par.p0_H, eps()))
end

t0 = time()
hs_hist = solve_cattaneo_history_halfplane(
    xhp, par.R_eq, μ, hp,
    NamedTuple[(; name=st.name, P=st.P, Q=st.Q) for st in load_steps];
    tol=1e-9)
dt_hs = time() - t0
hs = Dict(s.name => s for s in hs_hist)
for sol in hs_hist
    @printf("  HS  %s  a=%.4f (%.2f%% vs %.4f)  p0=%.2f (%.2f%% vs %.2f)  Ft=%.2f\n",
        sol.name, sol.a, relerr(sol.a, par.a_H), par.a_H,
        sol.p0, relerr(sol.p0, par.p0_H), par.p0_H, sol.force_t)
end
@printf("  half-plane wall time %.2fs\n", dt_hs)

# -----------------------------------------------------------------------------
# [2] Two-body BEM (pad + specimen)
# -----------------------------------------------------------------------------
println("\n[2] Two-body BEM  (quadratic, ~61 contact node-pairs)")
ndiv_c, ndiv_f = 12, 6
n_leg = 6

prob, _ = load_loyola_bulk_contact(; μ=μ, ndiv_c=ndiv_c, ndiv_f=ndiv_f,
    ndiv_s=4, ndiv_top=5, ndiv_out=6, ndiv_bot=12, ndiv_bulk=10,
    tipo=2, nome="loyola932")
npc = length(prob.contacts)
@printf("  pad n=%d  specimen n=%d  NPc=%d  gap0 ∈ [%.5f, %.5f]\n",
    prob.regions[1].n, prob.regions[2].n, npc,
    extrema(cp.gap0 for cp in prob.contacts)...)

corners = Dict{Symbol,Any}()
residues = Dict{Symbol,Vector{Float64}}()
t_bem = time()

println("\n  [A] P only, Q=0, B=0")
# Top face is pure Neumann (tx=0, ty=-P). Without a holder u_x pin the pad
# has only frictional stick as an x-restraint and two edge pairs chatter
# open ↔ slip forever. Pin is overwritten by apply_pad_top_mixed! on B–E.
let
    dad = prob.regions[1]
    ymax = maximum(pt[2] for pt in dad.Nodes)
    best, bx = 0, Inf
    for i in 1:dad.n
        abs(dad.Nodes[i][2] - ymax) <= 1e-9 * max(abs(ymax), 1.0) || continue
        ax = abs(dad.Nodes[i][1])
        ax < bx && (bx = ax; best = i)
    end
    dad.BC[2best-1] = 0
    dad.BV[2best-1] = 0.0
end
solA = contato_newton!(prob; tol=1e-9, maxiter=80, verbose=true)
mA = metrics(prob)
corners[:A] = mA
residues[:A] = solA.hist
_, uyA = pad_top_u_mean(prob)
@printf("  A  ok=%s it=%d  P=%.2f (%.1f%% of %.0f)  Q=%+.2f  p0=%.2f  a=%.4f  a_sup=%.4f  xc=%.4f  st/sl/op=%d/%d/%d\n",
    solA.ok, solA.nit, mA.P, 100 * mA.P / par.P, par.P, mA.Q, mA.p0, mA.a, mA.a_sup, mA.xc,
    mA.n_stick, mA.n_slip, mA.n_open)
@printf("  freeze uyA=%.6f mm\n", uyA)

# Pad holder (Fig. 9.31): mixed (t_x, u_y=uyA) on the pad top — force Q,
# frozen approach. Specimen bulk t_x = B in phase. Q_line = tx * 2w;
# tx_max = cargah = 15 ⇒ Qmax = 195 = 0.5 f P.
tx_max = par.cargah
println("\n  B–E  pad mixed (tx, uyA)  tx ∈ [−$tx_max, $tx_max]  B ∈ [−$(par.σB), $(par.σB)]  in phase")
legs = ((:B, 0.0, +tx_max, 0.0, +par.σB),
        (:C, +tx_max, 0.0, +par.σB, 0.0),
        (:D, 0.0, -tx_max, 0.0, -par.σB),
        (:E, -tx_max, 0.0, -par.σB, 0.0))
for (name, t0, t1, b0, b1) in legs
    println("\n  [$name] tx: $t0 → $t1   B: $b0 → $b1")
    local m, sol
    for s in 1:n_leg
        t = s / n_leg
        tx = (1 - t) * t0 + t * t1
        σ = (1 - t) * b0 + t * b1
        apply_pad_top_mixed!(prob, tx; uy=uyA)
        apply_specimen_bulk!(prob, σ)
        sol = contato_newton!(prob; tol=1e-9, maxiter=80, verbose=(s == n_leg))
        m = metrics(prob)
        if s == n_leg || s == 1
            @printf("    s=%d tx=%+.2f B=%+.1f it=%d P=%.1f Q=%+.1f |Q|/fP=%.3f st/sl=%d/%d xc=%.3f ok=%s\n",
                s, tx, σ, sol.nit, m.P, m.Q, abs(m.Q) / max(μ * abs(m.P), eps()),
                m.n_stick, m.n_slip, m.xc, sol.ok)
        end
    end
    corners[name] = m
    residues[name] = sol.hist
end
dt_bem = time() - t_bem
@printf("\n  two-body wall time %.1fs\n", dt_bem)

# -----------------------------------------------------------------------------
println("\n" * "="^72)
println(" Summary  (Hertz a=$(round(par.a_H, digits=4))  p0=$(round(par.p0_H, digits=2)); thesis a=$(par.a_th) p0=$(par.p0_th))")
println("="^72)
@printf("%-5s %10s %10s %8s %8s %8s %8s %6s %6s %6s\n",
    "step", "P", "Q", "|Q|/fP", "p0", "a", "xc", "stick", "slip", "open")
for s in (:A, :B, :C, :D, :E)
    m = corners[s]
    @printf("%-5s %10.2f %10.2f %8.3f %8.1f %8.4f %8.4f %6d %6d %6d\n",
        s, m.P, m.Q, abs(m.Q) / max(μ * abs(m.P), eps()),
        m.p0, m.a, m.xc, m.n_stick, m.n_slip, m.n_open)
end

mA = corners[:A]
println("\nStep A vs thesis Tables 9.24–9.25 (p0=486.92  a=1.6997)")
@printf("  BEM     p0=%.3f  err_th=%.3f%%  err_H=%.3f%%     a=%.4f  err_th=%.3f%%  a_sup=%.4f  NPc=%d  it=%d\n",
    mA.p0, relerr(mA.p0, par.p0_th), relerr(mA.p0, par.p0_H),
    mA.a, relerr(mA.a, par.a_th), mA.a_sup, npc, solA.nit)
@printf("  HS      p0=%.3f  err_H=%.3f%%     a=%.4f  err_H=%.3f%%\n",
    hs[:A].p0, relerr(hs[:A].p0, par.p0_H), hs[:A].a, relerr(hs[:A].a, par.a_H))

# -----------------------------------------------------------------------------
# Figures 9.34 and 9.36
# -----------------------------------------------------------------------------
println("\n[plots] $OUTDIR")
default(linewidth=1.8, legendfontsize=7, tickfontsize=9, guidefontsize=10,
    titlefontsize=11, grid=true, framestyle=:box, legend_foreground_color=:black)

aH = par.a_H
p0H = par.p0_H
fp0 = μ * p0H
xa = collect(range(-1.05aH, 1.05aH; length=500))
pAna = cattaneo_pressure(xa, aH, p0H)
qAna = Dict{Symbol,Vector{Float64}}()
let seq = Float64[]
    for st in load_steps
        push!(seq, st.Q)
        e = nowell_e(aH, st.σ, μ, p0H)
        qAna[st.name] = if st.name === :A
            zeros(length(xa))
        elseif st.name === :B || st.name === :D
            nowell_shear(xa, aH, p0H, st.Q, μ, par.P, e)
        else
            mindlin_shear_history(xa, aH, p0H, μ, par.P, seq)
        end
    end
end

yl_934 = Dict(:B => (-0.05, 1.08), :C => (-0.62, 1.08),
              :D => (-0.72, 1.08), :E => (-0.42, 1.08))
ttl_934 = Dict(:B => "a) Load step B.", :C => "b) Load step C.",
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
    p = plot(xlabel="x/a (mm)", ylabel="t / p0", title=ttl_934[st],
        xlims=(-1.05, 1.05), ylims=yl_934[st], legend=false, size=(420, 360))
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
    st === :B && plot!(p; legend=:top, legendfontsize=6,
        background_color_legend=:white, foreground_color_legend=:black)
    push!(plts, p)
end
fig934 = plot(plts...; layout=(2, 2), size=(920, 760),
    plot_title="Figure 9.34  Bulk-stress tractions", plot_titlefontsize=12)
savefig(fig934, joinpath(OUTDIR, "fig_9_34.png"))
savefig(fig934, joinpath(OUTDIR, "fig_9_34.pdf"))

ttl_936 = Dict(:B => "a) Step 2.", :C => "b) Step 3.",
               :D => "c) Step 4.", :E => "d) Step 5.")
yl_936 = Dict(:B => (-0.055, 0.015), :C => (-0.055, 0.015),
              :D => (-0.055, 0.015), :E => (-0.055, 0.015))

plts = Plots.Plot[]
for st in (:B, :C, :D, :E)
    m = corners[st]
    ξ = (m.x .- m.xc) ./ aH
    cl = abs.(m.state) .!= 1 .&& abs.(ξ) .<= 1.05
    p = plot(xlabel="x/a (mm)", ylabel="u (mm)", title=ttl_936[st],
        xlims=(-1.05, 1.05), ylims=yl_936[st], legend=false, size=(420, 360))
    scatter!(p, ξ[cl], m.un[cl]; marker=:star, markersize=6, color=:crimson,
        markerstrokewidth=0.2, label="un BEM")
    scatter!(p, ξ[cl], m.ut[cl]; marker=:utriangle, markersize=6, color=:darkorange,
        markerstrokecolor=:darkorange, label="ut BEM")
    st === :B && plot!(p; legend=:top, legendfontsize=7)
    push!(plts, p)
end
fig936 = plot(plts...; layout=(2, 2), size=(920, 760),
    plot_title="Figure 9.36  Bulk-stress displacements", plot_titlefontsize=12)
savefig(fig936, joinpath(OUTDIR, "fig_9_36.png"))
savefig(fig936, joinpath(OUTDIR, "fig_9_36.pdf"))

# Step A pressure
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

# mesh Fig. 9.33
figm = plot(aspect_ratio=1, xlabel="x (mm)", ylabel="y (mm)",
    title="bulk mesh  NPc=$npc", size=(640, 420), legend=:topright,
    framestyle=:box, xlims=(-15, 15), ylims=(-15, 8), grid=false)
for (dad, col, lab) in ((prob.regions[1], :crimson, "pad"),
                        (prob.regions[2], :dodgerblue, "specimen"))
    xs = [pt[1] for pt in dad.Nodes]
    ys = [pt[2] for pt in dad.Nodes]
    scatter!(figm, xs, ys; color=col, markersize=3, markerstrokewidth=0, label=lab)
end
savefig(figm, joinpath(OUTDIR, "mesh.png"))
savefig(figm, joinpath(OUTDIR, "mesh.pdf"))

figN = plot(xlabel="iteration", ylabel="||Δx||", yscale=:log10,
    title="Newton residue (Fig. 9.39)", size=(520, 360), legend=:topright)
for st in (:A, :B, :C, :D, :E)
    h = residues[st]
    isempty(h) && continue
    plot!(figN, 1:length(h), max.(h, 1e-16); label=String(st), marker=:circle, markersize=4)
end
hline!(figN, [1e-9]; color=:gray, linestyle=:dash, label="ε=1e-9")
savefig(figN, joinpath(OUTDIR, "newton_residue.png"))
savefig(figN, joinpath(OUTDIR, "newton_residue.pdf"))

open(joinpath(OUTDIR, "summary.txt"), "w") do io
    println(io, "Loyola 2022 §9.3.2 bulk stress")
    println(io, Dates.now())
    @printf(io, "NPc=%d  two-body %.1fs  half-plane %.2fs\n", npc, dt_bem, dt_hs)
    @printf(io, "P=%.4f  a_H=%.6f  p0_H=%.4f  a_th=%.4f  p0_th=%.4f\n",
        par.P, par.a_H, par.p0_H, par.a_th, par.p0_th)
    for s in (:A, :B, :C, :D, :E)
        m = corners[s]
        @printf(io, "%s P=%.4f Q=%.4f p0=%.4f a=%.6f a_sup=%.6f stick=%d slip=%d open=%d\n",
            s, m.P, m.Q, m.p0, m.a, m.a_sup, m.n_stick, m.n_slip, m.n_open)
    end
end

println("\nDone. Figures → $OUTDIR")
