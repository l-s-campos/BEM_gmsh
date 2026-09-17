# =============================================================================
# Multi-body fretting for BEM_contact paper  (fast, fretting=true sides free)
#
# Critical fix: top-block side rollers (ux=0) were killing bulk fretting shear.
# load_two_blocks_contact(; fretting=true) uses traction-free top sides.
# =============================================================================
using DrWatson
@quickactivate :BEM

using LinearAlgebra
using Statistics
using Printf
using Dates
using Plots
using LaTeXStrings

include(datadir("elastico", "two_blocks_contact.jl"))

const OUTDIR = raw"C:\Users\lucas.s.campos\OneDrive\artigos\escritos\2026\BEM_contact"
const FIGDIR = joinpath(OUTDIR, "figures")
mkpath(FIGDIR)

println("="^72)
println(" Multi-body fretting (free sides + bulk u_x)")
println(" ", Dates.now())
println("="^72)

E, ν, μ = 100.0, 0.3, 0.3
props = Elasticity(E, ν, 1.0; plane_strain=true)
R, W, H = 10.0, 4.0, 2.0
gap_flat = 0.04
ndiv, ndiv_y = 24, 4
δn = 0.05
u_max = 0.08
n_normal = 6
n_leg = 4
solvers = (:activeset, :ssn)   # two solvers only — keep runtime reasonable

function load_mb(nome)
    load_two_blocks_contact(props; W=W, H=H, gap=gap_flat, μ=μ,
        ndiv_bot=ndiv, ndiv_top=ndiv, ndiv_y=ndiv_y, nome=nome, fretting=true) |>
        p -> (apply_parabolic_contact_gap!(p; R=R, method=:ntn); p)
end

function metrics(prob)
    fr = contact_interface_xyτ(prob)
    h = length(fr.x) > 1 ? abs(fr.x[2] - fr.x[1]) : 1.0
    p = .-fr.tn; q = .-fr.tt
    cl = abs.(fr.state) .!= 1
    P = any(cl) ? sum(p[cl]) * h : 0.0
    Q = any(cl) ? sum(q[cl]) * h : 0.0
    a = any(cl) ? 0.5 * (maximum(fr.x[cl]) - minimum(fr.x[cl])) : 0.0
    p0 = any(cl) ? maximum(p[cl]) : 0.0
    xc = any(cl) ? mean(fr.x[cl]) : mean(fr.x)
    qrms = any(cl) ? sqrt(mean(abs2, q[cl])) : 0.0
    return (; fr..., p, q, P, Q, a, p0, xc, h, qrms,
        n_stick=count(==(3), fr.state),
        n_slip=count(s -> abs(s) == 2, fr.state),
        n_open=count(==(1), fr.state))
end

function step!(prob, δ, ux, sol, xw; maxit=80)
    set_farfield_displacement!(prob.regions[2]; ux=ux, uy=0.0, face=:top)
    _, xw = solve_contact_friction!(prob; δ=δ, solver=sol, method=:ntn,
        tol=1e-6, maxiter=maxit, npg=8, x0=xw,
        reset_states=(xw === nothing), return_x=true)
    return xw
end

rows = NamedTuple[]
sols = Dict{Symbol,Any}()
hyst = Dict{Symbol,Any}()

for sol in solvers
    println("\n=== $sol ===")
    prob = load_mb("mb_$sol")
    xw = nothing
    t0 = time()
    maxit = sol === :activeset ? 100 : 60

    # normal ramp at ux=0
    for δ in range(δn / n_normal, δn; length=n_normal)
        xw = step!(prob, δ, 0.0, sol, xw; maxit=maxit)
    end
    corners = Dict{Symbol,Any}(:A => metrics(prob))
    @printf("  A  P=%.3f Q=%+.3f a=%.3f st/sl=%d/%d qrms=%.3f\n",
        corners[:A].P, corners[:A].Q, corners[:A].a,
        corners[:A].n_stick, corners[:A].n_slip, corners[:A].qrms)

    ux_h = Float64[0.0]; Q_h = Float64[corners[:A].Q]
    # legs A→B→C→D→E
    legs = (( :B, 0.0, +u_max), (:C, +u_max, 0.0), (:D, 0.0, -u_max), (:E, -u_max, 0.0))
    for (nm, u0, u1) in legs
        for s in 1:n_leg
            t = s / n_leg
            ux = (1 - t) * u0 + t * u1
            xw = step!(prob, δn, ux, sol, xw; maxit=maxit)
            m = metrics(prob)
            push!(ux_h, ux); push!(Q_h, m.Q)
            if s == n_leg
                corners[nm] = m
                @printf("  %s  ux=%+.3f P=%.3f Q=%+.3f st/sl=%d/%d qrms=%.3f |Q|/fP=%.2f\n",
                    nm, ux, m.P, m.Q, m.n_stick, m.n_slip, m.qrms,
                    abs(m.Q) / max(μ * abs(m.P), eps()))
            end
        end
    end
    dt = time() - t0
    sols[sol] = (; corners, dt)
    hyst[sol] = (; ux=ux_h, Q=Q_h)
    @printf("  time %.1fs\n", dt)
    for (nm, m) in corners
        push!(rows, (; solver=String(sol), step=String(nm),
            P=m.P, Q=m.Q, a=m.a, p0=m.p0, xc=m.xc, qrms=m.qrms,
            n_stick=m.n_stick, n_slip=m.n_slip, n_open=m.n_open,
            Q_over_fP=abs(m.P) < eps() ? 0.0 : abs(m.Q) / max(μ * abs(m.P), eps())))
    end
end

open(joinpath(OUTDIR, "results_multibody.csv"), "w") do io
    println(io, "solver,step,P,Q,a,p0,xc,qrms,n_stick,n_slip,n_open,Q_over_fP")
    for r in rows
        @printf(io, "%s,%s,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%d,%d,%d,%.8e\n",
            r.solver, r.step, r.P, r.Q, r.a, r.p0, r.xc, r.qrms,
            r.n_stick, r.n_slip, r.n_open, r.Q_over_fP)
    end
end

cols = Dict(:activeset => :steelblue, :ssn => :darkorange)

# --- Hertz A ---
ref = sols[:activeset].corners[:A]
a = max(ref.a, eps()); p0 = max(ref.p0, eps())
xa = collect(range(-1.2a, 1.2a; length=300))
fig = plot(xa ./ a, cattaneo_pressure(xa, a, p0) ./ p0; color=:black, linestyle=:dash,
    label="Hertz", xlabel=L"(x-x_c)/a", ylabel=L"p/p_0", title="Multi-body pressure (A)",
    legend=:topright, legendfontsize=9, xlim=(-1.4, 1.4), size=(600, 340), framestyle=:box)
for sol in solvers
    m = sols[sol].corners[:A]
    plot!(fig, (m.x .- m.xc) ./ max(m.a, eps()), m.p ./ max(m.p0, eps());
        color=cols[sol], label=String(sol))
end
savefig(fig, joinpath(FIGDIR, "fig_mb_pressure_A.pdf"))
savefig(fig, joinpath(FIGDIR, "fig_mb_pressure_A.png"))

# --- hysteresis ---
fig = plot(; xlabel=L"u_x^{\mathrm{bulk}}", ylabel=L"Q", title="Fretting hysteresis Q(u_x)",
    legend=:topright, legendfontsize=9, size=(600, 380), framestyle=:box)
for sol in solvers
    h = hyst[sol]
    plot!(fig, h.ux, h.Q; color=cols[sol], marker=:circle, markersize=4, label=String(sol))
end
hline!(fig, [0.0]; color=:gray70, linestyle=:dot, label="")
vline!(fig, [0.0]; color=:gray70, linestyle=:dot, label="")
savefig(fig, joinpath(FIGDIR, "fig_mb_hysteresis.pdf"))
savefig(fig, joinpath(FIGDIR, "fig_mb_hysteresis.png"))

# --- shear B/C active-set ---
shear_panels = Plots.Plot[]
for (j, st) in enumerate((:B, :C))
    m = sols[:activeset].corners[st]
    ξ = (m.x .- m.xc) ./ max(m.a, eps()); p0 = max(m.p0, eps())
    ax = plot(; xlabel=L"(x-x_c)/a", ylabel=L"q/p_0", title="Shear step $st",
        legend=j == 1 ? :topright : false, legendfontsize=9, xlim=(-1.4, 1.4),
        framestyle=:box)
    if st === :B
        qref = cattaneo_shear(ξ .* m.a, m.a, m.p0, m.Q, μ, abs(m.P))
        plot!(ax, ξ, qref ./ p0; color=:black, linestyle=:dash, label="Cattaneo")
    else
        mB = sols[:activeset].corners[:B]
        xr = collect(range(-1.2mB.a, 1.2mB.a; length=300))
        qref = mindlin_shear_history(xr, mB.a, mB.p0, μ, abs(mB.P), [0.0, mB.Q, 0.0])
        plot!(ax, xr ./ max(mB.a, eps()), qref ./ max(mB.p0, eps());
            color=:black, linestyle=:dash, label="Mindlin")
    end
    plot!(ax, ξ, m.q ./ p0; color=cols[:activeset], label="AS")
    if haskey(sols, :ssn)
        ms = sols[:ssn].corners[st]
        plot!(ax, (ms.x .- ms.xc) ./ max(ms.a, eps()), ms.q ./ max(ms.p0, eps());
            color=cols[:ssn], linestyle=:dot, label="SSN")
    end
    push!(shear_panels, ax)
end
fig = plot(shear_panels...; layout=(1, 2), size=(700, 340))
savefig(fig, joinpath(FIGDIR, "fig_mb_shear_BC.pdf"))
savefig(fig, joinpath(FIGDIR, "fig_mb_shear_BC.png"))

# --- residual metrics ---
steps = (:A, :B, :C, :D, :E)
qrms = [sols[:activeset].corners[s].qrms for s in steps]
QfP = [abs(sols[:activeset].corners[s].Q) /
       max(μ * abs(sols[:activeset].corners[s].P), eps()) for s in steps]
fig = plot(1:5, qrms; color=:steelblue, marker=:circle, label=L"q_{\mathrm{rms}}",
    xlabel="step", title="Active-set residual metrics",
    xticks=(1:5, ["A", "B", "C", "D", "E"]), legend=:topright, legendfontsize=9,
    size=(520, 320), framestyle=:box)
plot!(fig, 1:5, QfP; color=:darkorange, marker=:circle, label=L"|Q|/(fP)")
savefig(fig, joinpath(FIGDIR, "fig_mb_residual_metrics.pdf"))
savefig(fig, joinpath(FIGDIR, "fig_mb_residual_metrics.png"))

# --- stick/slip B ---
m = sols[:activeset].corners[:B]
fig = scatter((m.x .- m.xc) ./ max(m.a, eps()), Float64.(m.state); color=cols[:activeset],
    xlabel=L"(x-x_c)/a", ylabel="state", title="AS stick/slip (B)",
    yticks=([-2, 1, 2, 3], ["slip−", "open", "slip+", "stick"]),
    xlim=(-1.4, 1.4), legend=false, size=(560, 260), framestyle=:box)
savefig(fig, joinpath(FIGDIR, "fig_mb_stick_slip_B.pdf"))
savefig(fig, joinpath(FIGDIR, "fig_mb_stick_slip_B.png"))

# --- solver B ---
tn = Float64[]; tt = Float64[]
for sol in solvers
    m = sols[sol].corners[:B]; cl = abs.(m.state) .!= 1
    push!(tn, mean(abs.(m.tn[cl]))); push!(tt, mean(abs.(m.tt[cl])))
end
xs = 1:length(solvers)
fig = bar(xs .- 0.15, tn; bar_width=0.28, color=:steelblue, label=L"|t_n|",
    xlabel="solver", ylabel="mean |t|", title="Closed-pair |t| at B",
    xticks=(xs, collect(string.(solvers))), legend=:topright, legendfontsize=9,
    size=(480, 300), framestyle=:box)
bar!(fig, xs .+ 0.15, tt; bar_width=0.28, color=:darkorange, label=L"|t_t|")
savefig(fig, joinpath(FIGDIR, "fig_mb_solver_B.pdf"))
savefig(fig, joinpath(FIGDIR, "fig_mb_solver_B.png"))

println("\nResidual (AS):")
for s in steps
    m = sols[:activeset].corners[s]
    @printf("  %s Q=%+.3f |Q|/fP=%.2f qrms=%.3f slip=%d\n",
        s, m.Q, abs(m.Q)/max(μ*abs(m.P),eps()), m.qrms, m.n_slip)
end
println("Done → ", FIGDIR)
