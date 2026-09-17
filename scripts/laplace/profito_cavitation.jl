# Profito et al., Tribol. Lett. 60:18 (2015) §4.1 validation cases.
# Mass-conserving Elrod–Adams p–θ Reynolds (structured FVM / Ausas).
#
#   julia --project=. scripts/laplace/profito_cavitation.jl
#
using DrWatson
@quickactivate :BEM
using LinearAlgebra
using Printf
using Plots

const OUTDIR = projectdir("plots", "profito_cavitation")
mkpath(OUTDIR)

function _run_1d(case)
    dx = case.x[2] - case.x[1]
    opt = ElrodOptions(; pcav=case.opt.pcav, maxiter=120, tol=1e-8, verbose=true)
    p, θ = solve_elrod_1d(case.h, dx; U=case.U, rheo=case.rheo,
        pleft=case.pleft, pright=case.pright, opt=opt)
    return p, θ
end

function fig_slider()
    println("\n## 4.1.1  parabolic sliders")
    sc = profito_single_slider(; compressible=true)
    si = profito_single_slider(; compressible=false)
    pc, θc = _run_1d(sc)
    pi, _ = _run_1d(si)
    ξs = sc.x ./ sc.L
    @printf("  single  pmax compressible = %.2f MPa  (incompressible %.2f MPa)\n",
        maximum(pc) / 1e6, maximum(pi) / 1e6)
    @printf("  single  cavitation starts at ξ ≈ %.3f\n",
        ξs[findfirst(<(1 - 1e-8), θc)])

    dc = profito_double_slider(; compressible=true)
    pd, θd = _run_1d(dc)
    ξd = dc.x ./ dc.L
    @printf("  double  pmax = %.3f MPa\n", maximum(pd) / 1e6)

    ax1 = plot(ξs, pi ./ 1e6; color=:black, label="Incompressible",
        xlabel="Normalized coordinate", ylabel="Hydrodynamic pressure (MPa)",
        title="Single parabolic slider", legend=:topleft)
    plot!(ax1, ξs, pc ./ 1e6; color=:red, lw=2, label="EbFVM (DH compressible)")
    ax2 = plot(ξd, pd ./ 1e6; color=:red, lw=2, label="EbFVM (current)",
        xlabel="Normalized coordinate", ylabel="Hydrodynamic pressure (MPa)",
        title="Double parabolic slider", legend=:topleft)
    fig = plot(ax1, ax2; layout=(2, 1), size=(720, 720))
    savefig(fig, joinpath(OUTDIR, "fig3_sliders.png"))
    println("  wrote fig3_sliders.png")
    return nothing
end

function fig_journal()
    println("\n## 4.1.2  infinitely long journal (Barus)")
    axes = Plots.Plot[]
    for (k, ε) in enumerate((0.93, 0.95))
        jp = profito_journal(; ε=ε, piezoviscous=true, nθ=121, ny=3)
        ji = profito_journal(; ε=ε, piezoviscous=false, nθ=121, ny=3)
        dx = jp.x[2] - jp.x[1]
        h1 = vec(jp.h[:, 1])
        function _run(case)
            p = zeros(length(case.x))
            θ = ones(length(case.x))
            solve_elrod_1d_periodic!(p, θ, vec(case.h[:, 1]), dx;
                U=case.U, rheo=case.rheo,
                opt=ElrodOptions(; pcav=case.opt.pcav, maxiter=40, tol=1e-7, verbose=true))
            return p, θ
        end
        pp = try
            first(_run(jp))
        catch e
            @warn "Barus ε=$ε failed" exception=e
            zeros(length(jp.x))
        end
        pi = try
            first(_run(ji))
        catch e
            @warn "isoviscous ε=$ε failed" exception=e
            zeros(length(ji.x))
        end
        ξ = jp.x ./ (2π * jp.R)
        @printf("  ε=%.2f  pmax Barus = %.1f MPa  isoviscous = %.1f MPa\n",
            ε, maximum(pp) / 1e6, maximum(pi) / 1e6)
        ax = plot(ξ, pi ./ 1e6; color=:black, label="Isoviscous",
            xlabel="Normalized circumferential coordinate",
            ylabel="Hydrodynamic pressure (MPa)", title="ε = $ε",
            legend=:topleft, xlim=(0, 0.6))
        plot!(ax, ξ, pp ./ 1e6; color=:red, lw=2, label="Barus (current)")
        push!(axes, ax)
    end
    fig = plot(axes...; layout=(2, 1), size=(720, 720))
    savefig(fig, joinpath(OUTDIR, "fig4_journal.png"))
    println("  wrote fig4_journal.png")
    return nothing
end

function fig_squeeze()
    println("\n## 4.1.3  squeeze circular plates")
    s = profito_squeeze()
    nstep = s.ncycle * s.nt
    dt = s.T / s.nt
    tmax_plot = 0.10
    p = fill(s.p0, length(s.r))
    θ = ones(length(s.r))
    t_hist = Float64[]
    rc_hist = Float64[]
    h_old = s.hmin + s.ha * (1 - cos(0.0))
    θ_old = copy(θ)
    opt = ElrodOptions(; pcav=s.pcav, maxiter=400, tol=1e-8, ωp=0.8, ωθ=0.5)
    for k in 1:nstep
        t = k * dt
        h = s.hmin + s.ha * (1 - cos(s.ω * t))
        solve_elrod_radial!(p, θ, s.r, h; μ=s.μ, ρ=s.ρ, opt=opt,
            hold=h_old, θold=θ_old, dt=dt, pouter=s.p0)
        if t <= tmax_plot + 0.5 * dt
            ic = findlast(<(1 - 1e-4), θ)
            rc = ic === nothing ? 0.0 : s.r[ic] / s.R
            push!(t_hist, t)
            push!(rc_hist, rc)
        end
        h_old = h
        θ_old .= θ
        if t > tmax_plot
            break
        end
    end
    @printf("  max cavitation radius / R = %.3f\n", maximum(rc_hist; init=0.0))
    fig = plot(t_hist, rc_hist; color=:red, lw=2, label="Elrod–Adams (current)",
        xlabel="Time (s)", ylabel="Normalized cavitation radius",
        title="Pure squeeze, circular plates", legend=:topleft, size=(720, 420))
    savefig(fig, joinpath(OUTDIR, "fig5_squeeze.png"))
    println("  wrote fig5_squeeze.png")
    return nothing
end

function fig_pocket()
    println("\n## 4.1.4  sliding pocket bearing")
    fig = plot(; size=(720, 420), xlabel="Normalized length",
        ylabel="Hydrodynamic pressure (MPa)", title="Sliding pocket", legend=:topleft)
    for infinite in (true, false)
        c = profito_pocket(; infinite=infinite, ny=infinite ? 5 : 17)
        dx = c.x[2] - c.x[1]
        ξ = c.x ./ c.a
        lbl = infinite ? "b = 300 mm" : "b = 10 mm"
        if infinite
            p, _ = solve_elrod_1d(vec(c.h[:, 1]), dx; U=c.U, rheo=c.rheo,
                pleft=0.0, pright=0.0,
                opt=ElrodOptions(; pcav=0.0, maxiter=120, tol=1e-8, verbose=true))
            pmid = p
        else
            p = fill(0.0, size(c.h))
            θ = ones(size(c.h))
            dy = c.y[2] - c.y[1]
            solve_elrod_2d!(p, θ, c.h, dx, dy; U=c.U, rheo=c.rheo,
                opt=ElrodOptions(; pcav=0.0, maxiter=8_000, tol=1e-6, ωp=0.9, ωθ=0.5),
                bc=(left=0.0, right=0.0, bottom=c.pside, top=c.pside))
            pmid = p[:, (size(p, 2) + 1) ÷ 2]
        end
        plot!(fig, ξ, pmid ./ 1e6; lw=2, label=lbl)
        @printf("  %s  pmax(mid) = %.2f MPa\n", lbl, maximum(pmid) / 1e6)
        if infinite
            plot!(twinx(fig), ξ, vec(c.h[:, 1]) .* 1e6; color=:gray, linestyle=:dash,
                ylabel="Pocket profile (μm)", label="")
        end
    end
    savefig(fig, joinpath(OUTDIR, "fig6_pocket.png"))
    println("  wrote fig6_pocket.png")
    return nothing
end

function main()
    println("="^72)
    println("Profito et al. 2015  §4.1  — Elrod–Adams p–θ FVM")
    println("="^72)
    fig_slider()
    try
        fig_journal()
    catch e
        @warn "journal figure failed" exception=e
    end
    try
        fig_squeeze()
    catch e
        @warn "squeeze figure failed" exception=e
    end
    try
        fig_pocket()
    catch e
        @warn "pocket figure failed" exception=e
    end
    println("\nDone.  ", OUTDIR)
end

main()
