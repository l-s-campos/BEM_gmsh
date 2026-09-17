# Santos Ch.4 wave cases with SBM–DRM (PHS1).
# julia --project=. scripts/thesis_wave_sbm.jl
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf, Statistics
using Plots, LaTeXStrings
gr()
default(fontfamily="Computer Modern", linewidth=1.5, framestyle=:box,
        grid=false, dpi=160, size=(520, 360), legendfontsize=8)

include(datadir("Laplace", "Laplace_dad.jl"))
include(datadir("Laplace", "potencial_problems.jl"))
include(datadir("Laplace", "wave_propagation.jl"))
include(joinpath(@__DIR__, "thesis_annulus_houbolt.jl"))

const OUT = get(ENV, "STUDY_OUT",
    raw"C:\Users\lucas.s.campos\OneDrive\artigos\escritos\2026\SBM transient\results")
const FIG = joinpath(OUT, "figures")
mkpath(FIG)

function probe_idx(dad, probe)
    pts = vcat(dad.Nodes, dad.internalNodes)
    return argmin(norm(p - probe) for p in pts)
end

function rel_hist(un, ua)
    den = norm(ua)
    return den > 0 ? norm(un .- ua) / den : NaN
end

function run_bar(name; ndiv=16, n_int=12, tf=12.0, Δt=0.04, scheme=:houbolt)
    dad, meta = wave_problem(name; ndiv=ndiv, n_int=n_int)
    load = hasproperty(meta, :load) ? meta.load : nothing
    sol = solve_sbm_wave(dad, Δt=Δt, tf=tf; scheme=scheme, load=load)
    ip = probe_idx(dad, meta.probe)
    un = sol.U[ip, :]
    p = ip <= dad.n ? dad.Nodes[ip] : dad.internalNodes[ip - dad.n]
    ua = [meta.ana.u(p; t=ti) for ti in sol.t]
    rel = rel_hist(un, ua)
    @printf("%-16s N=%d ni=%d  rel=%.3e  max=%.3f  ok=%s\n",
        name, dad.n, dad.ni, rel, maximum(abs, un), string(all(isfinite, un)))

    plt = plot(sol.t, ua; color=:black, ls=:dash, label="series")
    plot!(plt, sol.t, un; color=:steelblue, label="SBM–DRM")
    plot!(plt; xlabel=L"t", ylabel=L"u", title=string("SBM ", name))
    savefig(plt, joinpath(FIG, "sbm_$(name).pdf"))
    savefig(plt, joinpath(FIG, "sbm_$(name).png"))
    return (; name, rel, ok=all(isfinite, un), maxu=maximum(abs, un))
end

function run_annulus(; ndiv=16, n_int=12, tf=12.0, Δt=0.04, scheme=:houbolt)
    roots = fig_roots()
    Cn = fig_coeffs(roots)
    dad, _ = wave_problem(:annulus; ndiv=ndiv, n_int=n_int)
    set_internal_annulus!(dad; nr=n_int, nθ=n_int, a=RA, b=RB, pad=0.15)
    set_fig_bc!(dad)
    sol = solve_sbm_wave(dad, Δt=Δt, tf=tf; scheme=scheme)
    probe = Point2D(RA / sqrt(2), RA / sqrt(2))
    ip = probe_idx(dad, probe)
    p = ip <= dad.n ? dad.Nodes[ip] : dad.internalNodes[ip - dad.n]
    r = hypot(p[1], p[2])
    un = sol.U[ip, :]
    ua = [u_ana(r, ti, roots, Cn) for ti in sol.t]
    rel = rel_hist(un, ua)
    @printf("%-16s N=%d ni=%d  rel=%.3e  max=%.3f  ok=%s\n",
        :annulus, dad.n, dad.ni, rel, maximum(abs, un), string(all(isfinite, un)))

    plt = plot(sol.t, ua; color=:black, ls=:dash, label="series")
    plot!(plt, sol.t, un; color=:steelblue, label="SBM–DRM")
    plot!(plt; xlabel=L"t", ylabel=L"u", title="SBM annulus (figure BC)")
    savefig(plt, joinpath(FIG, "sbm_annulus.pdf"))
    savefig(plt, joinpath(FIG, "sbm_annulus.png"))
    return (; name=:annulus, rel, ok=all(isfinite, un), maxu=maximum(abs, un))
end

function main()
    run_bar(:bar_sudden; ndiv=16, n_int=12, tf=12.0, Δt=0.04)
    run_bar(:bar_periodic; ndiv=16, n_int=12, tf=20.0, Δt=0.04)
    run_bar(:membrane_forced; ndiv=16, n_int=12, tf=8.0, Δt=0.02)
    run_annulus(; ndiv=16, n_int=12, tf=12.0, Δt=0.04)
end

if endswith(replace(PROGRAM_FILE, "\\" => "/"), "thesis_wave_sbm.jl")
    main()
end
