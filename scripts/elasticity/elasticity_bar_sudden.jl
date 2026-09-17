# Sudden end load: Laplace wave vs plane-stress elasticity.
# Houbolt vs MMM on each physics.
# julia --project=. scripts/elasticity_bar_sudden.jl
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf, Statistics
using Plots, LaTeXStrings
gr()
default(fontfamily="Computer Modern", linewidth=1.5, framestyle=:box,
        grid=false, dpi=160, size=(560, 360), legendfontsize=8)

include(datadir("Laplace", "Laplace_dad.jl"))
include(datadir("Laplace", "potencial_problems.jl"))
include(datadir("Laplace", "wave_propagation.jl"))
include(datadir("elastico", "iso", "bar_sudden.jl"))

const OUT = get(ENV, "STUDY_OUT",
    raw"C:\Users\lucas.s.campos\OneDrive\artigos\escritos\2026\SBM transient\results")
const FIG = joinpath(OUT, "figures")
mkpath(FIG)

function probe_hist(U, dad, probe; comp=1)
    pts = vcat(dad.Nodes, dad.internalNodes)
    ip = argmin(norm(p - probe) for p in pts)
    ndof_pt = size(U, 1) ÷ dad.nt
    return U[ndof_pt * (ip - 1) + comp, :], ip
end

function rel_hist(un, ua)
    den = norm(ua)
    return den > 0 ? norm(un .- ua) / den : NaN
end

function run_laplace(; ndiv=10, n_int=6, Δt=0.05, tf=4.0)
    dad, meta = wave_problem(:bar_sudden; ndiv=ndiv, n_int=n_int)
    H_G_full_direct(dad; npg=8, threaded=false)
    DIBEM(dad; method=:dense, rbf=PHS(1; poly_deg=-1))
    dadH = deepcopy(dad)
    dadM = deepcopy(dad)
    Uh = solve_Houbolt(dadH, Δt, tf)
    Um, t, b = solve_mmm!(dadM, Δt, tf; alg=:houbolt)
    un_h, _ = probe_hist(Uh, dadH, meta.probe)
    un_m, _ = probe_hist(Um, dadM, meta.probe)
    ua = [meta.ana.u(meta.probe; t=ti) for ti in t]
    return (; t, ua, un_h, un_m, ω1=b.ω[1], nmodes=length(b.ω),
        rel_h=rel_hist(un_h, ua), rel_m=rel_hist(un_m, ua),
        max_h=maximum(abs, Uh), max_m=maximum(abs, Um),
        nt=dad.nt, physics=:laplace)
end

function run_elasticity(; ndiv=10, n_int=6, Δt=0.05, tf=4.0, ν=0.0)
    dad, meta = elasticity_bar_sudden(; ndiv=ndiv, n_int=n_int, ν=ν)
    H_G_full_direct(dad; npg=8, threaded=false)
    DIBEM(dad; method=:dense, rbf=PHS(1; poly_deg=-1))
    dadH = deepcopy(dad)
    dadM = deepcopy(dad)
    Uh = solve_Houbolt(dadH, Δt, tf)
    Um, t, b = solve_mmm!(dadM, Δt, tf; alg=:houbolt)
    un_h, _ = probe_hist(Uh, dadH, meta.probe; comp=1)
    un_m, _ = probe_hist(Um, dadM, meta.probe; comp=1)
    ua = [meta.ana.u(meta.probe; t=ti) for ti in t]
    return (; t, ua, un_h, un_m, ω1=b.ω[1], nmodes=length(b.ω),
        rel_h=rel_hist(un_h, ua), rel_m=rel_hist(un_m, ua),
        max_h=maximum(abs, Uh), max_m=maximum(abs, Um),
        nt=dad.nt, physics=:elasticity, ν)
end

function main()
    ndiv, n_int, Δt, tf = 10, 6, 0.05, 4.0
    lap = run_laplace(; ndiv, n_int, Δt, tf)
    el0 = run_elasticity(; ndiv, n_int, Δt, tf, ν=0.0)
    elν = run_elasticity(; ndiv, n_int, Δt, tf, ν=0.3)
    @printf("Laplace     nt=%d  modes=%d  ω1=%.4f  Houbolt rel=%.3e  MMM rel=%.3e  maxH=%.3f\n",
        lap.nt, lap.nmodes, lap.ω1, lap.rel_h, lap.rel_m, lap.max_h)
    @printf("Elast ν=0   nt=%d  modes=%d  ω1=%.4f  Houbolt rel=%.3e  MMM rel=%.3e  maxH=%.3f\n",
        el0.nt, el0.nmodes, el0.ω1, el0.rel_h, el0.rel_m, el0.max_h)
    @printf("Elast ν=0.3 nt=%d  modes=%d  ω1=%.4f  Houbolt rel=%.3e  MMM rel=%.3e  maxH=%.3f\n",
        elν.nt, elν.nmodes, elν.ω1, elν.rel_h, elν.rel_m, elν.max_h)

    plt = plot(lap.t, lap.ua; color=:black, ls=:dash, label="1D series")
    lap.max_h < 20 && plot!(plt, lap.t, lap.un_h; color=:steelblue, label="Laplace Houbolt")
    lap.max_m < 20 && plot!(plt, lap.t, lap.un_m; color=:royalblue, ls=:dot, label="Laplace MMM")
    el0.max_h < 20 && plot!(plt, el0.t, el0.un_h; color=:darkorange, label="Elast ν=0 Houbolt")
    el0.max_m < 20 && plot!(plt, el0.t, el0.un_m; color=:orangered, ls=:dot, label="Elast ν=0 MMM")
    elν.max_h < 20 && plot!(plt, elν.t, elν.un_h; color=:seagreen, label="Elast ν=0.3 Houbolt")
    elν.max_m < 20 && plot!(plt, elν.t, elν.un_m; color=:green, ls=:dot, label="Elast ν=0.3 MMM")
    plot!(plt; xlabel=L"t", ylabel=L"u_x(L,L/2)", title="sudden end load")
    savefig(plt, joinpath(FIG, "elast_bar_sudden.pdf"))
    savefig(plt, joinpath(FIG, "elast_bar_sudden.png"))
    println("wrote ", joinpath(FIG, "elast_bar_sudden.pdf"))
    return (; lap, el0, elν)
end

if endswith(replace(PROGRAM_FILE, "\\" => "/"), "elasticity_bar_sudden.jl")
    main()
end
