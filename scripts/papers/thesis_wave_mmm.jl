# Santos 2024 Ch.4 wave cases with MMM — three discretizations per figure.
# julia --project=. scripts/thesis_wave_mmm.jl
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf, Statistics
using Plots, LaTeXStrings
gr()
default(fontfamily="Computer Modern", linewidth=1.5, framestyle=:box,
        grid=false, dpi=160, size=(540, 360), legendfontsize=8)

include(datadir("Laplace", "Laplace_dad.jl"))
include(datadir("Laplace", "potencial_problems.jl"))
include(datadir("Laplace", "wave_propagation.jl"))
include(joinpath(@__DIR__, "thesis_annulus_houbolt.jl"))

const OUT = get(ENV, "STUDY_OUT",
    raw"C:\Users\lucas.s.campos\OneDrive\artigos\escritos\2026\SBM transient\results")
const FIG = joinpath(OUT, "figures")
mkpath(FIG)

const MESHES = let s = get(ENV, "MMM_MESHES", "16,12;32,24;48,36")
    [Tuple(parse.(Int, split(p, ','))) for p in split(s, ';'; keepempty=false)]
end

const COLS = [:steelblue, :darkorange, :crimson]
const STY = [:solid, :dashdot, :solid]

function assemble!(dad; rbf=PHS(1; poly_deg=-1))
    H_G_full_direct(dad; npg=10, threaded=false)
    DIBEM(dad; method=:dense, rbf=rbf)
    return dad
end

function probe_idx(dad, probe)
    pts = vcat(dad.Nodes, dad.internalNodes)
    return argmin(norm(p - probe) for p in pts)
end

function rel_hist(un, ua)
    den = norm(ua)
    return den > 0 ? norm(un .- ua) / den : NaN
end

function probe_point(dad, probe)
    ip = probe_idx(dad, probe)
    p = ip <= dad.n ? dad.Nodes[ip] : dad.internalNodes[ip - dad.n]
    return ip, p
end

function overlay_plot(t_ana, ua, curves, title, dest)
    plt = plot(t_ana, ua; color=:black, ls=:dash, lw=1.8, label="series")
    for (lab, t, un, col, st) in curves
        plot!(plt, t, un; color=col, ls=st, label=lab)
    end
    plot!(plt; xlabel=L"t", ylabel=L"u", title=title)
    savefig(plt, dest * ".pdf")
    savefig(plt, dest * ".png")
    return plt
end

function run_bar(name; meshes=MESHES, tf=12.0, Δt=0.04, alg=:houbolt,
        ωmax=Inf, nmodes=nothing)
    curves = Tuple[]
    rows = NamedTuple[]
    ua = nothing
    t_ana = nothing
    for (k, (ndiv, n_int)) in enumerate(meshes)
        dad, meta = wave_problem(name; ndiv=ndiv, n_int=n_int)
        assemble!(dad)
        U, t, basis = solve_mmm!(dad, Δt, tf; select=:freq, alg=alg, ωmax=ωmax,
            nmodes=nmodes, load=hasproperty(meta, :load) ? meta.load : nothing)
        ip, p = probe_point(dad, meta.probe)
        un = U[ip, :]
        ua_k = [meta.ana.u(p; t=ti) for ti in t]
        rel = rel_hist(un, ua_k)
        lab = "nt=$(dad.nt)"



        @printf("%-16s ndiv=%2d ni=%2d N=%d nt=%d modes=%d ω1=%.4f  rel=%.3e  max=%.3f  ok=%s\n",
            name, ndiv, n_int, dad.n, dad.nt, length(basis.ω), basis.ω[1], rel,
            maximum(abs, un), string(all(isfinite, un)))
        push!(curves, (lab, t, un, COLS[k], STY[k]))
        push!(rows, (; name, ndiv, n_int, N=dad.n, ni=dad.ni, nmodes=length(basis.ω),
            ω1=basis.ω[1], rel, maxu=maximum(abs, un)))
        if k == length(meshes)
            t_ana, ua = t, ua_k
        end
    end
    overlay_plot(t_ana, ua, curves, string(name), joinpath(FIG, "mmm_$(name)"))
    return rows
end

function run_annulus(; meshes=MESHES, tf=12.0, Δt=0.04, alg=:houbolt)
    roots = fig_roots()
    Cn = fig_coeffs(roots)
    probe = Point2D(RA / sqrt(2), RA / sqrt(2))
    curves = Tuple[]
    rows = NamedTuple[]
    ua = nothing
    t_ana = nothing
    for (k, (ndiv, n_int)) in enumerate(meshes)
        dad, _ = wave_problem(:annulus; ndiv=ndiv, n_int=n_int)
        set_internal_annulus!(dad; nr=n_int, nθ=n_int, a=RA, b=RB, pad=0.15)
        set_fig_bc!(dad)
        assemble!(dad)
        U, t, basis = solve_mmm!(dad, Δt, tf; select=:freq, alg=alg)
        ip, p = probe_point(dad, probe)
        r = hypot(p[1], p[2])
        un = U[ip, :]
        ua_k = [u_ana(r, ti, roots, Cn) for ti in t]
        rel = rel_hist(un, ua_k)
        lab = "nt=$(dad.nt)"




        @printf("%-16s ndiv=%2d ni=%2d N=%d nt=%d modes=%d ω1=%.4f  rel=%.3e  max=%.3f  ok=%s\n",
            :annulus, ndiv, n_int, dad.n, dad.nt, length(basis.ω), basis.ω[1], rel,
            maximum(abs, un), string(all(isfinite, un)))
        push!(curves, (lab, t, un, COLS[k], STY[k]))
        push!(rows, (; name=:annulus, ndiv, n_int, N=dad.n, ni=dad.ni,
            nmodes=length(basis.ω), ω1=basis.ω[1], rel, maxu=maximum(abs, un)))
        if k == length(meshes)
            t_ana, ua = t, ua_k
        end
    end
    overlay_plot(t_ana, ua, curves, "annulus (figure BC)", joinpath(FIG, "mmm_annulus"))
    return rows
end

function main()
    run_bar(:bar_sudden; tf=12.0, Δt=0.04)
    run_bar(:bar_periodic; tf=20.0, Δt=0.04)
    run_bar(:membrane_forced; tf=8.0, Δt=0.02, ωmax=24.0)
    run_annulus(; tf=12.0, Δt=0.04)
end

if endswith(replace(PROGRAM_FILE, "\\" => "/"), "thesis_wave_mmm.jl")
    main()
end
