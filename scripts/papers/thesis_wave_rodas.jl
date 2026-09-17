# Santos 2024 thesis Ch.4 wave cases, adaptive Rodas5P.
# julia --project=. scripts/thesis_wave_rodas.jl
using DrWatson
@quickactivate :BEM
using Printf
using Plots
using LaTeXStrings
gr()
default(
    fontfamily = "Computer Modern",
    linewidth = 1.6,
    framestyle = :box,
    grid = false,
    dpi = 160,
    legendfontsize = 8,
    guidefontsize = 10,
    tickfontsize = 8,
    size = (520, 360),
)

include(datadir("Laplace", "Laplace_dad.jl"))
include(datadir("Laplace", "potencial_problems.jl"))
include(datadir("Laplace", "wave_propagation.jl"))

const OUT = get(ENV, "STUDY_OUT",
    raw"C:\Users\lucas.s.campos\OneDrive\artigos\escritos\2026\SBM transient\results")
const FIG = joinpath(OUT, "figures")
mkpath(FIG)

function assemble!(dad; rbf=PHS(1; poly_deg=-1))
    H_G_full_direct(dad; npg=10, threaded=false)
    DIBEM(dad; method=:dense, rbf=rbf)
    return dad
end

function probe_history(dad, meta)
    pts = vcat(dad.Nodes, dad.internalNodes)
    ip = argmin(norm(Point2D(p) - Point2D(meta.probe)) for p in pts)
    t = dad.t isa AbstractRange ? collect(dad.t) : collect(dad.t)
    unum = dad.T[ip, :]
    uana = meta.ana === nothing ? fill(NaN, length(t)) :
           [meta.ana.u(pts[ip]; t=tt) for tt in t]
    return t, unum, uana, pts[ip]
end

function rel_err(unum, uana)
    i0 = min(length(unum), max(2, length(unum) ÷ 10))
    den = norm(@view uana[i0:end])
    return den > 0 && all(isfinite, unum) ?
        norm((@view unum[i0:end]) .- (@view uana[i0:end])) / den : NaN
end

function run_one(name; ndiv, n_int, tf, Δt_save, ω=1.0)
    dad, meta = wave_problem(name; ndiv=ndiv, n_int=n_int, ω=ω)
    if name === :annulus
        meta = merge(meta, (; probe=Point2D(3 / sqrt(2), 3 / sqrt(2))))
    end
    assemble!(dad)
    load = name === :bar_periodic ? (t -> sin(ω * t)) : nothing
    t_cpu = @elapsed solve_transient_o2(dad, Δt_save, tf;
        abstol=1e-5, reltol=1e-5, load=load, progress=false)
    t, unum, uana, xp = probe_history(dad, meta)
    err = rel_err(unum, uana)
    nst = has_cache(dad, :ode_sol) ? length(dad.ode_sol.t) : length(t)
    ok = all(isfinite, unum) && maximum(abs, unum) < 1e6
    @printf("%-18s n=%3d ni=%3d  steps≈%-4d  probe=(%.2f,%.2f)  rel=%.3e  max=%.3e  t=%.2fs %s\n",
            name, dad.n, dad.ni, nst, xp[1], xp[2], err, maximum(abs, unum),
            t_cpu, ok ? "ok" : "UNSTABLE")
    return (; name, t, unum, uana, err, t_cpu, n=dad.n, ni=dad.ni, nst, xp, ok)
end
function save_hist(r)
    plt = plot(r.t, r.uana; color=:black, ls=:dash, label="series",
               xlabel=L"t", ylabel=L"u_{\mathrm{probe}}",
               title=string(r.name))
    un = Float64.(r.unum)
    @inbounds for i in eachindex(un)
        (isfinite(un[i]) && abs(un[i]) < 50) || (un[i] = NaN)
    end
    plot!(plt, r.t, un; color=:crimson, label="Rodas5P")
    savefig(plt, joinpath(FIG, "wave_$(r.name).pdf"))
    savefig(plt, joinpath(FIG, "wave_$(r.name).png"))
end

function main()
    specs = (
        (:bar_sudden,      20, 20, 12.0, 0.04, 1.0),
        (:annulus,         16, 16, 30.0, 0.08, 1.0),
        (:bar_periodic,    20, 20, 40.0, 0.04, 1.0),
        (:membrane_forced, 20, 20,  8.0, 0.02, 1.0),
    )
    rows = NamedTuple[]
    for (name, ndiv, n_int, tf, dts, ω) in specs
        try
            r = run_one(name; ndiv=ndiv, n_int=n_int, tf=tf, Δt_save=dts, ω=ω)
            save_hist(r)
            push!(rows, r)
        catch e
            @warn "failed $name" exception=e
        end
    end
    return rows
end

if endswith(replace(PROGRAM_FILE, "\\" => "/"), "thesis_wave_rodas.jl")
    main()
end
