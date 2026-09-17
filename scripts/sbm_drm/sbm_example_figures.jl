# Four-method refinement up to ~5000 total nodes + error figures.
#
# julia --project=. scripts/sbm_example_figures.jl
using DrWatson
@quickactivate :BEM
using Printf
using Statistics
using Plots
using LaTeXStrings
gr()
default(
    fontfamily = "Computer Modern",
    linewidth = 1.8,
    framestyle = :box,
    grid = false,
    dpi = 180,
    legendfontsize = 8,
    guidefontsize = 10,
    tickfontsize = 8,
    titlefontsize = 10,
    size = (520, 400),
)

include(joinpath(@__DIR__, "sbm_transient_study.jl"))

const OUT = STUDY_OUT
const FIG = joinpath(OUT, "figures")
mkpath(FIG)

const METHS = (:sbm, :dibem, :drm, :kansa)
const COL = Dict(:sbm=>:black, :dibem=>:royalblue, :drm=>:darkorange, :kansa=>:seagreen)
const MKR = Dict(:sbm=>:circle, :dibem=>:square, :drm=>:utriangle, :kansa=>:diamond)
meth_label(m) = m === :sbm ? "SBM–DRM" :
                m === :dibem ? "BEM–DIBEM" :
                m === :drm ? "BEM–DRM" : "Kansa–BEM"

# (nb, nint) → N + M = 4 nb + nint². Last point ≈ 5017 nodes.
const GRIDS = (
    (8, 8),     # 96
    (16, 16),   # 320
    (24, 24),   # 672
    (32, 32),   # 1152
    (48, 48),   # 2496
    (64, 69),   # 5017
)

nsteps_of(ex) = ex == 3 ? 60 : (ex == 4 ? 40 : 120)

function run_all()
    rows = NamedTuple[]
    for ex in 1:4
        ns = nsteps_of(ex)
        for (nb, nint) in GRIDS, meth in METHS
            r = run_case(; ex=ex, method=meth, scheme=:houbolt, nsteps=ns, nb=nb, nint=nint)
            nt = r.n + r.ni
            row = (
                ex=ex, method=string(meth), label=meth_label(meth),
                nb=nb, nint=nint, n=r.n, ni=r.ni, nt=nt, nsteps=ns,
                rmse=r.rmse, rinf=r.rinf, maxu=r.maxu, meanu=r.meanu,
                ucenter=r.ucenter, t_cpu=r.t_cpu, ok=r.ok,
            )
            push!(rows, row)
            @printf("ex%d %-10s nb=%2d ni=%2d nt=%4d RMSE=%9.2e ctr=%8.3f t=%6.1fs %s\n",
                    ex, meth_label(meth), nb, nint, nt, r.rmse, r.ucenter, r.t_cpu,
                    r.ok ? "ok" : "FAIL")
        end
        write_csv(joinpath(OUT, "refine_ex$(ex).csv"), filter(r -> r.ex == ex, rows))
    end
    write_csv(joinpath(OUT, "refine_all.csv"), rows)
    return rows
end

function pick(rows; ex, meth)
    sub = filter(r -> r.ex == ex && r.method == string(meth) && r.ok, rows)
    sort!(sub; by=r -> r.nt)
    return sub
end

function fig_error(ex, rows; ylab, yof, fname, yscale=:log10)
    plt = plot(xlabel=L"n_t = N+M", ylabel=ylab,
               xscale=:log10, yscale=yscale, legend=:bottomleft)
    for meth in METHS
        sub = pick(rows; ex=ex, meth=meth)
        ys = [yof(r) for r in sub]
        keep = findall(isfinite, ys)
        isempty(keep) && continue
        plot!(plt, [float(sub[i].nt) for i in keep], ys[keep];
              color=COL[meth], marker=MKR[meth], label=meth_label(meth))
    end
    savefig(plt, joinpath(FIG, fname * ".pdf"))
    savefig(plt, joinpath(FIG, fname * ".png"))
    println("wrote $fname")
end

function fig_all(rows)
    fig_error(1, rows; ylab="RMSE", yof=r -> r.rmse, fname="ex1_error4")
    fig_error(2, rows; ylab="RMSE", yof=r -> r.rmse, fname="ex2_error4")
    fig_error(3, rows; ylab=L"u_{\mathrm{ctr}}", yof=r -> r.ucenter,
              fname="ex3_error4", yscale=:identity)
    fig_error(4, rows; ylab="RMSE", yof=r -> r.rmse, fname="ex4_error4")
end
function load_refine()
    path = joinpath(OUT, "refine_all.csv")
    lines = readlines(path)
    hdr = Symbol.(split(lines[1], ','))
    rows = NamedTuple[]
    for line in Iterators.drop(lines, 1)
        isempty(strip(line)) && continue
        cols = split(line, ',')
        d = Dict(hdr[i] => get(cols, i, "") for i in eachindex(hdr))
        push!(rows, (
            ex=parse(Int, d[:ex]),
            method=String(d[:method]),
            nt=parse(Int, d[:nt]),
            n=parse(Int, d[:n]),
            rmse=try parse(Float64, d[:rmse]) catch; NaN end,
            ucenter=parse(Float64, d[:ucenter]),
            ok=d[:ok] == "true",
        ))
    end
    return rows
end

function main()
    rows = get(ENV, "FIG_ONLY", "0") == "1" ? load_refine() : run_all()
    fig_all(rows)
    println("figures → $FIG")
    return rows
end

if endswith(replace(PROGRAM_FILE, "\\" => "/"), "sbm_example_figures.jl")
    main()
end
