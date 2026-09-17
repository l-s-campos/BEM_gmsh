# Plot DIBEM RBF×compression timing from dibem_rbf_compress.tsv
#
#   julia --project=. scripts/plot_dibem_rbf_times.jl
#
# ENV:
#   DIBEM_TSV=path/to/dibem_rbf_compress.tsv
#   DIBEM_OUT=output directory (default: paper folder or plots/)

using DrWatson
@quickactivate :BEM
using Plots
using DelimitedFiles
using Statistics
using Printf

const PAPER = raw"C:\Users\lucas.s.campos\OneDrive\artigos\escritos\2025\Fast_DIBEM_potencial - h2"
const OUTDIR = get(ENV, "DIBEM_OUT", isdir(PAPER) ? PAPER : projectdir("plots"))
const TSV = get(ENV, "DIBEM_TSV", joinpath(OUTDIR, "dibem_rbf_compress.tsv"))

function load_tsv(path)
    data, header = readdlm(path, '\t', header=true)
    h = String.(vec(header))
    col(name) = findfirst(==(name), h)
    rows = NamedTuple[]
    for i in 1:size(data, 1)
        push!(rows, (;
            rbf=String(data[i, col("rbf")]),
            method=String(data[i, col("method")]),
            ndiv=Int(data[i, col("ndiv")]),
            n=Int(data[i, col("n")]),
            ni=Int(data[i, col("ni")]),
            nt=Int(data[i, col("nt")]),
            t_HG=Float64(data[i, col("t_HG")]),
            t_M=Float64(data[i, col("t_M")]),
            t_sol=Float64(data[i, col("t_sol")]),
            t_total=Float64(data[i, col("t_total")]),
            err_i=Float64(data[i, col("err_i")]),
        ))
    end
    return rows
end

const METHOD_STYLE = Dict(
    "dense"   => (; label="dense",    color=:black,      marker=:circle,  ls=:solid),
    "hmatrix" => (; label="H-matrix", color=:dodgerblue, marker=:square,  ls=:dash),
    "fmm"     => (; label="FMM",      color=:crimson,    marker=:star5,   ls=:dot),
)

function _series(rows, rbf::String, method::String; y=:t_M)
    sub = filter(r -> r.rbf == rbf && r.method == method, rows)
    isempty(sub) && return Float64[], Float64[]
    sort!(sub; by=r -> r.nt)
    xs = Float64[r.nt for r in sub]
    ys = Float64[getfield(r, y) for r in sub]
    return xs, ys
end

function plot_time_vs_nt(rows; y=:t_M, rbfs=("PHS2", "PHS3", "FS", "PHS1"),
                         outfile="dibem_time_elapsed.pdf")
    methods = unique(r.method for r in rows)
    morder = filter(m -> m in methods, ["dense", "hmatrix", "fmm"])
    isempty(morder) && (morder = collect(methods))
    rbfs = filter(rb -> any(r -> r.rbf == rb, rows), collect(rbfs))

    ylabel = y === :t_M ? "t_M [s]" :
             y === :t_total ? "t_total [s]" :
             y === :t_sol ? "t_sol [s]" : String(y)

    plts = Plots.Plot[]
    for (k, rb) in enumerate(rbfs)
        p = plot(;
            xlabel="n_t = n_b + n_i",
            ylabel=k == 1 ? ylabel : "",
            title=rb,
            xscale=:log10,
            yscale=:log10,
            legend=(k == length(rbfs) ? :topleft : false),
            framestyle=:box,
            size=(320, 400),
        )
        for m in morder
            st = get(METHOD_STYLE, m, (; label=m, color=:gray, marker=:circle, ls=:solid))
            xs, ys = _series(rows, rb, m; y=y)
            isempty(xs) && continue
            mask = ys .> 0
            xs, ys = xs[mask], ys[mask]
            isempty(xs) && continue
            plot!(p, xs, ys;
                label=st.label,
                color=st.color,
                marker=st.marker,
                linestyle=st.ls,
                markersize=6,
                linewidth=2,
            )
        end
        push!(plts, p)
    end

    fig = plot(plts...; layout=(1, length(plts)), size=(320 * length(plts), 420))
    mkpath(OUTDIR)
    path = joinpath(OUTDIR, outfile)
    savefig(fig, path)
    println("wrote $path")
    return fig, path
end

function plot_time_breakdown(rows; rbf="PHS2", outfile="dibem_time_breakdown.pdf")
    sub = filter(r -> r.rbf == rbf, rows)
    methods = unique(r.method for r in sub)
    morder = filter(m -> m in methods, ["dense", "hmatrix", "fmm"])
    ndivs = sort(unique(r.ndiv for r in sub))

    # For each ndiv: grouped stacked bars per method
    # Build a simple multi-panel: one subplot per ndiv
    plts = Plots.Plot[]
    for nd in ndivs
        names = String[]
        tHG = Float64[]; tM = Float64[]; tS = Float64[]
        for m in morder
            rs = filter(r -> r.ndiv == nd && r.method == m, sub)
            isempty(rs) && continue
            r = rs[1]
            push!(names, m)
            push!(tHG, r.t_HG)
            push!(tM, r.t_M)
            push!(tS, r.t_sol)
        end
        p = plot(;
            title="ndiv=$nd",
            ylabel="time [s]",
            legend=:topleft,
            size=(400, 350),
            framestyle=:box,
            xticks=(1:length(names), names),
            xrotation=30,
        )
        # stacked bar via three series
        bar!(p, 1:length(names), tHG; label="H/G", color=:steelblue, bar_width=0.6)
        bar!(p, 1:length(names), tM; label="DIBEM M", color=:darkorange, bar_width=0.6,
            bar_position=:stack)
        # Plots.jl stack: use grouped recipe — manual stack
        # Re-do properly:
        p = groupedbar_stack(names, tHG, tM, tS; title="ndiv=$nd")
        push!(plts, p)
    end
    fig = plot(plts...; layout=(1, length(plts)), size=(280 * length(plts), 400))
    mkpath(OUTDIR)
    path = joinpath(OUTDIR, outfile)
    savefig(fig, path)
    println("wrote $path")
    return fig, path
end

"""Stacked bars: H/G, M, solve for each method name."""
function groupedbar_stack(names, tHG, tM, tS; title="")
    n = length(names)
    p = plot(;
        title=title,
        ylabel="time [s]",
        legend=:topleft,
        framestyle=:box,
        xticks=(1:n, names),
        xrotation=30,
        size=(400, 350),
    )
    # bottom: H/G
    bar!(p, 1:n, tHG; label="H/G", color=:steelblue, bar_width=0.65, linewidth=0)
    # middle: M on top of H/G
    bar!(p, 1:n, tHG .+ tM; label="", color=:darkorange, bar_width=0.65, linewidth=0)
    bar!(p, 1:n, tHG; label="DIBEM M", color=:darkorange, bar_width=0.65, linewidth=0,
        alpha=0)  # legend only
    # top: total
    tot = tHG .+ tM .+ tS
    bar!(p, 1:n, tot; label="", color=:seagreen, bar_width=0.65, linewidth=0)
    # redraw lower layers on top of green? better approach: plot from top down with full heights
    # Clean approach: three layers from bottom
    p = plot(;
        title=title,
        ylabel="time [s]",
        legend=:topleft,
        framestyle=:box,
        xticks=(1:n, names),
        xrotation=30,
        size=(400, 350),
    )
    # use Plots bar with series matrix (columns = series) — stacked by default in some backends
    ymat = hcat(tHG, tM, tS)
    # manual stacked bars
    for i in 1:n
        y0 = 0.0
        for (yi, col, lab) in zip((tHG[i], tM[i], tS[i]),
                                   (:steelblue, :darkorange, :seagreen),
                                   ("H/G", "DIBEM M", "solve"))
            plot!(p, Shape([i - 0.3, i + 0.3, i + 0.3, i - 0.3],
                           [y0, y0, y0 + yi, y0 + yi]);
                color=col, linecolor=:black, linewidth=0.3,
                label=(i == 1 ? lab : ""))
            y0 += yi
        end
    end
    return p
end

function main()
    println("TSV: $TSV")
    println("OUT: $OUTDIR")
    isfile(TSV) || error("missing $TSV — run fast_dibem_potencial_rbf_compare.jl first")
    rows = load_tsv(TSV)
    println("$(length(rows)) rows")

    plot_time_vs_nt(rows; y=:t_M, outfile="dibem_time_M.pdf")
    plot_time_vs_nt(rows; y=:t_total, outfile="dibem_time_total.pdf")
    plot_time_vs_nt(rows; y=:t_sol, outfile="dibem_time_sol.pdf")
    plot_time_breakdown(rows; rbf="PHS2", outfile="dibem_time_breakdown_PHS2.pdf")
    plot_time_breakdown(rows; rbf="FS", outfile="dibem_time_breakdown_FS.pdf")

    plot_time_vs_nt(rows; y=:t_M, outfile="dibem_time_M.html")
    plot_time_vs_nt(rows; y=:t_total, outfile="dibem_time_total.html")
    println("done")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
