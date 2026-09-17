# Matvec scaling: dense / H-matrix / NNCA H² / FMM
# 2-D Laplace log|x-y| on the unit square, N up to ~1e5.
#
#   julia --project=. -t auto scripts/hmat_fmm/matvec_compare.jl
#   MATVEC_QUICK=1 julia --project=. -t auto scripts/hmat_fmm/matvec_compare.jl
#
# Dense is skipped above MATVEC_DENSE_MAX (default 4500). Output:
#   .scratch/matvec-compare/fig_matvec.png
#   .scratch/matvec-compare/fig_error.png
#   .scratch/matvec-compare/results.csv

using DrWatson
@quickactivate :BEM
using LinearAlgebra
using Printf
using Random
using Statistics
using StaticArrays
using Dates
using BEM.HMatrices
using BEM.FMM
using Plots
using LaTeXStrings
gr()
default(fontfamily="Computer Modern", linewidth=2.0, framestyle=:box,
    grid=true, gridalpha=0.25, dpi=160, legendfontsize=9, guidefontsize=11,
    tickfontsize=9, size=(720, 460))

const OUT = get(ENV, "MATVEC_OUT",
    joinpath(dirname(dirname(@__DIR__)), ".scratch", "matvec-compare"))
const QUICK = get(ENV, "MATVEC_QUICK", "0") in ("1", "true", "yes")
const NMAX = parse(Int, get(ENV, "MATVEC_NMAX", "32"))
const RTOL = parse(Float64, get(ENV, "MATVEC_RTOL", "1e-6"))
const FMM_EPS = parse(Float64, get(ENV, "MATVEC_FMM_EPS", "1e-6"))
const DENSE_MAX = parse(Int, get(ENV, "MATVEC_DENSE_MAX", "4500"))
const NWARM = parse(Int, get(ENV, "MATVEC_NWARM", "3"))
const NRUN = parse(Int, get(ENV, "MATVEC_NRUN", "7"))
const THREADS = get(ENV, "MATVEC_HTHREADS", "1") in ("1", "true", "yes")

const NSIDES = if QUICK
    [16, 32, 48, 64]
else
    [16, 24, 32, 48, 64, 80, 100, 128, 160, 200, 256, 317]
end

const METHODS = (:dense, :hmatrix, :h2, :fmm)
const METHOD_STYLE = Dict(
    :dense => (:black, :circle, "dense"),
    :hmatrix => (:crimson, :utriangle, "H-matrix"),
    :h2 => (:royalblue, :diamond, "H2 (NNCA)"),
    :fmm => (:seagreen, :xcross, "FMM"),
)

mkpath(OUT)

function _pts(n1d)
    xs = range(0.0, 1.0; length=n1d)
    return [SVector(float(x), float(y)) for y in xs for x in xs]
end

function _pmat(pts)
    P = Matrix{Float64}(undef, 2, length(pts))
    @inbounds for j in eachindex(pts)
        P[1, j] = pts[j][1]
        P[2, j] = pts[j][2]
    end
    return P
end

function _kernel(pts)
    return KernelMatrix(pts, pts) do x, y
        r = hypot(x[1] - y[1], x[2] - y[2])
        return r < 1e-30 ? 0.0 : log(r)
    end
end

function _assemble(method, K, pts, P)
    if method === :dense
        return Matrix(K)
    elseif method === :hmatrix
        tree = ClusterTree(pts, hmatrix_splitter(; nmax=NMAX); cube=true)
        return assemble_hmatrix(K, tree, tree;
            adm=StrongAdmissibilityStd(2.0),
            comp=PartialACA(; rtol=RTOL),
            threads=THREADS)
    elseif method === :h2
        tree = ClusterTree(pts, DyadicSplitter(; nmax=NMAX, tight=false); cube=true)
        return assemble_h2(K, tree; rtol=RTOL, threads=THREADS)
    elseif method === :fmm
        return FMM.fmm_laplace2d_matrix(P; eps=FMM_EPS, nmax=NMAX)
    end
    throw(ArgumentError("unknown method $method"))
end

function _bench_mul(A, x; nwarm=NWARM, nrun=NRUN)
    y = similar(x)
    mul!(y, A, x)
    for _ in 1:nwarm
        mul!(y, A, x)
    end
    ts = Vector{Float64}(undef, nrun)
    @inbounds for i in 1:nrun
        t0 = time_ns()
        mul!(y, A, x)
        ts[i] = (time_ns() - t0) / 1e9
    end
    return median(ts), copy(y)
end

function _rel(a, b)
    nb = norm(b)
    return nb > 0 ? norm(a .- b) / nb : (norm(a) == 0 ? 0.0 : Inf)
end

function run()
    Random.seed!(1)
    println("Matvec  dense / H-matrix / H² / FMM")
    println("  ", Dates.now())
    println("  threads=", Threads.nthreads(), "  H-ACA/NNCA threads=", THREADS)
    println("  rtol=", RTOL, "  fmm_eps=", FMM_EPS, "  nmax=", NMAX)
    println("  n1d=", NSIDES, "  dense_max=", DENSE_MAX)
    println()
    @printf("%-8s %7s %10s %10s %8s %10s\n",
        "method", "N", "t_asm[s]", "t_mv[s]", "mem[MB]", "err")
    println("-"^62)

    rows = NamedTuple[]
    warmed = Set{Symbol}()
    for n1d in NSIDES
        pts = _pts(n1d)
        n = length(pts)
        K = _kernel(pts)
        P = _pmat(pts)
        x = randn(n)
        ydense = nothing
        yfmm = nothing
        for method in METHODS
            method === :dense && n > DENSE_MAX && continue
            try
                if method ∉ warmed
                    _assemble(method, K, pts, P)
                    push!(warmed, method)
                end
                A = nothing
                t_asm = @elapsed A = _assemble(method, K, pts, P)
                t_mv, y = _bench_mul(A, x)
                method === :dense && (ydense = copy(y))
                method === :fmm && (yfmm = copy(y))
                ref = ydense === nothing ? yfmm : ydense
                err = ref === nothing ? NaN : _rel(y, ref)
                mem = Base.summarysize(A) / 1024^2
                @printf("%-8s %7d %10.3e %10.3e %8.2f %10.2e\n",
                    method, n, t_asm, t_mv, mem, err)
                flush(stdout)
                push!(rows, (; method, n, n1d, t_asm, t_mv, mem_mb=mem, err))
                A = nothing
            catch e
                @warn "method $method failed at n=$n" exception = e
            end
        end
        GC.gc(false)
    end
    return rows
end

function write_csv(path, rows)
    open(path, "w") do io
        println(io, "method,n,n1d,t_asm,t_mv,mem_mb,err")
        for r in rows
            @printf(io, "%s,%d,%d,%.6e,%.6e,%.6e,%.6e\n",
                r.method, r.n, r.n1d, r.t_asm, r.t_mv, r.mem_mb, r.err)
        end
    end
    return path
end

function by_method(rows)
    d = Dict{Symbol,Vector{NamedTuple}}()
    for r in rows
        push!(get!(d, r.method, NamedTuple[]), r)
    end
    for k in keys(d)
        sort!(d[k]; by=r -> r.n)
    end
    return d
end

function add_guides!(plt, nmin, nmax, tref, nref)
    ns = 10 .^ range(log10(nmin), log10(nmax); length=48)
    plot!(plt, ns, tref .* (ns ./ nref);
        ls=:dash, lc=:gray, lw=0.9, label=L"N")
    plot!(plt, ns, tref .* (ns .* log.(ns)) ./ (nref * log(nref));
        ls=:dot, lc=:gray, lw=0.9, label=L"N\log N")
    plot!(plt, ns, tref .* (ns ./ nref) .^ 2;
        ls=:dashdot, lc=:gray, lw=0.9, label=L"N^2")
    return plt
end

function plot_rows(rows)
    R = by_method(rows)
    nall = [r.n for r in rows]
    nmin, nmax = extrema(nall)

    plt = plot(; xscale=:log10, yscale=:log10, xlabel=L"N",
        ylabel="matvec time [s]",
        title="2D Laplace " * L"\log|x-y|" * "  (unit square)",
        legend=:bottomright, size=(760, 480))
    tguide = nothing
    nguide = nothing
    for m in METHODS
        haskey(R, m) || continue
        rs = R[m]
        ns = [r.n for r in rs]
        ts = [r.t_mv for r in rs]
        col, mk, lab = METHOD_STYLE[m]
        plot!(plt, ns, ts; marker=mk, color=col, markersize=5, label=lab)
        if tguide === nothing && m !== :dense && ts[1] > 0
            tguide = ts[1]
            nguide = Float64(ns[1])
        end
    end
    tguide === nothing || add_guides!(plt, nmin, nmax, tguide, nguide)
    p_mv = joinpath(OUT, "fig_matvec.png")
    savefig(plt, p_mv)

    plt_e = plot(; xscale=:log10, yscale=:log10, xlabel=L"N",
        ylabel=L"\|y - y_{\mathrm{ref}}\| / \|y_{\mathrm{ref}}\|",
        title="relative error (vs dense, else vs FMM)",
        legend=:bottomright, size=(760, 480),
        ylims=(1e-16, 1e-2))
    for m in METHODS
        haskey(R, m) || continue
        m === :dense && continue
        rs = R[m]
        ns = [r.n for r in rs]
        es = [max(r.err, 1e-16) for r in rs]
        col, mk, lab = METHOD_STYLE[m]
        plot!(plt_e, ns, es; marker=mk, color=col, markersize=5, label=lab)
    end
    p_err = joinpath(OUT, "fig_error.png")
    savefig(plt_e, p_err)

    plt_a = plot(; xscale=:log10, yscale=:log10, xlabel=L"N",
        ylabel="assembly time [s]",
        title="assembly (host)",
        legend=:bottomright, size=(760, 480))
    for m in METHODS
        haskey(R, m) || continue
        rs = R[m]
        ns = [r.n for r in rs]
        ts = [r.t_asm for r in rs]
        col, mk, lab = METHOD_STYLE[m]
        plot!(plt_a, ns, ts; marker=mk, color=col, markersize=5, label=lab)
    end
    p_asm = joinpath(OUT, "fig_assemble.png")
    savefig(plt_a, p_asm)
    return p_mv, p_err, p_asm
end

function main()
    rows = run()
    csv = write_csv(joinpath(OUT, "results.csv"), rows)
    p_mv, p_err, p_asm = plot_rows(rows)
    println()
    println("wrote ", csv)
    println("wrote ", p_mv)
    println("wrote ", p_err)
    println("wrote ", p_asm)
    return rows
end

main()
