#!/usr/bin/env julia
# =============================================================================
# Cost order: dense / H-matrix / FMM  — assembly + matvec, 2D and 3D
#
# Laplace single-layer kernel (matches FMM entries):
#   2D  log|x-y|          3D  1/(4π|x-y|)
#
# Size ladders go to N ≈ 10⁴.
#
#   julia --project=. scripts/hmat_fmm/cost_order.jl --quick
#   julia --project=. scripts/hmat_fmm/cost_order.jl
#   julia --project=. scripts/hmat_fmm/cost_order.jl --dim=2 --geom=boundary
# =============================================================================
using Pkg
Pkg.activate(dirname(dirname(@__DIR__)))

using LinearAlgebra
using Printf
using Statistics
using Random
using StaticArrays
using Dates

using BEM
using BEM.HMatrices
using BEM.FMM

using Plots
using LaTeXStrings
gr()
default(fontfamily="Computer Modern", linewidth=1.8, framestyle=:box,
    grid=true, gridalpha=0.25, dpi=160, legendfontsize=8, guidefontsize=10,
    tickfontsize=8, size=(620, 420))

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
function parse_cli(args)
    dim = :all
    geom = :all
    quick = false
    out = joinpath(dirname(dirname(@__DIR__)), ".scratch", "cost-order")
    hthreads = false
    for a in args
        if startswith(a, "--dim=")
            v = split(a, "="; limit=2)[2]
            dim = v == "all" ? :all : Symbol(v)
            dim in (:all, :d2, :d3, Symbol("2"), Symbol("3")) ||
                error("--dim must be 2, 3, or all")
            dim === Symbol("2") && (dim = :d2)
            dim === Symbol("3") && (dim = :d3)
        elseif startswith(a, "--geom=")
            geom = Symbol(split(a, "="; limit=2)[2])
            geom in (:all, :volume, :boundary) || error("--geom must be volume, boundary, or all")
        elseif a == "--quick"
            quick = true
        elseif startswith(a, "--out=")
            out = split(a, "="; limit=2)[2]
        elseif a == "--threads"
            hthreads = true
        elseif a in ("-h", "--help")
            println("""
            cost_order.jl — dense / H / FMM assembly+matvec scaling
              --dim=2|3|all          (default all)
              --geom=volume|boundary|all
              --quick                first three sizes of each ladder
              --out=DIR              default .scratch/cost-order
              --threads              H-matrix ACA threads (default serial)
            """)
            exit(0)
        else
            error("unknown argument $a (try --help)")
        end
    end
    return (; dim, geom, quick, out, hthreads)
end

const CLI = parse_cli(ARGS)
mkpath(CLI.out)

const NMAX = 32
const RTOL = 1e-6
const FMM_EPS = 1e-8
const N_MV_WARM = 2
const N_MV_RUN = 7

const METHODS = (
    :full, :fmm, :hmatrix,
)

const METHOD_ORDER = (
    "full", "hmatrix",
    "fmm",
)

const METHOD_STYLE = Dict(
    "full" => (:black, :circle, "dense"),
    "hmatrix" => (:crimson, :utriangle, "H-matrix"),
    "fmm" => (:seagreen, :xcross, "FMM"),
)

# -----------------------------------------------------------------------------
# Geometry
# -----------------------------------------------------------------------------
function volume_grid(dim::Int, nside::Int)
    xs = range(0.0, 1.0; length=nside)
    if dim == 2
        return [SVector(float(x), float(y)) for y in xs for x in xs]
    end
    return [SVector(float(x), float(y), float(z)) for z in xs for y in xs for x in xs]
end

function square_boundary(n::Int)
    n >= 8 || throw(ArgumentError("square_boundary needs n ≥ 8"))
    s = range(0.0, 4.0; length=n + 1)[1:n]
    pts = Vector{SVector{2,Float64}}(undef, n)
    @inbounds for (i, t) in enumerate(s)
        if t < 1
            pts[i] = SVector(t, 0.0)
        elseif t < 2
            pts[i] = SVector(1.0, t - 1)
        elseif t < 3
            pts[i] = SVector(1 - (t - 2), 1.0)
        else
            pts[i] = SVector(0.0, 1 - (t - 3))
        end
    end
    return pts
end

function cube_surface(ndiv::Int)
    xs = range(0.0, 1.0; length=ndiv + 1)
    seen = Set{NTuple{3,Float64}}()
    pts = SVector{3,Float64}[]
    function pushpt!(x, y, z)
        k = (x, y, z)
        k in seen && return
        push!(seen, k)
        push!(pts, SVector(x, y, z))
        return nothing
    end
    for y in xs, x in xs
        pushpt!(x, y, 0.0)
        pushpt!(x, y, 1.0)
    end
    for z in xs, x in xs
        pushpt!(x, 0.0, z)
        pushpt!(x, 1.0, z)
    end
    for z in xs, y in xs
        pushpt!(0.0, y, z)
        pushpt!(1.0, y, z)
    end
    return pts
end

function make_points(dim::Int, geom::Symbol, size_param::Int)
    if geom === :volume
        return volume_grid(dim, size_param)
    elseif dim == 2
        return square_boundary(size_param)
    else
        return cube_surface(size_param)
    end
end

function points_matrix(pts)
    d = length(pts[1])
    P = Matrix{Float64}(undef, d, length(pts))
    @inbounds for j in eachindex(pts)
        p = pts[j]
        for a in 1:d
            P[a, j] = p[a]
        end
    end
    return P
end

function laplace_kernel(dim::Int)
    if dim == 2
        return (x, y) -> begin
            r = norm(x - y)
            return r < 1e-30 ? 0.0 : log(r)
        end
    end
    inv4π = 1 / (4π)
    return (x, y) -> begin
        r = norm(x - y)
        return r < 1e-30 ? 0.0 : inv4π / r
    end
end

function size_ladders(quick::Bool)
    if quick
        return (
            volume2d=[16, 24, 32],
            boundary2d=[256, 512, 1024],
            volume3d=[8, 10, 12],
            boundary3d=[4, 6, 8],
        )
    end
    return (
        volume2d=[16, 24, 32, 48, 64, 80, 100],          # N = 256 … 10_000
        boundary2d=[256, 512, 1024, 2048, 4096, 8192, 10000],
        volume3d=[8, 10, 12, 16, 20, 22],                # N = 512 … 10_648
        boundary3d=[8, 12, 16, 24, 32, 41],              # N = 386 … 10_088
    )
end

function skip_method(method::Symbol, n::Int, dim::Int, geom::Symbol)
    # Dense N² fill: ~800 MB at 10⁴; keep N ≲ 4k on an 8 GB box.
    method === :full && return n > 4500
    method === :hmatrix && return n > 12000
    return false
end

# -----------------------------------------------------------------------------
# Assemble / time
# -----------------------------------------------------------------------------
function assemble_op(method::Symbol, K, tree, Pmat; hthreads::Bool)
    dim = size(Pmat, 1)
    if method === :full
        return Matrix(K)
    elseif method === :hmatrix
        return assemble_hmatrix(K, tree, tree;
            adm=StrongAdmissibilityStd(; eta=3.0),
            comp=PartialACA(; rtol=RTOL),
            threads=hthreads)
    elseif method === :fmm
        return dim == 2 ?
               FMM.fmm_laplace2d_matrix(Pmat; eps=FMM_EPS, nmax=NMAX) :
               FMM.fmm_laplace3d_matrix(Pmat; eps=FMM_EPS, nmax=NMAX)
    end
    throw(ArgumentError("unknown method $method"))
end

function time_matvec(A, x; nwarm=N_MV_WARM, nrun=N_MV_RUN)
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

function op_maxrank(A)
    try
        return maxrank(A)
    catch
        return 0
    end
end

function fit_exponent(ns, ts)
    mask = [t > 0 && isfinite(t) && n > 0 for (n, t) in zip(ns, ts)]
    n = collect(ns[mask])
    t = collect(ts[mask])
    length(n) < 3 && return NaN
    take = min(4, length(n))
    n = n[(end - take + 1):end]
    t = t[(end - take + 1):end]
    x = log.(Float64.(n))
    y = log.(t)
    M = hcat(x, ones(length(x)))
    α = (M \ y)[1]
    return α
end

# -----------------------------------------------------------------------------
# One (dim, geom) sweep
# -----------------------------------------------------------------------------
function run_sweep(dim::Int, geom::Symbol, sizes; hthreads::Bool)
    rows = NamedTuple[]
    ker = laplace_kernel(dim)
    println()
    println("="^72)
    @printf("  %d-D  %s   threads=%s\n", dim, geom, hthreads)
    println("="^72)
    @printf("  %-14s %6s %10s %10s %8s %9s %8s\n",
        "method", "n", "t_asm", "t_mv", "memMB", "err", "rank")

    yref = nothing
    nref = 0
    warmed = Set{Symbol}()

    for sp in sizes
        pts = make_points(dim, geom, sp)
        n = length(pts)
        Pmat = points_matrix(pts)
        tree = ClusterTree(pts, PrincipalComponentSplitter(; nmax=NMAX))
        K = KernelMatrix(ker, pts, pts)
        x = randn(n)
        yref_n = nothing

        for method in METHODS
            skip_method(method, n, dim, geom) && continue
            local A, t_asm, t_mv, y
            try
                if method ∉ warmed
                    assemble_op(method, K, tree, Pmat; hthreads=hthreads)
                    push!(warmed, method)
                end
                t_asm = @elapsed begin
                    A = assemble_op(method, K, tree, Pmat; hthreads=hthreads)
                end
                t_mv, y = time_matvec(A, x)
            catch e
                @warn "method $method failed at n=$n ($dim-D $geom)" exception = e
                continue
            end

            if method === :full || (yref_n === nothing && method === :fmm)
                yref_n = copy(y)
                if method === :full
                    yref = copy(y)
                    nref = n
                end
            end
            ref = yref_n
            if ref === nothing && yref !== nothing && nref == n
                ref = yref
            end
            err = ref === nothing ? NaN : norm(y - ref) / (norm(ref) + 1e-14)
            mem = Base.summarysize(A)
            cr = (n * n * sizeof(Float64)) / max(mem, 1)
            rk = op_maxrank(A)
            @printf("  %-14s %6d %10.3e %10.3e %8.2f %9.2e %8d\n",
                method, n, t_asm, t_mv, mem / 1024^2, err, rk)
            flush(stdout)
            push!(rows, (;
                dim, geom=String(geom), method=String(method), n, size_param=sp,
                t_asm, t_mv, mem_bytes=mem, compression=cr, rel_err=err, maxrank=rk,
            ))
            A = nothing
        end
        GC.gc(false)
    end
    return rows
end

function write_csv(path, rows)
    open(path, "w") do io
        println(io, "dim,geom,method,n,size_param,t_asm,t_mv,mem_bytes,compression,rel_err,maxrank")
        for r in rows
            @printf(io, "%d,%s,%s,%d,%d,%.6e,%.6e,%d,%.6e,%.6e,%d\n",
                r.dim, r.geom, r.method, r.n, r.size_param,
                r.t_asm, r.t_mv, r.mem_bytes, r.compression, r.rel_err, r.maxrank)
        end
    end
    return path
end

function by_method(rows)
    d = Dict{String,Vector{NamedTuple}}()
    for r in rows
        push!(get!(d, r.method, NamedTuple[]), r)
    end
    for k in keys(d)
        sort!(d[k]; by=r -> r.n)
    end
    return d
end

function add_guides!(plt, nmin, nmax, tref, nref)
    ns = 10 .^ range(log10(nmin), log10(nmax); length=40)
    tN = tref * (ns ./ nref)
    tNlog = tref * (ns .* log.(ns)) / (nref * log(nref))
    tN2 = tref * (ns ./ nref) .^ 2
    plot!(plt, ns, tN; ls=:dash, lc=:gray, lw=0.8, label="N")
    plot!(plt, ns, tNlog; ls=:dot, lc=:gray, lw=0.8, label="N log N")
    plot!(plt, ns, tN2; ls=:dashdot, lc=:gray, lw=0.8, label="N^2")
    return plt
end

function plot_sweep(rows, dim, geom, outdir)
    isempty(rows) && return
    R = by_method(rows)
    nall = [r.n for r in rows]
    nmin, nmax = extrema(nall)
    title = "$(dim)D $(geom)"

    function make_fig(field, ylab, fname)
        plt = plot(; xscale=:log10, yscale=:log10, xlabel=L"N", ylabel=ylab,
            title=title, legend=:outertopright, size=(760, 440))
        tguide = nothing
        nguide = nothing
        for m in METHOD_ORDER
            haskey(R, m) || continue
            rs = R[m]
            ns = [r.n for r in rs]
            ts = [getfield(r, field) for r in rs]
            col, mk, lab = METHOD_STYLE[m]
            plot!(plt, ns, ts; marker=mk, color=col, label=lab)
            if tguide === nothing && length(ts) >= 1 && ts[1] > 0
                tguide = ts[1]
                nguide = ns[1]
            end
        end
        if tguide !== nothing
            add_guides!(plt, nmin, nmax, tguide, Float64(nguide))
        end
        path = joinpath(outdir, fname)
        savefig(plt, path)
        println("  wrote ", path)
        return plt
    end

    tag = "$(dim)d_$(geom)"
    make_fig(:t_asm, "assembly time [s]", "fig_asm_$(tag).png")
    make_fig(:t_mv, "matvec time [s]", "fig_mv_$(tag).png")
    return nothing
end

function exponent_table(rows)
    lines = String[]
    R = by_method(rows)
    push!(lines, @sprintf("%-14s  %-8s  %6s  %6s", "method", "geom", "α_asm", "α_mv"))
    for m in METHOD_ORDER
        haskey(R, m) || continue
        rs = R[m]
        geoms = unique(r.geom for r in rs)
        for g in geoms
            sub = filter(r -> r.geom == g, rs)
            αa = fit_exponent([r.n for r in sub], [r.t_asm for r in sub])
            αm = fit_exponent([r.n for r in sub], [r.t_mv for r in sub])
            push!(lines, @sprintf("%-14s  %-8s  %6.2f  %6.2f", m, g, αa, αm))
        end
    end
    return join(lines, "\n")
end

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
function main()
    Random.seed!(1)
    println("="^72)
    println(" Cost-order  dense / H-matrix / FMM")
    println(" ", Dates.now())
    println(" threads=", Threads.nthreads(), "  H-ACA threads=", CLI.hthreads)
    println("="^72)

    dims = if CLI.dim === :all
        (2, 3)
    elseif CLI.dim === :d2
        (2,)
    else
        (3,)
    end
    geoms = CLI.geom === :all ? (:volume, :boundary) : (CLI.geom,)
    L = size_ladders(CLI.quick)

    all_rows = NamedTuple[]
    exp_io = IOBuffer()
    println(exp_io, "empirical t ~ N^α  (last up to 4 points, ≥3 required)")
    println(exp_io, "generated ", Dates.now())
    println(exp_io)

    for dim in dims, geom in geoms
        sizes = if dim == 2 && geom === :volume
            L.volume2d
        elseif dim == 2
            L.boundary2d
        elseif geom === :volume
            L.volume3d
        else
            L.boundary3d
        end
        rows = run_sweep(dim, geom, sizes; hthreads=CLI.hthreads)
        append!(all_rows, rows)
        plot_sweep(rows, dim, geom, CLI.out)
        println(exp_io, "$(dim)D $(geom)")
        println(exp_io, exponent_table(rows))
        println(exp_io)
        write_csv(joinpath(CLI.out, "results.csv"), all_rows)
        println("  checkpoint ", length(all_rows), " rows")
        flush(stdout)
    end

    csv = joinpath(CLI.out, "results.csv")
    write_csv(csv, all_rows)
    println("\nWrote ", csv)

    exp_path = joinpath(CLI.out, "exponents.txt")
    write(exp_path, String(take!(exp_io)))
    println("Wrote ", exp_path)
    print(read(exp_path, String))
    println("Done.")
    return all_rows
end

main()
