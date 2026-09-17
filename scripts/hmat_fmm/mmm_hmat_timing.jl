# H-matrix vs dense MMM setup cost across discretizations.
# Assembly (H,G,M) + condensed apply / shift-invert setup.
# julia --project=. scripts/mmm_hmat_timing.jl
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf, Statistics
using Plots, LaTeXStrings
gr()
default(fontfamily="Computer Modern", linewidth=1.8, framestyle=:box,
        grid=false, dpi=180, legendfontsize=8, guidefontsize=10,
        tickfontsize=8, size=(560, 400))

include(datadir("Laplace", "Laplace_dad.jl"))

const OUT = get(ENV, "STUDY_OUT",
    raw"C:\Users\lucas.s.campos\OneDrive\artigos\escritos\2026\SBM transient\results")
const FIG = joinpath(OUT, "figures")
mkpath(FIG)
const CSV = joinpath(OUT, "mmm_hmat_timing.csv")

const NDIVS = let s = get(ENV, "MMM_NDIVS", "8,12,16,24,32")
    parse.(Int, split(s, ','; keepempty=false))
end

function make_dad(ndiv)
    msh = quadrado(ndiv=ndiv, show=false, nome="mmm_time_$ndiv")
    dad = format2d(msh, Laplace(1.0); tipo=1, pontointerno=false)
    nint = max(4, ndiv - 4)
    xs = range(0.12, 0.88; length=nint)
    set_internal_nodes!(dad, [Point2D(x, y) for x in xs, y in xs])
    return dad
end

function assemble_dense!(dad)
    H_G_full_direct(dad; npg=8, threaded=false)
    DIBEM(dad; method=:dense, rbf=PHS(1; poly_deg=-1))
    return dad
end

function assemble_hmat!(dad)
    qq, ww = gausslegendre(8)
    set_cache!(dad; qsi=qq, w=ww)
    H_G_Hmat(dad; atol=1e-6, nmax=32, threads=false)
    DIBEM(dad; method=:hmatrix, rbf=PHS(1; poly_deg=-1), atol=1e-6, nmax=32)
    return dad
end

function time_apply(P; nrep=8)
    nf = length(P.free)
    x = randn(nf)
    y = zeros(nf)
    mul_Kbar!(y, P, x)
    mul_Mbar!(y, P, x)
    t = @elapsed for _ in 1:nrep
        mul_Kbar!(y, P, x)
        mul_Mbar!(y, P, x)
    end
    return t / nrep
end

function time_one(ndiv, kind::Symbol)
    dad = make_dad(ndiv)
    t_asm = if kind === :dense
        @elapsed assemble_dense!(dad)
    else
        @elapsed assemble_hmat!(dad)
    end
    t_pen = @elapsed (P = condensed_pencil(dad))
    t_app = time_apply(P)
    return (; ndiv, n=dad.nt, kind=String(kind), t_asm, t_pen, t_app,
            t_setup=t_asm + t_pen)
end

function main()
    rows = NamedTuple[]
    println("ndiv   n     kind     t_asm    t_pen    t_app    t_setup")
    for ndiv in NDIVS, kind in (:dense, :hmat)
        r = time_one(ndiv, kind)
        push!(rows, r)
        @printf("%4d %5d  %-6s  %8.3f %8.3f %8.4f %8.3f\n",
            r.ndiv, r.n, r.kind, r.t_asm, r.t_pen, r.t_app, r.t_setup)
    end
    open(CSV, "w") do io
        println(io, "ndiv,n,kind,t_asm,t_pen,t_app,t_setup")
        for r in rows
            @printf(io, "%d,%d,%s,%.6f,%.6f,%.6f,%.6f\n",
                r.ndiv, r.n, r.kind, r.t_asm, r.t_pen, r.t_app, r.t_setup)
        end
    end
    ns = [r.n for r in rows if r.kind == "dense" && r.ndiv > 8]

    td = [only(r.t_setup for r in rows if r.n == n && r.kind == "dense") for n in ns]
    th = [only(r.t_setup for r in rows if r.n == n && r.kind == "hmat") for n in ns]
    ad = [only(r.t_app for r in rows if r.n == n && r.kind == "dense") for n in ns]
    ah = [only(r.t_app for r in rows if r.n == n && r.kind == "hmat") for n in ns]

    p1 = plot(ns, td; marker=:circle, label="dense", color=:black,
        xscale=:log10, yscale=:log10, xlabel=L"n_t",
        ylabel="assembly + pencil (s)", title="MMM setup")
    plot!(p1, ns, th; marker=:diamond, label="H-matrix", color=:crimson)
    savefig(p1, joinpath(FIG, "mmm_hmat_vs_dense_setup.pdf"))
    savefig(p1, joinpath(FIG, "mmm_hmat_vs_dense_setup.png"))

    p2 = plot(ns, ad; marker=:circle, label="dense", color=:black,
        xscale=:log10, yscale=:log10, xlabel=L"n_t",
        ylabel="one condensed apply (s)", title="MMM matvec")
    plot!(p2, ns, ah; marker=:diamond, label="H-matrix", color=:crimson)
    savefig(p2, joinpath(FIG, "mmm_hmat_vs_dense_apply.pdf"))
    savefig(p2, joinpath(FIG, "mmm_hmat_vs_dense_apply.png"))

    p3 = plot(p1, p2; layout=(1, 2), size=(900, 380))
    savefig(p3, joinpath(FIG, "mmm_hmat_vs_dense.pdf"))
    savefig(p3, joinpath(FIG, "mmm_hmat_vs_dense.png"))
    println("wrote $CSV and figures in $FIG")
    return rows
end

if endswith(replace(PROGRAM_FILE, "\\" => "/"), "mmm_hmat_timing.jl")
    main()
end
