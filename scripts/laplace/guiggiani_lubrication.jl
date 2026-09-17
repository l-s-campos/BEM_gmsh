# Reproduce Guiggiani, EABE 119:183–188 (2020), with Laplace FS + DIBEM.
#
#   julia --project=. scripts/laplace/guiggiani_lubrication.jl
#
# Fig. 2  special films h1–h5 + linear wedge (hi/ho = 2)
# Fig. 3  infinite bearing (1-D Reynolds)
# Fig. 5  finite rounded pad, film h2: DIBEM vs particular-integral Laplace
#
using DrWatson
@quickactivate :BEM
using LinearAlgebra
using Printf
using Statistics: mean
using Plots

const OUTDIR = projectdir("plots", "guiggiani_lubrication")
mkpath(OUTDIR)

function _pad_internals!(dad; nx=13, ny=7, ys=(-0.1875, 0.0, 0.1875), nx_line=15)
    pts = internal_grid(dad, nx, ny; d_min=0.02, layout=:cell)
    xs = range(0.08, 0.92; length=nx_line)
    for y in ys, x in xs
        p = Point2D(float(x), float(y))
        point_in_domain(dad, p) || continue
        push!(pts, p)
    end
    seen = Set{NTuple{2,Float64}}()
    uniq = Point2D[]
    for p in pts
        key = (round(p[1]; digits=10), round(p[2]; digits=10))
        key in seen && continue
        push!(seen, key)
        push!(uniq, p)
    end
    return set_internal_nodes!(dad, uniq)
end

function _line_xy(dad, y; atol=0.02)
    xs = Float64[]
    ps = Float64[]
    @inbounds for i in 1:dad.nt
        pt = point(dad, i)
        abs(pt[2] - y) <= atol || continue
        push!(xs, pt[1])
        push!(ps, dad.T[i])
    end
    perm = sortperm(xs)
    return xs[perm], ps[perm]
end

function _save_tsv(path, cols, header)
    open(path, "w") do io
        println(io, join(header, "\t"))
        n = length(cols[1])
        for i in 1:n
            println(io, join((c[i] for c in cols), "\t"))
        end
    end
    return path
end

function main()
    a = 2.0
    films = guiggiani_films(; a=a, hi=a)
    xs = collect(range(0.0, 1.0; length=201))

    println("="^72)
    println("Guiggiani 2020 lubrication — Laplace FS + DIBEM")
    println("  hi/ho = $a,  L = 1,  μ = U = ho = 1  ⇒  p* = p")
    println("="^72)

    # --- Fig. 2 films ---
    println("\n## Fig. 2  film profiles")
    hmat = Dict(f.name => [f.h(x) for x in xs] for f in films)
    @printf("  %-8s  h(0.5)  h'(0)\n", "film")
    for f in films
        @printf("  %-8s  %6.4f  %7.4f\n", f.name, f.h(0.5), f.hx(0.0))
    end
    _save_tsv(joinpath(OUTDIR, "fig2_films.tsv"),
        (xs, hmat["h1"], hmat["h2"], hmat["h3"], hmat["h4"], hmat["h5"], hmat["linear"]),
        ("x", "h1", "h2", "h3", "h4", "h5", "linear"))

    # --- Fig. 3 infinite bearing ---
    println("\n## Fig. 3  infinite bearing  p ho²/(μ U L)")
    pinf = Dict(f.name => infinite_bearing_pressure(f, xs) for f in films)
    @printf("  %-8s  max p*\n", "film")
    for f in films
        @printf("  %-8s  %7.4f\n", f.name, maximum(pinf[f.name]))
    end
    _save_tsv(joinpath(OUTDIR, "fig3_infinite.tsv"),
        (xs, pinf["h1"], pinf["h2"], pinf["h3"], pinf["h4"], pinf["h5"], pinf["linear"]),
        ("x", "h1", "h2", "h3", "h4", "h5", "linear"))

    fig2 = plot(; size=(720, 420), xlabel="x", ylabel="film profile",
        title="Special film profiles (hi/ho = 2)", legend=:topright, ylim=(0, 4))
    plot!(fig2, xs, hmat["h1"]; label="h1 eq. (29)", linestyle=:dash, lw=2)
    plot!(fig2, xs, hmat["h2"]; label="h2 eq. (31)", linestyle=:dashdot, lw=2)
    plot!(fig2, xs, hmat["h3"]; label="h3 eq. (32)", linestyle=:dot, lw=2)
    plot!(fig2, xs, hmat["h4"]; label="h4 eq. (34)", lw=2)
    plot!(fig2, xs, hmat["h5"]; label="h5 eq. (35)", lw=2)
    plot!(fig2, xs, hmat["linear"]; label="linear wedge", color=:black, lw=2)
    savefig(fig2, joinpath(OUTDIR, "fig2_films.png"))

    fig3 = plot(; size=(720, 480), xlabel="x", ylabel="lubricant pressure p (non-dim)",
        title="Infinite bearing (hi/ho = 2)", legend=:topleft, ylim=(0, 0.3))
    plot!(fig3, xs, pinf["h1"]; label="film h1", linestyle=:dash, lw=2)
    plot!(fig3, xs, pinf["h2"]; label="film h2", linestyle=:dashdot, lw=2)
    plot!(fig3, xs, pinf["h3"]; label="film h3", linestyle=:dot, lw=2)
    plot!(fig3, xs, pinf["h4"]; label="film h4", lw=2)
    plot!(fig3, xs, pinf["h5"]; label="film h5", lw=2)
    plot!(fig3, xs, pinf["linear"]; label="linear wedge", color=:black, lw=2)
    savefig(fig3, joinpath(OUTDIR, "fig3_infinite.png"))
    println("  wrote ", joinpath(OUTDIR, "fig2_films.png"))
    println("  wrote ", joinpath(OUTDIR, "fig3_infinite.png"))

    # --- Fig. 5 finite pad, film h2 ---
    println("\n## Fig. 5  finite pad, film h2, Laplace FS + DIBEM")
    film = films.h2
    msh = mesh_guiggiani_pad(; nome="guiggiani_pad", show=false)
    dad = format2d(msh, Laplace(1.0); pontointerno=false, tipo=2)
    println("  elements = ", length(dad.elements), "  (paper: 16 quadratic)")
    println("  boundary nodes = ", dad.n)
    _pad_internals!(dad)
    println("  internal points = ", dad.ni)
    assemble!(dad; npg=12, threaded=true)
    dad_pi = deepcopy(dad)

    t0 = time()
    solve_reynolds_dibem!(dad, film; npg=12)
    t_dibem = time() - t0
    t0 = time()
    solve_reynolds_particular!(dad_pi, film; npg=12)
    t_pi = time() - t0

    pnd = reynolds_pressure.(dad.T, Ref(film))
    pnd_pi = reynolds_pressure.(dad_pi.T, Ref(film))
    rel = norm(dad.T - dad_pi.T) / (norm(dad_pi.T) + 1e-14)
    pmax_inf = maximum(pinf["h2"])
    pint = view(pnd, (dad.n + 1):dad.nt)
    println("  DIBEM wall time     ", round(t_dibem; digits=3), " s")
    println("  particular wall     ", round(t_pi; digits=3), " s")
    println("  ||p_DIBEM − p_PI|| / ||p_PI|| = ", @sprintf("%.3e", rel))
    println("  max p* interior (DIBEM) = ", @sprintf("%.4f", maximum(pint)))
    println("  max p* infinite h2      = ", @sprintf("%.4f", pmax_inf))

    x0, p0 = _line_xy(dad, 0.0; atol=1e-8)
    x1, p1 = _line_xy(dad, 0.1875; atol=1e-8)
    p0n = reynolds_pressure.(p0, Ref(film))
    p1n = reynolds_pressure.(p1, Ref(film))
    xs_line = collect(range(0.02, 0.98; length=81))
    function _pi_line(y)
        xx = Float64[]; pp = Float64[]
        for x in xs_line
            pf = Point2D(x, y)
            point_in_domain(dad_pi, pf) || continue
            push!(xx, x)
            push!(pp, eval_reynolds_particular(dad_pi, film, pf))
        end
        return xx, reynolds_pressure.(pp, Ref(film))
    end
    x0p, p0pn = _pi_line(0.0)
    x1p, p1pn = _pi_line(0.1875)
    println("  samples y=0      n=", length(x0), "  max p* DIBEM = ",
        isempty(p0n) ? "n/a" : @sprintf("%.4f", maximum(p0n)),
        "  PI = ", isempty(p0pn) ? "n/a" : @sprintf("%.4f", maximum(p0pn)))
    println("  samples y=0.1875 n=", length(x1), "  max p* DIBEM = ",
        isempty(p1n) ? "n/a" : @sprintf("%.4f", maximum(p1n)),
        "  PI = ", isempty(p1pn) ? "n/a" : @sprintf("%.4f", maximum(p1pn)))

    _save_tsv(joinpath(OUTDIR, "fig5_y0_dibem.tsv"), (x0, p0n), ("x", "p_nd"))
    _save_tsv(joinpath(OUTDIR, "fig5_y1875_dibem.tsv"), (x1, p1n), ("x", "p_nd"))
    _save_tsv(joinpath(OUTDIR, "fig5_y0_particular.tsv"), (x0p, p0pn), ("x", "p_nd"))
    _save_tsv(joinpath(OUTDIR, "fig5_y1875_particular.tsv"), (x1p, p1pn), ("x", "p_nd"))

    fig5 = plot(; size=(720, 480), xlabel="x", ylabel="lubricant pressure p (non-dim)",
        title="Finite bearing — film h2 (Laplace FS + DIBEM)", legend=:topleft, ylim=(0, 0.3))
    plot!(fig5, xs, pinf["h2"]; label="infinite bearing", color=:black, lw=2)
    plot!(fig5, x0, p0n; label="y = 0 (DIBEM)", marker=:diamond, markersize=10)
    plot!(fig5, x1, p1n; label="y = 0.1875 (DIBEM)", marker=:utriangle, markersize=10)
    plot!(fig5, x0p, p0pn; label="y = 0 (particular)", linestyle=:dash, lw=2)
    plot!(fig5, x1p, p1pn; label="y = 0.1875 (particular)", linestyle=:dot, lw=2)
    savefig(fig5, joinpath(OUTDIR, "fig5_finite.png"))
    println("  wrote ", joinpath(OUTDIR, "fig5_finite.png"))

    println("\nDone. Tables/figures in ", OUTDIR)
    return nothing
end

main()
