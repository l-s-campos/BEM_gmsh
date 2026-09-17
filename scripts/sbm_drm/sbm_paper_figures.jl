# Space–time refinement + figures for the SBM transient paper.
#
# julia --project=. scripts/sbm_paper_figures.jl
# ENV: STUDY_OUT=...  STUDY_QUICK=1
using DrWatson
@quickactivate :BEM
using LinearAlgebra
using StaticArrays
using Printf
using Statistics
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
    titlefontsize = 10,
)

# GR Computer Modern has no Unicode math (², Δ, —). Route those glyphs
# through LaTeXStrings so GR's TeX renderer uses CM math fonts.

include(joinpath(@__DIR__, "sbm_transient_study.jl"))

const OUT = STUDY_OUT
const FIG = joinpath(OUT, "figures")
mkpath(FIG)

const QUICK_FIG = get(ENV, "STUDY_QUICK", "0") == "1"
const NB_LIST = QUICK_FIG ? [8, 16] : [8, 12, 16, 24]
const NS_LIST = QUICK_FIG ? [30, 120] : [30, 60, 120, 240]
const NINT_OF = nb -> max(3, nb ÷ 2)          # interior grid ~ half boundary density
const METHODS = (:sbm, :dibem, :drm)
const SCHEME = :houbolt

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

function solve_field(ex::Int, method::Symbol; nb, nsteps, scheme=SCHEME)
    dad = make_dad(ex; nb=nb, nint=NINT_OF(nb))
    N, ni = dad.n, dad.ni
    u0 = fill(u_init, N + ni)
    @inbounds for i in 1:N
        dad.BC[i] == 0 && (u0[i] = 0.0)
    end
    Δt = tf0 / nsteps
    rH = ex == 3 ? 10.0 : nothing
    rUf = ex == 3 ? 50.0 : nothing
    t_cpu = @elapsed begin
        if method === :sbm
            sol = solve_sbm_drm(dad; κ=κ0, Δt=Δt, tf=tf0, u0=u0, scheme=scheme,
                                basis=RBF_SBM, robin_H=rH, robin_uf=rUf)
            U = sol.U
            tgrid = sol.t
        elseif method === :dibem
            sol = bem_dibem(dad, u0; κ=κ0, Δt=Δt, tf=tf0, scheme=scheme,
                            robin_H=rH, robin_uf=rUf)
            U = sol.U
            tgrid = sol.t
        elseif method === :drm
            sol = bem_drm(dad, u0; κ=κ0, Δt=Δt, tf=tf0, scheme=scheme,
                          robin_H=rH, robin_uf=rUf)
            U = sol.U
            tgrid = sol.t
        else
            error("method")
        end
    end
    pts = Point2D[Point2D(p) for p in vcat(dad.Nodes, dad.internalNodes)]
    return (; dad, pts, U, t=tgrid, t_cpu, N, ni, nb, nsteps, method, ex)
end

function field_errors(res)
    ex = res.ex
    Uend = res.U[:, end]
    if ex <= 2
        uex = [ex == 1 ? exact_ex1(p[1], p[2], tf0) : exact_ex2(p[1], p[2], tf0)
               for p in res.pts]
        rmse, rinf = rmse_rinf(Uend, uex)
        return (; rmse, rinf, uex, Uend)
    end
    return (; rmse=NaN, rinf=NaN, uex=nothing, Uend)
end

function scatter_field(pts, vals; title="", clims=nothing)
    x = [p[1] for p in pts]
    y = [p[2] for p in pts]
    kw = (xlabel="x", ylabel="y", title=title, aspect_ratio=1,
          colorbar_title="u", ms=3.5, marker=:circle, label=false)
    if clims !== nothing
        return scatter(x, y; zcolor=vals, clims=clims, kw...)
    end
    return scatter(x, y; zcolor=vals, kw...)
end

meth_label(m) = m === :sbm ? "SBM-DRM" : m === :dibem ? "BEM-DIBEM" : "BEM-DRM"

# ---------------------------------------------------------------------------
# 1) Space–time refinement tables
# ---------------------------------------------------------------------------

function run_refinement()
    rows = NamedTuple[]
    println("="^72)
    println(" Space–time refinement")
    println("="^72)
    for ex in 1:3, meth in METHODS, nb in NB_LIST, ns in NS_LIST
        # skip known-unstable BEM-DRM on ex2 fine time for speed? still run; mark ok
        try
            res = solve_field(ex, meth; nb=nb, nsteps=ns)
            err = field_errors(res)
            ok = all(isfinite, err.Uend) && maximum(abs, err.Uend) < 1e6
            r = (
                ex=ex, method=string(meth), label=meth_label(meth),
                nb=nb, n=res.N, ni=res.ni, nsteps=ns, dt=tf0/ns,
                rmse=err.rmse, rinf=err.rinf,
                maxu=maximum(abs, err.Uend),
                meanu=mean(err.Uend),
                ucenter=begin
                    ic = argmin(norm(p - Point2D(Lx0/2, Ly0/2)) for p in res.pts)
                    err.Uend[ic]
                end,
                t_cpu=res.t_cpu, ok=ok,
            )
            push!(rows, r)
            @printf("ex%d %-10s nb=%2d ns=%3d n=%3d RMSE=%9.2e max=%.3e t=%.2fs %s\n",
                    ex, meth_label(meth), nb, ns, res.N, err.rmse, r.maxu, r.t_cpu,
                    ok ? "ok" : "FAIL")
        catch e
            @warn "refine fail" ex meth nb ns exception=e
            push!(rows, (
                ex=ex, method=string(meth), label=meth_label(meth),
                nb=nb, n=0, ni=0, nsteps=ns, dt=tf0/ns,
                rmse=NaN, rinf=NaN, maxu=NaN, meanu=NaN, ucenter=NaN,
                t_cpu=NaN, ok=false,
            ))
        end
    end
    write_csv(joinpath(OUT, "refine_all.csv"), rows)
    for ex in 1:3
        write_csv(joinpath(OUT, "refine_ex$(ex).csv"), filter(r -> r.ex == ex, rows))
    end
    return rows
end

# ---------------------------------------------------------------------------
# 2) Figures
# ---------------------------------------------------------------------------

function fig_convergence(rows)
    for ex in 1:2
        # spatial: fixed fine time ns=max
        ns_fix = maximum(NS_LIST)
        plt_h = plot(xlabel="N (boundary nodes)", ylabel="RMSE",
                     title="Ex$ex — spatial refinement (Houbolt, $ns_fix steps)",
                     xscale=:log10, yscale=:log10, legend=:bottomleft)
        for meth in METHODS
            sub = filter(r -> r.ex == ex && r.method == string(meth) &&
                         r.nsteps == ns_fix && r.ok && isfinite(r.rmse), rows)
            isempty(sub) && continue
            sort!(sub; by=r -> r.n)
            plot!(plt_h, [r.n for r in sub], [r.rmse for r in sub];
                  marker=:circle, label=meth_label(meth))
        end
        # reference slopes
        if ex == 1
            nref = [40.0, 200.0]
            plot!(plt_h, nref, 2.0 ./ nref; ls=:dash, color=:gray, label=L"O(1/N)")
            plot!(plt_h, nref, 30.0 ./ nref .^ 2; ls=:dot, color=:gray, label=L"O(1/N^2)")
        end
        savefig(plt_h, joinpath(FIG, "ex$(ex)_conv_space.pdf"))
        savefig(plt_h, joinpath(FIG, "ex$(ex)_conv_space.png"))

        # temporal: fixed fine space nb=max
        nb_fix = maximum(NB_LIST)
        plt_t = plot(xlabel=L"\Delta t", ylabel="RMSE",
                     title="Ex$ex — temporal refinement (Houbolt, nb=$nb_fix)",
                     xscale=:log10, yscale=:log10, legend=:bottomright)
        for meth in METHODS
            sub = filter(r -> r.ex == ex && r.method == string(meth) &&
                         r.nb == nb_fix && r.ok && isfinite(r.rmse), rows)
            isempty(sub) && continue
            sort!(sub; by=r -> r.dt)
            plot!(plt_t, [r.dt for r in sub], [r.rmse for r in sub];
                  marker=:square, label=meth_label(meth))
        end
        dtref = [tf0/240, tf0/30]
        plot!(plt_t, dtref, 0.5 .* dtref; ls=:dash, color=:gray, label=L"O(\Delta t)")
        plot!(plt_t, dtref, 2.0 .* dtref .^ 2; ls=:dot, color=:gray, label=L"O(\Delta t^2)")
        savefig(plt_t, joinpath(FIG, "ex$(ex)_conv_time.pdf"))
        savefig(plt_t, joinpath(FIG, "ex$(ex)_conv_time.png"))
        println("wrote conv ex$ex")
    end

    # Ex3: centre value vs N and vs dt (no exact)
    for (xkey, xlab, fname) in (
        (:n, "N (boundary nodes)", "ex3_conv_space"),
        (:dt, L"\Delta t", "ex3_conv_time"),
    )
        plt = plot(xlabel=xlab, ylabel=L"u(\mathrm{center})",
                   title="Ex3 — centre temperature (Houbolt)",
                   xscale=:log10, legend=:best)
        fix = xkey === :n ? maximum(NS_LIST) : maximum(NB_LIST)
        for meth in METHODS
            sub = if xkey === :n
                filter(r -> r.ex == 3 && r.method == string(meth) &&
                       r.nsteps == fix && r.ok, rows)
            else
                filter(r -> r.ex == 3 && r.method == string(meth) &&
                       r.nb == fix && r.ok, rows)
            end
            isempty(sub) && continue
            sort!(sub; by=r -> getfield(r, xkey))
            plot!(plt, [getfield(r, xkey) for r in sub], [r.ucenter for r in sub];
                  marker=:circle, label=meth_label(meth))
        end
        savefig(plt, joinpath(FIG, fname * ".pdf"))
        savefig(plt, joinpath(FIG, fname * ".png"))
    end
    println("wrote conv ex3")
end

function fig_fields()
    # Representative discretizations: coarse / medium / fine (space), fixed fine time
    nb_show = QUICK_FIG ? [8, 16] : [8, 16, 24]
    ns_fix = maximum(NS_LIST)

    for ex in 1:3
        # common color limits from exact or SBM fine
        cl = nothing
        if ex <= 2
            # sample exact on fine grid
            xs = range(0, Lx0; length=40)
            ys = range(0, Ly0; length=40)
            vals = Float64[]
            for y in ys, x in xs
                push!(vals, ex == 1 ? exact_ex1(x, y, tf0) : exact_ex2(x, y, tf0))
            end
            cl = (minimum(vals), maximum(vals))
        end

        panels = Plots.Plot[]
        # exact panel for ex1/2
        if ex <= 2
            dad = make_dad(ex; nb=maximum(nb_show), nint=NINT_OF(maximum(nb_show)))
            pts = Point2D[Point2D(p) for p in vcat(dad.Nodes, dad.internalNodes)]
            uex = [ex == 1 ? exact_ex1(p[1], p[2], tf0) : exact_ex2(p[1], p[2], tf0)
                   for p in pts]
            push!(panels, scatter_field(pts, uex; title="Exact", clims=cl))
        end

        for meth in (:sbm, :dibem)
            for nb in nb_show
                res = solve_field(ex, meth; nb=nb, nsteps=ns_fix)
                err = field_errors(res)
                ok = all(isfinite, err.Uend) && maximum(abs, err.Uend) < 1e6
                ttl = "$(meth_label(meth))\nnb=$nb (N=$(res.N))"
                if !ok
                    push!(panels, plot(title=ttl * " FAIL", framestyle=:none))
                    continue
                end
                # for ex3 set clims from data range across methods later
                cl_use = cl
                if ex == 3
                    cl_use = (0.0, max(200.0, maximum(err.Uend)))
                end
                push!(panels, scatter_field(res.pts, err.Uend; title=ttl, clims=cl_use))
            end
        end

        ncol = ex <= 2 ? 1 + length(nb_show) : length(nb_show)
        # layout: rows = methods (+ exact), cols = discretizations
        # rebuild more structured layout
        plts = Plots.Plot[]
        if ex <= 2
            dad = make_dad(ex; nb=maximum(nb_show), nint=NINT_OF(maximum(nb_show)))
            pts = Point2D[Point2D(p) for p in vcat(dad.Nodes, dad.internalNodes)]
            uex = [ex == 1 ? exact_ex1(p[1], p[2], tf0) : exact_ex2(p[1], p[2], tf0)
                   for p in pts]
            push!(plts, scatter_field(pts, uex; title="Exact (ref mesh)", clims=cl))
            # pad exact row
            for _ in 2:length(nb_show)
                push!(plts, plot(framestyle=:none, axis=false, ticks=false, title=""))
            end
        end
        for meth in (:sbm, :dibem)
            for nb in nb_show
                res = solve_field(ex, meth; nb=nb, nsteps=ns_fix)
                err = field_errors(res)
                ok = all(isfinite, err.Uend) && maximum(abs, err.Uend) < 1e6
                ttl = "$(meth_label(meth)), nb=$nb"
                cl_use = ex == 3 ? (0.0, 200.0) : cl
                if ok
                    push!(plts, scatter_field(res.pts, err.Uend; title=ttl, clims=cl_use))
                else
                    push!(plts, plot(title=ttl * " FAIL", framestyle=:none))
                end
            end
        end
        nrow = ex <= 2 ? 3 : 2
        ncol = length(nb_show)
        # if exact row has ncol panels (1 real + pads)
        if ex <= 2
            lay = (3, ncol)
        else
            lay = (2, ncol)
        end
        fig = plot(plts...; layout=lay, size=(280 * ncol, 260 * lay[1]))
        savefig(fig, joinpath(FIG, "ex$(ex)_fields.pdf"))
        savefig(fig, joinpath(FIG, "ex$(ex)_fields.png"))
        println("wrote fields ex$ex")

        # error fields for ex1/2 at medium mesh
        if ex <= 2
            nb_m = nb_show[min(2, length(nb_show))]
            eplots = Plots.Plot[]
            for meth in (:sbm, :dibem, :drm)
                res = solve_field(ex, meth; nb=nb_m, nsteps=ns_fix)
                err = field_errors(res)
                ok = all(isfinite, err.Uend) && err.uex !== nothing
                if !ok
                    push!(eplots, plot(title="$(meth_label(meth)) FAIL", framestyle=:none))
                    continue
                end
                ev = abs.(err.Uend .- err.uex)
                push!(eplots, scatter_field(res.pts, ev;
                    title="$(meth_label(meth))\n|e|, RMSE=$(@sprintf("%.2e", err.rmse))"))
            end
            fig_e = plot(eplots...; layout=(1, 3), size=(900, 300))
            savefig(fig_e, joinpath(FIG, "ex$(ex)_abserr.pdf"))
            savefig(fig_e, joinpath(FIG, "ex$(ex)_abserr.png"))
            println("wrote abserr ex$ex")
        end
    end
end

function fig_histories()
    nb = QUICK_FIG ? 12 : 20
    ns = QUICK_FIG ? 60 : 120
    for ex in 1:2
        plt = plot(xlabel="t", ylabel="u(center)",
                   title="Ex$ex - centre history (nb=$nb, $ns steps)")
        # exact
        tgrid = collect(0.0:tf0/ns:tf0)
        ue = [ex == 1 ? exact_ex1(Lx0/2, Ly0/2, tt) : exact_ex2(Lx0/2, Ly0/2, tt)
              for tt in tgrid]
        plot!(plt, tgrid, ue; color=:black, label="exact", lw=2)
        for meth in (:sbm, :dibem)
            res = solve_field(ex, meth; nb=nb, nsteps=ns)
            ic = argmin(norm(p - Point2D(Lx0/2, Ly0/2)) for p in res.pts)
            plot!(plt, res.t, res.U[ic, :]; label=meth_label(meth), ls=:dash)
        end
        savefig(plt, joinpath(FIG, "ex$(ex)_history.pdf"))
        savefig(plt, joinpath(FIG, "ex$(ex)_history.png"))
        println("wrote history ex$ex")
    end
    # ex3 history
    plt = plot(xlabel="t", ylabel="u(center)",
               title="Ex3 - centre history (convection BC)")
    for meth in (:sbm, :dibem, :drm)
        res = solve_field(3, meth; nb=nb, nsteps=ns)
        ok = all(isfinite, res.U) && maximum(abs, res.U) < 1e6
        ok || continue
        ic = argmin(norm(p - Point2D(Lx0/2, Ly0/2)) for p in res.pts)
        plot!(plt, res.t, res.U[ic, :]; label=meth_label(meth))
    end
    savefig(plt, joinpath(FIG, "ex3_history.pdf"))
    savefig(plt, joinpath(FIG, "ex3_history.png"))
    println("wrote history ex3")
end

function fig_cpu(rows)
    plt = plot(xlabel="N", ylabel="CPU time (s)",
               title="Cost vs boundary size (Houbolt, $(maximum(NS_LIST)) steps)",
               xscale=:log10, yscale=:log10, legend=:topleft)
    ns = maximum(NS_LIST)
    for meth in METHODS
        sub = filter(r -> r.ex == 1 && r.method == string(meth) &&
                     r.nsteps == ns && r.ok && isfinite(r.t_cpu), rows)
        isempty(sub) && continue
        sort!(sub; by=r -> r.n)
        plot!(plt, [r.n for r in sub], [r.t_cpu for r in sub];
              marker=:circle, label=meth_label(meth))
    end
    savefig(plt, joinpath(FIG, "cpu_vs_n.pdf"))
    savefig(plt, joinpath(FIG, "cpu_vs_n.png"))
    println("wrote cpu_vs_n")
end

# ---------------------------------------------------------------------------

function load_refine_rows(path=joinpath(OUT, "refine_all.csv"))
    lines = readlines(path)
    hdr = split(lines[1], ',')
    rows = NamedTuple[]
    for line in Iterators.drop(lines, 1)
        isempty(strip(line)) && continue
        cols = split(line, ',')
        d = Dict(Symbol(hdr[i]) => cols[i] for i in eachindex(hdr))
        push!(rows, (
            ex=parse(Int, d[:ex]),
            method=String(d[:method]),
            label=String(d[:label]),
            nb=parse(Int, d[:nb]),
            n=parse(Int, d[:n]),
            ni=parse(Int, d[:ni]),
            nsteps=parse(Int, d[:nsteps]),
            dt=parse(Float64, d[:dt]),
            rmse=parse(Float64, d[:rmse]),
            rinf=parse(Float64, d[:rinf]),
            maxu=parse(Float64, d[:maxu]),
            meanu=parse(Float64, d[:meanu]),
            ucenter=parse(Float64, d[:ucenter]),
            t_cpu=parse(Float64, d[:t_cpu]),
            ok=d[:ok] == "true",
        ))
    end
    return rows
end

function main()
    if get(ENV, "FIG_ONLY", "0") == "1"
        rows = load_refine_rows()
        fig_convergence(rows)
        fig_cpu(rows)
        println("\nFigures (from refine_all.csv) → $FIG")
        return rows
    end
    rows = run_refinement()
    fig_convergence(rows)
    fig_fields()
    fig_histories()
    fig_cpu(rows)
    println("\nAll figures → $FIG")
    return rows
end

if endswith(replace(PROGRAM_FILE, "\\" => "/"), "sbm_paper_figures.jl")
    main()
end
