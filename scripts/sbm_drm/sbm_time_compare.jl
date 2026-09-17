# Example 2, moderate mesh (nb=nint=16): Houbolt vs fixed-dt Rodas5P.
# julia --project=. scripts/sbm_time_compare.jl
using DrWatson
@quickactivate :BEM
using Printf
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
    size = (540, 400),
)

include(joinpath(@__DIR__, "sbm_transient_study.jl"))

const FIG = joinpath(STUDY_OUT, "figures")
mkpath(FIG)

const NB, NINT = 16, 16
const NS_LIST = (1, 2, 3, 4, 5, 8, 15, 30, 60, 120, 240, 480)
const NS_SNAP = 5

function rodas_run(ns)
    par = case_params(2)
    dad = make_dad(2; nb=NB, nint=NINT)
    u0 = fill(par.u0, dad.nt)
    @inbounds for i in 1:dad.n
        dad.BC[i] == 0 && (u0[i] = dad.BV[i])
    end
    sol = bem_diffeq(dad, u0; κ=par.κ, tf=par.tf, alg=Rodas5P(),
                     dt=par.tf / ns, adaptive=false)
    U = sol.U[1:dad.n + dad.ni, end]
    uex = exact_vec(dad, 2, par.tf)
    rmse = all(isfinite, U) ? rmse_rinf(U, uex)[1] : NaN
    ok = all(isfinite, U) && maximum(abs, U) < 1e6 && string(sol.retcode) == "Success"
    pts = vcat(dad.Nodes, dad.internalNodes)
    return (; nsteps=ns, dt=par.tf / ns, rmse, maxu=maximum(abs, U), ok,
            U, uex, pts, dad)
end

function field_panel(pts, vals, title; clims)
    xs = [p[1] for p in pts]
    ys = [p[2] for p in pts]
    return scatter(xs, ys; zcolor=vals, marker=:circle, ms=3.2, msw=0,
                   clims=clims, colorbar=true, aspect_ratio=1,
                   xlabel=L"x", ylabel=L"y", title=title, legend=false,
                   xlims=(-0.1, 3.1), ylims=(-0.1, 3.1))
end

function main()
    rows_h = NamedTuple[]
    rows_r = NamedTuple[]
    snap = nothing
    for ns in NS_LIST
        rh = run_case(; ex=2, method=:dibem, scheme=:houbolt, nsteps=ns, nb=NB, nint=NINT)
        rr = rodas_run(ns)
        push!(rows_h, (nsteps=ns, dt=1.2 / ns, rmse=rh.ok ? rh.rmse : NaN,
                       maxu=rh.maxu, ok=rh.ok))
        push!(rows_r, (nsteps=ns, dt=rr.dt, rmse=rr.ok ? rr.rmse : NaN,
                       maxu=rr.maxu, ok=rr.ok))
        ns == NS_SNAP && (snap = rr)
        @printf("ns=%4d  H RMSE=%9.2e %-4s  R RMSE=%9.2e %-4s\n",
                ns, rh.rmse, rh.ok ? "ok" : "FAIL", rr.rmse, rr.ok ? "ok" : "FAIL")
    end
    write_csv(joinpath(STUDY_OUT, "time_houbolt_rodas_ex2.csv"),
              vcat([(method="houbolt", r...) for r in rows_h],
                   [(method="rodas", r...) for r in rows_r]))

    plt = plot(xlabel=L"\Delta t", ylabel="RMSE",
               xscale=:log10, yscale=:log10, legend=:topleft,
               xlims=(1.5e-3, 2.0), ylims=(3e-4, 6.0))
    plot!(plt, [r.dt for r in rows_h], [r.rmse for r in rows_h];
          color=:black, marker=:circle, label="Houbolt")
    plot!(plt, [r.dt for r in rows_r], [r.rmse for r in rows_r];
          color=:crimson, marker=:diamond, label="Rodas5P")
    savefig(plt, joinpath(FIG, "ex2_time_houbolt_rodas.pdf"))
    savefig(plt, joinpath(FIG, "ex2_time_houbolt_rodas.png"))
    println("wrote ex2_time_houbolt_rodas")

    # Snapshot at 5 steps: exact / Houbolt / Rodas
    dad = make_dad(2; nb=NB, nint=NINT)
    par = case_params(2)
    u0 = fill(par.u0, dad.nt)
    @inbounds for i in 1:dad.n
        dad.BC[i] == 0 && (u0[i] = dad.BV[i])
    end
    solh = bem_dibem(dad, u0; κ=par.κ, Δt=par.tf / NS_SNAP, tf=par.tf,
                     scheme=:houbolt)
    Uh = solh.U[1:dad.n + dad.ni, end]
    pts = vcat(dad.Nodes, dad.internalNodes)
    uex = exact_vec(dad, 2, par.tf)
    cl = (0.0, maximum(uex))
    p1 = field_panel(pts, uex, "exact"; clims=cl)
    p2 = field_panel(pts, Uh, "Houbolt, 5 steps"; clims=cl)
    p3 = field_panel(pts, snap.U, "Rodas5P, 5 steps"; clims=cl)
    lay = plot(p1, p2, p3; layout=(1, 3), size=(1080, 360),
               plot_title="Example 2, " * L"n_b=n_{\mathrm{int}}=16" * ", " * L"t_f")
    savefig(lay, joinpath(FIG, "ex2_time_fields.pdf"))
    savefig(lay, joinpath(FIG, "ex2_time_fields.png"))
    println("wrote ex2_time_fields")
end

if endswith(replace(PROGRAM_FILE, "\\" => "/"), "sbm_time_compare.jl")
    main()
end
