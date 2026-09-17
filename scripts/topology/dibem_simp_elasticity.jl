# Coelho 2021 plane-stress cases: Pacheco DT motion vs DIBEM-SIMP + iso-cut.
#
#   julia --project=. scripts/topology/dibem_simp_elasticity.jl
#   julia --project=. scripts/topology/dibem_simp_elasticity.jl --ids=3 --maxiter=8

using DrWatson
@quickactivate :BEM
using BEM.Topology
using Printf
using Statistics
using Dates
using Plots
Plots.gr()

const OUT = datadir("topology", "elasticity")
mkpath(OUT)

ids = [3, 4, 5]
maxiter = 36
ne = 8
nint = 10
degree = 1
do_plot = false
n_simp = 30
area_rtol = 0.08

for a in ARGS
    if startswith(a, "--ids=")
        global ids = parse.(Int, split(split(a, "=")[2], ","; keepempty=false))
    elseif startswith(a, "--maxiter=")
        global maxiter = parse(Int, split(a, "=")[2])
    elseif startswith(a, "--nsimp=")
        global n_simp = parse(Int, split(a, "=")[2])
    elseif startswith(a, "--ne=")
        global ne = parse(Int, split(a, "=")[2])
    elseif a == "--plot"
        global do_plot = true
    end
end

const NAMES = Dict(3 => "cantilever", 4 => "cc-beam", 5 => "ss-beam")

function _plot_design(d::TopologyDesign; title="", path=nothing)
    outer = loop_vertices(d.loops[1])
    xmin = minimum(p[1] for p in outer)
    xmax = maximum(p[1] for p in outer)
    ymin = minimum(p[2] for p in outer)
    ymax = maximum(p[2] for p in outer)
    pad = 0.05 * max(xmax - xmin, ymax - ymin, 1e-8)
    plt = plot(; aspect_ratio=1, legend=false, title=title,
        xlim=(xmin - pad, xmax + pad), ylim=(ymin - pad, ymax + pad), size=(640, 420))
    for (k, segs) in enumerate(d.loops)
        v = loop_vertices(segs)
        isempty(v) && continue
        xs = [p[1] for p in v]; push!(xs, v[1][1])
        ys = [p[2] for p in v]; push!(ys, v[1][2])
        plot!(plt, xs, ys; lw=k == 1 ? 2 : 1.5, color=k == 1 ? :black : :steelblue)
        for s in segs
            BEM.Topology._is_fixed_segment(s) || continue
            plot!(plt, [s.verts[1][1], s.verts[end][1]],
                      [s.verts[1][2], s.verts[end][2]];
                      lw=3, color=:crimson)
        end
    end
    path !== nothing && savefig(plt, path)
    return plt
end

function _J0(d)
    dad = bemdata_from_loops(d)
    H_G_full_direct(dad; npg=10, threaded=false)
    solve(dad)
    return elastic_compliance(dad)
end

println("="^72)
println(" Coelho 2021 plane-stress: TD motion vs SIMP-ρ vs DT-ρ (volume-matched to Ap)")
println(" ne=$ne  nint=$nint  degree=$degree  n_simp=$n_simp  Pacheco maxiter=$maxiter  area_rtol=$area_rtol")
println(" J = compliance (lower is better) at A ≈ Ap")
println("="^72)

@printf("\n%-12s %10s %8s %8s %8s %8s %7s %7s\n",
    "problem", "method", "A", "Ap", "J", "J/J0", "holes", "s")
@printf("%s\n", "-"^72)

for id in ids
    name = NAMES[id]
    dP, optP = coelho_problem(id; ne=ne, nint=nint, degree=degree)
    Ap = (1 - optP.ΔA) * design_area(dP)
    J0 = _J0(copy_design(dP))

    optP.maxiter = maxiter
    optP.verbose = false
    optP.nucleate_every = 2
    optP.area_rtol = area_rtol
    t0 = time()
    d1, dad1, JP, AP = match_volume!(copy_design(dP), Ap; opt=optP, rtol=area_rtol)
    tP = time() - t0
    hP = n_holes(d1)
    @printf("%-12s %10s %8.3f %8.3f %8.4e %8.3f %7d %7.1f\n",
        name, "Pacheco", AP, Ap, JP, JP / J0, hP, tP)
    do_plot && _plot_design(d1; title="Pacheco $name",
        path=joinpath(OUT, "cmp_pacheco_c$id.png"))

    optS = DibemSimpOptions(;
        volfrac=1 - optP.ΔA, n_simp=n_simp, rmin=0.22, ρ_cut=0.4, β_end=8.0,
        ngrid=61, cut=true, match_area=true, pacheco=true, verbose=false, npg=10,
        min_hole_dist=0.002, min_hole_area=2e-4,
        pacheco_opt=PachecoOptions(maxiter=maxiter, verbose=false,
            nucleate_first=false, nucleate_every=typemax(Int),
            area_rtol=area_rtol, vmax=optP.vmax, min_hole_dist=0.002, min_hole_area=2e-4),
    )
    t0 = time()
    d2, dad2, ρ, histS = solve_dibem_simp!(copy_design(dP), optS)
    tS = time() - t0
    JS = elastic_compliance(dad2)
    AS = design_area(d2)
    hS = n_holes(d2)
    gray = isempty(histS) ? NaN : histS[end].gray
    @printf("%-12s %10s %8.3f %8.3f %8.4e %8.3f %7d %7.1f\n",
        name, "SIMP-ρ", AS, Ap, JS, JS / J0, hS, tS)
    @printf("%12s %10s  gray=%.3f  SIMP-C=%.4e (heterogeneous, before cut)\n",
        "", "", gray, histS[end].C)
    do_plot && _plot_design(d2; title="SIMP-ρ $name",
        path=joinpath(OUT, "cmp_dibemsimp_c$id.png"))

    optD = DibemSimpOptions(;
        method=:dt, volfrac=1 - optP.ΔA, n_simp=1, rmin=0.22, ρ_cut=0.4, β_end=8.0,
        ngrid=61, cut=true, match_area=true, pacheco=true, verbose=false, npg=10,
        min_hole_dist=0.002, min_hole_area=2e-4,
        pacheco_opt=PachecoOptions(maxiter=maxiter, verbose=false,
            nucleate_first=false, nucleate_every=typemax(Int),
            area_rtol=area_rtol, vmax=optP.vmax, min_hole_dist=0.002, min_hole_area=2e-4),
    )
    t0 = time()
    d3, dad3, ρD, histD = solve_dt_density!(copy_design(dP), optD)
    tD = time() - t0
    JD = elastic_compliance(dad3)
    AD = design_area(d3)
    hD = n_holes(d3)
    grayD = isempty(histD) ? NaN : histD[end].gray
    @printf("%-12s %10s %8.3f %8.3f %8.4e %8.3f %7d %7.1f\n",
        name, "DT-ρ", AD, Ap, JD, JD / J0, hD, tD)
    @printf("%12s %10s  gray=%.3f  C0=%.4e (homogeneous DT, before cut)\n",
        "", "", grayD, histD[end].C)
    do_plot && _plot_design(d3; title="DT-ρ $name",
        path=joinpath(OUT, "cmp_dtrho_c$id.png"))
end
println("done  ", string(now()))
println("plots in ", OUT)
