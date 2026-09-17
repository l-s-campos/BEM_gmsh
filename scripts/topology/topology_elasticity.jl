# Compare Pacheco node-motion vs Amstutz vs Hamilton–Jacobi on Coelho (2021)
# plane-stress elasticity cases 3–5 (cantilever, clamped–clamped, clamp+roller).
# Gmsh is used only if via_gmsh=true for the initial mesh.
#
#   julia --project=. scripts/topology_elasticity.jl
#   julia --project=. scripts/topology_elasticity.jl --ids=3 --maxiter=12

using DrWatson
@quickactivate :BEM
using BEM.Topology
using JSON
using Statistics
using Dates
using Plots
Plots.gr()

const OUT = datadir("topology", "elasticity")
mkpath(OUT)

ids = [3, 4, 5]
maxiter = 20
ne = 10
nint = 12
degree = 1
via_gmsh = false
do_plot = true
methods = [:pacheco, :amstutz, :hj]

for a in ARGS
    if startswith(a, "--ids=")
        ids = parse.(Int, split(split(a, "=")[2], ","; keepempty=false))
    elseif startswith(a, "--maxiter=")
        maxiter = parse(Int, split(a, "=")[2])
    elseif startswith(a, "--ne=")
        ne = parse(Int, split(a, "=")[2])
    elseif a == "--no-plot"
        do_plot = false
    elseif a == "--gmsh"
        via_gmsh = true
    end
end

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
            BEM._is_fixed_segment(s) || continue
            plot!(plt, [s.verts[1][1], s.verts[end][1]],
                      [s.verts[1][2], s.verts[end][2]];
                      lw=3, color=:crimson)
        end
    end
    path !== nothing && savefig(plt, path)
    return plt
end

function _hist_dict(hist::TopologyHistory)
    return Dict(
        "area" => hist.area,
        "J" => hist.J,
        "maxDT" => hist.maxDT,
        "n_holes" => hist.n_holes,
        "niter" => length(hist.area) - 1,
    )
end

results = Dict{String,Any}[]
for id in ids
    println("\n========== Coelho case $id ==========")
    row = Dict{String,Any}("id" => id, "ne" => ne, "maxiter" => maxiter)

    if :pacheco in methods
        d, opt = coelho_problem(id; ne=ne, nint=nint, degree=degree, via_gmsh=via_gmsh)
        opt.maxiter = maxiter
        opt.verbose = true
        opt.nucleate_every = 4
        A0 = design_area(d)
        t0 = time()
        d, dad, hist = solve_topology!(d, opt)
        row["pacheco"] = merge(_hist_dict(hist), Dict(
            "seconds" => time() - t0,
            "A0" => A0,
            "A_final" => design_area(d),
            "Ap" => (1 - opt.ΔA) * A0,
            "J_final" => hist.J[end],
            "J0" => hist.J[1],
        ))
        if do_plot
            _plot_design(d; title="Pacheco Coelho $id",
                path=joinpath(OUT, "pacheco_c$(id).png"))
        end
    end

    if :amstutz in methods
        d, optp = coelho_problem(id; ne=ne, nint=nint, degree=degree, via_gmsh=via_gmsh)
        ls = LevelSetOptions(; ΔA=optp.ΔA, method=:amstutz, maxiter=maxiter,
            ngrid=48, verbose=true, nel_per_loop=max(16, ne))
        A0 = design_area(d)
        t0 = time()
        d, dad, hist, _ = solve_levelset!(d, ls)
        row["amstutz"] = merge(_hist_dict(hist), Dict(
            "seconds" => time() - t0,
            "A0" => A0,
            "A_final" => design_area(d),
            "Ap" => (1 - ls.ΔA) * A0,
            "J_final" => hist.J[end],
            "J0" => hist.J[1],
        ))
        if do_plot
            _plot_design(d; title="Amstutz Coelho $id",
                path=joinpath(OUT, "amstutz_c$(id).png"))
        end
    end

    if :hj in methods
        d, optp = coelho_problem(id; ne=ne, nint=nint, degree=degree, via_gmsh=via_gmsh)
        ls = LevelSetOptions(; ΔA=optp.ΔA, method=:hj, maxiter=maxiter,
            ngrid=48, verbose=true, nel_per_loop=max(16, ne), n_hj=6, n_reinit=4)
        A0 = design_area(d)
        t0 = time()
        d, dad, hist, _ = solve_levelset!(d, ls)
        row["hj"] = merge(_hist_dict(hist), Dict(
            "seconds" => time() - t0,
            "A0" => A0,
            "A_final" => design_area(d),
            "Ap" => (1 - ls.ΔA) * A0,
            "J_final" => hist.J[end],
            "J0" => hist.J[1],
        ))
        if do_plot
            _plot_design(d; title="HJ Coelho $id",
                path=joinpath(OUT, "hj_c$(id).png"))
        end
    end

    push!(results, row)
end

outjson = joinpath(OUT, "compare.json")
open(outjson, "w") do io
    JSON.print(io, Dict("when" => string(now()), "ne" => ne, "maxiter" => maxiter,
        "results" => results); indent=2)
end
println("\nWrote ", outjson)
for row in results
    println("c$(row["id"]):")
    for m in ("pacheco", "amstutz", "hj")
        haskey(row, m) || continue
        r = row[m]
        println("  ", m, "  A=$(round(r["A_final"]; digits=3))/$(round(r["Ap"]; digits=3))",
            "  J/J0=$(round(r["J_final"] / max(r["J0"], 1e-16); digits=3))",
            "  holes=$(r["n_holes"][end])  iters=$(r["niter"])")
    end
end
