using DrWatson
@quickactivate :BEM
using BEM.Topology
using Plots
Plots.gr()
mkpath(datadir("topology"))

for id in 1:4
    println("\n========== Pacheco problem $id ==========")
    d, opt = pacheco_problem(id; ne=12, nint=12, degree=1)
    opt.maxiter = 12
    opt.verbose = true
    opt.nucleate_every = 4          # later holes allowed, not every iter
    A0 = design_area(d)
    println("A0=", A0, " Ap=", (1 - opt.ΔA) * A0, " ΔA=", opt.ΔA)
    d, dad, hist = solve_topology!(d, opt)
    println("final A=", round(design_area(d); digits=4),
        "  Ap=", round((1 - opt.ΔA) * A0; digits=4),
        "  holes=", n_holes(d),
        "  J/J0=", round(hist.J[end] / hist.J[1]; digits=3))
    plt = plot(; aspect_ratio=1, legend=false, title="Pacheco p$id",
        xlim=(-0.08, 1.08), ylim=(-0.08, 1.08), size=(480, 480))
    for (k, segs) in enumerate(d.loops)
        v = loop_vertices(segs)
        isempty(v) && continue
        xs = [p[1] for p in v]; push!(xs, v[1][1])
        ys = [p[2] for p in v]; push!(ys, v[1][2])
        plot!(plt, xs, ys; lw=2, color=k == 1 ? :black : :steelblue)
    end
    savefig(plt, datadir("topology", "pacheco_p$(id).png"))
end
println("done")
