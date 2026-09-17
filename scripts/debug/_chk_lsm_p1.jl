using DrWatson
@quickactivate :BEM
using BEM.Topology
using Plots
Plots.gr()
mkpath(datadir("topology"))

function dump(d, title, path)
    plt = plot(; aspect_ratio=1, legend=false, title=title,
        xlim=(-0.08, 1.08), ylim=(-0.08, 1.08), size=(480, 480))
    for (k, segs) in enumerate(d.loops)
        v = loop_vertices(segs)
        isempty(v) && continue
        xs = [p[1] for p in v]; push!(xs, v[1][1])
        ys = [p[2] for p in v]; push!(ys, v[1][2])
        plot!(plt, xs, ys; lw=2, color=k == 1 ? :black : :steelblue)
    end
    savefig(plt, path)
end

for (meth, ngrid) in ((:amstutz, 40), (:hj, 40))
    println("\n========== LSM $meth p1 ==========")
    d, opt = pacheco_problem(1; ne=10, nint=10, degree=1)
    ls = LevelSetOptions(; ΔA=opt.ΔA, method=meth, maxiter=8, ngrid=ngrid,
        verbose=true, nel_per_loop=20, n_hj=5, n_reinit=4)
    d, dad, hist, _ = solve_levelset!(d, ls)
    println("final A=", round(design_area(d); digits=3),
        " holes=", n_holes(d),
        " J/J0=", round(hist.J[end] / hist.J[1]; digits=3))
    dump(d, "LSM $meth p1", datadir("topology", "lsm_$(meth)_p1.png"))
end
