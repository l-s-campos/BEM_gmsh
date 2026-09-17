using DrWatson
@quickactivate :BEM
using BEM.Topology
using Plots
Plots.gr()
mkpath(datadir("topology"))

d, opt = pacheco_problem(1; ne=10, nint=10, degree=1)
ls = LevelSetOptions(; ΔA=opt.ΔA, method=:hj, maxiter=10, ngrid=40,
    verbose=true, nel_per_loop=20, n_hj=8, n_reinit=3, volume_step=0.2)
println("A0=", design_area(d), " Ap=", (1 - ls.ΔA) * design_area(d))
d, dad, hist, g = solve_levelset!(d, ls)
println("final A=", round(design_area(d); digits=3),
    " holes=", n_holes(d),
    " J/J0=", round(hist.J[end] / max(hist.J[1], 1e-16); digits=3),
    " area_φ=", round(BEM._area_from_phi(g); digits=3))
plt = plot(; aspect_ratio=1, legend=false, title="LSM HJ p1",
    xlim=(-0.08, 1.08), ylim=(-0.08, 1.08), size=(480, 480))
for (k, segs) in enumerate(d.loops)
    v = loop_vertices(segs)
    isempty(v) && continue
    xs = [p[1] for p in v]; push!(xs, v[1][1])
    ys = [p[2] for p in v]; push!(ys, v[1][2])
    plot!(plt, xs, ys; lw=2, color=k == 1 ? :black : :steelblue)
end
savefig(plt, datadir("topology", "lsm_hj_p1.png"))
