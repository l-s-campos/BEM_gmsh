# Thesis Fig. 7.10: Ux(ω), Uy(ω) on the 90° ring.
# Geometry is Ri=0.6 m, Ro=0.9 m (thickness 0.3 m) — Fig. 7.8 ω-stations
# 94.25 / 124.25 / 265.62 / 295.62 cm. FEniCS anel_aniso.py uses the same.
#
#   julia --project=. scripts/debug/p3_fig710.jl
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf, StaticArrays
using Plots, LaTeXStrings
gr()
default(fontfamily="Computer Modern", linewidth=1.1, framestyle=:box,
    grid=false, dpi=180, legendfontsize=8, tickfontsize=10, guidefontsize=12,
    titlefontsize=13)

include(datadir("Laplace", "Laplace_dad.jl"))

const Ex, Ey, Gxy = 124.04e3, 10.09e3, 6.03e3   # MPa
const νyx, ηx, ηy = 0.344, 1.255, -0.031
const νzx, νzy, ηz = 0.40, 0.25, 0.50
const P = 1000.0                                 # MPa = 1 GPa
const Ri, Ro = 600.0, 900.0                      # mm = 0.6 m, 0.9 m
const OUT = joinpath(@__DIR__, "p3_fig710.png")

function p3_mesh()
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("anel_aniso")
    lc = 80.0
    c = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, lc)
    p_iy = gmsh.model.geo.addPoint(0.0, Ri, 0.0, lc)
    p_ix = gmsh.model.geo.addPoint(Ri, 0.0, 0.0, lc)
    p_ox = gmsh.model.geo.addPoint(Ro, 0.0, 0.0, lc)
    p_oy = gmsh.model.geo.addPoint(0.0, Ro, 0.0, lc)
    inner = gmsh.model.geo.addCircleArc(p_iy, c, p_ix)
    bottom = gmsh.model.geo.addLine(p_ix, p_ox)
    outer = gmsh.model.geo.addCircleArc(p_ox, c, p_oy)
    left = gmsh.model.geo.addLine(p_oy, p_iy)
    cl = gmsh.model.geo.addCurveLoop([inner, bottom, outer, left])
    s1 = gmsh.model.geo.addPlaneSurface([cl])
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.setTransfiniteCurve(inner, 11)
    gmsh.model.mesh.setTransfiniteCurve(bottom, 3)
    gmsh.model.mesh.setTransfiniteCurve(outer, 11)
    gmsh.model.mesh.setTransfiniteCurve(left, 3)
    gmsh.model.addPhysicalGroup(1, [left], -1, "0;0;0;0")
    gmsh.model.addPhysicalGroup(1, [bottom], -1, "1;0;1;-$P")
    gmsh.model.addPhysicalGroup(1, [inner, outer], -1, "1;0;1;0")
    gmsh.model.addPhysicalGroup(2, [s1], -1, "Domain")
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(2)
    out = datadir("elastico", "anel_aniso_06_09.msh")
    mkpath(dirname(out))
    gmsh.write(out)
    gmsh.finalize()
    return out
end

function solve_case(msh, props; bie=:cbie)
    dad = format2d(msh, props; tipo=2, pontointerno=false)
    if bie === :hbie
        H_G_hyper(dad; npg=50, threaded=false)
    else
        assemble!(dad; npg=16, threaded=false)
    end
    solve(dad)
    return dad
end

"""ω [cm] along Fig. 7.8: inner (0,Ri)→(Ri,0) → bottom → outer → left."""
function path_omega(dad)
    n = dad.n
    xy = [(dad.Nodes[i][1], dad.Nodes[i][2]) for i in 1:n]
    r = [hypot(x, y) for (x, y) in xy]
    ang = [atan(y, x) for (x, y) in xy]
    left = findall(i -> xy[i][1] < 1.0, 1:n)
    bot  = findall(i -> xy[i][2] < 1.0 && xy[i][1] > Ri - 1, 1:n)
    used = union(left, bot)
    inn  = findall(i -> r[i] < (Ri + Ro) / 2 && i ∉ used, 1:n)
    out  = findall(i -> r[i] ≥ (Ri + Ro) / 2 && i ∉ used, 1:n)
    si = sort(inn; by=i -> -ang[i])          # π/2 → 0
    sb = sort(bot; by=i -> xy[i][1])         # Ri → Ro
    so = sort(out; by=i -> ang[i])           # 0 → π/2
    sl = sort(left; by=i -> -xy[i][2])       # Ro → Ri
    order = vcat(si, sb, so, sl)
    ω = zeros(n)
    acc = 0.0
    ω[order[1]] = 0.0
    for k in 2:length(order)
        i, j = order[k - 1], order[k]
        acc += hypot(xy[j][1] - xy[i][1], xy[j][2] - xy[i][2])
        ω[j] = acc
    end
    return ω ./ 10, order
end

function series(dad)
    ω, order = path_omega(dad)
    ux = [dad.u[2i - 1] / 10 for i in order]
    uy = [dad.u[2i] / 10 for i in order]
    return ω[order], ux, uy
end

ept = AnisotropicElasticity(lekhnitskii_engineering(Ex, Ey, Gxy, νyx;
    η12_1=ηx, η12_2=ηy))
epd = AnisotropicElasticity(lekhnitskii_engineering(Ex, Ey, Gxy, νyx;
    η12_1=ηx, η12_2=ηy, plane_strain=true, E3=Ey,
    ν31=νzx, ν32=νzy, η12_3=ηz))

msh = p3_mesh()
println("mesh ", msh)
s_ept = series(solve_case(msh, ept; bie=:cbie))
h_ept = series(solve_case(msh, ept; bie=:hbie))
s_epd = series(solve_case(msh, epd; bie=:cbie))
h_epd = series(solve_case(msh, epd; bie=:hbie))
@printf("EPT CBIE  Ux=%.1f  Uy=%.1f cm   ωmax=%.1f cm\n",
    extrema(s_ept[2])[1], extrema(s_ept[3])[1], maximum(s_ept[1]))
@printf("EPT HBIE  Ux=%.1f  Uy=%.1f cm\n", extrema(h_ept[2])[1], extrema(h_ept[3])[1])
@printf("EPD CBIE  Ux=%.1f  Uy=%.1f cm\n", extrema(s_epd[2])[1], extrema(s_epd[3])[1])
@printf("EPD HBIE  Ux=%.1f  Uy=%.1f cm\n", extrema(h_epd[2])[1], extrema(h_epd[3])[1])

kw_line = (lw=0.8, alpha=0.9)
px = plot(; xlabel=L"\omega\ \mathrm{(cm)}", ylabel=L"U_x\ \mathrm{(cm)}",
    title="Deslocamento X", xlims=(0, 300), ylims=(-80, 10),
    xticks=0:50:300, yticks=-80:10:10, legend=:bottomright)
py = plot(; xlabel=L"\omega\ \mathrm{(cm)}", ylabel=L"U_y\ \mathrm{(cm)}",
    title="Deslocamento Y", xlims=(0, 300), ylims=(-80, 0),
    xticks=0:50:300, yticks=-80:10:0, legend=:bottomright)

function add!(ω, ux, uy; m, mc, ms, lab)
    plot!(px, ω, ux; color=:black, kw_line...,
        marker=m, markercolor=mc, markersize=ms, markerstrokecolor=:black,
        markerstrokewidth=0.6, label=lab)
    plot!(py, ω, uy; color=:black, kw_line...,
        marker=m, markercolor=mc, markersize=ms, markerstrokecolor=:black,
        markerstrokewidth=0.6, label=lab)
end
add!(s_ept...; m=:square, mc=:white, ms=5, lab="MEC S EPT")
add!(h_ept...; m=:square, mc=:black, ms=5, lab="MEC HS EPT")
add!(s_epd...; m=:utriangle, mc=:white, ms=6, lab="MEC S EPD")
add!(h_epd...; m=:circle, mc=:gray, ms=5, lab="MEC HS EPD")
for x in (94.25, 124.25, 265.62)
    vline!(px, [x]; color=:gray, lw=0.5, ls=:dot, label="")
    vline!(py, [x]; color=:gray, lw=0.5, ls=:dot, label="")
end

plt = plot(px, py; layout=(1, 2), size=(1100, 520),
    plot_title=L"90° ring $R_i=0.6\,\mathrm{m}$, $R_o=0.9\,\mathrm{m}$, $t_y=-1\,\mathrm{GPa}$")
savefig(plt, OUT)
println("wrote ", OUT)
