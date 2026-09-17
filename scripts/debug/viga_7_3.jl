# Cordeiro 2015 §7.3 — non-homogeneous anisotropic beam (Fig. 7.14–7.17).
# Top: isotropic Kelvin-limit Lekhnitskii. Bottom: Vanalli laminate (already
# rotated 30°). Perfect interface, EPT and EPD, CBIE and HBIE.
#
#   julia --project=. scripts/debug/viga_7_3.jl
using DrWatson
@quickactivate :BEM
using BEM.MultiRegion
using LinearAlgebra, Printf, Statistics, StaticArrays
using Plots, LaTeXStrings
gr()
default(fontfamily="Computer Modern", linewidth=1.1, framestyle=:box,
    grid=false, dpi=180, legendfontsize=8, tickfontsize=10, guidefontsize=12)

include(datadir("Laplace", "Laplace_dad.jl"))

const Lx, Hy = 3000.0, 1000.0          # mm (3 m × 1 m + 1 m)
const Ptop, Pright = 100.0, 10.0       # MPa
const Eiso, νiso = 25e3, 0.25
const Ex, Ey, Gxy = 19.681e3, 11.248e3, 7.933e3
const νyx, ηx, ηy = 0.529, -1.224, -0.042
const νzy, νzx, ηz = 0.30, 0.15, 0.75
const OUT = joinpath(@__DIR__, "viga_7_3_fig717.png")

function mesh_viga()
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("viga_73")
    lc = 200.0
    p00 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, lc)
    pL0 = gmsh.model.geo.addPoint(Lx, 0.0, 0.0, lc)
    pL1 = gmsh.model.geo.addPoint(Lx, Hy, 0.0, lc)
    p01 = gmsh.model.geo.addPoint(0.0, Hy, 0.0, lc)
    pL2 = gmsh.model.geo.addPoint(Lx, 2Hy, 0.0, lc)
    p02 = gmsh.model.geo.addPoint(0.0, 2Hy, 0.0, lc)
    # bottom region CCW
    b_bot = gmsh.model.geo.addLine(p00, pL0)
    b_r   = gmsh.model.geo.addLine(pL0, pL1)
    b_if  = gmsh.model.geo.addLine(pL1, p01)
    b_l   = gmsh.model.geo.addLine(p01, p00)
    # top region CCW
    t_if  = gmsh.model.geo.addLine(p01, pL1)
    t_r   = gmsh.model.geo.addLine(pL1, pL2)
    t_top = gmsh.model.geo.addLine(pL2, p02)
    t_l   = gmsh.model.geo.addLine(p02, p01)
    clb = gmsh.model.geo.addCurveLoop([b_bot, b_r, b_if, b_l])
    clt = gmsh.model.geo.addCurveLoop([t_if, t_r, t_top, t_l])
    sb = gmsh.model.geo.addPlaneSurface([clb])
    st = gmsh.model.geo.addPlaneSurface([clt])
    gmsh.model.geo.synchronize()
    for (ℓ, n) in ((b_bot, 5), (b_if, 5), (t_if, 5), (t_top, 5),
                   (b_r, 3), (b_l, 3), (t_r, 3), (t_l, 3))
        gmsh.model.mesh.setTransfiniteCurve(ℓ, n)
    end
    gmsh.model.mesh.setTransfiniteSurface(sb)
    gmsh.model.mesh.setTransfiniteSurface(st)
    gmsh.model.addPhysicalGroup(1, [b_l, t_l], -1, "0;0;0;0")             # clamp
    gmsh.model.addPhysicalGroup(1, [t_top], -1, "1;0;1;-$Ptop")           # ty = −100
    gmsh.model.addPhysicalGroup(1, [t_r], -1, "1;-$Pright;1;0")           # tx = −10
    gmsh.model.addPhysicalGroup(1, [b_bot, b_r], -1, "1;0;1;0")
    gmsh.model.addPhysicalGroup(1, [b_if, t_if], -1, "3;0;3;0")
    gmsh.model.addPhysicalGroup(2, [sb], -1, "Bottom")
    gmsh.model.addPhysicalGroup(2, [st], -1, "Top")
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(2)
    out = datadir("elastico", "viga_7_3.msh")
    mkpath(dirname(out))
    gmsh.write(out)
    gmsh.finalize()
    return out
end

function split_beam(msh, props_bot, props_top)
    dad = format2d(msh, props_bot; tipo=2, pontointerno=false)
    bot_e = Element[]; top_e = Element[]
    for e in dad.elements
        c = mean(dad.Nodes[e.index])
        n0 = mean(dad.Normal[e.index])
        is_if = any(dad.BC[2 * i - 1] == 3 || dad.BC[2 * i] == 3 for i in e.index)
        if is_if
            if n0[2] >= 0
                push!(bot_e, e)
            else
                push!(top_e, e)
            end
        elseif c[2] < Hy - 1e-6
            push!(bot_e, e)
        else
            push!(top_e, e)
        end
    end
    function subset(elems, name, props)
        old = sort(unique(vcat([e.index for e in elems]...)))
        map_n = Dict(old[i] => i for i in eachindex(old))
        Nodes = dad.Nodes[old]
        Normal = dad.Normal[old]
        BC = zeros(Int, 2 * length(old))
        BV = zeros(2 * length(old))
        for (k, i) in enumerate(old)
            BC[2k-1] = dad.BC[2i-1]; BC[2k] = dad.BC[2i]
            BV[2k-1] = dad.BV[2i-1]; BV[2k] = dad.BV[2i]
        end
        new_elems = [Element([map_n[i] for i in e.index], e.Jacobian, e.Length, e.Region)
                     for e in elems]
        n = length(Nodes)
        return BEMdata(name, 2, new_elems, dad.element_type, dad.elem_weight,
            Nodes, Normal, props, BC, BV, n, 0, n, BEMCache())
    end
    return MultiRegionProblem([subset(bot_e, "bot", props_bot),
                               subset(top_e, "top", props_top)]; name="viga_7_3")
end

function props_iso(; epd=false)
    G = Eiso / (2 * (1 + νiso))
    AnisotropicElasticity(lekhnitskii_engineering(Eiso, Eiso, G, νiso;
        plane_strain=epd, E3=Eiso, ν31=νiso, ν32=νiso))
end
function props_lam(; epd=false)
    AnisotropicElasticity(lekhnitskii_engineering(Ex, Ey, Gxy, νyx;
        η12_1=ηx, η12_2=ηy, plane_strain=epd, E3=Ey,
        ν31=νzx, ν32=νzy, η12_3=ηz))
end

"""Outer-contour ω [cm]: bottom → right → top → left (Fig. 7.15)."""
function contour_series(prob)
    pts = Tuple{Float64,Float64,Float64,Float64}[]  # x,y,ux,uy
    for dad in prob.regions
        for i in 1:dad.n
            if dad.BC[2i-1] == 3 || dad.BC[2i] == 3
                continue
            end
            push!(pts, (dad.Nodes[i][1], dad.Nodes[i][2],
                dad.u[2i-1], dad.u[2i]))
        end
    end
    function on_bot(p); p[2] < 1.0 && p[1] > 1.0; end
    function on_right(p); p[1] > Lx - 1.0; end
    function on_top(p); p[2] > 2Hy - 1.0; end
    function on_left(p); p[1] < 1.0; end
    bot = sort(filter(on_bot, pts); by=p -> p[1])
    rgt = sort(filter(p -> on_right(p) && !on_bot(p) && !on_top(p), pts); by=p -> p[2])
    top = sort(filter(on_top, pts); by=p -> -p[1])
    lft = sort(filter(p -> on_left(p) && !on_bot(p) && !on_top(p), pts); by=p -> -p[2])
    seq = vcat(bot, rgt, top, lft)
    ω = zeros(length(seq))
    for k in 2:length(seq)
        ω[k] = ω[k-1] + hypot(seq[k][1]-seq[k-1][1], seq[k][2]-seq[k-1][2])
    end
    return ω ./ 10, [p[3]/10 for p in seq], [p[4]/10 for p in seq]
end

function run_case(epd, bie)
    msh = datadir("elastico", "viga_7_3.msh")
    isfile(msh) || mesh_viga()
    prob = split_beam(msh, props_lam(; epd=epd), props_iso(; epd=epd))
    pair_interfaces!(prob; tol=5.0)
    assemble_multiregion(prob; npg=bie === :hbie ? 50 : 16, bie=bie, threaded=false)
    solve_multiregion!(prob)
    ω, ux, uy = contour_series(prob)
    @printf("  %s %s  n_if=%d  Ux=[%.2f, %.2f]  Uy=[%.2f, %.2f] cm  ωmax=%.1f\n",
        epd ? "EPD" : "EPT", bie === :hbie ? "HBIE" : "CBIE",
        length(prob.interfaces), extrema(ux)..., extrema(uy)..., maximum(ω))
    return ω, ux, uy, prob
end

println("mesh …")
mesh_viga()
println("EPT/EPD  CBIE/HBIE …")
s_ept = run_case(false, :cbie)
h_ept = run_case(false, :hbie)
s_epd = run_case(true, :cbie)
h_epd = run_case(true, :hbie)

kw = (lw=0.8,)
px = plot(; xlabel=L"\omega\ \mathrm{(cm)}", ylabel=L"U_x\ \mathrm{(cm)}",
    title="Deslocamento X", xlims=(0, 1000), ylims=(-2, 4),
    xticks=0:200:1000, legend=:topleft)
py = plot(; xlabel=L"\omega\ \mathrm{(cm)}", ylabel=L"U_y\ \mathrm{(cm)}",
    title="Deslocamento Y", xlims=(0, 1000), ylims=(-8, 0),
    xticks=0:200:1000, legend=:bottomleft)
function add!(ω, ux, uy; m, mc, ms, lab)
    plot!(px, ω, ux; color=:black, kw..., marker=m, markercolor=mc, markersize=ms,
        markerstrokecolor=:black, markerstrokewidth=0.5, label=lab)
    plot!(py, ω, uy; color=:black, kw..., marker=m, markercolor=mc, markersize=ms,
        markerstrokecolor=:black, markerstrokewidth=0.5, label=lab)
end
add!(s_ept[1], s_ept[2], s_ept[3]; m=:square, mc=:white, ms=4, lab="MEC S EPT")
add!(h_ept[1], h_ept[2], h_ept[3]; m=:square, mc=:black, ms=4, lab="MEC HS EPT")
add!(s_epd[1], s_epd[2], s_epd[3]; m=:utriangle, mc=:white, ms=5, lab="MEC S EPD")
add!(h_epd[1], h_epd[2], h_epd[3]; m=:circle, mc=:gray, ms=4, lab="MEC HS EPD")
plt = plot(px, py; layout=(1, 2), size=(1100, 480),
    plot_title="§7.3 viga não homogênea anisotrópica")
savefig(plt, OUT)
println("wrote ", OUT)
