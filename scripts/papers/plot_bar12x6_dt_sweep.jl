# 12×6 bar, PHS1 + linear, Laplace vs Kelvin. Houbolt Δt = 1, 1/2, 1/4, …
# until the first blow-up. One figure: rows = Δt, columns = physics.
#
#   julia --project=. scripts/plot_bar12x6_dt_sweep.jl
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf
using Plots, LaTeXStrings
gr()
default(fontfamily="Computer Modern", linewidth=1.4, framestyle=:box,
    grid=false, dpi=160, legendfontsize=8)

include(datadir("Laplace", "Laplace_dad.jl"))
include(datadir("Laplace", "potencial_problems.jl"))
include(datadir("Laplace", "wave_propagation.jl"))
include(datadir("elastico", "iso", "bar_sudden.jl"))

const Lx, Ly = 12.0, 6.0
const tf = 4 * 4 * Lx          # 4 periods
const PROBE = Point2D(Lx, Ly / 2)
const NPG = 12
const RBF = PHS(1; poly_deg=1)  # φ=r + linear (1,x,y)
const MASSES = (:drm, :dibem, :cells)
const DTS = (1.0, 0.5, 0.25, 0.1, 0.05, 0.025)
const BLOW = 80.0

function _rect_mesh(bc_bot, bc_right, bc_top, bc_left; subdir, nome, ndivx=13, ndivy=7)
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(nome)
    p1 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, 1.0)
    p2 = gmsh.model.geo.addPoint(Lx, 0.0, 0.0, 1.0)
    p3 = gmsh.model.geo.addPoint(Lx, Ly, 0.0, 1.0)
    p4 = gmsh.model.geo.addPoint(0.0, Ly, 0.0, 1.0)
    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p4)
    l4 = gmsh.model.geo.addLine(p4, p1)
    cl = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    s = gmsh.model.geo.addPlaneSurface([cl])
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.setTransfiniteCurve(l1, ndivx)
    gmsh.model.mesh.setTransfiniteCurve(l3, ndivx)
    gmsh.model.mesh.setTransfiniteCurve(l2, ndivy)
    gmsh.model.mesh.setTransfiniteCurve(l4, ndivy)
    gmsh.model.mesh.setTransfiniteSurface(s)
    gmsh.model.mesh.setRecombine(2, s)
    gmsh.model.addPhysicalGroup(1, [l1], -1, bc_bot)
    gmsh.model.addPhysicalGroup(1, [l2], -1, bc_right)
    gmsh.model.addPhysicalGroup(1, [l3], -1, bc_top)
    gmsh.model.addPhysicalGroup(1, [l4], -1, bc_left)
    gmsh.model.addPhysicalGroup(2, [s], -1, "Domain")
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(1)
    out = datadir(subdir, nome * ".msh")
    mkpath(dirname(out))
    gmsh.write(out)
    gmsh.finalize()
    return out
end

function make_laplace()
    msh = _rect_mesh("1;0", "1;-1", "1;0", "0;0"; subdir="Laplace", nome="sw_lap12")
    dad = format2d(msh, Laplace(1.0); tipo=1, pontointerno=true)
    attach_analytical!(dad, ana_bar_sudden(; N=400, c=1.0, L=Lx))
    return dad
end

function make_elastic()
    msh = _rect_mesh("1;0;1;0", "1;1;1;0", "1;0;1;0", "0;0;0;0";
        subdir="elastico", nome="sw_el12")
    dad = format2d(msh, Elasticity(E=1.0, nu=0.0, rho=1.0; plane_stress=true);
        tipo=1, pontointerno=true)
    attach_analytical!(dad, ana_bar_sudden(; N=400, c=1.0, L=Lx))
    return dad
end

function assemble_mass!(dad, mass)
    if mass === :drm
        if dad.properties isa Laplace
            build_drm_matrices(dad, RBF; npg=NPG)
        else
            build_drm_matrices(dad; npg=NPG, kernel=:r, poly_deg=1)
        end
    elseif mass === :dibem
        DIBEM(dad; method=:dense, rbf=RBF)
    else
        build_cell_mass(dad; npg=NPG)
    end
    return dad
end

function probe(dad)
    pts = all_points(dad)
    ip = argmin(norm(p - PROBE) for p in pts)
    if dad.properties isa Laplace
        return dad.T[ip, :]
    end
    return dad.u[2 * (ip - 1) + 1, :]
end

function clip_blow(ux)
    v = copy(ux)
    i0 = findfirst(i -> !isfinite(v[i]) || abs(v[i]) > BLOW, eachindex(v))
    i0 !== nothing && (v[i0:end] .= NaN)
    return v, i0
end

rel(a, b) = norm(a .- b) / (norm(b) + 1e-30)

const COLS = Dict(:drm => :darkorange, :dibem => :steelblue, :cells => :seagreen)
const LABS = Dict(:drm => "DRM", :dibem => "DIBEM", :cells => "cells")

println(" PHS1 + linear  (1, x, y)   12×6 bar   tf=$tf")
println(" lower Δt from 1 until the first blow-up")

prepared = Dict{Symbol,Dict{Symbol,Any}}()
for (tag, maker) in ((:laplace, make_laplace), (:elastic, make_elastic))
    dad0 = maker()
    H_G_full_direct(dad0; npg=NPG, threaded=true)
    @printf("  %-8s  n=%d  ni=%d  nt=%d\n", tag, dad0.n, dad0.ni, dad0.nt)
    ms = Dict{Symbol,Any}()
    for mass in MASSES
        d = deepcopy(dad0)
        assemble_mass!(d, mass)
        ms[mass] = d
    end
    prepared[tag] = ms
end
ana = ana_bar_sudden(; N=400, c=1.0, L=Lx)

rows = []   # (; Δt, panels=Dict(tag => Dict(mass => ux)), ua, t, blown)
for Δt in DTS
    println()
    @printf("── Δt = %g  nT = %d ──\n", Δt, length(0:Δt:tf))
    @printf("  %-8s %-8s  %10s  %10s  %s\n", "phys", "mass", "err", "max|u|", "blow")
    panels = Dict{Symbol,Dict{Symbol,Vector{Float64}}}()
    tref = nothing
    ua = nothing
    anyblow = false
    for tag in (:laplace, :elastic)
        panels[tag] = Dict{Symbol,Vector{Float64}}()
        for mass in MASSES
            dad = deepcopy(prepared[tag][mass])
            solve_Houbolt(dad, Δt, tf)
            ux = probe(dad)
            t = dad.time
            tref === nothing && (tref = t)
            ua === nothing && (ua = [float(ana.u(PROBE; t=ti)) for ti in t])
            uxc, i0 = clip_blow(ux)
            ok = all(isfinite, ux)
            blew = i0 !== nothing || !ok
            anyblow |= blew
            err = blew ? NaN : rel(ux, ua)
            mx = maximum(abs, filter(isfinite, ux); init=0.0)
            blow = i0 === nothing ? "—" : @sprintf("t=%.1f", t[i0])
            @printf("  %-8s %-8s  %10.3e  %10.3e  %s\n", tag, mass, err, mx, blow)
            panels[tag][mass] = uxc
        end
    end
    push!(rows, (; Δt, panels, ua, t=tref, blown=anyblow))
    anyblow && break
end

nrows = length(rows)
ps = Plots.Plot[]
for (i, row) in enumerate(rows)
    for (j, tag) in enumerate((:laplace, :elastic))
        p = plot()
        showleg = i == 1 && j == 2
        plot!(p, row.t, row.ua; color=:black, ls=:dash,
            label=showleg ? "1D series" : "")
        for mass in MASSES
            plot!(p, row.t, row.panels[tag][mass]; color=COLS[mass],
                label=showleg ? LABS[mass] : "")
        end
        phys = tag === :laplace ? "Laplace" : "Kelvin"
        plot!(p; xlabel=(i == nrows ? L"t" : ""),
            ylabel=(j == 1 ? L"u(L,H/2)" : ""),
            title="$(phys)  Δt=$(row.Δt)",
            ylim=(-2, 28),
            legend=showleg ? :outertopright : false)
        push!(ps, p)
    end
end
plt = plot(ps...; layout=(nrows, 2), size=(1100, 300 * nrows),
    plot_title="12×6 bar  PHS1+linear  Houbolt  4 periods")

mkpath(joinpath(projectdir(), "plots"))
out = joinpath(projectdir(), "plots", "bar12x6_phs1lin_dt_sweep")
savefig(plt, out * ".png")
savefig(plt, out * ".pdf")
println("\nwrote ", out * ".png")
