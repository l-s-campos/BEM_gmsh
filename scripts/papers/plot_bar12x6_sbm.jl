# 12×6 bar: SBM–DRM (Laplace) vs BEM DRM/DIBEM/cells. Kelvin has no SBM.
# PHS1 + linear on BEM; SBM–DRM uses PHS1 (no poly in the (α,β) split).
# Houbolt Δt = 1 and 0.1 (first Kelvin DRM blow-up from the last sweep).
#
#   julia --project=. scripts/plot_bar12x6_sbm.jl
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
const tf = 4 * 4 * Lx
const PROBE = Point2D(Lx, Ly / 2)
const NPG = 12
const RBF = PHS(1; poly_deg=1)
const SBM_RBF = PHS(1; poly_deg=-1)
const MASSES = (:drm, :dibem, :cells)
const DTS = (1.0, 0.1)
const BLOW = 80.0

function _rect_mesh(bc_bot, bc_right, bc_top, bc_left; subdir, nome)
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
    gmsh.model.mesh.setTransfiniteCurve(l1, 13)
    gmsh.model.mesh.setTransfiniteCurve(l3, 13)
    gmsh.model.mesh.setTransfiniteCurve(l2, 7)
    gmsh.model.mesh.setTransfiniteCurve(l4, 7)
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
    msh = _rect_mesh("1;0", "1;-1", "1;0", "0;0"; subdir="Laplace", nome="sbm_lap12")
    dad = format2d(msh, Laplace(1.0); tipo=1, pontointerno=true)
    attach_analytical!(dad, ana_bar_sudden(; N=400, c=1.0, L=Lx))
    return dad
end

function make_elastic()
    msh = _rect_mesh("1;0;1;0", "1;1;1;0", "1;0;1;0", "0;0;0;0";
        subdir="elastico", nome="sbm_el12")
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

function probe_bem(dad)
    pts = all_points(dad)
    ip = argmin(norm(p - PROBE) for p in pts)
    dad.properties isa Laplace && return dad.T[ip, :]
    return dad.u[2 * (ip - 1) + 1, :]
end

function probe_sbm(sol, dad)
    pts = all_points(dad)
    ip = argmin(norm(p - PROBE) for p in pts)
    return sol.U[ip, :]
end

function clip_blow(ux)
    v = copy(ux)
    i0 = findfirst(i -> !isfinite(v[i]) || abs(v[i]) > BLOW, eachindex(v))
    i0 !== nothing && (v[i0:end] .= NaN)
    return v, i0
end
rel(a, b) = norm(a .- b) / (norm(b) + 1e-30)

const COLS = Dict(:drm => :darkorange, :dibem => :steelblue, :cells => :seagreen,
    :sbm => :purple)
const LABS = Dict(:drm => "DRM", :dibem => "DIBEM", :cells => "cells",
    :sbm => "SBM–DRM")

println(" 12×6 bar  SBM–DRM + BEM  PHS1+linear  tf=$tf")
dadL = make_laplace()
dadE = make_elastic()
H_G_full_direct(dadL; npg=NPG, threaded=true)
H_G_full_direct(dadE; npg=NPG, threaded=true)
@printf(" Laplace n=%d ni=%d   Kelvin n=%d ni=%d\n", dadL.n, dadL.ni, dadE.n, dadE.ni)

bemL = Dict(m => assemble_mass!(deepcopy(dadL), m) for m in MASSES)
bemE = Dict(m => assemble_mass!(deepcopy(dadE), m) for m in MASSES)
ana = ana_bar_sudden(; N=400, c=1.0, L=Lx)

function report(tag, mass, ux, ua, t)
    uxc, i0 = clip_blow(ux)
    blew = i0 !== nothing || !all(isfinite, ux)
    err = blew ? NaN : rel(ux, ua)
    mx = maximum(abs, filter(isfinite, ux); init=0.0)
    blow = i0 === nothing ? "—" : @sprintf("t=%.1f", t[i0])
    @printf("  %-8s %-8s  %10.3e  %10.3e  %s\n", tag, mass, err, mx, blow)
    return uxc
end

rows = []
for Δt in DTS
    println()
    @printf("── Δt = %g ──\n", Δt)
    @printf("  %-8s %-8s  %10s  %10s  %s\n", "phys", "mass", "err", "max|u|", "blow")
    panels = Dict{Symbol,Dict{Symbol,Vector{Float64}}}()
    tref = nothing
    ua = nothing

    panels[:laplace] = Dict{Symbol,Vector{Float64}}()
    for mass in MASSES
        dad = deepcopy(bemL[mass])
        solve_Houbolt(dad, Δt, tf)
        tref === nothing && (tref = dad.time)
        ua === nothing && (ua = [float(ana.u(PROBE; t=ti)) for ti in tref])
        panels[:laplace][mass] = report("laplace", mass, probe_bem(dad), ua, dad.time)
    end
    sol = solve_sbm_wave(deepcopy(dadL); Δt=Δt, tf=tf, c=1.0, scheme=:houbolt,
        basis=SBM_RBF)
    panels[:laplace][:sbm] = report("laplace", :sbm, probe_sbm(sol, dadL), ua, sol.t)
    @printf("           SBM n_drop=%d  N=%d M=%d\n", sol.n_drop, sol.N, sol.M)

    panels[:elastic] = Dict{Symbol,Vector{Float64}}()
    for mass in MASSES
        dad = deepcopy(bemE[mass])
        solve_Houbolt(dad, Δt, tf)
        panels[:elastic][mass] = report("elastic", mass, probe_bem(dad), ua, dad.time)
    end
    push!(rows, (; Δt, panels, ua, t=tref))
end

ps = Plots.Plot[]
for (i, row) in enumerate(rows)
    for (j, tag) in enumerate((:laplace, :elastic))
        p = plot()
        showleg = i == 1 && j == 1
        plot!(p, row.t, row.ua; color=:black, ls=:dash,
            label=showleg ? "1D series" : "")
        for mass in MASSES
            plot!(p, row.t, row.panels[tag][mass]; color=COLS[mass],
                label=showleg ? LABS[mass] : "")
        end
        if tag === :laplace
            plot!(p, row.t, row.panels[:laplace][:sbm]; color=COLS[:sbm],
                label=showleg ? LABS[:sbm] : "")
        end
        phys = tag === :laplace ? "Laplace" : "Kelvin (no SBM)"
        plot!(p; xlabel=(i == 2 ? L"t" : ""),
            ylabel=(j == 1 ? L"u(L,H/2)" : ""),
            title="$(phys)  Δt=$(row.Δt)",
            ylim=(-2, 28), legend=showleg ? :outertopright : false)
        push!(ps, p)
    end
end
plt = plot(ps...; layout=(2, 2), size=(1100, 640),
    plot_title="12×6 bar  SBM–DRM + BEM  PHS1  Houbolt  4 periods")
mkpath(joinpath(projectdir(), "plots"))
out = joinpath(projectdir(), "plots", "bar12x6_sbm_phs1")
savefig(plt, out * ".png")
savefig(plt, out * ".pdf")
println("\nwrote ", out * ".png")
