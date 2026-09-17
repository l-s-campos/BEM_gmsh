# Same 12×6 bar / Houbolt Δt=1 / 4 periods, Kelvin elasticity kernel (ν=0).
# DIBEM: PHS3 + linear. DRM: f=r + linear poly (no r³ particular in Kelvin DRM).
#
#   julia --project=. scripts/plot_elast_bar12x6.jl
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf
using Plots, LaTeXStrings
gr()
default(fontfamily="Computer Modern", linewidth=1.6, framestyle=:box,
    grid=false, dpi=160, size=(880, 400), legendfontsize=8)

include(datadir("Laplace", "Laplace_dad.jl"))
include(datadir("Laplace", "potencial_problems.jl"))
include(datadir("Laplace", "wave_propagation.jl"))
include(datadir("elastico", "iso", "bar_sudden.jl"))

const Lx, Ly = 12.0, 6.0
const Tper = 4 * Lx
const Δt = 1.0
const tf = 4 * Tper
const PROBE = Point2D(Lx, Ly / 2)
const NPG = 12
const RBF = PHS(3; poly_deg=1)   # r³ + linear (1, x, y)

function mesh_elast_12x6(; ndivx=13, ndivy=7, ordem=1, P=1.0, nome="elast_bar12x6")
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(nome)
    lc = 1.0
    p1 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, lc)
    p2 = gmsh.model.geo.addPoint(Lx, 0.0, 0.0, lc)
    p3 = gmsh.model.geo.addPoint(Lx, Ly, 0.0, lc)
    p4 = gmsh.model.geo.addPoint(0.0, Ly, 0.0, lc)
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
    gmsh.model.addPhysicalGroup(1, [l1, l3], -1, "1;0;1;0")
    gmsh.model.addPhysicalGroup(1, [l2], -1, "1;$P;1;0")
    gmsh.model.addPhysicalGroup(1, [l4], -1, "0;0;0;0")
    gmsh.model.addPhysicalGroup(2, [s], -1, "Domain")
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(ordem)
    out = datadir("elastico", nome * ".msh")
    mkpath(dirname(out))
    gmsh.write(out)
    gmsh.finalize()
    return out
end

function make_dad()
    msh = mesh_elast_12x6()
    props = Elasticity(E=1.0, nu=0.0, rho=1.0; plane_stress=true)
    dad = format2d(msh, props; tipo=1, pontointerno=true)
    attach_analytical!(dad, ana_bar_sudden(; N=400, c=1.0, L=Lx))
    return dad
end

clip_blow(ux; lim=80) = begin
    v = copy(ux)
    i0 = findfirst(i -> !isfinite(v[i]) || abs(v[i]) > lim, eachindex(v))
    i0 !== nothing && (v[i0:end] .= NaN)
    v, i0
end
rel(a, b) = norm(a .- b) / (norm(b) + 1e-30)

function probe_ux(U, dad)
    pts = all_points(dad)
    ip = argmin(norm(p - PROBE) for p in pts)
    return U[2 * (ip - 1) + 1, :], ip
end

function mass!(dad, mass)
    if mass === :drm
        build_drm_matrices(dad; npg=NPG, kernel=:r, poly_deg=1)
    elseif mass === :dibem
        DIBEM(dad; method=:dense, rbf=RBF)
    else
        build_cell_mass(dad; npg=NPG)
    end
    return dad.M
end

println(" Elasticity (Kelvin, ν=0)  12×6 bar  Houbolt  Δt=$Δt  4 periods")
println(" DIBEM PHS3+linear; DRM f=r + linear poly; cells")
dad0 = make_dad()
@printf(" elements=%d  n=%d  ni=%d  nt=%d  ndof=%d  cells=%d\n",
    length(dad0.elements), dad0.n, dad0.ni, dad0.nt, 2 * dad0.nt,
    length(extract_domain_cells(dad0)))
H_G_full_direct(dad0; npg=NPG, threaded=true)
ana = ana_bar_sudden(; N=400, c=1.0, L=Lx)

cols = Dict(:drm => :darkorange, :dibem => :steelblue, :cells => :seagreen)
labs = Dict(:drm => "DRM f=r+lin", :dibem => "DIBEM PHS3+lin", :cells => "cells")
runs = Dict{Symbol,NamedTuple}()

@printf("\n  %-12s  %10s  %10s  %10s  %s\n", "mass", "u_end_err", "max|ux|", "blow", "finite")
for mass in (:drm, :dibem, :cells)
    dad = deepcopy(dad0)
    mass!(dad, mass)
    solve_Houbolt(dad, Δt, tf)
    ux, _ = probe_ux(dad.u, dad)
    t = dad.time
    ua = [float(ana.u(PROBE; t=ti)) for ti in t]
    uxc, i0 = clip_blow(ux)
    ok = all(isfinite, ux)
    err = ok && i0 === nothing ? rel(ux, ua) : NaN
    mx = maximum(abs, filter(isfinite, ux); init=0.0)
    blow = i0 === nothing ? "—" : @sprintf("t=%.1f", t[i0])
    @printf("  %-12s  %10.3e  %10.3e  %10s  %s\n", mass, err, mx, blow, ok)
    runs[mass] = (; t, ux=uxc, ua)
end

plt = plot(runs[:cells].t, runs[:cells].ua; color=:black, ls=:dash, label="1D series")
for mass in (:drm, :dibem, :cells)
    plot!(plt, runs[mass].t, runs[mass].ux; color=cols[mass], label=labs[mass])
end
plot!(plt; xlabel=L"t", ylabel=L"u_x(L, H/2)", ylim=(-2, 28),
    legend=:outertopright,
    title="12×6 bar  Kelvin \$\\nu=0\$  PHS3+linear  Houbolt  \$\\Delta t=1\$  4 periods")
mkpath(joinpath(projectdir(), "plots"))
out = joinpath(projectdir(), "plots", "elast_bar12x6_phs3_dt1")
savefig(plt, out * ".png")
savefig(plt, out * ".pdf")
println("wrote ", out * ".png")
