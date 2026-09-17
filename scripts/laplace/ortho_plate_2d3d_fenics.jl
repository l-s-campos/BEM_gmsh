# Perforated orthotropic plate: 2D + extruded 3D, three meshes, three
# anisotropic strategies, vs FEniCS T(x=1, y).
#
#   S2  AnisotropicLaplace Green's function
#   S1  isotropic Laplace + DIBEM residual (Hessian)
#   S3  one IBP + DIBEM (no Hessian)
using BEM
using LinearAlgebra
using StaticArrays
using DelimitedFiles
using Printf
using Plots

using DrWatson: datadir
include(joinpath(@__DIR__, "..", "..", "data", "Laplace", "Laplace_dad.jl"))

K2 = @SMatrix [5.0 0.0; 0.0 0.5]
K3 = @SMatrix [5.0 0.0 0.0; 0.0 0.5 0.0; 0.0 0.0 1.0]
rbf = PHS(3; poly_deg=2)
dz = 0.2
meshes = [
    (name="coarse", lc2=0.10, lc3=0.15),
    (name="medium", lc2=0.06, lc3=0.12),
    (name="fine",   lc2=0.04, lc3=0.10),
]
# Keep enough volume centroids that 3D RBF-FD stencils are not pancakes.
nmax_int3d = 800

csv = joinpath(@__DIR__, "..", "..", "data", "Laplace", "fenics_ortho_plate_Tright.csv")
data = readdlm(csv, ',')
i0 = data[1, 1] isa AbstractString ? 2 : 1
yref = Float64.(data[i0:end, 1])
Tref = Float64.(data[i0:end, 2])
Tref_mid = Tref[argmin(abs.(yref .- 0.5))]

function _interp_lin(x, y, xq)
    n = length(x)
    out = similar(xq)
    @inbounds for (k, z) in enumerate(xq)
        if z <= x[1]
            out[k] = y[1]
        elseif z >= x[end]
            out[k] = y[end]
        else
            j = searchsortedlast(x, z)
            t = (z - x[j]) / (x[j+1] - x[j] + eps())
            out[k] = (1 - t) * y[j] + t * y[j+1]
        end
    end
    return out
end

function right_T(dad; x0=1.0, atol=0.04, zmid=nothing, ztol=nothing)
    ys = Float64[]; Ts = Float64[]
    zt = ztol === nothing ? atol : ztol
    for i in 1:dad.n
        p = dad.Nodes[i]
        abs(p[1] - x0) < atol || continue
        dad.dimension == 3 && abs(dad.Normal[i][1]) < 0.7 && continue
        if zmid !== nothing && abs(p[3] - zmid) > zt
            continue
        end
        push!(ys, p[2]); push!(Ts, dad.T[i])
    end
    isempty(ys) && return ys, Ts
    perm = sortperm(ys)
    return ys[perm], Ts[perm]
end

function rms_vs_fenics(y, T)
    isempty(y) && return NaN
    Tb = _interp_lin(y, T, yref)
    return norm(Tb .- Tref) / (norm(Tref) + eps())
end

function tmid(y, T)
    isempty(y) && return NaN
    return T[argmin(abs.(y .- 0.5))]
end

function _thin_internals!(dad, nmax::Int)
    dad.ni <= nmax && return dad
    pts = collect(dad.internalNodes)
    step = max(1, ceil(Int, length(pts) / nmax))
    set_internal_nodes!(dad, pts[1:step:end])
    return dad
end

function solve_2d(msh, strat)
    if strat === :S2
        dad = format2d(msh, AnisotropicLaplace(K2); pontointerno=false, tipo=2)
        assemble!(dad; npg=10, threaded=false)
        solve(dad)
        return dad
    end
    dad = format2d(msh, Laplace(1.0); pontointerno=true, tipo=2)
    assemble!(dad; npg=10, threaded=false)
    DIBEM(dad; rbf=rbf)
    strat === :S1 ? solve_anisotropic_dibem!(dad, K2; rbf=rbf) :
                    solve_anisotropic_ibp!(dad, K2; rbf=rbf)
    return dad
end

function solve_3d(msh, strat)
    if strat === :S2
        dad = format3d(msh, AnisotropicLaplace(K3); pontointerno=false)
        assemble!(dad; npg=8, threaded=false)
        solve(dad)
        return dad
    end
    dad = format3d(msh, Laplace(1.0); pontointerno=true)
    _thin_internals!(dad, nmax_int3d)
    assemble!(dad; npg=8, threaded=false)
    DIBEM(dad; rbf=rbf)
    strat === :S1 ? solve_anisotropic_dibem!(dad, K3; rbf=rbf) :
                    solve_anisotropic_ibp!(dad, K3; rbf=rbf)
    return dad
end

strats = (:S2, :S1, :S3)
labels = Dict(:S2 => "S2 anisotropic FS", :S1 => "S1 DIBEM Hessian",
    :S3 => "S3 IBP + DIBEM")
styles = Dict(:S2 => :dash, :S1 => :dot, :S3 => :solid)

rows = NamedTuple[]
curves = Dict{Tuple{Symbol,String,Symbol},Tuple{Vector{Float64},Vector{Float64}}}()

println("FEniCS T(x=1, y=0.5) = ", Tref_mid)
for m in meshes
    msh2 = placa_furo_orto(; lc=m.lc2, nome="cmp2d_$(m.name)", show=false, ordem=2)
    msh3 = placa_furo_orto_3d(; lc=m.lc3, nome="cmp3d_$(m.name)", dz=dz,
        show=false, recombine=false)
    for strat in strats
        t0 = time()
        dad = solve_2d(msh2, strat)
        y, T = right_T(dad)
        rms = rms_vs_fenics(y, T)
        tm = tmid(y, T)
        dt = time() - t0
        push!(rows, (dim=2, mesh=m.name, lc=m.lc2, strat=String(strat),
            n=dad.n, ni=dad.ni, rms=rms, Tmid=tm, sec=dt))
        curves[(:d2, m.name, strat)] = (y, T)
        @printf("2D %-6s %-3s n=%4d ni=%4d  RMS=%.4f  Tmid=%.4f  %.1fs\n",
            m.name, strat, dad.n, dad.ni, rms, tm, dt)
        flush(stdout)
    end
    for strat in strats
        t0 = time()
        dad = solve_3d(msh3, strat)
        y, T = right_T(dad; zmid=dz / 2, ztol=0.35 * dz)
        if length(y) < 4
            y, T = right_T(dad)
        end
        rms = rms_vs_fenics(y, T)
        tm = tmid(y, T)
        dt = time() - t0
        push!(rows, (dim=3, mesh=m.name, lc=m.lc3, strat=String(strat),
            n=dad.n, ni=dad.ni, rms=rms, Tmid=tm, sec=dt))
        curves[(:d3, m.name, strat)] = (y, T)
        @printf("3D %-6s %-3s n=%4d ni=%4d  RMS=%.4f  Tmid=%.4f  %.1fs\n",
            m.name, strat, dad.n, dad.ni, rms, tm, dt)
        flush(stdout)
    end
end

outdir = joinpath(@__DIR__, "..", "..", "plots")
mkpath(outdir)

plt = plot(layout=(2, 3), size=(1400, 900), legend=false,
    plot_title="Orthotropic plate with hole  kx=5, ky=0.5  vs FEniCS")
for (j, m) in enumerate(meshes)
    for (i, dim) in enumerate((:d2, :d3))
        sp = (i - 1) * 3 + j
        plot!(plt, yref, Tref; subplot=sp, label="FEniCS", lw=2.5, color=:black)
        for strat in strats
            y, T = curves[(dim, m.name, strat)]
            plot!(plt, y, T; subplot=sp, label=labels[strat], lw=2, ls=styles[strat])
        end
        lc = dim === :d2 ? m.lc2 : m.lc3
        title!(plt, "$(dim === :d2 ? "2D" : "3D") $(m.name)  lc=$lc"; subplot=sp)
        xlabel!(plt, "y"; subplot=sp)
        ylabel!(plt, "T(x=1)"; subplot=sp)
        ylims!(plt, 1.0, 4.5; subplot=sp)
        sp == 1 && plot!(plt; subplot=sp, legend=:topleft, legendfontsize=7)
    end
end
outpng = joinpath(outdir, "ortho_plate_2d3d_fenics.png")
savefig(plt, outpng)
println("wrote ", outpng)

tab = joinpath(outdir, "ortho_plate_2d3d_fenics.csv")
open(tab, "w") do io
    println(io, "dim,mesh,lc,strategy,n,ni,rms_fenics,Tmid,seconds")
    for r in rows
        @printf(io, "%d,%s,%.3f,%s,%d,%d,%.6f,%.6f,%.2f\n",
            r.dim, r.mesh, r.lc, r.strat, r.n, r.ni, r.rms, r.Tmid, r.sec)
    end
end
println("wrote ", tab)
println("FEniCS Tmid = ", Tref_mid)
