# 2D-only figures for the anisotropic-Laplace Typst report.
using BEM
using LinearAlgebra
using StaticArrays
using DelimitedFiles
using Printf
using Plots

include(joinpath(@__DIR__, "..", "..", "data", "Laplace", "Laplace_dad.jl"))

K = @SMatrix [5.0 0.0; 0.0 0.5]
rbf = PHS(3; poly_deg=2)
csv = joinpath(@__DIR__, "..", "..", "data", "Laplace", "fenics_ortho_plate_Tright.csv")
data = readdlm(csv, ',')
i0 = data[1, 1] isa AbstractString ? 2 : 1
yref = Float64.(data[i0:end, 1])
Tref = Float64.(data[i0:end, 2])

outdir = joinpath(@__DIR__, "..", "..", ".scratch", "anisotropic-laplace", "figures")
mkpath(outdir)

function _interp_lin(x, y, xq)
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

function right_T(dad)
    ys = Float64[]; Ts = Float64[]
    for i in 1:dad.n
        p = dad.Nodes[i]
        abs(p[1] - 1) < 0.03 || continue
        push!(ys, p[2]); push!(Ts, dad.T[i])
    end
    perm = sortperm(ys)
    return ys[perm], Ts[perm]
end

function rms_vs_fenics(y, T)
    Tb = _interp_lin(y, T, yref)
    return norm(Tb .- Tref) / (norm(Tref) + eps())
end

# --- geometry ---
dadg = format2d(placa_furo_orto(; lc=0.06, nome="rep_geo", show=false, ordem=2),
    Laplace(1.0); pontointerno=true, tipo=2)
θ = range(0, 2π; length=200)
geo = plot(aspect_ratio=1, xlims=(-0.08, 1.18), ylims=(-0.12, 1.12),
    xlabel="x", ylabel="y", legend=:topright, size=(640, 560),
    title="Unit square minus disk  (r = 0.25)", grid=false, framestyle=:box)
plot!(geo, [0, 1, 1, 0, 0], [0, 0, 1, 1, 0]; color=:black, lw=2, label="")
plot!(geo, 0.5 .+ 0.25 .* cos.(θ), 0.5 .+ 0.25 .* sin.(θ);
    color=:black, lw=2, label="")
# collocation
bx = [p[1] for p in dadg.Nodes[1:dadg.n]]
by = [p[2] for p in dadg.Nodes[1:dadg.n]]
ix = [p[1] for p in dadg.internalNodes]
iy = [p[2] for p in dadg.internalNodes]
scatter!(geo, ix, iy; ms=2, msw=0, color=:gray70, label="interior (DIBEM)")
scatter!(geo, bx, by; ms=3, msw=0, color=:steelblue, label="boundary collocation")
annotate!(geo, -0.02, 0.50, text("T = 0", 10, :right, :red))
annotate!(geo, 1.02, 0.50, text("q = −5", 10, :left, :blue))
annotate!(geo, 0.50, -0.06, text("q = 0", 10, :center))
annotate!(geo, 0.50, 1.06, text("q = 0", 10, :center))
annotate!(geo, 0.50, 0.50, text("hole  q = 0", 9, :center))
savefig(geo, joinpath(outdir, "geometry.png"))
println("wrote geometry.png  n=", dadg.n, " ni=", dadg.ni)

# --- three meshes ---
meshes = [(name="coarse", lc=0.10), (name="medium", lc=0.06), (name="fine", lc=0.04)]
curves = Dict{Tuple{String,Symbol},Tuple{Vector{Float64},Vector{Float64}}}()
stats = []
for m in meshes
    msh = placa_furo_orto(; lc=m.lc, nome="rep2d_$(m.name)", show=false, ordem=2)
    d2 = format2d(msh, AnisotropicLaplace(K); pontointerno=false, tipo=2)
    assemble!(d2; npg=10, threaded=false); solve(d2)
    y, T = right_T(d2)
    curves[(m.name, :S2)] = (y, T)
    push!(stats, (m.name, :S2, d2.n, d2.ni, rms_vs_fenics(y, T), T[argmin(abs.(y .- 0.5))]))

    d1 = format2d(msh, Laplace(1.0); pontointerno=true, tipo=2)
    assemble!(d1; npg=10, threaded=false); DIBEM(d1; rbf=rbf)
    solve_anisotropic_dibem!(d1, K; rbf=rbf)
    y, T = right_T(d1)
    curves[(m.name, :S1)] = (y, T)
    push!(stats, (m.name, :S1, d1.n, d1.ni, rms_vs_fenics(y, T), T[argmin(abs.(y .- 0.5))]))

    d3 = format2d(msh, Laplace(1.0); pontointerno=true, tipo=2)
    assemble!(d3; npg=10, threaded=false); DIBEM(d3; rbf=rbf)
    solve_anisotropic_ibp!(d3, K; rbf=rbf)
    y, T = right_T(d3)
    curves[(m.name, :S3)] = (y, T)
    push!(stats, (m.name, :S3, d3.n, d3.ni, rms_vs_fenics(y, T), T[argmin(abs.(y .- 0.5))]))
    @printf("%-8s S2 RMS=%.4f Tmid=%.3f  S1 RMS=%.4f Tmid=%.3f  S3 RMS=%.4f Tmid=%.3f\n",
        m.name, stats[end-2][5], stats[end-2][6],
        stats[end-1][5], stats[end-1][6], stats[end][5], stats[end][6])
end

labs = Dict(:S2 => "S2 anisotropic FS", :S1 => "S1 DIBEM Hessian", :S3 => "S3 IBP + DIBEM")
stys = Dict(:S2 => :dash, :S1 => :dot, :S3 => :solid)
cols = Dict(:S2 => :darkorange, :S1 => :seagreen, :S3 => :purple)

plt = plot(layout=(1, 3), size=(1200, 420), legend=false,
    plot_title="Right-edge T(x=1, y)  vs FEniCS")
for (j, m) in enumerate(meshes)
    plot!(plt, yref, Tref; subplot=j, label="FEniCS", lw=2.5, color=:black)
    for s in (:S2, :S1, :S3)
        y, T = curves[(m.name, s)]
        plot!(plt, y, T; subplot=j, label=labs[s], lw=2, ls=stys[s], color=cols[s])
    end
    nS2 = stats[3*(j-1)+1][3]
    title!(plt, "$(m.name)  lc=$(m.lc)  n=$nS2"; subplot=j)
    xlabel!(plt, "y"; subplot=j)
    ylabel!(plt, "T(x=1)"; subplot=j)
    ylims!(plt, 1.2, 3.0; subplot=j)
    j == 1 && plot!(plt; subplot=j, legend=:bottom, legendfontsize=7)
end
savefig(plt, joinpath(outdir, "profiles_meshes.png"))
println("wrote profiles_meshes.png")

# medium overlay (report main figure)
med = plot(yref, Tref; label="FEniCS", lw=2.8, color=:black,
    xlabel="y", ylabel="T(x=1)", legend=:outerright, size=(880, 460),
    title="Medium mesh (lc = 0.06, quadratic)", grid=true)
plot!(med, curves[("medium", :S2)]...; label="S2 anisotropic FS", lw=2, ls=:dash, color=:darkorange)
plot!(med, curves[("medium", :S1)]...; label="S1 DIBEM Hessian", lw=2.2, ls=:dot, color=:seagreen)
plot!(med, curves[("medium", :S3)]...; label="S3 IBP + DIBEM", lw=2.2, color=:purple)
savefig(med, joinpath(outdir, "profiles_medium.png"))
println("wrote profiles_medium.png")

# pointwise error on medium
err = plot(xlabel="y", ylabel="T_BEM − T_FEniCS", legend=:outerright,
    size=(880, 360), title="Medium mesh: pointwise error on x=1", grid=true)
hline!(err, [0.0]; color=:black, lw=1, label="")
for s in (:S2, :S1, :S3)
    y, T = curves[("medium", s)]
    Tb = _interp_lin(y, T, yref)
    plot!(err, yref, Tb .- Tref; label=labs[s], lw=2, ls=stys[s], color=cols[s])
end
savefig(err, joinpath(outdir, "errors_medium.png"))
println("wrote errors_medium.png")

open(joinpath(outdir, "stats.csv"), "w") do io
    println(io, "mesh,strategy,n,ni,rms,Tmid")
    for s in stats
        @printf(io, "%s,%s,%d,%d,%.6f,%.6f\n", s...)
    end
end
