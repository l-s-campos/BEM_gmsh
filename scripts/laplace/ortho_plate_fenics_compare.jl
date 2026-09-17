# Orthotropic plate-with-hole: FEniCS vs BEM strategy 2 (anisotropic FS),
# strategy 1 (isotropic Laplace + DIBEM residual / Hess), and strategy 3
# (one IBP + DIBEM, no Hess; RIM via int(rbf) and analytic ∇Φ primitive).
using BEM
using LinearAlgebra
using StaticArrays
using DelimitedFiles
using Plots

using DrWatson: datadir
include(joinpath(@__DIR__, "..", "..", "data", "Laplace", "Laplace_dad.jl"))

K = @SMatrix [5.0 0.0; 0.0 0.5]
csv = joinpath(@__DIR__, "..", "..", "data", "Laplace", "fenics_ortho_plate_Tright.csv")
data = readdlm(csv, ',')
i0 = data[1, 1] isa AbstractString ? 2 : 1
yref = Float64.(data[i0:end, 1])
Tref = Float64.(data[i0:end, 2])

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

lc = 0.06
msh = placa_furo_orto(; lc=lc, nome="cmp_orto_fs", show=false, ordem=2)
dad2 = format2d(msh, AnisotropicLaplace(K); pontointerno=false, tipo=2)
assemble!(dad2; npg=12, threaded=true)
solve(dad2)

rbf = PHS(3; poly_deg=2)
dad1 = format2d(placa_furo_orto(; lc=lc, nome="cmp_orto_s1", show=false, ordem=2),
    Laplace(1.0); pontointerno=true, tipo=2)
assemble!(dad1; npg=12, threaded=true)
DIBEM(dad1; rbf=rbf)
solve_anisotropic_dibem!(dad1, K; rbf=rbf)

dad3 = format2d(placa_furo_orto(; lc=lc, nome="cmp_orto_s3", show=false, ordem=2),
    Laplace(1.0); pontointerno=true, tipo=2)
assemble!(dad3; npg=12, threaded=true)
DIBEM(dad3; rbf=rbf)
solve_anisotropic_ibp!(dad3, K; rbf=rbf)

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
y2, T2 = right_T(dad2)
y1, T1 = right_T(dad1)
y3, T3 = right_T(dad3)

function rms_vs_fenics(y, T)
    Tb = _interp_lin(y, T, yref)
    return norm(Tb .- Tref) / (norm(Tref) + eps())
end

plt = plot(yref, Tref; label="FEniCS", lw=2.5, xlabel="y", ylabel="T(x=1)",
    title="Orthotropic plate with hole, kx=5, ky=0.5",
    legend=:outerright, size=(900, 480), grid=true)
plot!(plt, y2, T2; label="anisotropic FS", lw=2, ls=:dash)
plot!(plt, y1, T1; label="S1 DIBEM Hessian", lw=2, ls=:dot)
plot!(plt, y3, T3; label="S3 IBP + DIBEM", lw=2.5)
mkpath(joinpath(@__DIR__, "..", "..", "plots"))
out = joinpath(@__DIR__, "..", "..", "plots", "ortho_plate_fenics_compare.png")
savefig(plt, out)
println("wrote ", out)
println("FS n=", dad2.n, "  S1 n=", dad1.n, "  S3 n=", dad3.n)
println("RMS vs FEniCS  FS=", rms_vs_fenics(y2, T2),
    "  S1=", rms_vs_fenics(y1, T1),
    "  S3=", rms_vs_fenics(y3, T3))
println("T(x=1) mid  FS=", T2[argmin(abs.(y2 .- 0.5))],
    "  S1=", T1[argmin(abs.(y1 .- 0.5))],
    "  S3=", T3[argmin(abs.(y3 .- 0.5))],
    "  FEniCS=", Tref[argmin(abs.(yref .- 0.5))])
