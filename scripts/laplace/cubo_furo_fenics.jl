# Perforated cube [0,1]³ vs thin extruded plate, three anisotropic strategies.
# T(x,y,z)=T_2D(x,y) so the FEniCS right-edge curve still applies.
using BEM
using LinearAlgebra
using StaticArrays
using DelimitedFiles
using Printf
using Statistics
using Plots

include(joinpath(@__DIR__, "..", "..", "data", "Laplace", "Laplace_dad.jl"))

K2 = @SMatrix [5.0 0.0; 0.0 0.5]
K3 = @SMatrix [5.0 0.0 0.0; 0.0 0.5 0.0; 0.0 0.0 1.0]
rbf = PHS(3; poly_deg=2)

csv = joinpath(@__DIR__, "..", "..", "data", "Laplace", "fenics_ortho_plate_Tright.csv")
data = readdlm(csv, ',')
i0 = data[1, 1] isa AbstractString ? 2 : 1
yref = Float64.(data[i0:end, 1])
Tref = Float64.(data[i0:end, 2])
Tref_mid = Tref[argmin(abs.(yref .- 0.5))]

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

function describe_mesh(label, dad)
    xs = [p[1] for p in dad.Nodes[1:dad.n]]
    ys = [p[2] for p in dad.Nodes[1:dad.n]]
    zs = [p[3] for p in dad.Nodes[1:dad.n]]
    @printf("%-18s n=%4d ni=%4d ne=%4d  bbox x[%.2f,%.2f] y[%.2f,%.2f] z[%.2f,%.2f]  Δz=%.3f\n",
        label, dad.n, dad.ni, length(dad.elements),
        minimum(xs), maximum(xs), minimum(ys), maximum(ys),
        minimum(zs), maximum(zs), maximum(zs) - minimum(zs))
end

function right_T(dad; L=1.0, zmid=nothing, ztol=nothing)
    ys = Float64[]; Ts = Float64[]
    zt = ztol === nothing ? 0.15 * L : ztol
    zm = zmid === nothing ? 0.5 * L : zmid
    for i in 1:dad.n
        p = dad.Nodes[i]
        abs(p[1] - L) < 0.04 || continue
        abs(dad.Normal[i][1]) > 0.7 || continue
        if dad.dimension == 3 && abs(p[3] - zm) > zt
            continue
        end
        push!(ys, p[2]); push!(Ts, dad.T[i])
    end
    isempty(ys) && return ys, Ts
    perm = sortperm(ys)
    return ys[perm], Ts[perm]
end

function rms_fenics(y, T)
    isempty(y) && return NaN
    Tb = _interp_lin(y, T, yref)
    return norm(Tb .- Tref) / (norm(Tref) + eps())
end

function tmid(y, T)
    isempty(y) && return NaN
    return T[argmin(abs.(y .- 0.5))]
end

println("=== geometry ===")
dthin = format3d(placa_furo_orto_3d(; lc=0.15, dz=0.2, nome="geo_thin",
        recombine=false), Laplace(1.0); pontointerno=true)
describe_mesh("thin plate dz=0.2", dthin)
dcube = format3d(cubo_furo_orto(; lc=0.15, nome="geo_cube", recombine=false),
    Laplace(1.0); pontointerno=true)
describe_mesh("perforated cube", dcube)

# --- 2D reference (quadratic) ---
println("\n=== 2D S2 (reference BEM) ===")
d2 = format2d(placa_furo_orto(; lc=0.06, nome="cube_ref2d", show=false, ordem=2),
    AnisotropicLaplace(K2); pontointerno=false, tipo=2)
assemble!(d2; npg=10, threaded=false)
solve(d2)
y2, T2 = right_T(d2)
@printf("2D S2  n=%d  RMS=%.4f  Tmid=%.3f  (FEniCS %.3f)\n",
    d2.n, rms_fenics(y2, T2), tmid(y2, T2), Tref_mid)

# --- 3D cube, three strategies ---
println("\n=== perforated cube S2 / S1 / S3 ===")
lc = 0.18
msh = cubo_furo_orto(; lc=lc, nome="cmp_cube", recombine=false)
L = 1.0

ds2 = format3d(msh, AnisotropicLaplace(K3); pontointerno=false)
assemble!(ds2; npg=8, threaded=false)
t0 = time(); solve(ds2); dt = time() - t0
y, T = right_T(ds2; L=L, zmid=0.5, ztol=0.2)
@printf("S2  n=%d ni=%d  RMS=%.4f  Tmid=%.3f  Tright∈(%.2f, %.2f)  %.1fs\n",
    ds2.n, ds2.ni, rms_fenics(y, T), tmid(y, T), extrema(T)..., dt)
curves = Dict(:S2 => (y, T))

dad = format3d(msh, Laplace(1.0); pontointerno=true)
if dad.ni > 500
    step = ceil(Int, dad.ni / 500)
    set_internal_nodes!(dad, collect(dad.internalNodes)[1:step:end])
end
assemble!(dad; npg=8, threaded=false)
DIBEM(dad; rbf=rbf)
@printf("DIBEM cloud n=%d ni=%d nt=%d\n", dad.n, dad.ni, dad.nt)

d1 = deepcopy(dad)
t0 = time(); solve_anisotropic_dibem!(d1, K3; rbf=rbf); dt = time() - t0
y, T = right_T(d1; L=L, zmid=0.5, ztol=0.2)
@printf("S1  n=%d ni=%d  RMS=%.4f  Tmid=%.3f  Tright∈(%.2f, %.2f)  %.1fs\n",
    d1.n, d1.ni, rms_fenics(y, T), tmid(y, T), extrema(T)..., dt)
curves[:S1] = (y, T)

d3 = deepcopy(dad)
t0 = time(); solve_anisotropic_ibp!(d3, K3; rbf=rbf); dt = time() - t0
y, T = right_T(d3; L=L, zmid=0.5, ztol=0.2)
@printf("S3  n=%d ni=%d  RMS=%.4f  Tmid=%.3f  Tright∈(%.2f, %.2f)  %.1fs  |S1-S3|/|S1|=%.3f\n",
    d3.n, d3.ni, rms_fenics(y, T), tmid(y, T), extrema(T)..., dt,
    norm(d3.T .- d1.T) / (norm(d1.T) + eps()))
curves[:S3] = (y, T)

outdir = joinpath(@__DIR__, "..", "..", "plots")
mkpath(outdir)
plt = plot(yref, Tref; label="FEniCS 2D", lw=2.5, color=:black,
    xlabel="y", ylabel="T(x=1, z=L/2)",
    title="Perforated cube  kx=5, ky=0.5, kz=1  vs FEniCS",
    legend=:outerright, size=(900, 480), ylims=(1.0, 4.5))
plot!(plt, y2, T2; label="2D S2 FS", lw=2, ls=:dashdot, color=:gray)
plot!(plt, curves[:S2][1], curves[:S2][2]; label="3D S2 FS", lw=2, ls=:dash)
plot!(plt, curves[:S1][1], curves[:S1][2]; label="3D S1 Hess (interior quad)", lw=2, ls=:dot)
plot!(plt, curves[:S3][1], curves[:S3][2]; label="3D S3 IBP", lw=2)
out = joinpath(outdir, "cubo_furo_fenics.png")
savefig(plt, out)
println("wrote ", out)
println("FEniCS Tmid = ", Tref_mid)
