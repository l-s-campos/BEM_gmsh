# Fast n_int check: DIBEM PHS3 residual vs cells. Two grids only.
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf

include(datadir("Laplace", "Laplace_dad.jl"))
include(datadir("Laplace", "potencial_problems.jl"))
include(datadir("Laplace", "wave_propagation.jl"))
include(datadir("elastico", "iso", "bar_sudden.jl"))

function fields!(u, tv, ddu, dad, t, meta)
    pts = all_points(dad)
    @inbounds for i in eachindex(pts)
        f = bar_sudden_fields(pts[i], t; N=200, c=meta.c, L=meta.L)
        u[2i-1] = f.u; u[2i] = 0.0
        ddu[2i-1] = f.ddu; ddu[2i] = 0.0
    end
    @inbounds for i in 1:dad.n
        f = bar_sudden_fields(dad.Nodes[i], t; N=200, c=meta.c, L=meta.L)
        tv[2i-1] = meta.E * f.dudx * dad.Normal[i][1]
        tv[2i] = 0.0
    end
end

relres(H, G, M, u, tv, ddu) = (r = H * u .+ M * ddu .- G * tv;
    norm(r) / (norm(H * u) + norm(M * ddu) + norm(G * tv) + 1e-30))

io = open(joinpath(@__DIR__, "dibem_nint_sweep.out"), "w")
println(io, "n_int  ni   c<0   ||ü||     t=0.25      t=0.50      t=1.00")
flush(io)

for ni in (4, 8)
    dad, meta = elasticity_bar_sudden(; ndiv=8, n_int=ni, ν=0.0)
    H_G_full_direct(dad; npg=8, threaded=false)
    H, G = Matrix(dad.H), Matrix(dad.G)
    DIBEM(dad; method=:dense, rbf=PHS(3; poly_deg=1))  # same remainder M as phs3
    Md = Matrix(dad.M)
    cneg = count(<(0), dad.dibem_c)
    Mc = build_cell_mass(deepcopy(dad); npg=8).M
    xs = unique(p[1] for p in dad.internalNodes)
    d50 = minimum(abs(x - 0.5) for x in xs)
    d75 = minimum(abs(x - 0.75) for x in xs)
    rD = Float64[]; rC = Float64[]; nddu = 0.0
    for t in (0.25, 0.5, 1.0)
        u = zeros(2dad.nt); tv = zeros(2dad.n); ddu = zeros(2dad.nt)
        fields!(u, tv, ddu, dad, t, meta)
        push!(rD, relres(H, G, Md, u, tv, ddu))
        push!(rC, relres(H, G, Mc, u, tv, ddu))
        t == 0.5 && (nddu = norm(ddu))
    end
    @printf(io, "DIBEM %2d  %3d  %3d  %.2e  %.3e   %.3e   %.3e\n",
        ni, dad.ni, cneg, nddu, rD[1], rD[2], rD[3])
    @printf(io, "cells %2d  %3d    -           %.3e   %.3e   %.3e\n",
        ni, dad.ni, rC[1], rC[2], rC[3])
    @printf(io, "      |x_int-0.50|=%.3f  |x_int-0.75|=%.3f\n", d50, d75)
    flush(io)
end
println(io, "done")
close(io)
