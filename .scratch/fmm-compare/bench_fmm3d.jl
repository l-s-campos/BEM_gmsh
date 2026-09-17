# Compare BEM.FMM vs FMM3D.jl vs FastMultipole.jl on 3D Laplace 1/(4πr).
#
# Usage:
#   OMP_NUM_THREADS=6 julia --project=/path/to/BEM_gmsh -t 6 \
#     .scratch/fmm-compare/bench_fmm3d.jl
#
# The scratch env (FMM3D, FastMultipole) is stacked via LOAD_PATH.

using Pkg
const BEM_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const SCRATCH = @__DIR__
Pkg.activate(BEM_ROOT)
using LinearAlgebra
using Random
using Statistics
using Printf
using BEM
using BEM.FMM

Pkg.activate(SCRATCH)
import FMM3D
import FastMultipole

include(joinpath(SCRATCH, "gravitational.jl"))

const INV4PI = 1 / (4π)
const OUT_CSV = joinpath(SCRATCH, "results_now.csv")
const OUT_LOG = joinpath(SCRATCH, "run_now.log")

relerr(a, b) = norm(a .- b) / (norm(b) + 1e-30)

function medtime(f; n::Int=5)
    f()
    ts = Vector{Float64}(undef, n)
    for i in 1:n
        ts[i] = @elapsed f()
    end
    return median(ts), minimum(ts)
end

function cube_points(rng, n)
    return rand(rng, 3, n)
end

function sphere_points(rng, n)
    P = randn(rng, 3, n)
    @inbounds for j in 1:n
        r = hypot(P[1, j], P[2, j], P[3, j])
        s = r < 1e-30 ? 1.0 : 1 / r
        P[1, j] *= s
        P[2, j] *= s
        P[3, j] *= s
    end
    return P
end

function make_grav(P, q)
    n = size(P, 2)
    bodies = zeros(8, n)
    @inbounds begin
        bodies[1:3, :] .= P
        bodies[5, :] .= q
    end
    return GravSystem(bodies)
end

function dense_laplace3d(P, q; skip_self=true)
    n = size(P, 2)
    y = zeros(n)
    @inbounds for j in 1:n
        acc = 0.0
        xj, yj, zj = P[1, j], P[2, j], P[3, j]
        for i in 1:n
            skip_self && i == j && continue
            dx = xj - P[1, i]
            dy = yj - P[2, i]
            dz = zj - P[3, i]
            r2 = dx * dx + dy * dy + dz * dz
            r2 < 1e-30 && continue
            acc += q[i] * INV4PI / sqrt(r2)
        end
        y[j] = acc
    end
    return y
end

function bem_p(eps)
    return FMM.laplace3d_nterms(eps; full_fmm=true)
end

function bem_nmax(eps)
    return FMM.laplace3d_ndiv(eps)
end

function align_sign!(y, yref)
    if dot(y, yref) < 0
        y .*= -1
    end
    return y
end

function run_case(; dist, n, eps, nrepeat)
    rng = Random.default_rng()
    Random.seed!(rng, 20260909)
    P = dist == "cube" ? cube_points(rng, n) : sphere_points(rng, n)
    q = randn(rng, n)
    p = bem_p(eps)
    nmax = bem_nmax(eps)
    yref = nothing
    y_bem = zeros(n)
    y_fmm3d = nothing
    y_fm = nothing
    y_tree = zeros(n)

    row = Dict{String,Any}(
        "dist" => dist,
        "N" => n,
        "eps" => eps,
        "p" => p,
        "nmax" => nmax,
        "threads" => Threads.nthreads(),
    )

    # --- FMM3D (rebuilds tree every call; OpenMP) ---
    t_fmm3d, tmin_fmm3d = medtime(n=nrepeat) do
        FMM3D.lfmm3d(eps, P; charges=q, pg=1)
    end
    v3 = FMM3D.lfmm3d(eps, P; charges=q, pg=1)
    y_fmm3d = copy(v3.pot)
    row["t_fmm3d_ms"] = 1e3 * t_fmm3d
    row["tmin_fmm3d_ms"] = 1e3 * tmin_fmm3d

    # --- BEM cached Gumerov FMM ---
    t_plan, _ = medtime(n=max(2, nrepeat - 2)) do
        FMM.build_laplace3d_plan(P; eps=eps, full_fmm=true)
    end
    plan = FMM.build_laplace3d_plan(P; eps=eps, full_fmm=true)
    t_apply, tmin_apply = medtime(n=nrepeat) do
        FMM.apply_laplace3d!(plan, y_bem; charges=q)
    end
    FMM.apply_laplace3d!(plan, y_bem; charges=q)
    row["t_bem_plan_ms"] = 1e3 * t_plan
    row["t_bem_apply_ms"] = 1e3 * t_apply
    row["tmin_bem_apply_ms"] = 1e3 * tmin_apply
    row["err_bem_vs_fmm3d"] = relerr(y_bem, y_fmm3d)

    # --- BEM cold (uncached lfmm3d). Legacy equivalent-sphere path; skip for N>2500. ---
    if n <= 2500
        t_cold, _ = medtime(n=2) do
            FMM.lfmm3d(eps, P; charges=q, pg=1, full_fmm=true)
        end
        vcold = FMM.lfmm3d(eps, P; charges=q, pg=1, full_fmm=true)
        row["t_bem_cold_ms"] = 1e3 * t_cold
        row["err_bem_cold_vs_fmm3d"] = relerr(vcold.pot, y_fmm3d)
    else
        row["t_bem_cold_ms"] = NaN
        row["err_bem_cold_vs_fmm3d"] = NaN
    end

    # Treecode (`full_fmm=false`) was dropped; apply is always octree FMM.
    row["t_bem_treecode_ms"] = NaN
    row["err_bem_treecode_vs_fmm3d"] = NaN

    # --- FastMultipole default ---
    sys0 = make_grav(P, q)
    t_fm_def, _ = medtime(n=max(2, nrepeat - 1)) do
        sys0.potential .= 0
        FastMultipole.fmm!(sys0; scalar_potential=true, gradient=false, hessian=false,
            silence_warnings=true)
    end
    sys0.potential .= 0
    FastMultipole.fmm!(sys0; scalar_potential=true, gradient=false, hessian=false,
        silence_warnings=true)
    y_fm_def = copy(sys0.potential[1, :])
    align_sign!(y_fm_def, y_fmm3d)
    row["t_fm_default_ms"] = 1e3 * t_fm_def
    row["err_fm_default_vs_fmm3d"] = relerr(y_fm_def, y_fmm3d)

    # --- FastMultipole matched p / leaf ---
    sys1 = make_grav(P, q)
    t_fm_m, _ = medtime(n=max(2, nrepeat - 1)) do
        sys1.potential .= 0
        FastMultipole.fmm!(sys1; scalar_potential=true, gradient=false, hessian=false,
            expansion_order=p, leaf_size=nmax, multipole_acceptance=0.5,
            silence_warnings=true)
    end
    sys1.potential .= 0
    FastMultipole.fmm!(sys1; scalar_potential=true, gradient=false, hessian=false,
        expansion_order=p, leaf_size=nmax, multipole_acceptance=0.5,
        silence_warnings=true)
    y_fm_m = copy(sys1.potential[1, :])
    align_sign!(y_fm_m, y_fmm3d)
    row["t_fm_matched_ms"] = 1e3 * t_fm_m
    row["err_fm_matched_vs_fmm3d"] = relerr(y_fm_m, y_fmm3d)

    if n <= 2500
        ydens = dense_laplace3d(P, q)
        row["err_fmm3d_vs_dense"] = relerr(y_fmm3d, ydens)
        row["err_bem_vs_dense"] = relerr(y_bem, ydens)
        row["err_fm_default_vs_dense"] = relerr(y_fm_def, ydens)
        row["err_fm_matched_vs_dense"] = relerr(y_fm_m, ydens)
        row["err_treecode_vs_dense"] = NaN
    end

    return row
end

function print_row(io, r)
    @printf(io,
        "%-6s N=%6d eps=%.0e  p=%2d nmax=%3d | FMM3D %8.2f ms | BEM apply %8.2f ms (plan %7.2f, cold %8.2f, tree %8.2f) | FM def %8.2f ms  match %8.2f | err vs FMM3D: BEM %.2e  tree %.2e  FM %.2e / %.2e\n",
        r["dist"], r["N"], r["eps"], r["p"], r["nmax"],
        r["t_fmm3d_ms"], r["t_bem_apply_ms"], r["t_bem_plan_ms"], r["t_bem_cold_ms"],
        r["t_bem_treecode_ms"], r["t_fm_default_ms"], r["t_fm_matched_ms"],
        r["err_bem_vs_fmm3d"], r["err_bem_treecode_vs_fmm3d"],
        r["err_fm_default_vs_fmm3d"], r["err_fm_matched_vs_fmm3d"])
    if haskey(r, "err_fmm3d_vs_dense")
        @printf(io,
            "         vs dense: FMM3D %.2e  BEM %.2e  tree %.2e  FM %.2e / %.2e\n",
            r["err_fmm3d_vs_dense"], r["err_bem_vs_dense"], r["err_treecode_vs_dense"],
            r["err_fm_default_vs_dense"], r["err_fm_matched_vs_dense"])
    end
    flush(io)
end

function csv_header()
    return join([
        "dist", "N", "eps", "p", "nmax", "threads",
        "t_fmm3d_ms", "t_bem_plan_ms", "t_bem_apply_ms", "t_bem_cold_ms", "t_bem_treecode_ms",
        "t_fm_default_ms", "t_fm_matched_ms",
        "err_bem_vs_fmm3d", "err_bem_cold_vs_fmm3d", "err_bem_treecode_vs_fmm3d",
        "err_fm_default_vs_fmm3d", "err_fm_matched_vs_fmm3d",
        "err_fmm3d_vs_dense", "err_bem_vs_dense", "err_treecode_vs_dense",
        "err_fm_default_vs_dense", "err_fm_matched_vs_dense",
    ], ",")
end

function csv_row(r)
    keys = [
        "dist", "N", "eps", "p", "nmax", "threads",
        "t_fmm3d_ms", "t_bem_plan_ms", "t_bem_apply_ms", "t_bem_cold_ms", "t_bem_treecode_ms",
        "t_fm_default_ms", "t_fm_matched_ms",
        "err_bem_vs_fmm3d", "err_bem_cold_vs_fmm3d", "err_bem_treecode_vs_fmm3d",
        "err_fm_default_vs_fmm3d", "err_fm_matched_vs_fmm3d",
        "err_fmm3d_vs_dense", "err_bem_vs_dense", "err_treecode_vs_dense",
        "err_fm_default_vs_dense", "err_fm_matched_vs_dense",
    ]
    vals = String[]
    for k in keys
        v = get(r, k, "")
        if v isa AbstractFloat
            push!(vals, @sprintf("%.6g", v))
        else
            push!(vals, string(v))
        end
    end
    return join(vals, ",")
end

function main()
    io = open(OUT_LOG, "w")
    csv = open(OUT_CSV, "w")
    println(csv, csv_header())
    info = sprint() do s
        println(s, "Julia ", VERSION, "  nthreads=", Threads.nthreads(),
            "  OMP_NUM_THREADS=", get(ENV, "OMP_NUM_THREADS", "(unset)"))
        println(s, "CPU: ", Sys.cpu_info()[1].model, "  ncpu=", Sys.CPU_THREADS)
        println(s, "BEM.FMM vs FMM3D v2.1.0 vs FastMultipole.jl v2.3.0")
        println(s, "kernel: 1/(4πr) charge-to-potential, sources=targets")
        println(s, "times are median of repeats after one warmup, milliseconds")
    end
    print(stdout, info)
    print(io, info)

    cases = [
        ("cube", 2000, 1e-6, 5),
        ("sphere", 2000, 1e-6, 5),
        ("cube", 10000, 1e-4, 4),
        ("cube", 10000, 1e-6, 4),
        ("sphere", 10000, 1e-6, 4),
        ("cube", 40000, 1e-6, 3),
        ("sphere", 40000, 1e-6, 3),
    ]

    rows = []
    for (dist, n, eps, nrep) in cases
        msg = @sprintf("\n=== %s N=%d eps=%.0e ===\n", dist, n, eps)
        print(stdout, msg); print(io, msg)
        r = run_case(dist=dist, n=n, eps=eps, nrepeat=nrep)
        push!(rows, r)
        print_row(stdout, r)
        print_row(io, r)
        println(csv, csv_row(r))
        flush(csv)
    end

    # potential + gradient at N=10k cube
    msg = "\n=== extra: potential+gradient cube N=10000 eps=1e-6 ===\n"
    print(stdout, msg); print(io, msg)
    rng = Random.default_rng()
    Random.seed!(rng, 20260909)
    n = 10000
    eps = 1e-6
    P = cube_points(rng, n)
    q = randn(rng, n)
    t3, _ = medtime(n=3) do
        FMM3D.lfmm3d(eps, P; charges=q, pg=2)
    end
    plan = FMM.build_laplace3d_plan(P; eps=eps, full_fmm=true)
    y = zeros(n); g = zeros(3, n)
    ta, _ = medtime(n=3) do
        FMM.apply_laplace3d!(plan, y; charges=q, grad=g)
    end
    sys = make_grav(P, q)
    tfm, _ = medtime(n=3) do
        sys.potential .= 0
        FastMultipole.fmm!(sys; scalar_potential=true, gradient=true, hessian=false,
            expansion_order=bem_p(eps), leaf_size=bem_nmax(eps),
            silence_warnings=true)
    end
    v3 = FMM3D.lfmm3d(eps, P; charges=q, pg=2)
    FMM.apply_laplace3d!(plan, y; charges=q, grad=g)
    @printf(stdout, "FMM3D pg=2 %8.2f ms | BEM apply+grad %8.2f ms | FM match+grad %8.2f ms | pot err %.2e  grad err %.2e\n",
        1e3 * t3, 1e3 * ta, 1e3 * tfm, relerr(y, v3.pot), relerr(g, v3.grad))
    @printf(io, "FMM3D pg=2 %8.2f ms | BEM apply+grad %8.2f ms | FM match+grad %8.2f ms | pot err %.2e  grad err %.2e\n",
        1e3 * t3, 1e3 * ta, 1e3 * tfm, relerr(y, v3.pot), relerr(g, v3.grad))

    close(io)
    close(csv)
    println("\nwrote ", OUT_CSV, " and ", OUT_LOG)
end

main()
