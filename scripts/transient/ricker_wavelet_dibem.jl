#=
================================================================================
RICKER WAVELET  —  acoustic source with DIBEM + MMM
================================================================================

Same continuum problem as  FVM testes/ricker_wavelet_trixi.jl:

    p_tt = c² Δp + A R(t) g(x)     on  Ω = (−L, L)²
    ∂p/∂n = 0                      on  ∂Ω   (hard wall)

Zero-phase Ricker  R(τ) = (1 − 2 π² f₀² τ²) exp(−π² f₀² τ²),  τ = t − t₀,
tapered by a unit-mass Gaussian blob g.  Trixi's bundled exact is free-space
Hankel (no walls).  This script also sums Neumann images of that field so
the reference *does* reflect; after t₀+L/c both Trixi and DIBEM should
track the image sum.

Discretization: stationary Laplace FS + DIBEM mass M,
    H p − G q = (M / c²) (p_tt − A R(t) g),
all-Neumann q = 0.  Dense Newmark/Houbolt on raw DIBEM M is unstable
(indefinite mass); the default stepper is MMM (drop ω² ≤ 0 ghosts).

Internal collocation points are the Gmsh cell centroids from `format2d`
(`pontointerno=true`), not a separate Cartesian grid.

Run (from the repo root)

    julia --project=. scripts/transient/ricker_wavelet_dibem.jl

ENV: RICKER_MESHES=16,24,32,48  (ndiv list; default finer sweep)
     RICKER_DT, RICKER_TF, RICKER_SOLVER=mmm|newmark|houbolt, RICKER_QUICK=1.
     Plots/GIF evaluate a PHS(3) interpolant of the collocation field
     (RICKER_NLINE, RICKER_NGRID).

Each mesh is saved under  scripts/transient/ricker_wavelet_dibem/out/ndiv_XXX/.
=#

using DrWatson
@quickactivate :BEM
using LinearAlgebra
using Printf
using StaticArrays
using Statistics
using DelimitedFiles
using SpecialFunctions: besselj0
using Plots

ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")

include(datadir("Laplace", "Laplace_dad.jl"))
include(datadir("Laplace", "potencial_problems.jl"))
include(datadir("Laplace", "wave_propagation.jl"))

# ============================================================================
# 1. Problem data  (defaults match the Trixi script)
# ============================================================================

Base.@kwdef struct RickerProblem
    L::Float64 = 1.5
    c::Float64 = 1.0
    f0::Float64 = 4.0
    t0::Float64 = 0.30
    σ::Float64 = 0.05
    xs::Float64 = 0.0
    ys::Float64 = 0.0
    amp::Float64 = 1.0
    t_end::Float64 = 4.0
    ndiv::Int = 16
    Δt::Float64 = 0.01
end

"""Zero-phase Ricker wavelet R(t − t₀).  Peak value 1 at t = t₀."""
function ricker(t::Real, prob::RickerProblem)
    τ = t - prob.t0
    aτ = (π * prob.f0 * τ)^2
    return (1 - 2 * aτ) * exp(-aτ)
end

"""Antiderivative of the Ricker (up to the spatial envelope).  ∂t[τ exp(−π² f₀² τ²)] = R(τ)."""
function ricker_antiderivative(t::Real, prob::RickerProblem)
    τ = t - prob.t0
    return τ * exp(-(π * prob.f0 * τ)^2)
end

"""Unit-mass Gaussian blob,  ∫ g dA = 1."""
spatial_envelope(x, y, prob::RickerProblem) =
    exp(-((x - prob.xs)^2 + (y - prob.ys)^2) / (2 * prob.σ^2)) /
    (2π * prob.σ^2)

t_wall(prob::RickerProblem) = prob.t0 + prob.L / prob.c
t_return(prob::RickerProblem) = prob.t0 + 2 * prob.L / prob.c

function first_reflection_time(x, y, prob::RickerProblem)
    xs, ys, L = prob.xs, prob.ys, prob.L
    images = ((2L - xs, ys), (-2L - xs, ys), (xs, 2L - ys), (xs, -2L - ys))
    dmin = minimum(hypot(x - ix, y - iy) for (ix, iy) in images)
    return prob.t0 + dmin / prob.c
end

function snapshot_times(prob::RickerProblem)
    tw, tr = t_wall(prob), t_return(prob)
    ts = (max(0.5, tw - 1.0), tw - 0.2, tw + 0.4, tr, prob.t_end)
    return Tuple(unique(min(t, prob.t_end) for t in ts))
end

receiver_points(prob::RickerProblem) = (
    (prob.xs + 0.40, prob.ys),
    (prob.xs + 0.70, prob.ys),
    (prob.xs + 0.50, prob.ys + 0.50),
)

radius_from_source(x, y, prob::RickerProblem) = hypot(x - prob.xs, y - prob.ys)

function mesh_list_from_env()
    quick = get(ENV, "RICKER_QUICK", "0") in ("1", "true", "TRUE")
    raw = get(ENV, "RICKER_MESHES", quick ? "8,12" : "64")
    return Int[parse(Int, strip(s)) for s in split(raw, ',') if !isempty(strip(s))]
end

function from_env(; ndiv=16)
    quick = get(ENV, "RICKER_QUICK", "0") in ("1", "true", "TRUE")
    Δt = parse(Float64, get(ENV, "RICKER_DT", quick ? "0.02" : "0.005"))
    t_end = parse(Float64, get(ENV, "RICKER_TF", quick ? "0.6" : "4.0"))
    return RickerProblem(; ndiv, Δt, t_end)
end

# ============================================================================
# 2. Analytical free-space solution  (Hankel + Duhamel)
# ============================================================================

struct HankelKernel
    k::Vector{Float64}
    Δk::Float64
    Fg::Vector{Float64}
    c::Float64
end

function HankelKernel(prob::RickerProblem; kmax_over_invσ=8.0, nk=2501)
    kmax = kmax_over_invσ / prob.σ
    k = collect(range(0.0, kmax; length=nk))
    Fg = @. prob.amp * exp(-0.5 * (prob.σ * k)^2)
    return HankelKernel(k, k[2] - k[1], Fg, prob.c)
end

function duhamel_response!(I::AbstractVector, kernel::HankelKernel, t::Real,
        prob::RickerProblem)
    fill!(I, 0.0)
    t <= 0 && return I
    nτ = max(16, round(Int, t / 0.002) + 1)
    τ = range(0.0, t; length=nτ)
    Δτ = τ[2] - τ[1]
    Rτ = ricker.(τ, Ref(prob))
    Rτ[1] *= 0.5
    Rτ[end] *= 0.5
    @inbounds for m in eachindex(kernel.k)
        ω = kernel.c * kernel.k[m]
        s = 0.0
        if ω < 1.0e-14
            for j in eachindex(τ)
                s += (t - τ[j]) * Rτ[j]
            end
            I[m] = s * Δτ
        else
            for j in eachindex(τ)
                s += sin(ω * (t - τ[j])) * Rτ[j]
            end
            I[m] = s * Δτ / ω
        end
    end
    return I
end

function hankel_invert(r::Real, kernel::HankelKernel, I::AbstractVector)
    s = 0.0
    @inbounds for m in eachindex(kernel.k)
        km = kernel.k[m]
        s += kernel.Fg[m] * I[m] * besselj0(km * r) * km
    end
    return s * kernel.Δk / (2π)
end

analytical_pressure(r::Real, t::Real, kernel::HankelKernel, I::AbstractVector) =
    hankel_invert(r, kernel, I)

"""
Neumann (hard-wall) images of the source on ``(-L,L)²``.

Even reflections: sources at ``(2nL ± x_s,\\, 2mL ± y_s)``.  Truncated
to images that can have arrived by time `t` (plus a few σ of smear).
When ``x_s=0`` or ``y_s=0`` the ± copies coincide and are kept once.
"""
function image_sources(prob::RickerProblem, t::Real)
    L, xs, ys, c = prob.L, prob.xs, prob.ys, prob.c
    rmax = c * max(t, 0.0) + 12 * prob.σ + 1.0
    N = max(1, ceil(Int, rmax / (2L)) + 1)
    sxs = abs(xs) < 1e-14 ? (1.0,) : (1.0, -1.0)
    sys = abs(ys) < 1e-14 ? (1.0,) : (1.0, -1.0)
    srcs = NTuple{2,Float64}[]
    seen = Set{UInt64}()
    for n in -N:N, m in -N:N, sx in sxs, sy in sys
        ix = 2 * n * L + sx * xs
        iy = 2 * m * L + sy * ys
        h = hash((round(ix; digits=10), round(iy; digits=10)))
        h in seen && continue
        push!(seen, h)
        push!(srcs, (ix, iy))
    end
    return srcs
end

"""Free-space Hankel field plus hard-wall images (method of images)."""
function analytical_pressure_walls(x, y, t, kernel::HankelKernel, I, prob::RickerProblem)
    s = 0.0
    @inbounds for (ix, iy) in image_sources(prob, t)
        s += hankel_invert(hypot(x - ix, y - iy), kernel, I)
    end
    return s
end

# ============================================================================
# 3. Mesh, DIBEM, Newmark
# ============================================================================

function build_dad(prob::RickerProblem)
    nome = @sprintf("ricker_hardwall_n%03d", prob.ndiv)
    msh = mesh_square_hardwall(; ndiv=prob.ndiv, L=prob.L, nome=nome, show=false)
    dad = format2d(msh, Laplace(1.0); tipo=1, pontointerno=true)
    dad.ni > 0 || error("format2d returned no internal (cell-centroid) points")
    return dad
end

"""Regular centroid lattice `(xs, ys, loc)` with `loc[j,i]` the internal index, or `nothing`."""
function centroid_grid(dad)
    pts = collect(dad.internalNodes)
    isempty(pts) && return nothing
    xr = [round(p[1]; digits=10) for p in pts]
    yr = [round(p[2]; digits=10) for p in pts]
    xs = sort!(unique(xr))
    ys = sort!(unique(yr))
    nx, ny = length(xs), length(ys)
    nx * ny == length(pts) || return nothing
    xmap = Dict{Float64,Int}(x => i for (i, x) in enumerate(xs))
    ymap = Dict{Float64,Int}(y => j for (j, y) in enumerate(ys))
    loc = zeros(Int, ny, nx)
    for k in eachindex(pts)
        loc[ymap[yr[k]], xmap[xr[k]]] = k
    end
    any(iszero, loc) && return nothing
    return xs, ys, loc
end

function nearest_sample(pts, vals, x, y)
    best = 1
    d2 = Inf
    @inbounds for k in eachindex(pts)
        dx = pts[k][1] - x
        dy = pts[k][2] - y
        d = dx * dx + dy * dy
        if d < d2
            d2 = d
            best = k
        end
    end
    return vals[best]
end

"""PHS interpolant of the collocation field, for plots/GIF on a fine grid."""
struct FieldInterp
    rbf::RBF
    xline::Vector{Float64}
    qline::Vector{Point2D}
    xfine::Vector{Float64}
    yfine::Vector{Float64}
    qgrid::Vector{Point2D}
    nx::Int
    ny::Int
    qrec::Vector{Point2D}
end

function FieldInterp(dad, prob::RickerProblem)
    nline = parse(Int, get(ENV, "RICKER_NLINE", "301"))
    ngrid = parse(Int, get(ENV, "RICKER_NGRID", "201"))
    pts = collect(all_points(dad))
    rbf = RBF(pts, PHS(3; poly_deg=1))
    xline = collect(range(-prob.L, prob.L; length=nline))
    qline = Point2D[SA[x, 0.0] for x in xline]
    xfine = collect(range(-prob.L, prob.L; length=ngrid))
    yfine = copy(xfine)
    qgrid = Point2D[SA[x, y] for y in yfine for x in xfine]
    qrec = Point2D[SA[float(p[1]), float(p[2])] for p in receiver_points(prob)]
    return FieldInterp(rbf, xline, qline, xfine, yfine, qgrid, ngrid, ngrid, qrec)
end

eval_line(ip::FieldInterp, u) = rbf_evaluate(ip.rbf, ip.qline, u)
eval_rec(ip::FieldInterp, u) = rbf_evaluate(ip.rbf, ip.qrec, u)
function eval_grid(ip::FieldInterp, u)
    v = rbf_evaluate(ip.rbf, ip.qgrid, u)
    return reshape(v, ip.nx, ip.ny)'   # heatmap Z[y, x]
end

function nodal_envelope(dad, prob::RickerProblem)
    g = zeros(dad.nt)
    @inbounds for i in 1:dad.nt
        p = point(dad, i)
        g[i] = spatial_envelope(p[1], p[2], prob)
    end
    return g
end

function nearest_index(dad, xy)
    pts = all_points(dad)
    return argmin(i -> (pts[i][1] - xy[1])^2 + (pts[i][2] - xy[2])^2, eachindex(pts))
end

function centreline_indices(dad; y0=0.0)
    pts = all_points(dad)
    # internals on the Cartesian grid sit exactly on y = 0 when n_int is odd
    idx = Int[]
    for i in (dad.n + 1):dad.nt
        abs(pts[i][2] - y0) <= 1e-9 && push!(idx, i)
    end
    if isempty(idx)
        # fallback: nearest-to-y0 per unique x among internals
        byx = Dict{Float64,Tuple{Int,Float64}}()
        for i in (dad.n + 1):dad.nt
            x, y = pts[i][1], pts[i][2]
            d = abs(y - y0)
            if !haskey(byx, x) || d < byx[x][2]
                byx[x] = (i, d)
            end
        end
        idx = [byx[x][1] for x in sort!(collect(keys(byx)))]
    else
        sort!(idx; by=i -> pts[i][1])
    end
    return idx
end

function nearest_time_index(tgrid, t)
    return argmin(i -> abs(tgrid[i] - t), eachindex(tgrid))
end

# ============================================================================
# 4. Plots
# ============================================================================

function plot_wavelet(prob::RickerProblem, outdir)
    t = range(0.0, prob.t_end; length=800)
    R = ricker.(t, Ref(prob))
    S = ricker_antiderivative.(t, Ref(prob))
    plt = plot(
        t, R;
        xlabel="t", ylabel="amplitude",
        title="Ricker wavelet   f₀ = $(prob.f0),   t₀ = $(prob.t0)",
        label="R(t)  (wave-equation source)",
        lw=2.2, c=:black, size=(900, 420), legend=:topright,
    )
    plot!(plt, t, S; lw=1.8, c=:steelblue,
        label="∫ R  (injected into p'_t)")
    vline!(plt, [prob.t0]; ls=:dash, c=:gray, label="t₀")
    savefig(plt, joinpath(outdir, "ricker_wavelet.png"))
    return nothing
end

function plot_source_blob(prob::RickerProblem, outdir)
    n = 201
    x = range(-prob.L, prob.L; length=n)
    G = [spatial_envelope(xi, yj, prob) for xi in x, yj in x]
    plt = heatmap(
        x, x, G';
        xlabel="x", ylabel="y",
        title="source envelope g(x)   σ = $(prob.σ)",
        aspect_ratio=:equal, c=:viridis, size=(720, 640),
    )
    savefig(plt, joinpath(outdir, "source_envelope.png"))
    return nothing
end

function _locate_bracket(x, xs)
    n = length(xs)
    x <= xs[1] && return 1, 0.0
    x >= xs[n] && return n - 1, 1.0
    i = searchsortedlast(xs, x)
    i = clamp(i, 1, n - 1)
    dx = xs[i + 1] - xs[i]
    return i, dx == 0 ? 0.0 : (x - xs[i]) / dx
end

"""Bilinear sample. `Z[j,i]` lives at `(xs[i], ys[j])`."""
function bilinear_sample(xs, ys, Z, x, y)
    i, tx = _locate_bracket(x, xs)
    j, ty = _locate_bracket(y, ys)
    z00 = Z[j, i]
    z10 = Z[j, i + 1]
    z01 = Z[j + 1, i]
    z11 = Z[j + 1, i + 1]
    return (1 - ty) * ((1 - tx) * z00 + tx * z10) + ty * ((1 - tx) * z01 + tx * z11)
end

function plot_snapshots(prob::RickerProblem, dad, kernel::HankelKernel, outdir,
        ip::FieldInterp)
    tgrid = dad.t isa AbstractRange ? collect(dad.t) : collect(dad.t)
    xs = ip.xline
    I = zeros(length(kernel.k))
    plt_cut = plot(
        xlabel="x", ylabel="p'(x, y=0, t)",
        title="Centreline  —  DIBEM (solid) vs. hard-wall exact (dashed)",
        size=(900, 500), legend=:topright,
    )
    plot!(plt_cut, Float64[], Float64[]; c=:black, ls=:dash, label="exact")
    plt_err = plot(
        xlabel="x", ylabel="|p'_DIBEM − p'_exact|",
        title="Centreline error",
        yscale=:log10, size=(900, 420), legend=:topright,
    )

    csv = joinpath(outdir, "centreline.csv")
    snaps = snapshot_times(prob)
    clim = 0.0
    fields = Tuple{Float64,Vector{Float64}}[]
    open(csv, "w") do io
        println(io, "t,x,p_dibem,p_exact,error")
        for ts in snaps
            it = nearest_time_index(tgrid, ts)
            t = tgrid[it]
            u = view(dad.T, :, it)
            p_num = eval_line(ip, u)
            duhamel_response!(I, kernel, t, prob)
            p_ex = [analytical_pressure_walls(x, 0.0, t, kernel, I, prob) for x in xs]
            err = p_num .- p_ex
            tlab = "t = $(round(t; digits=2))"
            plot!(plt_cut, xs, p_num; lw=1.8, label=tlab)
            plot!(plt_cut, xs, p_ex; lw=1.6, ls=:dash, label=false)
            plot!(plt_err, xs, max.(abs.(err), 1.0e-16); lw=1.6, label=tlab)
            for i in eachindex(xs)
                @printf(io, "%.12e,%.12e,%.12e,%.12e,%.12e\n",
                    t, xs[i], p_num[i], p_ex[i], err[i])
            end
            clim = max(clim, maximum(abs, u))
            push!(fields, (t, collect(u)))
        end
    end
    clim = max(clim, 1.0e-12)
    savefig(plt_cut, joinpath(outdir, "centreline_pressure.png"))
    savefig(plt_err, joinpath(outdir, "centreline_error.png"))

    for f in readdir(outdir; join=true)
        bn = basename(f)
        startswith(bn, "heatmap_t") && endswith(bn, ".png") && rm(f)
    end
    for (k, (t, u)) in enumerate(fields)
        Zf = eval_grid(ip, u)
        plt = heatmap(
            ip.xfine, ip.yfine, Zf;
            xlabel="x", ylabel="y",
            title="p'(x, y)   t = $(round(t; digits=2))",
            clims=(-clim, clim), c=:seismic, aspect_ratio=:equal, size=(720, 640),
            legend=false,
        )
        scatter!(plt, [prob.xs], [prob.ys];
            ms=6, mc=:black, msw=0, label=false)
        ttag = replace(@sprintf("%.2f", t), "." => "p")
        savefig(plt, joinpath(outdir, "heatmap_t$(k)_$(ttag).png"))
    end
    return nothing
end

function plot_receivers(prob::RickerProblem, tgrid, traces, coords, labels,
        p_exact, outdir)
    colors = palette(:tab10)
    plt_src = plot(
        tgrid, ricker.(tgrid, Ref(prob));
        xlabel="t", ylabel="R(t)",
        title="Source wavelet",
        label="Ricker R(t)",
        lw=2.2, c=:black, legend=:topright,
    )
    vline!(plt_src, [prob.t0]; ls=:dash, c=:gray, label="t₀")

    plt_rec = plot(
        xlabel="t", ylabel="p'",
        title="Receiver traces  —  DIBEM (solid) vs. hard-wall exact (dashed)",
        legend=:outertopright,
    )
    plt_err = plot(
        xlabel="t", ylabel="p'_DIBEM − p'_exact",
        title="DIBEM − hard-wall exact  (images; should stay small after the bounce)",
        legend=:outertopright,
    )
    for k in eachindex(traces)
        c = colors[k]
        plot!(plt_rec, tgrid, traces[k];
            lw=1.8, c=c, label=labels[k] * "  DIBEM")
        plot!(plt_rec, tgrid, p_exact[k];
            lw=1.6, c=c, ls=:dash, label=labels[k] * "  exact")
        plot!(plt_err, tgrid, traces[k] .- p_exact[k];
            lw=1.4, c=c, label=labels[k])
        r = hypot(coords[k][1] - prob.xs, coords[k][2] - prob.ys)
        vline!(plt_rec, [prob.t0 + r / prob.c]; ls=:dot, c=:gray, label=false)
        vline!(plt_rec, [first_reflection_time(coords[k][1], coords[k][2], prob)];
            ls=:dash, c=:indianred, label=false)
    end
    vline!(plt_rec, [t_wall(prob)]; ls=:dot, c=:black, label="peak hits wall")

    plt = plot(plt_src, plt_rec, plt_err; layout=(3, 1), size=(900, 980))
    savefig(plt, joinpath(outdir, "receiver_traces.png"))
    return nothing
end

function add_peak_markers!(plt, t, prob::RickerProblem)
    rp = prob.c * max(0.0, t - prob.t0)
    L = prob.L
    if 0 < rp < L
        vline!(plt, [-rp, rp]; ls=:dash, c=:indianred, label="outgoing peak")
    elseif rp >= L
        xR = 2L - rp
        xL = -2L + rp
        xs = Float64[]
        -L < xR < L && push!(xs, xR)
        -L < xL < L && push!(xs, xL)
        if !isempty(xs)
            vline!(plt, xs; ls=:dash, c=:indianred, label="reflected peak")
        end
    end
    return plt
end

function write_centreline_gif(prob::RickerProblem, dad, outdir, ip::FieldInterp;
        fps=20, nframes=160)
    tgrid = dad.t isa AbstractRange ? collect(dad.t) : collect(dad.t)
    xs = ip.xline
    n = length(tgrid)
    n >= 2 || error("no time history")
    nframes = min(nframes, n)
    idx = unique(round.(Int, range(1, n; length=nframes)))
    ymax = 0.0
    plines = Vector{Vector{Float64}}(undef, length(idx))
    for (k, i) in enumerate(idx)
        plines[k] = eval_line(ip, view(dad.T, :, i))
        ymax = max(ymax, maximum(abs, plines[k]))
    end
    ymax = max(1.15 * ymax, 1.0e-6)

    anim = Animation()
    for (k, i) in enumerate(idx)
        t = tgrid[i]
        plt = plot(
            xs, plines[k];
            xlabel="x", ylabel="p'(x, y = 0)",
            title=@sprintf("Centreline pressure   t = %.3f", t),
            xlims=(-prob.L, prob.L),
            ylims=(-ymax, ymax),
            lw=2.2, c=:steelblue, label="DIBEM (PHS interp)",
            size=(900, 420), legend=:topright,
        )
        vline!(plt, [0.0]; ls=:dot, c=:gray, label="source")
        add_peak_markers!(plt, t, prob)
        frame(anim, plt)
    end
    outfile = joinpath(outdir, "centreline_pressure.gif")
    try
        gif(anim, outfile; fps=fps)
    catch err
        @warn "Plots.gif failed" exception = err
    end
    @info "wrote $(outfile)  ($(length(idx)) frames @ $(fps) fps)"
    return outfile
end

function rms_linf(a, b)
    e = a .- b
    return sqrt(sum(abs2, e) / length(e)), maximum(abs, e)
end

function print_error_table(prob, tgrid, traces, labels, p_exact, dad, kernel, ip)
    tw = t_wall(prob)
    println()
    println("="^78)
    println("DIBEM  vs.  hard-wall Hankel (method of images)")
    println("  images of the Gaussian blob, Neumann copies on (-L,L)²")
    println("  first wall hit at t₀ + L/c = $(round(tw; digits=2))")
    println("="^78)
    @printf("%-36s %12s %12s\n", "probe", "RMS", "Linf")
    for k in eachindex(traces)
        pre = [i for i in eachindex(tgrid) if tgrid[i] < tw]
        post = [i for i in eachindex(tgrid) if tgrid[i] >= tw]
        rms, linf = rms_linf(traces[k][pre], p_exact[k][pre])
        @printf("%-36s %12.3e %12.3e\n",
            "receiver $(labels[k])  t<wall", rms, linf)
        if !isempty(post)
            rms, linf = rms_linf(traces[k][post], p_exact[k][post])
            @printf("%-36s %12.3e %12.3e\n",
                "receiver $(labels[k])  t≥wall", rms, linf)
        end
    end
    println("-"^78)
    xs = ip.xline
    I = zeros(length(kernel.k))
    for ts in snapshot_times(prob)
        it = nearest_time_index(tgrid, ts)
        t = tgrid[it]
        p_num = eval_line(ip, view(dad.T, :, it))
        duhamel_response!(I, kernel, t, prob)
        p_ex = [analytical_pressure_walls(x, 0.0, t, kernel, I, prob) for x in xs]
        rms, linf = rms_linf(p_num, p_ex)
        tag = t < tw ? "" : "  (after wall)"
        @printf("%-36s %12.3e %12.3e\n",
            "centreline t=$(round(t; digits=2))$tag", rms, linf)
    end
    println("="^78)
    return nothing
end

# ============================================================================
# 5. Compare with Trixi DGSEM output
# ============================================================================

function interp1(x::AbstractVector, y::AbstractVector, xq::AbstractVector)
    n = length(x)
    out = Vector{Float64}(undef, length(xq))
    @inbounds for (k, q) in enumerate(xq)
        if q <= x[1]
            out[k] = y[1]
        elseif q >= x[n]
            out[k] = y[n]
        else
            i = searchsortedlast(x, q)
            i = clamp(i, 1, n - 1)
            den = x[i + 1] - x[i]
            t = den == 0 ? 0.0 : (q - x[i]) / den
            out[k] = (1 - t) * y[i] + t * y[i + 1]
        end
    end
    return out
end

function load_trixi_receivers(path)
    data, hdr = readdlm(path, ','; header=true)
    header = String.(vec(hdr))
    t = Float64.(data[:, 1])
    # columns: t, (p_trixi, p_exact, error) × 3 receivers
    traces = Vector{Vector{Float64}}(undef, 3)
    exact = Vector{Vector{Float64}}(undef, 3)
    for k in 1:3
        traces[k] = Float64.(data[:, 3k - 1])
        exact[k] = Float64.(data[:, 3k])
    end
    return t, traces, exact, header
end

function load_trixi_centreline(path)
    data, _ = readdlm(path, ','; header=true)
    t = Float64.(data[:, 1])
    x = Float64.(data[:, 2])
    p = Float64.(data[:, 3])
    pex = Float64.(data[:, 4])
    snaps = unique(t)
    xs = Dict{Float64,Vector{Float64}}()
    ps = Dict{Float64,Vector{Float64}}()
    pexs = Dict{Float64,Vector{Float64}}()
    for ts in snaps
        mask = t .== ts
        ord = sortperm(x[mask])
        xs[ts] = x[mask][ord]
        ps[ts] = p[mask][ord]
        pexs[ts] = pex[mask][ord]
    end
    return snaps, xs, ps, pexs
end

function plot_compare_trixi(prob, tgrid, traces, labels, p_exact, dad, outdir, trixi_out,
        ip::FieldInterp, kernel::HankelKernel)
    rec_csv = joinpath(trixi_out, "receivers.csv")
    cl_csv = joinpath(trixi_out, "centreline.csv")
    isfile(rec_csv) && isfile(cl_csv) || begin
        @warn "Trixi output not found; skip comparison" trixi_out
        return nothing
    end

    tT, pT, eT, _ = load_trixi_receivers(rec_csv)
    colors = palette(:tab10)

    plt_src = plot(
        tgrid, ricker.(tgrid, Ref(prob));
        xlabel="t", ylabel="R(t)",
        title="Source wavelet",
        label="Ricker R(t)",
        lw=2.2, c=:black, legend=:topright,
    )
    vline!(plt_src, [prob.t0]; ls=:dash, c=:gray, label="t₀")

    plt_rec = plot(
        xlabel="t", ylabel="p'",
        title="Receiver traces  —  Trixi (solid)  DIBEM (dash-dot)  hard-wall exact (dashed)",
        legend=:outertopright,
    )
    plt_err = plot(
        xlabel="t", ylabel="p' − p'_exact",
        title="Error vs. hard-wall exact  (Trixi solid, DIBEM dash-dot)",
        legend=:outertopright,
    )
    pT_on_d = Vector{Vector{Float64}}(undef, length(traces))
    for k in eachindex(traces)
        c = colors[k]
        pT_on_d[k] = interp1(tT, pT[k], tgrid)
        plot!(plt_rec, tT, pT[k]; lw=1.8, c=c, label=labels[k] * "  Trixi")
        plot!(plt_rec, tgrid, traces[k]; lw=1.7, c=c, ls=:dashdot, label=labels[k] * "  DIBEM")
        plot!(plt_rec, tgrid, p_exact[k]; lw=1.4, c=c, ls=:dash, label=labels[k] * "  exact")
        plot!(plt_err, tgrid, pT_on_d[k] .- p_exact[k];
            lw=1.4, c=c, label=labels[k] * "  Trixi")
        plot!(plt_err, tgrid, traces[k] .- p_exact[k];
            lw=1.4, c=c, ls=:dashdot, label=labels[k] * "  DIBEM")
    end
    pts = receiver_points(prob)
    for (k, pt) in enumerate(pts)
        vline!(plt_rec, [prob.t0 + hypot(pt[1] - prob.xs, pt[2] - prob.ys) / prob.c];
            ls=:dot, c=:gray, label=false)
        vline!(plt_rec, [first_reflection_time(pt[1], pt[2], prob)];
            ls=:dash, c=:indianred, label=false)
    end
    vline!(plt_rec, [t_wall(prob)]; ls=:dot, c=:black, label="peak hits wall")
    plt = plot(plt_src, plt_rec, plt_err; layout=(3, 1), size=(900, 980))
    savefig(plt, joinpath(outdir, "compare_receivers.png"))

    snapsT, xsT, pTcl, pexT = load_trixi_centreline(cl_csv)
    tD = dad.t isa AbstractRange ? collect(dad.t) : collect(dad.t)
    nsnap = length(snapsT)
    panels = Plots.Plot[]
    tw = t_wall(prob)
    println()
    println("="^78)
    println("Trixi DGSEM  vs.  DIBEM  vs.  hard-wall Hankel (images)")
    println("="^78)
    @printf("%-36s %12s %12s %12s\n", "probe", "Trixi RMS", "DIBEM RMS", "DIBEM−Trixi")
    for k in eachindex(traces)
        preT = [i for i in eachindex(tT) if tT[i] < tw]
        postT = [i for i in eachindex(tT) if tT[i] >= tw]
        pre = [i for i in eachindex(tgrid) if tgrid[i] < tw]
        post = [i for i in eachindex(tgrid) if tgrid[i] >= tw]
        rmsT, _ = rms_linf(pT_on_d[k][pre], p_exact[k][pre])
        rmsD, _ = rms_linf(traces[k][pre], p_exact[k][pre])
        rmsDT, _ = rms_linf(traces[k][pre], pT_on_d[k][pre])
        @printf("%-36s %12.3e %12.3e %12.3e\n",
            "receiver $(labels[k])  t<wall", rmsT, rmsD, rmsDT)
        if !isempty(post)
            rmsT, _ = rms_linf(pT_on_d[k][post], p_exact[k][post])
            rmsD, _ = rms_linf(traces[k][post], p_exact[k][post])
            rmsDT, _ = rms_linf(traces[k][post], pT_on_d[k][post])
            @printf("%-36s %12.3e %12.3e %12.3e\n",
                "receiver $(labels[k])  t≥wall", rmsT, rmsD, rmsDT)
        end
    end
    println("-"^78)

    plt_err_cl = plot(
        xlabel="x", ylabel="|p' − p'_exact|",
        title="Centreline error vs. hard-wall exact",
        yscale=:log10, size=(900, 420), legend=:outertopright,
    )
    Icl = zeros(length(kernel.k))
    for (k, ts) in enumerate(snapsT)
        it = nearest_time_index(tD, ts)
        t = tD[it]
        xT = xsT[ts]
        pTv = pTcl[ts]
        duhamel_response!(Icl, kernel, t, prob)
        pex = [analytical_pressure_walls(x, 0.0, t, kernel, Icl, prob) for x in xT]
        qT = Point2D[SA[x, 0.0] for x in xT]
        pD_on_T = rbf_evaluate(ip.rbf, qT, view(dad.T, :, it))
        tlab = "t = $(round(t; digits=2))"
        plt = plot(
            xlabel="x", ylabel="p'(x, y=0)",
            title=tlab,
            legend=k == 1 ? :topright : false,
            size=(500, 320),
        )
        plot!(plt, xT, pTv; lw=1.8, c=:steelblue, label="Trixi")
        plot!(plt, xT, pD_on_T; lw=1.7, c=:darkorange, ls=:dashdot, label="DIBEM")
        plot!(plt, xT, pex; lw=1.4, c=:black, ls=:dash, label="exact")
        push!(panels, plt)
        plot!(plt_err_cl, xT, max.(abs.(pTv .- pex), 1e-16);
            lw=1.5, label=tlab * "  Trixi")
        plot!(plt_err_cl, xT, max.(abs.(pD_on_T .- pex), 1e-16);
            lw=1.5, ls=:dashdot, label=tlab * "  DIBEM")
        rmsT, linfT = rms_linf(pTv, pex)
        rmsD, linfD = rms_linf(pD_on_T, pex)
        rmsDT, _ = rms_linf(pD_on_T, pTv)
        tag = t < tw ? "" : "  (after wall)"
        @printf("%-36s %12.3e %12.3e %12.3e\n",
            "centreline t=$(round(t; digits=2))$tag", rmsT, rmsD, rmsDT)
    end
    println("="^78)
    plt_cl = plot(panels...; layout=(nsnap, 1), size=(900, 280 * nsnap),
        plot_title="Centreline  —  Trixi vs. DIBEM vs. hard-wall exact")
    savefig(plt_cl, joinpath(outdir, "compare_centreline.png"))
    savefig(plt_err_cl, joinpath(outdir, "compare_centreline_error.png"))

    csv_cmp = joinpath(outdir, "compare_receivers.csv")
    open(csv_cmp, "w") do io
        write(io, "t")
        for lab in labels
            tag = replace(lab, r"[^0-9A-Za-z]+" => "_")
            write(io, ",p_trixi$(tag),p_dibem$(tag),p_exact$(tag)")
        end
        println(io)
        for i in eachindex(tgrid)
            @printf(io, "%.12e", tgrid[i])
            for k in eachindex(traces)
                @printf(io, ",%.12e,%.12e,%.12e",
                    pT_on_d[k][i], traces[k][i], p_exact[k][i])
            end
            println(io)
        end
    end
    println("wrote comparison figures to $(outdir)")
    tw = t_wall(prob)
    pre = [i for i in eachindex(tgrid) if tgrid[i] < tw]
    preT = [i for i in eachindex(tT) if tT[i] < tw]
    recv_dibem = Float64[rms_linf(traces[k][pre], p_exact[k][pre])[1] for k in eachindex(traces)]
    recv_trixi = Float64[rms_linf(pT[k][preT], eT[k][preT])[1] for k in eachindex(traces)]
    recv_vs_trixi = Float64[rms_linf(traces[k][pre], pT_on_d[k][pre])[1] for k in eachindex(traces)]
    return (;
        recv_dibem_rms=recv_dibem,
        recv_trixi_rms=recv_trixi,
        recv_dibem_vs_trixi=recv_vs_trixi,
        recv_dibem_mean=mean(recv_dibem),
        recv_trixi_mean=mean(recv_trixi),
    )
end

# ============================================================================
# 6. One mesh + sweep
# ============================================================================

function run_case(prob::RickerProblem, outdir, kernel, trixi_out; solver="mmm")
    mkpath(outdir)
    dx = 2 * prob.L / (prob.ndiv - 1)
    λ = prob.c / prob.f0
    println()
    println("="^78)
    println(" mesh ndiv=$(prob.ndiv)  Δx=$(round(dx; digits=4))  points/λ ≈ $(round(λ / dx; digits=2))")
    println("="^78)

    plot_wavelet(prob, outdir)
    plot_source_blob(prob, outdir)

    dad = build_dad(prob)
    println(dad)
    @info "format2d internals (cell centroids)" ni = dad.ni n = dad.n nt = dad.nt

    t_asm = @elapsed begin
        H_G_full_direct(dad; npg=10, threaded=false)
        DIBEM(dad; rbf=PHS(3; poly_deg=1), npg=12)
    end
    println("  assembly+DIBEM: $(round(t_asm; digits=2)) s")

    if prob.c != 1
        dad.M ./= prob.c^2
    end

    # Wave:  p_tt = c² Δp + R(t) g(x)
    # DIBEM: H p − G q = M Δp = (M/c²)(p_tt − R g)
    # q=0:   (H − α M) p = −M (R g) + Newmark/Houbolt history.
    # MMM:   M ü + K u = M (R g), then ÿ + ω² y = Φ̃ᵀ (R g).
    nneg = dad.nt <= 1200 ? count(<( -1e-8), real.(eigvals(Matrix(dad.M)))) : -1
    @printf("  DIBEM M: %d×%d  nneg(M)=%s  ‖M‖_F=%.3e\n",
        size(dad.M, 1), size(dad.M, 2), nneg < 0 ? "skip" : string(nneg), norm(dad.M))

    g = nodal_envelope(dad, prob)
    f_body = t -> (prob.amp * ricker(t, prob)) .* g
    println("  load: f_body(t) = R(t) g(x)  at $(dad.nt) collocation points")
    println("        Newmark/Houbolt apply  −M f_body  on the BEM residual")
    println("        MMM applies  f = M f_body  then  Φ̃'(M\\f) = Φ̃' f_body")

    t_sol = @elapsed if solver == "mmm"
        Mg = dad.M * (prob.amp .* g)
        solve_mmm!(dad, prob.Δt, prob.t_end; f=t -> ricker(t, prob) .* Mg)
    elseif solver == "newmark"
        solve_Newmark(dad, prob.Δt, prob.t_end; force=f_body)
    elseif solver == "houbolt"
        solve_Houbolt(dad, prob.Δt, prob.t_end; force=f_body)
    else
        error("RICKER_SOLVER must be mmm|newmark|houbolt (got $solver)")
    end
    tgrid = dad.t isa AbstractRange ? collect(dad.t) : collect(dad.t)
    println("  $solver: $(round(t_sol; digits=2)) s, steps=$(length(tgrid))")
    nmodes = 0
    ω1 = NaN
    if solver == "mmm" && has_cache(dad, :modal_basis)
        b = dad.modal_basis
        nmodes = length(b.ω)
        ω1 = b.ω[1]
        @printf("  MMM modes: %d  ω₁=%.4f  ω_max=%.2f  (Neumann square ω₁₀=π/(2L)=%.4f)\n",
            nmodes, ω1, b.ω[end], π / (2 * prob.L))
    end
    ok = all(isfinite, dad.T)
    mx = ok ? maximum(abs, dad.T) : NaN
    @info "solution finite" ok maxabs = mx

    t_rbf = @elapsed ip = FieldInterp(dad, prob)
    println("  PHS interpolant: $(length(ip.rbf.x)) centres, line=$(length(ip.xline)), grid=$(ip.nx)×$(ip.ny)  ($(round(t_rbf; digits=2)) s)")

    pts_req = receiver_points(prob)
    labels = ["($(p[1]), $(p[2]))" for p in pts_req]
    coords = [(p[1], p[2]) for p in pts_req]
    traces = [zeros(length(tgrid)) for _ in pts_req]
    for i in eachindex(tgrid)
        pr = eval_rec(ip, view(dad.T, :, i))
        for k in eachindex(pr)
            traces[k][i] = pr[k]
        end
    end
    for (lab, pt) in zip(labels, pts_req)
        @info "receiver $lab  PHS-interpolated at $pt"
    end

    I = zeros(length(kernel.k))
    p_exact = [zeros(length(tgrid)) for _ in traces]
    radii = [radius_from_source(xy[1], xy[2], prob) for xy in coords]
    for (i, t) in enumerate(tgrid)
        duhamel_response!(I, kernel, t, prob)
        for k in eachindex(radii)
            p_exact[k][i] = analytical_pressure_walls(coords[k][1], coords[k][2], t, kernel, I, prob)
        end
    end

    plot_snapshots(prob, dad, kernel, outdir, ip)
    plot_receivers(prob, tgrid, traces, coords, labels, p_exact, outdir)
    write_centreline_gif(prob, dad, outdir, ip)
    print_error_table(prob, tgrid, traces, labels, p_exact, dad, kernel, ip)

    open(joinpath(outdir, "ricker_source.csv"), "w") do io
        println(io, "t,ricker,antiderivative")
        for t in range(0.0, prob.t_end; length=1001)
            @printf(io, "%.12e,%.12e,%.12e\n",
                t, ricker(t, prob), ricker_antiderivative(t, prob))
        end
    end

    open(joinpath(outdir, "receivers.csv"), "w") do io
        write(io, "t")
        for lab in labels
            tag = replace(lab, r"[^0-9A-Za-z]+" => "_")
            write(io, ",p_dibem$(tag),p_exact$(tag),error$(tag)")
        end
        println(io)
        for i in eachindex(tgrid)
            @printf(io, "%.12e", tgrid[i])
            for k in eachindex(traces)
                @printf(io, ",%.12e,%.12e,%.12e",
                    traces[k][i], p_exact[k][i], traces[k][i] - p_exact[k][i])
            end
            println(io)
        end
    end

    cmp = plot_compare_trixi(prob, tgrid, traces, labels, p_exact, dad, outdir, trixi_out, ip, kernel)
    println("wrote figures and CSVs to $(outdir)")
    return (;
        ndiv=prob.ndiv, n=dad.n, ni=dad.ni, nt=dad.nt, dx=dx, nneg=nneg,
        ok=ok, maxabs=mx, nmodes=nmodes, ω1=ω1, solver=solver,
        t_asm=t_asm, t_sol=t_sol, cmp=cmp, labels=labels,
    )
end

function write_summary(rows, labels, outdir)
    csv = joinpath(outdir, "summary.csv")
    open(csv, "w") do io
        write(io, "solver,ndiv,n,ni,nt,dx,nneg,ok,maxabs,nmodes,omega1,t_asm,t_sol,recv_dibem_mean,recv_trixi_mean")
        for lab in labels
            tag = replace(lab, r"[^0-9A-Za-z]+" => "_")
            write(io, ",dibem$(tag),trixi$(tag),dibem_vs_trixi$(tag)")
        end
        println(io)
        for r in rows
            c = r.cmp
            @printf(io, "%s,%d,%d,%d,%d,%.8e,%d,%d,%.8e,%d,%.8e,%.4f,%.4f",
                r.solver, r.ndiv, r.n, r.ni, r.nt, r.dx, r.nneg, r.ok ? 1 : 0, r.maxabs,
                r.nmodes, r.ω1, r.t_asm, r.t_sol)
            if c === nothing
                write(io, ",NaN,NaN")
                for _ in labels
                    write(io, ",NaN,NaN,NaN")
                end
            else
                @printf(io, ",%.8e,%.8e", c.recv_dibem_mean, c.recv_trixi_mean)
                for k in eachindex(labels)
                    @printf(io, ",%.8e,%.8e,%.8e",
                        c.recv_dibem_rms[k], c.recv_trixi_rms[k], c.recv_dibem_vs_trixi[k])
                end
            end
            println(io)
        end
    end

    ok = filter(r -> r.ok && r.cmp !== nothing, rows)
    if length(ok) >= 1
        plt = plot(
            xlabel="Δx (cell size)", ylabel="RMS  t < wall  (vs. Hankel exact)",
            title="DIBEM mesh sweep vs. Trixi / exact",
            xscale=:log10, yscale=:log10, size=(900, 520), legend=:bottomright,
        )
        dxs = [r.dx for r in ok]
        plot!(plt, dxs, [r.cmp.recv_dibem_mean for r in ok];
            marker=:circle, lw=2, c=:darkorange, label="DIBEM vs exact")
        plot!(plt, dxs, [r.cmp.recv_trixi_mean for r in ok];
            marker=:square, lw=2, c=:steelblue, label="Trixi vs exact")
        savefig(plt, joinpath(outdir, "convergence.png"))
    end
    println("wrote $(csv)")
    return csv
end

function main()
    meshes = mesh_list_from_env()
    solver = lowercase(get(ENV, "RICKER_SOLVER", "newmark"))
    root = joinpath(@__DIR__, "ricker_wavelet_dibem", "out")
    mkpath(root)
    trixi_out = get(ENV, "TRIXI_OUT",
        "/data/OneDrive/pesquisa/FVM testes/ricker_wavelet/out")

    base = from_env(; ndiv=meshes[1])
    λ = base.c / base.f0
    println()
    println("Ricker wavelet in DIBEM  (format2d cell centroids + $solver)")
    println("  Ω = (−$(base.L), $(base.L))² ,  c = $(base.c)")
    println("  f₀ = $(base.f0) ,  t₀ = $(base.t0) ,  σ = $(base.σ) ,  λ = $(λ)")
    println("  meshes ndiv = $(meshes)")
    println("  Δt = $(base.Δt) ,  t_end = $(base.t_end)")
    println("  peak hits wall at t₀ + L/c = $(round(t_wall(base); digits=2))")
    println()

    kernel = HankelKernel(base)
    @info "Hankel k-grid" kmax = kernel.k[end] nk = length(kernel.k)
    plot_wavelet(base, root)
    plot_source_blob(base, root)

    rows = []
    labels = ["($(p[1]), $(p[2]))" for p in receiver_points(base)]
    for ndiv in meshes
        prob = from_env(; ndiv=ndiv)
        tag = solver == "mmm" ? @sprintf("ndiv_%03d", ndiv) :
            @sprintf("ndiv_%03d_%s", ndiv, solver)
        outdir = joinpath(root, tag)
        try
            push!(rows, run_case(prob, outdir, kernel, trixi_out; solver=solver))
        catch e
            @error "mesh ndiv=$ndiv failed" exception = (e, catch_backtrace())
        end
    end
    isempty(rows) && error("no mesh completed")
    write_summary(rows, labels, root)

    println()
    println("="^78)
    println(" summary  (receiver RMS, t < wall, vs Hankel exact)")
    println("="^78)
    @printf("%-8s %-8s %6s %6s %5s %10s %12s %12s\n",
        "solver", "ndiv", "n", "ni", "nneg", "Δx", "vs exact", "max|p|")
    for r in rows
        dib = r.cmp === nothing ? NaN : r.cmp.recv_dibem_mean
        @printf("%-8s %-8d %6d %6d %5d %10.4f %12.3e %12.3e  %s\n",
            r.solver, r.ndiv, r.n, r.ni, r.nneg, r.dx, dib, r.maxabs,
            r.ok ? "ok" : "FAIL")
    end
    println("results in $(root)")
    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
