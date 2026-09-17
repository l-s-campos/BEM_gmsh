#=
================================================================================
RICKER WAVELET IN AN ANISOTROPIC (ELLIPTICAL) MEDIUM  —  DIBEM
================================================================================

Same continuum problem as  FVM testes/ricker_anisotropic_trixi.jl:

    p_tt = c_x² p_xx + c_y² p_yy + A R(t) g(x)     on  Ω = (−L, L)²
    ∂p/∂n = 0                                      on  ∂Ω   (hard wall)

Not full elastic VTI (one wave family).  Default  c_x = 1.6,  c_y = 1.0.
Wavefronts of the peak are the ellipse  x²/c_x² + y²/c_y² = (t − t₀)².

Geometry is the physical square.  Anisotropy is a uniform tensor
K = diag(c_x², c_y²) treated as in the heterogeneous/anisotropic DIBEM
path (`AnisotropicDIBEM.jl`): isotropic Laplace FS + DIBEM residual,
not a stretched mesh and not the anisotropic Green's function.

    ∇·(K ∇p) = k Δp + ∇·(ΔK ∇p),   k = (det K)^{1/2},   ΔK = K − k I.

Package DIBEM is H p − G q = M Δp.  Default strategy `:ibp` applies
Green's theorem once to ∇·(ΔK ∇p) (no Hessian of p), same split as
`solve_anisotropic_ibp!`.  `:hess` is the RBF-Hessian residual of
`solve_anisotropic_dibem!`.  Newmark then runs on

    H_eff p − G_eff q = (M/k) (p_tt − A R g).

On this axis-aligned square the hard wall ∂p/∂n = 0 is the Laplace
flux q = 0.  Free-space reference: 2-D Fourier (circular Gaussian,
elliptical dispersion), valid until a wall.

Run (from the repo root)

    julia --project=. scripts/transient/ricker_anisotropic_dibem.jl

ENV: RICKER_MESHES, RICKER_DT, RICKER_TF, RICKER_SOLVER=mmm|newmark|houbolt,
     RICKER_ANISO=ibp|hess|both (default ibp), RICKER_QUICK=1,
     RICKER_NLINE, RICKER_NGRID, RICKER_NLOCAL (RBF stencil; default global),
     TRIXI_OUT.

Figures: scripts/transient/ricker_anisotropic_dibem/out/
=#

using DrWatson
@quickactivate :BEM
using LinearAlgebra
using Printf
using StaticArrays
using Statistics
using DelimitedFiles
using Plots

ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")

include(datadir("Laplace", "Laplace_dad.jl"))
include(datadir("Laplace", "potencial_problems.jl"))
include(datadir("Laplace", "wave_propagation.jl"))

# ============================================================================
# 1. Problem data  (defaults match the Trixi anisotropic script)
# ============================================================================

Base.@kwdef struct AnisoRickerProblem
    L::Float64 = 2.0
    cx::Float64 = 1.6
    cy::Float64 = 1.0
    f0::Float64 = 4.0
    t0::Float64 = 0.25
    σ::Float64 = 0.06
    xs::Float64 = 0.0
    ys::Float64 = 0.0
    amp::Float64 = 1.0
    t_end::Float64 = 1.45
    ndiv::Int = 16
    Δt::Float64 = 0.01
end

function ricker(t::Real, prob::AnisoRickerProblem)
    τ = t - prob.t0
    aτ = (π * prob.f0 * τ)^2
    return (1 - 2 * aτ) * exp(-aτ)
end

ricker_antiderivative(t::Real, prob::AnisoRickerProblem) =
    (t - prob.t0) * exp(-(π * prob.f0 * (t - prob.t0))^2)

spatial_envelope(x, y, prob::AnisoRickerProblem) =
    exp(-((x - prob.xs)^2 + (y - prob.ys)^2) / (2 * prob.σ^2)) / (2π * prob.σ^2)

aniso_traveltime(x, y, prob::AnisoRickerProblem) =
    hypot((x - prob.xs) / prob.cx, (y - prob.ys) / prob.cy)

t_wall_x(prob::AnisoRickerProblem) = prob.t0 + prob.L / prob.cx
t_wall_y(prob::AnisoRickerProblem) = prob.t0 + prob.L / prob.cy

function snapshot_times(prob::AnisoRickerProblem)
    ts = (prob.t0 + 0.20, 0.70, 0.95, prob.t_end)
    return Tuple(unique(min(t, prob.t_end) for t in ts))
end

receiver_points(prob::AnisoRickerProblem) = (
    (prob.xs + 0.90, prob.ys),
    (prob.xs,         prob.ys + 0.90),
    (prob.xs + 0.65,  prob.ys + 0.65),
)

aniso_K(prob::AnisoRickerProblem) = @SMatrix [prob.cx^2 0; 0 prob.cy^2]

function ellipse_xy(prob::AnisoRickerProblem, t; n=240)
    τ = max(0.0, t - prob.t0)
    θ = range(0, 2π; length=n)
    return (prob.xs .+ prob.cx .* τ .* cos.(θ),
            prob.ys .+ prob.cy .* τ .* sin.(θ))
end

function mesh_list_from_env()
    quick = get(ENV, "RICKER_QUICK", "0") in ("1", "true", "TRUE")
    raw = get(ENV, "RICKER_MESHES", quick ? "8,12" : "64")
    return Int[parse(Int, strip(s)) for s in split(raw, ',') if !isempty(strip(s))]
end

function from_env(; ndiv=16)
    quick = get(ENV, "RICKER_QUICK", "0") in ("1", "true", "TRUE")
    Δt = parse(Float64, get(ENV, "RICKER_DT", quick ? "0.02" : "0.005"))
    t_end = parse(Float64, get(ENV, "RICKER_TF", quick ? "0.6" : "1.45"))
    return AnisoRickerProblem(; ndiv, Δt, t_end)
end

# ============================================================================
# 2. Free-space Fourier (elliptical dispersion, circular Gaussian)
# ============================================================================
#
#   P_tt + (c_x² k_x² + c_y² k_y²) P = A R(t) exp(−σ² |k|² / 2)
#   p(x,t) = (1/(2π)²) ∫ P(k,t) e^{ik·x} d²k
#
# Even in (x,y) and (k_x, k_y) ⇒ cosine quadrature on the first quadrant.

struct AnisoFourier
    kx::Vector{Float64}
    ky::Vector{Float64}
    Δkx::Float64
    Δky::Float64
    Fg::Matrix{Float64}
    ω::Matrix{Float64}
    ω1d::Vector{Float64}
end

function AnisoFourier(prob::AnisoRickerProblem; kmax_over_invσ=8.0, nk=257)
    kmax = kmax_over_invσ / prob.σ
    kx = collect(range(0.0, kmax; length=nk))
    ky = copy(kx)
    Δk = kx[2] - kx[1]
    Fg = zeros(nk, nk)
    ω = zeros(nk, nk)
    @inbounds for j in 1:nk, i in 1:nk
        Fg[i, j] = prob.amp * exp(-0.5 * prob.σ^2 * (kx[i]^2 + ky[j]^2))
        ω[i, j] = hypot(prob.cx * kx[i], prob.cy * ky[j])
    end
    ωmax = maximum(ω)
    nω = 2 * nk + 1
    ω1d = collect(range(0.0, ωmax; length=nω))
    return AnisoFourier(kx, ky, Δk, Δk, Fg, ω, ω1d)
end

function duhamel_I_omega!(I::AbstractVector, ω::AbstractVector, t::Real,
        prob::AnisoRickerProblem)
    fill!(I, 0.0)
    t <= 0 && return I
    nτ = max(16, round(Int, t / 0.002) + 1)
    τ = range(0.0, t; length=nτ)
    Δτ = τ[2] - τ[1]
    Rτ = ricker.(τ, Ref(prob))
    Rτ[1] *= 0.5
    Rτ[end] *= 0.5
    @inbounds for m in eachindex(ω)
        ωm = ω[m]
        s = 0.0
        if ωm < 1.0e-14
            for j in eachindex(τ)
                s += (t - τ[j]) * Rτ[j]
            end
            I[m] = s * Δτ
        else
            for j in eachindex(τ)
                s += sin(ωm * (t - τ[j])) * Rτ[j]
            end
            I[m] = s * Δτ / ωm
        end
    end
    return I
end

function interp1_scalar(x::AbstractVector, y::AbstractVector, q::Real)
    n = length(x)
    q <= x[1] && return y[1]
    q >= x[n] && return y[n]
    i = searchsortedlast(x, q)
    i = clamp(i, 1, n - 1)
    den = x[i + 1] - x[i]
    t = den == 0 ? 0.0 : (q - x[i]) / den
    return (1 - t) * y[i] + t * y[i + 1]
end

function I2d_from_omega(kernel::AnisoFourier, I1d::AbstractVector)
    I2 = similar(kernel.ω)
    @inbounds for j in axes(I2, 2), i in axes(I2, 1)
        I2[i, j] = interp1_scalar(kernel.ω1d, I1d, kernel.ω[i, j])
    end
    return I2
end

function analytical_pressure_aniso(x, y, kernel::AnisoFourier, I2::AbstractMatrix)
    s = 0.0
    nx, ny = length(kernel.kx), length(kernel.ky)
    @inbounds for j in 1:ny
        cy = cos(kernel.ky[j] * y)
        wy = (j == 1 || j == ny) ? 0.5 : 1.0
        sj = 0.0
        for i in 1:nx
            wx = (i == 1 || i == nx) ? 0.5 : 1.0
            sj += wx * kernel.Fg[i, j] * I2[i, j] * cos(kernel.kx[i] * x)
        end
        s += wy * cy * sj
    end
    return s * kernel.Δkx * kernel.Δky / π^2
end

# ============================================================================
# 3. Physical square + isotropic FS + DIBEM residual for K
# ============================================================================

function build_dad(prob::AnisoRickerProblem)
    nome = @sprintf("ricker_aniso_n%03d", prob.ndiv)
    msh = mesh_square_hardwall(; ndiv=prob.ndiv, L=prob.L, nome=nome, show=false)
    dad = format2d(msh, Laplace(1.0); tipo=1, pontointerno=true)
    dad.ni > 0 || error("format2d returned no internal (cell-centroid) points")
    return dad
end

function nlocal_from_env()
    raw = get(ENV, "RICKER_NLOCAL", "")
    isempty(strip(raw)) && return nothing
    return parse(Int, strip(raw))
end

function strategy_list_from_env()
    raw = lowercase(strip(get(ENV, "RICKER_ANISO", "ibp")))
    raw == "both" && return [:ibp, :hess]
    raw in ("ibp", "hess") || error("RICKER_ANISO must be ibp|hess|both (got $raw)")
    return [Symbol(raw)]
end

struct FieldInterp
    rbf::RBF
    xline::Vector{Float64}
    yline::Vector{Float64}
    qx::Vector{Point2D}
    qy::Vector{Point2D}
    xfine::Vector{Float64}
    yfine::Vector{Float64}
    qgrid::Vector{Point2D}
    nx::Int
    ny::Int
    qrec::Vector{Point2D}
end

function FieldInterp(dad, prob::AnisoRickerProblem)
    nline = parse(Int, get(ENV, "RICKER_NLINE", "301"))
    ngrid = parse(Int, get(ENV, "RICKER_NGRID", "161"))
    pts = collect(all_points(dad))
    rbf = RBF(pts, PHS(3; poly_deg=1))
    xline = collect(range(-prob.L, prob.L; length=nline))
    yline = copy(xline)
    qx = Point2D[SA[x, 0.0] for x in xline]
    qy = Point2D[SA[0.0, y] for y in yline]
    xfine = collect(range(-prob.L, prob.L; length=ngrid))
    yfine = copy(xfine)
    qgrid = Point2D[SA[x, y] for y in yfine for x in xfine]
    qrec = Point2D[SA[float(p[1]), float(p[2])] for p in receiver_points(prob)]
    return FieldInterp(rbf, xline, yline, qx, qy, xfine, yfine, qgrid, ngrid, ngrid, qrec)
end

eval_xcut(ip::FieldInterp, u) = rbf_evaluate(ip.rbf, ip.qx, u)
eval_ycut(ip::FieldInterp, u) = rbf_evaluate(ip.rbf, ip.qy, u)
eval_rec(ip::FieldInterp, u) = rbf_evaluate(ip.rbf, ip.qrec, u)
function eval_grid(ip::FieldInterp, u)
    v = rbf_evaluate(ip.rbf, ip.qgrid, u)
    return reshape(v, ip.nx, ip.ny)'
end

function nodal_envelope(dad, prob::AnisoRickerProblem)
    g = zeros(dad.nt)
    @inbounds for i in 1:dad.nt
        p = point(dad, i)
        g[i] = spatial_envelope(p[1], p[2], prob)
    end
    return g
end

function nearest_time_index(tgrid, t)
    return argmin(i -> abs(tgrid[i] - t), eachindex(tgrid))
end

rms_linf(a, b) = (e = a .- b; (sqrt(sum(abs2, e) / length(e)), maximum(abs, e)))

function interp1(x::AbstractVector, y::AbstractVector, xq::AbstractVector)
    n = length(x)
    out = Vector{Float64}(undef, length(xq))
    @inbounds for (k, q) in enumerate(xq)
        out[k] = interp1_scalar(x, y, q)
    end
    return out
end

# ============================================================================
# 4. Plots
# ============================================================================

function plot_wavelet(prob::AnisoRickerProblem, outdir)
    t = range(0.0, prob.t_end; length=800)
    plt = plot(
        t, ricker.(t, Ref(prob));
        xlabel="t", ylabel="amplitude",
        title="Ricker wavelet   f₀ = $(prob.f0),   t₀ = $(prob.t0)",
        label="R(t)",
        lw=2.2, c=:black, size=(900, 420), legend=:topright,
    )
    plot!(plt, t, ricker_antiderivative.(t, Ref(prob));
        lw=1.8, c=:steelblue, label="∫ R  (injected into p'_t)")
    vline!(plt, [prob.t0]; ls=:dash, c=:gray, label="t₀")
    savefig(plt, joinpath(outdir, "ricker_wavelet.png"))
    return nothing
end

function plot_source_blob(prob::AnisoRickerProblem, outdir)
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

function plot_snapshots(prob::AnisoRickerProblem, dad, kernel::AnisoFourier,
        outdir, ip::FieldInterp)
    tgrid = dad.t isa AbstractRange ? collect(dad.t) : collect(dad.t)
    I1 = zeros(length(kernel.ω1d))
    plt_x = plot(xlabel="x", ylabel="p'(x, y=0)",
        title="Fast axis  (y = 0)", legend=:topright, size=(900, 360))
    plt_y = plot(xlabel="y", ylabel="p'(x=0, y)",
        title="Slow axis  (x = 0)", legend=:topright, size=(900, 360))
    plt_err = plot(xlabel="coordinate", ylabel="|p'_DIBEM − p'_Fourier|",
        title="Axis-cut error vs. free-space Fourier",
        yscale=:log10, size=(900, 360), legend=:outertopright)

    csv = joinpath(outdir, "axis_cuts.csv")
    snaps = snapshot_times(prob)
    clim = 0.0
    fields = Tuple{Float64,Vector{Float64}}[]
    open(csv, "w") do io
        println(io, "t,axis,s,p_dibem,p_exact,error")
        for ts in snaps
            it = nearest_time_index(tgrid, ts)
            t = tgrid[it]
            u = view(dad.T, :, it)
            px = eval_xcut(ip, u)
            py = eval_ycut(ip, u)
            duhamel_I_omega!(I1, kernel.ω1d, t, prob)
            I2 = I2d_from_omega(kernel, I1)
            pex = [analytical_pressure_aniso(x, 0.0, kernel, I2) for x in ip.xline]
            pey = [analytical_pressure_aniso(0.0, y, kernel, I2) for y in ip.yline]
            tlab = "t = $(round(t; digits=2))"
            plot!(plt_x, ip.xline, px; lw=1.8, label=tlab)
            plot!(plt_x, ip.xline, pex; lw=1.4, ls=:dash, label=false)
            plot!(plt_y, ip.yline, py; lw=1.8, label=tlab)
            plot!(plt_y, ip.yline, pey; lw=1.4, ls=:dash, label=false)
            plot!(plt_err, ip.xline, max.(abs.(px .- pex), 1e-16);
                lw=1.5, label=tlab * "  x")
            plot!(plt_err, ip.yline, max.(abs.(py .- pey), 1e-16);
                lw=1.5, ls=:dash, label=tlab * "  y")
            rx = prob.cx * max(0.0, t - prob.t0)
            ry = prob.cy * max(0.0, t - prob.t0)
            vline!(plt_x, [-rx, rx]; ls=:dot, c=:gray, label=false)
            vline!(plt_y, [-ry, ry]; ls=:dot, c=:gray, label=false)
            for i in eachindex(ip.xline)
                @printf(io, "%.12e,x,%.12e,%.12e,%.12e,%.12e\n",
                    t, ip.xline[i], px[i], pex[i], px[i] - pex[i])
            end
            for i in eachindex(ip.yline)
                @printf(io, "%.12e,y,%.12e,%.12e,%.12e,%.12e\n",
                    t, ip.yline[i], py[i], pey[i], py[i] - pey[i])
            end
            clim = max(clim, maximum(abs, u))
            push!(fields, (t, collect(u)))
        end
    end
    clim = max(clim, 1.0e-12)
    savefig(plot(plt_x, plt_y; layout=(2, 1), size=(900, 720)),
        joinpath(outdir, "axis_cuts.png"))
    savefig(plt_err, joinpath(outdir, "axis_cuts_error.png"))

    for f in readdir(outdir; join=true)
        bn = basename(f)
        startswith(bn, "heatmap_t") && endswith(bn, ".png") && rm(f)
    end
    for (k, (t, u)) in enumerate(fields)
        Zf = eval_grid(ip, u)
        plt = heatmap(
            ip.xfine, ip.yfine, Zf;
            xlabel="x", ylabel="y",
            title=@sprintf("anisotropic p'   t = %.2f    (c_x=%.2f, c_y=%.2f)",
                t, prob.cx, prob.cy),
            clims=(-clim, clim), c=:seismic, aspect_ratio=:equal, size=(720, 640),
            legend=:topright,
        )
        scatter!(plt, [prob.xs], [prob.ys]; ms=6, mc=:black, msw=0, label="source")
        ex, ey = ellipse_xy(prob, t)
        plot!(plt, ex, ey; lw=1.8, c=:black, ls=:dash, label="ellipse (t−t₀)")
        ttag = replace(@sprintf("%.2f", t), "." => "p")
        savefig(plt, joinpath(outdir, "heatmap_t$(k)_$(ttag).png"))
    end
    return nothing
end

function plot_receivers(prob::AnisoRickerProblem, tgrid, traces, coords, labels,
        p_exact, outdir)
    colors = palette(:tab10)
    plt_src = plot(
        tgrid, ricker.(tgrid, Ref(prob));
        xlabel="t", ylabel="R(t)", title="Source wavelet",
        label="Ricker R(t)", lw=2.2, c=:black, legend=:topright,
    )
    vline!(plt_src, [prob.t0]; ls=:dash, c=:gray, label="t₀")
    plt_rec = plot(
        xlabel="t", ylabel="p'",
        title="Receivers  —  DIBEM (solid) vs. free-space Fourier (dashed)",
        legend=:outertopright,
    )
    plt_err = plot(
        xlabel="t", ylabel="p'_DIBEM − p'_Fourier",
        title="DIBEM − free-space Fourier",
        legend=:outertopright,
    )
    for k in eachindex(traces)
        c = colors[k]
        plot!(plt_rec, tgrid, traces[k]; lw=1.8, c=c, label=labels[k] * "  DIBEM")
        plot!(plt_rec, tgrid, p_exact[k]; lw=1.6, c=c, ls=:dash, label=labels[k] * "  exact")
        plot!(plt_err, tgrid, traces[k] .- p_exact[k]; lw=1.4, c=c, label=labels[k])
        x, y = coords[k]
        vline!(plt_rec, [prob.t0 + aniso_traveltime(x, y, prob)];
            ls=:dot, c=c, label=false)
    end
    vline!(plt_rec, [prob.t0]; ls=:dash, c=:gray, label="t₀")
    vline!(plt_rec, [t_wall_x(prob)]; ls=:dot, c=:black, label="fast-axis wall")
    plt = plot(plt_src, plt_rec, plt_err; layout=(3, 1), size=(900, 980))
    savefig(plt, joinpath(outdir, "receiver_traces.png"))
    return nothing
end

function write_pressure_gif(prob::AnisoRickerProblem, dad, outdir, ip::FieldInterp;
        fps=12, nframes=48)
    tgrid = dad.t isa AbstractRange ? collect(dad.t) : collect(dad.t)
    n = length(tgrid)
    n >= 2 || error("no time history")
    nframes = min(nframes, n)
    idx = unique(round.(Int, range(1, n; length=nframes)))
    clim = 0.0
    frames = Vector{Matrix{Float64}}(undef, length(idx))
    for (k, i) in enumerate(idx)
        Z = eval_grid(ip, view(dad.T, :, i))
        frames[k] = Z
        clim = max(clim, maximum(abs, Z))
    end
    clim = max(clim, 1.0e-12)
    anim = Animation()
    for (k, i) in enumerate(idx)
        t = tgrid[i]
        plt = heatmap(
            ip.xfine, ip.yfine, frames[k];
            title=@sprintf("anisotropic p'   t = %.3f", t),
            clims=(-clim, clim), c=:seismic, aspect_ratio=:equal,
            size=(720, 640), legend=:topright,
        )
        scatter!(plt, [prob.xs], [prob.ys]; ms=6, mc=:black, msw=0, label="source")
        ex, ey = ellipse_xy(prob, t)
        plot!(plt, ex, ey; lw=1.8, c=:black, ls=:dash, label="ellipse (t−t₀)")
        frame(anim, plt)
    end
    outfile = joinpath(outdir, "pressure_anisotropic.gif")
    try
        gif(anim, outfile; fps=fps)
    catch err
        @warn "Plots.gif failed" exception = err
    end
    @info "wrote $(outfile)  ($(length(idx)) frames @ $(fps) fps)"
    return outfile
end

function print_error_table(prob, tgrid, traces, labels, p_exact, dad, kernel, ip)
    tw = t_wall_x(prob)
    println()
    println("="^78)
    strat = has_cache(dad, :aniso_strategy) ? dad.aniso_strategy : :unknown
    println("DIBEM  vs.  free-space Fourier  (elliptical acoustics)")
    println("  p_tt = ∇·(K ∇p) + R g ,  K=diag(c_x²,c_y²),  isotropic FS + $strat")
    println("  fast-axis wall at t₀ + L/c_x = $(round(tw; digits=2))")
    println("="^78)
    @printf("%-36s %12s %12s\n", "probe", "RMS", "Linf")
    for k in eachindex(traces)
        pre = [i for i in eachindex(tgrid) if tgrid[i] < tw]
        rms, linf = rms_linf(traces[k][pre], p_exact[k][pre])
        @printf("%-36s %12.3e %12.3e\n", "receiver $(labels[k])", rms, linf)
    end
    println("-"^78)
    I1 = zeros(length(kernel.ω1d))
    for ts in snapshot_times(prob)
        it = nearest_time_index(dad.t isa AbstractRange ? collect(dad.t) : collect(dad.t), ts)
        t = (dad.t isa AbstractRange ? collect(dad.t) : collect(dad.t))[it]
        u = view(dad.T, :, it)
        px = eval_xcut(ip, u)
        py = eval_ycut(ip, u)
        duhamel_I_omega!(I1, kernel.ω1d, t, prob)
        I2 = I2d_from_omega(kernel, I1)
        pex = [analytical_pressure_aniso(x, 0.0, kernel, I2) for x in ip.xline]
        pey = [analytical_pressure_aniso(0.0, y, kernel, I2) for y in ip.yline]
        rmsx, _ = rms_linf(px, pex)
        rmsy, _ = rms_linf(py, pey)
        @printf("%-36s %12.3e %12s\n", "fast-axis t=$(round(t; digits=2))", rmsx, "")
        @printf("%-36s %12.3e %12s\n", "slow-axis t=$(round(t; digits=2))", rmsy, "")
    end
    println("="^78)
    return nothing
end

# ============================================================================
# 5. Trixi comparison
# ============================================================================

function load_trixi_aniso_receivers(path)
    data, hdr = readdlm(path, ','; header=true)
    t = Float64.(data[:, 1])
    ncol = size(data, 2) - 1
    traces = [Float64.(data[:, k + 1]) for k in 1:ncol]
    return t, traces, String.(vec(hdr))
end

function plot_compare_trixi(prob, tgrid, traces, labels, p_exact, outdir, trixi_out)
    rec_csv = joinpath(trixi_out, "receivers.csv")
    isfile(rec_csv) || begin
        @warn "Trixi output not found; skip comparison" trixi_out
        return nothing
    end
    tT, pT, _ = load_trixi_aniso_receivers(rec_csv)
    n = min(length(traces), length(pT))
    colors = palette(:tab10)
    plt_rec = plot(
        xlabel="t", ylabel="p'",
        title="Receivers  —  Trixi (solid)  DIBEM (dash-dot)  Fourier (dashed)",
        legend=:outertopright,
    )
    plt_err = plot(
        xlabel="t", ylabel="p' − p'_Fourier",
        title="Error vs. free-space Fourier  (Trixi solid, DIBEM dash-dot)",
        legend=:outertopright,
    )
    pT_on_d = Vector{Vector{Float64}}(undef, n)
    println()
    println("="^78)
    println("Trixi DGSEM  vs.  DIBEM  vs.  free-space Fourier")
    println("="^78)
    @printf("%-36s %12s %12s %12s\n", "probe", "Trixi RMS", "DIBEM RMS", "DIBEM−Trixi")
    tw = t_wall_x(prob)
    for k in 1:n
        c = colors[k]
        pT_on_d[k] = interp1(tT, pT[k], tgrid)
        plot!(plt_rec, tT, pT[k]; lw=1.8, c=c, label=labels[k] * "  Trixi")
        plot!(plt_rec, tgrid, traces[k]; lw=1.7, c=c, ls=:dashdot, label=labels[k] * "  DIBEM")
        plot!(plt_rec, tgrid, p_exact[k]; lw=1.4, c=c, ls=:dash, label=labels[k] * "  exact")
        plot!(plt_err, tgrid, pT_on_d[k] .- p_exact[k]; lw=1.4, c=c, label=labels[k] * "  Trixi")
        plot!(plt_err, tgrid, traces[k] .- p_exact[k];
            lw=1.4, c=c, ls=:dashdot, label=labels[k] * "  DIBEM")
        pre = [i for i in eachindex(tgrid) if tgrid[i] < tw]
        rmsT, _ = rms_linf(pT_on_d[k][pre], p_exact[k][pre])
        rmsD, _ = rms_linf(traces[k][pre], p_exact[k][pre])
        rmsDT, _ = rms_linf(traces[k][pre], pT_on_d[k][pre])
        @printf("%-36s %12.3e %12.3e %12.3e\n", labels[k], rmsT, rmsD, rmsDT)
    end
    println("="^78)
    pts = receiver_points(prob)
    for (k, pt) in enumerate(pts)
        vline!(plt_rec, [prob.t0 + aniso_traveltime(pt[1], pt[2], prob)];
            ls=:dot, c=colors[k], label=false)
    end
    vline!(plt_rec, [prob.t0]; ls=:dash, c=:gray, label="t₀")
    plt = plot(plt_rec, plt_err; layout=(2, 1), size=(900, 720))
    savefig(plt, joinpath(outdir, "compare_receivers.png"))

    csv_cmp = joinpath(outdir, "compare_receivers.csv")
    open(csv_cmp, "w") do io
        write(io, "t")
        for lab in labels[1:n]
            tag = replace(lab, r"[^0-9A-Za-z]+" => "_")
            write(io, ",p_trixi$(tag),p_dibem$(tag),p_exact$(tag)")
        end
        println(io)
        for i in eachindex(tgrid)
            @printf(io, "%.12e", tgrid[i])
            for k in 1:n
                @printf(io, ",%.12e,%.12e,%.12e",
                    pT_on_d[k][i], traces[k][i], p_exact[k][i])
            end
            println(io)
        end
    end
    println("wrote comparison figures to $(outdir)")
    recv_dibem = Float64[rms_linf(traces[k][tgrid .< tw], p_exact[k][tgrid .< tw])[1] for k in 1:n]
    recv_trixi = Float64[rms_linf(pT_on_d[k][tgrid .< tw], p_exact[k][tgrid .< tw])[1] for k in 1:n]
    recv_vs_trixi = Float64[rms_linf(traces[k][tgrid .< tw], pT_on_d[k][tgrid .< tw])[1] for k in 1:n]
    return (;
        recv_dibem_rms=recv_dibem,
        recv_trixi_rms=recv_trixi,
        recv_dibem_vs_trixi=recv_vs_trixi,
        recv_dibem_mean=mean(recv_dibem),
        recv_trixi_mean=mean(recv_trixi),
    )
end

function plot_compare_strategies(prob, rows, labels, outdir)
    ibp = filter(r -> r.strategy === :ibp && r.ok && r.cmp !== nothing, rows)
    hess = filter(r -> r.strategy === :hess && r.ok && r.cmp !== nothing, rows)
    (isempty(ibp) || isempty(hess)) && return nothing
    csv_ibp = joinpath(ibp[end].outdir, "compare_receivers.csv")
    csv_hess = joinpath(hess[end].outdir, "compare_receivers.csv")
    (isfile(csv_ibp) && isfile(csv_hess)) || return nothing
    dI, hI = readdlm(csv_ibp, ','; header=true)
    dH, _ = readdlm(csv_hess, ','; header=true)
    t = Float64.(dI[:, 1])
    n = length(labels)
    colors = palette(:tab10)
    plt = plot(
        xlabel="t", ylabel="p'",
        title="IBP (default) vs. Hessian residual vs. Trixi",
        legend=:outertopright, size=(900, 480),
    )
    plt_err = plot(
        xlabel="t", ylabel="p' − p'_Fourier",
        title="Error vs. Fourier  (IBP solid, Hessian dash-dot, Trixi dotted)",
        legend=:outertopright, size=(900, 420),
    )
    println()
    println("="^78)
    println("IBP vs. Hessian residual  (finest mesh in this run)")
    println("="^78)
    @printf("%-24s %12s %12s %12s\n", "probe", "IBP RMS", "Hess RMS", "Trixi RMS")
    for k in 1:n
        c = colors[k]
        pT = Float64.(dI[:, 3k - 1])
        pI = Float64.(dI[:, 3k])
        pex = Float64.(dI[:, 3k + 1])
        pH = Float64.(dH[:, 3k])
        plot!(plt, t, pT; lw=1.4, c=c, ls=:dot, label=labels[k] * "  Trixi")
        plot!(plt, t, pI; lw=1.8, c=c, label=labels[k] * "  IBP")
        plot!(plt, t, pH; lw=1.6, c=c, ls=:dashdot, label=labels[k] * "  Hess")
        plot!(plt, t, pex; lw=1.2, c=c, ls=:dash, label=false)
        plot!(plt_err, t, pI .- pex; lw=1.6, c=c, label=labels[k] * "  IBP")
        plot!(plt_err, t, pH .- pex; lw=1.5, c=c, ls=:dashdot, label=labels[k] * "  Hess")
        plot!(plt_err, t, pT .- pex; lw=1.3, c=c, ls=:dot, label=labels[k] * "  Trixi")
        tw = t_wall_x(prob)
        pre = t .< tw
        rI, _ = rms_linf(pI[pre], pex[pre])
        rH, _ = rms_linf(pH[pre], pex[pre])
        rT, _ = rms_linf(pT[pre], pex[pre])
        @printf("%-24s %12.3e %12.3e %12.3e\n", labels[k], rI, rH, rT)
    end
    println("="^78)
    savefig(plot(plt, plt_err; layout=(2, 1), size=(900, 900)),
        joinpath(outdir, "compare_ibp_hess.png"))
    println("wrote $(joinpath(outdir, "compare_ibp_hess.png"))")
    return nothing
end

# ============================================================================
# 6. One mesh + sweep
# ============================================================================

function run_case(prob::AnisoRickerProblem, outdir, kernel, trixi_out;
        solver="newmark", strategy::Symbol=:ibp)
    mkpath(outdir)
    dx = 2 * prob.L / (prob.ndiv - 1)
    λs = prob.cy / prob.f0
    println()
    println("="^78)
    println(" mesh ndiv=$(prob.ndiv)  strategy=$strategy  Δx=$(round(dx; digits=4))  pts/λ_slow ≈ $(round(λs / dx; digits=2))")
    println(" Ω = (−$(prob.L), $(prob.L))²  (physical square, isotropic Laplace FS)")
    println("="^78)

    plot_wavelet(prob, outdir)
    plot_source_blob(prob, outdir)

    dad = build_dad(prob)
    println(dad)
    @info "format2d internals (cell centroids, physical square)" ni = dad.ni n = dad.n nt = dad.nt

    K = aniso_K(prob)
    kiso = isotropic_scale(K)
    nloc = nlocal_from_env()
    t_asm = @elapsed begin
        H_G_full_direct(dad; npg=10, threaded=false)
        DIBEM(dad; rbf=PHS(3; poly_deg=1), npg=12)
        anisotropic_wave_shift!(dad, K; strategy=strategy,
            rbf=PHS(3; poly_deg=2), nlocal=nloc)
    end
    println("  assembly+DIBEM+$strategy: $(round(t_asm; digits=2)) s")
    println("  K = diag($(prob.cx)^2, $(prob.cy)^2) = diag($(K[1,1]), $(K[2,2]))")
    println("  k = (det K)^{1/2} = $kiso   nlocal = $(nloc === nothing ? "global" : nloc)")
    if strategy === :ibp
        println("  IBP: H_eff = H + (G SΓ − AV)/k ,  M_eff = M/k ,  q = 0")
    else
        println("  Hessian residual: H_eff = H + M Af ,  M_eff = M/k ,  q = 0")
    end

    nneg = dad.nt <= 1200 ? count(<(-1e-8), real.(eigvals(Matrix(dad.M)))) : -1
    extra = if strategy === :hess && has_cache(dad, :aniso_Af)
        @sprintf("  ‖Af‖_F=%.3e", norm(dad.aniso_Af))
    elseif strategy === :ibp && has_cache(dad, :aniso_AV)
        @sprintf("  ‖AV‖_F=%.3e", norm(dad.aniso_AV))
    else
        ""
    end
    @printf("  DIBEM M/k: %d×%d  nneg(M)=%s  ‖M‖_F=%.3e%s\n",
        size(dad.M, 1), size(dad.M, 2), nneg < 0 ? "skip" : string(nneg),
        norm(dad.M), extra)

    g = nodal_envelope(dad, prob)
    f_body = t -> (prob.amp * ricker(t, prob)) .* g
    println("  load: f_body(t) = R(t) g(x,y)  at $(dad.nt) collocation points")

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
        @printf("  MMM modes: %d  ω₁=%.4f  ω_max=%.2f\n", nmodes, ω1, b.ω[end])
    end
    ok = all(isfinite, dad.T)
    mx = ok ? maximum(abs, dad.T) : NaN
    @info "solution finite" ok maxabs = mx

    t_rbf = @elapsed ip = FieldInterp(dad, prob)
    println("  PHS interpolant: $(length(ip.rbf.x)) centres, " *
            "grid=$(ip.nx)×$(ip.ny)  ($(round(t_rbf; digits=2)) s)")

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
        τ = aniso_traveltime(pt[1], pt[2], prob)
        @info "receiver $lab  PHS at $pt  peak ~ t₀+τ = $(round(prob.t0 + τ; digits=3))"
    end

    I1 = zeros(length(kernel.ω1d))
    p_exact = [zeros(length(tgrid)) for _ in traces]
    t_ex = @elapsed for (i, t) in enumerate(tgrid)
        duhamel_I_omega!(I1, kernel.ω1d, t, prob)
        I2 = I2d_from_omega(kernel, I1)
        for k in eachindex(coords)
            p_exact[k][i] = analytical_pressure_aniso(coords[k][1], coords[k][2], kernel, I2)
        end
    end
    println("  Fourier exact traces: $(round(t_ex; digits=2)) s")

    plot_snapshots(prob, dad, kernel, outdir, ip)
    plot_receivers(prob, tgrid, traces, coords, labels, p_exact, outdir)
    write_pressure_gif(prob, dad, outdir, ip)
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

    cmp = plot_compare_trixi(prob, tgrid, traces, labels, p_exact, outdir, trixi_out)
    println("wrote figures and CSVs to $(outdir)")
    return (;
        ndiv=prob.ndiv, n=dad.n, ni=dad.ni, nt=dad.nt, dx=dx, nneg=nneg,
        ok=ok, maxabs=mx, nmodes=nmodes, ω1=ω1, solver=solver, strategy=strategy,
        t_asm=t_asm, t_sol=t_sol, cmp=cmp, labels=labels, outdir=outdir,
    )
end

function write_summary(rows, labels, outdir)
    csv = joinpath(outdir, "summary.csv")
    open(csv, "w") do io
        write(io, "strategy,solver,ndiv,n,ni,nt,dx,nneg,ok,maxabs,nmodes,omega1,t_asm,t_sol,recv_dibem_mean,recv_trixi_mean")
        for lab in labels
            tag = replace(lab, r"[^0-9A-Za-z]+" => "_")
            write(io, ",dibem$(tag),trixi$(tag),dibem_vs_trixi$(tag)")
        end
        println(io)
        for r in rows
            c = r.cmp
            @printf(io, "%s,%s,%d,%d,%d,%d,%.8e,%d,%d,%.8e,%d,%.8e,%.4f,%.4f",
                r.strategy, r.solver, r.ndiv, r.n, r.ni, r.nt, r.dx, r.nneg, r.ok ? 1 : 0, r.maxabs,
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
    println("wrote $(csv)")
    return csv
end

function main()
    meshes = mesh_list_from_env()
    solver = lowercase(get(ENV, "RICKER_SOLVER", "newmark"))
    strats = strategy_list_from_env()
    root = joinpath(@__DIR__, "ricker_anisotropic_dibem", "out")
    mkpath(root)
    trixi_out = get(ENV, "TRIXI_OUT",
        "/data/OneDrive/pesquisa/FVM testes/ricker_anisotropic/out")

    base = from_env(; ndiv=meshes[1])
    println()
    println("Ricker wavelet, anisotropic acoustics  (isotropic FS + DIBEM)")
    println("  Ω = (−$(base.L), $(base.L))²   (geometry unchanged)")
    println("  c_x = $(base.cx)  (fast),  c_y = $(base.cy)  (slow)")
    println("  K = diag(c_x², c_y²)  on Laplace(1) kernels")
    println("  strategy = $(join(strats, ", "))   (default IBP / Green's theorem once)")
    println("  f₀ = $(base.f0),  t₀ = $(base.t0),  σ = $(base.σ),  t_end = $(base.t_end)")
    println("  meshes ndiv = $(meshes)  Δt = $(base.Δt)  solver = $solver")
    println("  peak hits x-wall at t₀ + L/c_x = $(round(t_wall_x(base); digits=2))")
    println("  peak hits y-wall at t₀ + L/c_y = $(round(t_wall_y(base); digits=2))")
    println()

    kernel = AnisoFourier(base)
    @info "Fourier k-grid" kmax = kernel.kx[end] nk = length(kernel.kx)
    plot_wavelet(base, root)
    plot_source_blob(base, root)

    rows = []
    labels = ["($(p[1]), $(p[2]))" for p in receiver_points(base)]
    for ndiv in meshes, strategy in strats
        prob = from_env(; ndiv=ndiv)
        nloc = nlocal_from_env()
        tag = nloc === nothing ?
              @sprintf("ndiv_%03d_%s_%s", ndiv, solver, strategy) :
              @sprintf("ndiv_%03d_%s_%s_n%d", ndiv, solver, strategy, nloc)
        outdir = joinpath(root, tag)
        try
            push!(rows, run_case(prob, outdir, kernel, trixi_out;
                solver=solver, strategy=strategy))
        catch e
            @error "mesh ndiv=$ndiv strategy=$strategy failed" exception = (e, catch_backtrace())
        end
    end
    isempty(rows) && error("no mesh completed")
    write_summary(rows, labels, root)
    plot_compare_strategies(base, rows, labels, root)

    println()
    println("="^78)
    println(" summary  (receiver RMS vs. Fourier exact / Trixi)")
    println("="^78)
    @printf("%-6s %-8s %-8s %6s %6s %5s %10s %12s %12s\n",
        "strat", "solver", "ndiv", "n", "ni", "nneg", "Δx", "vs exact", "max|p|")
    for r in rows
        dib = r.cmp === nothing ? NaN : r.cmp.recv_dibem_mean
        @printf("%-6s %-8s %-8d %6d %6d %5d %10.4f %12.3e %12.3e  %s\n",
            r.strategy, r.solver, r.ndiv, r.n, r.ni, r.nneg, r.dx, dib, r.maxabs,
            r.ok ? "ok" : "FAIL")
    end
    println("Expected peak arrivals  (t₀ + elliptical traveltime)")
    for (lab, xy) in zip(labels, receiver_points(base))
        τ = aniso_traveltime(xy[1], xy[2], base)
        @printf("  %-20s  t = %.3f\n", lab, base.t0 + τ)
    end
    println("results in $(root)")
    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
