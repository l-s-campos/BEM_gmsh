# =============================================================================
# Performance comparison: dense / FFT / H-matrix / FMM for 2D half-space contact
# Publication-quality PDF figures → BEM-wear article folder
# =============================================================================
using DrWatson
@quickactivate :BEM
using BEM.Contact
using BEM.HMatrices

using LinearAlgebra
using Statistics
using Printf
using Random
using Dates

# plotting
using Plots

# Article root (all figures next to main.tex)
const OUTDIR = raw"C:\Users\lucas.s.campos\OneDrive\artigos\escritos\2026\BEM-wear"
mkpath(OUTDIR)

println("="^64)
println(" Half-space BEM acceleration comparison")
println(" ", Dates.now())
println("="^64)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
function time_matvec(K, p; nwarm=3, nrun=10)
    y = similar(p)
    for _ in 1:nwarm
        mul!(y, K isa AbstractMatrix ? K : K, p)  # fallback
        try
            y .= K * p
        catch
            mul!(y, K, p)
        end
    end
    # unified apply
    apply = p -> begin
        if K isa AbstractMatrix && !(K isa HMatrix)
            return K * p
        else
            return K * p
        end
    end
    apply(p)
    t0 = time_ns()
    local acc
    for _ in 1:nrun
        acc = apply(p)
    end
    t1 = time_ns()
    return (t1 - t0) / nrun / 1e9, acc
end

function bench_backends(ns; E=1.0)
    methods = [:dense, :fft, :hmatrix, :fmm]
    results = Dict{Symbol,Any}()
    for m in methods
        results[m] = (n=Int[], t_build=Float64[], t_mv=Float64[], err=Float64[], mem=Float64[])
    end
    Random.seed!(1)
    for n in ns
        println("\n--- N = $n ---")
        # parabolic gap (cylinder-like)
        dad = HalfSpace2D(-1.0, 1.0, n; E=E)
        dad.h0 .= (dad.x .^ 2) ./ 2
        p = rand(n)
        p ./= sum(p .* dad.al)   # unit load

        # dense reference
        t0 = time_ns()
        Kd = build_operator(dad, :dense)
        t_bd = (time_ns() - t0) / 1e9
        t_md, y_ref = begin
            t0 = time_ns()
            local y
            for _ in 1:5
                y = Kd * p
            end
            ((time_ns() - t0) / 5 / 1e9, Kd * p)
        end
        push!(results[:dense].n, n)
        push!(results[:dense].t_build, t_bd)
        push!(results[:dense].t_mv, t_md)
        push!(results[:dense].err, 0.0)
        push!(results[:dense].mem, Base.summarysize(Kd) / 1024^2)

        for m in (:fft, :hmatrix, :fmm)
            try
                t0 = time_ns()
                K = if m === :hmatrix
                    build_operator(dad, m; nmax=max(16, n ÷ 16), atol=1e-6)
                elseif m === :fmm
                    build_operator(dad, m; θ=0.7, near_factor=4.0)
                else
                    build_operator(dad, m)
                end
                t_b = (time_ns() - t0) / 1e9
                # matvec timing
                y = K * p
                t0 = time_ns()
                nrun = m === :fmm ? 3 : 10
                for _ in 1:nrun
                    y = K * p
                end
                t_m = (time_ns() - t0) / nrun / 1e9
                err = norm(y - y_ref) / max(norm(y_ref), eps())
                mem = Base.summarysize(K) / 1024^2
                push!(results[m].n, n)
                push!(results[m].t_build, t_b)
                push!(results[m].t_mv, t_m)
                push!(results[m].err, err)
                push!(results[m].mem, mem)
                @printf("  %-8s  build=%8.3f s  matvec=%8.4f s  err=%9.2e  mem=%7.2f MB\n",
                    m, t_b, t_m, err, mem)
            catch e
                @warn "method $m failed at n=$n" exception=e
            end
        end
    end
    return results
end

# -----------------------------------------------------------------------------
# 1) Scaling study
# -----------------------------------------------------------------------------
ns = [64, 128, 256, 512, 1024]
println("\n[1] Scaling study N ∈ $ns")
results = bench_backends(ns)

# Plot matvec time
cols = Dict(:dense=>:black, :fft=>:dodgerblue, :hmatrix=>:crimson, :fmm=>:seagreen)
marks = Dict(:dense=>:circle, :fft=>:rect, :hmatrix=>:utriangle, :fmm=>:diamond)
fig1 = plot(; size=(480, 360), xlabel="N", ylabel="matvec time [s]",
    xscale=:log2, yscale=:log10, title="Half-space matvec cost",
    legend=:topleft, legendfontsize=9, framestyle=:box, background_color=:white)
for m in (:dense, :fft, :hmatrix, :fmm)
    r = results[m]
    isempty(r.n) && continue
    plot!(fig1, r.n, r.t_mv; color=cols[m], marker=marks[m], label=String(m))
end
nref = [64.0, 1024.0]
plot!(fig1, nref, 1e-7 .* nref .^ 2; color=:gray, linestyle=:dash, label="O(N²)")
plot!(fig1, nref, 3e-8 .* nref .* log2.(nref); color=:gray, linestyle=:dot, label="O(N log N)")
savefig(fig1, joinpath(OUTDIR, "fig_matvec_scaling.pdf"))
savefig(fig1, joinpath(OUTDIR, "fig_matvec_scaling.png"))
println("  saved fig_matvec_scaling.pdf")

# Plot relative error
fig2 = plot(; size=(480, 360), xlabel="N", ylabel="relative matvec error",
    xscale=:log2, yscale=:log10, title="Accuracy vs dense reference",
    legend=:topright, legendfontsize=9, framestyle=:box, background_color=:white)
for m in (:fft, :hmatrix, :fmm)
    r = results[m]
    isempty(r.n) && continue
    plot!(fig2, r.n, max.(r.err, 1e-16); color=cols[m], marker=marks[m], label=String(m))
end
savefig(fig2, joinpath(OUTDIR, "fig_matvec_error.pdf"))
savefig(fig2, joinpath(OUTDIR, "fig_matvec_error.png"))
println("  saved fig_matvec_error.pdf")

# Memory
fig3 = plot(; size=(480, 360), xlabel="N", ylabel="memory [MB]",
    xscale=:log2, yscale=:log10, title="Operator storage",
    legend=:topleft, legendfontsize=9, framestyle=:box, background_color=:white)
for m in (:dense, :fft, :hmatrix, :fmm)
    r = results[m]
    isempty(r.n) && continue
    plot!(fig3, r.n, max.(r.mem, 1e-4); color=cols[m], marker=marks[m], label=String(m))
end
savefig(fig3, joinpath(OUTDIR, "fig_memory.pdf"))
savefig(fig3, joinpath(OUTDIR, "fig_memory.png"))
println("  saved fig_memory.pdf")

# -----------------------------------------------------------------------------
# 2) Hertz line contact pressure (forward) + wear evolution
# -----------------------------------------------------------------------------
println("\n[2] Hertz line pressure + wear demo")
N = 512
dad = HalfSpace2D(-2.0, 2.0, N; E=1.0)
R = 1.0
# initial cylindrical gap
dad.h0 .= dad.x .^ 2 ./ (2R)
W = 0.05
Kfft = build_operator(dad, :fft)
p, g = contact_pressure_force(dad, Kfft, W; err_tol=1e-6, it_max=300)

# analytical Hertz with E* = E (dad.E already contact modulus)
a = sqrt(4 * W * R / (π * dad.E))
p0 = 2 * W / (π * a)
p_hz = [abs(xi) < a ? p0 * sqrt(1 - (xi / a)^2) : 0.0 for xi in dad.x]

fig4 = plot(dad.x, p; color=:dodgerblue, label="BEM (FFT)",
    xlabel="x", ylabel="p(x)", title="Line contact pressure",
    legend=:topright, legendfontsize=9, size=(480, 360), framestyle=:box,
    background_color=:white)
plot!(fig4, dad.x, p_hz; color=:black, linestyle=:dash, label="Hertz")
savefig(fig4, joinpath(OUTDIR, "fig_hertz_pressure.pdf"))
savefig(fig4, joinpath(OUTDIR, "fig_hertz_pressure.png"))
println("  saved fig_hertz_pressure.pdf  F_num=$(sum(p .* dad.al)) W=$W")

# Wear
println("  running wear_2d …")
w1, w2, ph = wear_2d(dad, Kfft, W; k_ar1=5e-4, k_ar2=1e-4, δ=0.02, nsteps=40,
    err_tol=1e-5, it_max=80)

steps_show = [1, 10, 20, 40]
cmap = palette(:viridis, length(steps_show))
fig5 = plot(; xlabel="x", ylabel="wear depth", title="Archard wear evolution",
    legend=:topright, legendfontsize=9, size=(480, 360), framestyle=:box,
    background_color=:white)
for (k, s) in enumerate(steps_show)
    s <= size(w1, 2) || continue
    plot!(fig5, dad.x, w1[:, s] .+ w2[:, s]; color=cmap[k], label="step $s")
end
savefig(fig5, joinpath(OUTDIR, "fig_wear_evolution.pdf"))
savefig(fig5, joinpath(OUTDIR, "fig_wear_evolution.png"))
println("  saved fig_wear_evolution.pdf")

fig6 = plot(; xlabel="x", ylabel="p(x)", title="Pressure during wear",
    legend=:topright, legendfontsize=9, size=(480, 360), framestyle=:box,
    background_color=:white)
for (k, s) in enumerate(steps_show)
    s <= size(ph, 2) || continue
    plot!(fig6, dad.x, ph[:, s]; color=cmap[k], label="step $s")
end
savefig(fig6, joinpath(OUTDIR, "fig_wear_pressure.pdf"))
savefig(fig6, joinpath(OUTDIR, "fig_wear_pressure.png"))
println("  saved fig_wear_pressure.pdf")

# -----------------------------------------------------------------------------
# 3) Contact solve time by backend
# -----------------------------------------------------------------------------
println("\n[3] Full contact solve timing")
ns2 = [128, 256, 512]
t_contact = Dict(m => Float64[] for m in (:dense, :fft, :hmatrix, :fmm))
for n in ns2
    dad = HalfSpace2D(-1.5, 1.5, n; E=1.0)
    dad.h0 .= dad.x .^ 2 ./ 2
    W = 0.03
    for m in (:dense, :fft, :hmatrix, :fmm)
        try
            K = m === :hmatrix ? build_operator(dad, m; nmax=24, atol=1e-5) :
                m === :fmm ? build_operator(dad, m; θ=0.75) :
                build_operator(dad, m)
            t0 = time_ns()
            contact_pressure_force(dad, K, W; err_tol=1e-5, it_max=100)
            dt = (time_ns() - t0) / 1e9
            push!(t_contact[m], dt)
            @printf("  N=%4d  %-8s  contact solve = %.3f s\n", n, m, dt)
        catch e
            push!(t_contact[m], NaN)
            @warn "contact $m n=$n failed" exception=e
        end
    end
end

fig7 = plot(; xlabel="N", ylabel="contact solve time [s]",
    xscale=:log2, yscale=:log10, title="Force-controlled contact solve",
    legend=:topleft, legendfontsize=9, size=(480, 360), framestyle=:box,
    background_color=:white)
for m in (:dense, :fft, :hmatrix, :fmm)
    ts = t_contact[m]
    ok = findall(isfinite, ts)
    isempty(ok) && continue
    plot!(fig7, ns2[ok], ts[ok]; color=cols[m], marker=marks[m], label=String(m))
end
savefig(fig7, joinpath(OUTDIR, "fig_contact_solve_time.pdf"))
savefig(fig7, joinpath(OUTDIR, "fig_contact_solve_time.png"))
println("  saved fig_contact_solve_time.pdf")

# -----------------------------------------------------------------------------
# Save raw data as CSV for the paper
# -----------------------------------------------------------------------------
open(joinpath(OUTDIR, "scaling_data.csv"), "w") do io
    println(io, "method,n,t_build,t_matvec,rel_err,mem_MB")
    for m in (:dense, :fft, :hmatrix, :fmm)
        r = results[m]
        for i in eachindex(r.n)
            @printf(io, "%s,%d,%.6e,%.6e,%.6e,%.6e\n",
                m, r.n[i], r.t_build[i], r.t_mv[i], r.err[i], r.mem[i])
        end
    end
end
println("\nAll figures → $OUTDIR")
println("Done.")
