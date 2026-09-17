# How small can Δt get before condensed Houbolt on the 12×6 DRM bar blows up?
# julia --project=. scripts/sweep_dt_bar12x6.jl
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf, Statistics

include(joinpath(@__DIR__, "..", "papers", "plot_bar_rect12x6.jl"))

dad0 = make_dad()
H_G_full_direct(dad0; npg=12, threaded=false)
drm_mass!(dad0; flip=true)
sys = build_modal_system(dad0)
evM = real.(eigvals(sys.M))
@printf("n=%d ni=%d  nneg(M̄)=%d  minλ(M̄)=%.3e  maxλ(M̄)=%.3e  cond(M̄)=%.3e\n",
    dad0.n, dad0.ni, count(<( -1e-8), evM), minimum(evM), maximum(evM),
    maximum(abs, evM) / (minimum(abs, evM) + 1e-30))
@printf("tr(K)=%.3e  nneg(K)=%d\n", tr(sys.K), count(<( -1e-8), real.(eigvals(sys.K))))

ana = ana_bar_sudden(; N=400, c=1.0, L=Lx)
# Yee upper bound ΔL/c ≈ 1 (quadratic element length on the long side)
dts = (2.0, 1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001)
println("Δt        nT     finite   max|ux|     rel        cond(K+2M/Δt²)  first |u|>50 at")
for Δt in dts
    dad = deepcopy(dad0)
    U, t, _ = houbolt_condensed!(dad, Δt, tf)
    ux = probe_ux(U, dad)
    ua = [ana.u(PROBE; t=ti) for ti in t]
    fin = all(isfinite, ux)
    mx = maximum(abs, filter(isfinite, ux); init=0.0)
    r = fin ? rel(ux, ua) : NaN
    A = sys.K .+ (2 / Δt^2) .* sys.M
    κ = try
        cond(A)
    catch
        NaN
    end
    blow = findfirst(i -> !isfinite(ux[i]) || abs(ux[i]) > 50, eachindex(ux))
    tblow = blow === nothing ? "—" : @sprintf("t=%.3f (step %d)", t[blow], blow)
    @printf("%8.4f  %6d  %-7s  %10.3e  %9.3e  %10.3e  %s\n",
        Δt, length(t), string(fin), mx, r, κ, tblow)
    flush(stdout)
end
