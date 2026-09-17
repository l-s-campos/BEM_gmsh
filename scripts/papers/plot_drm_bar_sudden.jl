# DRM sudden bar, Newmark on condensed free DOFs (no MMM).
# julia --project=. scripts/plot_drm_bar_sudden.jl
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf, Statistics
using Plots, LaTeXStrings
gr()
default(fontfamily="Computer Modern", linewidth=1.6, framestyle=:box,
    grid=false, dpi=160, size=(640, 380), legendfontsize=9)

include(datadir("Laplace", "Laplace_dad.jl"))
include(datadir("Laplace", "potencial_problems.jl"))
include(datadir("Laplace", "wave_propagation.jl"))
include(datadir("elastico", "iso", "bar_sudden.jl"))

function probe_ux(U, dad, probe)
    pts = vcat(dad.Nodes, dad.internalNodes)
    ip = argmin(norm(p - probe) for p in pts)
    return U[2 * (ip - 1) + 1, :], ip
end

# Average-acceleration Newmark on M̈u + K u = f. Negative-mass modes of the
# BEM operator are flipped (abs of symmetric eigenvalues) so the step is stable.
function newmark_condensed!(dad, Δt, tf)
    sys = build_modal_system(dad)
    S = Symmetric(0.5 .* (sys.M .+ sys.M'))
    F = eigen(S)
    M = F.vectors * (abs.(F.values) .* F.vectors')
    K, f0, free = sys.K, sys.f0, sys.free
    ndof, nb = BEM._neq(dad)
    t = collect(0:Δt:tf)
    nT = length(t)
    nf = length(free)
    β, γ = 0.25, 0.5
    Y = zeros(nf, nT)
    V = zeros(nf)
    Aacc = M \ (f0 - K * Y[:, 1])
    Afact = factorize(M .+ (β * Δt^2) .* K)
    for i in 2:nT
        Up = Y[:, i - 1] .+ Δt .* V .+ (Δt^2 * (0.5 - β)) .* Aacc
        Vp = V .+ (Δt * (1 - γ)) .* Aacc
        Aacc = Afact \ (f0 - K * Up)
        Y[:, i] .= Up .+ (β * Δt^2) .* Aacc
        V .= Vp .+ (γ * Δt) .* Aacc
    end
    U = zeros(ndof, nT)
    q = zeros(nb, nT)
    for j in 1:nT
        U[free, j] .= Y[:, j]
        BEM._pin_known!(view(U, :, j), dad)
        BEM._scatter_step!(dad, view(U, :, j), view(q, :, j))
    end
    set_cache!(dad; u=U, T=U, traction=q, q=q, time=t)
    return U
end

ndiv, n_int = 8, 4
Δt, tf = 0.04, 8.0
dad, meta = elasticity_bar_sudden(; ndiv=ndiv, n_int=n_int, ν=0.0)
H_G_full_direct(dad; npg=8, threaded=false)
build_drm_matrices(dad; npg=8)
U = newmark_condensed!(dad, Δt, tf)
t = dad.time
ux, ip = probe_ux(U, dad, meta.probe)
ua = [meta.ana.u(meta.probe; t=ti) for ti in t]
rel = norm(ux .- ua) / (norm(ua) + eps())

@printf("DRM bar sudden  Newmark  nt=%d  rel=%.3e  max=%.3f  probe=%s\n",
    dad.nt, rel, maximum(abs, ux), all_points(dad)[ip])

plt = plot(t, ua; color=:black, ls=:dash, label="1D series")
plot!(plt, t, ux; color=:steelblue, label="Newmark + DRM")
plot!(plt; xlabel=L"t", ylabel=L"u_x(L, L/2)",
    title="elast bar sudden  DRM  ndiv=$ndiv  ν=0")
mkpath(joinpath(projectdir(), "plots"))
out = joinpath(projectdir(), "plots", "drm_bar_sudden")
savefig(plt, out * ".png")
savefig(plt, out * ".pdf")
println("wrote ", out * ".png")
