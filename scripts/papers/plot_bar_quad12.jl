# Sudden bar: 12 quadratic elements/side, 3×3 internals. DIBEM vs DRM (f=r).
# julia --project=. scripts/plot_bar_quad12.jl
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf, Statistics
using Plots, LaTeXStrings
gr()
default(fontfamily="Computer Modern", linewidth=1.6, framestyle=:box,
    grid=false, dpi=160, size=(640, 380), legendfontsize=8)

include(datadir("Laplace", "Laplace_dad.jl"))
include(datadir("Laplace", "potencial_problems.jl"))
include(datadir("Laplace", "wave_propagation.jl"))
include(datadir("elastico", "iso", "bar_sudden.jl"))

spd_abs(A) = (F = eigen(Symmetric(0.5 .* (A .+ A'))); F.vectors * (abs.(F.values) .* F.vectors'))

function houbolt_condensed!(dad, Δt, tf)
    sys = build_modal_system(dad)
    M, K, f0, free = spd_abs(sys.M), sys.K, sys.f0, sys.free
    ndof, nb = BEM._neq(dad)
    t = collect(0:Δt:tf)
    nT = length(t)
    Y = zeros(length(free), nT)
    V = zeros(length(free))
    Ae = factorize(M .+ (Δt^2) .* K)
    for i in 2:min(3, nT)
        a = Ae \ (f0 .- K * (Y[:, i - 1] .+ Δt .* V))
        Y[:, i] .= Y[:, i - 1] .+ Δt .* V .+ (Δt^2) .* a
        V .+= Δt .* a
    end
    if nT >= 4
        Mt = M ./ Δt^2
        A = factorize(K .+ 2 .* Mt)
        for i in 4:nT
            rhs = f0 .+ Mt * (5 .* Y[:, i - 1] .- 4 .* Y[:, i - 2] .+ Y[:, i - 3])
            Y[:, i] .= A \ rhs
        end
    end
    U = zeros(ndof, nT)
    q = zeros(nb, nT)
    for j in 1:nT
        U[free, j] .= Y[:, j]
        BEM._pin_known!(view(U, :, j), dad)
        BEM._scatter_step!(dad, view(U, :, j), view(q, :, j))
    end
    set_cache!(dad; u=U, T=U, traction=q, q=q, time=t)
    return U, t, sys
end

function probe_ux(U, dad, probe)
    pts = vcat(dad.Nodes, dad.internalNodes)
    ip = argmin(norm(p - probe) for p in pts)
    return U[2 * (ip - 1) + 1, :], ip
end

# 12 quadratic elements/side: transfinite 13 nodes, then order 2.
dad, meta = elasticity_bar_sudden(; ndiv=13, n_int=3, tipo=2, ordem=2, ν=0.0, pad=0.15)
@printf("elements=%d  n=%d  ni=%d  nt=%d  tipo=%s\n",
    length(dad.elements), dad.n, dad.ni, dad.nt, typeof(dad.element_type))
@assert length(dad.elements) == 48 "expected 12 elements × 4 sides"
@assert dad.ni == 9
@assert dad.n == 48 * 3  # quadratic discontinuous: 3 colloc / element

Δt, tf = 0.04, 8.0
H_G_full_direct(dad; npg=12, threaded=false)

dadD = deepcopy(dad)
DIBEM(dadD; method=:dense, rbf=PHS(1; poly_deg=-1))
UD, t, sysD = houbolt_condensed!(dadD, Δt, tf)
uxD, _ = probe_ux(UD, dadD, meta.probe)
dadDM = deepcopy(dadD)
UmD, tmD, bD = solve_mmm!(dadDM, Δt, tf; alg=:houbolt)
uxMD, _ = probe_ux(UmD, dadDM, meta.probe)

dadR = deepcopy(dad)
build_drm_matrices(dadR; npg=12, kernel=:r)
UR, _, sysR = houbolt_condensed!(dadR, Δt, tf)
uxR, _ = probe_ux(UR, dadR, meta.probe)
dadRM = deepcopy(dadR)
UmR, tmR, bR = solve_mmm!(dadRM, Δt, tf; alg=:houbolt)
uxMR, _ = probe_ux(UmR, dadRM, meta.probe)

ua = [meta.ana.u(meta.probe; t=ti) for ti in t]
rel(a, b) = norm(a .- b) / (norm(b) + eps())
@printf("DIBEM Houbolt  rel=%.3e max=%.3f  nneg(M)=%d\n",
    rel(uxD, ua), maximum(abs, uxD), count(<( -1e-8), real.(eigvals(Matrix(dadD.M)))))
@printf("DRM   Houbolt  rel=%.3e max=%.3f  nneg(M)=%d\n",
    rel(uxR, ua), maximum(abs, uxR), count(<( -1e-8), real.(eigvals(Matrix(dadR.M)))))
@printf("DIBEM MMM      rel=%.3e max=%.3f  nmodes=%d ω1=%.4f\n",
    rel(uxMD, ua), maximum(abs, uxMD), length(bD.ω), bD.ω[1])
@printf("DRM   MMM      rel=%.3e max=%.3f  nmodes=%d ω1=%.4f\n",
    rel(uxMR, ua), maximum(abs, uxMR), length(bR.ω), bR.ω[1])

plt = plot(t, ua; color=:black, ls=:dash, label="1D series")
plot!(plt, t, uxD; color=:steelblue, label="DIBEM Houbolt")
plot!(plt, t, uxR; color=:darkorange, ls=:dashdot, label="DRM (\$f=r\$) Houbolt")
plot!(plt, tmD, uxMD; color=:steelblue, ls=:dot, label="DIBEM MMM")
plot!(plt, tmR, uxMR; color=:darkorange, ls=:dot, label="DRM MMM")
plot!(plt; xlabel=L"t", ylabel=L"u_x(L, L/2)", ylim=(-0.2, 2.4),
    title="quad 12/side, \$n_i=9\$  \$n_b=$(dad.n)\$")
mkpath(joinpath(projectdir(), "plots"))
out = joinpath(projectdir(), "plots", "bar_quad12_ni9")
savefig(plt, out * ".png")
savefig(plt, out * ".pdf")
println("wrote ", out * ".png")
