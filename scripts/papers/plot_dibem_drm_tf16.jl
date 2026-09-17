# DIBEM vs DRM sudden bar, 3 meshes, tf=16. Houbolt (no MMM) and MMM figures.
# julia --project=. scripts/plot_dibem_drm_tf16.jl
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf, Statistics
using Plots, LaTeXStrings
gr()
default(fontfamily="Computer Modern", linewidth=1.5, framestyle=:box,
    grid=false, dpi=160, size=(720, 400), legendfontsize=8)

include(datadir("Laplace", "Laplace_dad.jl"))
include(datadir("Laplace", "potencial_problems.jl"))
include(datadir("Laplace", "wave_propagation.jl"))
include(datadir("elastico", "iso", "bar_sudden.jl"))

# format2d centroids: ni = (ndiv-1)², nb = 8(ndiv-1).
# ndiv=8 → nb=56 ni=49; ndiv=16 → 120/225; ndiv=29 → 224/784 (nt≈1000).
const DISCS = (8, 16, 29)
const Δt = 0.04
const tf = 16.0
const PROBE = Point2D(1.0, 0.5)

function make_bar(ndiv)
    msh = mesh_elasticity_bar(; ndiv=ndiv, L=1.0, P=1.0, nome="bar_tf16_cent_n$(ndiv)")
    props = Elasticity(E=1.0, nu=0.0, rho=1.0; plane_stress=true)
    dad = format2d(msh, props; tipo=1, pontointerno=true)
    attach_analytical!(dad, ana_bar_sudden(; N=400, c=1.0, L=1.0))
    return dad
end

function probe_ux(U, dad, probe=PROBE)
    pts = vcat(dad.Nodes, dad.internalNodes)
    ip = argmin(norm(p - probe) for p in pts)
    return U[2 * (ip - 1) + 1, :]
end

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
    return U, t
end

function lab(method, dad)
    return "$(method)  \$n_b=$(dad.n)\$, \$n_i=$(dad.ni)\$"
end

ana = ana_bar_sudden(; N=400, c=1.0, L=1.0)
t_ref = collect(0:Δt:tf)
ua = [ana.u(PROBE; t=ti) for ti in t_ref]

runs = []
for ndiv in DISCS
    @printf("\n--- mesh ndiv=%d (format2d centroids) ---\n", ndiv)
    dad0 = make_bar(ndiv)
    @printf("nb=%d ni=%d nt=%d\n", dad0.n, dad0.ni, dad0.nt)
    flush(stdout)
    H_G_full_direct(dad0; npg=8, threaded=true)

    dadD = deepcopy(dad0)
    @printf("  DIBEM ...\n"); flush(stdout)
    DIBEM(dadD; method=:dense, rbf=PHS(1; poly_deg=-1))
    UD, tD = houbolt_condensed!(dadD, Δt, tf)
    uxD = probe_ux(UD, dadD)
    relD = norm(uxD .- ua) / (norm(ua) + eps())
    @printf("  DIBEM Houbolt  rel=%.3e  max=%.3f\n", relD, maximum(abs, uxD))

    dadR = deepcopy(dad0)
    @printf("  DRM ...\n"); flush(stdout)
    build_drm_matrices(dadR; npg=8)
    UR, tR = houbolt_condensed!(dadR, Δt, tf)
    uxR = probe_ux(UR, dadR)
    relR = norm(uxR .- ua) / (norm(ua) + eps())
    @printf("  DRM Houbolt    rel=%.3e  max=%.3f\n", relR, maximum(abs, uxR))

    dadDM = deepcopy(dadD)
    @printf("  DIBEM MMM ...\n"); flush(stdout)
    UmD, tmD, bD = solve_mmm!(dadDM, Δt, tf; alg=:houbolt)
    uxMD = probe_ux(UmD, dadDM)
    relMD = norm(uxMD .- ua) / (norm(ua) + eps())
    @printf("  DIBEM MMM      rel=%.3e  max=%.3f  nmodes=%d\n",
        relMD, maximum(abs, uxMD), length(bD.ω))

    dadRM = deepcopy(dadR)
    @printf("  DRM MMM ...\n"); flush(stdout)
    UmR, tmR, bR = solve_mmm!(dadRM, Δt, tf; alg=:houbolt)
    uxMR = probe_ux(UmR, dadRM)
    relMR = norm(uxMR .- ua) / (norm(ua) + eps())
    @printf("  DRM MMM        rel=%.3e  max=%.3f  nmodes=%d\n",
        relMR, maximum(abs, uxMR), length(bR.ω))

    push!(runs, (; dad=dad0, tD, uxD, uxR, tmD, uxMD, tmR, uxMR,
        labD=lab("DIBEM", dad0), labR=lab("DRM", dad0),
        relD, relR, relMD, relMR))
end

cols = [:steelblue, :darkorange, :seagreen]
mkpath(joinpath(projectdir(), "plots"))

finite_or_nan(u) = (v = copy(u); v[.!isfinite.(v) .| (abs.(v) .> 20)] .= NaN; v)

pltH = plot(t_ref, ua; color=:black, ls=:dash, label="1D series",
    xlabel=L"t", ylabel=L"u_x(L, L/2)", ylim=(-0.2, 2.4),
    title="elast bar sudden  Houbolt  \$t_f=16\$")
for (k, r) in enumerate(runs)
    plot!(pltH, r.tD, finite_or_nan(r.uxD); color=cols[k], label=r.labD)
    plot!(pltH, r.tD, finite_or_nan(r.uxR); color=cols[k], ls=:dashdot, label=r.labR)
end
outH = joinpath(projectdir(), "plots", "dibem_drm_houbolt_tf16")
savefig(pltH, outH * ".png")
savefig(pltH, outH * ".pdf")
println("wrote ", outH * ".png")

pltM = plot(t_ref, ua; color=:black, ls=:dash, label="1D series",
    xlabel=L"t", ylabel=L"u_x(L, L/2)", ylim=(-0.2, 2.4),
    title="elast bar sudden  MMM  \$t_f=16\$")
for (k, r) in enumerate(runs)
    plot!(pltM, r.tmD, finite_or_nan(r.uxMD); color=cols[k], label=r.labD)
    plot!(pltM, r.tmR, finite_or_nan(r.uxMR); color=cols[k], ls=:dashdot, label=r.labR)
end
outM = joinpath(projectdir(), "plots", "dibem_drm_mmm_tf16")
savefig(pltM, outM * ".png")
savefig(pltM, outM * ".pdf")
println("wrote ", outM * ".png")
