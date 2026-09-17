# DIBEM + Houbolt (no MMM) on the sudden bar, with properly inset internals.
# Also reports format2d centroids vs set_internal_grid!.
# julia --project=. scripts/plot_dibem_bar_houbolt.jl
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

function min_int_dist(dad)
    isempty(dad.internalNodes) && return NaN
    return minimum(norm(p - q) for p in dad.internalNodes, q in dad.Nodes)
end

function nneg_M(M)
    ev = real.(eigvals(M))
    return count(<( -1e-8), ev), minimum(ev), tr(M)
end

function make_bar(; ndiv=8, internals=:centroids, pad=0.15, n_int=4)
    msh = mesh_elasticity_bar(; ndiv=ndiv, L=1.0, P=1.0, nome="bar_$(internals)_n$ndiv")
    props = Elasticity(E=1.0, nu=0.0, rho=1.0; plane_stress=true)
    if internals === :centroids
        dad = format2d(msh, props; tipo=1, pontointerno=true)
    else
        dad = format2d(msh, props; tipo=1, pontointerno=false)
        if internals === :grid
            set_internal_grid!(dad; nx=n_int, ny=n_int, x=(0, 1), y=(0, 1), pad=pad)
        end
    end
    ana = ana_bar_sudden(; N=400, c=1.0, L=1.0)
    attach_analytical!(dad, ana)
    return dad, (; probe=Point2D(1.0, 0.5), ana)
end

function report(tag, dad)
    dmin = min_int_dist(dad)
    xs = isempty(dad.internalNodes) ? Float64[] : getindex.(dad.internalNodes, 1)
    @printf("%-28s n=%3d ni=%3d nt=%3d  xmin_int=%s  dmin=%.3e\n",
        tag, dad.n, dad.ni, dad.nt,
        isempty(xs) ? "—" : @sprintf("%.4f", minimum(xs)), dmin)
end

# Houbolt on condensed free DOFs: M̄ ü + K̄ u = f  (no modal truncation).
# Steps 2–3: implicit Euler on (u̇, v̇)=(v, M⁻¹(f−Ku)); then classical Houbolt.
function houbolt_condensed!(dad, Δt, tf)
    sys = build_modal_system(dad)
    M, K, f0, free = sys.M, sys.K, sys.f0, sys.free
    ndof, nb = BEM._neq(dad)
    t = collect(0:Δt:tf)
    nT = length(t)
    Y = zeros(length(free), nT)
    V = zeros(length(free))
    nT >= 2 || (set_cache!(dad; u=zeros(ndof, nT), time=t); return zeros(ndof, nT), sys)
    Ae = factorize(M .+ (Δt^2) .* K)
    for i in 2:min(3, nT)
        a = Ae \ (f0 .- K * (Y[:, i - 1] .+ Δt .* V))
        Y[:, i] .= Y[:, i - 1] .+ Δt .* V .+ (Δt^2) .* a
        V .+= Δt .* a
    end
    nT >= 4 || begin
        U = zeros(ndof, nT)
        q = zeros(nb, nT)
        for j in 1:nT
            U[free, j] .= Y[:, j]
            BEM._pin_known!(view(U, :, j), dad)
            BEM._scatter_step!(dad, view(U, :, j), view(q, :, j))
        end
        set_cache!(dad; u=U, T=U, traction=q, q=q, time=t)
        return U, sys
    end
    Mt = M ./ Δt^2
    A = factorize(K .+ 2 .* Mt)
    for i in 4:nT
        rhs = f0 .+ Mt * (5 .* Y[:, i - 1] .- 4 .* Y[:, i - 2] .+ Y[:, i - 3])
        Y[:, i] .= A \ rhs
    end
    U = zeros(ndof, nT)
    q = zeros(nb, nT)
    for j in 1:nT
        U[free, j] .= Y[:, j]
        BEM._pin_known!(view(U, :, j), dad)
        BEM._scatter_step!(dad, view(U, :, j), view(q, :, j))
    end
    set_cache!(dad; u=U, T=U, traction=q, q=q, time=t)
    return U, sys
end

function probe_ux(U, dad, probe)
    pts = vcat(dad.Nodes, dad.internalNodes)
    ip = argmin(norm(p - probe) for p in pts)
    return U[2 * (ip - 1) + 1, :], ip
end

println("========== Houbolt DIBEM (Euler start) ==========")
ndiv, Δt, tf = 8, 0.04, 8.0
dad, meta = make_bar(; ndiv=ndiv, internals=:centroids)
H_G_full_direct(dad; npg=8, threaded=false)
DIBEM(dad; method=:dense, rbf=PHS(1; poly_deg=-1))
@printf("ni=%d dmin=%.3e  nneg(M)=%d\n", dad.ni, min_int_dist(dad), nneg_M(Matrix(dad.M))[1])

U, sys = houbolt_condensed!(dad, Δt, tf)
t = dad.time
ux, ip = probe_ux(U, dad, meta.probe)
ua = [meta.ana.u(meta.probe; t=ti) for ti in t]
rel = norm(ux .- ua) / (norm(ua) + eps())
@printf("Houbolt (Euler steps 2–3)  rel=%.3e  max=%.3f  u(Δt)=%.3f  probe=%s\n",
    rel, maximum(abs, ux), ux[2], all_points(dad)[ip])

plt = plot(t, ua; color=:black, ls=:dash, label="1D series")
plot!(plt, t, ux; color=:steelblue, label="Houbolt + DIBEM")
plot!(plt; xlabel=L"t", ylabel=L"u_x(L, L/2)",
    title="elast bar sudden  DIBEM  format2d centroids  ndiv=$ndiv")
mkpath(joinpath(projectdir(), "plots"))
out = joinpath(projectdir(), "plots", "dibem_bar_houbolt")
savefig(plt, out * ".png")
savefig(plt, out * ".pdf")
println("wrote ", out * ".png")
