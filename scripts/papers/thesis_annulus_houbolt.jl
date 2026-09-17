# Santos §4.2 figure BCs + Houbolt on a fine polar cloud.
# julia --project=. scripts/thesis_annulus_houbolt.jl
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf, Statistics
using SpecialFunctions: besselj0, bessely0, besselj1, bessely1
using Plots, LaTeXStrings
gr()
default(fontfamily="Computer Modern", linewidth=1.5, framestyle=:box,
        grid=false, dpi=160, size=(560, 360), legendfontsize=8)

include(datadir("Laplace", "Laplace_dad.jl"))
include(datadir("Laplace", "potencial_problems.jl"))
include(datadir("Laplace", "wave_propagation.jl"))

const OUT = get(ENV, "STUDY_OUT",
    raw"C:\Users\lucas.s.campos\OneDrive\artigos\escritos\2026\SBM transient\results")
const FIG = joinpath(OUT, "figures")
mkpath(FIG)

const RA, RB = 1.0, 5.0

function fig_roots(a=RA, b=RB; N=30, nscan=8000)
    f(λ) = besselj1(λ * a) * bessely0(λ * b) - besselj0(λ * b) * bessely1(λ * a)
    roots = Float64[]
    λs = range(1e-6, N * π / (b - a) * 4; length=nscan)
    for i in 1:(length(λs) - 1)
        f1, f2 = f(λs[i]), f(λs[i + 1])
        f1 * f2 < 0 || continue
        lo, hi = λs[i], λs[i + 1]
        for _ in 1:60
            mid = 0.5 * (lo + hi)
            f(mid) * f(lo) <= 0 ? (hi = mid) : (lo = mid)
        end
        push!(roots, 0.5 * (lo + hi))
        length(roots) >= N && break
    end
    return roots
end

Zn(λ, r, b=RB) = besselj0(λ * r) * bessely0(λ * b) - besselj0(λ * b) * bessely0(λ * r)
us(r, a=RA, b=RB) = a * log(b / max(r, a * 1e-15))

function fig_coeffs(roots; a=RA, b=RB, nq=96)
    ξ, w = gausslegendre(nq)
    jr = (b - a) / 2
    Cn = zeros(length(roots))
    for (n, λ) in enumerate(roots)
        num = den = 0.0
        for (g, ww) in zip(ξ, w)
            r = (b + a) / 2 + jr * g
            Z = Zn(λ, r, b)
            num += ww * jr * r * us(r, a, b) * Z
            den += ww * jr * r * Z^2
        end
        Cn[n] = -num / (den + 1e-30)
    end
    return Cn
end

function u_ana(r, t, roots, Cn; a=RA, b=RB)
    s = us(r, a, b)
    @inbounds for n in eachindex(roots)
        s += Cn[n] * Zn(roots[n], r, b) * cos(roots[n] * t)
    end
    return s
end

function set_fig_bc!(dad)
    for i in 1:dad.n
        p = dad.Nodes[i]
        r = hypot(p[1], p[2])
        nr = (dad.Normal[i][1] * p[1] + dad.Normal[i][2] * p[2]) / r
        if nr > 0.9
            dad.BC[i] = 0
            dad.BV[i] = 0.0
        elseif nr < -0.9
            dad.BC[i] = 1
            dad.BV[i] = -1.0
        else
            dad.BC[i] = 1
            dad.BV[i] = 0.0
        end
    end
    return dad
end

function main()
    ndiv, nint, Δt, tf = 80, 36, 0.04, 80.0
    roots = fig_roots()
    Cn = fig_coeffs(roots)
    dad, _ = wave_problem(:annulus; ndiv=ndiv, n_int=nint)
    set_internal_annulus!(dad; nr=nint, nθ=nint, a=RA, b=RB, pad=0.15)
    set_fig_bc!(dad)
    @printf("mesh N=%d ni=%d nt=%d  Δt=%.3f tf=%.1f steps=%d\n",
        dad.n, dad.ni, dad.nt, Δt, tf, length(0:Δt:tf))
    @time H_G_full_direct(dad; npg=12, threaded=false)
    @time DIBEM(dad; method=:dense, rbf=PHS(1; poly_deg=-1))
    @time solve_Houbolt(dad, Δt, tf)
    t = dad.time
    T = dad.T
    ok = all(isfinite, T)
    mx = maximum(abs, T)
    probe = Point2D(RA / sqrt(2), RA / sqrt(2))
    pts = vcat(dad.Nodes, dad.internalNodes)
    ip = argmin(norm(p - probe) for p in pts)
    rprobe = hypot(pts[ip][1], pts[ip][2])
    un = T[ip, :]
    ua = [u_ana(rprobe, ti, roots, Cn) for ti in t]
    rel = (ok && norm(ua) > 0) ? norm(un - ua) / norm(ua) : NaN
    @printf("Houbolt finite=%s  max|T|=%.3e  probe r=%.3f  rel=%.3e  max num/ana=%.3f/%.3f\n",
        string(ok), mx, rprobe, rel, maximum(abs, un), maximum(abs, ua))

    plt = plot(t, ua; color=:black, ls=:dash, label="series",
               xlabel=L"t", ylabel=L"u",
               title="inner arc, Houbolt fine mesh")
    plot!(plt, t, un; color=:crimson, label="Houbolt")
    savefig(plt, joinpath(FIG, "annulus_houbolt_fine.pdf"))
    savefig(plt, joinpath(FIG, "annulus_houbolt_fine.png"))
    println("wrote ", joinpath(FIG, "annulus_houbolt_fine.png"))
    return nothing
end

if endswith(replace(PROGRAM_FILE, "\\" => "/"), "thesis_annulus_houbolt.jl")
    main()
end
