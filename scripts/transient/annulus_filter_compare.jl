# Figure-BC annulus (outer u=0, inner ∂u/∂n=1) vs Bessel series.
# julia --project=. scripts/annulus_filter_compare.jl
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf, Statistics
using SpecialFunctions: besselj0, bessely0, besselj1, bessely1
using Plots
using LaTeXStrings
gr()


default(fontfamily="Computer Modern", linewidth=1.6, framestyle=:box,
        grid=false, dpi=160, size=(560, 360), legendfontsize=8)

include(datadir("Laplace", "Laplace_dad.jl"))
include(datadir("Laplace", "potencial_problems.jl"))
include(datadir("Laplace", "wave_propagation.jl"))

const OUT = get(ENV, "STUDY_OUT",
    raw"C:\Users\lucas.s.campos\OneDrive\artigos\escritos\2026\SBM transient\results")
const FIG = joinpath(OUT, "figures")
mkpath(FIG)

const A = 1.0
const B = 5.0

function fig_roots(a=A, b=B; N=30, nscan=8000)
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

Zn(λ, r, b=B) = besselj0(λ * r) * bessely0(λ * b) - besselj0(λ * b) * bessely0(λ * r)
us(r, a=A, b=B) = a * log(b / max(r, a * 1e-15))

function fig_coeffs(roots; a=A, b=B, nq=96)
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

function u_ana(r, t, roots, Cn; a=A, b=B)
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
    roots = fig_roots()
    Cn = fig_coeffs(roots)
    @printf("modes=%d  λ1=%.4f  us(a)=%.4f  u(a,0)=%.3e\n",
        length(roots), roots[1], us(A), u_ana(A, 0.0, roots, Cn))

    dad, _ = wave_problem(:annulus; ndiv=16, n_int=12)
    set_internal_annulus!(dad; nr=12, nθ=12, a=A, b=B, pad=0.15)
    set_fig_bc!(dad)
    H_G_full_direct(dad; npg=12, threaded=false)
    DIBEM(dad; method=:dense, rbf=PHS(1; poly_deg=-1))
    tf, dts = 12.0, 0.08
    sol = solve_transient_o2(dad, dts, tf; filter_unstable=true,
                             abstol=1e-5, reltol=1e-5, progress=false)
    t = dad.time
    probe = Point2D(A / sqrt(2), A / sqrt(2))
    pts = vcat(dad.Nodes, dad.internalNodes)
    ip = argmin(norm(p - probe) for p in pts)
    rprobe = hypot(pts[ip][1], pts[ip][2])
    un = dad.T[ip, :]
    ua = [u_ana(rprobe, ti, roots, Cn) for ti in t]
    rel = norm(un - ua) / norm(ua)
    @printf("n_drop=%s  probe r=%.3f  rel=%.3e  max num/ana=%.3f/%.3f\n",
        string(has_cache(dad, :wave_n_drop) ? dad.wave_n_drop : "?"),
        rprobe, rel, maximum(abs, un), maximum(abs, ua))

    plt = plot(t, ua; color=:black, ls=:dash, label="series",
               xlabel=L"t", ylabel=L"u",
               title="inner arc, filtered Rodas")
    plot!(plt, t, un; color=:crimson, label="Rodas + Schur")
    savefig(plt, joinpath(FIG, "annulus_filter_inner.pdf"))
    savefig(plt, joinpath(FIG, "annulus_filter_inner.png"))
    println("wrote ", joinpath(FIG, "annulus_filter_inner.png"))
    return plt
end

main()
