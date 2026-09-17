# Compare Guiggiani Laurent coefficients: polynomial interpolation (20 Gauss
# points on the singular element) vs Richardson, on integrals with closed forms.
using DrWatson
@quickactivate :BEM
using Printf, LinearAlgebra, FastGaussQuadrature

const ninterp = 20
const qsi, w = gausslegendre(16)

hfp(a, k::Int) = if k == 1
    log(abs((1 - a) / (1 + a)))
elseif k == 2
    -2 / (1 - a^2)
elseif k == 3
    -1 / (2 * (1 - a)^2) + 1 / (2 * (1 + a)^2)
elseif k == 4
    -1 / (3 * (1 - a)^3) - 1 / (3 * (1 + a)^3)
else
    throw(ArgumentError("k ∈ {1,2,3,4}"))
end

relerr(x::Number, y::Number) = abs(x - y) / max(abs(y), 1e-16)
relerr(A, B) = norm(A - B) / max(norm(B), 1e-16)
abserr(x, y) = abs(x - y)

function print_header(title)
    println()
    println("="^78)
    println(title)
    println("="^78)
end

# -----------------------------------------------------------------------------
print_header("1) Laurent coefficients of a polynomial series (exact)")
# f(ξ) = 5/(ξ-a)⁴ + 4/(ξ-a)³ + 3/(ξ-a)² + 2/(ξ-a) + 7
# Right-ray Richardson (s=+1) matches the signed interpolation coefficients.
a = 0.25
fpoly(ξ) = 5 / (ξ - a)^4 + 4 / (ξ - a)^3 + 3 / (ξ - a)^2 + 2 / (ξ - a) + 7
exact = (5.0, 4.0, 3.0, 2.0, 7.0)
Ci = laurent_coefficients_interp(fpoly, a, -4; n=ninterp)
CR = laurent_coefficients(ρ -> fpoly(a + ρ), 1e-3, Val(-4); maxeval=10, atol=1e-12)
@printf "  a = %.2f   ninterp = %d\n" a ninterp
@printf "  %-8s  %14s  %14s  %14s  %12s  %12s\n" "term" "exact" "interp" "Richardson" "err interp" "err Rich."
for (lab, e, i, r) in zip(("F₋₄", "F₋₃", "F₋₂", "F₋₁", "F₀"), exact, Ci, CR)
    @printf "  %-8s  %14.6e  %14.6e  %14.6e  %12.3e  %12.3e\n" lab e i r abserr(i, e) abserr(r, e)
end

# -----------------------------------------------------------------------------
print_header("2) Taylor coefficients of non-polynomial kernels")
a = 0.1
cases = (
    ("exp(u)/u²", ξ -> exp(ξ - a) / (ξ - a)^2, -2, (1.0, 1.0, 0.5)),
    ("cos(u)/u²", ξ -> cos(ξ - a) / (ξ - a)^2, -2, (1.0, 0.0, -0.5)),
    ("cos(u)/u⁴", ξ -> cos(ξ - a) / (ξ - a)^4, -4, (1.0, 0.0, -0.5, 0.0, 1 / 24)),
    ("e^u/u⁴ + e^u/u", ξ -> exp(ξ - a) / (ξ - a)^4 + exp(ξ - a) / (ξ - a), -4,
        (1.0, 1.0, 0.5, 1 + 1 / 6, 1 + 1 / 24)),
)
# e^u/u⁴ = u^{-4} + u^{-3} + u^{-2}/2 + u^{-1}/6 + 1/24 + …
# e^u/u  =           u^{-1} + 1 + u/2 + …
# sum    : F₋₄=1, F₋₃=1, F₋₂=1/2, F₋₁=1+1/6, F₀=1+1/24
for (name, f, o, ex) in cases
    local Ci = laurent_coefficients_interp(f, a, o; n=ninterp)
    local CR = laurent_coefficients(ρ -> f(a + ρ), 1e-3, Val(o); maxeval=10, atol=1e-12)
    println("  ", name, "   order=", o)
    nterm = length(ex)
    labs = o <= -3 ? ("F₋₄", "F₋₃", "F₋₂", "F₋₁", "F₀")[end-nterm+1:end] :
           ("F₋₂", "F₋₁", "F₀")[end-nterm+1:end]
    # pack: order -4 is 5-tuple; -2 is 3-tuple. `ex` is the non-leading-zero terms
    # aligned to the returned tuple.
    @printf "    %-6s  %14s  %14s  %14s  %12s  %12s\n" "term" "exact" "interp" "Richardson" "err interp" "err Rich."
    for k in 1:nterm
        e = ex[k]
        i = Ci[k]
        r = CR[k]
        @printf "    %-6s  %14.6e  %14.6e  %14.6e  %12.3e  %12.3e\n" labs[k] e i r abserr(i, e) abserr(r, e)
    end
end

# -----------------------------------------------------------------------------
print_header("3) CPV / HFP integrals  ∫_{-1}^{1} (ξ-a)^{-k} dξ")
@printf "  %-6s %-3s  %14s  %14s  %14s  %12s  %12s\n" "a" "k" "exact" "interp" "Richardson" "rel interp" "rel Rich."
for a in (-0.6, -0.3, 0.0, 0.2, 0.7)
    for k in (1, 2, 3, 4)
        exactI = hfp(a, k)
        f = ξ -> 1 / (ξ - a)^k
        Ii = guiggiani_integral(f, a, -k; qsi=qsi, w=w, laurent=:interp, ninterp=ninterp)
        Ir = guiggiani_integral(f, a, -k; qsi=qsi, w=w, h=1e-3, maxeval=10, atol=1e-12)
        @printf "  %6.2f %-3d  %14.6e  %14.6e  %14.6e  %12.3e  %12.3e\n" a k exactI Ii Ir relerr(Ii, exactI) relerr(Ir, exactI)
    end
end

# -----------------------------------------------------------------------------
print_header("4) Mixed integrand with a known HFP (polynomial remainder)")
# f = 5/(ξ-a)⁴ + 4/(ξ-a)³ + 3/(ξ-a)² + 2/(ξ-a) + 7 + 6(ξ-a)
# I = 5 HFP₄ + 4 HFP₃ + 3 HFP₂ + 2 CPV + 7*2 + 6*(-2a)
a = 0.3
fmix(ξ) = 5 / (ξ - a)^4 + 4 / (ξ - a)^3 + 3 / (ξ - a)^2 + 2 / (ξ - a) + 7 + 6 * (ξ - a)
Iex = 5 * hfp(a, 4) + 4 * hfp(a, 3) + 3 * hfp(a, 2) + 2 * hfp(a, 1) + 14 - 12 * a
Ii = guiggiani_integral(fmix, a, -4; qsi=qsi, w=w, laurent=:interp, ninterp=ninterp)
Ir = guiggiani_integral(fmix, a, -4; qsi=qsi, w=w, h=1e-3, maxeval=10, atol=1e-12)
@printf "  exact      = %.16e\n" Iex
@printf "  interp     = %.16e   rel = %.3e\n" Ii relerr(Ii, Iex)
@printf "  Richardson = %.16e   rel = %.3e\n" Ir relerr(Ir, Iex)

# -----------------------------------------------------------------------------
print_header("5) ninterp sweep on HFP ∫ (ξ)^{-4} dξ = -2/3  (a = 0)")
Iex = -2 / 3
@printf "  %-8s  %16s  %12s  %16s  %12s\n" "n" "interp I" "rel interp" "Richardson I" "rel Rich."
Ir = guiggiani_integral(ξ -> 1 / ξ^4, 0.0, -4; qsi=qsi, w=w, h=1e-3, maxeval=10, atol=1e-12)
for n in (4, 6, 8, 12, 16, 20, 24, 32)
    local Ii = guiggiani_integral(ξ -> 1 / ξ^4, 0.0, -4; qsi=qsi, w=w, laurent=:interp, ninterp=n)
    @printf "  %-8d  %16.8e  %12.3e  %16.8e  %12.3e\n" n Ii relerr(Ii, Iex) Ir relerr(Ir, Iex)
end

# -----------------------------------------------------------------------------
print_header("6) Laplace T (regular) and Kelvin T (CPV) on a straight element")
poly = BEM.Equispaced(1)
nodes = [Point2D(0.0, 0.0), Point2D(1.0, 0.0)]
a = 0.0
pf = Point2D(0.5, 0.0)
propsL = Laplace(1.0)
propsE = Elasticity(1.0, 0.3, 1.0)

function geom_at(ξ)
    N, dN = BEM.shapefun(poly, ξ)
    pg = N[1, 1] * nodes[1] + N[1, 2] * nodes[2]
    dx = dN[1, 1] * nodes[1] + dN[1, 2] * nodes[2]
    J = norm(dx)
    nrm = Point2D(dx[2], -dx[1]) / J
    return N, pg, J, nrm
end

flap = ξ -> begin
    N, pg, J, nrm = geom_at(ξ)
    r = pg - pf
    norm(r) < 1e-30 && return zeros(2), zeros(2)
    U, T = fundamental(propsL, r, nrm)
    Fg = [U * N[1, j] * J for j in 1:2]
    Fh = [T * N[1, j] * J for j in 1:2]
    return Fg, Fh
end
fkel = ξ -> begin
    N, pg, J, nrm = geom_at(ξ)
    r = pg - pf
    norm(r) < 1e-30 && return zeros(2, 4), zeros(2, 4)
    U, T = fundamental(propsE, r, nrm)
    Fg = zeros(2, 4)
    Fh = zeros(2, 4)
    for j in 1:2
        cols = (2j - 1):(2j)
        Fg[:, cols] .= U .* (N[1, j] * J)
        Fh[:, cols] .= T .* (N[1, j] * J)
    end
    return Fg, Fh
end

function report_GH(lab, fker, props, oG, oH)
    IgA, IhA = guiggiani_GH(fker, a; order_G=oG, order_H=oH, qsi=qsi, w=w,
        props=props, poly=poly, nodes=nodes)
    IgI, IhI = guiggiani_GH(fker, a; order_G=oG, order_H=oH, qsi=qsi, w=w,
        laurent=:interp, ninterp=ninterp)
    IgR, IhR = guiggiani_GH(fker, a; order_G=oG, order_H=oH, qsi=qsi, w=w)
    println("  ", lab)
    @printf "    ||H analytic|| = %.6e   ||H interp|| = %.6e   ||H Rich.|| = %.6e\n" norm(IhA) norm(IhI) norm(IhR)
    @printf "    rel H  interp vs analytic = %.3e    interp vs Rich. = %.3e\n" relerr(IhI, IhA) relerr(IhI, IhR)
    @printf "    rel H  Rich.  vs analytic = %.3e\n" relerr(IhR, IhA)
    @printf "    rel G  interp vs Rich.    = %.3e    analytic vs Rich. = %.3e\n" relerr(IgI, IgR) relerr(IgA, IgR)
end

report_GH("Laplace CBIE", flap, propsL, 0, -1)
report_GH("Kelvin CBIE", fkel, propsE, 0, -1)

# Kelvin HBIE ~ 1/ρ²
n_el = Point2D(0.0, -1.0)
fhyp = ξ -> begin
    N, dN = BEM.shapefun(poly, ξ)
    pg = N[1, 1] * nodes[1] + N[1, 2] * nodes[2]
    dx = dN[1, 1] * nodes[1] + dN[1, 2] * nodes[2]
    J = norm(dx)
    nrm = Point2D(dx[2], -dx[1]) / J
    Uh, Th = fundamental_hyper(propsE, pg - pf, nrm, n_el)
    Fg = zeros(2, 4)
    Fh = zeros(2, 4)
    for j in 1:2
        cols = (2j - 1):(2j)
        Fg[:, cols] .= Uh .* (N[1, j] * J)
        Fh[:, cols] .= Th .* (N[1, j] * J)
    end
    return Fg, Fh
end
report_GH("Kelvin HBIE (order G=-1, H=-2)", fhyp, propsE, -1, -2)

println()
println("done.")
