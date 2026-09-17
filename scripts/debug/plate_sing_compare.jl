# Compare plate on-element integrals: closed-form moments vs Guiggiani/Richardson.
using DrWatson
@quickactivate :BEM
using LinearAlgebra, FastGaussQuadrature, StaticArrays, Richardson
using BEM.Plate
using BEM.Plate.ThinPlate: ThinPlate as TP
const ArbitraryPolynomial = BEM.ArbitraryPolynomial
const shapefun = BEM.shapefun
const Legendre = BEM.Legendre

const Point2D = SVector{2,Float64}

# Pala / Levy SS immovable: not used here
D, ν = 1.0, 0.3
L = 1.0
x1 = Point2D(0.0, 0.0)
x3 = Point2D(L, 0.0)
geo = Point2D[x1, Point2D(L / 2, 0.0), x3]
el = Element(; index=[1, 2, 3], Jacobian=fill(L / 2, 3), Length=L, Region=1, geo=geo)

function orig_moments_pm23(L)
    # hardcoded original for ξ0 = -2/3, N_disc at ±2/3, 0
    Nlog = (
        (L * (-39 + 10 * log(5) - 27 * log(6) + 27 * log(L))) / 72,
        (L * (-5 + (25 * log(5)) / 6 - 3 * log(6) + 3 * log(L))) / 12,
        -(L * (3 + 25 * log(6 / 5) + log(36) - 27 * log(L))) / 72,
    )
    intN = (0.75, 0.5, 0.75)
    intNsr = ((3 * (-4 + (4 * log(5)) / 3)) / 4, 3.0, 0.0)
    intNsr2 = ((3 * (-9 / 5 - 3 * log(5))) / 4, (-9 + 6 * log(5)) / 2, (3 * (3 - log(5))) / 4)
    return intN, Nlog, intNsr, intNsr2
end

function orig_moments_0(L)
    Nlog = (
        -(L * (1 + log(8) - 3 * log(L))) / 8,
        (L * (-3 + log(L / 2))) / 4,
        -(L * (1 + log(8) - 3 * log(L))) / 8,
    )
    return (0.75, 0.5, 0.75), Nlog, (-1.5, 0.0, 1.5), (2.25, -6.5, 2.25)
end

println("="^70)
println("1) Moments: N_disc at ±2/3 vs original hardcoded")
poly23 = ArbitraryPolynomial([-2 / 3, 0.0, 2 / 3])
for (lab, ξ0, origf) in (("ξ0=-2/3", -2 / 3, orig_moments_pm23), ("ξ0=0", 0.0, orig_moments_0))
    I0, Ilog, Is, Is2 = TP._poly_moments(poly23, ξ0)
    oN, oLog, oS, oS2 = origf(L)
    Nlog = (L / 2) .* (Ilog .+ log(L / 2) .* I0)
    println("  ", lab)
    println("    ΔintN   = ", collect(I0) .- collect(oN))
    println("    ΔNlog   = ", collect(Nlog) .- collect(oLog))
    println("    ΔintNsr = ", collect(Is) .- collect(oS))
    println("    ΔintNsr2= ", collect(Is2) .- collect(oS2))
end

# ---- Guiggiani / Richardson per kernel entry ----
"""Orders of (U,P) entries: G log in U22; H has 1/r, log, 1/r², 1/r."""
const ORDER_U = @SMatrix [0 0; 0 0]
const ORDER_P = @SMatrix [-1 0; -2 -1]

function _pack_kernel(el, poly, pf, nf, ξ, D, ν)
    pg, J, n̂ = TP.elem_geom(el, ξ)
    r = norm(pg - pf)
    r < 1e-14 && return nothing
    U, P = TP.plate_kernels(pg, pf, n̂, nf, D, ν)
    Nf, _ = shapefun(poly, ξ)
    return U, P, Nf, J
end

function plate_guiggiani_components(el, poly, pf, nf, ξ0, D, ν; npg=20, h=1e-3)
    nN = length(el.index)
    qsi, w = gausslegendre(npg)
    a = clamp(float(ξ0), nextfloat(-1.0), prevfloat(1.0))
    g = zeros(2, 2nN)
    hmat = zeros(2, 2nN)
    for α in 1:2, β in 1:2
        fU = ξ -> begin
            samp = _pack_kernel(el, poly, pf, nf, ξ, D, ν)
            samp === nothing && return zeros(nN)
            U, _, Nf, J = samp
            return [U[α, β] * Nf[1, j] * J for j in 1:nN]
        end
        fP = ξ -> begin
            samp = _pack_kernel(el, poly, pf, nf, ξ, D, ν)
            samp === nothing && return zeros(nN)
            _, P, Nf, J = samp
            return [P[α, β] * Nf[1, j] * J for j in 1:nN]
        end
        IU = guiggiani_integral(fU, a, ORDER_U[α, β]; qsi=qsi, w=w, h=h)
        IP = guiggiani_integral(fP, a, ORDER_P[α, β]; qsi=qsi, w=w, h=h)
        for j in 1:nN
            g[α, 2j-2+β] = IU[j]
            hmat[α, 2j-2+β] = IP[j]
        end
    end
    return hmat, g
end

function plate_guiggiani_fused(el, poly, pf, nf, ξ0, D, ν; npg=20, h=1e-3, oG=0, oH=-2)
    nN = length(el.index)
    qsi, w = gausslegendre(npg)
    a = clamp(float(ξ0), nextfloat(-1.0), prevfloat(1.0))
    f = ξ -> begin
        samp = _pack_kernel(el, poly, pf, nf, ξ, D, ν)
        samp === nothing && return (zeros(2, 2nN), zeros(2, 2nN))
        U, P, Nf, J = samp
        Fg = zeros(2, 2nN)
        Fh = zeros(2, 2nN)
        for j in 1:nN
            NjJ = Nf[1, j] * J
            Fg[:, 2j-1:2j] .= U .* NjJ
            Fh[:, 2j-1:2j] .= P .* NjJ
        end
        return Fg, Fh
    end
    return guiggiani_GH(f, a; order_G=oG, order_H=oH, qsi=qsi, w=w, h=h)
end

function rel(A, B)
    nB = norm(B)
    nB < 1e-16 && return norm(A)
    return norm(A - B) / nB
end

println()
println("="^70)
println("2) Closed-form vs Guiggiani on a straight element (GL collocation)")
poly = Legendre(2)
for loc in 1:3
    ξ0 = poly.nodes[loc]
    pf, J, nf = TP.elem_geom(el, ξ0)
    h_cf, g2 = TP.integraelemsing(x1, x3, D, ν, ξ0, poly)
    g_cf = zeros(2, 6)
    g_cf[2, 2:2:6] .= g2
    h_comp, g_comp = plate_guiggiani_components(el, poly, pf, nf, ξ0, D, ν)
    g_fused, h_fused = plate_guiggiani_fused(el, poly, pf, nf, ξ0, D, ν)
    println("  loc=$loc  ξ0=$(round(ξ0; digits=6))")
    println("    ||H_cf|| = ", norm(h_cf), "  ||H_comp|| = ", norm(h_comp),
        "  ||H_fused(-2)|| = ", norm(h_fused))
    println("    rel H  cf vs comp     = ", rel(h_cf, h_comp))
    println("    rel H  fused vs comp  = ", rel(h_fused, h_comp))
    println("    rel G22 cf vs comp    = ", rel(g_cf[2, 2:2:6], g_comp[2, 2:2:6]))
    println("    rel G   fused vs comp = ", rel(g_fused, g_comp))
    println("    H_cf   row1 = ", h_cf[1, :])
    println("    H_comp row1 = ", h_comp[1, :])
    println("    H_cf   row2 = ", h_cf[2, :])
    println("    H_comp row2 = ", h_comp[2, :])
end

println()
println("="^70)
println("3) Richardson Laurent coeffs vs known 1/ρ², 1/ρ, log on a toy integrand")
# F(ρ) = 2/ρ^2 - 3/ρ + 4*log(ρ) + 5 + 6ρ
toy(ρ) = 2 / ρ^2 - 3 / ρ + 4 * log(ρ) + 5 + 6ρ
h = 1e-3
C_hfp = laurent_coefficients(toy, h, Val(-2))
C_cpv = laurent_coefficients(toy, h, Val(-1))
C_log = begin
    F0, _ = extrapolate(h; x0=0.0, contract=1 / 2, atol=1e-12, rtol=1e-10) do ρ
        ρ ≤ eps(h) && return 0.0
        return toy(ρ) / log(ρ)   # WRONG for mixed 1/ρ²+log — shows the issue
    end
    (0.0, 0.0, F0)
end
println("  true   F-2,F-1,F0_log = 2, -3, 4")
println("  HFP-2 Richardson      = ", C_hfp)
println("  CPV-1 Richardson      = ", C_cpv, "  (F-2 not extracted)")
println("  order-0  F/log(ρ)     = ", C_log, "  (garbage if 1/ρ² present)")

# plate kernel sample: P[2,1] ~ 1/r², P[1,1] ~ 1/r, P[1,2] ~ log
println()
println("4) Richardson on actual plate P components (source at GL node 1)")
ξ0 = poly.nodes[1]
pf, J, nf = TP.elem_geom(el, ξ0)
s = 1.0  # ray to the right
hh = 1e-3
Fr_P(ρ) = begin
    samp = _pack_kernel(el, poly, pf, nf, ξ0 + s * ρ, D, ν)
    samp === nothing && return zeros(2, 2)
    return samp[2]  # P
end
probe = Fr_P(hh)
for (lab, α, β, o) in (("Vn P11 ~1/r", 1, 1, -1), ("Mn P12 ~log", 1, 2, 0),
    ("dVn P21 ~1/r²", 2, 1, -2), ("dMn P22 ~1/r", 2, 2, -1))
    f = ρ -> Fr_P(ρ)[α, β]
    C = if o == 0
        F0, _ = extrapolate(hh; x0=0.0, contract=1 / 2, atol=1e-12, rtol=1e-10) do ρ
            ρ ≤ eps(hh) && return 0.0
            return f(ρ) / log(ρ)
        end
        (0.0, 0.0, F0)
    else
        laurent_coefficients(f, hh, Val(o))
    end
    println("  ", lab, "  order=$o  C=", C)
end
println("done.")
