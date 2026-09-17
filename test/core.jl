# Core: interpolation, kernels, analytics (no mesh solves except a 4-edge square).
using Test
using LinearAlgebra
using StaticArrays
using BEM

@testset "3D surface near-field (parent square)" begin
    # ∫∫ dξ dη / R and z/R³ on [-1,1]², source (0,0,d).
    G(x, y, z) = begin
        R = hypot(x, y, z)
        rx, ry = hypot(x, z), hypot(y, z)
        t1 = rx > 0 ? x * asinh(y / rx) : 0.0
        t2 = ry > 0 ? y * asinh(x / ry) : 0.0
        t3 = (abs(z) < 1e-16 || R < 1e-16) ? 0.0 : z * atan(x * y / (z * R))
        t1 + t2 - t3
    end
    I1(d) = G(1, 1, d) - G(-1, 1, d) - G(1, -1, d) + G(-1, -1, d)
    H(x, y, z) = begin
        R = hypot(x, y, z)
        (abs(z) < 1e-16 || R < 1e-16) && return 0.0
        atan(x * y / (z * R))
    end
    I3(d) = H(1, 1, d) - H(-1, 1, d) - H(1, -1, d) + H(-1, -1, d)
    d = 1e-3
    t1, t3 = I1(d), I3(d)
    function Iq(mode, f)
        ξ, η, w = BEM._nearfield_2d(0.0, 0.0, d; n=8, mode=mode)
        s = 0.0
        @inbounds for i in eachindex(w)
            s += w[i] * f(ξ[i], η[i])
        end
        return s
    end
    f1 = (ξ, η) -> 1 / hypot(ξ, η, d)
    f3 = (ξ, η) -> d / hypot(ξ, η, d)^3
    e1p = abs(Iq(:polar, f1) - t1) / t1
    e1t = abs(Iq(:tensor, f1) - t1) / t1
    e1g = abs(Iq(:plain, f1) - t1) / t1
    e3p = abs(Iq(:polar, f3) - t3) / t3
    e3g = abs(Iq(:plain, f3) - t3) / t3
    e1tp = abs(Iq(:tanp3c, f1) - t1) / t1
    e3tp = abs(Iq(:tanp3c, f3) - t3) / t3
    @test e1p < 1e-6
    @test e1t < e1g
    @test e3p < 0.05
    @test e3g > 0.5
    # radial tan-p3c: natural for z/R³ (1/r²); 1/R stays sinh's job
    @test e3tp < 1e-6
    @test e1p < e1tp
    # polar about a parent corner (projection on a vertex)
    ξc, ηc, wc = BEM._nearfield_2d(1.0, 1.0, 1e-3; n=8, mode=:polar)
    @test all(abs.(ξc) .<= 1 + 1e-9) && all(abs.(ηc) .<= 1 + 1e-9)
    @test abs(sum(wc) - 4) < 1e-3
    @test all(>(0), wc)
    ξt, ηt, wt = BEM._nearfield_2d(0.0, 0.0, 1e-3; n=8, mode=:tanp3c)
    @test all(abs.(ξt) .<= 1 + 1e-9) && all(abs.(ηt) .<= 1 + 1e-9)
    @test all(>(0), wt)
end

@testset "polar Guiggiani surface" begin
    # Parent square [-1,1]², collocation at origin: ∫ ρ dρ dθ = area = 4.
    Ig, Ih = guiggiani_GH_surface(0.0, 0.0; nθ=10) do ξ, η
        1.0, 0.0
    end
    @test Ig ≈ 4.0 rtol=1e-6
    @test abs(Ih) < 1e-10

    # Flat unit square in z=0, φ_G = 1/R (no 4πk). Collocation at centre.
    # ∫_{[-1,1]²} dA/R = 8 * log(1+√2) ≈ 7.327.
    poly = BEM.Equispaced(1)
    corners = [Point3D(-1, -1, 0), Point3D(1, -1, 0), Point3D(-1, 1, 0), Point3D(1, 1, 0)]
    pf = Point3D(0, 0, 0)
    IgR, IhR = guiggiani_GH_surface(0.0, 0.0; nθ=12) do ξ, η
        L, Lξ, Lη = BEM.shapefun2D(poly, poly, ξ, η)
        pg = zero(Point3D); tξ = zero(Point3D); tη = zero(Point3D)
        for k in 1:4
            pg += L[1, k] * corners[k]
            tξ += Lξ[1, k] * corners[k]
            tη += Lη[1, k] * corners[k]
        end
        J = norm(cross(tξ, tη))
        R = norm(pg - pf)
        R < 1e-14 && return 0.0, 0.0
        nrm = cross(tξ, tη) / J
        return (1 / R) * J, (dot(pg - pf, nrm) / R^3) * J
    end
    @test IgR ≈ 8 * log(1 + sqrt(2)) rtol=1e-3
    @test abs(IhR) < 1e-6
end

@testset "four-term Richardson Laurent (order -4)" begin
    f(ρ) = 5 / ρ^4 + 4 / ρ^3 + 3 / ρ^2 + 2 / ρ + 7
    C = laurent_coefficients(f, 0.05, Val(-4); maxeval=8, atol=1e-8)
    @test length(C) == 5
    @test C[1] ≈ 5 rtol = 1e-10
    @test C[2] ≈ 4 rtol = 1e-8
    @test C[3] ≈ 3 rtol = 1e-6
    @test C[4] ≈ 2 rtol = 1e-5
    @test C[5] ≈ 7 rtol = 1e-4
    # HFP ∫_{-1}^{1} ξ^{-4} dξ = -2/3 (two rays, F₋₄=1)
    qsi, w = BEM.gausslegendre(16)
    I = guiggiani_integral(0.0, -4; qsi=qsi, w=w, h=0.05, maxeval=8, atol=1e-8,
        laurent=:richardson) do ξ
        1 / ξ^4
    end
    @test I ≈ -2 / 3 rtol = 1e-6
end

@testset "interpolated Guiggiani Laurent" begin
    # Gauss-n nodes/weights/barycentric wᵢ are cached; only Nᵢ(a) depends on a.
    ξa, wa, Na, _ = BEM._interp_rule(0.2, 20)
    ξb, wb, Nb, _ = BEM._interp_rule(0.55, 20)
    @test ξa === ξb && wa === wb
    @test Na ≉ Nb
    ξd, _, _, δd = BEM._interp_rule(ξa[1], 20)
    @test length(ξd) == 19
    @test all(abs.(δd) .> 1e-12)

    # Polynomial Laurent series: interpolant of (ξ-a)^p f is exact for n=20.
    a = 0.2
    f(ξ) = 5 / (ξ - a)^4 + 4 / (ξ - a)^3 + 3 / (ξ - a)^2 + 2 / (ξ - a) + 7
    C = laurent_coefficients_interp(f, a, -4; n=20)
    @test length(C) == 5
    @test C[1] ≈ 5 rtol = 1e-12
    @test C[2] ≈ 4 rtol = 1e-12
    @test C[3] ≈ 3 rtol = 1e-10
    @test C[4] ≈ 2 rtol = 1e-10
    @test C[5] ≈ 7 rtol = 1e-10
    # Right-ray Richardson recovers the leading terms; F₀ is poorly resolved.
    CR = laurent_coefficients(ρ -> f(a + ρ), 1e-3, Val(-4); maxeval=8, atol=1e-10)
    @test CR[1] ≈ 5 rtol = 1e-8
    @test CR[2] ≈ 4 rtol = 1e-6
    @test abs(C[1] - CR[1]) < abs(CR[1] - 5) + 1e-12
    @test abs(C[5] - 7) < abs(CR[5] - 7)

    # Taylor of exp(u)/u^2: A₋₂=1, A₋₁=1, A₀=1/2
    g(ξ) = exp(ξ - a) / (ξ - a)^2
    Cg = laurent_coefficients_interp(g, a, -2; n=20)
    @test Cg[1] ≈ 1 rtol = 1e-10
    @test Cg[2] ≈ 1 rtol = 1e-8
    @test Cg[3] ≈ 0.5 rtol = 1e-6

    qsi, w = BEM.gausslegendre(16)
    hfp4(a) = -1 / (3 * (1 - a)^3) - 1 / (3 * (1 + a)^3)
    hfp2(a) = -2 / (1 - a^2)
    cpv1(a) = log(abs((1 - a) / (1 + a)))
    for s in (-0.4, 0.0, 0.55)
        I4i = guiggiani_integral(ξ -> 1 / (ξ - s)^4, s, -4; qsi=qsi, w=w, laurent=:interp)
        I4r = guiggiani_integral(ξ -> 1 / (ξ - s)^4, s, -4; qsi=qsi, w=w, h=0.05,
            maxeval=8, atol=1e-8, laurent=:richardson)
        @test I4i ≈ hfp4(s) rtol = 1e-5
        @test I4r ≈ hfp4(s) rtol = 1e-5
        I2i = guiggiani_integral(ξ -> 1 / (ξ - s)^2, s, -2; qsi=qsi, w=w, laurent=:interp)
        I2r = guiggiani_integral(ξ -> 1 / (ξ - s)^2, s, -2; qsi=qsi, w=w,
            laurent=:richardson)
        @test I2i ≈ hfp2(s) rtol = 1e-8
        @test I2r ≈ hfp2(s) rtol = 1e-7
        I1i = guiggiani_integral(ξ -> 1 / (ξ - s), s, -1; qsi=qsi, w=w, laurent=:interp)
        I1r = guiggiani_integral(ξ -> 1 / (ξ - s), s, -1; qsi=qsi, w=w,
            laurent=:richardson)
        @test I1i ≈ cpv1(s) rtol = 1e-8 atol = 1e-14
        @test I1r ≈ cpv1(s) rtol = 1e-8 atol = 1e-14
    end

    # Log: strip 1/ρ first, then F₀ from the remainder / log|δ|.
    # Identities at ξ=0 (Gauss nodes symmetric): ξ log|ξ| is odd → F₋₁=0.
    Clog = laurent_coefficients_interp(ξ -> 3 * log(abs(ξ)), 0.0, 0; n=20)
    @test Clog[2] ≈ 0 atol = 1e-12
    @test Clog[3] ≈ 3 rtol = 1e-12
    Cmix = laurent_coefficients_interp(ξ -> 2 / ξ + 3 * log(abs(ξ)), 0.0, 0; n=20)
    @test Cmix[2] ≈ 2 rtol = 1e-10
    @test Cmix[3] ≈ 3 rtol = 1e-10
    Ilog0 = -2.0   # ∫_{-1}^{1} log|ξ| dξ
    I0 = guiggiani_integral(ξ -> log(abs(ξ)), 0.0, 0; laurent=:interp)
    @test I0 ≈ Ilog0 rtol = 1e-12
    Imix = guiggiani_integral(ξ -> 2 / ξ + 3 * log(abs(ξ)), 0.0, 0; laurent=:interp)
    @test Imix ≈ 3 * Ilog0 rtol = 1e-10   # CPV of 1/ξ is 0 at a=0
    Iglog, _ = guiggiani_GH(ξ -> (2 * log(abs(ξ)), 0.0), 0.0;
        order_G=0, order_H=-1, laurent=:interp)
    @test Iglog ≈ 2 * Ilog0 rtol = 1e-12

    # Fused pair: interp CPV + HFP vs Richardson / closed form.
    pair = ξ -> (1 / (ξ - 0.1), 1 / (ξ - 0.1)^2)
    IgI, IhI = guiggiani_GH(pair, 0.1; order_G=-1, order_H=-2, qsi=qsi, w=w,
        laurent=:interp)
    IgR, IhR = guiggiani_GH(pair, 0.1; order_G=-1, order_H=-2, qsi=qsi, w=w,
        laurent=:richardson)
    @test IgI ≈ cpv1(0.1) rtol = 1e-8
    @test IgR ≈ cpv1(0.1) rtol = 1e-8
    @test IhI ≈ hfp2(0.1) rtol = 1e-8
    @test IhR ≈ hfp2(0.1) rtol = 1e-7
end

@testset "near-field maps" begin
    @test default_nearfield(Laplace(1.0)) === :tanp3c
    @test default_nearfield(Elasticity(1.0, 0.3, 1.0)) === :tanp3c
    @test default_nearfield(AnisotropicElasticity(
        lekhnitskii_engineering(124.04, 10.09, 6.03, 0.334))) === :zsinh
    a, b = 0.25, 0.04
    u, w = BEM.gausslegendre(8)
    R(ξ) = hypot(ξ - a, b)
    ex2 = (atan((1 - a) / b) - atan((-1 - a) / b)) / b
    xt, wt = BEM._tangenttrans(u, w, a, b)
    I2 = sum(wt[i] / R(xt[i])^2 for i in eachindex(wt))
    @test I2 ≈ ex2 rtol = 1e-12
    x3, w3 = BEM._tanp3ctrans(u, w, a, b)
    I2p = sum(w3[i] / R(x3[i])^2 for i in eachindex(w3))
    @test I2p ≈ ex2 rtol = 1e-12
    xs, ws = BEM._sinhtrans(u, w, a, b)
    I1 = sum(ws[i] / R(xs[i]) for i in eachindex(ws))
    ex1 = asinh((1 - a) / b) - asinh((-1 - a) / b)
    @test I1 ≈ ex1 rtol = 1e-12

    # Eq. (40) η0=0: ξ = ζ0 + sign(ξ-ζ0) exp((ξ̃-B)/A). Both pole sides.
    for ζ0 in (-4.58, -4.0, 4.0, 4.58)
        s = -sign(ζ0)
        A, B, sAB = BEM._eq40_AB(ζ0)
        @test sAB == s
        @test (ζ0 < -1 && A > 0) || (ζ0 > 1 && A < 0)
        I1true = log((abs(ζ0) + 1) / (abs(ζ0) - 1))
        x40, w40 = BEM._eq40_real_pole(u, w, ζ0)
        xs, ws = BEM._sinhtrans(u, w, ζ0, 0.0)
        xss, wss = BEM._sinhtrans_iterated(u, w, ζ0, 0.0; niter=2)
        @test x40 ≈ xs
        @test x40 ≈ xss
        @test all(-1 - 1e-12 .<= x40 .<= 1 + 1e-12)
        @test all(>(0), w40)
        @test sum(w40) ≈ 2 rtol = 1e-12
        @test issorted(x40)
        # paper formula and ξ(±1)=±1
        @test ζ0 + s * exp((-1 - B) / A) ≈ -1 atol = 1e-12
        @test ζ0 + s * exp((1 - B) / A) ≈ 1 atol = 1e-12
        @test x40 ≈ [ζ0 + s * exp((ui - B) / A) for ui in u]
        # dξ/dξ̃ = s exp((ξ̃-B)/A) / A
        @test w40 ./ w ≈ [s * exp((ui - B) / A) / A for ui in u] rtol = 1e-12
        I1q = sum(w40[i] / abs(x40[i] - ζ0) for i in eachindex(x40))
        @test I1q ≈ I1true rtol = 1e-12
        @test sum(wss[i] / abs(xss[i] - ζ0) for i in eachindex(xss)) ≈ I1true rtol = 1e-12
        # η0→0 sinh → exponential
        xlim, _ = BEM._sinh_map(u, ζ0, 1e-14)
        @test x40 ≈ xlim rtol = 1e-12
    end

    # Eq. (41) η0=0: Möbius ξ = ζ0 - A/(ξ̃-B). Both pole sides.
    for ζ0 in (-4.58, -4.0, 4.0, 4.58)
        I2true = 2 / (ζ0^2 - 1)
        x41, w41 = BEM._eq41_real_pole(u, w, ζ0)
        xt, wt = BEM._tangenttrans(u, w, ζ0, 0.0)
        xp3, wp3 = BEM._tanp3ctrans(u, w, ζ0, 0.0)
        for (x, ww) in ((x41, w41), (xt, wt), (xp3, wp3))
            @test all(-1 - 1e-12 .<= x .<= 1 + 1e-12)
            @test all(>(0), ww)
            @test sum(ww) ≈ 2 rtol = 1e-12
            @test issorted(x)
        end
        Am, Bm = ζ0^2 - 1, -ζ0
        @test ζ0 - Am / (-1 - Bm) ≈ -1 atol = 1e-12
        @test ζ0 - Am / (1 - Bm) ≈ 1 atol = 1e-12
        @test x41 ≈ xt
        @test x41 ≈ xp3
        I2q = sum(w41[i] / (x41[i] - ζ0)^2 for i in eachindex(x41))
        @test I2q ≈ I2true rtol = 1e-12
        @test sum(wt[i] / (xt[i] - ζ0)^2 for i in eachindex(xt)) ≈ I2true rtol = 1e-12
        @test sum(wp3[i] / (xp3[i] - ζ0)^2 for i in eachindex(xp3)) ≈ I2true rtol = 1e-12
    end
    # atan-saturated collinear (η0 tiny, |ζ0|>1): must stay on Möbius, not tan→∞
    for ζ0 in (-3.5, 4.0)
        I2true = 2 / (ζ0^2 - 1)
        I1true = log((abs(ζ0) + 1) / (abs(ζ0) - 1))
        for η0 in (0.0, 1e-16, 1e-12, 1e-10)
            xt, wt = BEM._tanp3ctrans(u, w, ζ0, η0)
            @test all(-1 - 1e-12 .<= xt .<= 1 + 1e-12)
            @test all(>(0), wt)
            @test sum(wt) ≈ 2 rtol = 1e-10
            @test sum(wt[i] / (xt[i] - ζ0)^2 for i in eachindex(xt)) ≈ I2true rtol = 1e-8
            @test sum(wt[i] / abs(xt[i] - ζ0) for i in eachindex(xt)) ≈ I1true rtol = 1e-8
        end
    end
end

@testset "analytic Guiggiani coeffs (Marczak / cap-iso)" begin
    poly = BEM.Equispaced(1)
    nodes = [Point2D(0.0, 0.0), Point2D(1.0, 0.0)]
    a = 0.0
    qsi, w = BEM.gausslegendre(16)

    # Laplace: flux kernel is regular → F_{-1}=F_{-2}=0
    CH = laurent_coefficients(Laplace(1.0), poly, nodes, a, 1.0, :H, -1)
    @test CH !== nothing
    Fm2, Fm1, F0 = CH
    @test all(iszero, Fm2) && all(iszero, Fm1) && all(iszero, F0)
    _, _, F0G = laurent_coefficients(Laplace(1.0), poly, nodes, a, 1.0, :G, 0)
    # log coeff of U φ J = −φ J / (2πk); J=0.5, φ=0.5 at mid for linear
    @test F0G[1] ≈ -0.5 * 0.5 / (2π) atol=1e-14
    @test F0G[2] ≈ -0.5 * 0.5 / (2π) atol=1e-14
    CH2L = laurent_coefficients(Laplace(1.0), poly, nodes, a, 1.0, :H, -2)
    @test CH2L !== nothing
    Fm2L, _, _ = CH2L
    # F_{-2} = −φ/(2π J); J=1/2, φ=1/2 → −1/(2π)
    @test Fm2L[1] ≈ -1 / (2π) atol=1e-14
    @test Fm2L[2] ≈ -1 / (2π) atol=1e-14
    CG1L = laurent_coefficients(Laplace(1.0), poly, nodes, a, 1.0, :G, -1)
    @test CG1L !== nothing
    @test all(iszero, CG1L[2])

    # Kelvin traction: F_{-2}=0, F_{-1} = −((1-2ν)/(4π(1-ν))) (n_α t_β − n_β t_α) φ
    props = Elasticity(1.0, 0.3, 1.0)
    ν = effective_nu(props)
    cT = -(1 - 2ν) / (4π * (1 - ν))
    Fm2e, Fm1p, _ = laurent_coefficients(props, poly, nodes, a, 1.0, :H, -1)
    @test all(iszero, Fm2e)
    # node 1 and 2 each have φ=0.5 at ξ=0; T12 = cT (n1 t2 − n2 t1)= cT
    @test Fm1p[1, 2] ≈ cT * 0.5 atol=1e-14   # α=1, β=2, node 1
    @test Fm1p[2, 1] ≈ -cT * 0.5 atol=1e-14
    @test Fm1p[1, 4] ≈ cT * 0.5 atol=1e-14   # node 2
    _, Fm1m, _ = laurent_coefficients(props, poly, nodes, a, -1.0, :H, -1)
    @test Fm1m ≈ -Fm1p
    # Kelvin HBIE: T^h → μ I / (2π(1-ν) R²) on the element
    CH2 = laurent_coefficients(props, poly, nodes, a, 1.0, :H, -2)
    @test CH2 !== nothing
    Fm2h, Fm1h, _ = CH2
    Jmid = 0.5
    cS = float(props.mu) / (2π * (1 - ν) * Jmid)
    @test Fm2h[1, 1] ≈ cS * 0.5 atol=1e-14
    @test Fm2h[2, 2] ≈ cS * 0.5 atol=1e-14
    @test Fm2h[1, 2] ≈ 0 atol=1e-14
    CG1 = laurent_coefficients(props, poly, nodes, a, 1.0, :G, -1)
    @test CG1 !== nothing
    _, Fm1g, _ = CG1
    # U^h antisymmetric; n=(0,-1), t=(1,0) → t1 n2 - n1 t2 = -1
    cD = (1 - 2ν) / (4π * (1 - ν))
    @test Fm1g[1, 2] ≈ cD * (-1) * 0.5 atol=1e-14
    @test Fm1g[2, 1] ≈ -Fm1g[1, 2] atol=1e-14

    # fused integral: analytic H matches Richardson CPV on this element
    pf = Point2D(0.5, 0.0)
    fker = ξ -> begin
        N, dN = BEM.shapefun(poly, ξ)
        pg = N[1, 1] * nodes[1] + N[1, 2] * nodes[2]
        dx = dN[1, 1] * nodes[1] + dN[1, 2] * nodes[2]
        J = norm(dx)
        r = pg - pf
        norm(r) < 1e-30 && return zeros(2, 4), zeros(2, 4)
        nrm = Point2D(dx[2], -dx[1]) / J
        U, T = fundamental(props, r, nrm)
        Fg = zeros(2, 4); Fh = zeros(2, 4)
        for j in 1:2
            cols = (2j - 1):(2j)
            Fg[:, cols] .= U .* (N[1, j] * J)
            Fh[:, cols] .= T .* (N[1, j] * J)
        end
        return Fg, Fh
    end
    IgA, IhA = guiggiani_GH(fker, a; order_G=0, order_H=-1, qsi=qsi, w=w,
        props=props, poly=poly, nodes=nodes, laurent=:auto)
    IgR, IhR = guiggiani_GH(fker, a; order_G=0, order_H=-1, qsi=qsi, w=w,
        laurent=:richardson)
    # Traction CPV (the cap-iso F_{-1}) must match Richardson. The log
    # coefficient of U converges slowly under Richardson (log J / log ρ).
    @test IhA ≈ IhR rtol=1e-5 atol=1e-7
    @test IgA ≈ IgR rtol=2e-4 atol=1e-6

    n_el = Point2D(0.0, -1.0)
    fhyp = ξ -> begin
        N, dN = BEM.shapefun(poly, ξ)
        pg = N[1, 1] * nodes[1] + N[1, 2] * nodes[2]
        dx = dN[1, 1] * nodes[1] + dN[1, 2] * nodes[2]
        J = norm(dx)
        nrm = Point2D(dx[2], -dx[1]) / J
        Uh, Th = fundamental_hyper(props, pg - pf, nrm, n_el)
        Fg = zeros(2, 4); Fh = zeros(2, 4)
        for j in 1:2
            cols = (2j - 1):(2j)
            Fg[:, cols] .= Uh .* (N[1, j] * J)
            Fh[:, cols] .= Th .* (N[1, j] * J)
        end
        return Fg, Fh
    end
    ρ = 1e-6
    _, Fh = fhyp(a + ρ)
    @test Fm2h ≈ ρ^2 .* Fh rtol=1e-4 atol=1e-6
    Fm2R, Fm1R, _ = BEM.laurent_coefficients(ρ -> fhyp(a + ρ)[2], 1e-3, Val(-2))
    @test Fm2h ≈ Fm2R rtol=1e-8 atol=1e-8
    @test Fm1h ≈ Fm1R rtol=1e-6 atol=1e-6
    IgH, IhH = guiggiani_GH(fhyp, a; order_G=-1, order_H=-2, qsi=qsi, w=w,
        props=props, poly=poly, nodes=nodes, laurent=:auto)
    IgR2, IhR2 = guiggiani_GH(fhyp, a; order_G=-1, order_H=-2, qsi=qsi, w=w,
        laurent=:richardson)
    @test all(isfinite, IgH) && all(isfinite, IhH)
    @test IhH ≈ IhR2 rtol=1e-4 atol=1e-6
end

@testset "analytic Guiggiani coeffs (Cordeiro / Lekhnitskii)" begin
    poly = BEM.Equispaced(1)
    nodes = [Point2D(0.0, 0.0), Point2D(1.0, 0.0)]
    a = 0.0
    qsi, w = BEM.gausslegendre(16)
    pars = lekhnitskii_engineering(124.04, 10.09, 6.03, 0.334; η12_1=1.255, η12_2=-0.031)
    props = AnisotropicElasticity(pars)
    D = inv(pars.C)
    for μ in pars.mi
        rμ = D[1, 1] * μ^4 - 2 * D[1, 3] * μ^3 + (2 * D[1, 2] + D[3, 3]) * μ^2 -
             2 * D[2, 3] * μ + D[2, 2]
        @test abs(rμ) < 1e-12 * abs(D[2, 2])
    end
    E, ν = 124.04e3, 0.334
    piso = lekhnitskii_engineering(E, E, E / (2 * (1 + ν)), ν)
    @test all(abs.(imag.(piso.mi) .- 1) .< 1e-8)
    CH = laurent_coefficients(props, poly, nodes, a, 1.0, :H, -1)
    @test CH !== nothing
    Fm2, Fm1, F0 = CH
    @test all(iszero, Fm2) && all(iszero, F0)
    @test size(Fm1) == (2, 4)
    CG = laurent_coefficients(props, poly, nodes, a, 1.0, :G, 0)
    @test CG !== nothing
    _, _, F0G = CG
    Ulog = 2 * real(pars.A * transpose(pars.q))
    @test F0G[1, 1] ≈ Ulog[1, 1] * 0.5 * 0.5 atol=1e-14  # φ=1/2, J=1/2 at mid
    @test F0G[2, 2] ≈ Ulog[2, 2] * 0.5 * 0.5 atol=1e-14

    pf = Point2D(0.5, 0.0)
    fker = ξ -> begin
        N, dN = BEM.shapefun(poly, ξ)
        pg = N[1, 1] * nodes[1] + N[1, 2] * nodes[2]
        dx = dN[1, 1] * nodes[1] + dN[1, 2] * nodes[2]
        J = norm(dx)
        r = pg - pf
        norm(r) < 1e-30 && return zeros(2, 4), zeros(2, 4)
        nrm = Point2D(dx[2], -dx[1]) / J
        U, T = fundamental(props, pg, pf, nrm)
        U = BEM._to_smat(U); T = BEM._to_smat(T)
        Fg = zeros(2, 4); Fh = zeros(2, 4)
        for j in 1:2
            cols = (2j - 1):(2j)
            Fg[:, cols] .= U .* (N[1, j] * J)
            Fh[:, cols] .= T .* (N[1, j] * J)
        end
        return Fg, Fh
    end
    IgA, IhA = guiggiani_GH(fker, a; order_G=0, order_H=-1, qsi=qsi, w=w,
        props=props, poly=poly, nodes=nodes, laurent=:auto)
    IgR, IhR = guiggiani_GH(fker, a; order_G=0, order_H=-1, qsi=qsi, w=w,
        laurent=:richardson)
    @test IhA ≈ IhR rtol=1e-5 atol=1e-7
    # log coefficient of U converges slowly under Richardson (same as Kelvin)
    @test IgA ≈ IgR rtol=5e-3 atol=1e-6

    # HBIE: paper SST F₋₂ matches ρ² (nξ·S) φ J. Richardson at parent h=1e-3
    # recovers F₋₂ and F₋₁ on this straight element.
    n_el = Point2D(0.0, -1.0)
    fhyp = ξ -> begin
        N, dN = BEM.shapefun(poly, ξ)
        pg = N[1, 1] * nodes[1] + N[1, 2] * nodes[2]
        dx = dN[1, 1] * nodes[1] + dN[1, 2] * nodes[2]
        J = norm(dx)
        nrm = Point2D(dx[2], -dx[1]) / J
        Uh, Th = fundamental_hyper(props, pg, pf, nrm, n_el)
        Uh = BEM._to_smat(Uh); Th = BEM._to_smat(Th)
        Fg = zeros(2, 4); Fh = zeros(2, 4)
        for j in 1:2
            cols = (2j - 1):(2j)
            Fg[:, cols] .= Uh .* (N[1, j] * J)
            Fh[:, cols] .= Th .* (N[1, j] * J)
        end
        return Fg, Fh
    end
    CH2 = laurent_coefficients(props, poly, nodes, a, 1.0, :H, -2)
    @test CH2 !== nothing
    Fm2, Fm1, _ = CH2
    ρ = 1e-6
    _, Fh = fhyp(a + ρ)
    @test Fm2 ≈ ρ^2 .* Fh rtol=1e-4 atol=1e-6
    # eq. 37 is T_lead φ / J, not T_lead φ / J²  (J=1/2 on this element)
    _, Thlead = BEM._lekh_Uh_Th_lead(pars, Point2D(0.0, -1.0))
    @test Fm2[1, 1] ≈ Thlead[1, 1] * 0.5 / 0.5 atol=1e-12
    @test !(Fm2[1, 1] ≈ Thlead[1, 1] * 0.5 / 0.5^2)
    CG1 = laurent_coefficients(props, poly, nodes, a, 1.0, :G, -1)
    @test CG1 !== nothing
    Fm2R, Fm1R, _ = BEM.laurent_coefficients(ρ -> fhyp(a + ρ)[2], 1e-3, Val(-2))
    @test Fm2 ≈ Fm2R rtol=1e-8 atol=1e-8
    @test Fm1 ≈ Fm1R rtol=1e-6 atol=1e-6
    IgH, IhH = guiggiani_GH(fhyp, a; order_G=-1, order_H=-2, qsi=qsi, w=w,
        props=props, poly=poly, nodes=nodes, laurent=:auto)
    IgS, IhS = sst_GH(fhyp, a; qsi=qsi, w=w, props=props, poly=poly, nodes=nodes)
    IgR2, IhR2 = guiggiani_GH(fhyp, a; order_G=-1, order_H=-2, qsi=qsi, w=w,
        laurent=:richardson)
    @test all(isfinite, IgH) && all(isfinite, IhH)
    @test IhS ≈ IhH rtol=1e-12 atol=1e-12
    @test IgS ≈ IgH rtol=1e-12 atol=1e-12
    @test IhH ≈ IhR2 rtol=1e-4 atol=1e-6

    # Curved quadratic: SST-factor Guiggiani vs Richardson.
    polyq = BEM.Legendre(2)
    θ = (0.0, π / 8, π / 4)
    Xarc = [Point2D(cos(t), sin(t)) for t in θ]
    a_c = polyq.nodes[2]
    Nc, _ = BEM.shapefun(polyq, a_c)
    pf_c = (Nc * Xarc)[1]
    g_c = BEM._geom_1d(polyq, Xarc, a_c)
    _, _, _, n_c = g_c
    farc = ξ -> begin
        N, dN = BEM.shapefun(polyq, ξ)
        pg = (N * Xarc)[1]; dx = (dN * Xarc)[1]; J = norm(dx)
        nrm = Point2D(dx[2], -dx[1]) / J
        Uh, Th = fundamental_hyper(props, pg, pf_c, nrm, n_c)
        Uh = BEM._to_smat(Uh); Th = BEM._to_smat(Th)
        Fg = zeros(2, 6); Fh = zeros(2, 6)
        for j in 1:3
            cols = (2j - 1):(2j)
            Fg[:, cols] .= Uh .* (N[1, j] * J)
            Fh[:, cols] .= Th .* (N[1, j] * J)
        end
        return Fg, Fh
    end
    IgC, IhC = sst_GH(farc, a_c; qsi=qsi, w=w, props=props, poly=polyq, nodes=Xarc)
    IgR, IhR = guiggiani_GH(farc, a_c; order_G=-1, order_H=-2, qsi=qsi, w=w,
        laurent=:richardson)
    @test all(isfinite, IgC) && all(isfinite, IhC)
    @test IhC ≈ IhR rtol=1e-6 atol=1e-6
    @test IgC ≈ IgR rtol=1e-6 atol=1e-6
end

@testset "closest-point" begin
    poly = BEM.Equispaced(1)
    nodes = [Point2D(0.0, 0.0), Point2D(1.0, 0.0)]
    ξ, x, d = closest_point_1d(poly, nodes, Point2D(0.5, 0.1))
    @test abs(x[1] - 0.5) < 1e-6
    @test d ≈ 0.1 atol=1e-6
    ξ2, _, d2 = closest_point_1d(poly, nodes, Point2D(-1.0, 0.0))
    @test ξ2 ≈ -1.0 atol=1e-8
    @test d2 ≈ 1.0 atol=1e-8
end

@testset "Granados complex pole / tangent" begin
    poly = BEM.Equispaced(1)
    nodes = [Point2D(0.0, 0.0), Point2D(2.0, 0.0)]  # J=1, L=2
    ζ0, η0 = BEM._complex_pole_1d(poly, nodes, Point2D(1.0, 0.1))
    @test ζ0 ≈ 0.0 atol=1e-12
    @test η0 ≈ 0.1 atol=1e-12
    u, w = BEM.gausslegendre(8)
    xt, wt = BEM._tangenttrans(u, w, 0.0, 0.1)
    @test all(-1 .<= xt .<= 1)
    @test sum(wt) > 0
    xs, ws = BEM._sinhtrans_iterated(u, w, 0.0, 0.1; niter=2)
    x1, w1 = BEM._sinhtrans(u, w, 0.0, 0.1)
    @test norm(xs - x1) > 0  # second sinh moves nodes
    @test all(-1 .<= xs .<= 1)
end

@testset "Granados p3c / tan-p3c" begin
    # JACM 12 (2026) p.562 numerical example
    ζ0, η0 = 0.7524705856764565, 0.09447435736471754
    ζ̃, η̃ = BEM._p3c_inv_pole(ζ0, η0)
    @test ζ̃ ≈ 0.4375586278465583 atol=1e-14
    @test η̃ ≈ 0.47346508373606316 atol=1e-14
    ends, _ = BEM._p3ctrans([-1.0, 1.0], [1.0, 1.0], ζ0, η0)
    @test ends[1] ≈ -1 atol=1e-14
    @test ends[2] ≈ 1 atol=1e-14
    # η0=0 cubic (82): residual of the inverse pole
    z, e = BEM._p3c_inv_pole(0.5, 0.0)
    @test abs(e) < 1e-14
    @test abs(z^3 - 1.5 * z^2 + 3 * z - 0.5) < 1e-12
    u, w = BEM.gausslegendre(8)
    xp, wp = BEM._p3ctrans(u, w, 0.0, 0.1)
    @test all(-1 .<= xp .<= 1)
    @test sum(wp) ≈ 2 atol=1e-12
    # interior pole: tan-p3c splits at B (2n nodes)
    xt, wt = BEM._tanp3ctrans(u, w, 0.0, 0.1)
    @test length(xt) == 2 * length(u)
    @test all(-1 .<= xt .<= 1)
    @test sum(wt) ≈ 2 atol=1e-6
    # Case 3 (ζ0≥1): no split
    x3, w3 = BEM._tanp3ctrans(u, w, 1.225, 6e-5)
    @test length(x3) == length(u)
    @test all(-1 .<= x3 .<= 1)
    @test sum(w3) ≈ 2 atol=1e-8
    # 1/r² on a straight parent: I = [atan((1-ζ0)/η0)+atan((1+ζ0)/η0)]/η0
    Itrue(ζ, η) = (atan((1 - ζ) / η) + atan((1 + ζ) / η)) / η
    Iquad(x, ww, ζ, η) = sum(ww[i] / ((x[i] - ζ)^2 + η^2) for i in eachindex(x))
    I0 = Itrue(0.0, 0.1)
    Iplain = Iquad(u, w, 0.0, 0.1)
    @test abs(Iquad(xp, wp, 0.0, 0.1) - I0) < abs(Iplain - I0)  # p3c beats plain GL
    xtan, wtan = BEM._tangenttrans(u, w, 0.0, 0.1)
    @test abs(Iquad(xtan, wtan, 0.0, 0.1) - I0) / I0 < 1e-10
    @test abs(Iquad(xt, wt, 0.0, 0.1) - I0) / I0 < 1e-8
    I3 = Itrue(1.225, 6e-5)
    @test abs(Iquad(x3, w3, 1.225, 6e-5) - I3) / I3 < 1e-6
end

@testset "analytical T=x flux" begin
    ana = ana_laplace_linear(; direction=SA[1.0, 0.0], k=1.0)
    @test ana(Point2D(0.5, 0.3)) ≈ 0.5
    @test ana.q(Point2D(1, 0), Point2D(1, 0)) ≈ -1.0
end

@testset "fundamental solutions" begin
    r = Point2D(0.3, 0.4)
    n = Point2D(1.0, 0.0)
    R = norm(r)
    lap = Laplace(1.0)
    kp = fundamental(lap, r, n)
    @test kp.U ≈ -log(R) / (2π)
    @test kp.T ≈ dot(r, n) / (R^2 * 2π)
    @test fundamental_U(lap, r) ≈ kp.U
    @test fundamental_T(lap, r, n) ≈ kp.T

    nf = Point2D(0.0, 1.0)
    kh = fundamental_hyper(lap, r, n, nf)
    e = r / R
    @test kh.U ≈ dot(e, nf) / (2π * R) atol=1e-14

    r3 = Point3D(0.3, 0.4, 0.5)
    n3 = Point3D(0.0, 0.0, 1.0)
    R3 = norm(r3)
    kp3 = fundamental(lap, r3, n3)
    @test kp3.U ≈ 1 / (4π * R3)
    @test kp3.T ≈ dot(r3, n3) / (4π * R3^3)
    @test fundamental_U(lap, r3) ≈ kp3.U
    @test fundamental_T(lap, r3, n3) ≈ kp3.T

    el = Elasticity(1.0, 0.3, 1.0)
    kpe = fundamental(el, r, n)
    @test kpe.U ≈ kpe.U' atol=1e-14
    @test fundamental_U(el, r) ≈ kpe.U
    pars = lekhnitskii_engineering(124.04, 10.09, 6.03, 0.334; η12_1=1.255, η12_2=-0.031)
    aniso = AnisotropicElasticity(pars)
    kpa = fundamental(aniso, r, zero(r), n)
    @test all(isfinite, kpa.U) && all(isfinite, kpa.T)
    kph = fundamental_hyper(aniso, r, zero(r), n, nf)
    @test all(isfinite, kph.U) && all(isfinite, kph.T)
    @test size(kph.U) == (2, 2)
    kpe3 = fundamental(el, r3, n3)
    @test kpe3.U ≈ kpe3.U' atol=1e-12
    nf3 = Point3D(0.0, 1.0, 0.0)
    khe3 = fundamental_hyper(el, r3, n3, nf3)
    @test all(isfinite, khe3.U) && all(isfinite, khe3.T)
    @test size(khe3.U) == (3, 3)

    ana3 = ana_laplace_linear(; direction=SA[0.0, 0.0, 1.0], k=1.0)
    @test ana3(Point3D(0.2, 0.3, 0.4)) ≈ 0.4
    @test ana3.q(Point3D(0, 0, 1), Point3D(0, 0, 1)) ≈ -1.0
    anaq = ana_laplace_quadratic(; dim=3)
    @test anaq(Point3D(1.0, 1.0, 1.0)) ≈ 0.0
    anael = ana_elasticity_patch(; εxx=0.01, dim=3)
    @test anael(Point3D(2.0, 0.0, 0.0)) ≈ SA[0.02, 0.0, 0.0]
    n̂ = Point3D(1.0, 0.0, 0.0)
    t̂ = anael.q(Point3D(1, 0, 0), n̂)
    λ = 1.0 * 0.3 / ((1 + 0.3) * (1 - 2 * 0.3))
    μ = 1.0 / (2 * 1.3)
    @test t̂[1] ≈ (λ + 2μ) * 0.01 atol=1e-14
    @test t̂[2] ≈ 0 atol=1e-14
    @test t̂[3] ≈ 0 atol=1e-14
    helm = Helmholtz(; ω=2.0, c=1.0)
    kph = fundamental(helm, r, n)
    @test kph.U isa Complex
    @test isfinite(real(kph.U))
    @test fundamental_U(helm, r) ≈ kph.U
    @test fundamental_T(helm, r, n) ≈ kph.T
end

@testset "3D surface DIBEM (parent square)" begin
    # PHS3 RIM of r³ on [-1,1]² vs tensor Gauss.
    ξc, ηc = BEM._surf_dibem_centers(8)
    @test length(ξc) == 4 + 8 * 4
    c = BEM._surf_dibem_weights(ξc, ηc)
    @test abs(sum(c) - 4) < 1e-10
    @test abs(dot(c, ξc)) < 1e-10
    @test abs(dot(c, ηc)) < 1e-10
    u, w = BEM.gausslegendre(24)
    function IFgauss(cx, cy)
        acc = 0.0
        @inbounds for i in eachindex(u), j in eachindex(u)
            acc += w[i] * w[j] * hypot(u[i] - cx, u[j] - cy)^3
        end
        return acc
    end
    @test abs(BEM._surf_dibem_IF_phs3(0.0, 0.0) - IFgauss(0.0, 0.0)) /
          IFgauss(0.0, 0.0) < 1e-5
    @test abs(BEM._surf_dibem_IF_phs3(1.0, 1.0) - IFgauss(1.0, 1.0)) < 1e-10

    # Closed form ∫ 1/R and z/R³ on [-1,1]², source (0,0,d).
    G(x, y, z) = begin
        R = hypot(x, y, z)
        rx, ry = hypot(x, z), hypot(y, z)
        t1 = rx > 0 ? x * asinh(y / rx) : 0.0
        t2 = ry > 0 ? y * asinh(x / ry) : 0.0
        t3 = (abs(z) < 1e-16 || R < 1e-16) ? 0.0 : z * atan(x * y / (z * R))
        t1 + t2 - t3
    end
    I1(d) = G(1, 1, d) - G(-1, 1, d) - G(1, -1, d) + G(-1, -1, d)
    Hfn(x, y, z) = begin
        R = hypot(x, y, z)
        (abs(z) < 1e-16 || R < 1e-16) && return 0.0
        atan(x * y / (z * R))
    end
    I3(d) = Hfn(1, 1, d) - Hfn(-1, 1, d) - Hfn(1, -1, d) + Hfn(-1, -1, d)

    d = 1e-3
    @test abs(BEM._id_radial_parent(0.0, 0.0, d, 1) - I1(d)) / I1(d) < 1e-8
    @test abs(d * BEM._id_radial_parent(0.0, 0.0, d, 3) - I3(d)) / I3(d) < 1e-8

    poly = BEM.Equispaced(1)
    nodes = [Point3D(-1.0, -1.0, 0.0), Point3D(1.0, -1.0, 0.0),
             Point3D(-1.0, 1.0, 0.0), Point3D(1.0, 1.0, 0.0)]
    nrm = [Point3D(0.0, 0.0, 1.0) for _ in 1:4]
    el = Element([1, 2, 3, 4], ones(4), 2.0, 1)
    dad = BEMdata(; name="surf_dibem_sq", dimension=3, elements=[el],
        element_type=poly, elem_weight=SA[1.0, 1.0, 1.0, 1.0],
        collocation=nodes, Normal=nrm, properties=Laplace(1.0),
        BC=ones(Int, 4), BV=zeros(4), n=4, ni=0, nt=4)
    qq, ww = BEM.gausslegendre(8)
    set_cache!(dad; qsi=qq, w=ww, nearfield=:dibem)
    pf = Point3D(0.0, 0.0, d)
    h = zeros(4); g = zeros(4)
    BEM.integrate_element_dibem!(h, g, dad, el, nodes, pf)
    # Row-sum = analytic ID (k=1, J=1, n=+z, r·n=−d).
    @test abs(sum(g) - I1(d) / (4π)) / (I1(d) / (4π)) < 1e-8
    @test abs(sum(h) + I3(d) / (4π)) / (I3(d) / (4π)) < 1e-8
    # Nodal split vs polar Gauss of N J K.
    set_cache!(dad; nearfield=:polar)
    hp = zeros(4); gp = zeros(4)
    integrate_element(dad, el, nodes, pf, hp, gp)
    @test norm(g - gp) / norm(gp) < 0.05
    @test norm(h - hp) / norm(hp) < 0.05
end

@testset "internal_grid 2-D square" begin
    dad = format2d(quadrado(ndiv=8, show=false, nome="t_igrid2"), Laplace(1.0);
        pontointerno=false)
    pts = internal_grid(dad, 5, 5; d_min=0.05)
    @test !isempty(pts)
    @test all(p -> 0 < p[1] < 1 && 0 < p[2] < 1, pts)
    @test all(p -> point_in_domain(dad, p), pts)
    @test !point_in_domain(dad, Point2D(-0.1, 0.5))
    @test !point_in_domain(dad, Point2D(1.1, 0.5))
    internal_grid!(dad, 4, 4; d_min=0.02)
    @test dad.ni == length(internal_grid(dad, 4, 4; d_min=0.02))
    @test gera_p_in(dad, 3, 3; d_min=0) == internal_grid(dad, 3, 3; d_min=0)
end

@testset "internal_grid 2-D hole" begin
    dad = format2d(placa_furo_orto(; lc=0.08, nome="t_igrid_h", show=false),
        Laplace(1.0); pontointerno=false)
    pts = internal_grid(dad, 10, 10; d_min=0.02)
    @test !isempty(pts)
    @test all(p -> (p[1] - 0.5)^2 + (p[2] - 0.5)^2 >= 0.25^2 - 1e-6, pts)
    @test !point_in_domain(dad, Point2D(0.5, 0.5))  # hole centre
    @test point_in_domain(dad, Point2D(0.1, 0.1))
end

@testset "internal_grid 3-D cube" begin
    include(joinpath(@__DIR__, "..", "data", "Laplace", "cube_mesh.jl"))
    dad = format3d(mesh_cube(; L=1.0, ndiv=2, nome="t_igrid3"), Laplace(1.0);
        pontointerno=false)
    pts = internal_grid(dad, 3, 3, 3; d_min=0.05)
    @test !isempty(pts)
    @test all(p -> all(0 .< p .< 1), pts)
    @test all(p -> point_in_domain(dad, p), pts)
    @test !point_in_domain(dad, Point3D(-0.1, 0.5, 0.5))
    @test !point_in_domain(dad, Point3D(0.5, 0.5, -0.1))
    @test point_in_domain(dad, Point3D(0.5, 0.5, 0.5))
    internal_grid!(dad, 2, 2, 2; d_min=0.02)
    @test dad.ni > 0
    @test dad.nt == dad.n + dad.ni
end

@testset "internal_layer 2-D square and hole" begin
    dad = format2d(quadrado(ndiv=21, show=false, nome="t_ilayer2"), Laplace(1.0);
        pontointerno=false)
    ls = sort(Float64[el.Length for el in dad.elements])
    h = ls[(length(ls) + 1) ÷ 2]
    layer = internal_layer(dad; every=2, fill=4)
    @test !isempty(layer)
    @test length(layer) < dad.n
    @test all(p -> point_in_domain(dad, p), layer)
    dΓ = minimum(BEM._dist_to_boundary(dad, p) for p in layer)
    @test 0.4 * h < dΓ < 3 * h
    internal_layer!(dad; every=2, fill=4)
    @test dad.ni == length(layer)
    dadh = format2d(placa_furo_orto(; lc=0.08, nome="t_ilayer_h", show=false),
        Laplace(1.0); pontointerno=false)
    ph = internal_layer(dadh; every=2, fill=3)
    @test !isempty(ph)
    @test all(p -> (p[1] - 0.5)^2 + (p[2] - 0.5)^2 >= 0.25^2 - 1e-4, ph)
end

@testset "internal_layer 3-D cube" begin
    include(joinpath(@__DIR__, "..", "data", "Laplace", "cube_mesh.jl"))
    dad = format3d(mesh_cube(; L=1.0, ndiv=2, nome="t_ilayer3"), Laplace(1.0);
        pontointerno=false)
    pts = internal_layer(dad; every=2, fill=2)
    @test !isempty(pts)
    @test all(p -> all(0 .< p .< 1), pts)
    @test all(p -> point_in_domain(dad, p), pts)
end
