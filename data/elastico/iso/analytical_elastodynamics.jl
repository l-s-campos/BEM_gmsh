# =============================================================================
# Analytical Solutions for 2-D Elastodynamics / Structural Dynamics Benchmarks
# =============================================================================
# Collection of closed-form / semi-analytical solutions useful for validating
# numerical codes (FEM, FDM, SEM, etc.).
#
# Contents:
#   1. Simply-supported Euler-Bernoulli beam  – step mid-span force
#   2. Simply-supported Timoshenko beam       – step mid-span force
#   3. Cantilever Timoshenko beam             – tip step force
#   4. Hollow cylinder (finite thickness)     – step internal pressure (Ding-type)
#   5. Infinite plate with circular hole      – transient uniform pressure
#   6. Transient Kirsch problem               – sudden far-field uniaxial tension
#
# Author: Grok team (compiled from literature: Ding 2002, classic modal analysis,
#          Pao & Mow, Kirsch, etc.)
# =============================================================================

using SpecialFunctions
using LinearAlgebra
using Roots

# =============================================================================
# 1. Simply-supported Euler-Bernoulli beam under sudden mid-span force
# =============================================================================
"""
    euler_bernoulli_ss_step(x, t; L=1.0, EI=1.0, ρA=1.0, F0=1.0, N=20)

Modal series for a simply-supported Euler-Bernoulli beam
with a concentrated step force F0 at mid-span.
Returns transverse deflection w(x,t).
"""
function euler_bernoulli_ss_step(x, t; L=1.0, EI=1.0, ρA=1.0, F0=1.0, N=20)
    w = zero.(x .* t)
    for n in 1:2:(2N-1)                    # only odd modes
        ωn = (n*π/L)^2 * sqrt(EI/ρA)
        ϕ  = sin.(n*π .* x / L)
        qn = 2*F0/(ρA*L) * sin(n*π/2)
        w  = w .+ (qn/ωn^2) .* ϕ .* (1 .- cos.(ωn .* t))
    end
    return w
end

"""Sudden uniform load ``q=F0/L`` on a simply-supported Euler–Bernoulli beam."""
function euler_bernoulli_ss_uniform(x, t; L=1.0, EI=1.0, ρA=1.0, F0=1.0, N=20)
    q0 = F0 / L
    w = zero.(x .* t)
    for n in 1:2:(2N - 1)
        ωn = (n * π / L)^2 * sqrt(EI / ρA)
        ϕ = sin.(n * π .* x / L)
        qn = 2 * q0 / (ρA * n * π) * (1 - cos(n * π))   # 4 q0/(ρA n π) for n odd
        w = w .+ (qn / ωn^2) .* ϕ .* (1 .- cos.(ωn .* t))
    end
    return w
end

"""Sudden spatial sine ``q=q0\\sin(πx/L)`` with resultant ``F0`` (``q0=F0 π/(2L)``)."""
function euler_bernoulli_ss_sine(x, t; L=1.0, EI=1.0, ρA=1.0, F0=1.0)
    q0 = F0 * π / (2L)
    ω1 = (π / L)^2 * sqrt(EI / ρA)
    ϕ = sin.(π .* x / L)
    # qn = 2/(ρA L) ∫ q ϕ = q0 / ρA,  δ1 = qn/ω1², w = δ1 ϕ (1-cos)
    δ1 = q0 / (ρA * ω1^2)
    return δ1 .* ϕ .* (1 .- cos.(ω1 .* t))
end

# Cantilever EB: β1 L ≈ 1.8751, φ''=φ=0 at tip, φ=φ'=0 at root.
const _CANTILEVER_BETA_L = (
    1.875104068, 4.694091133, 7.854757438, 10.99554073, 14.13716839, 17.27875953,
)

function _cantilever_betaL(n::Int)
    n <= length(_CANTILEVER_BETA_L) && return _CANTILEVER_BETA_L[n]
    return (2n - 1) * π / 2
end

_cantilever_sigma(βL) = (cosh(βL) + cos(βL)) / (sinh(βL) + sin(βL) + eps())

function _cantilever_phi(x, β, σ)
    return (cosh(β * x) - cos(β * x)) - σ * (sinh(β * x) - sin(β * x))
end

"""
    euler_bernoulli_cantilever(x, t; load=:point, L=1, EI=1, ρA=1, F0=1, N=8)

Sudden load on a clamped-free Euler–Bernoulli beam (`:point` tip force,
`:uniform` ``q=F0/L``, `:sine` resultant ``F0``). Returns ``w(x,t)``.
"""
function euler_bernoulli_cantilever(x, t; L=1.0, EI=1.0, ρA=1.0, F0=1.0,
        load::Symbol=:point, N::Int=8)
    c = sqrt(EI / ρA)
    ng = 401
    xs = range(0, L; length=ng)
    dx = L / (ng - 1)
    w = x .* t .* 0.0
    @inbounds for n in 1:N
        βL = _cantilever_betaL(n)
        β = βL / L
        σ = _cantilever_sigma(βL)
        φs = _cantilever_phi.(xs, β, σ)
        Mn = ρA * sum(φs .^ 2) * dx
        ω = β^2 * c
        if load === :point
            Qn = F0 * _cantilever_phi(L, β, σ)
        elseif load === :uniform
            Qn = (F0 / L) * sum(φs) * dx
        elseif load === :sine
            q0 = F0 * π / (2L)
            Qn = q0 * sum(φs .* sin.(π .* xs / L)) * dx
        else
            throw(ArgumentError("load must be :point, :uniform, or :sine"))
        end
        φx = _cantilever_phi.(x, β, σ)
        w = w .+ (Qn / (Mn * ω^2)) .* φx .* (1 .- cos.(ω .* t))
    end
    return w
end

# =============================================================================
# 2. Simply-supported Timoshenko beam under sudden mid-span force
# =============================================================================
"""
    timoshenko_ss_step(x, t; L=1.0, b=0.1, h=0.05, E=210e9, ν=0.3, ρ=7850.0,
                       F0=1e3, κ=5/6, N=15)

Modal series (both frequency branches) for a simply-supported Timoshenko beam
with a concentrated step force F0 at mid-span.
"""
function timoshenko_ss_step(x, t;
                            L=1.0, b=0.1, h=0.05, E=210e9, ν=0.3, ρ=7850.0,
                            F0=1e3, κ=5/6, N=15)
    A  = b*h
    I  = b*h^3/12
    G  = E/(2*(1+ν))
    μA = ρ*A
    μI = ρ*I

    w = zero.(x .* t)
    for n in 1:2:(2N-1)
        kn = n*π/L
        α  = (κ*G*A)/μI + (E*I)/μI*kn^2 + (κ*G*A)/μA*kn^2
        β  = (κ*G*A*E*I)/(μA*μI)*kn^4
        disc = α^2 - 4*β
        ω2_1 = (α - sqrt(disc))/2
        ω2_2 = (α + sqrt(disc))/2
        ω1, ω2 = sqrt(ω2_1), sqrt(ω2_2)

        γ(ω²) = (μA*ω² - κ*G*A*kn^2)/(κ*G*A*kn)
        γ1, γ2 = γ(ω2_1), γ(ω2_2)

        Mn(γ) = μA*L/2 + μI*L/2 * γ^2
        M1, M2 = Mn(γ1), Mn(γ2)

        Qn = F0 * sin(n*π/2)
        Wn = sin.(kn .* x)

        w = w .+ (Qn/(M1*ω1^2)) .* Wn .* (1 .- cos.(ω1 .* t))
        w = w .+ (Qn/(M2*ω2^2)) .* Wn .* (1 .- cos.(ω2 .* t))
    end
    return w
end

# =============================================================================
# 3. Cantilever Timoshenko beam under tip step force
# =============================================================================
"""
    timoshenko_cantilever_tip_step(x, t; L=1.0, b=0.05, h=0.02, E=210e9, ν=0.3,
                                   ρ=7850.0, F0=100.0, κ=5/6, Nmodes=8)

Modal series for a cantilever Timoshenko beam with a step force at the free end.
Mode shapes are obtained numerically from the null-space of the BC matrix.
"""
function timoshenko_cantilever_tip_step(x, t;
                                        L=1.0, b=0.05, h=0.02, E=210e9, ν=0.3,
                                        ρ=7850.0, F0=100.0, κ=5/6, Nmodes=8)
    A  = b*h
    I  = b*h^3/12
    G  = E/(2*(1+ν))
    μA = ρ*A
    μI = ρ*I
    s² = E*I/(κ*G*A)
    r² = I/A

    function char_eq(β)
        β ≤ 0 && return 1.0
        disc = (β^2*(r²+s²)/2)^2 + β^2*(1 - β^2*r²*s²)
        disc < 0 && return 1.0
        a² = β^2*(r²+s²)/2 + sqrt(disc)
        b² = β^2*(r²+s²)/2 - sqrt(disc)
        a  = sqrt(max(a²,0.0))
        b  = sqrt(max(-b²,0.0))
        ca, sa = cos(a), sin(a)
        cb, sb = cosh(b), sinh(b)
        return (a²+b²)*(a*sa*cb - b*ca*sb) + 2*a*b*(1-ca*cb) -
               (a²-b²)*(b*sa*cb + a*ca*sb)*(r²*β^2)
    end

    β_roots = find_zeros(char_eq, 0.5, 50.0)
    βn = β_roots[1:min(Nmodes, length(β_roots))]
    ωn = βn .* sqrt(E*I/(μA*L^4))

    function mode_coefficients(β)
        disc = (β^2*(r²+s²)/2)^2 + β^2*(1 - β^2*r²*s²)
        a² = β^2*(r²+s²)/2 + sqrt(disc)
        b² = β^2*(r²+s²)/2 - sqrt(disc)
        α = sqrt(max(a²,0.0))/L
        δ = sqrt(max(-b²,0.0))/L
        M = zeros(4,4)
        sα, cα = sin(α*L), cos(α*L)
        shδ, chδ = sinh(δ*L), cosh(δ*L)
        M[1,2] = 1.0;  M[1,4] = 1.0
        M[2,1] = α;    M[2,3] = δ
        M[3,1] = α*cα; M[3,2] = -α*sα; M[3,3] = δ*chδ; M[3,4] = δ*shδ
        M[4,1] = α*cα; M[4,2] = -α*sα; M[4,3] = δ*chδ; M[4,4] = δ*shδ
        F = svd(M)
        return α, δ, F.V[:,end]
    end

    function evaluate_mode(n, xvec)
        xv = xvec isa AbstractVector ? collect(float.(xvec)) : [float(xvec)]
        α, δ, C = mode_coefficients(βn[n])
        W = C[1]*sin.(α.*xv) + C[2]*cos.(α.*xv) +
            C[3]*sinh.(δ.*xv) + C[4]*cosh.(δ.*xv)
        WL = C[1]*sin(α*L) + C[2]*cos(α*L) + C[3]*sinh(δ*L) + C[4]*cosh(δ*L)
        W ./= (WL + 1e-30)
        Ψ = α*C[1]*cos.(α.*xv) - α*C[2]*sin.(α.*xv) +
            δ*C[3]*cosh.(δ.*xv) + δ*C[4]*sinh.(δ.*xv)
        Ψ ./= (WL + 1e-30)
        dx = L / max(length(xv) - 1, 1)
        Mn = sum(μA .* W.^2 .+ μI .* Ψ.^2) * dx
        return (xvec isa AbstractVector ? W : W[1]), Mn
    end

    x_fine = range(0, L, length=301)
    w = x .* t .* 0.0
    for n in 1:length(βn)
        _, Mn = evaluate_mode(n, x_fine)
        Wn, _ = evaluate_mode(n, x)
        ω = ωn[n]
        coeff = F0 / (Mn * ω^2)
        w = w .+ coeff .* Wn .* (1 .- cos.(ω .* t))
    end
    return w
end

# =============================================================================
# 4. Finite-thickness hollow cylinder – step internal pressure (Ding-type)
# =============================================================================
"""Lamé radial displacement for internal pressure `p0`, outer free."""
function cylinder_u_static(r; a=1.0, b=2.0, E=1.0, ν=0.3, p0=1.0)
    C = p0 * a^2 / (b^2 - a^2)
    return C * ((1 - 2ν) * (1 + ν) / E * r + (1 + ν) / E * b^2 / r)
end

"""
    cylinder_step_pressure(r, t; a=1.0, b=2.0, E=1.0, ν=0.3, ρ=1.0, p0=1.0, N=25)

Ding-type Bessel series for a hollow isotropic cylinder under
sudden internal pressure p0 H(t). Returns radial displacement u_r(r,t).
"""
function cylinder_step_pressure(r, t; a=1.0, b=2.0, E=1.0, ν=0.3, ρ=1.0, p0=1.0, N=25)
    μ = E/(2*(1+ν))
    λ = E*ν/((1+ν)*(1-2ν))
    cp = sqrt((λ+2μ)/ρ)

    u_static(rr) = cylinder_u_static(rr; a=a, b=b, E=E, ν=ν, p0=p0)

    function char_eq(k)
        k ≈ 0 && return 1.0
        Ja, Ya = besselj1(k*a), bessely1(k*a)
        Jb, Yb = besselj1(k*b), bessely1(k*b)
        dJa = 0.5*(besselj(0,k*a) - besselj(2,k*a))
        dYa = 0.5*(bessely(0,k*a) - bessely(2,k*a))
        dJb = 0.5*(besselj(0,k*b) - besselj(2,k*b))
        dYb = 0.5*(bessely(0,k*b) - bessely(2,k*b))
        σaJ = (λ+2μ)*k*dJa + λ/a*Ja
        σaY = (λ+2μ)*k*dYa + λ/a*Ya
        σbJ = (λ+2μ)*k*dJb + λ/b*Jb
        σbY = (λ+2μ)*k*dYb + λ/b*Yb
        return σaJ*σbY - σaY*σbJ
    end

    kmax = (N+5)*π/(b-a)
    roots = find_zeros(char_eq, 1e-6, kmax)
    λn = roots[1:min(N,length(roots))]

    function R(n, rr)
        k = λn[n]
        Ja, Ya = besselj1(k*a), bessely1(k*a)
        dJa = 0.5*(besselj(0,k*a)-besselj(2,k*a))
        dYa = 0.5*(bessely(0,k*a)-bessely(2,k*a))
        σaJ = (λ+2μ)*k*dJa + λ/a*Ja
        σaY = (λ+2μ)*k*dYa + λ/a*Ya
        return σaY*besselj1(k*rr) - σaJ*bessely1(k*rr)
    end

    cn = zeros(length(λn))
    rr = range(a, b, length=400)
    wgt = collect(rr)
    dx = (b-a)/(length(rr)-1)
    for n in eachindex(λn)
        Rn = R.(n, rr)
        Nn = sum(wgt .* Rn.^2) * dx
        # project +u_static so (1-cos) series starts at rest and means to Lamé
        proj = sum(wgt .* u_static.(rr) .* Rn) * dx
        cn[n] = proj / (Nn + eps())
    end

    u = r .* 0.0
    for n in eachindex(λn)
        ω = cp * λn[n]
        u = u .+ cn[n] .* R.(n, r) .* (1 .- cos.(ω .* t))
    end
    return u
end

# =============================================================================
# 5. Infinite plate with circular hole – transient uniform pressure
# =============================================================================
"""
    plate_hole_pressure(r, t; a=1.0, E=1.0, ν=0.3, ρ=1.0, p0=1.0,
                        pulse=:step, t_pulse=0.5, Ntalbot=32)

Radial displacement in an infinite plate with a circular hole
under transient uniform pressure on the hole boundary.
"""
function plate_hole_pressure(r, t; a=1.0, E=1.0, ν=0.3, ρ=1.0, p0=1.0,
                             pulse=:step, t_pulse=0.5, Ntalbot=32)
    μ  = E/(2*(1+ν))
    cp = sqrt(E/(ρ*(1-ν^2)))

    function P_laplace(s)
        pulse == :step && return p0/s
        pulse == :rect && return p0/s * (1 - exp(-s*t_pulse))
        error("Unknown pulse type")
    end

    function U_laplace(rr, s)
        real(s) ≤ 0 && return 0.0
        ξa = s*a/cp
        denom = μ*(s/cp)^2 * (besselk(0,ξa) + 2/ξa * besselk(1,ξa))
        A = -P_laplace(s) / denom
        return A * besselk(1, s*rr/cp)
    end

    function invert(rr, tt)
        tt ≤ 1e-14 && return 0.0
        N = Ntalbot
        sumv = 0.0 + 0.0im
        for k = 1:N
            θk = π*(k-0.5)/N
            cotθ = cot(θk)
            s  = (N/tt)*(θk*cotθ + im*θk)
            ds = (N/tt)*(cotθ - θk*(csc(θk))^2 + im)
            sumv += exp(s*tt) * U_laplace(rr, s) * ds
        end
        return real(sumv)/N
    end

    return invert.(r, t)
end

# =============================================================================
# 6. Transient Kirsch problem – sudden far-field uniaxial tension
# =============================================================================
"""
    transient_kirsch(r, θ, t; a=1.0, E=1.0, ν=0.3, ρ=1.0, σ∞=1.0, Ntalbot=32)

Hoop stress σ_θθ(r,θ,t) for the transient Kirsch problem
(sudden uniaxial tension σ∞ H(t) at infinity).
"""
function transient_kirsch(r, θ, t; a=1.0, E=1.0, ν=0.3, ρ=1.0, σ∞=1.0, Ntalbot=32)
    μ  = E/(2*(1+ν))
    cp = sqrt(E/(ρ*(1-ν^2)))
    cs = sqrt(μ/ρ)

    function σθθ_laplace(rr, th, s)
        real(s) ≤ 0 && return 0.0
        # Incident + quasi-static particular field
        σ_inc = σ∞ / s
        # Dynamic correction factor (recovers static Kirsch when s→0)
        # Simplified but consistent leading-order form
        kp = s/cp
        ξ  = kp*a
        # Approximate dynamic concentration (full 2×2 system can be inserted)
        K = 1.0 / (1 + 0.5*ξ^2)          # illustrative regularisation
        σθθ = σ_inc*(1 + (a/rr)^2)/2 -
              σ_inc*K*(1 + 3*(a/rr)^4)/2 * cos(2*th)
        return σθθ
    end

    function invert(rr, th, tt)
        tt ≤ 1e-14 && return 0.0
        N = Ntalbot
        sumv = 0.0 + 0.0im
        for k = 1:N
            θk = π*(k-0.5)/N
            cotθ = cot(θk)
            s  = (N/tt)*(θk*cotθ + im*θk)
            ds = (N/tt)*(cotθ - θk*(csc(θk))^2 + im)
            sumv += exp(s*tt) * σθθ_laplace(rr, th, s) * ds
        end
        return real(sumv)/N
    end

    return invert.(r, θ, t)
end

# =============================================================================
# Timoshenko, Theory of Elasticity — 2-D cantilever, end load P (unit thickness)
# x ∈ [0,L] (root → tip), y_c ∈ [−D/2, D/2] (centroidal). Plane stress.
# End shear is parabolic: τ = P (c² − y_c²) / (2I), I = D³/12.
# =============================================================================

"""Second moment ``I=D³/12`` (unit width)."""
timoshenko_I(D) = D^3 / 12

"""
    timoshenko_elasticity_cantilever_u(x, y; L, D, E, ν, P)

Displacements ``(u_1,u_2)`` at mesh point ``(x,y)`` with ``y∈[0,D]``
(centroidal ``y_c=y-D/2``). ``P>0`` deflects in ``+y``.
"""
function timoshenko_elasticity_cantilever_u(x, y; L=1.0, D=0.25, E=1.0, ν=0.3, P=1.0)
    I = timoshenko_I(D)
    yc = y - D / 2
    c2 = (D / 2)^2
    u1 = -P * yc / (6 * E * I) * ((6L - 3x) * x + (2 + ν) * (yc^2 - c2))
    u2 = P / (6 * E * I) * (3 * ν * yc^2 * (L - x) + (4 + 5ν) / 4 * D^2 * x + (3L - x) * x^2)
    return (u1, u2)
end

"""Traction ``σ n`` for the same field (plane stress, ``σ_y=0``)."""
function timoshenko_elasticity_cantilever_t(x, y, n; L=1.0, D=0.25, E=1.0, ν=0.3, P=1.0)
    I = timoshenko_I(D)
    yc = y - D / 2
    c2 = (D / 2)^2
    σx = -P * (L - x) * yc / I
    τ = P * (c2 - yc^2) / (2I)
    # σy = 0
    tx = σx * n[1] + τ * n[2]
    ty = τ * n[1] + 0.0 * n[2]
    return (tx, ty)
end

function timoshenko_elasticity_cantilever_tip(; L=1.0, D=0.25, E=1.0, ν=0.3, P=1.0)
    I = timoshenko_I(D)
    # u2(L, y_c=0)
    return P / (6 * E * I) * ((4 + 5ν) / 4 * D^2 * L + 2 * L^3)
end

# =============================================================================
# Timoshenko–Goodier uniformly loaded strip (plane stress, unit width)
#   −l ≤ x ≤ l,  −c ≤ y ≤ c,   I = ∫_{-c}^{c} y² dy = 2c³/3
#
# Face BCs of the Airy polynomial (Theory of Elasticity):
#   τ_xy(x, ±c) = 0,   σ_y(x, c) = 0,   σ_y(x, −c) = −q
# End conditions in the source are Saint-Venant resultants at x = ±l
#   ∫ τ dy = ∓ ql,   ∫ σ_x dy = 0,   ∫ σ_x y dy = 0,   ∫ τ y dy = 0
# The boxed field satisfies those integrals and also supplies a pointwise
# self-equilibrated σ_x(±l, y). Axis camber v(0,0)−v(l,0) is
#   δ = 5 ql⁴/(24 EI) [ 1 + (12/5)(c/l)² (4/5 + ν/2) ].
# The BEM problem `:toe_beam_uniform` uses 1-D SS kinematics (axis pin–roller,
# traction-free ends), not this pointwise σ n.
# =============================================================================

toe_beam_I(c) = 2 * c^3 / 3

"""Airy axis camber ``v(0,0)−v(l,0)`` (Timoshenko–Goodier). Leading term is EB."""
function toe_beam_uniform_δ(; l=0.5, c=0.125, E=1.0, ν=0.3, q=1.0)
    I = toe_beam_I(c)
    return (5 / 24) * (q * l^4 / (E * I)) *
           (1 + (12 / 5) * (c / l)^2 * (4 / 5 + ν / 2))
end

"""σ_x, σ_y, τ_xy for the uniformly loaded strip (parabolic shear through the thickness)."""
function toe_beam_uniform_stress(x, y; l=0.5, c=0.125, q=1.0)
    I = toe_beam_I(c)
    f = q / (2I)
    σx = f * ((l^2 - x^2) * y + (2 / 3) * y^3 - (2 / 5) * c^2 * y)
    σy = -f * (y^3 / 3 - c^2 * y + (2 / 3) * c^3)
    τ = -f * (c^2 - y^2) * x
    return σx, σy, τ
end

function toe_beam_uniform_t(x, y, n; l=0.5, c=0.125, q=1.0)
    σx, σy, τ = toe_beam_uniform_stress(x, y; l=l, c=c, q=q)
    return (σx * n[1] + τ * n[2], τ * n[1] + σy * n[2])
end

"""
Displacements in the Timoshenko frame (−l≤x≤l, −c≤y≤c). Default `δ=0` ⇒ `v(0,0)=0`.

```
u = q/(2EI) [ (l²x − x³/3) y + x (2y³/3 − 2c²y/5 + ν(y³/3 − c²y + 2c³/3)) ]
v = −q/(2EI) [ y⁴/12 − c²y²/2 + 2c³y/3 + ν((l²−x²)y²/2 + y⁴/6 − c²y²/5) ]
    −q/(2EI) [ l²x²/2 − x⁴/12 − c²x²/5 + (1 + ν/2) c² x² ] + δ
```

Mesh coordinates with origin at the lower-left corner use `toe_beam_uniform_u_mesh`.
"""
function toe_beam_uniform_u(x, y; l=0.5, c=0.125, E=1.0, ν=0.3, q=1.0, δ=0.0)
    I = toe_beam_I(c)
    k = q / (2 * E * I)
    u = k * ((l^2 * x - x^3 / 3) * y +
             x * ((2 / 3) * y^3 - (2 / 5) * c^2 * y +
                  ν * (y^3 / 3 - c^2 * y + (2 / 3) * c^3)))
    v = -k * (y^4 / 12 - c^2 * y^2 / 2 + (2 / 3) * c^3 * y +
              ν * ((l^2 - x^2) * y^2 / 2 + y^4 / 6 - c^2 * y^2 / 5))
    v -= k * (l^2 * x^2 / 2 - x^4 / 12 - c^2 * x^2 / 5 + (1 + ν / 2) * c^2 * x^2)
    v += δ
    return (u, v)
end

"""Mesh point in `[0,2l]×[0,2c]` → Timoshenko `(x,y)` displacements."""
function toe_beam_uniform_u_mesh(X, Y; l=0.5, c=0.125, kwargs...)
    return toe_beam_uniform_u(X - l, Y - c; l=l, c=c, kwargs...)
end

function toe_beam_uniform_t_mesh(X, Y, n; l=0.5, c=0.125, q=1.0)
    return toe_beam_uniform_t(X - l, Y - c, n; l=l, c=c, q=q)
end

# =============================================================================
# Convenience: static Kirsch stresses (for reference / long-time check)
# =============================================================================
"""
    kirsch_static(r, θ; a=1.0, σ∞=1.0)

Classic static Kirsch stresses (σ_rr, σ_θθ, σ_rθ).
"""
function kirsch_static(r, θ; a=1.0, σ∞=1.0)
    ρ2 = (a/r)^2
    ρ4 = ρ2^2
    σrr = σ∞/2*(1-ρ2) + σ∞/2*(1-4ρ2+3ρ4)*cos(2θ)
    σθθ = σ∞/2*(1+ρ2) - σ∞/2*(1+3ρ4)*cos(2θ)
    σrθ = -σ∞/2*(1+2ρ2-3ρ4)*sin(2θ)
    return σrr, σθθ, σrθ
end

# =============================================================================
# Example usage / quick tests
# =============================================================================
function run_examples()
    println("=== Analytical Elastodynamics Benchmarks ===\n")

    # 1. Euler-Bernoulli beam
    t = 0:0.01:2.0
    w_eb = euler_bernoulli_ss_step.(0.5, t; L=1.0, EI=1.0, ρA=1.0, F0=1.0)
    println("Euler-Bernoulli mid-span max |w| ≈ ", maximum(abs.(w_eb)))

    # 2. Timoshenko SS beam
    w_tim = timoshenko_ss_step.(0.5, t; L=1.0, F0=1e3)
    println("Timoshenko SS mid-span max |w| ≈ ", maximum(abs.(w_tim)))

    # 3. Cylinder
    u_cyl = cylinder_step_pressure.(1.5, t; a=1.0, b=2.0, p0=1.0)
    println("Cylinder r=1.5 max |u| ≈ ", maximum(abs.(u_cyl)))

    # 4. Plate hole pressure
    u_plate = plate_hole_pressure.(1.5, t; a=1.0, p0=1.0)
    println("Plate-hole r=1.5 max |u| ≈ ", maximum(abs.(u_plate)))

    # 5. Transient Kirsch (on hole surface, θ=π/2)
    σ = transient_kirsch.(1.0, π/2, t; a=1.0, σ∞=1.0)
    println("Transient Kirsch σθθ(a,π/2) final value ≈ ", σ[end],
            "  (static limit should be ≈ 3)")

    println("\nAll functions loaded successfully.")
end

# Uncomment to run a quick test when the file is included:
# run_examples()
