"""
    ContactHalfPlane2D

2D (plane-strain) line-contact BEM on an elastic half-plane, the 1-D analogue of
Pohrt & Li (2014). Kernels from the Flamant solution; comparison with classical
**Hertz line contact** (cylinder on a flat).

Sign convention: positive pressure and positive ``u`` point into the solid.
"""
module ContactHalfPlane2D

using LinearAlgebra
using Statistics
using FFTW

export ElasticHalfPlane2D
export influence_coeff_2d, influence_kernel_2d, precompute_kernel_2d
export fc_forward_2d, fc_inverse_2d
export solve_line_contact, solve_line_contact_force, solve_line_contact_hertz
export hertz_line, hertz_line_pressure

"""
    ElasticHalfPlane2D(G, ν; h=1.0)

Plane-strain elastic half-plane. Cell size `h` along the surface line.
"""
struct ElasticHalfPlane2D{T<:Real}
    G::T
    ν::T
    h::T
end
ElasticHalfPlane2D(G::Real, ν::Real; h::Real=1.0) =
    ElasticHalfPlane2D(promote(float(G), float(ν), float(h))...)

"""Contact modulus ``E* = 2G/(1-ν)`` (plane strain)."""
contact_modulus(hp::ElasticHalfPlane2D) = 2hp.G / (1 - hp.ν)

# =============================================================================
# Flamant influence for uniform pressure on a segment of length h
# =============================================================================

"""
    influence_coeff_2d(di, hp) -> Float64

Normal deflection at cell `i` due to unit pressure on cell `i+di`
(relative offset `di`).

Closed form of
```math
K = -\\frac{1-ν}{π G}\\int_{-h/2}^{h/2}\\ln|x-ξ|\\,dξ
```
with ``x = di·h`` (relative), up to an additive constant fixed so that
``K(0)`` matches the self-influence integral.
"""
function influence_coeff_2d(di::Real, hp::ElasticHalfPlane2D)
    G, ν, h = hp.G, hp.ν, hp.h
    # integration of -ln|x-ξ| over ξ ∈ [c-h/2, c+h/2], c = di*h, x = 0
    # I(a,b) = ∫_a^b ln|s| ds = [s ln|s| - s]_a^b   (principal value)
    a = (di - 0.5) * h
    b = (di + 0.5) * h
    I = _int_ln_abs(b) - _int_ln_abs(a)
    # u = -((1-ν)/(πG)) * ∫ p ln|r| dξ   with p=1
    return -(1 - ν) / (π * G) * I
end

@inline function _int_ln_abs(s::Real)
    # ∫ ln|s| ds = s ln|s| - s
    as = abs(s)
    as < eps(typeof(float(s))) && return 0.0
    return s * log(as) - s
end

function influence_kernel_2d(n::Int, hp::ElasticHalfPlane2D)
    K = zeros(Float64, 2n - 1)
    @inbounds for di in -(n - 1):(n - 1)
        K[di + n] = influence_coeff_2d(di, hp)
    end
    return K
end

function precompute_kernel_2d(n::Int, hp::ElasticHalfPlane2D)
    M = 2n
    pad = zeros(Float64, M)
    K = influence_kernel_2d(n, hp)
    # embed with origin at index 1 (circular)
    for di in -(n - 1):(n - 1)
        ii = di >= 0 ? di + 1 : M + di + 1
        pad[ii] = K[di + n]
    end
    return (; n, M, Khat=rfft(pad), hp, K)
end

function fc_forward_2d(p::AbstractVector{<:Real}, prep)
    n, M = prep.n, prep.M
    length(p) == n || throw(DimensionMismatch())
    pad = zeros(Float64, M)
    pad[1:n] .= p
    û = prep.Khat .* rfft(pad)
    full = irfft(û, M)
    return full[1:n]
end

"""
Non-negative inverse on `mask` with mean-removed log-kernel (rank-safe).

Uses projector `P = I - 11ᵀ/m` so that `P K P p = P u`, then `p ← max(p,0)`
with one active-set sweep. The log kernel has a near-null rigid mode; removing
the mean makes the system well-posed up to the force level (use
[`solve_line_contact_force`](@ref) when the load is prescribed).
"""
function fc_inverse_2d(u::AbstractVector{<:Real}, mask::AbstractVector{Bool}, prep;
    tol=1e-12, maxiter=200, p0=nothing)
    n = prep.n
    hp = prep.hp
    idx = findall(mask)
    m = length(idx)
    p = zeros(n)
    m == 0 && return p

    K = zeros(m, m)
    @inbounds for (a, i) in enumerate(idx), (c, j) in enumerate(idx)
        K[a, c] = influence_coeff_2d(i - j, hp)
    end
    # mean-removal projector
    P = Matrix{Float64}(I, m, m) .- 1 / m
    A = P * K * P
    b = P * u[idx]

    x = zeros(m)
    if p0 !== nothing
        @inbounds for (a, i) in enumerate(idx)
            x[a] = max(p0[i], 0.0)
        end
        x .-= mean(x)
    end
    active = trues(m)
    for _ in 1:maxiter
        ia = findall(active)
        length(ia) < 2 && break
        # solve on active set with mean-removal restricted to ia
        ma = length(ia)
        Pa = Matrix{Float64}(I, ma, ma) .- 1 / ma
        Ka = K[ia, ia]
        Aa = Pa * Ka * Pa
        ba = Pa * u[idx[ia]]
        xs = pinv(Aa; rtol=tol) * ba
        x[ia] .= xs
        @inbounds for a in 1:m
            if !active[a]
                x[a] = 0.0
            end
        end
        done = true
        @inbounds for a in ia
            if x[a] < -tol
                x[a] = 0.0
                active[a] = false
                done = false
            end
        end
        # residual on inactive
        res = K * x - u[idx]
        res .-= mean(res[active])  # only meaningful on active
        @inbounds for a in 1:m
            if !active[a] && res[a] < -tol
                active[a] = true
                done = false
            end
        end
        done && break
    end
    x .= max.(x, 0.0)
    @inbounds for (a, i) in enumerate(idx)
        p[i] = x[a]
    end
    return p
end

"""
    solve_line_contact(gap0, δ, hp; tol=1e-10) -> NamedTuple

Frictionless line contact. `gap0` = initial gap, `δ` = rigid indentation
(into the solid). Returns `(; p, u, contact, force, prep)`.
"""
function solve_line_contact(gap0::AbstractVector{<:Real}, δ::Real, hp::ElasticHalfPlane2D;
    tol=1e-10, maxiter=200)
    n = length(gap0)
    prep = precompute_kernel_2d(n, hp)
    u_tgt = δ .- gap0
    contact = u_tgt .> 0
    p = zeros(n)
    for _ in 1:maxiter
        any(contact) || break
        p = fc_inverse_2d(u_tgt, contact, prep; tol=tol, p0=p)
        updated = false
        @inbounds for i in eachindex(p)
            if contact[i] && p[i] < 0
                p[i] = 0; contact[i] = false; updated = true
            end
        end
        u = fc_forward_2d(p, prep)
        @inbounds for i in eachindex(u)
            if !contact[i] && (u[i] + gap0[i] - δ) < -tol * max(abs(δ), 1.0)
                contact[i] = true; updated = true
            end
        end
        !updated && break
    end
    u = fc_forward_2d(p, prep)
    return (; p, u, contact, force=sum(p) * hp.h, prep)
end

"""
    solve_line_contact_hertz(x, R, F, hp) -> NamedTuple

Force-driven comparison with Hertz line theory: fix the contact set to the
analytical half-width ``a(F)`` and solve for pressure under the Hertz gap
``u = (a² - x²)/(2R)`` (relative to the edge). Compares ``p`` and ``F`` to Hertz.
"""
function solve_line_contact_hertz(x::AbstractVector, R::Real, F::Real, hp::ElasticHalfPlane2D;
    tol=1e-12)
    hz = hertz_line(F, R, hp)
    n = length(x)
    prep = precompute_kernel_2d(n, hp)
    contact = abs.(x) .<= hz.a + 0.5hp.h
    # Hertz geometric closure (relative): parabolic gap
    u_tgt = zeros(n)
    @inbounds for i in 1:n
        if contact[i]
            u_tgt[i] = (hz.a^2 - x[i]^2) / (2R)
        end
    end
    p = fc_inverse_2d(u_tgt, contact, prep; tol=tol)
    # remove tensile if any
    @inbounds for i in eachindex(p)
        if p[i] < 0
            p[i] = 0
            contact[i] = false
        end
    end
    p = fc_inverse_2d(u_tgt, contact, prep; tol=tol, p0=p)
    u = fc_forward_2d(p, prep)
    return (; p, u, contact, force=sum(p) * hp.h, hz, prep)
end

"""
    solve_line_contact_force(gap0, F, hp; tol=1e-10) -> NamedTuple

Force-controlled frictionless line contact (Polonsky–Keer active-set CG).
Prescribes normal load `F = ∫ p dx` and returns `(; p, u, contact, force, prep)`.

Works with the Flamant log kernel by using mean-removed gaps on the contact
set (absolute deflection is defined only up to a constant on a half-plane).
"""
function solve_line_contact_force(
    gap0::AbstractVector{<:Real},
    F::Real,
    hp::ElasticHalfPlane2D;
    tol=1e-10,
    maxiter=400,
    p_init=nothing,
)
    n = length(gap0)
    h = hp.h
    prep = precompute_kernel_2d(n, hp)
    area = n * h
    p = p_init === nothing ? fill(float(F) / area, n) : copy(p_init)
    p .= max.(p, 0.0)
    s = sum(p) * h
    s > 0 && (p .*= F / s)

    g = zeros(n)
    t = zeros(n)
    for it in 1:maxiter
        u = fc_forward_2d(p, prep)
        @. g = gap0 + u
        Ael = findall(p .> 0)
        isempty(Ael) && break
        g .-= minimum(g[Ael])           # rigid shift: min gap on contact = 0
        ḡ = mean(g[Ael])
        r = zeros(n)
        @inbounds for i in Ael
            r[i] = g[i] - ḡ
        end
        # steepest descent / PK search direction
        t .= r
        dt = fc_forward_2d(t, prep)
        dt .-= mean(dt[Ael])
        num = dot(r[Ael], r[Ael])
        den = dot(t[Ael], dt[Ael])
        abs(den) < eps() && break
        α = num / den
        @. p = p - α * t
        @inbounds for i in 1:n
            p[i] < 0 && (p[i] = 0.0)
        end
        # points that penetrate must re-enter contact
        u = fc_forward_2d(p, prep)
        @. g = gap0 + u
        Ael = findall(p .> 0)
        isempty(Ael) && break
        g .-= minimum(g[Ael])
        @inbounds for i in 1:n
            if p[i] == 0 && g[i] < -tol * max(maximum(abs, g), 1.0)
                p[i] = eps()
            end
        end
        s = sum(p) * h
        s > 0 && (p .*= F / s)
        num < tol^2 * max(F, 1.0)^2 && break
    end
    u = fc_forward_2d(p, prep)
    contact = p .> 0
    return (; p, u, contact, force=sum(p) * h, prep)
end

# =============================================================================
# Hertz line contact (cylinder on flat)
# =============================================================================

"""
    hertz_line(F, R, hp) -> (; a, p0, Estar)

Classical Hertz line contact:
```math
a = \\sqrt{\\frac{4 F R}{π E*}},\\qquad p_0 = \\frac{2F}{π a},\\qquad
p(x)=p_0\\sqrt{1-(x/a)^2}.
```
"""
function hertz_line(F::Real, R::Real, hp::ElasticHalfPlane2D)
    Estar = contact_modulus(hp)
    a = sqrt(4 * F * R / (π * Estar))
    p0 = 2 * F / (π * a)
    return (; a, p0, Estar)
end

"""Sample Hertz pressure on coordinate vector `x` for given force `F` and radius `R`."""
function hertz_line_pressure(F, R, hp, x::AbstractVector)
    hz = hertz_line(F, R, hp)
    p = zeros(length(x))
    @inbounds for i in eachindex(x)
        ξ = x[i] / hz.a
        abs(ξ) < 1 && (p[i] = hz.p0 * sqrt(1 - ξ^2))
    end
    return p, hz
end

end # module
