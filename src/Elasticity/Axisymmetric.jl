# Axisymmetric elasticity fundamental solutions (ElastQuadraticoAxis)
# Kernels involve complete elliptic integrals K(m), E(m).

export AxisymmetricElasticity
export axisym_fundamental

"""
    AxisymmetricElasticity(; E=1.0, ν=0.3)

Axisymmetric linear elasticity in the ``(r,z)`` meridional plane.
Unknowns: ``(u_r, u_z)``. Tractions: ``(t_r, t_z)``.

Lamé ``λ, μ`` are cached at construction (same formulas as 3D / plane strain).
"""
mutable struct AxisymmetricElasticity{T} <: Vectorial
    E::T
    ν::T
    lambda::T
    mu::T
end

function AxisymmetricElasticity{T}(; E=one(T), ν=T(0.3)) where {T}
    λ, μ = lame_constants(E, ν, true)  # axisym uses 3D/plane-strain Lamé
    return AxisymmetricElasticity{T}(T(E), T(ν), T(λ), T(μ))
end

function AxisymmetricElasticity(; E=1.0, ν=0.3)
    T = float(promote_type(typeof(E), typeof(ν)))
    return AxisymmetricElasticity{T}(; E=T(E), ν=T(ν))
end

function AxisymmetricElasticity(E::Real, ν::Real)
    return AxisymmetricElasticity(; E, ν)
end

function Base.setproperty!(e::AxisymmetricElasticity{T}, name::Symbol, v) where {T}
    if name === :lambda || name === :mu
        throw(ArgumentError("set E or ν instead of `$name`"))
    end
    if name === :E || name === :ν || name === :nu
        fname = name === :nu ? :ν : name
        setfield!(e, fname, convert(T, v))
        λ, μ = lame_constants(e.E, e.ν, true)
        setfield!(e, :lambda, T(λ))
        setfield!(e, :mu, T(μ))
        return v
    end
    return setfield!(e, name, v)
end

shear_modulus(e::AxisymmetricElasticity) = e.mu
lame_μ(e::AxisymmetricElasticity) = e.mu
lame_λ(e::AxisymmetricElasticity) = e.lambda

"""
    axisym_fundamental(rp, zp, rq, zq, nr, nz, E, ν) -> (U, T)
    axisym_fundamental(rp, zp, rq, zq, nr, nz, props::AxisymmetricElasticity)

Displacement (`U`) and traction (`T`) kernels 2×2 at source ``(r_p,z_p)``,
field ``(r_q,z_q)``, normal ``(n_r,n_z)`` at the field point.

Ported from `calc_solfund.m` (Bakri / Bakri–elliptic form).
Uses `SpecialFunctions.ellipk` / `ellipe`.
"""
function axisym_fundamental(rp, zp, rq, zq, nr, nz, props::AxisymmetricElasticity)
    return axisym_fundamental(rp, zp, rq, zq, nr, nz, props.E, props.ν, props.mu)
end

function axisym_fundamental(rp, zp, rq, zq, nr, nz, E, ν)
    μ = E / (2(1 + ν))
    return axisym_fundamental(rp, zp, rq, zq, nr, nz, E, ν, μ)
end

function axisym_fundamental(rp, zp, rq, zq, nr, nz, E, ν, μ)
    XA = 1 / (16 * π^2 * μ * (1 - ν))
    ZZ = (zp - zq)^2
    XC = sqrt((rp + rq)^2 + ZZ)
    XD = (rp - rq)^2 + ZZ
    XD = max(XD, 1e-30)
    XC = max(XC, 1e-30)
    XDC2 = XD * XC * XC
    AM = min(max(2 * sqrt(max(rp * rq, 0)) / XC, 0.0), 1 - 1e-15)  # m parameter
    XN1 = 1 - 2ν
    XN2 = 2ν - 3
    XN3 = 3 - 4ν
    X1 = 2rp^2 + rq^2 + 2ZZ
    X2 = 2rp^2 + 3ZZ
    X3 = rp^2 + rq^2 + ZZ
    X4 = rp^2 - rq^2 + ZZ
    X5 = rq^2 - rp^2 + ZZ
    X6 = 2rp^2 - 3rq^2 + 4ZZ
    X7 = 3rq^2 + 2ZZ
    X8 = ZZ * X3 * X4 / XDC2

    rp_s = max(rp, 1e-14)
    rq_s = max(rq, 1e-14)
    UA1 = XA / (rp_s * rq_s * XC)
    UA2 = XA * (zp - zq) / (rp_s * XC)
    UA3 = XA * (zp - zq) / (rq_s * XC)
    UA4 = 2 * XA / XC

    UXXK = UA1 * (XN3 * X3 + ZZ)
    UXXE = UA1 * (-XN3 * XC * XC - ZZ * X3 / XD)
    UXYK = UA2
    UXYE = UA2 * (-X5 / XD)
    UYXK = -UA3
    UYXE = UA3 * X4 / XD
    UYYK = UA4 * XN3
    UYYE = UA4 * ZZ / XD

    # elliptic integrals K(m), E(m) with m = AM^2 in MATLAB ELLIPT convention
    # SpecialFunctions uses parameter m ∈ [0,1]
    m = AM^2
    m = min(m, 1 - 1e-15)
    K = SpecialFunctions.ellipk(m)
    Eell = SpecialFunctions.ellipe(m)

    Urr = UXXK * K + UXXE * Eell
    Urz = UXYK * K + UXYE * Eell
    Uzr = UYXK * K + UYXE * Eell
    Uzz = UYYK * K + UYYE * Eell
    U = @SMatrix [Urr Urz; Uzr Uzz]

    # Traction kernels (abbreviated full form from calc_solfund.m)
    XA1 = XA / (rp_s * rq_s * rq_s * XC)
    XA2 = (zp - zq) * XA / (rp_s * rq_s * XC * XD)
    XA3 = XA / (rp_s * XC * XD)
    XA5 = (zp - zq) * XA / (XC * XD)
    XA6 = XA / (rq_s * XC * XD)
    XA7 = 2 * (zp - zq) * XA / (XC * XD)

    TK1 = XA1 * (2ν * X1 - 1.5 * X2 + 0.5 * X8)
    TE1 = (XA1 / XD) * (-2ν * (rp^2 * X6 + ZZ * X7 + rq^4) + 3 * (rp^2 * X4 + ZZ * X1) - 2 * X8 * X3)
    TK2 = XA2 * (XN2 * XD + ZZ * X3 / (XC * XC))
    TE2 = XA2 * (3ZZ - XN2 * X3 - 4 * ZZ * X3 * X3 / XDC2)
    TK3 = XA3 * (-XN1 * XD + ZZ * X5 / (XC * XC))
    TE3 = XA3 * (-ZZ + XN1 * X5 + 8 * ZZ * rp^2 * X4 / XDC2)
    TK5 = XA5 * (XD / (rq_s * rq_s) + 2 * ZZ / (XC * XC))
    TE5 = XA5 * (4 * (1 + ν) - X3 / (rq_s * rq_s) - 8 * ZZ * X3 / XDC2)
    TK6 = XA6 * (-XN1 * XD - ZZ * X4 / (XC * XC))
    TE6 = XA6 * (-3ZZ + XN1 * X4 + 4 * ZZ * X3 * X4 / XDC2)
    TK7 = XA7 * (-ZZ / (XC * XC))
    TE7 = XA7 * (XN1 + 4 * ZZ * X3 / XDC2)

    TXXK = 2μ * (TK1 * nr + TK2 * nz)
    TXXE = 2μ * (TE1 * nr + TE2 * nz)
    TXYK = 2μ * (TK3 * nz + TK2 * nr)
    TXYE = 2μ * (TE3 * nz + TE2 * nr)
    TYXK = 2μ * (TK5 * nr + TK6 * nz)
    TYXE = 2μ * (TE5 * nr + TE6 * nz)
    TYYK = 2μ * (TK7 * nz + TK6 * nr)
    TYYE = 2μ * (TE7 * nz + TE6 * nr)

    Trr = TXXK * K + TXXE * Eell
    Trz = TXYK * K + TXYE * Eell
    Tzr = TYXK * K + TYXE * Eell
    Tzz = TYYK * K + TYYE * Eell
    Tker = @SMatrix [Trr Trz; Tzr Tzz]
    return U, Tker
end

function fundamental(props::AxisymmetricElasticity, r::SVector{2}, n::SVector{2})
    # interpret r as (rq - rp) is NOT enough — axisym needs absolute r,z.
    # Store field at y=r absolute via convention: caller should use axisym_fundamental.
    # Fallback: treat as local with source at origin on axis (rp=0) — limited.
    error("Axisymmetric kernels need absolute (r,z); use axisym_fundamental(rp,zp,rq,zq,nr,nz,E,ν)")
end
