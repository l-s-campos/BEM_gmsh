"""
    ContactHalfSpace

Complete BEM formulation for normal and tangential contact of an elastic
half-space on a **uniform rectangular grid**, following

> R. Pohrt & Q. Li (2014), *Complete Boundary Element Formulation for
> Normal and Tangential Contact Problems*, Physical Mesomechanics 17(4).

# Contents
1. Influence coefficients ``K_{ab}`` (Boussinesq / Cerruti integrated over
   flat rectangular patches) — Eqs. (12)–(21)
2. FFT fast convolution ``u = \\mathcal{FC}_{ab}(b)`` — Eq. (23)
3. CG inverse ``b = \\mathcal{FC}_{ab}^{-1}(u, I_s)`` — Eqs. (24)–(31)
4. Coulomb partial-slip solver — Sect. 5

Sign convention (contact mechanics): positive normal pressure and positive
``u_z`` point **into** the half-space.
"""
module ContactHalfSpace

using LinearAlgebra
using FFTW
using Printf

export ElasticHalfSpace, InfluenceComponent
export Kxx, Kxy, Kxz, Kyx, Kyy, Kyz, Kzx, Kzy, Kzz
export influence_coeff, influence_kernel, precompute_kernels
export fc_forward!, fc_forward, fc_inverse
export solve_normal_contact, solve_partial_slip
export contact_modulus, hertz_pressure, hertz_halfwidth

# =============================================================================
# Material / grid
# =============================================================================

"""
    ElasticHalfSpace(G, ν; hx=1.0, hy=hx)

Elastic half-space with shear modulus `G`, Poisson ratio `ν`, and
rectangular cell sizes `hx`, `hy`.
"""
struct ElasticHalfSpace{T<:Real}
    G::T
    ν::T
    hx::T
    hy::T
end

ElasticHalfSpace(G::Real, ν::Real; hx::Real=1.0, hy::Real=hx) =
    ElasticHalfSpace(promote(float(G), float(ν), float(hx), float(hy))...)

shear_to_E(hs::ElasticHalfSpace) = 2hs.G * (1 + hs.ν)
"""Plane-strain contact modulus ``E* = E / (1-ν²) = 2G/(1-ν)``."""
contact_modulus(hs::ElasticHalfSpace) = 2hs.G / (1 - hs.ν)

@enum InfluenceComponent begin
    Kxx = 1
    Kxy = 2
    Kxz = 3
    Kyx = 4
    Kyy = 5
    Kyz = 6
    Kzx = 7
    Kzy = 8
    Kzz = 9
end

const COMPONENT_NAMES = Dict(
    Kxx => "xx", Kxy => "xy", Kxz => "xz",
    Kyx => "yx", Kyy => "yy", Kyz => "yz",
    Kzx => "zx", Kzy => "zy", Kzz => "zz",
)

# =============================================================================
# Geometry helpers for Love / Cerruti integrals  — Eq. (13)
# =============================================================================

"""Return `(k, m, l, n)` for relative indices `(di, dj) = (i-i', j-j')`."""
@inline function _kmnl(di::Real, dj::Real, hx::Real, hy::Real)
    k = (di + 0.5) * hx
    m = (dj + 0.5) * hy
    l = (di - 0.5) * hx
    n = (dj - 0.5) * hy
    return k, m, l, n
end

@inline _hypot2(a, b) = sqrt(a * a + b * b)

# safe logs / arctan used in the closed forms
@inline function _ln_ratio(num, den)
    # log(num/den) with floor to avoid log(0)
    return log(max(num, floatmin(typeof(num))) / max(den, floatmin(typeof(den))))
end

@inline function _atan_diff(y1, x1, y2, x2)
    # arctan(y1/x1) - arctan(y2/x2) via atan2 for quadrant safety
    return atan(y1, x1) - atan(y2, x2)
end

# =============================================================================
# Influence coefficients — Eqs. (12)–(21)
# =============================================================================

"""
    influence_coeff(comp::InfluenceComponent, di, dj, hs::ElasticHalfSpace) -> Float64

Influence coefficient ``K_{ab}^{ij,i'j'}`` for relative cell offset
`(di,dj) = (i-i', j-j')`.
"""
function influence_coeff(comp::InfluenceComponent, di::Real, dj::Real, hs::ElasticHalfSpace)
    G, ν, hx, hy = hs.G, hs.ν, hs.hx, hs.hy
    k, m, l, n = _kmnl(di, dj, hx, hy)
    return _coeff(comp, k, m, l, n, G, ν)
end

# Prefactors include π from classical Boussinesq/Cerruti (the printed paper
# formulae drop π in several places due to typesetting; continuum kernels are
# ∼1/(π G R)).

function _coeff(::Val{Kzz}, k, m, l, n, G, ν)
    # Love (1929) / Eq. (12)
    s_km = _hypot2(k, m); s_kn = _hypot2(k, n)
    s_lm = _hypot2(l, m); s_ln = _hypot2(l, n)
    F = (
        k * _ln_ratio(m + s_km, n + s_kn) +
        l * _ln_ratio(n + s_ln, m + s_lm) +
        m * _ln_ratio(k + s_km, l + s_lm) +
        n * _ln_ratio(l + s_ln, k + s_kn)
    )
    return (1 - ν) / (2π * G) * F
end

function _coeff(::Val{Kxz}, k, m, l, n, G, ν)
    # Eq. (14) — u_x due to p_z
    s_km2 = k * k + m * m
    s_lm2 = l * l + m * m
    s_ln2 = l * l + n * n
    s_kn2 = k * k + n * n
    F = 0.5 * (m * _ln_ratio(s_km2, s_lm2) + n * _ln_ratio(s_ln2, s_kn2)) +
        k * _atan_diff(m, k, n, k) +
        l * _atan_diff(n, l, m, l)
    return -(1 - 2ν) / (4π * G) * F
end

function _coeff(::Val{Kyz}, k, m, l, n, G, ν)
    # Eq. (15) — u_y due to p_z
    s_km2 = k * k + m * m
    s_kn2 = k * k + n * n
    s_ln2 = l * l + n * n
    s_lm2 = l * l + m * m
    F = 0.5 * (k * _ln_ratio(s_km2, s_kn2) + l * _ln_ratio(s_ln2, s_lm2)) +
        m * _atan_diff(k, m, l, m) +
        n * _atan_diff(l, n, k, n)
    return -(1 - 2ν) / (4π * G) * F
end

function _coeff(::Val{Kzx}, k, m, l, n, G, ν)
    # Eq. (16)
    return -_coeff(Val(Kxz), k, m, l, n, G, ν)
end

function _coeff(::Val{Kxx}, k, m, l, n, G, ν)
    # Eq. (17) — u_x due to τ_x
    s_km = _hypot2(k, m); s_kn = _hypot2(k, n)
    s_lm = _hypot2(l, m); s_ln = _hypot2(l, n)
    Fν = (
        k * _ln_ratio(m + s_km, n + s_kn) +
        l * _ln_ratio(n + s_ln, m + s_lm)
    )
    F1 = (
        m * _ln_ratio(k + s_km, l + s_lm) +
        n * _ln_ratio(l + s_ln, k + s_kn)
    )
    return 1 / (2π * G) * ((1 - ν) * Fν + F1)
end

function _coeff(::Val{Kyx}, k, m, l, n, G, ν)
    # Eq. (18) — u_y due to τ_x
    F = (
        _hypot2(n, k) - _hypot2(m, k) +
        _hypot2(m, l) - _hypot2(n, l)
    )
    return ν / (2π * G) * F
end

function _coeff(::Val{Kzy}, k, m, l, n, G, ν)
    # Eq. (19) — u_z due to τ_y
    return -_coeff(Val(Kyz), k, m, l, n, G, ν)
end

function _coeff(::Val{Kyy}, k, m, l, n, G, ν)
    # Eq. (20) — swap roles of (k,l)↔(m,n) relative to Kxx
    s_km = _hypot2(k, m); s_kn = _hypot2(k, n)
    s_lm = _hypot2(l, m); s_ln = _hypot2(l, n)
    Fν = (
        m * _ln_ratio(k + s_km, l + s_lm) +
        n * _ln_ratio(l + s_ln, k + s_kn)
    )
    F1 = (
        k * _ln_ratio(m + s_km, n + s_kn) +
        l * _ln_ratio(n + s_ln, m + s_lm)
    )
    return 1 / (2π * G) * ((1 - ν) * Fν + F1)
end

function _coeff(::Val{Kxy}, k, m, l, n, G, ν)
    # Eq. (21)
    return _coeff(Val(Kyx), k, m, l, n, G, ν)
end

# dispatch on enum
_coeff(comp::InfluenceComponent, k, m, l, n, G, ν) =
    _coeff(Val(comp), k, m, l, n, G, ν)

# =============================================================================
# Kernel construction for FFT convolution
# =============================================================================

"""
    influence_kernel(comp, nx, ny, hs) -> Matrix

Build the influence kernel on relative offsets
`di ∈ -(nx-1):(nx-1)`, `dj ∈ -(ny-1):(ny-1)`, stored in an array of size
`(2nx-1, 2ny-1)` with zero-frequency (di=dj=0) at index `(nx, ny)`.

Suitable for zero-padded FFT convolution of an `(nx × ny)` stress field.
"""
function influence_kernel(comp::InfluenceComponent, nx::Int, ny::Int, hs::ElasticHalfSpace)
    K = zeros(Float64, 2nx - 1, 2ny - 1)
    @inbounds for dj in -(ny - 1):(ny - 1)
        for di in -(nx - 1):(nx - 1)
            K[di + nx, dj + ny] = influence_coeff(comp, di, dj, hs)
        end
    end
    return K
end

"""
    precompute_kernels(nx, ny, hs; components=...) -> NamedTuple

FFT of zero-padded influence kernels for the requested components.
Default: all 9.
"""
function precompute_kernels(
    nx::Int, ny::Int, hs::ElasticHalfSpace;
    components=instances(InfluenceComponent),
)
    # padded FFT size (next pow2 optional; use exact 2N for clarity)
    Mx, My = 2nx, 2ny
    kernels = Dict{InfluenceComponent,Matrix{ComplexF64}}()
    scratch = zeros(Float64, Mx, My)
    for comp in components
        fill!(scratch, 0)
        K = influence_kernel(comp, nx, ny, hs)
        # place kernel with origin at (1,1) using circular wrap
        _embed_kernel!(scratch, K, nx, ny)
        kernels[comp] = rfft(scratch)
    end
    return (; nx, ny, Mx, My, kernels, hs)
end

function _embed_kernel!(dest::AbstractMatrix, K::AbstractMatrix, nx, ny)
    # K is (2nx-1)×(2ny-1) with centre at (nx,ny) = offset (0,0)
    # dest is Mx×My = 2nx × 2ny
    fill!(dest, 0)
    @inbounds for j in axes(K, 2), i in axes(K, 1)
        di = i - nx          # ∈ -(nx-1):(nx-1)
        dj = j - ny
        # circular indices in dest
        ii = di >= 0 ? di + 1 : size(dest, 1) + di + 1
        jj = dj >= 0 ? dj + 1 : size(dest, 2) + dj + 1
        dest[ii, jj] = K[i, j]
    end
    return dest
end

# =============================================================================
# Fast convolution  FC_ab(b)  — Eq. (23)
# =============================================================================

"""
    fc_forward(stress, comp, prep) -> deflection

``u = \\mathcal{FC}_{ab}(b)`` via FFT convolution using precomputed kernels
from [`precompute_kernels`](@ref).
"""
function fc_forward(stress::AbstractMatrix{<:Real}, comp::InfluenceComponent, prep)
    nx, ny = prep.nx, prep.ny
    size(stress) == (nx, ny) || throw(DimensionMismatch("stress size $(size(stress)) ≠ ($nx,$ny)"))
    u = zeros(Float64, nx, ny)
    fc_forward!(u, stress, comp, prep)
    return u
end

function fc_forward!(
    u::AbstractMatrix{<:Real},
    stress::AbstractMatrix{<:Real},
    comp::InfluenceComponent,
    prep,
)
    nx, ny, Mx, My = prep.nx, prep.ny, prep.Mx, prep.My
    haskey(prep.kernels, comp) || error("kernel $comp not precomputed")
    # pad stress
    pad = zeros(Float64, Mx, My)
    @inbounds for j in 1:ny, i in 1:nx
        pad[i, j] = stress[i, j]
    end
    ŝ = rfft(pad)
    û = prep.kernels[comp] .* ŝ
    full = irfft(û, Mx)
    @inbounds for j in 1:ny, i in 1:nx
        u[i, j] = full[i, j]
    end
    return u
end

"""Apply several components and accumulate (e.g. full 3×3 coupling)."""
function fc_forward_coupled(stresses::NamedTuple, comps::Vector{Pair{Symbol,InfluenceComponent}}, prep)
    nx, ny = prep.nx, prep.ny
    u = zeros(Float64, nx, ny)
    tmp = similar(u)
    for (skey, comp) in comps
        s = stresses[skey]
        fc_forward!(tmp, s, comp, prep)
        u .+= tmp
    end
    return u
end

# =============================================================================
# CG inverse  FC^{-1}(u, I_s)  — Eqs. (24)–(31)
# =============================================================================

"""
    fc_inverse(u, mask, comp, prep; tol=1e-8, maxiter=1000) -> stress

Find stresses supported on `mask` (BitMatrix / Bool) that produce deflection
`u` on that same set, with zero stress outside.  Polonsky–Keer / Pohrt–Li CG.
"""
function fc_inverse(
    u::AbstractMatrix{<:Real},
    mask::AbstractMatrix{Bool},
    comp::InfluenceComponent,
    prep;
    tol::Float64=1e-8,
    maxiter::Int=4 * length(u),
    σ0::Union{Nothing,AbstractMatrix}=nothing,
)
    nx, ny = prep.nx, prep.ny
    size(u) == (nx, ny) == size(mask) || throw(DimensionMismatch())

    σ = isnothing(σ0) ? zeros(Float64, nx, ny) : copy(σ0)
    σ .*= mask   # enforce support

    # r0 = u - FC(σ) on mask
    uσ = fc_forward(σ, comp, prep)
    r = zeros(Float64, nx, ny)
    d = zeros(Float64, nx, ny)
    @inbounds for j in 1:ny, i in 1:nx
        if mask[i, j]
            r[i, j] = u[i, j] - uσ[i, j]
            d[i, j] = r[i, j]
        end
    end

    rr = _dot_mask(r, r, mask)
    tol2 = tol^2 * max(rr, 1.0)
    z = zeros(Float64, nx, ny)

    for _ in 1:maxiter
        rr < tol2 && break
        fc_forward!(z, d, comp, prep)
        # z only used on mask
        dz = _dot_mask(d, z, mask)
        abs(dz) < eps(Float64) && break
        α = rr / dz
        @inbounds for j in 1:ny, i in 1:nx
            if mask[i, j]
                σ[i, j] += α * d[i, j]
                r[i, j] -= α * z[i, j]
            else
                σ[i, j] = 0
            end
        end
        rr_new = _dot_mask(r, r, mask)
        β = rr_new / rr
        @inbounds for j in 1:ny, i in 1:nx
            if mask[i, j]
                d[i, j] = r[i, j] + β * d[i, j]
            end
        end
        rr = rr_new
    end
    return σ
end

@inline function _dot_mask(a, b, mask)
    s = 0.0
    @inbounds for i in eachindex(mask)
        mask[i] && (s += a[i] * b[i])
    end
    return s
end

# =============================================================================
# Normal frictionless contact (CG with active set)
# =============================================================================

"""
    solve_normal_contact(gap0, δ, hs; tol=1e-8, maxiter=200) -> (;p, u, contact, prep)

Frictionless normal contact for a rigid indenter / elastic half-space.

- `gap0[i,j]`: initial gap (≥0 out of contact) at zero penetration
- `δ`: rigid-body indentation (positive into the half-space)

Solves ``u_z = δ - g_0`` on the contact set with ``p ≥ 0``, ``p = 0`` outside,
using CG + active-set (Polonsky–Keer style).
"""
function solve_normal_contact(
    gap0::AbstractMatrix{<:Real},
    δ::Real,
    hs::ElasticHalfSpace;
    tol=1e-8,
    maxiter=200,
    prep=nothing,
)
    nx, ny = size(gap0)
    prep = isnothing(prep) ? precompute_kernels(nx, ny, hs; components=(Kzz,)) : prep

    # target deflection if in full contact: u = δ - gap0
    u_target = δ .- gap0
    # initial contact guess: geometric interference
    contact = u_target .> 0
    p = zeros(Float64, nx, ny)

    for _it in 1:maxiter
        any(contact) || break
        # solve p on contact set for u = u_target
        p = fc_inverse(u_target, contact, Kzz, prep; tol=tol, σ0=p)
        # remove tensile
        updated = false
        @inbounds for i in eachindex(p)
            if contact[i] && p[i] < 0
                p[i] = 0
                contact[i] = false
                updated = true
            end
        end
        # compute deflection everywhere
        u = fc_forward(p, Kzz, prep)
        # add points that penetrate
        @inbounds for i in eachindex(u)
            if !contact[i] && (u[i] + gap0[i] - δ) < -tol * max(δ, 1.0)
                # still interpenetrating
                contact[i] = true
                updated = true
            end
        end
        !updated && all(p[i] >= -tol || !contact[i] for i in eachindex(p)) && break
    end
    u = fc_forward(p, Kzz, prep)
    return (; p, u, contact, prep, force=sum(p) * hs.hx * hs.hy)
end

# =============================================================================
# Partial slip with Coulomb friction — Sect. 5
# =============================================================================

"""
    solve_partial_slip(p, contact, d, μ_fric, hs; direction=:x, tol=1e-8) -> NamedTuple

Tangential partial-slip problem under Coulomb friction (Pohrt–Li §5).

# Arguments
- `p`: normal pressure from a prior normal solve (same grid)
- `contact`: contact mask
- `d`: imposed rigid tangential displacement in `direction`
- `μ_fric`: friction coefficient
- `hs`: half-space

# Returns
`(; τ, u_t, stick, slip, force_t, prep)`

Assumes decoupling (exact for ``ν = 1/2``); for other ``ν`` normal/tangential
coupling is neglected as in the paper's simplified algorithm.
"""
function solve_partial_slip(
    p::AbstractMatrix{<:Real},
    contact::AbstractMatrix{Bool},
    d::Real,
    μ_fric::Real,
    hs::ElasticHalfSpace;
    direction::Symbol=:x,
    tol=1e-8,
    maxiter=100,
    prep=nothing,
)
    nx, ny = size(p)
    comp = direction === :x ? Kxx : direction === :y ? Kyy :
           throw(ArgumentError("direction must be :x or :y"))

    comps = direction === :x ? (Kxx,) : (Kyy,)
    prep = isnothing(prep) ? precompute_kernels(nx, ny, hs; components=comps) : prep

    # start: full stick
    stick = copy(contact)
    slip = falses(nx, ny)
    τ = zeros(Float64, nx, ny)
    τ_slip = zeros(Float64, nx, ny)

    for _it in 1:maxiter
        # slip tractions at Coulomb limit, opposite to imposed motion sign
        fill!(τ_slip, 0)
        @inbounds for i in eachindex(slip)
            if slip[i]
                τ_slip[i] = sign(d) * μ_fric * p[i]   # resisting friction
            end
        end

        # deflection from slip tractions alone
        u_slip = fc_forward(τ_slip, comp, prep)

        # additional deflection needed in stick zone
        u_add = zeros(Float64, nx, ny)
        @inbounds for i in eachindex(stick)
            if stick[i]
                u_add[i] = d - u_slip[i]
            end
        end

        # stick tractions
        τ_stick = fc_inverse(u_add, stick, comp, prep; tol=tol)

        # points that exceed Coulomb → move to slip
        changed = false
        @inbounds for i in eachindex(stick)
            if stick[i] && abs(τ_stick[i]) > μ_fric * p[i] + tol
                stick[i] = false
                slip[i] = true
                changed = true
            end
        end

        # combine
        fill!(τ, 0)
        @inbounds for i in eachindex(τ)
            if stick[i]
                τ[i] = τ_stick[i]
            elseif slip[i]
                τ[i] = sign(d) * μ_fric * p[i]
            end
        end
        u_t = fc_forward(τ, comp, prep)

        # slip points that should stick (tangential deflection < d)
        @inbounds for i in eachindex(slip)
            if slip[i] && contact[i] && abs(u_t[i]) < abs(d) - tol &&
               abs(τ[i]) < μ_fric * p[i] - tol
                slip[i] = false
                stick[i] = true
                changed = true
            end
        end

        !changed && break
    end

    force_t = sum(τ) * hs.hx * hs.hy
    return (; τ, u_t=fc_forward(τ, comp, prep), stick, slip, force_t, prep)
end

# =============================================================================
# Analytical references (Hertz / Mindlin helpers)
# =============================================================================

"""Hertz half-width for cylinder is 2D; here sphere on flat: ``a = (3 F R / 4 E*)^{1/3}``."""
function hertz_halfwidth(F, R, hs::ElasticHalfSpace)
    Estar = contact_modulus(hs)
    return (3 * F * R / (4 * Estar))^(1 / 3)
end

"""Hertz pressure distribution on grid centred at origin."""
function hertz_pressure(F, R, hs::ElasticHalfSpace, x::AbstractVector, y::AbstractVector)
    a = hertz_halfwidth(F, R, hs)
    Estar = contact_modulus(hs)
    p0 = (6 * F * Estar^2 / (π^3 * R^2))^(1 / 3)
    nx, ny = length(x), length(y)
    p = zeros(Float64, nx, ny)
    @inbounds for j in 1:ny, i in 1:nx
        ρ2 = (x[i]^2 + y[j]^2) / a^2
        if ρ2 < 1
            p[i, j] = p0 * sqrt(1 - ρ2)
        end
    end
    return p, a, p0
end

function Base.show(io::IO, hs::ElasticHalfSpace)
    print(io, "ElasticHalfSpace(G=$(hs.G), ν=$(hs.ν), hx=$(hs.hx), hy=$(hs.hy))")
end

end # module
