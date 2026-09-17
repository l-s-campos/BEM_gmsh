# =============================================================================
# 2D Laplace multipole / local expansions (complex log kernel)
#
# pot ≈ a₀ log|z| + Σ_{n=1}^p aₙ (rscale/z)^n     (outgoing / multipole)
# pot ≈ Σ_{n=0}^p bₙ (z/rscale)^n                   (incoming / local)
#
# Conventions match Flatiron FMM2D (l2d* routines): translations are incremental.
# Gradients are stored as the complex derivative d/dz; physical components are
#   ∂u/∂x = Re(du/dz),  ∂u/∂y = -Im(du/dz)
# =============================================================================

"""Scratch arrays for 2D M2M / L2L / M2L / P2M (sized to `nterms+1`)."""
mutable struct Laplace2DTransWS
    z0pow1::Vector{ComplexF64}
    z0pow2::Vector{ComplexF64}
    tmp1::Vector{ComplexF64}
    tmp2::Vector{ComplexF64}
end
function Laplace2DTransWS(nterms::Int)
    n = max(nterms + 1, 1)
    return Laplace2DTransWS(
        zeros(ComplexF64, n), zeros(ComplexF64, n),
        zeros(ComplexF64, n), zeros(ComplexF64, n),
    )
end
function _ensure_trans!(ws::Laplace2DTransWS, nmax::Int)
    n = nmax + 1
    if length(ws.z0pow1) < n
        resize!(ws.z0pow1, n); resize!(ws.z0pow2, n)
        resize!(ws.tmp1, n); resize!(ws.tmp2, n)
    end
    return ws
end

"""Number of expansion terms for relative precision `eps` (worst-case M2L).

Matches Flatiron `l2dterms`: decay of `(√2/2)^n / 1.5^{n+1}`.
"""
function laplace_nterms(eps::Real)
    ntmax = 200
    z1 = 1.5
    z2 = sqrt(2) / 2
    for j in 2:ntmax
        if abs((z2^j) / (z1^(j + 1))) < eps
            return j
        end
    end
    return ntmax
end

"""3D expansion order (Flatiron `l3dterms`): `(√3/2)^n / 1.5^{n+1}`."""
function laplace3d_nterms_flatiron(eps::Real)
    ntmax = 200
    z1 = 1.5
    z2 = sqrt(3) / 2
    for j in 2:ntmax
        if abs((z2^j) / (z1^(j + 1))) < eps
            return j
        end
    end
    return ntmax
end

"""Leaf size vs `eps`, Flatiron `lndiv2d` (quadtree `ndiv`)."""
function laplace2d_ndiv(eps::Real)
    e = Float64(eps)
    e >= 5e-1 && return 3
    e >= 5e-2 && return 5
    e >= 5e-3 && return 8
    e >= 5e-4 && return 10
    e >= 5e-7 && return 15
    e >= 5e-10 && return 20
    e >= 5e-13 && return 25
    return 45
end

"""Leaf size vs `eps` for 3D Laplace. Same buckets as Flatiron `lndiv`.

Plane-wave M2L is cheap, so FMM3D keeps large leaves (200 at `1e-6`, 400 at
`1e-8`). Equivalent-sphere M2L is O(K²) per pair; pass a smaller `nmax=` if
you want a deeper octree, or `p=` to cut K.
"""
function laplace3d_ndiv(eps::Real)
    e = Float64(eps)
    e >= 5e-3 && return 40
    e >= 5e-4 && return 100
    e >= 5e-7 && return 200
    e >= 5e-10 && return 400
    e >= 5e-13 && return 600
    return 700
end

"""Binomial coefficient table `carray[i,j] = C(i,j)` for `0 ≤ j ≤ i ≤ n`."""
function binomial_table(n::Int)
    C = zeros(Float64, n + 1, n + 1)
    for i in 0:n
        C[i + 1, 1] = 1.0
        for j in 1:i
            C[i + 1, j + 1] = C[i + 1, j] * (i - j + 1) / j
        end
    end
    return C
end

@inline function _to_complex(c::SVector{2,T}) where {T}
    return complex(c[1], c[2])
end

# ---------------------------------------------------------------------------
# Form multipole from charges / complex dipoles
# ---------------------------------------------------------------------------

"""
    form_multipole_charge!(mpole, rscale, center, sources, charges, irange)

Increment multipole coefficients about `center` from charges on `sources[irange]`.
`mpole` is indexed `0:nterms`.
"""
function form_multipole_charge!(
    mpole::AbstractVector{ComplexF64},
    rscale::Float64,
    center::SVector{2,Float64},
    sources::AbstractVector{SVector{2,Float64}},
    charges::AbstractVector{<:Number},
    irange::AbstractUnitRange,
)
    nterms = length(mpole) - 1
    zpow = Vector{ComplexF64}(undef, nterms + 1)
    return form_multipole_charge!(mpole, rscale, center, sources, charges, irange, zpow)
end
function form_multipole_charge!(
    mpole::AbstractVector{ComplexF64},
    rscale::Float64,
    center::SVector{2,Float64},
    sources::AbstractVector{SVector{2,Float64}},
    charges::AbstractVector{<:Number},
    irange::AbstractUnitRange,
    zpow::Vector{ComplexF64},
)
    nterms = length(mpole) - 1
    length(zpow) >= nterms + 1 || throw(ArgumentError("zpow too short"))
    for j in irange
        z0 = _to_complex(sources[j] - center)
        ztemp = z0 / rscale
        zpow[1] = 1
        zn = -one(ComplexF64)
        for n in 1:nterms
            zn *= ztemp
            zpow[n + 1] = zn / n
        end
        q = complex(charges[j])
        @inbounds for n in 0:nterms
            mpole[n + 1] += q * zpow[n + 1]
        end
    end
    return mpole
end

"""
    form_multipole_dipole!(mpole, rscale, center, sources, dipoles, irange)

Increment multipole from complex dipole strengths (`dipoles[j]` already includes
orientation: typically `dipstr * (-(vx + im*vy))`).
"""
function form_multipole_dipole!(
    mpole::AbstractVector{ComplexF64},
    rscale::Float64,
    center::SVector{2,Float64},
    sources::AbstractVector{SVector{2,Float64}},
    dipoles::AbstractVector{<:Number},
    irange::AbstractUnitRange,
)
    nterms = length(mpole) - 1
    nterms < 1 && return mpole
    zpow = Vector{ComplexF64}(undef, nterms)
    return form_multipole_dipole!(mpole, rscale, center, sources, dipoles, irange, zpow)
end
function form_multipole_dipole!(
    mpole::AbstractVector{ComplexF64},
    rscale::Float64,
    center::SVector{2,Float64},
    sources::AbstractVector{SVector{2,Float64}},
    dipoles::AbstractVector{<:Number},
    irange::AbstractUnitRange,
    zpow::Vector{ComplexF64},
)
    nterms = length(mpole) - 1
    nterms < 1 && return mpole
    length(zpow) >= nterms || throw(ArgumentError("zpow too short"))
    for j in irange
        z0 = _to_complex(sources[j] - center)
        ztemp = z0 / rscale
        zpow[1] = 1 / rscale
        for n in 2:nterms
            zpow[n] = zpow[n - 1] * ztemp
        end
        d = complex(dipoles[j])
        @inbounds for n in 1:nterms
            mpole[n + 1] += d * zpow[n]
        end
    end
    return mpole
end

# ---------------------------------------------------------------------------
# Form local from charges / dipoles (direct P2L)
# ---------------------------------------------------------------------------

function form_local_charge!(
    localexp::AbstractVector{ComplexF64},
    rscale::Float64,
    center::SVector{2,Float64},
    sources::AbstractVector{SVector{2,Float64}},
    charges::AbstractVector{<:Number},
    irange::AbstractUnitRange,
)
    nterms = length(localexp) - 1
    zpow = Vector{ComplexF64}(undef, nterms + 1)
    for j in irange
        z0 = _to_complex(sources[j] - center)
        abs(z0) == 0 && continue
        ztemp = rscale / z0
        zpow[1] = log(abs(z0))
        zn = -one(ComplexF64)
        for n in 1:nterms
            zn *= ztemp
            zpow[n + 1] = zn / n
        end
        q = complex(charges[j])
        @inbounds for n in 0:nterms
            localexp[n + 1] += q * zpow[n + 1]
        end
    end
    return localexp
end

function form_local_dipole!(
    localexp::AbstractVector{ComplexF64},
    rscale::Float64,
    center::SVector{2,Float64},
    sources::AbstractVector{SVector{2,Float64}},
    dipoles::AbstractVector{<:Number},
    irange::AbstractUnitRange,
)
    nterms = length(localexp) - 1
    zpow = Vector{ComplexF64}(undef, nterms + 1)
    for j in irange
        z0 = _to_complex(sources[j] - center)
        abs(z0) == 0 && continue
        zinv = 1 / z0
        ztemp = rscale * zinv
        # local_n += - dip * rscale^n / z0^(n+1)
        zn = -zinv
        for n in 0:nterms
            zpow[n + 1] = zn
            zn *= ztemp
        end
        d = complex(dipoles[j])
        @inbounds for n in 0:nterms
            localexp[n + 1] += d * zpow[n + 1]
        end
    end
    return localexp
end

# ---------------------------------------------------------------------------
# Evaluate multipole / local
# ---------------------------------------------------------------------------

"""Evaluate multipole expansion. Adds `real(φ)` to `pot` (log-kernel convention)."""
function eval_multipole!(
    pot::AbstractVector{<:Real},
    grad::Union{AbstractVector{<:Complex},Nothing},
    rscale::Float64,
    center::SVector{2,Float64},
    mpole::AbstractVector{ComplexF64},
    targets::AbstractVector{SVector{2,Float64}},
    irange::AbstractUnitRange,
)
    nterms = length(mpole) - 1
    rinv = 1 / rscale
    for k in irange
        z = _to_complex(targets[k] - center)
        absz = abs(z)
        absz == 0 && continue
        ztemp = rscale / z
        # φ = a0 log|z| + Σ an (rscale/z)^n  — physical pot is Re(φ)
        p = mpole[1] * log(absz)
        zn = ztemp
        for n in 1:nterms
            p += mpole[n + 1] * zn
            zn *= ztemp
        end
        pot[k] += real(p)
        if grad !== nothing
            g = mpole[1] / z
            zn = ztemp
            for n in 1:nterms
                g += mpole[n + 1] * (-n * rinv) * (zn * ztemp)
                zn *= ztemp
            end
            grad[k] += g
        end
    end
    return nothing
end

"""Evaluate local expansion. Adds `real(φ)` to `pot` (log-kernel convention)."""
function eval_local!(
    pot::AbstractVector{<:Real},
    grad::Union{AbstractVector{<:Complex},Nothing},
    rscale::Float64,
    center::SVector{2,Float64},
    localexp::AbstractVector{ComplexF64},
    targets::AbstractVector{SVector{2,Float64}},
    irange::AbstractUnitRange,
)
    nterms = length(localexp) - 1
    rinv = 1 / rscale
    for k in irange
        z = _to_complex(targets[k] - center) / rscale
        zn = one(ComplexF64)
        p = localexp[1]
        g = zero(ComplexF64)
        for n in 1:nterms
            g += localexp[n + 1] * n * rinv * zn
            zn *= z
            p += localexp[n + 1] * zn
        end
        pot[k] += real(p)
        if grad !== nothing
            grad[k] += g
        end
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Translations M2M, L2L, M2L
# ---------------------------------------------------------------------------

"""
    m2m!(hexp2, rscale2, center2, hexp1, rscale1, center1, carray)

Shift multipole `hexp1` (about `center1`) into multipole about `center2` (incremental).
"""
function m2m!(
    hexp2::AbstractVector{ComplexF64},
    rscale2::Float64,
    center2::SVector{2,Float64},
    hexp1::AbstractVector{ComplexF64},
    rscale1::Float64,
    center1::SVector{2,Float64},
    carray::AbstractMatrix{Float64},
)
    nmax = max(length(hexp1) - 1, length(hexp2) - 1)
    return m2m!(hexp2, rscale2, center2, hexp1, rscale1, center1, carray,
        Laplace2DTransWS(nmax))
end
function m2m!(
    hexp2::AbstractVector{ComplexF64},
    rscale2::Float64,
    center2::SVector{2,Float64},
    hexp1::AbstractVector{ComplexF64},
    rscale1::Float64,
    center1::SVector{2,Float64},
    carray::AbstractMatrix{Float64},
    ws::Laplace2DTransWS,
)
    nterms1 = length(hexp1) - 1
    nterms2 = length(hexp2) - 1
    nmax = max(nterms1, nterms2)
    _ensure_trans!(ws, nmax)
    z0 = -_to_complex(center2 - center1)
    z0pow1, z0pow2 = ws.z0pow1, ws.z0pow2
    hexp1tmp, hexp2tmp = ws.tmp1, ws.tmp2
    z0pow1[1] = 1
    z0pow2[1] = 1
    if nmax >= 1
        ztemp = rscale1 / z0
        z0pow1[2] = ztemp
        for i in 2:nmax
            z0pow1[i + 1] = z0pow1[i] * ztemp
        end
        ztemp = z0 / rscale2
        z0pow2[2] = ztemp
        for i in 2:nmax
            z0pow2[i + 1] = z0pow2[i] * ztemp
        end
    end
    @inbounds for i in 0:nterms1
        hexp1tmp[i + 1] = hexp1[i + 1] * z0pow1[i + 1]
    end
    fill!(hexp2tmp, 0)
    hexp2tmp[1] = hexp1[1]
    for i in 1:nterms2
        s = -hexp1tmp[1] / i
        for j in 1:min(i, nterms1)
            s += hexp1tmp[j + 1] * carray[i, j]
        end
        hexp2tmp[i + 1] = s * z0pow2[i + 1]
    end
    @inbounds for i in 0:nterms2
        hexp2[i + 1] += hexp2tmp[i + 1]
    end
    return hexp2
end

"""
    l2l!(jexp2, rscale2, center2, jexp1, rscale1, center1, carray)

Shift local expansion `jexp1` about `center1` to `center2` (incremental).
"""
function l2l!(
    jexp2::AbstractVector{ComplexF64},
    rscale2::Float64,
    center2::SVector{2,Float64},
    jexp1::AbstractVector{ComplexF64},
    rscale1::Float64,
    center1::SVector{2,Float64},
    carray::AbstractMatrix{Float64},
)
    nmax = max(length(jexp1) - 1, length(jexp2) - 1)
    return l2l!(jexp2, rscale2, center2, jexp1, rscale1, center1, carray,
        Laplace2DTransWS(nmax))
end
function l2l!(
    jexp2::AbstractVector{ComplexF64},
    rscale2::Float64,
    center2::SVector{2,Float64},
    jexp1::AbstractVector{ComplexF64},
    rscale1::Float64,
    center1::SVector{2,Float64},
    carray::AbstractMatrix{Float64},
    ws::Laplace2DTransWS,
)
    nterms1 = length(jexp1) - 1
    nterms2 = length(jexp2) - 1
    nmax = max(nterms1, nterms2)
    _ensure_trans!(ws, nmax)
    z0 = _to_complex(center2 - center1)
    z0pow1, z0pow2 = ws.z0pow1, ws.z0pow2
    jexp1tmp, jexp2tmp = ws.tmp1, ws.tmp2
    z0pow1[1] = 1
    z0pow2[1] = 1
    if nmax >= 1
        ztemp = z0 / rscale1
        z0pow1[2] = ztemp
        for i in 2:nmax
            z0pow1[i + 1] = z0pow1[i] * ztemp
        end
    end
    if nmax >= 1 && abs(z0) > 0
        ztemp = rscale2 / z0
        z0pow2[2] = ztemp
        for i in 2:nmax
            z0pow2[i + 1] = z0pow2[i] * ztemp
        end
    end
    @inbounds for i in 0:nterms1
        jexp1tmp[i + 1] = jexp1[i + 1] * z0pow1[i + 1]
    end
    fill!(jexp2tmp, 0)
    for i in 0:nterms2
        s = zero(ComplexF64)
        for j in i:nterms1
            s += jexp1tmp[j + 1] * carray[j + 1, i + 1]
        end
        jexp2tmp[i + 1] = s * z0pow2[i + 1]
    end
    @inbounds for i in 0:nterms2
        jexp2[i + 1] += jexp2tmp[i + 1]
    end
    return jexp2
end

"""
    m2l!(jexp2, rscale2, center2, hexp1, rscale1, center1, carray)

Convert multipole about `center1` to local about `center2` (incremental).
"""
function m2l!(
    jexp2::AbstractVector{ComplexF64},
    rscale2::Float64,
    center2::SVector{2,Float64},
    hexp1::AbstractVector{ComplexF64},
    rscale1::Float64,
    center1::SVector{2,Float64},
    carray::AbstractMatrix{Float64},
)
    nmax = max(length(hexp1) - 1, length(jexp2) - 1)
    return m2l!(jexp2, rscale2, center2, hexp1, rscale1, center1, carray,
        Laplace2DTransWS(nmax))
end
function m2l!(
    jexp2::AbstractVector{ComplexF64},
    rscale2::Float64,
    center2::SVector{2,Float64},
    hexp1::AbstractVector{ComplexF64},
    rscale1::Float64,
    center1::SVector{2,Float64},
    carray::AbstractMatrix{Float64},
    ws::Laplace2DTransWS,
)
    nterms1 = length(hexp1) - 1
    nterms2 = length(jexp2) - 1
    nmax = max(nterms1, nterms2)
    z0 = -_to_complex(center2 - center1)
    abs(z0) == 0 && return jexp2
    _ensure_trans!(ws, nmax)
    z0pow1, z0pow2 = ws.z0pow1, ws.z0pow2
    hexp1tmp, jexp2tmp = ws.tmp1, ws.tmp2
    z0pow1[1] = 1
    z0pow2[1] = 1
    ztemp1 = 1 / z0
    ztemp2 = ztemp1 * rscale2
    ztemp3 = -ztemp1 * rscale1
    for i in 1:nmax
        z0pow2[i + 1] = ztemp2
        z0pow1[i + 1] = ztemp3
        ztemp2 *= ztemp1 * rscale2
        ztemp3 = -ztemp3 * ztemp1 * rscale1
    end
    @inbounds for i in 0:nterms1
        hexp1tmp[i + 1] = hexp1[i + 1] * z0pow1[i + 1]
    end
    fill!(jexp2tmp, 0)
    rtmp = log(abs(z0))
    @inbounds begin
        jexp2tmp[1] = hexp1tmp[1] * rtmp
        for j in 1:nterms1
            jexp2tmp[1] += hexp1tmp[j + 1]
        end
        for i in 1:nterms2
            s = -hexp1tmp[1] / i
            for j in 1:nterms1
                s += hexp1tmp[j + 1] * carray[i + j, j]
            end
            jexp2tmp[i + 1] = s * z0pow2[i + 1]
        end
        for i in 0:nterms2
            jexp2[i + 1] += jexp2tmp[i + 1]
        end
    end
    return jexp2
end
