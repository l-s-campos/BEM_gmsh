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

"""Number of expansion terms for relative precision `eps` (worst-case M2L)."""
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
    for (iloc, k) in enumerate(irange)
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
        pot[iloc] += real(p)
        if grad !== nothing
            g = mpole[1] / z
            zn = ztemp
            for n in 1:nterms
                g += mpole[n + 1] * (-n * rinv) * (zn * ztemp)
                zn *= ztemp
            end
            grad[iloc] += g
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
    for (iloc, k) in enumerate(irange)
        z = _to_complex(targets[k] - center) / rscale
        zn = one(ComplexF64)
        p = localexp[1]
        g = zero(ComplexF64)
        for n in 1:nterms
            g += localexp[n + 1] * n * rinv * zn
            zn *= z
            p += localexp[n + 1] * zn
        end
        pot[iloc] += real(p)
        if grad !== nothing
            grad[iloc] += g
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
    nterms1 = length(hexp1) - 1
    nterms2 = length(hexp2) - 1
    nmax = max(nterms1, nterms2)
    z0 = -_to_complex(center2 - center1)

    z0pow1 = ones(ComplexF64, nmax + 1)
    if nmax >= 1
        ztemp = rscale1 / z0   # 1/(z0/rscale1)
        z0pow1[2] = ztemp
        for i in 2:nmax
            z0pow1[i + 1] = z0pow1[i] * ztemp
        end
    end
    z0pow2 = ones(ComplexF64, nmax + 1)
    if nmax >= 1
        ztemp = z0 / rscale2
        z0pow2[2] = ztemp
        for i in 2:nmax
            z0pow2[i + 1] = z0pow2[i] * ztemp
        end
    end

    hexp1tmp = [hexp1[i + 1] * z0pow1[i + 1] for i in 0:nterms1]
    hexp2tmp = zeros(ComplexF64, nterms2 + 1)
    hexp2tmp[1] = hexp1[1]
    for i in 1:nterms2
        s = -hexp1tmp[1] / i
        for j in 1:min(i, nterms1)
            # carray(i-1, j-1) in 0-based Fortran = C[i, j] in 1-based with C(n,k)=binom
            s += hexp1tmp[j + 1] * carray[i, j]  # C(i-1, j-1) -> row i, col j in 1-based binom table of order i-1 choose j-1
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
    nterms1 = length(jexp1) - 1
    nterms2 = length(jexp2) - 1
    nmax = max(nterms1, nterms2)
    z0 = _to_complex(center2 - center1)

    z0pow1 = ones(ComplexF64, nmax + 1)
    if nmax >= 1
        ztemp = z0 / rscale1
        z0pow1[2] = ztemp
        for i in 2:nmax
            z0pow1[i + 1] = z0pow1[i] * ztemp
        end
    end
    z0pow2 = ones(ComplexF64, nmax + 1)
    if nmax >= 1 && abs(z0) > 0
        ztemp = rscale2 / z0
        z0pow2[2] = ztemp
        for i in 2:nmax
            z0pow2[i + 1] = z0pow2[i] * ztemp
        end
    end

    jexp1tmp = [jexp1[i + 1] * z0pow1[i + 1] for i in 0:nterms1]
    jexp2tmp = zeros(ComplexF64, nterms2 + 1)
    for i in 0:nterms2
        s = zero(ComplexF64)
        for j in i:nterms1
            s += jexp1tmp[j + 1] * carray[j + 1, i + 1]  # C(j,i)
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
    nterms1 = length(hexp1) - 1
    nterms2 = length(jexp2) - 1
    nmax = max(nterms1, nterms2)
    z0 = -_to_complex(center2 - center1)
    abs(z0) == 0 && return jexp2

    # z0pow1[i] ~ (-rscale1/z0)^i for scaling hexp
    # z0pow2[i] ~ (rscale2/z0)^i
    z0pow1 = ones(ComplexF64, nmax + 1)
    z0pow2 = ones(ComplexF64, nmax + 1)
    ztemp1 = 1 / z0
    ztemp2 = ztemp1 * rscale2
    ztemp3 = -ztemp1 * rscale1
    for i in 1:nmax
        z0pow2[i + 1] = ztemp2
        z0pow1[i + 1] = ztemp3
        ztemp2 *= ztemp1 * rscale2
        ztemp3 = -ztemp3 * ztemp1 * rscale1
    end

    hexp1tmp = [hexp1[i + 1] * z0pow1[i + 1] for i in 0:nterms1]
    jexp2tmp = zeros(ComplexF64, nterms2 + 1)

    rtmp = log(abs(z0))
    jexp2tmp[1] = hexp1tmp[1] * rtmp
    for j in 1:nterms1
        jexp2tmp[1] += hexp1tmp[j + 1]
    end

    for i in 1:nterms2
        s = -hexp1tmp[1] / i
        for j in 1:nterms1
            # carray(i+j-1, j-1) = C(i+j-1, j-1)
            s += hexp1tmp[j + 1] * carray[i + j, j]  # row i+j-1+1, col j-1+1
        end
        jexp2tmp[i + 1] = s * z0pow2[i + 1]
    end

    @inbounds for i in 0:nterms2
        jexp2[i + 1] += jexp2tmp[i + 1]
    end
    return jexp2
end
