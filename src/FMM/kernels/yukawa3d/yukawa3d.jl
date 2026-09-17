# =============================================================================
# 3D Modified Helmholtz (Yukawa) FMM — kernel e^{-κ r}/(4π r)
#
# From exafmm-t. Expansion (|R| > |r|):
#
#   e^{-κ|R-r|}/|R-r|
#     = κ Σ_n (2n+1) i_n(κ r) k_n(κ R) P_n(cos γ)
#
# with modified spherical Bessel i_n, k_n.
# =============================================================================

using SpecialFunctions: besseli, besselk

const INV4PI_Y = 1 / (4π)

"""Modified spherical Bessel i_n(x) = √(π/(2x)) I_{n+1/2}(x)."""
@inline function spherical_in(n::Int, x::Float64)
    x < 1e-14 && return n == 0 ? 1.0 : 0.0
    return sqrt(π / (2x)) * besseli(n + 0.5, x)
end

"""Modified spherical Bessel k_n(x) = √(2/(π x)) K_{n+1/2}(x)."""
@inline function spherical_kn(n::Int, x::Float64)
    return sqrt(2 / (π * x)) * besselk(n + 0.5, x)
end

function yukawa3d_nterms(eps::Real, κ::Real, boxscale::Real)
    p0 = laplace_nterms(Float64(eps))
    extra = ceil(Int, abs(κ) * boxscale)
    return min(max(p0 + extra + 2, 6), 30)
end

@inline _yncoeff(p::Int) = (p + 1)^2
@inline _yoff(n::Int) = n * n

function form_ympole3d!(
    mpole::Vector{Float64},
    center::SVector{3,Float64},
    κ::Float64,
    sources::Vector{SVector{3,Float64}},
    charges::AbstractVector{<:Real},
    irange,
    p::Int,
)
    fill!(mpole, 0.0)
    for i in irange
        d = sources[i] - center
        rs, θs, φs = _cart2sph(d)
        cts = cos(θs)
        q = Float64(charges[i])
        for n in 0:p
            iv = spherical_in(n, κ * rs)
            off = _yoff(n)
            mpole[off + 1] += q * iv * _legendre_pm(n, 0, cts)
            for m in 1:n
                nmf = _fact_ratio(n, m)
                P = _legendre_pm(n, m, cts)
                c = q * iv * nmf * P
                mpole[off + 2m] += c * cos(-m * φs)
                mpole[off + 2m + 1] += c * sin(-m * φs)
            end
        end
    end
    return mpole
end

function eval_ympole3d_series(
    d::SVector{3,Float64},
    mpole::Vector{Float64},
    κ::Float64,
    p::Int,
)
    rt, θt, φt = _cart2sph(d)
    rt < 1e-30 && return 0.0
    ctt = cos(θt)
    s = 0.0
    for n in 0:p
        kv = spherical_kn(n, κ * rt)
        fac = (2n + 1) * kv
        off = _yoff(n)
        s += fac * mpole[off + 1] * _legendre_pm(n, 0, ctt)
        for m in 1:n
            Pt = _legendre_pm(n, m, ctt)
            Mr, Mi = mpole[off + 2m], mpole[off + 2m + 1]
            s += fac * 2 * (Mr * cos(m * φt) - Mi * sin(m * φt)) * Pt
        end
    end
    return INV4PI_Y * κ * s
end

function eval_ympole3d!(
    pot::AbstractVector{<:Real},
    center::SVector{3,Float64},
    mpole::Vector{Float64},
    κ::Float64,
    targets::Vector{SVector{3,Float64}},
    irange,
    p::Int,
)
    for (jt, j) in enumerate(irange)
        pot[jt] += eval_ympole3d_series(targets[j] - center, mpole, κ, p)
    end
    return pot
end

"""
```julia
vals = yfmm3d(eps, κ, sources; charges, targets=nothing, pg=0, pgt=0, nmax=40, η=1.2)
```

3D modified Helmholtz (Yukawa) FMM for ``e^{-κ r}/(4π r)`` (exafmm-t kernel).
"""
function yfmm3d(
    eps::Real,
    κ::Real,
    sources::AbstractMatrix{<:Real};
    charges,
    targets=nothing,
    pg::Integer=0,
    pgt::Integer=0,
    nmax::Integer=40,
    η::Real=1.2,
    threaded::Bool=false,
)
    κ = Float64(κ)
    @assert κ > 0 "Yukawa parameter κ must be positive"
    @assert size(sources, 1) == 3
    ns = size(sources, 2)
    @assert pg > 0 || pgt > 0

    spl = GeometricSplitter(nmax=Int(nmax))
    stree, sl2g, src = build_point_tree(sources, spl)
    L = diameter(container(stree))
    p = yukawa3d_nterms(eps, κ, L / 4)
    ncoeff = _yncoeff(p)
    adm = StrongAdmissibility(η=Float64(η))
    ch = permute_to_local(vec(charges), sl2g)

    leaf_mp = Dict{Int,Vector{Float64}}()
    for leaf in leaves(stree)
        ctr = SVector{3,Float64}(center(container(leaf)))
        mp = zeros(Float64, ncoeff)
        form_ympole3d!(mp, ctr, κ, src, ch, index_range(leaf), p)
        leaf_mp[node_id(leaf)] = mp
    end

    vals = FMMVals()

    function run(tmat, is_src)
        nt = size(tmat, 2)
        if is_src
            ttree, tl2g, tpts = stree, sl2g, src
            same = true
        else
            ttree, tl2g, tpts = build_point_tree(tmat, spl)
            same = false
        end
        pot_loc = zeros(Float64, nt)

        function m2p!(tnode, snode, _, __)
            mp = leaf_mp[node_id(snode)]
            ctr = SVector{3,Float64}(center(container(snode)))
            buf = zeros(Float64, length(index_range(tnode)))
            eval_ympole3d!(buf, ctr, mp, κ, tpts, index_range(tnode), p)
            for (jt, j) in enumerate(index_range(tnode))
                pot_loc[j] += buf[jt]
            end
        end

        function p2p!(tnode, snode)
            for j in index_range(tnode)
                tj = tpts[j]
                s = 0.0
                for i in index_range(snode)
                    same && i == j && continue
                    r = norm(tj - src[i])
                    r < 1e-30 && continue
                    s += ch[i] * exp(-κ * r) * INV4PI_Y / r
                end
                pot_loc[j] += s
            end
        end

        sdata = allocate_expdata(stree, 1)
        dualtree_upward!(stree, sdata, (n, e) -> nothing, (a, b) -> nothing)
        dualtree_interact!(ttree, stree, sdata, sdata, adm; m2p!, p2p!, threaded=threaded)

        pot_g = zeros(Float64, nt)
        unpermute!(pot_g, pot_loc, tl2g)
        return pot_g
    end

    pg > 0 && (vals.pot = run(sources, true))
    pgt > 0 && targets !== nothing && (vals.pottarg = run(targets, false))
    vals.ier = 0
    return vals
end

function y3ddir(
    κ::Real,
    sources::AbstractMatrix{<:Real},
    targets::AbstractMatrix{<:Real};
    charges,
    thresh::Float64=0.0,
)
    κ = Float64(κ)
    ns = size(sources, 2)
    nt = size(targets, 2)
    pot = zeros(Float64, nt)
    @inbounds for j in 1:nt
        s = 0.0
        for i in 1:ns
            r = hypot(
                targets[1, j] - sources[1, i],
                targets[2, j] - sources[2, i],
                targets[3, j] - sources[3, i],
            )
            r <= thresh && continue
            s += charges[i] * exp(-κ * r) * INV4PI_Y / r
        end
        pot[j] = s
    end
    vals = FMMVals()
    vals.pottarg = pot
    vals.ier = 0
    return vals
end
