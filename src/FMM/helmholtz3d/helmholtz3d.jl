# =============================================================================
# 3D Helmholtz FMM — kernel e^{ikr}/(4π r)
#
# Spherical multipole expansion (|R| > |r|):
#
#   e^{ik|R-r|}/|R-r|
#     = ik Σ_n (2n+1) j_n(kr) h_n^{(1)}(kR) P_n(cos γ)
#
# with associated-Legendre expansion of P_n. Leaf multipoles + dual-tree M2P.
# =============================================================================

using SpecialFunctions: sphericalbesselj, sphericalbessely

const INV4PI_H = 1 / (4π)

@inline sphericalhankel1(n, z) = sphericalbesselj(n, z) + im * sphericalbessely(n, z)

function helmholtz3d_nterms(eps::Real, zk::Number, boxscale::Real)
    p0 = laplace_nterms(Float64(eps))
    extra = ceil(Int, abs(complex(zk)) * boxscale)
    return min(max(p0 + extra + 2, 6), 40)
end

@inline _hncoeff(p::Int) = (p + 1)^2
@inline _hoff(n::Int) = n * n  # 0-based offset for degree n (m = -n..n)

"""
Form multipole about `center`.
Index layout: for degree n, entries `mpole[n² + (m+n) + 1]` hold M_n^m for m=-n..n.

M_n^0 = q j_n(k r_s) P_n(cos θ_s)
M_n^{±m} = q j_n nmf P_n^m e^{∓ i m φ_s}
"""
function form_hmpole3d!(
    mpole::Vector{ComplexF64},
    center::SVector{3,Float64},
    zk::ComplexF64,
    sources::Vector{SVector{3,Float64}},
    charges::AbstractVector{<:Number},
    dipoles::Union{Nothing,Vector{SVector{3,ComplexF64}}},
    irange,
    p::Int,
)
    fill!(mpole, 0)
    for i in irange
        d = sources[i] - center
        rs, θs, φs = _cart2sph(d)
        cts = cos(θs)
        q = complex(charges[i])
        for n in 0:p
            jn = rs < 1e-14 ? (n == 0 ? one(ComplexF64) : zero(ComplexF64)) :
                 complex(sphericalbesselj(n, zk * rs))
            off = _hoff(n)
            mpole[off + n + 1] += q * jn * _legendre_pm(n, 0, cts)
            for m in 1:n
                nmf = _fact_ratio(n, m)
                P = _legendre_pm(n, m, cts)
                c = q * jn * nmf * P
                mpole[off + (m + n) + 1] += c * exp(-im * m * φs)
                mpole[off + (-m + n) + 1] += c * exp(im * m * φs)
            end
        end
        if dipoles !== nothing
            δ = 1e-7
            dip = dipoles[i]
            for (ehat, dval) in (
                (SVector(1.0, 0.0, 0.0), dip[1]),
                (SVector(0.0, 1.0, 0.0), dip[2]),
                (SVector(0.0, 0.0, 1.0), dip[3]),
            )
                abs(dval) < 1e-30 && continue
                for sgn in (1.0, -1.0)
                    ds = sources[i] + (sgn * δ) * ehat - center
                    rs2, θ2, φ2 = _cart2sph(ds)
                    ct2 = cos(θ2)
                    w = dval / (2δ) * sgn
                    for n in 0:p
                        jn = rs2 < 1e-14 ? (n == 0 ? one(ComplexF64) : zero(ComplexF64)) :
                             complex(sphericalbesselj(n, zk * rs2))
                        off = _hoff(n)
                        mpole[off + n + 1] += w * jn * _legendre_pm(n, 0, ct2)
                        for m in 1:n
                            nmf = _fact_ratio(n, m)
                            P = _legendre_pm(n, m, ct2)
                            c = w * jn * nmf * P
                            mpole[off + (m + n) + 1] += c * exp(-im * m * φ2)
                            mpole[off + (-m + n) + 1] += c * exp(im * m * φ2)
                        end
                    end
                end
            end
        end
    end
    return mpole
end

function eval_hmpole3d_series(
    d::SVector{3,Float64},
    mpole::Vector{ComplexF64},
    zk::ComplexF64,
    p::Int,
)
    rt, θt, φt = _cart2sph(d)
    rt < 1e-30 && return zero(ComplexF64)
    ctt = cos(θt)
    s = zero(ComplexF64)
    for n in 0:p
        hn = sphericalhankel1(n, zk * rt)
        fac = (2n + 1) * hn
        off = _hoff(n)
        s += fac * mpole[off + n + 1] * _legendre_pm(n, 0, ctt)
        for m in 1:n
            Pt = _legendre_pm(n, m, ctt)
            Mp = mpole[off + (m + n) + 1]
            Mm = mpole[off + (-m + n) + 1]
            s += fac * Pt * (Mp * exp(im * m * φt) + Mm * exp(-im * m * φt))
        end
    end
    return INV4PI_H * im * zk * s
end

function eval_hmpole3d!(
    pot::AbstractVector{<:Complex},
    center::SVector{3,Float64},
    mpole::Vector{ComplexF64},
    zk::ComplexF64,
    targets::Vector{SVector{3,Float64}},
    irange,
    p::Int,
)
    for (jt, j) in enumerate(irange)
        pot[jt] += eval_hmpole3d_series(targets[j] - center, mpole, zk, p)
    end
    return pot
end

"""
```julia
vals = hfmm3d(eps, zk, sources; charges=nothing, dipvecs=nothing,
              targets=nothing, pg=0, pgt=0, nmax=40, η=1.2)
```

3D Helmholtz FMM for ``e^{ikr}/(4\\pi r)``.
"""
function hfmm3d(
    eps::Real,
    zk::Number,
    sources::AbstractMatrix{<:Real};
    charges=nothing,
    dipvecs=nothing,
    targets=nothing,
    pg::Integer=0,
    pgt::Integer=0,
    nmax::Integer=40,
    η::Real=1.2,
    threaded::Bool=false,
)
    zk = complex(Float64(real(zk)), Float64(imag(complex(zk))))
    @assert size(sources, 1) == 3
    ns = size(sources, 2)
    has_c = charges !== nothing
    has_d = dipvecs !== nothing
    @assert has_c || has_d
    @assert pg > 0 || pgt > 0

    spl = GeometricSplitter(nmax=Int(nmax))
    stree, sl2g, src = build_point_tree(sources, spl)
    L = diameter(container(stree))
    p = helmholtz3d_nterms(eps, zk, L / 4)
    ncoeff = _hncoeff(p)
    adm = StrongAdmissibility(η=Float64(η))

    ch = has_c ? permute_to_local_complex(vec(charges), sl2g) : zeros(ComplexF64, ns)
    dips = nothing
    if has_d
        dips = Vector{SVector{3,ComplexF64}}(undef, ns)
        @inbounds for i in 1:ns
            g = sl2g[i]
            dips[i] = SVector{3,ComplexF64}(
                complex(dipvecs[1, g]), complex(dipvecs[2, g]), complex(dipvecs[3, g]),
            )
        end
    end

    leaf_mp = Dict{UInt,Vector{ComplexF64}}()
    for leaf in leaves(stree)
        ctr = SVector{3,Float64}(center(container(leaf)))
        mp = zeros(ComplexF64, ncoeff)
        form_hmpole3d!(mp, ctr, zk, src, ch, dips, index_range(leaf), p)
        leaf_mp[objectid(leaf)] = mp
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
        pot_loc = zeros(ComplexF64, nt)

        function m2p!(tnode, snode, _, __)
            mp = leaf_mp[objectid(snode)]
            ctr = SVector{3,Float64}(center(container(snode)))
            buf = zeros(ComplexF64, length(index_range(tnode)))
            eval_hmpole3d!(buf, ctr, mp, zk, tpts, index_range(tnode), p)
            for (jt, j) in enumerate(index_range(tnode))
                pot_loc[j] += buf[jt]
            end
        end

        function p2p!(tnode, snode)
            for j in index_range(tnode)
                tj = tpts[j]
                s = zero(ComplexF64)
                for i in index_range(snode)
                    same && i == j && continue
                    rvec = tj - src[i]
                    r = norm(rvec)
                    r < 1e-30 && continue
                    eikr = exp(im * zk * r)
                    s += ch[i] * eikr * INV4PI_H / r
                    if dips !== nothing
                        fdot = dips[i][1] * rvec[1] + dips[i][2] * rvec[2] + dips[i][3] * rvec[3]
                        s += fdot / r * (im * zk - 1 / r) * eikr * INV4PI_H / r
                    end
                end
                pot_loc[j] += s
            end
        end

        sdata = allocate_expdata(stree, 1)
        dualtree_upward!(stree, sdata, (n, e) -> nothing, (a, b) -> nothing)
        dualtree_interact!(ttree, stree, sdata, sdata, adm; m2p!, p2p!, threaded=threaded)

        pot_g = zeros(ComplexF64, nt)
        unpermute!(pot_g, pot_loc, tl2g)
        return pot_g
    end

    pg > 0 && (vals.pot = run(sources, true))
    pgt > 0 && targets !== nothing && (vals.pottarg = run(targets, false))
    vals.ier = 0
    return vals
end

function h3ddir(
    zk::Number,
    sources::AbstractMatrix{<:Real},
    targets::AbstractMatrix{<:Real};
    charges=nothing,
    dipvecs=nothing,
    pgt::Integer=1,
    thresh::Float64=0.0,
)
    zk = complex(zk)
    ns = size(sources, 2)
    nt = size(targets, 2)
    pot = zeros(ComplexF64, nt)
    has_c = charges !== nothing
    has_d = dipvecs !== nothing
    @inbounds for j in 1:nt
        s = zero(ComplexF64)
        for i in 1:ns
            rx = targets[1, j] - sources[1, i]
            ry = targets[2, j] - sources[2, i]
            rz = targets[3, j] - sources[3, i]
            r = sqrt(rx * rx + ry * ry + rz * rz)
            r <= thresh && continue
            eikr = exp(im * zk * r)
            has_c && (s += complex(charges[i]) * eikr * INV4PI_H / r)
            if has_d
                fdot = complex(dipvecs[1, i]) * rx + complex(dipvecs[2, i]) * ry +
                       complex(dipvecs[3, i]) * rz
                s += fdot / r * (im * zk - 1 / r) * eikr * INV4PI_H / r
            end
        end
        pot[j] = s
    end
    vals = FMMVals()
    vals.pottarg = pot
    vals.ier = 0
    return vals
end
