# =============================================================================
# 2D Modified Helmholtz (Yukawa) FMM — kernel K_0(κ r)/(2π)
#
# Multipole: M_n += q I_n(κ r) e^{-inθ}
# Eval:      pot += (1/2π) sum M_n K_n(κ R) e^{inθ}
# =============================================================================

using SpecialFunctions: besseli, besselk

const INV2PI = 1 / (2π)

function yukawa2d_nterms(eps::Real, κ::Real, boxscale::Real)
    p0 = laplace_nterms(Float64(eps))
    extra = ceil(Int, abs(κ) * boxscale)
    return min(max(p0 + extra + 2, 6), 40)
end

"""
```julia
vals = yfmm2d(eps, κ, sources; charges, targets=nothing, pg=0, pgt=0, nmax=40, η=1.2)
```

2D modified Helmholtz FMM for ``K_0(κ r)/(2π)``.
"""
function yfmm2d(
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
    @assert κ > 0
    @assert size(sources, 1) == 2
    ns = size(sources, 2)
    @assert pg > 0 || pgt > 0

    spl = GeometricSplitter(nmax=Int(nmax))
    stree, sl2g, src = build_point_tree(sources, spl)
    L = diameter(container(stree))
    nterms = yukawa2d_nterms(eps, κ, L / 4)
    ncoeff = 2 * nterms + 1
    idx0 = nterms + 1
    n_index(n) = idx0 + n
    adm = StrongAdmissibility(η=Float64(η))
    ch = permute_to_local(vec(charges), sl2g)

    leaf_mp = Dict{Int,Vector{ComplexF64}}()
    for leaf in leaves(stree)
        ctr = SVector{2,Float64}(center(container(leaf)))
        mp = zeros(ComplexF64, ncoeff)
        for i in index_range(leaf)
            dxy = src[i] - ctr
            r = hypot(dxy[1], dxy[2])
            θ = atan(dxy[2], dxy[1])
            q = ch[i]
            # M_0 += q I_0
            mp[n_index(0)] += q * (r < 1e-14 ? 1.0 : besseli(0, κ * r))
            if r >= 1e-14
                e = exp(-im * θ)
                ep, em = e, conj(e)
                for n in 1:nterms
                    In = besseli(n, κ * r)
                    mp[n_index(n)] += q * In * ep
                    mp[n_index(-n)] += q * In * em
                    ep *= e
                    em *= conj(e)
                end
            end
        end
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
            ctr = SVector{2,Float64}(center(container(snode)))
            for j in index_range(tnode)
                dxy = tpts[j] - ctr
                r = hypot(dxy[1], dxy[2])
                r < 1e-30 && continue
                θ = atan(dxy[2], dxy[1])
                s = mp[n_index(0)] * besselk(0, κ * r)
                e = exp(im * θ)
                ep, em = e, conj(e)
                for n in 1:nterms
                    Kn = besselk(n, κ * r)
                    s += mp[n_index(n)] * Kn * ep
                    s += mp[n_index(-n)] * Kn * em
                    ep *= e
                    em *= conj(e)
                end
                pot_loc[j] += real(s) * INV2PI
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
                    s += ch[i] * besselk(0, κ * r) * INV2PI
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

function y2ddir(
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
            r = hypot(targets[1, j] - sources[1, i], targets[2, j] - sources[2, i])
            r <= thresh && continue
            s += charges[i] * besselk(0, κ * r) * INV2PI
        end
        pot[j] = s
    end
    vals = FMMVals()
    vals.pottarg = pot
    vals.ier = 0
    return vals
end
