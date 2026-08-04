# =============================================================================
# 2D Helmholtz FMM
# kernel:  (i/4) H_0^{(1)}(k|x-y|)
# multipole:  sum_{n=-p}^{p} M_n H_n(kr) e^{inθ}
# form: M_n += q J_n(kr_s) e^{-inθ_s}
# =============================================================================

using SpecialFunctions: besselj, besselh

"""Scaled Bessel J_n(z) with multipole scaling rscale."""
function _jbessel_scaled(nterms::Int, z::ComplexF64, rscale::Float64)
    # jval[n+1] ≈ J_n(z) / rscale^n   (n≥0); negative via J_{-n}=(-1)^n J_n for integer
    jval = Vector{ComplexF64}(undef, nterms + 1)
    if abs(z) < 1e-14
        jval[1] = 1
        for n in 1:nterms
            jval[n + 1] = 0
        end
        return jval
    end
    # unscaled J then scale
    sc = one(ComplexF64)
    for n in 0:nterms
        jval[n + 1] = besselj(n, z) / sc
        sc *= rscale
    end
    return jval
end

function _hankel_scaled(nterms::Int, z::ComplexF64, rscale::Float64)
    # hval[n+1] = H_n(z) * rscale^n
    hval = Vector{ComplexF64}(undef, nterms + 1)
    sc = one(ComplexF64)
    for n in 0:nterms
        hval[n + 1] = besselh(n, 1, z) * sc
        sc *= rscale
    end
    return hval
end

"""
Number of Helmholtz multipole terms for precision `eps`.
"""
function helmholtz_nterms(eps::Real, zk::Number, boxsize::Real)
    # rough estimate: similar to Laplace plus k*R factor
    p = laplace_nterms(Float64(eps))
    extra = ceil(Int, abs(complex(zk)) * boxsize)
    return min(p + extra + 5, 80)
end

"""
```julia
vals = hfmm2d(eps, zk, sources; charges=nothing, dipstr=nothing, dipvec=nothing,
              targets=nothing, pg=0, pgt=0, nmax=40, η=1.2)
```

2D Helmholtz FMM for kernel ``(i/4) H_0^{(1)}(k r)``.
"""
function hfmm2d(
    eps::Real,
    zk::Number,
    sources::AbstractMatrix{<:Real};
    charges=nothing,
    dipstr=nothing,
    dipvec=nothing,
    targets=nothing,
    pg::Integer=0,
    pgt::Integer=0,
    nmax::Integer=40,
    η::Real=1.2,
)
    zk = complex(Float64(real(zk)), Float64(imag(complex(zk))))
    @assert size(sources, 1) == 2
    ns = size(sources, 2)
    has_c = charges !== nothing
    has_d = dipstr !== nothing
    @assert has_c || has_d
    @assert pg > 0 || pgt > 0

    spl = GeometricSplitter(nmax=Int(nmax))
    stree, sl2g, src = build_point_tree(sources, spl)
    L = diameter(container(stree))
    nterms = helmholtz_nterms(eps, zk, L / 4)
    # multipole layout: index 1 = n=0, then pairs (n,-n) packed as
    # coeffs[1 + 2(n-1) + 1] = +n, coeffs[1 + 2(n-1) + 2] = -n for n>=1
    # simpler: Vector length 2p+1 with offset p+1 for n=0
    ncoeff = 2 * nterms + 1
    idx0 = nterms + 1  # location of n=0
    n_index(n) = idx0 + n   # n in -p:p

    sdata = allocate_expdata(stree, ncoeff)
    ch = has_c ? permute_to_local_complex(vec(charges), sl2g) : nothing
    # dipoles: convert to complex strength * direction
    dp = nothing
    if has_d
        @assert dipvec !== nothing
        dp = Vector{ComplexF64}(undef, ns)
        dstr = vec(dipstr)
        @inbounds for i in 1:ns
            g = sl2g[i]
            dp[i] = complex(dstr[g]) * complex(dipvec[1, g], dipvec[2, g])
        end
    end

    ima = im
    ima4 = ima / 4

    function form_mp_leaf!(node, ed)
        for i in index_range(node)
            dxy = src[i] - ed.center
            r = hypot(dxy[1], dxy[2])
            θ = atan(dxy[2], dxy[1])
            z = zk * r
            jval = _jbessel_scaled(nterms, z, ed.rscale)
            e = exp(-ima * θ)
            ecur = one(ComplexF64)
            if has_c
                q = ch[i]
                ed.multipole[n_index(0)] += q * jval[1]
                ep = e
                em = conj(e)
                for n in 1:nterms
                    ed.multipole[n_index(n)] += q * jval[n + 1] * ep
                    ed.multipole[n_index(-n)] += q * jval[n + 1] * em
                    ep *= e
                    em *= conj(e)
                end
            end
            if has_d
                # dipole approx: d · ∇_src of charge expansion ≈ ik * (direction factor)
                # use finite formula: charge multipole of strength 0 with derivative
                # Simplified: treat as pair of charges (central difference) — skip for accuracy
                # Use: M_n += (d_complex) * (J' terms) — approximate with ik * d_r * J
                d = dp[i]
                # radial dipole contribution ~ d_r * ∂r (J_n e^{-inθ})
                # fallback: small shift
                δ = 1e-8
                for sgn in (1.0, -1.0)
                    # not ideal; use analytical: multipole of dipole = -d·∇_center of charge mpole
                    nothing
                end
                # Analytical for complex dipole d_z = dx+i dy acting as ∂/∂conj-like:
                # For Helmholtz, dipole orientation v with strength s:
                # pot ~ s v·∇_y H0 = -s k (v·rhat) H1
                # Multipole: M_n += s * (something)
                # Use finite difference of charge form:
                v = d
                for (dx, dy, w) in (
                    (δ, 0.0, real(v) / (2δ)),
                    (-δ, 0.0, -real(v) / (2δ)),
                    (0.0, δ, imag(v) / (2δ)),
                    (0.0, -δ, -imag(v) / (2δ)),
                )
                    dxy2 = dxy + SVector(dx, dy)
                    r2 = hypot(dxy2[1], dxy2[2])
                    θ2 = atan(dxy2[2], dxy2[1])
                    j2 = _jbessel_scaled(nterms, zk * r2, ed.rscale)
                    e2 = exp(-ima * θ2)
                    ed.multipole[n_index(0)] += w * j2[1]
                    ep = e2
                    em = conj(e2)
                    for n in 1:nterms
                        ed.multipole[n_index(n)] += w * j2[n + 1] * ep
                        ed.multipole[n_index(-n)] += w * j2[n + 1] * em
                        ep *= e2
                        em *= conj(e2)
                    end
                end
            end
        end
    end

    # Leaf-only multipoles (no M2M) — dual-tree M2P expands sources to leaves
    m2m_helm! = (parent_ed, child_ed) -> nothing

    function eval_mp_at_targets!(pot, tnode, snode, tdata, sdata, tpts)
        sd = _edata(sdata, snode)
        for (jt, j) in enumerate(index_range(tnode))
            dxy = tpts[j] - sd.center
            r = hypot(dxy[1], dxy[2])
            r < 1e-30 && continue
            θ = atan(dxy[2], dxy[1])
            z = zk * r
            hval = _hankel_scaled(nterms, z, sd.rscale)
            s = sd.multipole[n_index(0)] * hval[1]
            ep = exp(ima * θ)
            em = conj(ep)
            ecp, ecm = ep, em
            for n in 1:nterms
                s += sd.multipole[n_index(n)] * hval[n + 1] * ecp
                s += sd.multipole[n_index(-n)] * hval[n + 1] * ecm
                ecp *= ep
                ecm *= em
            end
            pot[jt] += ima4 * s
        end
    end

    function p2p_helm!(pot, tnode, snode, tpts, exclude)
        for (jt, j) in enumerate(index_range(tnode))
            tj = tpts[j]
            s = zero(ComplexF64)
            for i in index_range(snode)
                exclude && i == j && continue
                dxy = tj - src[i]
                r = hypot(dxy[1], dxy[2])
                r < 1e-30 && continue
                z = zk * r
                if has_c
                    s += ch[i] * besselh(0, 1, z)
                end
                if has_d
                    # s v·∇_y H0 = -s k (v·rhat) H1
                    d = dp[i]
                    rhat_dot = (real(d) * dxy[1] + imag(d) * dxy[2]) / r
                    s += -zk * rhat_dot * besselh(1, 1, z)
                end
            end
            pot[jt] += ima4 * s
        end
    end

    adm = StrongAdmissibility(η=Float64(η))
    vals = FMMVals()

    function run(tpts_mat, do_src)
        nt = size(tpts_mat, 2)
        if do_src
            ttree, tl2g, tpts = stree, sl2g, src
            same = true
            tdata = sdata
        else
            ttree, tl2g, tpts = build_point_tree(tpts_mat, spl)
            same = false
            tdata = allocate_expdata(ttree, ncoeff)
        end
        pot_loc = zeros(ComplexF64, nt)
        pot_leaf = Dict{UInt,Any}()
        for leaf in leaves(ttree)
            pot_leaf[objectid(leaf)] = view(pot_loc, index_range(leaf))
        end
        zero_expansions!(sdata)
        dualtree_upward!(stree, sdata, form_mp_leaf!, m2m_helm!)

        dualtree_interact!(
            ttree, stree, tdata, sdata, adm;
            m2p! = (tn, sn, td, sd) -> begin
                p = pot_leaf[objectid(tn)]
                eval_mp_at_targets!(p, tn, sn, td, sd, tpts)
            end,
            p2p! = (tn, sn) -> begin
                p = pot_leaf[objectid(tn)]
                p2p_helm!(p, tn, sn, tpts, same)
            end,
        )
        pot_g = zeros(ComplexF64, nt)
        unpermute!(pot_g, pot_loc, tl2g)
        return pot_g
    end

    pg > 0 && (vals.pot = run(sources, true))
    pgt > 0 && targets !== nothing && (vals.pottarg = run(targets, false))
    vals.ier = 0
    return vals
end

function h2ddir(
    zk::Number,
    sources::AbstractMatrix{<:Real},
    targets::AbstractMatrix{<:Real};
    charges=nothing,
    dipstr=nothing,
    dipvec=nothing,
    pgt::Integer=1,
    thresh::Float64=0.0,
)
    zk = complex(zk)
    nt = size(targets, 2)
    ns = size(sources, 2)
    pot = zeros(ComplexF64, nt)
    ima4 = im / 4
    has_c = charges !== nothing
    has_d = dipstr !== nothing
    @inbounds for j in 1:nt
        s = zero(ComplexF64)
        for i in 1:ns
            dx = targets[1, j] - sources[1, i]
            dy = targets[2, j] - sources[2, i]
            r = hypot(dx, dy)
            r <= thresh && continue
            z = zk * r
            if has_c
                s += complex(charges[i]) * besselh(0, 1, z)
            end
            if has_d
                d = complex(dipstr[i]) * complex(dipvec[1, i], dipvec[2, i])
                rhat = (real(d) * dx + imag(d) * dy) / r
                s += -zk * rhat * besselh(1, 1, z)
            end
        end
        pot[j] = ima4 * s
    end
    vals = FMMVals()
    vals.pottarg = pot
    vals.ier = 0
    return vals
end
