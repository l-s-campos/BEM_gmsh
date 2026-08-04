# =============================================================================
# 2D Cauchy FMM  —  kernel 1/(z - z_j)
# pot(z) = Σ q_j / (z - z_j)   (+ dipole terms q'/(z-z_j)^2)
# =============================================================================

"""
    cfmm2d(eps, sources; charges=nothing, dipstr=nothing, targets=nothing, pg=0, pgt=0)

Cauchy kernel FMM: ``u(z)=\\sum_j c_j/(z-z_j) + d_j/(z-z_j)^2``.
"""
function cfmm2d(
    eps::Real,
    sources::AbstractMatrix{<:Real};
    charges=nothing,
    dipstr=nothing,
    targets=nothing,
    pg::Integer=0,
    pgt::Integer=0,
    nmax::Integer=50,
    η::Real=1.0,
)
    @assert size(sources, 1) == 2
    ns = size(sources, 2)
    has_c = charges !== nothing
    has_d = dipstr !== nothing
    @assert has_c || has_d
    @assert pg > 0 || pgt > 0

    nterms = laplace_nterms(Float64(eps))
    ncoeff = nterms + 1
    carray = binomial_table(2 * nterms + 2)
    adm = StrongAdmissibility(η=Float64(η))
    spl = GeometricSplitter(nmax=Int(nmax))

    stree, sl2g, src = build_point_tree(sources, spl)
    sdata = allocate_expdata(stree, ncoeff)
    ch = has_c ? permute_to_local_complex(vec(charges), sl2g) : nothing
    dp = has_d ? permute_to_local_complex(vec(dipstr), sl2g) : nothing

    function upward_leaf!(node, ed)
        for i in index_range(node)
            z0 = complex((src[i] - ed.center)...)
            # multipole: q * z0^n  for pot = q/(z-z0) = q/z * 1/(1-z0/z) = sum q z0^n / z^{n+1}
            # store a_n = q z0^n  (n=0..), eval sum a_n / z^{n+1}
            if has_c
                zn = one(ComplexF64)
                q = ch[i]
                for n in 0:nterms
                    ed.multipole[n + 1] += q * zn
                    zn *= z0
                end
            end
            if has_d
                # d/(z-z0)^2 = d * sum (n+1) z0^n / z^{n+2}
                # store in same basis carefully: contrib to a_n from dipole
                zn = one(ComplexF64)
                d = dp[i]
                for n in 0:nterms
                    # a_n gets d * (n) * z0^{n-1} for n>=1 from d/(z-z0)^2 expansion
                    nothing
                end
                # d/(z-z0)^2 = sum_{n=0} (n+1) d z0^n / z^{n+2}
                # Let b_m = coeff of 1/z^{m+1}, m=n+1 => b_{n+1} += (n+1) d z0^n
                zn = one(ComplexF64)
                for n in 0:(nterms - 1)
                    ed.multipole[n + 2] += (n + 1) * d * zn
                    zn *= z0
                end
            end
        end
    end

    function m2m!(parent_ed, child_ed)
        # shift multipoles a_n (about child) to parent: z = z' + δ, δ = child - parent
        δ = complex((child_ed.center - parent_ed.center)...)
        # 1/(z-c_c)^{n+1} = 1/(z'-c_p - (c_c-c_p))^{n+1}
        # Use binomial: (w-δ)^{-(n+1)} = sum_k C(n+k,k) δ^k / w^{n+k+1}
        a = child_ed.multipole
        for n in 0:nterms
            # a_n contributes to parent coeffs
            δk = one(ComplexF64)
            for k in 0:(nterms - n)
                # C(n+k, k) * a_n * δ^k  -> parent[n+k]
                bin = carray[n + k + 1, k + 1]  # C(n+k, k)
                parent_ed.multipole[n + k + 1] += a[n + 1] * bin * δk
                δk *= δ
            end
        end
    end

    function m2l!(tnode, snode, tdata, sdata)
        td = _edata(tdata, tnode)
        sd = _edata(sdata, snode)
        # Convert multipole about sc to local about tc
        # 1/(z-sc)^{n+1} = 1/(δ + (z-tc))^{n+1} with δ = tc-sc
        # = sum_k (-1)^k C(n+k,k) (z-tc)^k / δ^{n+k+1}
        δ = complex((td.center - sd.center)...)
        abs(δ) < 1e-30 && return
        a = sd.multipole
        for n in 0:nterms
            invδ = 1 / δ
            pwr = invδ^(n + 1)
            for k in 0:nterms
                # local_k += a_n * (-1)^k * C(n+k,k) / δ^{n+k+1}
                bin = carray[n + k + 1, k + 1]
                td.localexp[k + 1] += a[n + 1] * bin * pwr * ((-1)^k)
                pwr *= invδ
            end
        end
    end

    function eval_local_cauchy!(pot, node, ed, pts)
        for (jt, j) in enumerate(index_range(node))
            z = complex((pts[j] - ed.center)...)
            s = zero(ComplexF64)
            zn = one(ComplexF64)
            for n in 0:nterms
                s += ed.localexp[n + 1] * zn
                zn *= z
            end
            pot[jt] += s
        end
    end

    function p2p_cauchy!(pot, tnode, snode, tpts, spts, exclude)
        for (jt, j) in enumerate(index_range(tnode))
            zt = complex(tpts[j]...)
            s = zero(ComplexF64)
            for i in index_range(snode)
                exclude && i == j && continue
                zs = complex(spts[i]...)
                dz = zt - zs
                abs(dz) < 1e-30 && continue
                has_c && (s += ch[i] / dz)
                has_d && (s += dp[i] / (dz * dz))
            end
            pot[jt] += s
        end
    end

    vals = FMMVals()

    function run_on_targets(tpts_mat, pflag)
        nt = size(tpts_mat, 2)
        ttree, tl2g, tpts = if tpts_mat === sources && pflag # same tree
            (stree, sl2g, src)
        else
            build_point_tree(tpts_mat, spl)
        end
        same = ttree === stree
        tdata = same ? sdata : allocate_expdata(ttree, ncoeff)
        same || zero_expansions!(tdata)

        pot_loc = zeros(ComplexF64, nt)
        pot_leaf = Dict{UInt,Any}()
        for leaf in leaves(ttree)
            pot_leaf[objectid(leaf)] = view(pot_loc, index_range(leaf))
        end

        zero_expansions!(sdata)
        dualtree_upward!(stree, sdata, upward_leaf!, m2m!)
        zero_expansions!(tdata; multipole=false, localexp=true)

        dualtree_interact!(
            ttree, stree, tdata, sdata, adm;
            m2l! = m2l!,
            p2p! = (tn, sn) -> begin
                p = pot_leaf[objectid(tn)]
                p2p_cauchy!(p, tn, sn, tpts, src, same)
            end,
        )
        dualtree_downward!(
            ttree, tdata,
            (cd, pd) -> begin
                # L2L: local about parent → child
                δ = complex((cd.center - pd.center)...)
                for n in 0:nterms
                    δk = one(ComplexF64)
                    for k in 0:(nterms - n)
                        bin = carray[n + k + 1, k + 1]
                        # b_n (z-c_p)^n = b_n (z-c_c + δ)^n = sum C(n,k) b_n δ^{n-k} (z-c_c)^k
                        nothing
                    end
                end
                for n in 0:nterms
                    for k in 0:n
                        bin = carray[n + 1, k + 1]  # C(n,k)
                        cd.localexp[k + 1] += pd.localexp[n + 1] * bin * δ^(n - k)
                    end
                end
            end,
            (node, ed) -> begin
                p = pot_leaf[objectid(node)]
                eval_local_cauchy!(p, node, ed, tpts)
            end,
        )

        pot_g = zeros(ComplexF64, nt)
        unpermute!(pot_g, pot_loc, tl2g)
        return pot_g
    end

    if pg > 0
        vals.pot = run_on_targets(sources, true)
    end
    if pgt > 0 && targets !== nothing
        vals.pottarg = run_on_targets(targets, false)
    end
    vals.ier = 0
    return vals
end

function c2ddir(
    sources::AbstractMatrix{<:Real},
    targets::AbstractMatrix{<:Real};
    charges=nothing,
    dipstr=nothing,
    pgt::Integer=1,
)
    nt = size(targets, 2)
    ns = size(sources, 2)
    pot = zeros(ComplexF64, nt)
    has_c = charges !== nothing
    has_d = dipstr !== nothing
    @inbounds for j in 1:nt
        zt = complex(targets[1, j], targets[2, j])
        s = zero(ComplexF64)
        for i in 1:ns
            zs = complex(sources[1, i], sources[2, i])
            dz = zt - zs
            abs(dz) < 1e-30 && continue
            has_c && (s += complex(charges[i]) / dz)
            has_d && (s += complex(dipstr[i]) / (dz * dz))
        end
        pot[j] = s
    end
    vals = FMMVals()
    vals.pottarg = pot
    vals.ier = 0
    return vals
end
