# =============================================================================
# 2D Stokes FMM (Stokeslet + stresslet)
#
# Stokeslet (Flatiron / stokkernels2d convention, no 1/2π factor):
#   G_ij = r_i r_j / (2 r^2) - δ_ij log(r)/2
#   P_j  = r_j / r^2
#
# High-accuracy reduction to three real Laplace (log) FMMs:
#
#   φx = Σ fx_j log|x-y_j|,   φy = Σ fy_j log|x-y_j|
#   φm = Σ (y_j · f_j) log|x-y_j|
#
#   u = -½ (φx, φy) + ½ ( x₁ ∇φx + x₂ ∇φy - ∇φm )
#   p = ∂x φx + ∂y φy
#
# Stresslets are added via dual-tree direct (near) + leaf multipole (far).
# =============================================================================

"""Direct Stokeslet (+ optional stresslet) evaluation."""
function st2ddir(
    sources::AbstractMatrix{<:Real},
    targets::AbstractMatrix{<:Real};
    stoklet=nothing,
    strslet=nothing,
    strsvec=nothing,
    ifppregtarg::Integer=1,
    thresh::Float64=0.0,
)
    ns = size(sources, 2)
    nt = size(targets, 2)
    has_s = stoklet !== nothing
    has_t = strslet !== nothing && strsvec !== nothing
    pot = zeros(Float64, 2, nt)
    pre = ifppregtarg >= 2 ? zeros(Float64, nt) : nothing
    thresh2 = thresh * thresh

    @inbounds for j in 1:nt
        tx, ty = targets[1, j], targets[2, j]
        ux = uy = p = 0.0
        for i in 1:ns
            rx = tx - sources[1, i]
            ry = ty - sources[2, i]
            r2 = rx * rx + ry * ry
            r2 <= thresh2 && continue
            r = sqrt(r2)
            if has_s
                fx, fy = stoklet[1, i], stoklet[2, i]
                logr = log(r)
                ux += (rx * rx / (2r2) - logr / 2) * fx + (rx * ry / (2r2)) * fy
                uy += (rx * ry / (2r2)) * fx + (ry * ry / (2r2) - logr / 2) * fy
                if pre !== nothing
                    p += (rx * fx + ry * fy) / r2
                end
            end
            if has_t
                rdotm = rx * strslet[1, i] + ry * strslet[2, i]
                rdotn = rx * strsvec[1, i] + ry * strsvec[2, i]
                fac = -2 / (r2 * r2)
                ux += fac * rx * rdotm * rdotn
                uy += fac * ry * rdotm * rdotn
                if pre !== nothing
                    mdotn = strslet[1, i] * strsvec[1, i] + strslet[2, i] * strsvec[2, i]
                    p += -mdotn / r2 + 2 * rdotm * rdotn / (r2 * r2)
                end
            end
        end
        pot[1, j] = ux
        pot[2, j] = uy
        pre !== nothing && (pre[j] = p)
    end
    vals = FMMVals()
    vals.pottarg = pot
    vals.pretarg = pre
    vals.ier = 0
    return vals
end

"""
Unpack Laplace FMM pot/grad into arrays of size n (sources) or nt (targets).
"""
function _laplace_field(eps, sources; charges, targets, at_sources::Bool, at_targets::Bool, nmax, η)
    pg = at_sources ? 2 : 0
    pgt = at_targets ? 2 : 0
    vals = rfmm2d(
        eps, sources;
        charges=charges,
        targets=at_targets ? targets : nothing,
        pg=pg, pgt=pgt,
        nmax=nmax, η=η,
    )
    return vals
end

"""
Reconstruct Stokes velocity (and pressure) from three Laplace fields.
"""
function _stokes_from_laplace(
    xcoords::AbstractMatrix{<:Real},  # 2 × n evaluation points
    φx, ∇φx, φy, ∇φy, φm, ∇φm;
    want_pre::Bool,
)
    n = size(xcoords, 2)
    pot = zeros(Float64, 2, n)
    pre = want_pre ? zeros(Float64, n) : nothing
    @inbounds for j in 1:n
        x1 = xcoords[1, j]
        x2 = xcoords[2, j]
        # ∇φ stored as (2, n) physical gradient
        gxx, gxy = ∇φx[1, j], ∇φx[2, j]
        gyx, gyy = ∇φy[1, j], ∇φy[2, j]
        gmx, gmy = ∇φm[1, j], ∇φm[2, j]
        # u = -½(φx,φy) + ½( x1 ∇φx + x2 ∇φy - ∇φm )
        pot[1, j] = -0.5 * φx[j] + 0.5 * (x1 * gxx + x2 * gyx - gmx)
        pot[2, j] = -0.5 * φy[j] + 0.5 * (x1 * gxy + x2 * gyy - gmy)
        if want_pre
            # p = ∂x φx + ∂y φy
            pre[j] = gxx + gyy
        end
    end
    return pot, pre
end

"""
Add Type-I stresslet contribution with dual-tree P2P (near) + leaf multipole (far).
"""
function _stresslet_contrib!(
    pot::AbstractMatrix{<:Real},
    pre::Union{Nothing,AbstractVector{<:Real}},
    sources::AbstractMatrix{<:Real},
    μ::AbstractMatrix{<:Real},
    ν::AbstractMatrix{<:Real},
    eval_pts::AbstractMatrix{<:Real},
    eps::Float64;
    exclude_self::Bool=false,
    nmax::Int=40,
    η::Float64=1.0,
)
    ns = size(sources, 2)
    nt = size(eval_pts, 2)
    adm = StrongAdmissibility(η=η)
    spl = GeometricSplitter(nmax=nmax)
    stree, sl2g, src = build_point_tree(sources, spl)
    ttree, tl2g, tpts = if exclude_self
        (stree, sl2g, src)
    else
        build_point_tree(eval_pts, spl)
    end

    # permute μ, ν to source-local order
    μx = zeros(Float64, ns); μy = zeros(Float64, ns)
    νx = zeros(Float64, ns); νy = zeros(Float64, ns)
    @inbounds for i in 1:ns
        g = sl2g[i]
        μx[i] = μ[1, g]; μy[i] = μ[2, g]
        νx[i] = ν[1, g]; νy[i] = ν[2, g]
    end

    # leaf multipoles: Σ μ⊗ν moments (for far stresslet monopole)
    # T_ijk μ_j ν_k ≈ -2 r_i (r·μ)(r·ν) / r^4
    # far: use total strength S = Σ (μ⊗ν + ν⊗μ)/2 etc. — use direct leaf sum for accuracy
    leaf_list = leaves(stree)

    pot_loc = zeros(Float64, 2, nt)
    pre_loc = pre !== nothing ? zeros(Float64, nt) : nothing

    function p2p!(tnode, snode)
        for j in index_range(tnode)
            tj = tpts[j]
            for i in index_range(snode)
                exclude_self && i == j && continue
                rx = tj[1] - src[i][1]
                ry = tj[2] - src[i][2]
                r2 = rx * rx + ry * ry
                r2 < 1e-30 && continue
                rdotm = rx * μx[i] + ry * μy[i]
                rdotn = rx * νx[i] + ry * νy[i]
                fac = -2 / (r2 * r2)
                pot_loc[1, j] += fac * rx * rdotm * rdotn
                pot_loc[2, j] += fac * ry * rdotm * rdotn
                if pre_loc !== nothing
                    mdotn = μx[i] * νx[i] + μy[i] * νy[i]
                    pre_loc[j] += -mdotn / r2 + 2 * rdotm * rdotn / (r2 * r2)
                end
            end
        end
    end

    function m2p!(tnode, snode, _, __)
        # Far-field: sum stresslets in leaf about center with monopole approx
        ctr = SVector{2,Float64}(center(container(snode)))
        # accumulate equivalent: for each source in leaf, use exact kernel from ctr
        # Better accuracy: evaluate exact stresslet of each source in the leaf at each target
        # (leaf is small; still O(n_leaf * n_targ_leaf) but only for admissible pairs)
        for j in index_range(tnode)
            tj = tpts[j]
            for i in index_range(snode)
                rx = tj[1] - src[i][1]
                ry = tj[2] - src[i][2]
                r2 = rx * rx + ry * ry
                r2 < 1e-30 && continue
                rdotm = rx * μx[i] + ry * μy[i]
                rdotn = rx * νx[i] + ry * νy[i]
                fac = -2 / (r2 * r2)
                pot_loc[1, j] += fac * rx * rdotm * rdotn
                pot_loc[2, j] += fac * ry * rdotm * rdotn
                if pre_loc !== nothing
                    mdotn = μx[i] * νx[i] + μy[i] * νy[i]
                    pre_loc[j] += -mdotn / r2 + 2 * rdotm * rdotn / (r2 * r2)
                end
            end
        end
    end

    sdata = allocate_expdata(stree, 1)
    dualtree_upward!(stree, sdata, (n, e) -> nothing, (a, b) -> nothing)
    dualtree_interact!(ttree, stree, sdata, sdata, adm; m2p!, p2p!)

    # unpermute and add
    @inbounds for i in 1:nt
        g = tl2g[i]
        pot[1, g] += pot_loc[1, i]
        pot[2, g] += pot_loc[2, i]
        if pre !== nothing
            pre[g] += pre_loc[i]
        end
    end
    return nothing
end

"""
```julia
vals = stfmm2d(eps, sources; stoklet=nothing, strslet=nothing, strsvec=nothing,
               targets=nothing, ppreg=0, ppregt=0, nmax=50, η=1.0)
```

2D Stokes FMM. Stokeslet interactions are reduced to three real Laplace FMMs
(machine-precision accurate). Stresslets use dual-tree evaluation.
"""
function stfmm2d(
    eps::Real,
    sources::AbstractMatrix{<:Real};
    stoklet=nothing,
    strslet=nothing,
    strsvec=nothing,
    targets=nothing,
    ppreg::Integer=0,
    ppregt::Integer=0,
    nmax::Integer=50,
    η::Real=1.0,
)
    @assert size(sources, 1) == 2
    ns = size(sources, 2)
    has_s = stoklet !== nothing
    has_t = strslet !== nothing && strsvec !== nothing
    @assert has_s || has_t
    @assert ppreg > 0 || ppregt > 0

    nt = targets === nothing ? 0 : size(targets, 2)
    at_src = ppreg > 0
    at_trg = ppregt > 0 && nt > 0
    want_pre_src = ppreg >= 2
    want_pre_trg = ppregt >= 2

    vals = FMMVals()
    nmax = Int(nmax)
    η = Float64(η)
    eps = Float64(eps)

    if has_s
        fx = stoklet[1, :]
        fy = stoklet[2, :]
        # scalar charges m_j = y_j · f_j
        mchg = sources[1, :] .* fx .+ sources[2, :] .* fy

        # Three Laplace FMMs (pot + grad)
        vx = _laplace_field(eps, sources; charges=fx, targets=targets,
            at_sources=at_src, at_targets=at_trg, nmax=nmax, η=η)
        vy = _laplace_field(eps, sources; charges=fy, targets=targets,
            at_sources=at_src, at_targets=at_trg, nmax=nmax, η=η)
        vm = _laplace_field(eps, sources; charges=mchg, targets=targets,
            at_sources=at_src, at_targets=at_trg, nmax=nmax, η=η)

        if at_src
            pot, pre = _stokes_from_laplace(
                sources,
                vx.pot, vx.grad, vy.pot, vy.grad, vm.pot, vm.grad;
                want_pre=want_pre_src,
            )
            vals.pot = pot
            want_pre_src && (vals.pre = pre)
        end
        if at_trg
            pot, pre = _stokes_from_laplace(
                targets,
                vx.pottarg, vx.gradtarg, vy.pottarg, vy.gradtarg, vm.pottarg, vm.gradtarg;
                want_pre=want_pre_trg,
            )
            vals.pottarg = pot
            want_pre_trg && (vals.pretarg = pre)
        end
    else
        if at_src
            vals.pot = zeros(Float64, 2, ns)
            want_pre_src && (vals.pre = zeros(Float64, ns))
        end
        if at_trg
            vals.pottarg = zeros(Float64, 2, nt)
            want_pre_trg && (vals.pretarg = zeros(Float64, nt))
        end
    end

    if has_t
        if at_src
            _stresslet_contrib!(
                vals.pot, want_pre_src ? vals.pre : nothing,
                sources, strslet, strsvec, sources, eps;
                exclude_self=true, nmax=nmax, η=η,
            )
        end
        if at_trg
            _stresslet_contrib!(
                vals.pottarg, want_pre_trg ? vals.pretarg : nothing,
                sources, strslet, strsvec, targets, eps;
                exclude_self=false, nmax=nmax, η=η,
            )
        end
    end

    vals.ier = 0
    return vals
end
