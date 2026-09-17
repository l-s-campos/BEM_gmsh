# =============================================================================
# 3D Stokes FMM — Stokeslet
#   G_ij = (δ_ij/r + r_i r_j/r^3) / (8π)
#   P_j  = r_j / (4π r^3)
#
# High-accuracy reduction to four 3D Laplace FMMs (kernel 1/(4πr)):
#
#   φα = Σ f_α,j / (4π |x-y_j|),   φm = Σ (y_j·f_j)/(4π |x-y_j|)
#
#   u = ½ φ − ½ ( x₁∇φx + x₂∇φy + x₃∇φz − ∇φm )
#   p = −(∂x φx + ∂y φy + ∂z φz)
# =============================================================================

const INV8PI = 1 / (8π)

function st3ddir(
    sources::AbstractMatrix{<:Real},
    targets::AbstractMatrix{<:Real};
    stoklet=nothing,
    strslet=nothing,
    strsvec=nothing,
    ppregt::Integer=1,
    thresh::Float64=0.0,
)
    ns = size(sources, 2)
    nt = size(targets, 2)
    pot = zeros(Float64, 3, nt)
    pre = ppregt >= 2 ? zeros(Float64, nt) : nothing
    has_s = stoklet !== nothing
    has_t = strslet !== nothing && strsvec !== nothing
    @inbounds for j in 1:nt
        ux = uy = uz = p = 0.0
        for i in 1:ns
            rx = targets[1, j] - sources[1, i]
            ry = targets[2, j] - sources[2, i]
            rz = targets[3, j] - sources[3, i]
            r2 = rx * rx + ry * ry + rz * rz
            r = sqrt(r2)
            r <= thresh && continue
            if has_s
                fx, fy, fz = stoklet[1, i], stoklet[2, i], stoklet[3, i]
                fdotr = rx * fx + ry * fy + rz * fz
                invr = 1 / r
                invr3 = invr / r2
                ux += (fx * invr + rx * fdotr * invr3) * INV8PI
                uy += (fy * invr + ry * fdotr * invr3) * INV8PI
                uz += (fz * invr + rz * fdotr * invr3) * INV8PI
                pre !== nothing && (p += fdotr / (4π * r2 * r))
            end
            if has_t
                μx, μy, μz = strslet[1, i], strslet[2, i], strslet[3, i]
                νx, νy, νz = strsvec[1, i], strsvec[2, i], strsvec[3, i]
                rdotm = rx * μx + ry * μy + rz * μz
                rdotn = rx * νx + ry * νy + rz * νz
                fac = -3 * rdotm * rdotn / (4π * r2 * r2 * r)
                ux += fac * rx; uy += fac * ry; uz += fac * rz
            end
        end
        pot[1, j] = ux; pot[2, j] = uy; pot[3, j] = uz
        pre !== nothing && (pre[j] = p)
    end
    vals = FMMVals()
    vals.pottarg = pot
    vals.pretarg = pre
    vals.ier = 0
    return vals
end

function _stokes3d_from_laplace(
    xcoords::AbstractMatrix{<:Real},
    φx, ∇φx, φy, ∇φy, φz, ∇φz, φm, ∇φm;
    want_pre::Bool,
)
    n = size(xcoords, 2)
    pot = zeros(Float64, 3, n)
    pre = want_pre ? zeros(Float64, n) : nothing
    @inbounds for j in 1:n
        x1, x2, x3 = xcoords[1, j], xcoords[2, j], xcoords[3, j]
        # u = ½ φ − ½ (x1∇φx + x2∇φy + x3∇φz − ∇φm)
        gx = x1 * ∇φx[1, j] + x2 * ∇φy[1, j] + x3 * ∇φz[1, j] - ∇φm[1, j]
        gy = x1 * ∇φx[2, j] + x2 * ∇φy[2, j] + x3 * ∇φz[2, j] - ∇φm[2, j]
        gz = x1 * ∇φx[3, j] + x2 * ∇φy[3, j] + x3 * ∇φz[3, j] - ∇φm[3, j]
        pot[1, j] = 0.5 * φx[j] - 0.5 * gx
        pot[2, j] = 0.5 * φy[j] - 0.5 * gy
        pot[3, j] = 0.5 * φz[j] - 0.5 * gz
        if want_pre
            # p = −(∂x φx + ∂y φy + ∂z φz)
            pre[j] = -(∇φx[1, j] + ∇φy[2, j] + ∇φz[3, j])
        end
    end
    return pot, pre
end

"""
```julia
vals = stfmm3d(eps, sources; stoklet=nothing, strslet=nothing, strsvec=nothing,
               targets=nothing, ppreg=0, ppregt=0, nmax=40, η=1.0)
```

3D Stokes FMM. Stokeslets are reduced to four Laplace FMMs (high accuracy).
Stresslets are added by dual-tree direct evaluation on leaves.
"""
function stfmm3d(
    eps::Real,
    sources::AbstractMatrix{<:Real};
    stoklet=nothing,
    strslet=nothing,
    strsvec=nothing,
    targets=nothing,
    ppreg::Integer=0,
    ppregt::Integer=0,
    nmax::Integer=40,
    η::Real=1.0,
)
    @assert size(sources, 1) == 3
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
    eps = Float64(eps)
    nmax = Int(nmax)
    η = Float64(η)

    vals = FMMVals()

    if has_s
        fx = stoklet[1, :]; fy = stoklet[2, :]; fz = stoklet[3, :]
        mchg = sources[1, :] .* fx .+ sources[2, :] .* fy .+ sources[3, :] .* fz

        pg = at_src ? 2 : 0
        pgt = at_trg ? 2 : 0
        vx = lfmm3d(eps, sources; charges=fx, targets=targets, pg=pg, pgt=pgt, nmax=nmax, η=η)
        vy = lfmm3d(eps, sources; charges=fy, targets=targets, pg=pg, pgt=pgt, nmax=nmax, η=η)
        vz = lfmm3d(eps, sources; charges=fz, targets=targets, pg=pg, pgt=pgt, nmax=nmax, η=η)
        vm = lfmm3d(eps, sources; charges=mchg, targets=targets, pg=pg, pgt=pgt, nmax=nmax, η=η)

        if at_src
            pot, pre = _stokes3d_from_laplace(
                sources, vx.pot, vx.grad, vy.pot, vy.grad, vz.pot, vz.grad, vm.pot, vm.grad;
                want_pre=want_pre_src,
            )
            vals.pot = pot
            want_pre_src && (vals.pre = pre)
        end
        if at_trg
            pot, pre = _stokes3d_from_laplace(
                targets,
                vx.pottarg, vx.gradtarg, vy.pottarg, vy.gradtarg,
                vz.pottarg, vz.gradtarg, vm.pottarg, vm.gradtarg;
                want_pre=want_pre_trg,
            )
            vals.pottarg = pot
            want_pre_trg && (vals.pretarg = pre)
        end
    else
        at_src && (vals.pot = zeros(Float64, 3, ns); want_pre_src && (vals.pre = zeros(Float64, ns)))
        at_trg && (vals.pottarg = zeros(Float64, 3, nt); want_pre_trg && (vals.pretarg = zeros(Float64, nt)))
    end

    if has_t
        jobs = Tuple{Matrix{Float64},Bool}[]
        at_src && push!(jobs, (Matrix{Float64}(sources), true))
        at_trg && push!(jobs, (Matrix{Float64}(targets), false))
        for (eval_pts, is_src) in jobs
            pot = is_src ? vals.pot : vals.pottarg
            nt_e = size(eval_pts, 2)
            adm = StrongAdmissibility(η=η)
            spl = GeometricSplitter(nmax=nmax)
            stree, sl2g, src = build_point_tree(sources, spl)
            ttree, tl2g, tpts = is_src ? (stree, sl2g, src) : build_point_tree(eval_pts, spl)
            μx = zeros(ns); μy = zeros(ns); μz = zeros(ns)
            νx = zeros(ns); νy = zeros(ns); νz = zeros(ns)
            @inbounds for i in 1:ns
                g = sl2g[i]
                μx[i] = strslet[1, g]; μy[i] = strslet[2, g]; μz[i] = strslet[3, g]
                νx[i] = strsvec[1, g]; νy[i] = strsvec[2, g]; νz[i] = strsvec[3, g]
            end
            pot_loc = zeros(Float64, 3, nt_e)
            function add_pair!(j, i)
                rvec = tpts[j] - src[i]
                r = norm(rvec)
                r < 1e-30 && return
                rdotm = rvec[1] * μx[i] + rvec[2] * μy[i] + rvec[3] * μz[i]
                rdotn = rvec[1] * νx[i] + rvec[2] * νy[i] + rvec[3] * νz[i]
                fac = -3 * rdotm * rdotn / (4π * r^5)
                pot_loc[1, j] += fac * rvec[1]
                pot_loc[2, j] += fac * rvec[2]
                pot_loc[3, j] += fac * rvec[3]
            end
            function p2p!(tn, sn)
                for j in index_range(tn), i in index_range(sn)
                    is_src && i == j && continue
                    add_pair!(j, i)
                end
            end
            function m2p!(tn, sn, _, __)
                for j in index_range(tn), i in index_range(sn)
                    add_pair!(j, i)
                end
            end
            sdata = allocate_expdata(stree, 1)
            dualtree_upward!(stree, sdata, (n, e) -> nothing, (a, b) -> nothing)
            dualtree_interact!(ttree, stree, sdata, sdata, adm; m2p!, p2p!)
            @inbounds for i in 1:nt_e
                g = tl2g[i]
                pot[1, g] += pot_loc[1, i]
                pot[2, g] += pot_loc[2, i]
                pot[3, g] += pot_loc[3, i]
            end
        end
    end

    vals.ier = 0
    return vals
end
