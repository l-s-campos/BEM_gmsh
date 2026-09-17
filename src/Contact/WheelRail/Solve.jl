# FZ iteration + per-patch Pohrt–Uzawa rolling (CONTACT `wr_contact` / `wr_solve_cp`).

Base.@kwdef struct PatchResult
    FN::Float64 = 0.0
    FX::Float64 = 0.0
    FS::Float64 = 0.0
    pmax::Float64 = 0.0
    approach::Float64 = 0.0
    ncon::Int = 0
    ξx::Float64 = 0.0
    ξy::Float64 = 0.0
    φ::Float64 = 0.0
    YCP_tr::Float64 = 0.0
    ZCP_tr::Float64 = 0.0
    XCP_tr::Float64 = 0.0
    DELT::Float64 = 0.0
    mx::Int = 0
    my::Int = 0
end

Base.@kwdef struct WRResult
    z_ws::Float64 = 0.0
    FZ_tr::Float64 = 0.0
    FX_tr::Float64 = 0.0
    FY_tr::Float64 = 0.0
    patches::Vector{PatchResult} = PatchResult[]
    niter::Int = 0
end

"""Move `mpot` so the (pen − h)+ centroid sits at the origin (CONTACT-like wgt)."""
function _recenter_gap!(cp, rail, wheel_spl, m_rail, m_wheel_trk, nom_radius, x, s, h, pen;
                        wheel=nothing, pitch=0.0)
    mx, my = size(h)
    mx == 0 && return x, s, h, pen
    wsum = 0.0
    xw = 0.0
    sw = 0.0
    @inbounds for j in 1:my, i in 1:mx
        w = max(pen - h[i, j], 0.0)
        w == 0 && continue
        wsum += w
        xw += w * x[i]
        sw += w * s[j]
    end
    wsum == 0 && return x, s, h, pen
    dx, ds = xw / wsum, sw / wsum
    hypot(dx, ds) < 0.5 * max(cp.dx_eff, cp.ds_eff) && return x, s, h, pen
    p = vec_2glob([dx, ds, 0.0], cp.mpot)
    cp.mref = Marker()
    marker_roll!(cp.mref, cp.delttr)
    marker_shift!(cp.mref, p[1], p[2], p[3])
    cp.mpot = cp.mref
    return undeformed_distance(cp, rail, wheel_spl, m_rail, m_wheel_trk, nom_radius;
                               wheel=wheel, pitch=pitch)
end

function _solve_one_patch(cp::ContactPatch, rail, wheel_spl, m_rail, m_wheel_ws, m_ws_trk,
                          ws, sgn, μ, G_r, ν_r, G_w, ν_w; tol=1e-6, maxiter=250,
                          wheel=nothing)
    m_wheel_trk = marker_2glob(m_wheel_ws, m_ws_trk)
    x, s, h, pen = undeformed_distance(cp, rail, wheel_spl, m_rail, m_wheel_trk, ws.nom_radius;
                                       wheel=wheel, pitch=ws.pitch)
    # shift the contact origin to the gap centroid so spin (odd in s) has ~zero net FX
    x, s, h, pen = _recenter_gap!(cp, rail, wheel_spl, m_rail, m_wheel_trk, ws.nom_radius,
                                  x, s, h, pen; wheel=wheel, pitch=ws.pitch)
    x, s, h, pen = _recenter_gap!(cp, rail, wheel_spl, m_rail, m_wheel_trk, ws.nom_radius,
                                  x, s, h, pen; wheel=wheel, pitch=ws.pitch)
    hs = combined_halfspace(G_r, ν_r, G_w, ν_w; hx=cp.dx_eff, hy=cp.ds_eff)
    grid = make_grid(x, s, hs, h)
    prep = precompute_kernels(length(x), length(s), hs)
    st = init_state(grid)
    V, ξx, ξy, φ = creepage_at_patch(cp, ws, m_wheel_ws, m_ws_trk, sgn; pen=pen)
    kin = RollingKinematics(V, ξx, ξy, φ)
    law = isotropic_law(μ, 0.0)
    niter, Ψ = solve_rolling_step!(st, grid, prep, law, kin, pen;
                                   wear=false, tol=tol, maxiter=maxiter, johnson=true)
    P, Qx, Qy = contact_resultants(st, hs)
    stats = contact_patch_stats(st, hs)
    # rotate contact-frame forces to track: F_tr = R_mref * F_cp
    # contact: FX along x (rolling), FS along s, FN along n (into rail, +z_cp)
    F_cp = [Qx, Qy, P]
    F_tr = cp.mref.R * F_cp
    return PatchResult(
        FN=P, FX=Qx, FS=Qy, pmax=stats.pmax, approach=pen,
        ncon=stats.n, ξx=ξx, ξy=ξy, φ=φ,
        YCP_tr=sgn * oy(cp.mref), ZCP_tr=oz(cp.mref), XCP_tr=ox(cp.mref),
        DELT=sgn * cp.delttr, mx=cp.mx, my=cp.my,
    ), F_tr, niter, Ψ
end

function _forces_at_z(z_ws, rail, wheel, trk, ws_tmpl, sgn, μ, G_r, ν_r, G_w, ν_w;
                      dx, ds, npot_max, wheel_spl, kwargs...)
    ws = deepcopy(ws_tmpl)
    ws.z = z_ws
    m_rail, _, _ = set_rail_marker(trk, rail, sgn)
    m_wheel_ws, m_ws_trk = set_wheel_markers(ws, sgn)
    patches = locate_patches(rail, wheel, m_rail, m_wheel_ws, m_ws_trk, ws, sgn;
                             dx=dx, ds=ds, npot_max=npot_max)
    isempty(patches) && return 0.0, 0.0, 0.0, PatchResult[], 0
    Ftr = zeros(3)
    pres = PatchResult[]
    ntot = 0
    for cp in patches
        pr, ftr, niter, _ = _solve_one_patch(cp, rail, wheel_spl, m_rail, m_wheel_ws, m_ws_trk,
                                             ws, sgn, μ, G_r, ν_r, G_w, ν_w;
                                             wheel=wheel, kwargs...)
        Ftr .+= ftr
        push!(pres, pr)
        ntot += niter
    end
    return Ftr[3], Ftr[1], Ftr[2], pres, ntot
end

"""1D gap at a trial `z_ws`. Increasing `z_ws` makes the gap more negative (dF/dz > 0)."""
function _gap_min_at_z(z_ws, rail, wheel, trk, ws_tmpl, sgn; ds=0.2)
    ws = deepcopy(ws_tmpl)
    ws.z = z_ws
    m_rail, _, _ = set_rail_marker(trk, rail, sgn)
    m_w, m_ws = set_wheel_markers(ws, sgn)
    return rigid_gap_min(rail, wheel, m_rail, m_w, m_ws, ws, sgn; ds=ds)
end

"""
N=1 on `z_ws` so `FZ_track ≈ ws.fz` (CONTACT Brent, frictionless then Johnson).

CONTACT has dFz/dz > 0: the wheel sits below the rail and raising `z_ws` increases
interpenetration. A warm-started `z` from a neighbouring yaw can be several
millimetres off (flange, z ≈ −8 mm vs tread z ≈ 0.2 mm). We therefore shift to
the rigid just-touch `z ← z + gmin` *before* any BEM solve, then secant on Fz.
"""
function match_fz!(ws::WheelsetGeom, rail, wheel, trk, sgn;
                   μ=0.3, G_r=82000.0, ν_r=0.28, G_w=82000.0, ν_w=0.28,
                   dx=0.2, ds=0.2, npot_max=4000, rtol=1e-3, maxit=12,
                   tol=1e-6, maxiter=250)
    wheel_spl = is_varprof(wheel) ? profile_at_theta(wheel, ws.pitch) : wheel.spl
    Ftarget = ws.fz
    z = ws.z
    gmin = 0.0
    for _ in 1:8
        gmin = _gap_min_at_z(z, rail, wheel, trk, ws, sgn; ds=ds)
        abs(gmin) < 5e-4 && break
        abs(gmin) > 500 && break
        z += clamp(gmin, -8.0, 8.0)
    end
    if abs(gmin) > 2.0
        ws.z = z
        return WRResult(z_ws=z, FZ_tr=0.0, FX_tr=0.0, FY_tr=0.0,
                        patches=PatchResult[], niter=0)
    end
    z_touch = z

    # frictionless Fz while iterating z (CONTACT MAXOUT=1 / Johnson split)
    z1 = z_touch
    F1 = 0.0
    z2 = z_touch + 0.015
    F2, FX, FY, pres, n = _forces_at_z(z2, rail, wheel, trk, ws, sgn, 0.0, G_r, ν_r, G_w, ν_w;
                                       dx=dx, ds=ds, npot_max=npot_max, wheel_spl=wheel_spl,
                                       tol=tol, maxiter=maxiter)
    ntot = n
    if F2 < 1.0
        for δ in (0.04, 0.10, 0.25, 0.60)
            z2 = z_touch + δ
            F2, FX, FY, pres, n = _forces_at_z(z2, rail, wheel, trk, ws, sgn, 0.0, G_r, ν_r, G_w, ν_w;
                                               dx=dx, ds=ds, npot_max=npot_max, wheel_spl=wheel_spl,
                                               tol=tol, maxiter=maxiter)
            ntot += n
            F2 >= 1.0 && break
        end
    end
    if F2 > 8 * Ftarget
        z2 = z_touch + 0.004
        F2, FX, FY, pres, n = _forces_at_z(z2, rail, wheel, trk, ws, sgn, 0.0, G_r, ν_r, G_w, ν_w;
                                           dx=dx, ds=ds, npot_max=npot_max, wheel_spl=wheel_spl,
                                           tol=tol, maxiter=maxiter)
        ntot += n
    end
    z = z2
    FZ = F2
    if F2 < 1.0
        ws.z = z
        return WRResult(z_ws=z, FZ_tr=0.0, FX_tr=0.0, FY_tr=0.0,
                        patches=pres, niter=ntot)
    end
    for it in 1:maxit
        denom = F2 - F1
        if abs(denom) < 1e-3
            z = z2 + 0.004 * sign(Ftarget - F2)   # dF/dz > 0
        else
            z = z2 - (F2 - Ftarget) * (z2 - z1) / denom
        end
        z = clamp(z, z_touch - 0.05, z_touch + 1.50)
        z = clamp(z, z2 - 0.25, z2 + 0.25)
        FZ, FX, FY, pres, n = _forces_at_z(z, rail, wheel, trk, ws, sgn, 0.0, G_r, ν_r, G_w, ν_w;
                                           dx=dx, ds=ds, npot_max=npot_max, wheel_spl=wheel_spl,
                                           tol=tol, maxiter=maxiter)
        ntot += n
        if FZ > 8 * Ftarget
            z = min(z, z2) - 0.01
            continue
        end
        abs(FZ - Ftarget) <= rtol * max(Ftarget, 1.0) && break
        z1, F1 = z2, F2
        z2, F2 = z, FZ
    end
    # final frictional solve at the matched z
    FZ, FX, FY, pres, n = _forces_at_z(z, rail, wheel, trk, ws, sgn, μ, G_r, ν_r, G_w, ν_w;
                                       dx=dx, ds=ds, npot_max=npot_max, wheel_spl=wheel_spl,
                                       tol=tol, maxiter=maxiter)
    ntot += n
    ws.z = z
    return WRResult(z_ws=z, FZ_tr=FZ, FX_tr=FX, FY_tr=FY, patches=pres, niter=ntot)
end

"""Solve a wheel–rail case. `z` in `ws` is used when `force=false`; else N=1 on `ws.fz`."""
function solve_wheel_rail(trk::TrackGeom, ws::WheelsetGeom, rail::WRProfile, wheel::WRProfile;
                          side=:left, μ=0.3, G_r=82000.0, ν_r=0.28, G_w=82000.0, ν_w=0.28,
                          dx=0.2, ds=0.2, npot_max=4000, force=true,
                          rtol=1e-3, maxit=12, tol=1e-6, maxiter=250)
    sgn = side === :left || side === 0 ? -1.0 : 1.0
    if force
        return match_fz!(ws, rail, wheel, trk, sgn; μ=μ, G_r=G_r, ν_r=ν_r, G_w=G_w, ν_w=ν_w,
                         dx=dx, ds=ds, npot_max=npot_max, rtol=rtol, maxit=maxit,
                         tol=tol, maxiter=maxiter)
    end
    wheel_spl = is_varprof(wheel) ? profile_at_theta(wheel, ws.pitch) : wheel.spl
    FZ, FX, FY, pres, n = _forces_at_z(ws.z, rail, wheel, trk, ws, sgn, μ, G_r, ν_r, G_w, ν_w;
                                       dx=dx, ds=ds, npot_max=npot_max, wheel_spl=wheel_spl,
                                       tol=tol, maxiter=maxiter)
    return WRResult(z_ws=ws.z, FZ_tr=FZ, FX_tr=FX, FY_tr=FY, patches=pres, niter=n)
end
