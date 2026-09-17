# Planar D=2 contact location (CONTACT `compute_wr_locus` + `locate_interpen_1d`).

Base.@kwdef mutable struct ContactPatch
    xsta::Float64 = 0.0
    xend::Float64 = 0.0
    ysta::Float64 = 0.0
    yend::Float64 = 0.0
    zsta::Float64 = 0.0
    zend::Float64 = 0.0
    micp::Vector{Float64} = zeros(3)     # max interpenetration (track)
    wgt::Vector{Float64} = zeros(3)      # weighted centre (track)
    gap_min::Float64 = 0.0
    delttr::Float64 = 0.0                # contact angle
    sr_ref::Float64 = 0.0
    sr_pot::Float64 = 0.0
    mref::Marker = Marker()
    mpot::Marker = Marker()
    dx_eff::Float64 = 0.2
    ds_eff::Float64 = 0.2
    sp_sta::Float64 = 0.0
    sp_end::Float64 = 0.0
    mx::Int = 0
    my::Int = 0
    xl::Float64 = 0.0
    yl::Float64 = 0.0
end

"""Prismatic contact locus in wheel profile coords (CONTACT `locus_prismatic`)."""
function locus_prismatic(spl::ProfileSpline, nom_radius, R_vw)
    n = length(spl)
    x = similar(spl.y)
    z = similar(spl.z)
    y = copy(spl.y)
    R11, R12, R13 = R_vw[1,1], R_vw[1,2], R_vw[1,3]
    @inbounds for i in 1:n
        ry = nom_radius + spl.z[i]
        dy = eval_dy(spl, spl.s[i])
        dz = eval_dz(spl, spl.s[i])
        dzdy = dz / (signbit(dy) ? min(dy, -1e-6) : max(dy, 1e-6))
        xi = (ry * dzdy * R12 - nom_radius * R13) / max(R11, 1e-12)
        xi = clamp(xi, -0.5 * nom_radius, 0.5 * nom_radius)
        x[i] = xi
        z[i] = ry^2 - xi^2 > 0 ? sqrt(ry^2 - xi^2) - nom_radius : -999.0
    end
    return x, y, z
end

"""Wheel (y,z) spline at this pitch: variable profiles use the θ-slice."""
_wheel_spline(wheel::WRProfile, pitch) =
    is_varprof(wheel) ? profile_at_theta(wheel, pitch) : wheel.spl

"""
Prismatic 1D gap `zr − zw` in track y (CONTACT `compute_wr_locus` + gap).

Negative = interpenetration. Always returns `gmin` even when the surfaces are
open, so N=1 can shift `z_ws` to just-touch (`z ← z + gmin`; dF/dz > 0).
"""
function rigid_gap_1d(rail::WRProfile, wheel::WRProfile,
                      m_rail::Marker, m_wheel_ws::Marker, m_ws_trk::Marker,
                      ws::WheelsetGeom, sgn; ds=0.2)
    m_wheel_trk = marker_2glob(m_wheel_ws, m_ws_trk)
    rail_trk = transform_profile(rail.spl, m_rail)
    spl_w = _wheel_spline(wheel, ws.pitch)
    xw, yw, zw = locus_prismatic(spl_w, ws.nom_radius, m_wheel_trk.R)
    n = length(yw)
    xt = similar(xw); yt = similar(yw); zt = similar(zw)
    @inbounds for i in 1:n
        p = vec_2glob([xw[i], yw[i], zw[i]], m_wheel_trk)
        xt[i] = p[1]; yt[i] = p[2]; zt[i] = p[3]
    end
    perm = sortperm(yt)
    xt, yt, zt = xt[perm], yt[perm], zt[perm]

    # CONTACT: gap = 999 outside the rail surface. Do not sample the wheel
    # flange past the rail head — that invented a −12 mm “contact” at y=695
    # on r300_wide (true tread YCP ≈ 759).
    ymin_r, ymax_r = extrema(rail_trk.y)
    ymin = ymin_r - 1.0
    ymax = ymax_r + 1.0
    dy = 0.3 * ds
    ny = max(32, Int(ceil((ymax - ymin) / dy)) + 1)
    ys = collect(range(ymin, ymax; length=ny))
    zr = [z_at_y(rail_trk, y) for y in ys]
    zwg = [_interp_xy(yt, zt, y) for y in ys]
    xlc = [_interp_xy(yt, xt, y) for y in ys]
    gap = zr .- zwg
    @inbounds for i in 1:ny
        outside = ys[i] < ymin_r || ys[i] > ymax_r
        (outside || zwg[i] < -500 || !isfinite(zwg[i]) || !isfinite(zr[i])) && (gap[i] = 999.0)
    end
    gmin, imin = findmin(gap)
    return (; ys, gap, zr, zwg, xlc, rail_trk, gmin, imin, m_wheel_trk)
end

"""
2D gap for a variable (OOR) wheel (CONTACT `compute_oor_whl` + `locate_interpen_2d`).

Slices around `θ = −pitch` (contact at +z in wheel-centre coords) are revolved
and compared to the prismatic rail. Returns the same named tuple as
`rigid_gap_1d` plus `xsta/xend` from the 2D interpenetration bbox.
"""
function rigid_gap_oor(rail::WRProfile, wheel::WRProfile,
                       m_rail::Marker, m_wheel_ws::Marker, m_ws_trk::Marker,
                       ws::WheelsetGeom, sgn; ds=0.4, dx=0.4)
    R = ws.nom_radius
    span = 0.20
    nslc = max(11, Int(round(2 * span * R / (4 * max(dx, ds))) + 2))
    iseven(nslc) && (nslc += 1)
    dθ = 2 * span / (nslc - 1)
    θc = -ws.pitch
    m_wheel_trk = marker_2glob(m_wheel_ws, m_ws_trk)
    rail_trk = transform_profile(rail.spl, m_rail)
    ymin_r, ymax_r = extrema(rail_trk.y)
    ymin, ymax = ymin_r - 1.0, ymax_r + 1.0
    dy = 0.3 * ds
    ny = max(32, Int(ceil((ymax - ymin) / dy)) + 1)
    ys = collect(range(ymin, ymax; length=ny))
    zr = [z_at_y(rail_trk, y) for y in ys]
    gap = fill(999.0, ny)
    xlc = zeros(ny)
    zwg = fill(999.0, ny)
    wsum = 0.0
    xwsum = 0.0
    ywsum = 0.0
    xsta, xend = Inf, -Inf
    @inbounds for k in 1:nslc
        Δθ = -span + (k - 1) * dθ
        spl = profile_at_theta(wheel, θc + Δθ)
        n = length(spl)
        xt = Vector{Float64}(undef, n)
        yt = Vector{Float64}(undef, n)
        zt = Vector{Float64}(undef, n)
        for i in 1:n
            r = R + spl.z[i]
            p = vec_2glob([r * sin(Δθ), spl.y[i], r * cos(Δθ) - R], m_wheel_trk)
            xt[i] = p[1]; yt[i] = p[2]; zt[i] = p[3]
        end
        perm = sortperm(yt)
        yt, zt, xt = yt[perm], zt[perm], xt[perm]
        for j in 1:ny
            (ys[j] < ymin_r || ys[j] > ymax_r) && continue
            zw = _interp_xy(yt, zt, ys[j])
            (!isfinite(zw) || zw < -500) && continue
            gj = zr[j] - zw
            if gj < gap[j]
                gap[j] = gj
                xlc[j] = _interp_xy(yt, xt, ys[j])
                zwg[j] = zw
            end
            if gj < 0
                w = -gj
                wsum += w
                xwsum += w * _interp_xy(yt, xt, ys[j])
                ywsum += w * ys[j]
                xk = _interp_xy(yt, xt, ys[j])
                xsta = min(xsta, xk)
                xend = max(xend, xk)
            end
        end
    end
    gmin, imin = findmin(gap)
    return (; ys, gap, zr, zwg, xlc, rail_trk, gmin, imin, m_wheel_trk,
              xsta, xend, wgt_x = wsum > 0 ? xwsum / wsum : xlc[imin],
              wgt_y = wsum > 0 ? ywsum / wsum : ys[imin])
end

function rigid_gap_min(rail, wheel, m_rail, m_wheel_ws, m_ws_trk, ws, sgn; ds=0.2)
    g = is_varprof(wheel) ?
        rigid_gap_oor(rail, wheel, m_rail, m_wheel_ws, m_ws_trk, ws, sgn; ds=ds, dx=ds) :
        rigid_gap_1d(rail, wheel, m_rail, m_wheel_ws, m_ws_trk, ws, sgn; ds=ds)
    return g.gmin
end

"""
Locate planar contact patches for one wheel/rail pair.

`rail_trk` / `wheel_ws` are the stored profiles; markers place them in the
internal right-rail track frame.
"""
function locate_patches(rail::WRProfile, wheel::WRProfile,
                        m_rail::Marker, m_wheel_ws::Marker, m_ws_trk::Marker,
                        ws::WheelsetGeom, sgn;
                        dx=0.2, ds=0.2, npot_max=4000,
                        a_sep=π/2, d_sep=8.0, d_comb=4.0)
    g = is_varprof(wheel) ?
        rigid_gap_oor(rail, wheel, m_rail, m_wheel_ws, m_ws_trk, ws, sgn; ds=ds, dx=dx) :
        rigid_gap_1d(rail, wheel, m_rail, m_wheel_ws, m_ws_trk, ws, sgn; ds=ds)
    ys, gap, zr, xlc, rail_trk = g.ys, g.gap, g.zr, g.xlc, g.rail_trk
    gmin, imin, ny = g.gmin, g.imin, length(g.ys)
    gmin > 0 && return ContactPatch[]   # no contact

    # interpenetration interval
    i0 = imin
    while i0 > 1 && gap[i0-1] < 0
        i0 -= 1
    end
    i1 = imin
    while i1 < ny && gap[i1+1] < 0
        i1 += 1
    end
    # one guard cell
    i0 = max(1, i0 - 1)
    i1 = min(ny, i1 + 1)

    # linear zeros at the edges
    ysta, zsta = _zero_cross(ys, gap, zr, i0, +1)
    yend, zend = _zero_cross(ys, gap, zr, i1, -1)

    # weighted centre (gap as weight, only gap<0)
    wsum = 0.0
    ywsum = 0.0
    xwsum = 0.0
    zwsum = 0.0
    asums = 0.0
    @inbounds for i in i0:i1
        gap[i] >= 0 && continue
        w = -gap[i]
        wsum += w
        ywsum += w * ys[i]
        xwsum += w * xlc[i]
        zwsum += w * zr[i]
        asums += w * eval_alpha(rail_trk, s_at_y(rail_trk, ys[i]))
    end
    if wsum > 0
        wgt_x = xwsum / wsum
        wgt_y = ywsum / wsum
        wgt_z = z_at_y(rail_trk, wgt_y)
        wgt_a = asums / wsum
    else
        wgt_x, wgt_y, wgt_z = xlc[imin], ys[imin], zr[imin]
        wgt_a = eval_alpha(rail_trk, s_at_y(rail_trk, wgt_y))
    end
    # CONTACT D=2 "wgt-wgt": origin at the gap-weighted centre so spin (odd in y)
    # integrates to ~zero net FX. Angle is the gap-weighted rail inclination.
    cref_x = wgt_x
    cref_y = wgt_y
    cref_z = wgt_z
    delt = wgt_a

    # rolling-direction extent: OOR uses the 2D bbox; prismatic uses 1d gap + curvature
    if is_varprof(wheel) && haskey(g, :xsta) && isfinite(g.xsta)
        xmin, xmax = g.xsta, g.xend
        wgt_x, wgt_y = g.wgt_x, g.wgt_y
        wgt_z = z_at_y(rail_trk, wgt_y)
        wgt_a = eval_alpha(rail_trk, s_at_y(rail_trk, wgt_y))
        cref_x, cref_y, cref_z, delt = wgt_x, wgt_y, wgt_z, wgt_a
    else
        curv = 0.5 / ws.nom_radius
        xmin, xmax = _contact_length_x(xlc, gap, i0, i1, curv)
    end

    cp = ContactPatch()
    cp.xsta = xmin; cp.xend = xmax
    cp.ysta = ysta; cp.yend = yend
    cp.zsta = zsta; cp.zend = zend
    cp.micp = [cref_x, cref_y, cref_z]
    cp.wgt = [wgt_x, wgt_y, wgt_z]
    cp.gap_min = gmin
    cp.delttr = delt
    cp.sr_ref = s_at_y(rail_trk, cref_y)
    cp.mref = Marker()
    marker_roll!(cp.mref, delt)
    marker_shift!(cp.mref, cref_x, cref_y, cref_z)
    cp.mpot = cp.mref
    cp.sr_pot = cp.sr_ref

    set_planar_potcon!(cp, rail_trk; dx=dx, ds=ds, npot_max=npot_max)
    return [cp]
end

"""Linear interpolate `f(x)` without assuming `x` is sorted."""
function _interp_xy(x, f, xq)
    n = length(x)
    n == 1 && return f[1]
    k = 1
    dmin = abs(x[1] - xq)
    @inbounds for i in 2:n
        d = abs(x[i] - xq)
        if d < dmin
            dmin = d
            k = i
        end
    end
    # prefer a neighbour that brackets xq
    if k < n && (x[k] - xq) * (x[k+1] - xq) <= 0
        den = x[k+1] - x[k]
        t = abs(den) < 1e-16 ? 0.0 : (xq - x[k]) / den
        return f[k] + t * (f[k+1] - f[k])
    elseif k > 1 && (x[k] - xq) * (x[k-1] - xq) <= 0
        den = x[k] - x[k-1]
        t = abs(den) < 1e-16 ? 0.0 : (xq - x[k-1]) / den
        return f[k-1] + t * (f[k] - f[k-1])
    end
    return f[k]
end

function _zero_cross(ys, gap, zr, i, dir)
    n = length(ys)
    j = clamp(i + dir, 1, n)
    g0, g1 = gap[i], gap[j]
    if g0 * g1 < 0 && abs(g1 - g0) > 1e-14
        t = -g0 / (g1 - g0)
        return ys[i] + t * (ys[j] - ys[i]), zr[i] + t * (zr[j] - zr[i])
    end
    return ys[i], zr[i]
end

function _contact_length_x(x, gap, i0, i1, curv)
    # CONTACT: gap(x) ≈ gap_1d + curv x^2  →  |x| = sqrt(-gap/curv)
    xmin = Inf; xmax = -Inf
    @inbounds for i in i0:i1
        g = gap[i]
        g >= 0 && continue
        half = curv > 0 ? sqrt(max(-g / curv, 0.0)) : 0.0
        xmin = min(xmin, x[i] - half)
        xmax = max(xmax, x[i] + half)
    end
    isfinite(xmin) || (xmin = 0.0; xmax = 0.0)
    return xmin, xmax
end

function set_planar_potcon!(cp::ContactPatch, rail_trk::ProfileSpline; dx=0.2, ds=0.2, npot_max=4000)
    sr_sta = s_at_y(rail_trk, cp.ysta)
    sr_end = s_at_y(rail_trk, cp.yend)
    if sr_sta > sr_end
        sr_sta, sr_end = sr_end, sr_sta
    end
    cp.sp_sta = sr_sta - cp.sr_pot
    cp.sp_end = sr_end - cp.sr_pot
    xp_l = cp.xsta - ox(cp.mpot)
    xp_h = cp.xend - ox(cp.mpot)
    if xp_l > xp_h
        xp_l, xp_h = xp_h, xp_l
    end
    dx_eff, ds_eff = float(dx), float(ds)
    ix_l = floor(Int, xp_l / dx_eff) - 1
    ix_h = ceil(Int, xp_h / dx_eff) + 1
    iy_l = floor(Int, cp.sp_sta / ds_eff) - 1
    iy_h = ceil(Int, cp.sp_end / ds_eff) + 1
    mx = ix_h - ix_l + 1
    my = iy_h - iy_l + 1
    while mx * my >= npot_max && npot_max >= 100
        if mx > 10
            dx_eff *= 2
            ix_l = floor(Int, xp_l / dx_eff) - 1
            ix_h = ceil(Int, xp_h / dx_eff) + 1
            mx = ix_h - ix_l + 1
        end
        if mx * my >= npot_max && my > 10
            ds_eff *= 2
            iy_l = floor(Int, cp.sp_sta / ds_eff) - 1
            iy_h = ceil(Int, cp.sp_end / ds_eff) + 1
            my = iy_h - iy_l + 1
        end
        (mx <= 10 && my <= 10) && break
    end
    # even sizes for FFT
    isodd(mx) && (mx += 1; ix_h += 1)
    isodd(my) && (my += 1; iy_h += 1)
    cp.dx_eff = dx_eff
    cp.ds_eff = ds_eff
    cp.mx = mx
    cp.my = my
    cp.xl = (ix_l + 0.5) * dx_eff
    cp.yl = (iy_l + 0.5) * ds_eff
    return cp
end
