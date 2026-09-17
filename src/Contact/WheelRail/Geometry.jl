# Track / wheelset placement. Internal frame is CONTACT's right-rail convention;
# left-side inputs (Y, roll, yaw) are mirrored with `sgn = -1` and track-y outputs
# are multiplied by `sgn` when reported.

Base.@kwdef mutable struct TrackGeom
    gauge::Float64 = 1435.0
    gauge_height::Float64 = 14.0
    cant::Float64 = 0.0
    rail_dy::Float64 = 0.0
    rail_dz::Float64 = 0.0
    rail_roll::Float64 = 0.0
end

Base.@kwdef mutable struct WheelsetGeom
    nom_radius::Float64 = 460.0
    flback_dist::Float64 = 1360.0
    flback_pos::Float64 = -70.0
    y::Float64 = 0.0
    z::Float64 = 0.0          # z_ws (CONTACT)
    roll::Float64 = 0.0
    yaw::Float64 = 0.0
    pitch::Float64 = 0.0
    vs::Float64 = 2000.0
    vy::Float64 = 0.0
    vz::Float64 = 0.0
    vroll::Float64 = 0.0
    vyaw::Float64 = 0.0
    vpitch::Float64 = 0.0
    fz::Float64 = 10000.0
end

"""Cant the (y,z) polyline by angle `α` about the origin (CONTACT `cartgrid_roll(-cant)`)."""
function roll_points(y, z, α)
    c, s = cos(α), sin(α)
    return c .* y .+ s .* z, -s .* y .+ c .* z   # R_x(-α) applied to (y,z)? 
end

# CONTACT cartgrid_roll(prr, -cant): rotate profile points by -cant about x.
# Point (0,y,z) → (0, c y - s z, s y + c z) with c=cos(-cant)=cos(cant), s=sin(-cant)=-sin(cant)
# i.e. y' =  cos(cant) y + sin(cant) z
#      z' = -sin(cant) y + cos(cant) z
function cant_profile(spl::ProfileSpline, cant)
    iszero(cant) && return spl
    yp, zp = roll_points(spl.y, spl.z, cant)
    return make_profile_spline(yp, zp)
end

"""
Gauge-point placement (CONTACT `find_gauge_meas_pt` + `wr_set_rail_marker`).

Returns `(ygauge0, zmin)` on the canted profile: the inner-face y at gauge
height, and the minimum z (rail head after cant, in profile coords).
"""
function gauge_meas_pt(spl::ProfileSpline, cant, gauge_height)
    cspl = cant_profile(spl, cant)
    ymn, zmin, smin = yz_at_minz(cspl)
    # first point with z-zmin ≤ gaught and s < smin (inside face)
    ygauge0 = cspl.y[1]
    ygauge1 = Inf
    i1st = 0
    @inbounds for i in 1:length(cspl.s)
        cspl.s[i] >= smin && break
        if cspl.z[i] - zmin <= gauge_height
            i1st == 0 && (i1st = i)
            if cspl.y[i] < ygauge1
                ygauge1 = cspl.y[i]
            end
        end
    end
    # y at z = zmin + gaught
    zq = zmin + gauge_height
    # scan for z-crossing on the inside face
    found = false
    @inbounds for i in 1:length(cspl.s)-1
        cspl.s[i] >= smin && break
        zi, zj = cspl.z[i] - zmin, cspl.z[i+1] - zmin
        if (zi - gauge_height) * (zj - gauge_height) <= 0 && abs(zj - zi) > 1e-14
            t = (gauge_height - zi) / (zj - zi)
            ygauge0 = cspl.y[i] + t * (cspl.y[i+1] - cspl.y[i])
            found = true
            break
        end
    end
    found || (ygauge0 = isfinite(ygauge1) ? ygauge1 : cspl.y[1])
    return ygauge0, zmin, cspl
end

"""Rail marker in the internal (right-rail) track frame."""
function set_rail_marker(trk::TrackGeom, rail::WRProfile, sgn)
    ygauge0, zmin, _ = gauge_meas_pt(rail.spl, trk.cant, trk.gauge_height)
    rail_phi = -trk.cant + sgn * trk.rail_roll
    if trk.gauge_height > 0
        rail_y = trk.gauge / 2 - ygauge0 + sgn * trk.rail_dy
        rail_z = -zmin + trk.rail_dz
    else
        rail_y = sgn * trk.rail_dy
        rail_z = trk.rail_dz
    end
    m = Marker()
    marker_roll!(m, rail_phi)
    marker_shift!(m, 0.0, rail_y, rail_z)
    return m, ygauge0, zmin
end

"""Wheel profile marker in wheelset coords, and wheelset marker in track coords."""
function set_wheel_markers(ws::WheelsetGeom, sgn)
    yw_ws = ws.flback_dist / 2 - ws.flback_pos + sgn * 0.0
    zw_ws = ws.nom_radius
    m_ws = Marker()
    marker_shift!(m_ws, 0.0, yw_ws, zw_ws)

    m_trk = Marker()
    marker_rotate!(m_trk, sgn * ws.roll, sgn * ws.yaw, 0.0)
    marker_shift!(m_trk, 0.0, sgn * ws.y, ws.z - ws.nom_radius)
    return m_ws, m_trk
end

"""Transform a profile polyline with a marker (profile coords → parent)."""
function transform_profile(spl::ProfileSpline, m::Marker)
    n = length(spl)
    y = similar(spl.y); z = similar(spl.z)
    @inbounds for i in 1:n
        p = vec_2glob([0.0, spl.y[i], spl.z[i]], m)
        y[i] = p[2]
        z[i] = p[3]
    end
    return make_profile_spline(y, z)
end
