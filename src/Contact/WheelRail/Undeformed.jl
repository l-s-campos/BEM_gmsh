# Undeformed distance on the planar potential-contact grid.
# CONTACT `wr_ud_planar`: transform rail/wheel to pot.contact (x,s,n), then
# `h = n_rail − n_wheel`. Negative = interpenetration. After sampling we shift
# so min(h)=0 and return the approach `pen = −hmin` (CONTACT `kin%pen`).

"""Cell-centre coordinates of the planar pot.contact grid in contact axes `(x, s)`."""
function potcon_axes(cp::ContactPatch)
    x = [cp.xl + (i - 1) * cp.dx_eff for i in 1:cp.mx]
    s = [cp.yl + (j - 1) * cp.ds_eff for j in 1:cp.my]
    return x, s
end

"""Linear interpolate `f(s)` from a (possibly unsorted) polyline."""
function _interp_sorted(s, f, sq)
    n = length(s)
    n == 0 && return 999.0
    sq <= s[1] && return f[1]
    sq >= s[n] && return f[n]
    i = searchsortedlast(s, sq)
    i = clamp(i, 1, n - 1)
    den = s[i+1] - s[i]
    t = abs(den) < 1e-16 ? 0.0 : (sq - s[i]) / den
    return f[i] + t * (f[i+1] - f[i])
end

"""Rail `n(s)` in the contact frame (prismatic; independent of `x`)."""
function _rail_n_of_s(rail::WRProfile, m_rail::Marker, mpot::Marker)
    n = length(rail.spl)
    ss = Vector{Float64}(undef, n)
    nn = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        p = vec_2loc(vec_2glob([0.0, rail.spl.y[i], rail.spl.z[i]], m_rail), mpot)
        ss[i] = p[2]
        nn[i] = p[3]
    end
    perm = sortperm(ss)
    return ss[perm], nn[perm]
end

"""
Wheel `n(s)` at a fixed contact-`x` (body of revolution about the axle).

CONTACT `grid_revolve_profile` with `z_axle = −R`:
`z = −R + sqrt((R + z_p)² − x_w²)` in wheel profile coordinates.
"""
function _wheel_n_of_s(wheel_spl::ProfileSpline, m_wheel_trk::Marker, mpot::Marker,
                      nom_radius, x_c)
    # grid point (x_c, 0, 0) in contact → wheel, to get the wheel-x of this slice
    p0 = vec_2loc(vec_2glob([x_c, 0.0, 0.0], mpot), m_wheel_trk)
    xw = p0[1]
    n = length(wheel_spl)
    ss = Float64[]; nn = Float64[]
    sizehint!(ss, n); sizehint!(nn, n)
    @inbounds for i in 1:n
        yw = wheel_spl.y[i]
        zp = wheel_spl.z[i]
        ry = nom_radius + zp
        rad2 = ry * ry - xw * xw
        rad2 <= 0 && continue
        zw = sqrt(rad2) - nom_radius
        p = vec_2loc(vec_2glob([xw, yw, zw], m_wheel_trk), mpot)
        push!(ss, p[2])
        push!(nn, p[3])
    end
    perm = sortperm(ss)
    return ss[perm], nn[perm]
end

"""
Sample undeformed distance. Returns `(x, s, h, pen)` with `min(h) == 0` and
`pen = −min(h_raw)` (CONTACT).
"""
function undeformed_distance(cp::ContactPatch, rail::WRProfile, wheel_spl::ProfileSpline,
                             m_rail::Marker, m_wheel_trk::Marker, nom_radius;
                             wheel::Union{WRProfile,Nothing}=nothing, pitch=0.0)
    x, s = potcon_axes(cp)
    mx, my = length(x), length(s)
    h = fill(999.0, mx, my)
    mpot = cp.mpot
    sr, nr = _rail_n_of_s(rail, m_rail, mpot)
    oor = wheel !== nothing && is_varprof(wheel)
    @inbounds for i in 1:mx
        spl_i = wheel_spl
        if oor
            p0 = vec_2loc(vec_2glob([x[i], 0.0, 0.0], mpot), m_wheel_trk)
            Δθ = asin(clamp(p0[1] / max(nom_radius, 1e-6), -1.0, 1.0))
            spl_i = profile_at_theta(wheel, -pitch + Δθ)
        end
        sw, nw = _wheel_n_of_s(spl_i, m_wheel_trk, mpot, nom_radius, x[i])
        for j in 1:my
            n_r = _interp_sorted(sr, nr, s[j])
            n_w = _interp_sorted(sw, nw, s[j])
            h[i, j] = n_r - n_w
        end
    end
    hmin = minimum(h)
    h .-= hmin
    pen = -hmin
    return x, s, h, pen
end
