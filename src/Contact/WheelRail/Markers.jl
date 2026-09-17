# Rigid markers: x_glob = o + R * x_loc  (CONTACT m_markers).
# Roll is about x: y′ = c y − s z, z′ = s y + c z.

const _I3 = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]

rot_roll(α) = (c=cos(α); s=sin(α); [1.0 0.0 0.0; 0.0 c -s; 0.0 s c])
rot_yaw(ψ)  = (c=cos(ψ); s=sin(ψ); [c -s 0.0; s c 0.0; 0.0 0.0 1.0])
rot_pitch(θ) = (c=cos(θ); s=sin(θ); [c 0.0 s; 0.0 1.0 0.0; -s 0.0 c])

"""Origin `o` and rotation `R` (local → parent)."""
struct Marker
    o::Vector{Float64}   # length 3
    R::Matrix{Float64}   # 3×3
end

Marker() = Marker(zeros(3), copy(_I3))
Marker(x, y, z) = Marker(Float64[x, y, z], copy(_I3))

Base.copy(m::Marker) = Marker(copy(m.o), copy(m.R))

ox(m::Marker) = m.o[1]
oy(m::Marker) = m.o[2]
oz(m::Marker) = m.o[3]

function marker_shift!(m::Marker, dx, dy, dz)
    m.o[1] += dx
    m.o[2] += dy
    m.o[3] += dz
    return m
end

function marker_roll!(m::Marker, roll; yc=m.o[2], zc=m.o[3])
    c, s = cos(roll), sin(roll)
    yrel, zrel = m.o[2] - yc, m.o[3] - zc
    m.o[2] = yc + c * yrel - s * zrel
    m.o[3] = zc + s * yrel + c * zrel
    m.R .= m.R * rot_roll(roll)
    return m
end

function marker_rotate!(m::Marker, roll, yaw, pitch=0)
    # CONTACT: rotate first about origin, then the caller shifts.
    m.R .= m.R * rot_roll(roll) * rot_yaw(yaw) * rot_pitch(pitch)
    return m
end

"""`x_glob = o + R * x_loc`."""
vec_2glob(xloc, m::Marker) = m.o + m.R * xloc

"""`x_loc = R' * (x_glob - o)`."""
vec_2loc(xglob, m::Marker) = m.R' * (xglob - m.o)

"""Compose: `x_glob = o_ref + R_ref * (o_loc + R_loc * x)`."""
function marker_2glob(mloc_ref::Marker, mref_glb::Marker)
    return Marker(vec_2glob(mloc_ref.o, mref_glb), mref_glb.R * mloc_ref.R)
end

"""Express `mloc_glb` in the frame of `mref_glb` (CONTACT `marker_2loc`)."""
function marker_2loc(mloc_glb::Marker, mref_glb::Marker)
    R = mref_glb.R' * mloc_glb.R
    o = mref_glb.R' * (mloc_glb.o - mref_glb.o)
    return Marker(o, R)
end

"""`v_P = v_O + ω × (R r_loc) + R v_loc`."""
function vec_veloc2glob(vO, ω, R, r_loc, v_loc=zeros(3))
    return vO + cross(ω, R * r_loc) + R * v_loc
end
