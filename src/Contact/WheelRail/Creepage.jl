# Rigid creepage at the contact reference (CONTACT `wr_creep_cref`).
# Rail is at rest. Wheelset: translation `vs` and pitch `vpitch`.

"""
Return `(V, ξx, ξy, φ)` in the contact frame.

Kalker: `ξ = (v_wheel − v_rail) / V` at the contact reference, with
`V = |vs|`. Spin `φ` is the contact-frame z-component of `ω_wheel − ω_rail`
divided by `V`.
"""
function creepage_at_patch(cp::ContactPatch, ws::WheelsetGeom,
                           m_wheel_ws::Marker, m_ws_trk::Marker, sgn; pen=0.0)
    V = abs(ws.vs)
    V < 1e-12 && return 0.0, 0.0, 0.0, 0.0

    # rail at rest (prismatic, no curve): vp = 0
    ws_tvel_trk = [ws.vs, sgn * ws.vy, ws.vz]
    ws_rvel_ws  = [sgn * ws.vroll, ws.vpitch, sgn * ws.vyaw]
    ws_rvel_trk = m_ws_trk.R * ws_rvel_ws

    # Q on the wheel sits `pen` above the rail reference (CONTACT `mq_cp`)
    mq_trk = marker_2glob(Marker(0.0, 0.0, pen), cp.mref)
    mq_ws = marker_2loc(mq_trk, m_ws_trk)
    vQ_trk = vec_veloc2glob(ws_tvel_trk, ws_rvel_trk, m_ws_trk.R, mq_ws.o)
    pitch_tvel = cross(m_ws_trk.R * [0.0, ws.vpitch, 0.0], m_ws_trk.R * mq_ws.o)
    vs_est = vQ_trk[1] - pitch_tvel[1]
    Vuse = (abs(vs_est) + abs(pitch_tvel[1])) / 2
    Vuse < 1e-12 && (Vuse = V)

    vQ_cp = cp.mref.R' * vQ_trk
    ω_cp  = cp.mref.R' * ws_rvel_trk
    # CONTACT `wr_creep_cref`: ξ = −v_wheel / V at the contact reference
    # (rail at rest). Do not zero ξx with VS/|ω| — yawed Manchester cases have
    # CKSI ~ 10⁻⁴–10⁻² and FX of hundreds to thousands of newtons.
    ξx = -vQ_cp[1] / Vuse
    ξy = -vQ_cp[2] / Vuse
    φ  =  ω_cp[3] / Vuse
    return Vuse, ξx, ξy, φ
end
