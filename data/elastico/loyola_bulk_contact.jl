using DrWatson: datadir

# =============================================================================
# Loyola 2022 §9.3.2 — bulk-stress fretting (cylindrical pad on flat specimen)
#
# Fig. 9.31 / Table 9.23:
#   pad:    width 2w, height w, radius R, P on top, Q on the pad
#   specimen: width 4w, height 2w; left ux=0, bottom uy=0, right bulk B
#   Q and B in phase.  cargav=100 on pad top ⇒ P = 2 w cargav = 1300 N/mm
#   (Tables 9.24–9.25: a=1.6997 mm, p0=486.92 MPa — cylinder-on-flat Hertz).
# =============================================================================

"""Loyola bulk-stress (sapatas) parameters."""
function loyola_bulk_params()
    w = 6.5
    R = 70.0
    θ = asin(w / R)
    y = R - R * cos(θ)          # sagitta at |x|=w
    E = 73.4e3
    ν = 0.33
    μ = 0.3
    cargav = 100.0
    cargah = 15.0               # Q and B as traction densities (N/mm²)
    P = 2 * w * cargav          # line load on pad top
    East = E / (1 - ν^2)        # plane-strain modulus, one body
    E_eq = East / 2             # two similar bodies
    R_eq = R                    # cylinder on flat
    a_H = sqrt(4 * R_eq * P / (π * E_eq))
    p0_H = 2 * P / (π * a_H)
    a_th = 1.6997               # Tables 9.24–9.25
    p0_th = 486.92
    ya = R - sqrt(max(R^2 - a_H^2, 0.0))
    Qmax = 0.5 * μ * P
    σB = cargah                 # bulk traction on specimen right face
    return (; w, R, θ, y, ya, E, ν, μ, cargav, cargah, P, East, E_eq, R_eq,
        a_H, p0_H, a_th, p0_th, Qmax, σB)
end

"""Nowell offset  e/a = σ / (4 f p0)  (thesis eq. 3.27, at max |Q|)."""
nowell_e(a, σ, f, p0) = a * σ / (4 * f * max(p0, eps()))

"""
Cattaneo shear with a Nowell-shifted stick zone (centre at `e`).
"""
function nowell_shear(x::AbstractVector, a, p0, Q, f, P, e)
    q = zeros(float(typeof(p0)), length(x))
    abs(Q) < 1e-15 * max(f * P, 1.0) && return q
    s = sign(Q) == 0 ? 1.0 : sign(Q)
    c = a * sqrt(max(0.0, 1 - abs(Q) / max(f * P, eps())))
    @inbounds for i in eachindex(x)
        xa = abs(x[i])
        xa >= a && continue
        term = sqrt(1 - (x[i] / a)^2)
        if c > 0 && abs(x[i] - e) < c
            ξ = (x[i] - e) / c
            abs(ξ) < 1 && (term -= (c / a) * sqrt(1 - ξ^2))
        end
        q[i] = s * f * p0 * term
    end
    return q
end

"""Mesh the cylindrical pad (Fig. 9.31, height w, width 2w)."""
function mesh_loyola_bulk_pad(;
        ndiv_c::Int=12, ndiv_f::Int=6, ndiv_s::Int=4, ndiv_top::Int=5,
        μ::Real=0.3, cargav::Real=100.0, nome::AbstractString="bulk_pad",
        show::Bool=false, ordem::Int=1,
    )
    par = loyola_bulk_params()
    w, R, y, a, ya = par.w, par.R, par.y, par.a_H, par.ya
    μs = string(float(μ))
    cvs = string(-float(cargav))
    ytop = w

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(nome)
    lc = 2w / max(ndiv_c + 2 * ndiv_f, 8)

    c = gmsh.model.geo.addPoint(0.0, R, 0.0, lc)
    p1 = gmsh.model.geo.addPoint(-w, y, 0.0, lc)
    p2 = gmsh.model.geo.addPoint(-a, ya, 0.0, lc)
    p3 = gmsh.model.geo.addPoint(a, ya, 0.0, lc)
    p4 = gmsh.model.geo.addPoint(w, y, 0.0, lc)
    p5 = gmsh.model.geo.addPoint(w, ytop, 0.0, lc)
    p6 = gmsh.model.geo.addPoint(-w, ytop, 0.0, lc)

    a1 = gmsh.model.geo.addCircleArc(p1, c, p2)
    a2 = gmsh.model.geo.addCircleArc(p2, c, p3)
    a3 = gmsh.model.geo.addCircleArc(p3, c, p4)
    s4 = gmsh.model.geo.addLine(p4, p5)
    s5 = gmsh.model.geo.addLine(p5, p6)
    s6 = gmsh.model.geo.addLine(p6, p1)
    cl = gmsh.model.geo.addCurveLoop([a1, a2, a3, s4, s5, s6])
    surf = gmsh.model.geo.addPlaneSurface([cl])
    gmsh.model.geo.synchronize()

    gmsh.model.mesh.setTransfiniteCurve(a1, ndiv_f)
    gmsh.model.mesh.setTransfiniteCurve(a2, ndiv_c)
    gmsh.model.mesh.setTransfiniteCurve(a3, ndiv_f)
    gmsh.model.mesh.setTransfiniteCurve(s4, ndiv_s)
    gmsh.model.mesh.setTransfiniteCurve(s5, ndiv_top)
    gmsh.model.mesh.setTransfiniteCurve(s6, ndiv_s)

    gmsh.model.addPhysicalGroup(1, [a1], -1, "4;" * μs * ";4;" * μs * ";8;11")
    gmsh.model.addPhysicalGroup(1, [a2], -1, "4;" * μs * ";4;" * μs * ";8;12")
    gmsh.model.addPhysicalGroup(1, [a3], -1, "4;" * μs * ";4;" * μs * ";8;13")
    gmsh.model.addPhysicalGroup(1, [s4], -1, "1;0;1;0;8;14")
    gmsh.model.addPhysicalGroup(1, [s5], -1, "1;0;1;" * cvs * ";8;15")
    gmsh.model.addPhysicalGroup(1, [s6], -1, "1;0;1;0;8;16")
    gmsh.model.addPhysicalGroup(2, [surf], -1, "Domain")

    gmsh.option.setNumber("Mesh.SecondOrderLinear", 0)
    gmsh.model.mesh.generate(2)
    ordem > 1 && gmsh.model.mesh.setOrder(Int(ordem))
    out = datadir("elastico", nome * ".msh")
    mkpath(dirname(out))
    gmsh.write(out)
    show && gmsh.fltk.run()
    gmsh.finalize()
    return out
end

"""Mesh the flat specimen (width 4w, height 2w). Top split: free | contact | free."""
function mesh_loyola_bulk_specimen(;
        ndiv_c::Int=12, ndiv_f::Int=6, ndiv_out::Int=6,
        ndiv_s::Int=6, ndiv_bot::Int=12, ndiv_bulk::Int=10,
        μ::Real=0.3, nome::AbstractString="bulk_spec",
        show::Bool=false, ordem::Int=1,
    )
    par = loyola_bulk_params()
    w, a = par.w, par.a_H
    μs = string(float(μ))
    W = 2 * w
    H = 2 * w

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(nome)
    lc = 2W / max(ndiv_bot, 8)

    # top, left → right: -2w, -w, -a, a, w, 2w
    t1 = gmsh.model.geo.addPoint(-W, 0.0, 0.0, lc)
    t2 = gmsh.model.geo.addPoint(-w, 0.0, 0.0, lc)
    t3 = gmsh.model.geo.addPoint(-a, 0.0, 0.0, lc)
    t4 = gmsh.model.geo.addPoint(a, 0.0, 0.0, lc)
    t5 = gmsh.model.geo.addPoint(w, 0.0, 0.0, lc)
    t6 = gmsh.model.geo.addPoint(W, 0.0, 0.0, lc)
    bR = gmsh.model.geo.addPoint(W, -H, 0.0, lc)
    bL = gmsh.model.geo.addPoint(-W, -H, 0.0, lc)

    topL = gmsh.model.geo.addLine(t1, t2)
    cL = gmsh.model.geo.addLine(t2, t3)
    cC = gmsh.model.geo.addLine(t3, t4)
    cR = gmsh.model.geo.addLine(t4, t5)
    topR = gmsh.model.geo.addLine(t5, t6)
    right = gmsh.model.geo.addLine(t6, bR)
    bot = gmsh.model.geo.addLine(bR, bL)
    left = gmsh.model.geo.addLine(bL, t1)
    cl = gmsh.model.geo.addCurveLoop([topL, cL, cC, cR, topR, right, bot, left])
    surf = gmsh.model.geo.addPlaneSurface([cl])
    gmsh.model.geo.synchronize()

    gmsh.model.mesh.setTransfiniteCurve(topL, ndiv_out)
    gmsh.model.mesh.setTransfiniteCurve(cL, ndiv_f)
    gmsh.model.mesh.setTransfiniteCurve(cC, ndiv_c)
    gmsh.model.mesh.setTransfiniteCurve(cR, ndiv_f)
    gmsh.model.mesh.setTransfiniteCurve(topR, ndiv_out)
    gmsh.model.mesh.setTransfiniteCurve(right, ndiv_bulk)
    gmsh.model.mesh.setTransfiniteCurve(bot, ndiv_bot)
    gmsh.model.mesh.setTransfiniteCurve(left, ndiv_s)

    # contact on the three central top segments; free on the wings
    gmsh.model.addPhysicalGroup(1, [topL], -1, "1;0;1;0;8;31")
    gmsh.model.addPhysicalGroup(1, [cL], -1, "4;" * μs * ";4;" * μs * ";8;32")
    gmsh.model.addPhysicalGroup(1, [cC], -1, "4;" * μs * ";4;" * μs * ";8;33")
    gmsh.model.addPhysicalGroup(1, [cR], -1, "4;" * μs * ";4;" * μs * ";8;34")
    gmsh.model.addPhysicalGroup(1, [topR], -1, "1;0;1;0;8;35")
    gmsh.model.addPhysicalGroup(1, [right], -1, "1;0;1;0;8;36")          # tx = B later
    gmsh.model.addPhysicalGroup(1, [bot], -1, "1;0;0;0;8;37")            # uy = 0
    gmsh.model.addPhysicalGroup(1, [left], -1, "0;0;1;0;8;38")            # ux = 0
    gmsh.model.addPhysicalGroup(2, [surf], -1, "Domain")

    gmsh.option.setNumber("Mesh.SecondOrderLinear", 0)
    gmsh.model.mesh.generate(2)
    ordem > 1 && gmsh.model.mesh.setOrder(Int(ordem))
    out = datadir("elastico", nome * ".msh")
    mkpath(dirname(out))
    gmsh.write(out)
    show && gmsh.fltk.run()
    gmsh.finalize()
    return out
end

"""
    load_loyola_bulk_contact(; ndiv_c=12, μ=0.3, tipo=2, ...) -> (prob, par)

Pad (reg 1) on specimen (reg 2). NTN contact on the arc vs the flat.
"""
function load_loyola_bulk_contact(;
        ndiv_c::Int=12, ndiv_f::Int=6, ndiv_s::Int=4, ndiv_top::Int=5,
        ndiv_out::Int=6, ndiv_bot::Int=12, ndiv_bulk::Int=10,
        μ::Union{Nothing,Real}=nothing,
        nome::AbstractString="loyola932",
        tipo::Int=2,
        show::Bool=false,
        collocation::Symbol=:legendre,
        gap::Symbol=:normal,
    )
    par = loyola_bulk_params()
    μv = μ === nothing ? par.μ : float(μ)
    props = Elasticity(par.E, par.ν, 1.0; plane_strain=true)

    msh_p = mesh_loyola_bulk_pad(; ndiv_c=ndiv_c, ndiv_f=ndiv_f, ndiv_s=ndiv_s,
        ndiv_top=ndiv_top, μ=μv, cargav=par.cargav, nome=nome * "_pad", show=false,
        ordem=tipo)
    msh_s = mesh_loyola_bulk_specimen(; ndiv_c=ndiv_c, ndiv_f=ndiv_f,
        ndiv_out=ndiv_out, ndiv_s=ndiv_s, ndiv_bot=ndiv_bot, ndiv_bulk=ndiv_bulk,
        μ=μv, nome=nome * "_spec", show=show, ordem=tipo)

    dad_p = format2d(msh_p, props; pontointerno=false, tipo=tipo, collocation=collocation)
    dad_s = format2d(msh_s, props; pontointerno=false, tipo=tipo, collocation=collocation)
    dad_p.name = "pad"
    dad_s.name = "specimen"

    prob = MultiRegionProblem([dad_p, dad_s]; name="loyola_bulk")
    pair_contacts!(prob; method=:ntn, slave_reg=1, master_reg=2, gap=gap)
    for cp in prob.contacts
        cp.μ = μv
        cp.ut_lock = 0.0
        cp.state = 1
    end
    return prob, par
end

"""Mean (u_x, u_y) on the pad top after scatter."""
function pad_top_u_mean(prob)
    dad = prob.regions[1]
    has_cache(dad, :u) || return (0.0, 0.0)
    ymax = maximum(pt[2] for pt in dad.Nodes)
    thr = 1e-9 * max(abs(ymax), 1.0)
    sx = sy = 0.0
    n = 0
    for i in 1:dad.n
        if abs(dad.Nodes[i][2] - ymax) <= thr
            sx += dad.u[2i - 1]
            sy += dad.u[2i]
            n += 1
        end
    end
    n == 0 && return (0.0, 0.0)
    return (sx / n, sy / n)
end

"""Dirichlet (u_x, u_y) on the pad top."""
function apply_pad_top_u!(prob, ux::Real; uy::Real)
    dad = prob.regions[1]
    ymax = maximum(pt[2] for pt in dad.Nodes)
    thr = 1e-9 * max(abs(ymax), 1.0)
    ux = float(ux); uy = float(uy)
    nset = 0
    for i in 1:dad.n
        if abs(dad.Nodes[i][2] - ymax) <= thr
            dad.BC[2i - 1] = 0; dad.BV[2i - 1] = ux
            dad.BC[2i]     = 0; dad.BV[2i]     = uy
            nset += 1
        end
    end
    return nset
end

"""
Pad-holder BC (Fig. 9.31): mixed ``(t_x, u_y)`` on the pad top.

All top nodes share the same ``u_y`` (platen stays horizontal → no tip-over)
and carry a uniform tangential traction ``t_x = Q/(2w)``. Contact friction
is the only ``u_x`` restraint, as in the experiment.
"""
function apply_pad_top_mixed!(prob, tx::Real; uy::Real)
    dad = prob.regions[1]
    ymax = maximum(pt[2] for pt in dad.Nodes)
    thr = 1e-9 * max(abs(ymax), 1.0)
    tx = float(tx); uy = float(uy)
    nset = 0
    for i in 1:dad.n
        if abs(dad.Nodes[i][2] - ymax) <= thr
            dad.BC[2i - 1] = 1; dad.BV[2i - 1] = tx
            dad.BC[2i]     = 0; dad.BV[2i]     = uy
            nset += 1
        end
    end
    return nset
end

"""Bulk traction t_x = σ on the specimen right face (Fig. 9.31, B)."""
function apply_specimen_bulk!(prob, σ::Real)
    dad = prob.regions[2]
    xmax = maximum(pt[1] for pt in dad.Nodes)
    thr = 1e-9 * max(abs(xmax), 1.0)
    σ = float(σ)
    nset = 0
    for i in 1:dad.n
        if abs(dad.Nodes[i][1] - xmax) <= thr
            dad.BC[2i - 1] = 1; dad.BV[2i - 1] = σ
            dad.BC[2i]     = 1; dad.BV[2i]     = 0.0
            nset += 1
        end
    end
    return nset
end
