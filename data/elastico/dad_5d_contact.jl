using DrWatson: datadir

# =============================================================================
# dad_5d — Loyola (IGABEM-bulk) Cattaneo two-cylinder geometry
#
# Source: Desktop/IGABEM-bulk/dad.jl → dad_5d()
#   R=70, w=6.5, a≈1.186, E=73.4e3, ν=0.33, μ=0.3
#   cargav=100 on upper top  →  P = 2*w*cargav = 1300
#   cargah=15  (bulk horizontal, fretting — apply_dad5d_bulk_tx!)
#   Lower bottom fixed; contact = circular arcs (split at ±a)
# =============================================================================

"""Loyola / Contato sapatas parameters (dad_5d)."""
function loyola_dad5d_params()
    w = 6.5
    R = 70.0
    θ = asin(w / R)
    xchord = R * cos(θ)
    y = R - xchord
    a = 1.1860170907699792
    ya = 0.010048125031531185
    E = 73.4e3
    ν = 0.33
    μ = 0.3
    cargav = 100.0
    cargah = 15.0
    P = 2 * w * cargav
    East = E / (1 - ν^2)
    a_H = sqrt(4 * R * P / (π * East))
    p0_H = 2 * P / (π * a_H)
    return (; w, R, θ, y, a, ya, E, ν, μ, cargav, cargah, P, East, a_H, p0_H)
end

"""
    mesh_dad_5d_body(which; ndiv_c, ndiv_f, ndiv_s, ndiv_top, μ, nome)

Mesh one cylinder body from dad_5d. `which = :upper | :lower`.
"""
function mesh_dad_5d_body(which::Symbol;
        ndiv_c::Int=12,
        ndiv_f::Int=6,
        ndiv_s::Int=6,
        ndiv_top::Int=12,
        μ::Real=0.3,
        cargav::Real=100.0,
        nome::AbstractString="dad5d",
        show::Bool=false,
        ordem::Int=1,
    )
    par = loyola_dad5d_params()
    w, R, y, a, ya = par.w, par.R, par.y, par.a, par.ya
    μs = string(float(μ))
    cvs = string(-float(cargav))

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(nome)
    lc = 2w / max(ndiv_c + 2 * ndiv_f, 8)

    if which === :upper
        c = gmsh.model.geo.addPoint(0.0, R, 0.0, lc)
        p1 = gmsh.model.geo.addPoint(-w, y, 0.0, lc)
        p2 = gmsh.model.geo.addPoint(-a, ya, 0.0, lc)
        p3 = gmsh.model.geo.addPoint(a, ya, 0.0, lc)
        p4 = gmsh.model.geo.addPoint(w, y, 0.0, lc)
        p5 = gmsh.model.geo.addPoint(w, 2.5 * w, 0.0, lc)
        p6 = gmsh.model.geo.addPoint(-w, 2.5 * w, 0.0, lc)

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

        # numeric type;value only (parse_pairs); 8;id uniqueness
        gmsh.model.addPhysicalGroup(1, [a1], -1, "4;" * μs * ";4;" * μs * ";8;11")
        gmsh.model.addPhysicalGroup(1, [a2], -1, "4;" * μs * ";4;" * μs * ";8;12")
        gmsh.model.addPhysicalGroup(1, [a3], -1, "4;" * μs * ";4;" * μs * ";8;13")
        gmsh.model.addPhysicalGroup(1, [s4], -1, "1;0;1;0;8;14")
        gmsh.model.addPhysicalGroup(1, [s5], -1, "1;0;1;" * cvs * ";8;15")  # ty=-cargav
        gmsh.model.addPhysicalGroup(1, [s6], -1, "1;0;1;0;8;16")
        gmsh.model.addPhysicalGroup(2, [surf], -1, "Domain")

    elseif which === :lower
        c = gmsh.model.geo.addPoint(0.0, -R, 0.0, lc)
        p7  = gmsh.model.geo.addPoint(w, -y, 0.0, lc)
        p8  = gmsh.model.geo.addPoint(a, -ya, 0.0, lc)
        p9  = gmsh.model.geo.addPoint(-a, -ya, 0.0, lc)
        p10 = gmsh.model.geo.addPoint(-w, -y, 0.0, lc)
        p11 = gmsh.model.geo.addPoint(-w, -2.5 * w, 0.0, lc)
        p12 = gmsh.model.geo.addPoint(w, -2.5 * w, 0.0, lc)

        a7 = gmsh.model.geo.addCircleArc(p7, c, p8)
        a8 = gmsh.model.geo.addCircleArc(p8, c, p9)
        a9 = gmsh.model.geo.addCircleArc(p9, c, p10)
        s10 = gmsh.model.geo.addLine(p10, p11)
        s11 = gmsh.model.geo.addLine(p11, p12)
        s12 = gmsh.model.geo.addLine(p12, p7)
        cl = gmsh.model.geo.addCurveLoop([a7, a8, a9, s10, s11, s12])
        surf = gmsh.model.geo.addPlaneSurface([cl])
        gmsh.model.geo.synchronize()

        gmsh.model.mesh.setTransfiniteCurve(a7, ndiv_f)
        gmsh.model.mesh.setTransfiniteCurve(a8, ndiv_c)
        gmsh.model.mesh.setTransfiniteCurve(a9, ndiv_f)
        gmsh.model.mesh.setTransfiniteCurve(s10, ndiv_s)
        gmsh.model.mesh.setTransfiniteCurve(s11, ndiv_top)
        gmsh.model.mesh.setTransfiniteCurve(s12, ndiv_s)

        gmsh.model.addPhysicalGroup(1, [a7], -1, "4;" * μs * ";4;" * μs * ";8;21")
        gmsh.model.addPhysicalGroup(1, [a8], -1, "4;" * μs * ";4;" * μs * ";8;22")
        gmsh.model.addPhysicalGroup(1, [a9], -1, "4;" * μs * ";4;" * μs * ";8;23")
        gmsh.model.addPhysicalGroup(1, [s10], -1, "1;0;1;0;8;24")
        gmsh.model.addPhysicalGroup(1, [s11], -1, "0;0;0;0;8;25")
        gmsh.model.addPhysicalGroup(1, [s12], -1, "1;0;1;0;8;26")
        gmsh.model.addPhysicalGroup(2, [surf], -1, "Domain")
    else
        gmsh.finalize()
        throw(ArgumentError("which must be :upper or :lower"))
    end

    # Raise order *before* write, while the circle CAD is still live.
    # format2d(tipo=2) on a linear .msh only midpoints the chords — 3 GL
    # collocation points then share one n and one gap0, and tn sawtooths.
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
    load_dad_5d_contact(props=nothing; ndiv_c=12, μ=0.3, ...) -> (prob, par)

Build MultiRegionProblem matching Loyola dad_5d (two circular cylinders).
Region 1 = upper, region 2 = lower. NTN contact on arcs.
"""
function load_dad_5d_contact(
        props::Union{Nothing,Elasticity}=nothing;
        ndiv_c::Int=12,
        ndiv_f::Int=6,
        ndiv_s::Int=6,
        ndiv_top::Int=12,
        μ::Union{Nothing,Real}=nothing,
        nome::AbstractString="dad5d",
        show::Bool=false,
        pair_method::Symbol=:ntn,
        tipo::Int=2,
        collocation::Symbol=:legendre,
        gap::Symbol=:normal,
    )
    par = loyola_dad5d_params()
    μv = μ === nothing ? par.μ : float(μ)
    if props === nothing
        props = Elasticity(par.E, par.ν, 1.0; plane_strain=true)
    end

    msh_u = mesh_dad_5d_body(:upper; ndiv_c=ndiv_c, ndiv_f=ndiv_f, ndiv_s=ndiv_s,
        ndiv_top=ndiv_top, μ=μv, cargav=par.cargav, nome=nome * "_upper", show=false,
        ordem=tipo)
    msh_l = mesh_dad_5d_body(:lower; ndiv_c=ndiv_c, ndiv_f=ndiv_f, ndiv_s=ndiv_s,
        ndiv_top=ndiv_top, μ=μv, cargav=par.cargav, nome=nome * "_lower", show=show,
        ordem=tipo)

    # tipo=2 → quadratic discontinuous (3 Gauss colloc./elem), Contato default
    dad_u = format2d(msh_u, props; pontointerno=false, tipo=tipo, collocation=collocation)
    dad_l = format2d(msh_l, props; pontointerno=false, tipo=tipo, collocation=collocation)
    dad_u.name = "upper"
    dad_l.name = "lower"

    prob = MultiRegionProblem([dad_u, dad_l]; name="dad_5d_contact")
    pair_contacts!(prob; method=pair_method, slave_reg=1, master_reg=2, gap=gap)
    for cp in prob.contacts
        cp.μ = μv
        cp.ut_lock = 0.0
        cp.state = 1
    end
    return prob, par
end

"""
    apply_dad5d_bulk_tx!(prob, tx; cargav=100)

Set upper far-face traction `(tx, ty=-cargav)`.

!!! warning
    Pure Neumann fretting (`tx=cargah` on the tall top face) applies a large
    overturning moment and **tips** the upper cylinder — the contact patch
    migrates off-center. Prefer [`apply_dad5d_bulk_ux!`](@ref) for fretting.
"""
function apply_dad5d_bulk_tx!(prob, tx::Real; cargav::Real=100.0)
    dad = prob.regions[1]
    ymax = maximum(pt[2] for pt in dad.Nodes)
    thr = 1e-9 * max(abs(ymax), 1.0)
    tx = float(tx); ty = -float(cargav)
    nset = 0
    for i in 1:dad.n
        if abs(dad.Nodes[i][2] - ymax) <= thr
            dad.BC[2i - 1] = 1; dad.BV[2i - 1] = tx
            dad.BC[2i]     = 1; dad.BV[2i]     = ty
            nset += 1
        end
    end
    return nset
end

"""
    apply_dad5d_bulk_ux!(prob, ux; uy)

Fretting bulk load on the upper far face as **full Dirichlet**
``(u_x, u_y) = (ux, uy)``.

Typical path:
1. Step A force-controlled (`ty=-cargav`) → record mean top ``u_y``
2. B–E: `apply_dad5d_bulk_ux!(prob, ux; uy=uyA)` with ramped `ux`

Do **not** use pure Neumann `tx` on the tall top face — overturning moment
tips the cylinder. Mixed `(ux, ty)` leaves rotation free and kills net ``Q``.
"""
function apply_dad5d_bulk_ux!(prob, ux::Real; uy::Real)
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

"""Mean global ``(u_x, u_y)`` on the upper far face (after scatter)."""
function dad5d_top_u_mean(prob)
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

"""Set friction coefficient on every contact pair (IGABEM `milocal`)."""
function set_contact_mu!(prob, μ::Real)
    μv = float(μ)
    for cp in prob.contacts
        cp.μ = μv
    end
    return μv
end

"""Reset incremental stick locks (call after frictionless step A)."""
function reset_contact_ut_locks!(prob)
    for cp in prob.contacts
        cp.ut_lock = 0.0
    end
    return nothing
end

"""
    apply_dad5d_cargah!(prob, cargah; cargav=100, faces=:upper_sides)

IGABEM-bulk fretting bulk load: horizontal traction `cargah` on the upper
vertical sides (segments 4 & 6 of dad_5d), keeping top `ty=-cargav`.

IGABEM `Elastico_bulk.jl` applies `CDC = [1, cargah/npassos, …]` on side curves
after the frictionless normal steps (`milocal=0` → `milocal=mi`).
"""
function apply_dad5d_cargah!(prob, cargah::Real; cargav::Real=100.0,
        faces::Symbol=:upper_sides)
    dad = prob.regions[1]
    xs = [pt[1] for pt in dad.Nodes]
    ys = [pt[2] for pt in dad.Nodes]
    xmin, xmax = extrema(xs)
    ymax = maximum(ys)
    thr = 1e-9 * max(abs(xmax - xmin), abs(ymax), 1.0)
    tx = float(cargah); ty_top = -float(cargav)
    nside = ntop = 0
    for i in 1:dad.n
        x, y = dad.Nodes[i]
        on_top = abs(y - ymax) <= thr
        on_R = abs(x - xmax) <= thr
        on_L = abs(x - xmin) <= thr
        if on_top
            dad.BC[2i - 1] = 1; dad.BV[2i - 1] = 0.0
            dad.BC[2i]     = 1; dad.BV[2i]     = ty_top
            ntop += 1
        elseif faces === :upper_sides && (on_R || on_L)
            # same-sign tx on both sides → net Fx = tx * (height_R + height_L)
            dad.BC[2i - 1] = 1; dad.BV[2i - 1] = tx
            dad.BC[2i]     = 1; dad.BV[2i]     = 0.0
            nside += 1
        elseif faces === :upper_right && on_R
            dad.BC[2i - 1] = 1; dad.BV[2i - 1] = tx
            dad.BC[2i]     = 1; dad.BV[2i]     = 0.0
            nside += 1
        end
    end
    return (; nside, ntop)
end
