# Mesh generators for classic 2D Laplace benchmarks
# (ported from BEM.jl atual `data/dadpotencial.jl` / `scripts/potencial/potencial_direto.jl`)
#
# Physical-group BC strings use package convention:
#   "0;T"  → Dirichlet  T = value
#   "1;q"  → Neumann    q = value   with  q = -k ∂T/∂n

"""
    potencial1d_mesh(; ndiv=16, ordem=1, nome="potencial1d", show=false) -> path

Unit square with BCs for exact ``T = x`` (k=1):
left Dirichlet 0, right Neumann q=-1, top/bottom insulated.
Same as `quadrado` defaults.
"""
function potencial1d_mesh(; ndiv=16, ordem=1, Lx=1.0, Ly=1.0, nome="potencial1d", show=false)
    return quadrado(; nome=nome, Lx=Lx, Ly=Ly, ordem=ordem, ndiv=ndiv, show=show)
end

"""
    laquini1_mesh(; ndiv=16, ordem=1, nome="laquini1", show=false)

Unit square — Laquini problem 1:
T=0 on bottom/right/left, q=-1 on top.
"""
function laquini1_mesh(; ndiv=16, ordem=1, nome="laquini1", show=false)
    return _square_mesh_bc(;
        nome=nome, ndiv=ndiv, ordem=ordem, show=show,
        bottom="0;0", right="0;0", top="1;-1", left="0;0",
    )
end

"""
    laquini2_mesh(; ndiv=16, ordem=1, nome="laquini2", show=false)

Unit square — Laquini problem 2:
T=0 on bottom/left, q=-1 on right and top.
"""
function laquini2_mesh(; ndiv=16, ordem=1, nome="laquini2", show=false)
    return _square_mesh_bc(;
        nome=nome, ndiv=ndiv, ordem=ordem, show=show,
        bottom="0;0", right="1;-1", top="1;-1", left="0;0",
    )
end

"""
    laquini3_mesh(; ndiv=16, ordem=1, nome="laquini3", show=false)

Unit square — Laquini problem 3:
T=0 on bottom/right/left, T=1 on top.
"""
function laquini3_mesh(; ndiv=16, ordem=1, nome="laquini3", show=false)
    return _square_mesh_bc(;
        nome=nome, ndiv=ndiv, ordem=ordem, show=show,
        bottom="0;0", right="0;0", top="0;1", left="0;0",
    )
end

"""
    quarto_circ_mesh(; ndiv=12, ordem=1, ri=1.0, re=2.0, ti=100.0, qe=-200.0,
                     nome="quarto_circ", show=false)

Quarter annulus ``ri ≤ r ≤ re``, θ ∈ [0, π/2].

BCs (k=1):
- inner arc: Dirichlet ``T = ti``
- outer arc: Neumann ``q = qe``
- straight radial edges: insulated ``q = 0``

Exact: ``T = ti - qe·re·log(r/ri)``.
"""
function quarto_circ_mesh(;
    ndiv=12,
    ordem=1,
    ri=1.0,
    re=2.0,
    ti=100.0,
    qe=-200.0,
    nome="quarto_circ",
    show=false,
)
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(nome)
    lc = (re - ri) / max(ndiv, 4)

    # points: bottom-inner, bottom-outer, top-outer, top-inner, centre
    c = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, lc)
    p1 = gmsh.model.geo.addPoint(ri, 0.0, 0.0, lc)   # inner, θ=0
    p2 = gmsh.model.geo.addPoint(re, 0.0, 0.0, lc)   # outer, θ=0
    p3 = gmsh.model.geo.addPoint(0.0, re, 0.0, lc)   # outer, θ=π/2
    p4 = gmsh.model.geo.addPoint(0.0, ri, 0.0, lc)   # inner, θ=π/2

    l_bot = gmsh.model.geo.addLine(p1, p2)                 # radial bottom
    a_out = gmsh.model.geo.addCircleArc(p2, c, p3)         # outer arc
    l_left = gmsh.model.geo.addLine(p3, p4)                # radial left (top→inner)
    a_in = gmsh.model.geo.addCircleArc(p4, c, p1)          # inner arc (CW from p4 to p1)

    cl = gmsh.model.geo.addCurveLoop([l_bot, a_out, l_left, a_in])
    s = gmsh.model.geo.addPlaneSurface([cl])
    gmsh.model.geo.synchronize()

    nθ = ndiv
    nr = max(ndiv ÷ 2, 4)
    gmsh.model.mesh.setTransfiniteCurve(l_bot, nr)
    gmsh.model.mesh.setTransfiniteCurve(a_out, nθ)
    gmsh.model.mesh.setTransfiniteCurve(l_left, nr)
    gmsh.model.mesh.setTransfiniteCurve(a_in, nθ)
    gmsh.model.mesh.setTransfiniteSurface(s)
    gmsh.model.mesh.setRecombine(2, s)

    gmsh.model.addPhysicalGroup(1, [l_bot, l_left], -1, "1;0")          # insulated
    gmsh.model.addPhysicalGroup(1, [a_out], -1, "1;$qe")                # outer flux
    gmsh.model.addPhysicalGroup(1, [a_in], -1, "0;$ti")                 # inner T
    gmsh.model.addPhysicalGroup(2, [s], -1, "Domain")

    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(ordem)

    out = datadir("Laplace", nome * ".msh")
    mkpath(dirname(out))
    gmsh.write(out)
    show && gmsh.fltk.run()
    gmsh.finalize()
    return out
end

"""
    placa_moulton_mesh(; ndiv=12, ordem=1, nome="placa_moulton", show=false)

Moulton plate: rectangle ``[-1,1]×[0,1]`` with crack-tip field
``T = √r cos(θ/2)`` (θ = atan2(y,x)).

Mesh BC tags (placeholders — overwrite with analytical after `format2d`):
- bottom-left edge ``[-1,0]→[0,0]``: Dirichlet 0 (exact T=0 on the cut)
- remaining edges: Neumann 0 (filled from `ana_moulton` after load)
"""
function placa_moulton_mesh(; ndiv=12, ordem=1, nome="placa_moulton", show=false)
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(nome)
    lc = 0.1

    # (-1,0) (0,0) (1,0) (1,1) (-1,1)
    p1 = gmsh.model.geo.addPoint(-1.0, 0.0, 0.0, lc)
    p2 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, lc)
    p3 = gmsh.model.geo.addPoint(1.0, 0.0, 0.0, lc)
    p4 = gmsh.model.geo.addPoint(1.0, 1.0, 0.0, lc)
    p5 = gmsh.model.geo.addPoint(-1.0, 1.0, 0.0, lc)

    l1 = gmsh.model.geo.addLine(p1, p2)  # bottom-left (Dirichlet cut)
    l2 = gmsh.model.geo.addLine(p2, p3)  # bottom-right
    l3 = gmsh.model.geo.addLine(p3, p4)  # right
    l4 = gmsh.model.geo.addLine(p4, p5)  # top
    l5 = gmsh.model.geo.addLine(p5, p1)  # left

    cl = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4, l5])
    s = gmsh.model.geo.addPlaneSurface([cl])
    gmsh.model.geo.synchronize()

    n_half = max(ndiv ÷ 2, 3)
    gmsh.model.mesh.setTransfiniteCurve(l1, n_half)
    gmsh.model.mesh.setTransfiniteCurve(l2, n_half)
    gmsh.model.mesh.setTransfiniteCurve(l3, ndiv)
    gmsh.model.mesh.setTransfiniteCurve(l4, ndiv)
    gmsh.model.mesh.setTransfiniteCurve(l5, ndiv)
    # not fully transfinite (5 sides) — free mesh with size from points
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 2 / ndiv)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.5 / ndiv)

    gmsh.model.addPhysicalGroup(1, [l1], -1, "0;0")           # Dirichlet on cut
    gmsh.model.addPhysicalGroup(1, [l2, l3, l4, l5], -1, "1;0") # Neumann placeholder
    gmsh.model.addPhysicalGroup(2, [s], -1, "Domain")

    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(ordem)

    out = datadir("Laplace", nome * ".msh")
    mkpath(dirname(out))
    gmsh.write(out)
    show && gmsh.fltk.run()
    gmsh.finalize()
    return out
end

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

function _square_mesh_bc(;
    nome="square",
    Lx=1.0,
    Ly=1.0,
    x0=0.0,
    y0=0.0,
    ndiv=16,
    ordem=1,
    show=false,
    bottom="1;0",
    right="1;-1",
    top="1;0",
    left="0;0",
)
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(nome)
    lc = 0.1
    p1 = gmsh.model.geo.addPoint(x0, y0, 0.0, lc)
    p2 = gmsh.model.geo.addPoint(x0 + Lx, y0, 0.0, lc)
    p3 = gmsh.model.geo.addPoint(x0 + Lx, y0 + Ly, 0.0, lc)
    p4 = gmsh.model.geo.addPoint(x0, y0 + Ly, 0.0, lc)
    l1 = gmsh.model.geo.addLine(p1, p2)  # bottom
    l2 = gmsh.model.geo.addLine(p2, p3)  # right
    l3 = gmsh.model.geo.addLine(p3, p4)  # top
    l4 = gmsh.model.geo.addLine(p4, p1)  # left
    cl = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    s = gmsh.model.geo.addPlaneSurface([cl])
    gmsh.model.geo.synchronize()
    for l in (l1, l2, l3, l4)
        gmsh.model.mesh.setTransfiniteCurve(l, ndiv)
    end
    gmsh.model.mesh.setTransfiniteSurface(s)
    gmsh.model.mesh.setRecombine(2, s)

    # group edges that share the same BC string
    groups = Dict{String,Vector{Int}}()
    for (l, tag) in ((l1, bottom), (l2, right), (l3, top), (l4, left))
        push!(get!(groups, tag, Int[]), l)
    end
    for (tag, curves) in groups
        gmsh.model.addPhysicalGroup(1, curves, -1, tag)
    end
    gmsh.model.addPhysicalGroup(2, [s], -1, "Domain")

    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(ordem)

    out = datadir("Laplace", nome * ".msh")
    mkpath(dirname(out))
    gmsh.write(out)
    show && gmsh.fltk.run()
    gmsh.finalize()
    return out
end

"""
    apply_bc_keep_type!(dad, ana::AnalyticalSolution)

Keep mesh BC types (`dad.BC`) and fill `dad.BV` from the analytical field:
Dirichlet → `ana.u`, Neumann → `ana.q` (requires `ana.q`).
"""
function apply_bc_keep_type!(dad::BEMdata{<:Laplace}, ana::AnalyticalSolution)
    for i in 1:dad.n
        if dad.BC[i] == 0
            dad.BV[i] = float(ana.u(dad.Nodes[i]))
        else
            ana.q === nothing && error("analytical flux `q` required for Neumann node $i")
            dad.BV[i] = float(ana.q(dad.Nodes[i], dad.Normal[i]))
        end
    end
    attach_analytical!(dad, ana)
    return dad
end
