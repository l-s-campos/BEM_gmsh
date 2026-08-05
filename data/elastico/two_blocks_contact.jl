# Two elastic blocks with a horizontal contact joint (BC type 4).
# Bottom: [0,W]×[0,H]     top face = contact
# Top:    [0,W]×[H+g, 2H+g]  bottom face = contact
#
# Used for multibody NTN / NTS contact tests.

"""
    mesh_elastic_block(; x0, y0, W, H, ndiv_x, ndiv_y, μ, top_bc, bottom_bc, nome)

Rectangle boundary mesh with elasticity physical names `tx;vx;ty;vy`.
Contact faces use type 4 with friction `μ`.
"""
function mesh_elastic_block(;
        x0=0.0, y0=0.0, W=1.0, H=0.5,
        ndiv_x=6, ndiv_y=4,
        μ=0.3,
        bottom_bc::AbstractString = "0;0;0;0",
        top_bc::AbstractString = "4;$μ;4;$μ",
        left_bc::AbstractString = "1;0;1;0",
        right_bc::AbstractString = "1;0;1;0",
        nome="elast_block",
        show=false,
    )
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(nome)
    lc = min(W, H) / max(ndiv_x, ndiv_y)

    p1 = gmsh.model.geo.addPoint(x0,     y0,     0.0, lc)
    p2 = gmsh.model.geo.addPoint(x0 + W, y0,     0.0, lc)
    p3 = gmsh.model.geo.addPoint(x0 + W, y0 + H, 0.0, lc)
    p4 = gmsh.model.geo.addPoint(x0,     y0 + H, 0.0, lc)
    lbot = gmsh.model.geo.addLine(p1, p2)
    lright = gmsh.model.geo.addLine(p2, p3)
    ltop = gmsh.model.geo.addLine(p3, p4)
    lleft = gmsh.model.geo.addLine(p4, p1)
    cl = gmsh.model.geo.addCurveLoop([lbot, lright, ltop, lleft])
    s = gmsh.model.geo.addPlaneSurface([cl])
    gmsh.model.geo.synchronize()

    gmsh.model.mesh.setTransfiniteCurve(lbot, ndiv_x)
    gmsh.model.mesh.setTransfiniteCurve(ltop, ndiv_x)
    gmsh.model.mesh.setTransfiniteCurve(lleft, ndiv_y)
    gmsh.model.mesh.setTransfiniteCurve(lright, ndiv_y)
    gmsh.model.mesh.setTransfiniteSurface(s)
    gmsh.model.mesh.setRecombine(2, s)

    gmsh.model.addPhysicalGroup(1, [lbot], -1, bottom_bc)
    gmsh.model.addPhysicalGroup(1, [ltop], -1, top_bc)
    gmsh.model.addPhysicalGroup(1, [lleft], -1, left_bc)
    gmsh.model.addPhysicalGroup(1, [lright], -1, right_bc)
    gmsh.model.addPhysicalGroup(2, [s], -1, "Domain")
    gmsh.model.mesh.generate(2)

    out = datadir("elastico", nome * ".msh")
    mkpath(dirname(out))
    gmsh.write(out)
    show && gmsh.fltk.run()
    gmsh.finalize()
    return out
end

"""
    load_two_blocks_contact(props; W, H, gap, μ, ndiv_bot, ndiv_top, ...)

Build a [`MultiRegionProblem`](@ref) with bottom + top elastic blocks.

- Bottom block: base fully fixed, upper face = contact (type 4)
- Top block: upper face fully fixed, lower face = contact (type 4)
- Sides: rollers `ux=0` on the top block; bottom sides free

Contact is closed by the rigid approach `δ` in
[`solve_multibody_elasticity_contact!`](@ref) (gap ← gap0 − δ), not by a far-field
Dirichlet push. That yields a stable staggered penalty fixed-point.

Use `ndiv_bot != ndiv_top` for non-matching NTS tests.

Gmsh physical names must be unique per curve — trailing `;id;0` pairs are
ignored by elasticity (`dof=2`) but keep names distinct.
"""
function load_two_blocks_contact(
        props::Elasticity;
        W=1.0,
        H=0.5,
        gap=0.02,
        μ=0.3,
        ndiv_bot=6,
        ndiv_top=6,
        ndiv_y=3,
        nome="two_blocks",
        show=false,
        # kept for API compat; ignored (use solver `δ` instead)
        uy_top=nothing,
    )
    uy_top === nothing || @warn "uy_top is ignored; pass δ to solve_multibody_elasticity_contact!" uy_top

    # Bottom: fully fixed base (kills 3 rigid modes). Sides traction-free.
    msh_b = mesh_elastic_block(;
        x0=0.0, y0=0.0, W=W, H=H,
        ndiv_x=ndiv_bot, ndiv_y=ndiv_y, μ=μ,
        bottom_bc="0;0;0;0;8;1",
        top_bc="4;$μ;4;$μ;8;2",
        left_bc="1;0;1;0;8;3",
        right_bc="1;0;1;0;8;4",
        nome=nome * "_bot",
        show=false,
    )
    # Top: fixed far face; rollers on sides; contact on bottom face.
    msh_t = mesh_elastic_block(;
        x0=0.0, y0=H + gap, W=W, H=H,
        ndiv_x=ndiv_top, ndiv_y=ndiv_y, μ=μ,
        bottom_bc="4;$μ;4;$μ;9;1",
        top_bc="0;0;0;0;9;2",      # fixed support
        left_bc="0;0;1;0;9;3",     # ux = 0 roller
        right_bc="0;0;1;0;9;4",
        nome=nome * "_top",
        show=show,
    )
    dad_b = format2d(msh_b, props; pontointerno=false)
    dad_t = format2d(msh_t, props; pontointerno=false)
    dad_b.name = "bottom"
    dad_t.name = "top"
    return MultiRegionProblem([dad_b, dad_t]; name="two_blocks_contact")
end
