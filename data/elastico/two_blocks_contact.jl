# Two elastic blocks with a horizontal contact joint (BC type 4).
# Bottom: [0,W]×[0,H]     top face = contact
# Top:    [0,W]×[H+g, 2H+g]  bottom face = contact
#
# Used for multibody NTN / NTS contact tests and fretting (bulk u_x).

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
    load_two_blocks_contact(props; W, H, gap, μ, ndiv_bot, ndiv_top, fretting=false, ...)

Build a [`MultiRegionProblem`](@ref) with bottom + top elastic blocks.

- Bottom: base fully fixed, upper face = contact (type 4), sides free
- Top: far face Dirichlet (default fixed), lower face = contact (type 4)
- Top sides: rollers `ux=0` by default; with `fretting=true` sides are
  traction-free so bulk far-face `u_x` can drive Cattaneo slip
  (**rollers pin ux=0 and kill fretting shear — never use them for fretting**)

Contact is closed by the rigid approach `δ` (gap ← gap0 − δ).
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
        fretting::Bool=false,
        uy_top=nothing,
    )
    uy_top === nothing || @warn "uy_top is ignored; pass δ to solve_multibody_elasticity_contact!" uy_top

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
    # fretting: free sides; compression tests: rollers kill RBM
    side = fretting ? "1;0;1;0" : "0;0;1;0"
    msh_t = mesh_elastic_block(;
        x0=0.0, y0=H + gap, W=W, H=H,
        ndiv_x=ndiv_top, ndiv_y=ndiv_y, μ=μ,
        bottom_bc="4;$μ;4;$μ;9;1",
        top_bc="0;0;0;0;9;2",
        left_bc="$(side);9;3",
        right_bc="$(side);9;4",
        nome=nome * "_top",
        show=show,
    )
    dad_b = format2d(msh_b, props; pontointerno=false)
    dad_t = format2d(msh_t, props; pontointerno=false)
    dad_b.name = "bottom"
    dad_t.name = "top"
    return MultiRegionProblem([dad_b, dad_t]; name="two_blocks_contact")
end

"""
    apply_parabolic_contact_gap!(prob; R, gap_min=0.0, x0=nothing, method=:ntn)

Overwrite type-4 pair gaps with a cylinder-on-flat profile
``g₀(x) = gap_min + (x - x₀)² / (2R)`` after pairing.
"""
function apply_parabolic_contact_gap!(prob;
        R::Real,
        gap_min::Real=0.0,
        x0=nothing,
        method::Symbol=:ntn,
        slave_reg::Int=1,
        master_reg::Int=2)
    isempty(prob.contacts) && pair_contacts!(prob; method=method,
        slave_reg=slave_reg, master_reg=master_reg)
    R = float(R)
    R > 0 || throw(ArgumentError("R > 0 required"))
    xs = Float64[]
    for cp in prob.contacts
        dad = prob.regions[cp.reg_a]
        push!(xs, dad.Nodes[cp.node_a][1])
    end
    xc = x0 === nothing ? sum(xs) / max(length(xs), 1) : float(x0)
    for cp in prob.contacts
        dad = prob.regions[cp.reg_a]
        x = dad.Nodes[cp.node_a][1]
        cp.gap0 = float(gap_min) + (x - xc)^2 / (2R)
        cp.state = 1
        cp.ut_lock = 0.0
    end
    return prob
end

"""
    contact_interface_xyτ(prob) -> (; x, tn, tt, state)

Sample slave-side contact tractions ordered by arc coordinate (for plotting).
"""
function contact_interface_xyτ(prob)
    pairs = prob.contacts
    isempty(pairs) && return (; x=Float64[], tn=Float64[], tt=Float64[], state=Int[])
    dad = prob.regions[pairs[1].reg_a]
    x = [dad.Nodes[cp.node_a][1] for cp in pairs]
    tn = [cp.tn for cp in pairs]
    tt = [cp.tt for cp in pairs]
    st = [cp.state for cp in pairs]
    perm = sortperm(x)
    return (; x=x[perm], tn=tn[perm], tt=tt[perm], state=st[perm])
end
