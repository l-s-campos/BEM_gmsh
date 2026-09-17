# Finite-width Laplace crack — diamond (double-V) hole, standard CBIE, sinh

"""
    mesh_finite_width_crack(; W=5, H=10, a=1, gap=0.05, field=:y, ...) -> path

Flattened **diamond** (two V faces) of opening `gap` (`δ`) and tip-to-tip
length `2a`. Faces meet at the tips — no dummy end caps.

The hole loop is **clockwise** so `tan2normal` (90° CW of the tangent)
points **into the hole** (outward from the solid). The previous rectangle
walked CCW around the hole, so `n` pointed into the solid and the slit
did not act as an insulated cavity.

```
              (0, δ/2)
              /      \\
        (−a,0)        (a,0)     CW: L → D → R → U → L
              \\      /
              (0,−δ/2)
```

`field=:y` → insulated `q=0` on the diamond; outer `T=±H`.
`field=:x` → conducting `T=0` on the diamond; outer `T=±W`.
"""
function mesh_finite_width_crack(; W=5.0, H=10.0, a=1.0, gap=0.05,
        ndiv_b=10, ndiv_h=16, ndiv_crack=16, n_cap=3,
        field::Symbol=:y, ordem=1, nome="finite_width_crack", show=false,
        physics::Symbol=:laplace, σ=1.0)
    gap > 0 || error("mesh_finite_width_crack: gap must be positive, got $gap")
    gap < 1e-12 && error("mesh_finite_width_crack: gap=$gap is below 1e-12; use dual BEM")
    B = parentmodule(@__MODULE__)
    gmsh = B.gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(nome)
    lc = min(W, H) / max(ndiv_b, 8)
    δ2 = gap / 2
    lc_c = min(lc / 2, 2a / max(ndiv_crack, 2), gap)

    p1 = gmsh.model.geo.addPoint(-W, -H, 0, lc)
    p2 = gmsh.model.geo.addPoint(W, -H, 0, lc)
    p3 = gmsh.model.geo.addPoint(W, H, 0, lc)
    p4 = gmsh.model.geo.addPoint(-W, H, 0, lc)

    pL = gmsh.model.geo.addPoint(-a, 0, 0, lc_c)
    pR = gmsh.model.geo.addPoint(a, 0, 0, lc_c)
    pU = gmsh.model.geo.addPoint(0, δ2, 0, lc_c)
    pD = gmsh.model.geo.addPoint(0, -δ2, 0, lc_c)

    lb = gmsh.model.geo.addLine(p1, p2)
    lr = gmsh.model.geo.addLine(p2, p3)
    lt = gmsh.model.geo.addLine(p3, p4)
    ll = gmsh.model.geo.addLine(p4, p1)

    # Clockwise hole: solid on the right → n into the opening.
    eLD = gmsh.model.geo.addLine(pL, pD)
    eDR = gmsh.model.geo.addLine(pD, pR)
    eRU = gmsh.model.geo.addLine(pR, pU)
    eUL = gmsh.model.geo.addLine(pU, pL)
    slit = [eLD, eDR, eRU, eUL]
    cl_outer = gmsh.model.geo.addCurveLoop([lb, lr, lt, ll])
    cl_hole = gmsh.model.geo.addCurveLoop(slit)
    s = gmsh.model.geo.addPlaneSurface([cl_outer, cl_hole])
    gmsh.model.geo.synchronize()

    gmsh.model.mesh.setTransfiniteCurve(lb, ndiv_b)
    gmsh.model.mesh.setTransfiniteCurve(lt, ndiv_b)
    gmsh.model.mesh.setTransfiniteCurve(lr, ndiv_h)
    gmsh.model.mesh.setTransfiniteCurve(ll, ndiv_h)
    nslit = max(4, Int(ndiv_crack))
    for e in slit
        gmsh.model.mesh.setTransfiniteCurve(e, nslit)
    end

    if physics === :elasticity
        gmsh.model.addPhysicalGroup(1, [lb], -1, "1;0;1;$(-σ)")
        gmsh.model.addPhysicalGroup(1, [lt], -1, "1;0;1;$σ")
        gmsh.model.addPhysicalGroup(1, [ll, lr], -1, "1;0;1;0")
        gmsh.model.addPhysicalGroup(1, slit, -1, "1;0;1;0")
    elseif field === :y
        gmsh.model.addPhysicalGroup(1, [lb], -1, "0;$(-H)")
        gmsh.model.addPhysicalGroup(1, [lt], -1, "0;$H")
        gmsh.model.addPhysicalGroup(1, [ll, lr], -1, "1;0")
        gmsh.model.addPhysicalGroup(1, slit, -1, "1;0")
    elseif field === :x
        gmsh.model.addPhysicalGroup(1, [ll], -1, "0;$(-W)")
        gmsh.model.addPhysicalGroup(1, [lr], -1, "0;$W")
        gmsh.model.addPhysicalGroup(1, [lb, lt], -1, "1;0")
        gmsh.model.addPhysicalGroup(1, slit, -1, "0;0")
    else
        error("mesh_finite_width_crack: field must be :x or :y, got $field")
    end
    gmsh.model.addPhysicalGroup(2, [s], -1, "Domain")

    gmsh.model.mesh.generate(2)
    ordem > 1 && gmsh.model.mesh.setOrder(ordem)
    sub = physics === :elasticity ? ("elastico", "iso") : ("Laplace",)
    out = B.datadir(sub..., nome * ".msh")
    mkpath(dirname(out))
    gmsh.write(out)
    show && gmsh.fltk.run()
    gmsh.finalize()
    return out
end

"""
    check_slit_bc(dad; a=1, gap=0.05) -> NamedTuple

Inspect BC type/value and normal orientation on the diamond opening.
Upper-face normals must satisfy `n_y < 0` (into the hole). Insulated:
`BC=1`, `BV=0`.
"""
function check_slit_bc(dad; a=1.0, gap=0.05)
    idx = findall(i -> abs(dad.Nodes[i][1]) ≤ a + 1e-9 &&
                       abs(dad.Nodes[i][2]) ≤ 0.6 * gap + 1e-9, 1:dad.n)
    n_into = 0
    n_out = 0
    @inbounds for i in idx
        p = dad.Nodes[i]
        n = dad.Normal[i]
        # vector from node toward the opening centre (0,0)
        to_hole = -p
        if abs(p[2]) < 1e-14
            # tip nodes: n should have a component toward x=0
            (sign(n[1]) == -sign(p[1]) || abs(p[1]) < 1e-14) ? (n_into += 1) : (n_out += 1)
        else
            # upper y>0 → into hole is −êy; lower y<0 → into hole is +êy
            want = p[2] > 0 ? n[2] < 0 : n[2] > 0
            want ? (n_into += 1) : (n_out += 1)
        end
    end
    bc = dad.BC[idx]
    bv = dad.BV[idx]
    return (; n=length(idx), bc, bv, n_into_hole=n_into, n_into_solid=n_out,
        insulated=all(==(1), bc) && all(iszero, bv))
end

"""True if collocation `i` sits on the diamond opening."""
function _on_slit(dad, i, a, gap)
    p = dad.Nodes[i]
    return abs(p[1]) ≤ a + 1e-9 && abs(p[2]) ≤ 0.6 * gap + 1e-9
end

"""
Flip slit elements whose `n` points into the solid (`n·x > 0`).
Gmsh may reverse the hole walk relative to the geo loop; the BIE must use
outward-from-solid normals (into the opening).
"""
function orient_slit_normals!(dad; a=1.0, gap=0.05)
    flipped = 0
    for el in dad.elements
        all(i -> _on_slit(dad, i, a, gap), el.index) || continue
        c = mean(dad.Nodes[i] for i in el.index)
        n̄ = mean(dad.Normal[i] for i in el.index)
        # origin is inside the diamond; n should satisfy n·c < 0 (into the hole)
        n̄ ⋅ c < 0 && continue
        reverse!(el.index)
        reverse!(el.Jacobian)
        @inbounds for i in el.index
            dad.Normal[i] = -dad.Normal[i]
        end
        flipped += 1
    end
    return flipped
end

"""Minimum distance from a collocation node to a non-owning element."""
function min_opposite_distance(dad)
    B = parentmodule(@__MODULE__)
    dmin = Inf
    poly = dad.element_type
    @inbounds for i in 1:dad.n
        pf = dad.Nodes[i]
        for el in dad.elements
            B._source_on_element(el, i) && continue
            xj = dad.Nodes[el.index]
            _, _, dist = B.closest_point_1d(poly, xj, pf; ξ0=B._seed_1d(poly, xj, pf))
            dmin = min(dmin, dist)
        end
    end
    return dmin
end

"""Recommended Gauss count for a slit of width `δ` and typical element length `L`."""
function npg_finite_width(δ, L)
    ratio = max(L / max(δ, 1e-14), 1.0)
    return max(20, round(Int, 12 + 6 * log10(ratio)))
end

"""
    assemble_finite_width_laplace!(dad; npg=nothing, gap=nothing, threaded=true)

Standard CBIE with topological on-element Guiggiani (never on the opposite
face). Default `npg` grows as `log10(L/δ)` when the gap is small.
"""
function assemble_finite_width_laplace!(dad; npg=nothing, gap=nothing, threaded::Bool=true)
    B = parentmodule(@__MODULE__)
    δ = gap === nothing ? min_opposite_distance(dad) : float(gap)
    Ltyp = mean(el.Length for el in dad.elements)
    n = npg === nothing ? npg_finite_width(δ, Ltyp) : Int(npg)
    return B.H_G_full_direct(dad; npg=n, threaded=threaded)
end

"""
    finite_width_laplace_problem(; gap=0.05, field=:y, k=1, kwargs...) -> dad

Mesh + `format2d` for the sharp-slit geometry.
"""
function finite_width_laplace_problem(; W=5.0, H=10.0, a=1.0, gap=0.05, k=1.0,
        ndiv_b=10, ndiv_h=16, ndiv_crack=16, n_cap=3,
        field::Symbol=:y, ordem=1, nome="finite_width_crack", pontointerno=false)
    B = parentmodule(@__MODULE__)
    msh = mesh_finite_width_crack(; W, H, a, gap, ndiv_b, ndiv_h, ndiv_crack, n_cap,
        field, ordem, nome, show=false)
    dad = B.format2d(msh, B.Laplace(k); tipo=ordem, pontointerno=pontointerno)
    orient_slit_normals!(dad; a, gap)
    return dad
end

"""
    assemble_finite_width_elasticity!(dad; npg=nothing, gap=nothing, threaded=true)

Standard Kelvin CBIE with topological on-element Guiggiani (never on the
opposite diamond face). Default `npg` grows as `log10(L/δ)`.
"""
function assemble_finite_width_elasticity!(dad; npg=nothing, gap=nothing,
        threaded::Bool=true)
    B = parentmodule(@__MODULE__)
    δ = gap === nothing ? min_opposite_distance(dad) : float(gap)
    Ltyp = mean(el.Length for el in dad.elements)
    n = npg === nothing ? npg_finite_width(δ, Ltyp) : Int(npg)
    return B.H_G_full_direct(dad; npg=n, threaded=threaded)
end

"""
    finite_width_elasticity_problem(; gap=0.05, E=3000, ν=0.2, σ=1, ...) -> dad

Diamond slit + `format2d` + inward-hole normals + rigid-body pins.
"""
function finite_width_elasticity_problem(; W=5.0, H=10.0, a=1.0, gap=0.05,
        E=3000.0, ν=0.2, σ=1.0, ndiv_b=10, ndiv_h=16, ndiv_crack=16, n_cap=3,
        plane_strain=true, ordem=2, nome="finite_width_crack", pontointerno=false)
    B = parentmodule(@__MODULE__)
    msh = mesh_finite_width_crack(; W, H, a, gap, ndiv_b, ndiv_h, ndiv_crack, n_cap,
        field=:y, ordem, nome, show=false, physics=:elasticity, σ)
    props = B.Elasticity(E, ν, 1.0; plane_strain=plane_strain)
    dad = B.format2d(msh, props; tipo=ordem, pontointerno=pontointerno)
    orient_slit_normals!(dad; a, gap)
    pin_plate_rbm!(dad; W=W, H=H)
    return dad
end

"""Opening ``u_{\\mathrm{upper}} - u_{\\mathrm{lower}}`` at the diamond mid-span."""
function slit_opening_mid(dad::BEMdata{<:Elasticity}; a=1.0, gap=0.05)
    top = findall(i -> _on_slit(dad, i, a, gap) && dad.Nodes[i][2] > 0.02 * gap, 1:dad.n)
    bot = findall(i -> _on_slit(dad, i, a, gap) && dad.Nodes[i][2] < -0.02 * gap, 1:dad.n)
    (isempty(top) || isempty(bot)) && return SVector(NaN, NaN)
    it = top[argmin(abs.(getindex.(dad.Nodes[top], 1)))]
    xt = dad.Nodes[it][1]
    ib = bot[argmin(abs.(getindex.(dad.Nodes[bot], 1) .- xt))]
    u = dad.u
    uA = SVector(u[2it - 1], u[2it])
    uB = SVector(u[2ib - 1], u[2ib])
    return uA - uB
end

