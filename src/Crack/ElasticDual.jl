# Elasticity dual BEM on BEMdata (Portela–Aliabadi–Rooke / Albuquerque–Sato)
# Twin discontinuous faces from Gmsh + format2d, BC type 5:
#   "5;2;5;2"  face A — displacement BIE (CBIE)
#   "5;3;5;3"  face B — traction BIE (HBIE)
# Kelvin / Lekhnitskii hypersingular: `fundamental_hyper` (D,S contracted with n_ξ).

const ElasticCrackProps = Union{Elasticity, AnisotropicElasticity}

"""
    assemble_dual_elasticity!(dad; npg=20, threaded=true) -> (H, G)

Mixed CBIE / HBIE dual BEM on a mesh prepared by [`prepare_crack!`](@ref).
`Elasticity` uses Kelvin; `AnisotropicElasticity` uses Lekhnitskii
(`fundamental` / `fundamental_hyper`).

- Outer (`eq_type=1`) and crack face A (`eq=2`): collocation BIE.
- Crack face B (`eq=3`): hypersingular BIE (`fundamental_hyper`).
- Self and coincident twin: Guiggiani (anisotropic HBIE: Cordeiro SST).
  Parallel-but-offset faces stay sinh.
- Outer rows: rigid-body row-sum free term.
- Crack CBIE: ``H_{ii} \\mathrel{+}= \\tfrac12 I`` and the twin column.
- Crack HBIE: ``G_{ii} \\mathrel{-}= \\tfrac12 I`` and the twin column
  (not solid ``c_{ij}=0``).
"""
function assemble_dual_elasticity!(dad; npg::Int=20, threaded::Bool=true,
        laurent::Symbol=:interp, ninterp::Int=20)
    B = parentmodule(@__MODULE__)
    B.set_cache!(dad; laurent=laurent, ninterp=ninterp)
    dad.properties isa ElasticCrackProps ||
        error("assemble_dual_elasticity!: Elasticity or AnisotropicElasticity")
    if !B.has_cache(dad, :eq_type)
        prepare_crack!(dad)
    elseif !B.has_cache(dad, :twin)
        B.set_cache!(dad; twin=zeros(Int, dad.n))
    end
    dad.dimension == 2 || error("assemble_dual_elasticity!: 2D only")
    B._init_quadrature!(dad, npg)
    dim = dad.dimension
    n, nt = dad.n, dad.nt
    H = zeros(dim * nt, dim * nt)
    G = zeros(dim * nt, dim * n)
    eq = dad.eq_type
    twin = dad.twin
    elems = dad.elements

    B._collocation_loop!(threaded, n) do i
        pf = B.point(dad, i)
        nf = dad.Normal[i]
        ii = B.expand(i, dim)
        tipo = eq[i]
        if tipo == 3
            f = (d, r, nrm) -> B.fundamental_hyper(d, r, nrm, nf)
            orders = _HBIE_ORDERS_CRACK
        else
            f = B.fundamental
            orders = nothing
        end
        @inbounds for el in elems
            xj = dad.Nodes[el.index]
            jj = B.expand(el.index, dim)
            on = B._source_on_element(el, i) || B._source_on_twin_element(el, i, twin)
            near = on || B._near_element(pf, xj, el)
            # Never 1-point-lump HBIE 1/r² (second crack, outer wall).
            if near || tipo == 3
                hloc = zeros(dim, length(jj))
                gloc = zeros(dim, length(jj))
                B.integrate_element(dad, el, xj, pf, hloc, gloc, f;
                    orders=orders, source=i, twins=twin)
                H[ii, jj] .+= hloc
                G[ii, jj] .+= gloc
            else
                B._far_nodal_vec!(H, G, dad, el, pf, ii, dim)
            end
        end
    end

    if nt > n
        B._collocation_loop!(threaded, nt - n) do k
            i = n + k
            pf = B.point(dad, i)
            ii = B.expand(i, dim)
            @inbounds for el in elems
                xj = dad.Nodes[el.index]
                jj = B.expand(el.index, dim)
                if B._near_element(pf, xj, el)
                    hloc = zeros(dim, length(jj))
                    gloc = zeros(dim, length(jj))
                    B.integrate_element(dad, el, xj, pf, hloc, gloc; source=i)
                    H[ii, jj] .+= hloc
                    G[ii, jj] .+= gloc
                else
                    B._far_nodal_vec!(H, G, dad, el, pf, ii, dim)
                end
            end
        end
    end

    @inbounds for i in 1:n
        tipo = eq[i]
        ii = B.expand(i, dim)
        if tipo == 1
            H[ii, ii] .= 0.0
            for j in 1:dim
                H[ii, ii[j]] .= -sum(H[ii, j:dim:end]; dims=2)
            end
        elseif tipo == 2
            for d in 1:dim
                H[ii[d], ii[d]] += 0.5
            end
            tw = twin[i]
            if tw != 0
                tt = B.expand(tw, dim)
                for d in 1:dim
                    H[ii[d], tt[d]] += 0.5
                end
            end
        elseif tipo == 3
            for d in 1:dim
                G[ii[d], ii[d]] -= 0.5
            end
            tw = twin[i]
            if tw != 0
                tt = B.expand(tw, dim)
                for d in 1:dim
                    G[ii[d], tt[d]] -= 0.5
                end
            end
        end
    end
    @inbounds for i in (n + 1):nt
        ii = B.expand(i, dim)
        H[ii, ii] .= 0.0
        for j in 1:dim
            H[ii, ii[j]] .= -sum(H[ii, j:dim:end]; dims=2)
        end
    end

    B.set_cache!(dad; H, G)
    return H, G
end

"""
    assemble_dual!(dad; npg=20, threaded=true) -> (H, G)

Elasticity dual BEM: displacement BIE on the outer boundary and crack face A,
hypersingular traction BIE on crack face B. Alias of
`assemble_dual_elasticity!`. Laplace cracks use `assemble_dual_laplace!`.
"""
assemble_dual!(dad::BEMdata{<:ElasticCrackProps}; npg::Int=20, threaded::Bool=true,
        kwargs...) =
    assemble_dual_elasticity!(dad; npg=npg, threaded=threaded, kwargs...)

"""
    solve_dual!(dad; npg=20, threaded=true) -> u

[`prepare_crack!`](@ref) (if needed) → assemble if `H` is missing → [`BEM.solve`](@ref).
"""
function solve_dual!(dad::BEMdata{<:ElasticCrackProps}; npg::Int=20, threaded::Bool=true)
    B = parentmodule(@__MODULE__)
    B.has_cache(dad, :eq_type) || prepare_crack!(dad)
    B.has_cache(dad, :H) || assemble_dual_elasticity!(dad; npg=npg, threaded=threaded)
    return B.solve(dad)
end

# =============================================================================
# Rigid-body pins
# =============================================================================

"""
    pin_plate_rbm!(dad; W=5, H=10)

Three Dirichlet pins on the outer boundary of a centred plate `[-W,W]×[-H,H]`
to kill rigid translation/rotation (all-Neumann far-field tension).
"""
function pin_plate_rbm!(dad; W=5.0, H=10.0)
    B = parentmodule(@__MODULE__)
    eq = B.has_cache(dad, :eq_type) ? dad.eq_type : ones(Int, dad.n)
    function nearest(pred)
        best = 0
        bd = Inf
        @inbounds for i in 1:dad.n
            eq[i] == 1 || continue
            p = dad.Nodes[i]
            pred(p) || continue
            d = abs(p[1]) + abs(p[2])
            if d < bd
                bd = d
                best = i
            end
        end
        return best
    end
    i_left = nearest(p -> abs(p[1] + W) < 1e-6 * max(W, 1) && abs(p[2]) < 0.6H)
    i_bot = nearest(p -> abs(p[2] + H) < 1e-6 * max(H, 1) && abs(p[1]) < 0.6W)
    i_bot2 = nearest(p -> abs(p[2] + H) < 1e-6 * max(H, 1) && abs(p[1]) < 0.25W)
    function set_dir!(inode, dir, val=0.0)
        inode == 0 && return
        dad.BC[2 * (inode - 1) + dir] = 0
        dad.BV[2 * (inode - 1) + dir] = val
        return nothing
    end
    set_dir!(i_left, 1, 0.0)
    set_dir!(i_bot, 2, 0.0)
    if i_bot2 != 0 && i_bot2 != i_bot
        set_dir!(i_bot2, 1, 0.0)
    elseif i_left != 0
        set_dir!(i_left, 2, 0.0)
    end
    return dad
end

# =============================================================================
# COD / SIF
# =============================================================================

"""Crack opening ``u_A - u_B`` at a face-A (or any twinned) node."""
function crack_opening(dad, inode::Int)
    tw = dad.twin[inode]
    tw == 0 && error("node $inode has no twin")
    u = dad.u
    uA = SVector(u[2inode - 1], u[2inode])
    uB = SVector(u[2tw - 1], u[2tw])
    return uA - uB
end

"""
    sif_cod_dual(dad, tip_node; sample=2) -> (KI, KII)

COD correlation at the `sample`-th node behind the tip on face A.
Isotropic: Williams ``μ/(κ+1)``. Anisotropic: invert the Sih matrix
``M(π)-M(-π)`` in the tip frame.
"""
function sif_cod_dual(dad, tip_node::Int; sample::Int=2)
    faceA = crack_face_nodes(dad; face=2)
    isempty(faceA) && error("sif_cod_dual: no face-A nodes")
    tip = dad.Nodes[tip_node]
    sort!(faceA; by=i -> norm(dad.Nodes[i] - tip))
    isamp = faceA[min(1 + sample, length(faceA))]
    r = max(norm(dad.Nodes[isamp] - tip), 1e-14)
    Δu = crack_opening(dad, isamp)
    props = dad.properties
    if props isa Elasticity
        e1, e2 = tip_frame(dad, tip)
        ωA = crack_omega(e2, dad.Normal[isamp])
        ωA < 0 && (Δu = -Δu)   # face A is the lower face → flip to u⁺ − u⁻
        Δun = abs(dot(Δu, e2))
        Δut = dot(Δu, e1)
        μ = props.mu
        κ = kappa(props.E, props.nu, props.plane_strain)
        c = μ / (κ + 1) * sqrt(2π / r)
        return c * Δun, c * Δut
    elseif props isa AnisotropicElasticity
        e1, e2 = tip_frame(dad, tip)
        field = make_tip_field(props, e1, e2)
        ωA = crack_omega(e2, dad.Normal[isamp])
        ωA < 0 && (Δu = -Δu)   # face A is the lower face → flip to u⁺ − u⁻
        Δtip = SVector(dot(Δu, e1), dot(Δu, e2))
        Mcod = tip_M(field, r, π) - tip_M(field, r, -π)
        K = Mcod \ Δtip
        return K[1], K[2]
    else
        error("sif_cod_dual: Elasticity or AnisotropicElasticity")
    end
end

# =============================================================================
# Gmsh: coincident twin faces (Griffith plate)
# =============================================================================

"""
    mesh_center_crack(; W=5, H=10, a=1, α=0, σ=1, ...) -> path

Rectangle `[-W,W]×[-H,H]` with a centre crack of half-length `a` at angle
`α` to the x-axis (Erdogan–Sih `β = π/2 - α` to the tensile axis) as
**two coincident curves** of opposite orientation. Physical names:

| Curve | Name |
|-------|------|
| face A (L→R) | `"5;2;5;2"` CBIE |
| face B (R→L) | `"5;3;5;3"` HBIE |
| bottom / top | `"1;0;1;±σ"` |
| sides | `"1;0;1;0"` |

Do not call Gmsh `removeAllDuplicates` — that would merge the twins.
"""
function mesh_center_crack(; W=5.0, H=10.0, a=1.0, α=0.0,
        ndiv_b=10, ndiv_h=16, ndiv_crack=16, σ=1.0,
        ordem=2, nome="center_crack", show=false)
    B = parentmodule(@__MODULE__)
    gmsh = B.gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(nome)
    lc = min(W, H) / max(ndiv_b, 8)
    cα, sα = cos(α), sin(α)

    p1 = gmsh.model.geo.addPoint(-W, -H, 0, lc)
    p2 = gmsh.model.geo.addPoint(W, -H, 0, lc)
    p3 = gmsh.model.geo.addPoint(W, H, 0, lc)
    p4 = gmsh.model.geo.addPoint(-W, H, 0, lc)
    ptL = gmsh.model.geo.addPoint(-a * cα, -a * sα, 0, lc / 2)
    ptR = gmsh.model.geo.addPoint(a * cα, a * sα, 0, lc / 2)

    lb = gmsh.model.geo.addLine(p1, p2)
    lr = gmsh.model.geo.addLine(p2, p3)
    lt = gmsh.model.geo.addLine(p3, p4)
    ll = gmsh.model.geo.addLine(p4, p1)
    cA = gmsh.model.geo.addLine(ptL, ptR)
    cB = gmsh.model.geo.addLine(ptR, ptL)
    cl = gmsh.model.geo.addCurveLoop([lb, lr, lt, ll])
    s = gmsh.model.geo.addPlaneSurface([cl])
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.embed(1, [cA, cB], 2, s)

    gmsh.model.mesh.setTransfiniteCurve(lb, ndiv_b)
    gmsh.model.mesh.setTransfiniteCurve(lt, ndiv_b)
    gmsh.model.mesh.setTransfiniteCurve(lr, ndiv_h)
    gmsh.model.mesh.setTransfiniteCurve(ll, ndiv_h)
    gmsh.model.mesh.setTransfiniteCurve(cA, ndiv_crack)
    gmsh.model.mesh.setTransfiniteCurve(cB, ndiv_crack)

    gmsh.model.addPhysicalGroup(1, [lb], -1, "1;0;1;$(-σ)")
    gmsh.model.addPhysicalGroup(1, [lt], -1, "1;0;1;$σ")
    gmsh.model.addPhysicalGroup(1, [ll, lr], -1, "1;0;1;0")
    gmsh.model.addPhysicalGroup(1, [cA], -1, "5;2;5;2")
    gmsh.model.addPhysicalGroup(1, [cB], -1, "5;3;5;3")
    gmsh.model.addPhysicalGroup(2, [s], -1, "Domain")

    gmsh.model.mesh.generate(2)
    ordem > 1 && gmsh.model.mesh.setOrder(ordem)
    out = B.datadir("elastico", "iso", nome * ".msh")
    mkpath(dirname(out))
    gmsh.write(out)
    show && gmsh.fltk.run()
    gmsh.finalize()
    return out
end

"""
    dual_elasticity_problem(; W=5, H=10, a=1, E=3000, ν=0.2, σ=1, ...) -> dad

Mesh + `format2d` + [`prepare_crack!`](@ref) + rigid-body pins.
"""
function dual_elasticity_problem(; W=5.0, H=10.0, a=1.0, α=0.0, E=3000.0, ν=0.2, σ=1.0,
        ndiv_b=10, ndiv_h=16, ndiv_crack=16, plane_strain=true,
        ordem=2, nome="center_crack", pontointerno=false, props=nothing)
    B = parentmodule(@__MODULE__)
    msh = mesh_center_crack(; W, H, a, α, ndiv_b, ndiv_h, ndiv_crack, σ, ordem, nome, show=false)
    mat = props === nothing ? B.Elasticity(E, ν, 1.0; plane_strain=plane_strain) : props
    dad = B.format2d(msh, mat; tipo=ordem, pontointerno=pontointerno)
    prepare_crack!(dad)
    pin_plate_rbm!(dad; W=W, H=H)
    return dad
end

"""
    build_center_crack_mesh(; ...) -> BEMdata

Compatibility wrapper around [`dual_elasticity_problem`](@ref)
(legacy kwargs `n_bottom` / `n_crack` / …).
"""
function build_center_crack_mesh(; W=5.0, H=10.0, a=1.0, α=0.0,
        n_bottom=8, n_right=16, n_top=8, n_left=16, n_crack=16,
        E=3000.0, ν=0.2, σ=1.0, plane_strain=true, ordem=2, nome="center_crack")
    return dual_elasticity_problem(; W, H, a, α, E, ν, σ, plane_strain, ordem, nome,
        ndiv_b=max(n_bottom, n_top), ndiv_h=max(n_right, n_left),
        ndiv_crack=n_crack)
end

# =============================================================================
# CSTBD (Ke 2008 Fig. 8 / §4.2)
# =============================================================================

"""Hualien marble of Ke, Chen, Ku & Chen (IJNAMG 2009), plane-stress Lekhnitskii.
`E` in the isotropy plane, `E′` normal to it; `ψ` is that plane to the x-axis."""
function ke2008_marble(ψ_deg)
    B = parentmodule(@__MODULE__)
    return B.AnisotropicElasticity(B.lekhnitskii_params(
        78.302, 67.681, 25.336, 0.185; θ_deg=ψ_deg))
end

"""
    mesh_cstbd(; R, a, β, p, ...) -> path

Circle of radius `R` with a centre crack of half-length `a`. `β` is the
angle from the loaded diameter (y-axis, Fig. 8): `β=0` is a vertical
mode-I crack. Diametral compression `p` on small polar patches.
"""
function mesh_cstbd(; R=3.7, a=1.1, β=π / 4, p=1.0,
        ndiv_outer=28, ndiv_crack=12, ndiv_load=2, load_halfdeg=8.0,
        ordem=2, nome="cstbd", show=false)
    0 < a < R || error("mesh_cstbd: need 0 < a < R")
    B = parentmodule(@__MODULE__)
    gmsh = B.gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(nome)
    lc = 2π * R / max(ndiv_outer, 16)
    δ = deg2rad(load_halfdeg)
    pc = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, lc)
    θs = (0.0, π / 2 - δ, π / 2 + δ, π, 3π / 2 - δ, 3π / 2 + δ)
    pts = [gmsh.model.geo.addPoint(R * cos(θ), R * sin(θ), 0.0, lc) for θ in θs]
    arcs = Int[]
    for i in 1:6
        j = i == 6 ? 1 : i + 1
        push!(arcs, gmsh.model.geo.addCircleArc(pts[i], pc, pts[j]))
    end
    sβ, cβ = sin(β), cos(β)
    ptL = gmsh.model.geo.addPoint(-a * sβ, -a * cβ, 0.0, lc / 2)
    ptR = gmsh.model.geo.addPoint(a * sβ, a * cβ, 0.0, lc / 2)
    cA = gmsh.model.geo.addLine(ptL, ptR)
    cB = gmsh.model.geo.addLine(ptR, ptL)
    cl = gmsh.model.geo.addCurveLoop(arcs)
    s = gmsh.model.geo.addPlaneSurface([cl])
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.embed(1, [cA, cB], 2, s)

    n_free = max(ndiv_outer - 2 * ndiv_load, 16)
    n_side = max(n_free ÷ 4, 4)
    n_on = (n_side, ndiv_load, n_side, n_side, ndiv_load, n_side)
    for (arc, n) in zip(arcs, n_on)
        gmsh.model.mesh.setTransfiniteCurve(arc, n + 1)
    end
    gmsh.model.mesh.setTransfiniteCurve(cA, ndiv_crack)
    gmsh.model.mesh.setTransfiniteCurve(cB, ndiv_crack)

    # load patches: pressure t = −p n ≈ (0, ∓p) at the poles
    gmsh.model.addPhysicalGroup(1, [arcs[2]], -1, "1;0;1;$(-p)")
    gmsh.model.addPhysicalGroup(1, [arcs[5]], -1, "1;0;1;$p")
    gmsh.model.addPhysicalGroup(1, [arcs[1], arcs[3], arcs[4], arcs[6]], -1, "1;0;1;0")
    gmsh.model.addPhysicalGroup(1, [cA], -1, "5;2;5;2")
    gmsh.model.addPhysicalGroup(1, [cB], -1, "5;3;5;3")
    gmsh.model.addPhysicalGroup(2, [s], -1, "Domain")

    gmsh.model.mesh.generate(2)
    ordem > 1 && gmsh.model.mesh.setOrder(ordem)
    out = B.datadir("elastico", "iso", nome * ".msh")
    mkpath(dirname(out))
    gmsh.write(out)
    show && gmsh.fltk.run()
    gmsh.finalize()
    return out
end

function pin_disc_rbm!(dad; R=3.7)
    eq = dad.eq_type
    function nearest(target)
        best, bd = 0, Inf
        @inbounds for i in 1:dad.n
            eq[i] == 1 || continue
            d = norm(dad.Nodes[i] - target)
            if d < bd
                bd = d
                best = i
            end
        end
        return best
    end
    i_bot = nearest(Point2D(0.0, -R))
    i_right = nearest(Point2D(R, 0.0))
    function set_dir!(inode, dir)
        inode == 0 && return
        dad.BC[2 * (inode - 1) + dir] = 0
        dad.BV[2 * (inode - 1) + dir] = 0.0
        return nothing
    end
    set_dir!(i_bot, 1)
    set_dir!(i_bot, 2)
    i_right != i_bot && set_dir!(i_right, 2)
    return dad
end

"""
    cstbd_problem(; R=3.7, a=1.1, β=π/4, ψ=0, p=1, ...) -> dad

Ke CSTBD: diametral compression, centre crack at `β` from the load,
marble (or `props`) with isotropy-plane angle `ψ`.
"""
function cstbd_problem(; R=3.7, a=1.1, β=π / 4, ψ=0.0, p=1.0,
        ndiv_outer=28, ndiv_crack=12, ndiv_load=2, load_halfdeg=8.0,
        ordem=2, nome="cstbd", props=nothing)
    B = parentmodule(@__MODULE__)
    msh = mesh_cstbd(; R, a, β, p, ndiv_outer, ndiv_crack, ndiv_load,
        load_halfdeg, ordem, nome, show=false)
    mat = props === nothing ? ke2008_marble(ψ) : props
    dad = B.format2d(msh, mat; tipo=ordem, pontointerno=false)
    prepare_crack!(dad)
    pin_disc_rbm!(dad; R=R)
    return dad
end

# =============================================================================
# Dual-BEM increment (MTS / Erdogan–Sih)
# =============================================================================

"""Ligament unit vector at a geometric crack end (from the crack through the tip)."""
function _crack_ahead(dad, tip_pos)
    best_d, ahead = Inf, zero(tip_pos)
    for ie in dad.crack_face_a
        p0, p1 = _curve_end_points(dad, dad.elements[ie])
        for (free, other) in ((p0, p1), (p1, p0))
            d = norm(free - tip_pos)
            if d < best_d
                v = free - other
                nv = norm(v)
                nv < 1e-14 && continue
                best_d = d
                ahead = v / nv
            end
        end
    end
    isfinite(best_d) || error("extend_dual_crack_tip!: no face-A element at $tip_pos")
    return ahead
end

function _refresh_tip_nodes!(dad)
    B = parentmodule(@__MODULE__)
    nodesA = unique!(reduce(vcat, (dad.elements[ie].index for ie in dad.crack_face_a); init=Int[]))
    tips = _free_end_positions(dad, dad.crack_face_a)
    tip_nodes = [_nearest_index(dad, nodesA, p) for p in tips]
    B.set_cache!(dad; tip_nodes=tip_nodes)
    return tip_nodes
end

"""
    extend_dual_crack_tip!(dad, tip_pos, direction, da; n_new=1)

Append `n_new` coincident type-5 twins of length `da` along `direction`
ahead of the geometric end nearest `tip_pos`. Call [`assemble_dual!`](@ref)
again (cache `H` is cleared). Requires `dad.ni == 0`.
"""
function extend_dual_crack_tip!(
        mesh::BEMdata{<:ElasticCrackProps},
        tip_pos,
        direction,
        da::Real;
        n_new::Int=1,
    )
    B = parentmodule(@__MODULE__)
    mesh.ni == 0 || error("extend_dual_crack_tip!: no internal nodes")
    da > 0 || error("extend_dual_crack_tip!: da must be positive")
    n_new >= 1 || error("extend_dual_crack_tip!: n_new ≥ 1")
    d = Point2D(direction)
    d = d / (norm(d) + eps())
    ends = _free_end_positions(mesh, mesh.crack_face_a)
    isempty(ends) && error("extend_dual_crack_tip!: no free crack end")
    geo = ends[argmin(norm(p - Point2D(tip_pos)) for p in ends)]

    pdeg = length(mesh.elements[1]) - 1
    n_per = pdeg + 1
    qsi, wi = B.discontinuous_nodes_weights(pdeg)
    poly_geo = B.Equispaced(pdeg)
    Ngeo, dNgeo = B.shapefun(poly_geo, qsi)

    xs = [geo + (k / n_new) * da * d for k in 0:n_new]
    n̂_plus = B.tan2normal(d)
    n̂_minus = -n̂_plus
    eq_type = copy(mesh.eq_type)
    twin = copy(mesh.twin)
    face_a = copy(mesh.crack_face_a)
    face_b = copy(mesh.crack_face_b)
    coll = getfield(mesh, :collocation)
    nodes_plus = Int[]
    nodes_minus = Int[]

    function add_seg!(p1, p2, eq, nfix)
        X = [p1 + ((k - 1) / pdeg) * (p2 - p1) for k in 1:n_per]
        NOS = Ngeo * X
        dx = dNgeo * X
        J = norm.(dx)
        Lseg = abs(dot(J, wi))
        n0 = mesh.n
        idx = collect((n0 + 1):(n0 + n_per))
        append!(coll, Point2D.(NOS))
        append!(mesh.Normal, fill(nfix, n_per))
        for _ in 1:n_per
            append!(mesh.BC, [1, 1])
            append!(mesh.BV, [0.0, 0.0])
        end
        append!(eq_type, fill(eq, n_per))
        append!(twin, zeros(Int, n_per))
        mesh.n = n0 + n_per
        mesh.nt = mesh.n
        push!(mesh.elements, B.Element(idx, collect(Float64, J), Float64(Lseg), 0))
        return idx
    end

    for i in 1:n_new
        p1, p2 = xs[i], xs[i + 1]
        append!(nodes_plus, add_seg!(p1, p2, 2, n̂_plus))
        push!(face_a, length(mesh.elements))
        append!(nodes_minus, add_seg!(p2, p1, 3, n̂_minus))
        push!(face_b, length(mesh.elements))
    end

    for ip in nodes_plus
        xp = coll[ip]
        best, dmin = 0, Inf
        for im in nodes_minus
            dd = norm(coll[im] - xp)
            dd < dmin && ((dmin, best) = (dd, im))
        end
        if best > 0
            twin[ip] = best
            twin[best] = ip
        end
    end

    B.set_cache!(mesh; eq_type=eq_type, twin=twin,
        crack_face_a=face_a, crack_face_b=face_b,
        H=nothing, G=nothing, A=nothing, B=nothing, b=nothing, u=nothing,
        H_xbem=nothing, C_u=nothing, C_sif=nothing, KI=nothing, KII=nothing)
    _refresh_tip_nodes!(mesh)
    return nodes_plus
end
