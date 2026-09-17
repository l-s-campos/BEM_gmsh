# Laplace dual BEM on BEMdata (Portela–Aliabadi–Rooke)
# Twin discontinuous faces from Gmsh + format2d, BC type 5:
#   "5;2"  face A — collocation BIE (CBIE)
#   "5;3"  face B — hypersingular BIE (HBIE)

const _HBIE_ORDERS_CRACK = (-1, -2)

"""
    prepare_crack!(dad; bc=:insulated, T0=0)

Read Gmsh type-5 tags, store `eq_type` / `twin` / face element lists on
`dad.cache`, and rewrite `BC`/`BV`. Twin pairing is by collocation
proximity (opposite-oriented faces reverse the local index).

**Laplace:** `bc=:insulated` → Neumann `q=0`; `bc=:conducting` → Dirichlet `T=T0`.
**Elasticity:** type-5 faces become traction-free (`t=0`); `bc` is ignored.
Physical names: scalar `"5;2"` / `"5;3"`; elasticity / Kirchhoff
`"5;2;5;2"` / `"5;3;5;3"`; FSDT 3-DOF three pairs, Hsu–Hwu 5-DOF five pairs.
"""
function prepare_crack!(dad; bc::Symbol=:insulated, T0::Real=0.0)
    dad.dimension == 2 || error("prepare_crack!: 2D only")
    B = parentmodule(@__MODULE__)
    n = dad.n
    dof = length(dad.BC) ÷ n
    is_vec = dof > 1
    if B.has_cache(dad, :eq_type)
        eq_type = copy(dad.eq_type)
    else
        eq_type = ones(Int, n)
        @inbounds for i in 1:n
            bci, bvi = _node_bc_pair(dad, i, dof)
            if bci == CRACK_BC
                eq = Int(round(bvi))
                eq_type[i] = eq in (2, 3) ? eq : 2
            end
        end
    end
    count(==(2), eq_type) > 0 && count(==(3), eq_type) > 0 ||
        error("prepare_crack!: need type-5 faces tagged \"5;2\" and \"5;3\"")

    faceA = Int[]
    faceB = Int[]
    for (ie, el) in enumerate(dad.elements)
        eq = eq_type[el.index[1]]
        if eq == 2
            push!(faceA, ie)
        elseif eq == 3
            push!(faceB, ie)
        end
    end

    twin = zeros(Int, n)
    nodesA = unique!(reduce(vcat, (dad.elements[ie].index for ie in faceA); init=Int[]))
    nodesB = unique!(reduce(vcat, (dad.elements[ie].index for ie in faceB); init=Int[]))
    _pair_twins!(twin, dad, nodesA, nodesB)

    if is_vec
        bc = :traction_free
        @inbounds for i in 1:n
            eq_type[i] in (2, 3) || continue
            _set_node_bc!(dad, i, dof, 1, 0.0)
        end
    else
        bc in (:insulated, :conducting) ||
            throw(ArgumentError("prepare_crack!: bc must be :insulated or :conducting, got $bc"))
        @inbounds for i in 1:n
            eq_type[i] in (2, 3) || continue
            if bc === :insulated
                dad.BC[i] = 1
                dad.BV[i] = 0.0
            else
                dad.BC[i] = 0
                dad.BV[i] = float(T0)
            end
        end
    end

    tips = _free_end_positions(dad, faceA)
    tip_nodes = [_nearest_index(dad, nodesA, p) for p in tips]

    B.set_cache!(dad; eq_type=eq_type, twin=twin,
        crack_face_a=faceA, crack_face_b=faceB, crack_bc=bc, tip_nodes=tip_nodes)
    return dad
end

"""Parent-curve ends `ξ=±1` of a discontinuous element (not the inset collocation)."""
function _curve_end_points(dad, el)
    B = parentmodule(@__MODULE__)
    xj = dad.Nodes[el.index]
    Nm, _ = B.shapefun(dad.element_type, -1.0)
    Np, _ = B.shapefun(dad.element_type, 1.0)
    p0 = zero(xj[1])
    p1 = zero(xj[1])
    @inbounds for k in eachindex(xj)
        p0 += Nm[1, k] * xj[k]
        p1 += Np[1, k] * xj[k]
    end
    return p0, p1
end

"""Valence-1 endpoints of a crack-face polyline (geometric tips / mouths)."""
function _free_end_positions(dad, face_els)
    ends = typeof(dad.Nodes[1])[]
    counts = Int[]
    for ie in face_els
        p0, p1 = _curve_end_points(dad, dad.elements[ie])
        for p in (p0, p1)
            idx = findfirst(q -> norm(p - q) < 1e-9, ends)
            if idx === nothing
                push!(ends, p)
                push!(counts, 1)
            else
                counts[idx] += 1
            end
        end
    end
    return [ends[i] for i in eachindex(ends) if counts[i] == 1]
end

function _nearest_index(dad, idxs, pos)
    return idxs[argmin(norm(dad.Nodes[i] - pos) for i in idxs)]
end

@inline function _node_bc_pair(dad, i, dof)
    if dof == 1
        return dad.BC[i], dad.BV[i]
    end
    j = dof * (i - 1) + 1
    return dad.BC[j], dad.BV[j]
end

function _set_node_bc!(dad, i, dof, tipo, val)
    @inbounds for d in 1:dof
        j = dof * (i - 1) + d
        dad.BC[j] = tipo
        dad.BV[j] = val
    end
    return nothing
end

function _pair_twins!(twin, dad, faceA, faceB)
    (isempty(faceA) || isempty(faceB)) && return twin
    used = falses(length(faceB))
    @inbounds for ia in faceA
        best = 0
        bd = Inf
        pa = dad.Nodes[ia]
        for (k, ib) in enumerate(faceB)
            used[k] && continue
            d = norm(dad.Nodes[ib] - pa)
            if d < bd
                bd = d
                best = k
            end
        end
        best == 0 && continue
        ib = faceB[best]
        used[best] = true
        twin[ia] = ib
        twin[ib] = ia
        dad.Normal[ib] = -dad.Normal[ia]
    end
    return twin
end

"""
    integral_kind(dad, source, elem; twins=nothing) -> Symbol

`:guiggiani_self`, `:guiggiani_twin`, `:sinh`, or `:far`.
"""
function integral_kind(dad, source::Integer, elem; twins=nothing)
    B = parentmodule(@__MODULE__)
    B._source_on_element(elem, source) && return :guiggiani_self
    twins !== nothing && B._source_on_twin_element(elem, source, twins) &&
        return :guiggiani_twin
    pf = dad.Nodes[source]
    xj = dad.Nodes[elem.index]
    return B._near_element(pf, xj, elem) ? :sinh : :far
end

# =============================================================================
# Assembly
# =============================================================================

"""
    assemble_dual_laplace!(dad; npg=20, threaded=true) -> (H, G)

Mixed CBIE / HBIE Laplace dual BEM on a mesh prepared by [`prepare_crack!`](@ref).

- Outer (`eq_type=1`) and crack face A (`eq=2`): collocation BIE.
- Crack face B (`eq=3`): hypersingular BIE (`fundamental_hyper`).
- Self and coincident twin: Guiggiani. Parallel-but-offset faces stay sinh.
- Outer rows: constant-field row-sum free term. Crack CBIE: `H_ii += 1/2`.
  Crack HBIE: `G_ii -= 1/(2k)` (same jump as `H_G_hyper`).
"""
function assemble_dual_laplace!(dad; npg::Int=20, threaded::Bool=true,
        laurent::Symbol=:interp, ninterp::Int=20, near_factor::Real=1.5)
    B = parentmodule(@__MODULE__)
    B.set_cache!(dad; laurent=laurent, ninterp=ninterp)
    if !B.has_cache(dad, :eq_type)
        prepare_crack!(dad)
    elseif !B.has_cache(dad, :twin)
        B.set_cache!(dad; twin=zeros(Int, dad.n))
    end
    dad.dimension == 2 || error("assemble_dual_laplace!: 2D only")
    B._init_quadrature!(dad, npg)
    n, nt = dad.n, dad.nt
    H = zeros(nt, nt)
    G = zeros(nt, n)
    eq = dad.eq_type
    twin = dad.twin
    elems = dad.elements
    k = float(dad.properties.k)
    nE = length(elems)
    nodes_el, hbufs, gbufs = nE == 0 ?
        (nothing, nothing, nothing) :
        B._scalar_elem_bufs(dad, elems, eltype(H))

    B._collocation_loop!(threaded, n) do i
        pf = B.point(dad, i)
        nf = dad.Normal[i]
        tipo = eq[i]
        if tipo == 3
            f = (d, r, nrm) -> B.fundamental_hyper(d, r, nrm, nf)
            orders = _HBIE_ORDERS_CRACK
        else
            f = B.fundamental
            orders = nothing
        end
        far = if tipo == 3
            (H, G, dad, el, pf, i) -> B._far_nodal_hyper!(H, G, dad, el, pf, nf, i)
        else
            B._far_nodal_scalar!
        end
        tid = B._assembly_tid(threaded)
        @inbounds for eidx in 1:nE
            B._accumulate_element_scalar!(H, G, dad, elems[eidx], pf, i;
                f=f, orders=orders, twins=twin, far=far,
                xj=nodes_el[eidx], hbuf=hbufs[tid], gbuf=gbufs[tid],
                near_factor=near_factor)
        end
    end

    if nt > n
        B._collocation_loop!(threaded, nt - n) do ii
            i = n + ii
            pf = B.point(dad, i)
            tid = B._assembly_tid(threaded)
            @inbounds for eidx in 1:nE
                B._accumulate_element_scalar!(H, G, dad, elems[eidx], pf, i;
                    xj=nodes_el[eidx], hbuf=hbufs[tid], gbuf=gbufs[tid],
                    near_factor=near_factor)
            end
        end
    end

    inv2k = 0.5 / k
    @inbounds for i in 1:n
        tipo = eq[i]
        if tipo == 1
            H[i, i] = 0.0
            H[i, i] = -sum(view(H, i, :))
        elseif tipo == 2
            # Smooth-crack jump ½ T⁺ and ½ T⁻ (coincident opposite face).
            # With ∫_outer ∂G/∂n = −1 this makes H 1 = 0 for T⁺ = T⁻.
            H[i, i] += 0.5
            tw = twin[i]
            tw != 0 && (H[i, tw] += 0.5)
        elseif tipo == 3
            G[i, i] -= inv2k
        end
    end
    @inbounds for i in (n + 1):nt
        H[i, i] = 0.0
        H[i, i] = -sum(view(H, i, :))
    end

    B.set_cache!(dad; H, G)
    return H, G
end

"""
    solve_dual_laplace!(dad; npg=20, threaded=true) -> T

[`prepare_crack!`](@ref) (if needed) → assemble → [`BEM.solve`](@ref).
"""
function solve_dual_laplace!(dad; npg::Int=20, threaded::Bool=true, bc::Symbol=:insulated,
        T0::Real=0.0)
    parentmodule(@__MODULE__).has_cache(dad, :eq_type) || prepare_crack!(dad; bc=bc, T0=T0)
    assemble_dual_laplace!(dad; npg=npg, threaded=threaded)
    B = parentmodule(@__MODULE__)
    return B.solve(dad)
end

# =============================================================================
# Post-process
# =============================================================================

"""Potential jump `T[i] − T[twin]` at a face-A (or any twinned) node."""
function crack_jump(dad, i::Integer)
    tw = dad.twin[i]
    tw == 0 && error("node $i has no twin")
    return dad.T[i] - dad.T[tw]
end

"""Face-A collocation indices, sorted by `x`."""
function crack_face_nodes(dad; face::Integer=2)
    eq = dad.eq_type
    idx = findall(==(face), eq)
    sort!(idx; by=i -> dad.Nodes[i][1])
    return idx
end

# =============================================================================
# Closed-form infinite-plate fields (Griffith / anti-plane analog)
# =============================================================================

"""`√(z−a)√(z+a)` with Julia principal sqrts (cut on the slit). ~ `z` at ∞."""
@inline _sqrt_za(z, a) = sqrt(z - a) * sqrt(z + a)

"""
    analytical_insulated_crack_T(p, a; G=1)

Infinite plate, insulated crack `|x|<a`, far-field `T = G y`.
`T = G Re(−i √(z²−a²))`. Jump `ΔT = 2 G √(a²−x²)`.
"""
function analytical_insulated_crack_T(p, a; G=1.0)
    z = complex(p[1], p[2])
    return G * real(-im * _sqrt_za(z, a))
end

"""`ΔT(x) = 2 G √(a²−x²)` on the insulated slit."""
analytical_insulated_jump(x, a; G=1.0) = 2G * sqrt(max(a^2 - x^2, 0.0))

"""
    analytical_conducting_crack_T(p, a; G=1)

Infinite plate, conducting crack `T=0` on `|x|<a`, far-field `T = G x`.
`T = G Re √(z²−a²)`.
"""
function analytical_conducting_crack_T(p, a; G=1.0)
    z = complex(p[1], p[2])
    return G * real(_sqrt_za(z, a))
end

# =============================================================================
# Gmsh: coincident twin faces
# =============================================================================

"""
    mesh_center_crack_laplace(; W=5, H=10, a=1, field=:y, ...) -> path

Rectangle `[-W,W]×[-H,H]` with a centre crack `[-a,a]` as **two coincident
curves** of opposite orientation. Physical names:

| Curve | Name |
|-------|------|
| face A (L→R) | `"5;2"` CBIE |
| face B (R→L) | `"5;3"` HBIE |
| `field=:y` | top/bottom Dirichlet `T=±H`, sides `q=0` |
| `field=:x` | left/right Dirichlet `T=±W`, top/bottom `q=0` |

Do not call Gmsh `removeAllDuplicates` — that would merge the twins.
"""
function mesh_center_crack_laplace(; W=5.0, H=10.0, a=1.0,
        ndiv_b=10, ndiv_h=16, ndiv_crack=16,
        field::Symbol=:y, ordem=1, nome="center_crack_laplace", show=false)
    B = parentmodule(@__MODULE__)
    gmsh = B.gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(nome)
    lc = min(W, H) / max(ndiv_b, 8)

    p1 = gmsh.model.geo.addPoint(-W, -H, 0, lc)
    p2 = gmsh.model.geo.addPoint(W, -H, 0, lc)
    p3 = gmsh.model.geo.addPoint(W, H, 0, lc)
    p4 = gmsh.model.geo.addPoint(-W, H, 0, lc)
    ptL = gmsh.model.geo.addPoint(-a, 0, 0, lc / 2)
    ptR = gmsh.model.geo.addPoint(a, 0, 0, lc / 2)

    lb = gmsh.model.geo.addLine(p1, p2)
    lr = gmsh.model.geo.addLine(p2, p3)
    lt = gmsh.model.geo.addLine(p3, p4)
    ll = gmsh.model.geo.addLine(p4, p1)
    cA = gmsh.model.geo.addLine(ptL, ptR)   # face A, n ≈ −êy (upper)
    cB = gmsh.model.geo.addLine(ptR, ptL)   # face B, n ≈ +êy (lower)
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

    if field === :y
        gmsh.model.addPhysicalGroup(1, [lb], -1, "0;$(-H)")
        gmsh.model.addPhysicalGroup(1, [lt], -1, "0;$H")
        gmsh.model.addPhysicalGroup(1, [ll, lr], -1, "1;0")
    elseif field === :x
        gmsh.model.addPhysicalGroup(1, [ll], -1, "0;$(-W)")
        gmsh.model.addPhysicalGroup(1, [lr], -1, "0;$W")
        gmsh.model.addPhysicalGroup(1, [lb, lt], -1, "1;0")
    else
        error("mesh_center_crack_laplace: field must be :x or :y, got $field")
    end
    gmsh.model.addPhysicalGroup(1, [cA], -1, "5;2")
    gmsh.model.addPhysicalGroup(1, [cB], -1, "5;3")
    gmsh.model.addPhysicalGroup(2, [s], -1, "Domain")

    gmsh.model.mesh.generate(2)
    ordem > 1 && gmsh.model.mesh.setOrder(ordem)
    out = B.datadir("Laplace", nome * ".msh")
    mkpath(dirname(out))
    gmsh.write(out)
    show && gmsh.fltk.run()
    gmsh.finalize()
    return out
end

"""
    dual_laplace_problem(; field=:y, bc=:insulated, T0=0, k=1, kwargs...) -> dad

Mesh + `format2d` + [`prepare_crack!`](@ref). `field` sets the outer BCs
(`:y` → `T=y`, `:x` → `T=x`). `bc` is the crack law.
"""
function dual_laplace_problem(; W=5.0, H=10.0, a=1.0, k=1.0,
        ndiv_b=10, ndiv_h=16, ndiv_crack=16, field::Symbol=:y,
        bc::Symbol=:insulated, T0=0.0, ordem=1, nome="center_crack_laplace",
        pontointerno=false)
    B = parentmodule(@__MODULE__)
    msh = mesh_center_crack_laplace(; W, H, a, ndiv_b, ndiv_h, ndiv_crack,
        field, ordem, nome, show=false)
    dad = B.format2d(msh, B.Laplace(k); tipo=ordem, pontointerno=pontointerno)
    prepare_crack!(dad; bc=bc, T0=T0)
    return dad
end
