# 2-D topology optimization: explicit boundary loops → BEM (no Gmsh after t=0).
#
# Source of truth is `TopologyDesign` (closed loops of `BoundarySegment`s).
# `bemdata_from_loops` builds discontinuous Legendre collocation the same way
# `format2d_lagrange` does, then samples interior points by winding number.

export BoundarySegment, TopologyDesign
export bemdata_from_loops, extract_loops, extract_loops!
export design_area, sample_internal_points!
export resample_polyline, chaikin_smooth!
export in_polygon, polygon_area, loop_vertices
export vertex_outward_normals, freeze_dirichlet!
export copy_design, n_holes

"""One BC-homogeneous chain of vertices (open). Loops close by identification.

`bc` / `value` have length 1 (Laplace) or 2 (elasticity, x then y).
`0` = Dirichlet (potential / displacement), `1` = Neumann (flux / traction).
"""
mutable struct BoundarySegment
    verts::Vector{Point2D}
    bc::Vector{Int}
    value::Vector{Float64}
    frozen::BitVector
    n_el::Int
end

_has_dirichlet(seg::BoundarySegment) = any(==(0), seg.bc)
_has_dirichlet(bc::AbstractVector{<:Integer}) = any(==(0), bc)

"""True if the segment must not move: any Dirichlet DOF, or a non-zero Neumann (load patch)."""
function _is_fixed_bc(bc::AbstractVector{<:Integer}, val::AbstractVector{<:Real})
    _has_dirichlet(bc) && return true
    @inbounds for i in eachindex(bc)
        bc[i] != 0 && abs(val[i]) > 0 && return true
    end
    return false
end
_is_fixed_segment(seg::BoundarySegment) = _is_fixed_bc(seg.bc, seg.value)

function BoundarySegment(
        verts::AbstractVector{<:SVector{2}};
        bc=1,
        value=0.0,
        frozen::Union{Nothing,AbstractVector{Bool}}=nothing,
        n_el::Integer=max(length(verts) - 1, 1),
    )
    v = Point2D[Point2D(p[1], p[2]) for p in verts]
    n = length(v)
    bcv = bc isa Integer ? [Int(bc)] : collect(Int, bc)
    valv = value isa Real ? fill(Float64(value), length(bcv)) : collect(Float64, value)
    length(valv) == length(bcv) || throw(ArgumentError("bc and value length mismatch"))
    fr = frozen === nothing ? fill(_is_fixed_bc(bcv, valv), n) : BitVector(frozen)
    length(fr) == n || throw(ArgumentError("frozen length $(length(fr)) ≠ nverts $n"))
    return BoundarySegment(v, bcv, valv, fr, max(Int(n_el), 1))
end

"""Design = outer loop + holes. `loops[1]` is the outer contour (CCW)."""
mutable struct TopologyDesign
    loops::Vector{Vector{BoundarySegment}}
    degree::Int
    k::Float64
    nx_int::Int
    ny_int::Int
    name::String
    properties::Problem
end

function TopologyDesign(
        loops::Vector{Vector{BoundarySegment}};
        degree::Integer=2,
        k::Real=1.0,
        nx_int::Integer=20,
        ny_int::Integer=20,
        name::AbstractString="topology",
        properties::Union{Nothing,Problem}=nothing,
    )
    props = properties === nothing ? Laplace(Float64(k)) : properties
    kk = props isa Laplace ? Float64(props.k) : Float64(k)
    return TopologyDesign(loops, Int(degree), kk, Int(nx_int), Int(ny_int), String(name), props)
end

copy_design(d::TopologyDesign) = deepcopy(d)

n_holes(d::TopologyDesign) = max(length(d.loops) - 1, 0)

# =============================================================================
# Polygon helpers
# =============================================================================

# `polygon_area` is defined in Core/Input.jl and imported from BEM.

"""Winding-number test (boundary counts as inside)."""
function in_polygon(p::SVector{2}, verts::AbstractVector{<:SVector{2}})
    n = length(verts)
    n < 3 && return false
    wn = 0
    @inbounds for i in 1:n
        a = verts[i]
        b = verts[i == n ? 1 : i + 1]
        if a[2] <= p[2]
            if b[2] > p[2] && _cross(a, b, p) > 0
                wn += 1
            end
        else
            if b[2] <= p[2] && _cross(a, b, p) < 0
                wn -= 1
            end
        end
    end
    return wn != 0
end

@inline _cross(a, b, p) = (b[1] - a[1]) * (p[2] - a[2]) - (b[2] - a[2]) * (p[1] - a[1])

function loop_vertices(segs::Vector{BoundarySegment})
    pts = Point2D[]
    for s in segs
        isempty(s.verts) && continue
        if !isempty(pts) && pts[end] ≈ s.verts[1]
            append!(pts, @view s.verts[2:end])
        else
            append!(pts, s.verts)
        end
    end
    if length(pts) ≥ 2 && pts[1] ≈ pts[end]
        pop!(pts)
    end
    return pts
end

function in_design(p::SVector{2}, d::TopologyDesign)
    outer = loop_vertices(d.loops[1])
    in_polygon(p, outer) || return false
    for k in 2:length(d.loops)
        in_polygon(p, loop_vertices(d.loops[k])) && return false
    end
    return true
end

"""Material area (outer minus holes). Holes are stored clockwise (negative area)."""
function design_area(d::TopologyDesign)
    A = 0.0
    for (i, segs) in enumerate(d.loops)
        a = polygon_area(loop_vertices(segs))
        A += i == 1 ? abs(a) : -abs(a)
    end
    return A
end

function freeze_dirichlet!(d::TopologyDesign)
    dirip = Point2D[]
    for segs in d.loops, s in segs
        if _is_fixed_segment(s)
            fill!(s.frozen, true)
            append!(dirip, s.verts)
        end
    end
    isempty(dirip) && return d
    for segs in d.loops, s in segs
        _is_fixed_segment(s) && continue
        for i in eachindex(s.verts)
            for q in dirip
                if norm(s.verts[i] - q) < 1e-10
                    s.frozen[i] = true
                    break
                end
            end
        end
    end
    return d
end

# =============================================================================
# Polyline resample / smooth
# =============================================================================

function _arclength(verts::AbstractVector{<:SVector{2}})
    n = length(verts)
    s = zeros(n)
    @inbounds for i in 2:n
        s[i] = s[i - 1] + norm(verts[i] - verts[i - 1])
    end
    return s
end

"""Resample an open or closed polyline to `n` vertices (closed: `n` unique)."""
function resample_polyline(verts::AbstractVector{<:SVector{2}}, n::Integer; closed::Bool=false)
    n = max(Int(n), closed ? 3 : 2)
    pts = Point2D[Point2D(p[1], p[2]) for p in verts]
    if closed && (length(pts) < 2 || !(pts[1] ≈ pts[end]))
        push!(pts, pts[1])
    end
    s = _arclength(pts)
    L = s[end]
    L < 1e-16 && return pts[1:min(end, n)]
    t = closed ? range(0, L; length=n + 1)[1:n] : range(0, L; length=n)
    out = Vector{Point2D}(undef, n)
    j = 1
    @inbounds for i in 1:n
        ti = t[i]
        while j < length(s) - 1 && s[j + 1] < ti
            j += 1
        end
        ds = s[j + 1] - s[j]
        α = ds < 1e-16 ? 0.0 : (ti - s[j]) / ds
        out[i] = (1 - α) * pts[j] + α * pts[j + 1]
    end
    return out
end

function chaikin_smooth!(verts::Vector{Point2D}; closed::Bool=false, passes::Integer=1)
    passes < 1 && return verts
    pts = verts
    for _ in 1:passes
        n = length(pts)
        n < 3 && break
        if closed
            nxt = Point2D[]
            sizehint!(nxt, 2n)
            for i in 1:n
                a = pts[i]
                b = pts[mod1(i + 1, n)]
                push!(nxt, 0.75 * a + 0.25 * b)
                push!(nxt, 0.25 * a + 0.75 * b)
            end
            pts = nxt
        else
            nxt = Point2D[pts[1]]
            for i in 1:(n - 1)
                a = pts[i]
                b = pts[i + 1]
                push!(nxt, 0.75 * a + 0.25 * b)
                push!(nxt, 0.25 * a + 0.75 * b)
            end
            push!(nxt, pts[end])
            pts = nxt
        end
    end
    empty!(verts)
    append!(verts, pts)
    return verts
end

"""Outward unit normals at loop vertices (right of CCW tangent = outward)."""
function vertex_outward_normals(verts::AbstractVector{<:SVector{2}}; closed::Bool=true)
    n = length(verts)
    nrm = Vector{Point2D}(undef, n)
    @inbounds for i in 1:n
        im = i == 1 ? (closed ? n : 1) : i - 1
        ip = i == n ? (closed ? 1 : n) : i + 1
        t = verts[ip] - verts[im]
        L = norm(t)
        if L < 1e-16
            nrm[i] = Point2D(0.0, 0.0)
        else
            t /= L
            nrm[i] = Point2D(t[2], -t[1])   # tan2normal
        end
    end
    return nrm
end

# =============================================================================
# Loops → BEMdata
# =============================================================================

function _ensure_loop_orientation!(segs::Vector{BoundarySegment}; hole::Bool=false)
    verts = loop_vertices(segs)
    a = polygon_area(verts)
    want_negative = hole          # holes CW
    if (a > 0 && want_negative) || (a < 0 && !want_negative)
        reverse!(segs)
        for s in segs
            reverse!(s.verts)
            reverse!(s.frozen)
        end
    end
    return segs
end

"""
    bemdata_from_loops(design; d_min=0.01) -> BEMdata

Discontinuous collocation of degree `design.degree` on every segment, plus a
Cartesian interior grid clipped by the current topology. Does **not** call Gmsh.
"""
function bemdata_from_loops(d::TopologyDesign; d_min::Real=0.01)
    p = d.degree
    p ≥ 1 || throw(ArgumentError("degree must be ≥ 1"))
    qsi, wi = discontinuous_nodes_weights(p)
    Ngeo, dNgeo = shapefun(Equispaced(p), qsi)
    poly = Legendre(p)

    NOS = Point2D[]
    normal = Point2D[]
    ELEM = Element[]
    BC = Int[]
    BV = Float64[]
    n_per = p + 1

    for (iloop, segs) in enumerate(d.loops)
        _ensure_loop_orientation!(segs; hole=iloop > 1)
        for (iseg, seg) in enumerate(segs)
            nv = length(seg.verts)
            nv < 2 && continue
            # resample to n_el + 1 geometric corners, then p+1 nodes per element
            corners = resample_polyline(seg.verts, seg.n_el + 1; closed=false)
            for e in 1:seg.n_el
                a = corners[e]
                b = corners[e + 1]
                X = [_lerp_pt(a, b, k / p) for k in 0:p]
                idx0 = length(NOS) + 1
                idx = collect(idx0:(idx0 + n_per - 1))
                colloc = Ngeo * X
                dx = dNgeo * X
                J = norm.(dx)
                nrm = tan2normal.(dx ./ J)
                L = abs(dot(J, wi))
                append!(NOS, colloc)
                append!(normal, nrm)
                _append_segment_bc!(BC, BV, seg, n_per)
                push!(ELEM, Element(idx, collect(Float64, J), L, iseg))
            end
        end
    end

    dad = _bemdata_2d(d.name, ELEM, poly, wi, NOS, normal, d.properties, BC, BV, Point2D[])
    internals = sample_internal_points(d; d_min=d_min)
    set_internal_nodes!(dad, internals)
    set_cache!(dad; topology=d)
    return dad
end

@inline _lerp_pt(a, b, ξ) = (1 - ξ) * a + ξ * b

function _append_segment_bc!(BC, BV, seg::BoundarySegment, n_per::Integer)
    nd = length(seg.bc)
    for _ in 1:n_per
        for k in 1:nd
            push!(BC, seg.bc[k])
            push!(BV, seg.value[k])
        end
    end
    return nothing
end

"""Traction-free hole loop (Laplace `q=0` or elasticity `t=0`)."""
function _free_segment(pts, old::Vector{BoundarySegment}, nel::Integer)
    nd = isempty(old) ? 1 : length(old[1].bc)
    return BoundarySegment(pts; bc=ones(Int, nd), value=zeros(nd), n_el=nel)
end

function hole_segment(pts, d::TopologyDesign, nel::Integer)
    if d.properties isa Elasticity
        return BoundarySegment(pts; bc=[1, 1], value=[0.0, 0.0], n_el=nel)
    end
    return BoundarySegment(pts; bc=1, value=0.0, n_el=nel)
end

function sample_internal_points(d::TopologyDesign; d_min::Real=0.01)
    outer = loop_vertices(d.loops[1])
    isempty(outer) && return Point2D[]
    xmin = minimum(p[1] for p in outer)
    xmax = maximum(p[1] for p in outer)
    ymin = minimum(p[2] for p in outer)
    ymax = maximum(p[2] for p in outer)
    lx = max(xmax - xmin, 1e-12)
    dm = d_min * lx
    nx, ny = d.nx_int, d.ny_int
    xs = range(xmin, xmax; length=nx)
    ys = range(ymin, ymax; length=ny)
    pts = Point2D[]
    allverts = [loop_vertices(s) for s in d.loops]
    for x in xs, y in ys
        p = Point2D(x, y)
        in_design(p, d) || continue
        too_close = false
        for verts in allverts
            if _dist_to_polyline(p, verts) < dm
                too_close = true
                break
            end
        end
        too_close && continue
        push!(pts, p)
    end
    return pts
end

sample_internal_points!(dad::BEMdata, d::TopologyDesign; kwargs...) =
    set_internal_nodes!(dad, sample_internal_points(d; kwargs...))

function _dist_to_polyline(p, verts)
    n = length(verts)
    dmin = Inf
    @inbounds for i in 1:n
        a = verts[i]
        b = verts[i == n ? 1 : i + 1]
        dmin = min(dmin, _dist_point_seg(p, a, b))
    end
    return dmin
end

function _dist_point_seg(p, a, b)
    ab = b - a
    L2 = dot(ab, ab)
    L2 < 1e-30 && return norm(p - a)
    t = clamp(dot(p - a, ab) / L2, 0.0, 1.0)
    return norm(p - (a + t * ab))
end

# =============================================================================
# BEMdata → loops (initial Gmsh mesh)
# =============================================================================

"""
    extract_loops(dad) -> TopologyDesign

Recover polyline loops from a 2-D `BEMdata` by chaining element
endpoints (Legendre interpolant at `ξ = ±1`). Used once after `format2d`.
"""
function extract_loops(dad::BEMdata; nx_int=20, ny_int=20, name=dad.name)
    p = degree(dad.element_type)
    poly = dad.element_type
    Nends, _ = shapefun(poly, [-1.0, 1.0])
    n_el = length(dad.elements)
    ends = Vector{NTuple{2,Point2D}}(undef, n_el)
    bcs = Vector{Tuple{Vector{Int},Vector{Float64}}}(undef, n_el)
    @inbounds for (e, el) in enumerate(dad.elements)
        X = dad.Nodes[el.index]
        p0 = Nends[1, 1] * X[1]
        p1 = Nends[2, 1] * X[1]
        for k in 2:length(X)
            p0 += Nends[1, k] * X[k]
            p1 += Nends[2, k] * X[k]
        end
        ends[e] = (Point2D(p0[1], p0[2]), Point2D(p1[1], p1[2]))
        i0 = el.index[1]
        ndof = length(dad.BC) ÷ max(dad.n, 1)
        i0d = ndof * (i0 - 1)
        bcs[e] = (Int[dad.BC[i0d + k] for k in 1:ndof],
                  Float64[dad.BV[i0d + k] for k in 1:ndof])
    end

    used = falses(n_el)
    loops = Vector{Vector{BoundarySegment}}()
    for seed in 1:n_el
        used[seed] && continue
        chain = Int[seed]
        used[seed] = true
        # grow forward from p1
        cur = ends[seed][2]
        growing = true
        while growing
            growing = false
            for e in 1:n_el
                used[e] && continue
                a, b = ends[e]
                if a ≈ cur
                    push!(chain, e)
                    used[e] = true
                    cur = b
                    growing = true
                    break
                elseif b ≈ cur
                    ends[e] = (b, a)
                    push!(chain, e)
                    used[e] = true
                    cur = a
                    growing = true
                    break
                end
            end
        end
        segs = _chain_to_segments(chain, ends, bcs)
        push!(loops, segs)
    end

    # outer = largest |area|
    areas = [abs(polygon_area(loop_vertices(s))) for s in loops]
    perm = sortperm(areas; rev=true)
    loops = loops[perm]
    _ensure_loop_orientation!(loops[1]; hole=false)
    for k in 2:length(loops)
        _ensure_loop_orientation!(loops[k]; hole=true)
    end
    kk = dad.properties isa Laplace ? dad.properties.k : 1.0
    d = TopologyDesign(loops; degree=p, k=kk, nx_int=nx_int, ny_int=ny_int, name=name,
        properties=dad.properties)
    freeze_dirichlet!(d)
    return d
end

function _chain_to_segments(chain, ends, bcs)
    segs = BoundarySegment[]
    i = 1
    n = length(chain)
    while i <= n
        e0 = chain[i]
        bc, val = bcs[e0]
        verts = Point2D[ends[e0][1], ends[e0][2]]
        n_el = 1
        i += 1
        while i <= n
            e = chain[i]
            bcs[e][1] == bc && bcs[e][2] == val || break
            push!(verts, ends[e][2])
            n_el += 1
            i += 1
        end
        frozen = fill(_is_fixed_bc(bc, val), length(verts))
        push!(segs, BoundarySegment(verts; bc=bc, value=val, frozen=frozen, n_el=n_el))
    end
    return segs
end

extract_loops!(dad::BEMdata; kwargs...) = extract_loops(dad; kwargs...)
