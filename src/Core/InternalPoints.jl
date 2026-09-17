# Regular interior collocation grids (MATLAB `gera_p_in`, 2-D and 3-D).
# Inside: even–odd ray test on the BEM surface. Clearance: closest-point
# distance to the piecewise-linear interpolant of each element.

export internal_grid, internal_grid!, gera_p_in, internal_layer, internal_layer!
export point_in_domain

"""
    internal_grid(dad, nx, ny; d_min=0.01, layout=:gera) -> Vector{Point2D}
    internal_grid(dad, nx, ny, nz; d_min=0.01, layout=:gera) -> Vector{Point3D}

Cartesian candidates in the bounding box of `dad.Nodes`. Kept if inside `Ω`
and at least `d_min * L_max` from `Γ` (`d_min=0` skips clearance).

`layout`:
- `:gera` — MATLAB `gera_p_in`, first layer at `L/(n+1)` (default)
- `:cell` — cell centres, first layer at `0.5 L/n`
- `:cheb` — Chebyshev–Gauss on each axis (clustered toward the walls)

For **few** poles on a fine `Γ`, use [`internal_layer`](@ref) so the first
layer sits at `O(h_Γ)` no matter how coarse the fill is.
"""
function internal_grid(dad::BEMdata, nx::Integer, ny::Integer;
        d_min::Real=0.01, layout::Symbol=:gera)
    dad.dimension == 2 || throw(ArgumentError(
        "internal_grid(dad, nx, ny) is 2-D; pass nz for 3-D"))
    nx >= 1 && ny >= 1 || throw(ArgumentError("nx, ny must be ≥ 1"))
    bb = _boundary_bbox(dad)
    xs = _axis_nodes(nx, bb.xmin, bb.xmax, layout)
    ys = _axis_nodes(ny, bb.ymin, bb.ymax, layout)
    clear = float(d_min) * _max_el_length(dad)
    pts = Point2D[]
    sizehint!(pts, nx * ny)
    @inbounds for y in ys, x in xs
        p = Point2D(x, y)
        point_in_domain(dad, p) || continue
        clear > 0 && _dist_to_boundary(dad, p) < clear && continue
        push!(pts, p)
    end
    return pts
end

function internal_grid(dad::BEMdata, nx::Integer, ny::Integer, nz::Integer;
        d_min::Real=0.01, layout::Symbol=:gera)
    dad.dimension == 3 || throw(ArgumentError(
        "internal_grid(dad, nx, ny, nz) is 3-D; omit nz for 2-D"))
    nx >= 1 && ny >= 1 && nz >= 1 || throw(ArgumentError("nx, ny, nz must be ≥ 1"))
    bb = _boundary_bbox(dad)
    xs = _axis_nodes(nx, bb.xmin, bb.xmax, layout)
    ys = _axis_nodes(ny, bb.ymin, bb.ymax, layout)
    zs = _axis_nodes(nz, bb.zmin, bb.zmax, layout)
    clear = float(d_min) * _max_el_length(dad)
    pts = Point3D[]
    sizehint!(pts, nx * ny * nz)
    tris = _surface_triangles(dad)
    @inbounds for z in zs, y in ys, x in xs
        p = Point3D(x, y, z)
        _inside_surface(p, tris) || continue
        clear > 0 && _dist_to_boundary(dad, p) < clear && continue
        push!(pts, p)
    end
    return pts
end

function _axis_nodes(n::Integer, a, b, layout::Symbol)
    n >= 1 || throw(ArgumentError("n ≥ 1"))
    if layout === :gera
        return [a + (b - a) * i / (n + 1) for i in 1:n]
    elseif layout === :cell
        return [a + (b - a) * (i - 0.5) / n for i in 1:n]
    elseif layout === :cheb
        return [a + (b - a) * (1 - cos(π * (i - 0.5) / n)) / 2 for i in 1:n]
    end
    throw(ArgumentError("layout must be :gera, :cell, or :cheb (got $layout)"))
end

"""`set_internal_nodes!(dad, internal_grid(dad, …))`."""
internal_grid!(dad::BEMdata, nx::Integer, ny::Integer; kwargs...) =
    set_internal_nodes!(dad, internal_grid(dad, nx, ny; kwargs...))
internal_grid!(dad::BEMdata, nx::Integer, ny::Integer, nz::Integer; kwargs...) =
    set_internal_nodes!(dad, internal_grid(dad, nx, ny, nz; kwargs...))

"""Alias of [`internal_grid`](@ref) (MATLAB `gera_p_in`)."""
const gera_p_in = internal_grid

"""
    internal_layer(dad; δ=nothing, every=1, fill=0) -> Vector{<:Point}

Few-pole strategy: one inward pole per element (mid-point along `-n`),
then optionally fill the core with a cell-centred `fill×fill`
(`fill×fill×fill` in 3-D) lattice.

- `δ=nothing` (default) uses the median element length, so the layer sits
  at `O(h_Γ)` even when `fill` is small. That is what IBP's 21-neighbour
  stencil needs; a coarse `internal_grid` on a fine `Γ` does not.
- `every=k` keeps every `k`-th **element** (mid-point along `-n`).
- Core fill points closer than `2δ` to `Γ` are dropped.

Holes: `n` is outward from `Ω`, so `-n` from a hole edge still lands in `Ω`.
"""
function internal_layer(dad::BEMdata; δ=nothing, every::Integer=1, fill::Integer=0)
    every >= 1 || throw(ArgumentError("every must be ≥ 1"))
    fill >= 0 || throw(ArgumentError("fill must be ≥ 0"))
    δf = δ === nothing ? 1.25 * _typical_h(dad) : float(δ)
    δf > 0 || throw(ArgumentError("δ must be positive"))
    pts = _offset_layer(dad, δf, every)
    if fill > 0
        append!(pts, _core_fill(dad, fill, 2 * δf))
        pts = _dedup_points(pts, 0.25 * δf)
    end
    return pts
end

internal_layer!(dad::BEMdata; kwargs...) =
    set_internal_nodes!(dad, internal_layer(dad; kwargs...))

"""In-plane mesh size: square-like `L/(n_el/4)`, cube-like `L/√(n_el/6)`."""
function _typical_h(dad::BEMdata)
    bb = _boundary_bbox(dad)
    L = max(bb.xmax - bb.xmin, bb.ymax - bb.ymin,
        dad.dimension == 3 ? (bb.zmax - bb.zmin) : 0.0, 1e-12)
    ne = max(length(dad.elements), 1)
    return dad.dimension == 2 ? L / (ne / 4) : L / sqrt(ne / 6)
end

function _offset_layer(dad::BEMdata, δ, every)
    T = eltype(getfield(dad, :collocation))
    pts = T[]
    @inbounds for (e, el) in enumerate(dad.elements)
        ((e - 1) % every == 0) || continue
        idx = el.index
        isempty(idx) && continue
        mid = sum(point(dad, i) for i in idx) / length(idx)
        nrm = sum(dad.Normal[i] for i in idx)
        nn = norm(nrm)
        nn < 1e-16 && continue
        p = mid - (δ / nn) * nrm
        point_in_domain(dad, p) || continue
        # 2-D 90° corner midpoints sit 0.5 L from the adjacent edge — drop them.
        dmin = dad.dimension == 2 ? 0.75 * δ : 0.15 * δ
        _dist_to_boundary(dad, p) < dmin && continue
        push!(pts, p)
    end
    return _dedup_points(pts, 0.25 * δ)
end

function _core_fill(dad::BEMdata, nfill::Integer, clear)
    bb = _boundary_bbox(dad)
    xs = _axis_nodes(nfill, bb.xmin, bb.xmax, :cell)
    ys = _axis_nodes(nfill, bb.ymin, bb.ymax, :cell)
    pts = eltype(getfield(dad, :collocation))[]
    if dad.dimension == 2
        @inbounds for y in ys, x in xs
            p = Point2D(x, y)
            point_in_domain(dad, p) || continue
            _dist_to_boundary(dad, p) < clear && continue
            push!(pts, p)
        end
        return pts
    end
    zs = _axis_nodes(nfill, bb.zmin, bb.zmax, :cell)
    tris = _surface_triangles(dad)
    @inbounds for z in zs, y in ys, x in xs
        p = Point3D(x, y, z)
        _inside_surface(p, tris) || continue
        _dist_to_boundary(dad, p) < clear && continue
        push!(pts, p)
    end
    return pts
end

function _dedup_points(pts, tol)
    isempty(pts) && return pts
    tol2 = tol * tol
    kept = eltype(pts)[pts[1]]
    @inbounds for i in 2:length(pts)
        p = pts[i]
        dup = false
        for q in kept
            if sum(abs2, p - q) < tol2
                dup = true
                break
            end
        end
        dup || push!(kept, p)
    end
    return kept
end

"""
    point_in_domain(dad, p) -> Bool

`true` if `p` is inside `Ω`. 2-D: domain-cell even–odd when `dad.cells`
exists, otherwise even–odd on boundary chords. 3-D: even–odd ray vs
surface triangles. A point clearly outside the bounding box is `false`;
a point exactly on `Γ` may count either way (even–odd).
"""
function point_in_domain(dad::BEMdata, p::SVector{2})
    dad.dimension == 2 || throw(ArgumentError("2-D point vs $(dad.dimension)-D mesh"))
    if has_cache(dad, :cells)
        cells = dad.cells
        cells isa Vector{DomainCell} && !isempty(cells) && return _point_in_cells(p, cells)
    end
    return _inside_segments(p, _boundary_segments(dad))
end

function point_in_domain(dad::BEMdata, p::SVector{3})
    dad.dimension == 3 || throw(ArgumentError("3-D point vs $(dad.dimension)-D mesh"))
    return _inside_surface(p, _surface_triangles(dad))
end

# ---------------------------------------------------------------------------
# 2-D even–odd on chords
# ---------------------------------------------------------------------------

function _boundary_segments(dad::BEMdata)
    segs = Vector{Tuple{Point2D,Point2D}}(undef, 0)
    @inbounds for el in dad.elements
        g = el.geo
        if length(g) >= 2
            for k in 1:length(g)-1
                push!(segs, (g[k], g[k+1]))
            end
        else
            X = dad.Nodes[el.index]
            length(X) >= 2 && push!(segs, (Point2D(X[1][1], X[1][2]),
                Point2D(X[end][1], X[end][2])))
        end
    end
    return segs
end

function _inside_segments(p::SVector{2}, segs)
    inside = false
    px, py = p[1], p[2]
    @inbounds for (a, b) in segs
        yi, yj = a[2], b[2]
        if (yi > py) != (yj > py)
            xint = (b[1] - a[1]) * (py - yi) / (yj - yi + 1e-30) + a[1]
            px < xint && (inside = !inside)
        end
    end
    return inside
end

# ---------------------------------------------------------------------------
# 3-D even–odd vs surface triangles
# ---------------------------------------------------------------------------

function _surface_triangles(dad::BEMdata)
    tris = NTuple{3,Point3D}[]
    @inbounds for el in dad.elements
        c = _face_corners(dad, el)
        n = length(c)
        n < 3 && continue
        if n == 3 || (n >= 4 && norm(c[4] - c[3]) < 1e-14)
            push!(tris, (c[1], c[2], c[3]))
        else
            push!(tris, (c[1], c[2], c[4]))
            push!(tris, (c[1], c[4], c[3]))
        end
    end
    return tris
end

function _inside_surface(p::SVector{3}, tris)
    # Irrational-ish direction so the ray does not ride a mesh edge/vertex
    # (even–odd fails if a hit is shared by two triangles).
    dir = SVector(1.0, 0.031415926, 0.017453292)
    hits = 0
    @inbounds for (a, b, c) in tris
        hits += _ray_hits_triangle(p, dir, a, b, c)
    end
    return isodd(hits)
end

"""Möller–Trumbore. `true` if the ray `orig + t dir`, `t > 0`, hits the triangle."""
function _ray_hits_triangle(orig, dir, v0, v1, v2)
    e1 = v1 - v0
    e2 = v2 - v0
    nrm = cross(e1, e2)
    dot(nrm, nrm) < 1e-30 && return 0
    pvec = cross(dir, e2)
    det = dot(e1, pvec)
    abs(det) < 1e-16 && return 0
    invdet = 1 / det
    tvec = orig - v0
    u = dot(tvec, pvec) * invdet
    (u <= 1e-12 || u >= 1 - 1e-12) && return 0
    qvec = cross(tvec, e1)
    v = dot(dir, qvec) * invdet
    (v <= 1e-12 || u + v >= 1 - 1e-12) && return 0
    t = dot(e2, qvec) * invdet
    return t > 1e-12 ? 1 : 0
end

function _face_corners(dad::BEMdata, el::Element)
    X = [point(dad, i) for i in el.index]
    n = length(X)
    n == 3 && return (X[1], X[2], X[3])
    # Collocation may be Gauss (inset); evaluate the parent interpolant at ξ,η=±1.
    N, _, _ = shapefun2D(dad.element_type, [-1.0, 1.0])
    C = [_mix_shape(N, k, X) for k in 1:size(N, 1)]
    length(C) >= 4 && return (C[1], C[2], C[3], C[4])
    length(C) == 3 && return (C[1], C[2], C[3])
    n >= 4 && return (X[1], X[2], X[3], X[4])
    return (X[1], X[2], X[end])
end

function _mix_shape(N, k, X)
    s = zero(X[1])
    @inbounds for j in eachindex(X)
        s += N[k, j] * X[j]
    end
    return s
end

# ---------------------------------------------------------------------------
# Distance and bbox
# ---------------------------------------------------------------------------

function _el_xg(dad::BEMdata, el::Element)
    if dad.dimension == 2
        return !isempty(el.geo) ? el.geo : dad.Nodes[el.index]
    end
    c = _face_corners(dad, el)
    return collect(c)
end

function _dist_to_boundary(dad::BEMdata, p)
    dmin = Inf
    @inbounds for el in dad.elements
        xg = _el_xg(dad, el)
        isempty(xg) && continue
        lo, hi = _minmax_nodes(xg)
        _dist_aabb(p, lo, hi) >= dmin && continue
        dmin = min(dmin, _dist_element(p, xg))
    end
    return dmin
end

function _max_el_length(dad::BEMdata)
    L = 0.0
    @inbounds for el in dad.elements
        el.Length > L && (L = el.Length)
    end
    return L > 0 ? L : 1.0
end

function _boundary_bbox(dad::BEMdata)
    n = Int(dad.n)
    n >= 1 || throw(ArgumentError("BEMdata has no boundary nodes"))
    p0 = point(dad, 1)
    xmin = xmax = p0[1]
    ymin = ymax = p0[2]
    zmin = zmax = length(p0) == 3 ? p0[3] : 0.0
    @inbounds for i in 2:n
        p = point(dad, i)
        xmin = min(xmin, p[1]); xmax = max(xmax, p[1])
        ymin = min(ymin, p[2]); ymax = max(ymax, p[2])
        if length(p) == 3
            zmin = min(zmin, p[3]); zmax = max(zmax, p[3])
        end
    end
    if dad.dimension == 2
        @inbounds for el in dad.elements
            for q in el.geo
                xmin = min(xmin, q[1]); xmax = max(xmax, q[1])
                ymin = min(ymin, q[2]); ymax = max(ymax, q[2])
            end
        end
    end
    return (xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, zmin=zmin, zmax=zmax)
end
