# Marching squares on a Cartesian grid. Iso-points live on unique grid edges so
# adjacent cells share vertices exactly; segments are then stitched into polylines.

export marching_squares, stitch_segments

# case 0..15: edge pairs. Edges of a cell: 1=bottom, 2=right, 3=top, 4=left.
const _MS_CASES = (
    NTuple{2,Int}[],
    [(4, 1)],
    [(1, 2)],
    [(4, 2)],
    [(2, 3)],
    [(4, 1), (2, 3)],
    [(1, 3)],
    [(4, 3)],
    [(3, 4)],
    [(3, 1)],
    [(1, 2), (3, 4)],
    [(3, 2)],
    [(2, 4)],
    [(2, 1)],
    [(1, 4)],
    NTuple{2,Int}[],
)

function _ms_lerp(va, vb)
    den = vb - va
    abs(den) < 1e-30 && return 0.5
    return clamp(-va / den, 0.0, 1.0)
end

"""Unique iso-point on the horizontal grid edge `(i,j) — (i+1,j)`."""
function _hpoint(xs, ys, Z, level, i, j)
    t = _ms_lerp(Z[i, j] - level, Z[i + 1, j] - level)
    return Point2D((1 - t) * xs[i] + t * xs[i + 1], ys[j])
end

"""Unique iso-point on the vertical grid edge `(i,j) — (i,j+1)`."""
function _vpoint(xs, ys, Z, level, i, j)
    t = _ms_lerp(Z[i, j] - level, Z[i, j + 1] - level)
    return Point2D(xs[i], (1 - t) * ys[j] + t * ys[j + 1])
end

function _cell_edge_point(xs, ys, Z, level, i, j, edge)
    edge == 1 && return _hpoint(xs, ys, Z, level, i, j)         # bottom
    edge == 2 && return _vpoint(xs, ys, Z, level, i + 1, j)     # right
    edge == 3 && return _hpoint(xs, ys, Z, level, i, j + 1)     # top
    return _vpoint(xs, ys, Z, level, i, j)                      # left
end

"""
    marching_squares(xs, ys, Z, level) -> Vector{Vector{Point2D}}

`Z[i,j]` is the sample at `(xs[i], ys[j])`. Closed curves have `pts[1] ≈ pts[end]`.
"""
function marching_squares(xs::AbstractVector, ys::AbstractVector, Z::AbstractMatrix, level::Real)
    nx, ny = length(xs), length(ys)
    size(Z, 1) == nx && size(Z, 2) == ny ||
        throw(DimensionMismatch("Z size $(size(Z)) vs $(nx)×$(ny)"))
    segs = Vector{NTuple{2,Point2D}}()
    @inbounds for j in 1:(ny - 1), i in 1:(nx - 1)
        bits = 0
        Z[i, j] > level && (bits |= 1)
        Z[i + 1, j] > level && (bits |= 2)
        Z[i + 1, j + 1] > level && (bits |= 4)
        Z[i, j + 1] > level && (bits |= 8)
        for (e1, e2) in _MS_CASES[bits + 1]
            p1 = _cell_edge_point(xs, ys, Z, level, i, j, e1)
            p2 = _cell_edge_point(xs, ys, Z, level, i, j, e2)
            p1 == p2 && continue
            push!(segs, (p1, p2))
        end
    end
    return stitch_segments(segs)
end

function stitch_segments(segs::Vector{NTuple{2,Point2D}}; atol=nothing)
    isempty(segs) && return Vector{Point2D}[]
    if atol === nothing
        s = 0.0
        @inbounds for (a, b) in segs
            s += norm(b - a)
        end
        atol = max(1e-12, 1e-9 * s / length(segs))
    end
    used = falses(length(segs))
    lines = Vector{Vector{Point2D}}()

    function nearest_unused(p)
        kbest, dbest, other = 0, atol, p
        @inbounds for k in eachindex(segs)
            used[k] && continue
            a, b = segs[k]
            da = norm(a - p)
            db = norm(b - p)
            if da <= dbest
                kbest, dbest, other = k, da, b
            end
            if db <= dbest
                kbest, dbest, other = k, db, a
            end
        end
        return kbest, other
    end

    for s0 in eachindex(segs)
        used[s0] && continue
        a0, b0 = segs[s0]
        used[s0] = true
        pts = Point2D[a0, b0]
        growing = true
        while growing
            k, nxt = nearest_unused(pts[end])
            if k == 0
                growing = false
            else
                used[k] = true
                push!(pts, nxt)
            end
        end
        growing = true
        while growing
            k, nxt = nearest_unused(pts[1])
            if k == 0
                growing = false
            else
                used[k] = true
                pushfirst!(pts, nxt)
            end
        end
        if length(pts) ≥ 3 && norm(pts[1] - pts[end]) ≤ 10 * atol
            pts[end] = pts[1]
        end
        length(pts) ≥ 2 && push!(lines, pts)
    end
    return lines
end
