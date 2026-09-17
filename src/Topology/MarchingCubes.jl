# Marching tetrahedra on a Cartesian grid (5 tets / cube, even/odd to avoid cracks).

export marching_cubes

# Cube corners: 0=(i,j,k) … 7=(i,j+1,k+1) as in Topology3D density grids.
# 5-tet split (even / odd cells) so shared faces match.
const _TETS_EVEN = (
    (1, 2, 3, 6),
    (1, 3, 4, 8),
    (1, 6, 8, 5),
    (3, 6, 7, 8),
    (1, 3, 6, 8),
)
const _TETS_ODD = (
    (2, 3, 4, 7),
    (1, 2, 4, 5),
    (2, 6, 7, 5),
    (4, 7, 8, 5),
    (2, 4, 7, 5),
)

@inline function _mc_lerp(pa, pb, va, vb, level)
    t = _ms_lerp(va - level, vb - level)
    return (1 - t) * pa + t * pb
end

function _march_tet!(tris, P, V, level)
    bits = 0
    @inbounds for a in 1:4
        V[a] > level && (bits |= 1 << (a - 1))
    end
    (bits == 0 || bits == 15) && return
    ins = Int[]
    outs = Int[]
    @inbounds for a in 1:4
        if (bits >> (a - 1)) & 1 == 1
            push!(ins, a)
        else
            push!(outs, a)
        end
    end
    if length(ins) == 1
        a = ins[1]
        push!(tris, (
            _mc_lerp(P[a], P[outs[1]], V[a], V[outs[1]], level),
            _mc_lerp(P[a], P[outs[2]], V[a], V[outs[2]], level),
            _mc_lerp(P[a], P[outs[3]], V[a], V[outs[3]], level),
        ))
    elseif length(ins) == 3
        # complement of one-out: one triangle
        b = outs[1]
        push!(tris, (
            _mc_lerp(P[ins[1]], P[b], V[ins[1]], V[b], level),
            _mc_lerp(P[ins[2]], P[b], V[ins[2]], V[b], level),
            _mc_lerp(P[ins[3]], P[b], V[ins[3]], V[b], level),
        ))
    else
        # two inside: non-crossing quad
        p11 = _mc_lerp(P[ins[1]], P[outs[1]], V[ins[1]], V[outs[1]], level)
        p12 = _mc_lerp(P[ins[1]], P[outs[2]], V[ins[1]], V[outs[2]], level)
        p21 = _mc_lerp(P[ins[2]], P[outs[1]], V[ins[2]], V[outs[1]], level)
        p22 = _mc_lerp(P[ins[2]], P[outs[2]], V[ins[2]], V[outs[2]], level)
        push!(tris, (p11, p12, p22))
        push!(tris, (p11, p22, p21))
    end
    return
end

"""
    marching_cubes(xs, ys, zs, Z, level) -> Vector{NTuple{3,Point3D}}

Iso-surface of `Z[i,j,k]` at `(xs[i], ys[j], zs[k])`. Triangles from a
5-tetrahedron split of each voxel (even/odd so faces match).
"""
function marching_cubes(xs::AbstractVector, ys::AbstractVector, zs::AbstractVector,
        Z::AbstractArray{<:Real,3}, level::Real)
    nx, ny, nz = length(xs), length(ys), length(zs)
    size(Z) == (nx, ny, nz) || throw(DimensionMismatch("Z size $(size(Z)) vs ($nx,$ny,$nz)"))
    tris = NTuple{3,Point3D}[]
    (nx < 2 || ny < 2 || nz < 2) && return tris
    P = Vector{Point3D}(undef, 4)
    V = Vector{Float64}(undef, 4)
    @inbounds for k in 1:(nz - 1), j in 1:(ny - 1), i in 1:(nx - 1)
        odd = isodd(i + j + k)
        tets = odd ? _TETS_ODD : _TETS_EVEN
        corners = (
            (i, j, k), (i + 1, j, k), (i + 1, j + 1, k), (i, j + 1, k),
            (i, j, k + 1), (i + 1, j, k + 1), (i + 1, j + 1, k + 1), (i, j + 1, k + 1),
        )
        for tet in tets
            for a in 1:4
                ii, jj, kk = corners[tet[a]]
                P[a] = Point3D(xs[ii], ys[jj], zs[kk])
                V[a] = Z[ii, jj, kk]
            end
            _march_tet!(tris, P, V, level)
        end
    end
    return tris
end
