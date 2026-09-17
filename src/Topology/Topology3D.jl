# 3-D topology on a fixed surface mesh: DIBEM-SIMP / DT-ρ (no explicit Γ motion).

export heat_cube_3d, cantilever_cube_3d, box_interior_grid
export mesh_box_3d, apply_heat_bridge_bcs!, apply_cantilever_bcs!
export export_vtk_density, export_vtk_isosurface
export density_grid_3d, cut_density_3d!, grid_solid_volume
export bemdata_from_iso, extract_closed_cavities

# =============================================================================
# Box surface mesh
# =============================================================================

"""Build the six faces of `[0,Lx]×[0,Ly]×[0,Lz]` as transfinite surfaces."""
function _box_surfaces!(; Lx=1.0, Ly=1.0, Lz=1.0, ndiv=2, nome="box3d", recombine::Bool=true)
    nx, ny, nz = ndiv isa Integer ? (Int(ndiv), Int(ndiv), Int(ndiv)) : (Int(ndiv[1]), Int(ndiv[2]), Int(ndiv[3]))
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(nome)
    lc = min(Lx / nx, Ly / ny, Lz / nz)
    p = [
        gmsh.model.geo.addPoint(0, 0, 0, lc),
        gmsh.model.geo.addPoint(Lx, 0, 0, lc),
        gmsh.model.geo.addPoint(Lx, Ly, 0, lc),
        gmsh.model.geo.addPoint(0, Ly, 0, lc),
        gmsh.model.geo.addPoint(0, 0, Lz, lc),
        gmsh.model.geo.addPoint(Lx, 0, Lz, lc),
        gmsh.model.geo.addPoint(Lx, Ly, Lz, lc),
        gmsh.model.geo.addPoint(0, Ly, Lz, lc),
    ]
    l1 = gmsh.model.geo.addLine(p[1], p[2])   # x on z=0 y=0
    l2 = gmsh.model.geo.addLine(p[2], p[3])   # y on z=0 x=Lx
    l3 = gmsh.model.geo.addLine(p[3], p[4])
    l4 = gmsh.model.geo.addLine(p[4], p[1])
    l5 = gmsh.model.geo.addLine(p[5], p[6])
    l6 = gmsh.model.geo.addLine(p[6], p[7])
    l7 = gmsh.model.geo.addLine(p[7], p[8])
    l8 = gmsh.model.geo.addLine(p[8], p[5])
    l9 = gmsh.model.geo.addLine(p[1], p[5])
    l10 = gmsh.model.geo.addLine(p[2], p[6])
    l11 = gmsh.model.geo.addLine(p[3], p[7])
    l12 = gmsh.model.geo.addLine(p[4], p[8])
    cl_bot = gmsh.model.geo.addCurveLoop([-l4, -l3, -l2, -l1])
    cl_top = gmsh.model.geo.addCurveLoop([l5, l6, l7, l8])
    cl_f = gmsh.model.geo.addCurveLoop([l1, l10, -l5, -l9])
    cl_r = gmsh.model.geo.addCurveLoop([l2, l11, -l6, -l10])
    cl_b = gmsh.model.geo.addCurveLoop([l3, l12, -l7, -l11])
    cl_l = gmsh.model.geo.addCurveLoop([l4, l9, -l8, -l12])
    s_bot = gmsh.model.geo.addPlaneSurface([cl_bot])
    s_top = gmsh.model.geo.addPlaneSurface([cl_top])
    s_f = gmsh.model.geo.addPlaneSurface([cl_f])
    s_r = gmsh.model.geo.addPlaneSurface([cl_r])
    s_b = gmsh.model.geo.addPlaneSurface([cl_b])
    s_l = gmsh.model.geo.addPlaneSurface([cl_l])
    gmsh.model.geo.synchronize()
    for (ℓ, n) in zip((l1, l3, l5, l7), ntuple(_ -> nx + 1, 4))
        gmsh.model.mesh.setTransfiniteCurve(ℓ, n)
    end
    for (ℓ, n) in zip((l2, l4, l6, l8), ntuple(_ -> ny + 1, 4))
        gmsh.model.mesh.setTransfiniteCurve(ℓ, n)
    end
    for (ℓ, n) in zip((l9, l10, l11, l12), ntuple(_ -> nz + 1, 4))
        gmsh.model.mesh.setTransfiniteCurve(ℓ, n)
    end
    faces = (s_bot, s_top, s_f, s_r, s_b, s_l)
    for s in faces
        gmsh.model.mesh.setTransfiniteSurface(s)
        recombine && gmsh.model.mesh.setRecombine(2, s)
    end
    return faces
end

function _write_box_msh(nome)
    gmsh.model.mesh.generate(2)
    out = datadir("topology", nome * ".msh")
    mkpath(dirname(out))
    gmsh.write(out)
    gmsh.finalize()
    return out
end

"""
    mesh_box_3d(; Lx=1, Ly=1, Lz=1, ndiv=2, bc="1;0", nome="topo3d_box") -> path

Axis-aligned box surface mesh. Default `bc` is uniform Neumann; assign
Dirichlet / loads afterwards with [`apply_heat_bridge_bcs!`](@ref) or
[`apply_cantilever_bcs!`](@ref).
"""
function mesh_box_3d(; Lx=1.0, Ly=1.0, Lz=1.0, ndiv=2, nome="topo3d_box",
        bc="1;0", recombine::Bool=true)
    faces = _box_surfaces!(; Lx=Lx, Ly=Ly, Lz=Lz, ndiv=ndiv, nome=nome, recombine=recombine)
    gmsh.model.addPhysicalGroup(2, collect(faces), -1, bc)
    return _write_box_msh(nome)
end

"""Uniform interior sample on `(0,Lx)×(0,Ly)×(0,Lz)` (not on the boundary)."""
function box_interior_grid(Lx=1.0, Ly=1.0, Lz=1.0, n::Integer=2)
    n >= 1 || throw(ArgumentError("n ≥ 1"))
    xs = n == 1 ? (Lx / 2,) : range(0.25Lx, 0.75Lx; length=n)
    ys = n == 1 ? (Ly / 2,) : range(0.25Ly, 0.75Ly; length=n)
    zs = n == 1 ? (Lz / 2,) : range(0.25Lz, 0.75Lz; length=n)
    return vec([Point3D(x, y, z) for x in xs, y in ys, z in zs])
end

# =============================================================================
# BCs
# =============================================================================

"""
Pacheco inverted-V analogue on a cube: hot patches on `z=0` corners,
cold patch on the top centre, insulated elsewhere.
"""
function apply_heat_bridge_bcs!(dad::BEMdata{<:Laplace}; Lx=1.0, Ly=1.0, Lz=1.0,
        hot=0.25, cold=0.30)
    atol = 1e-8 * max(Lx, Ly, Lz, 1.0)
    @inbounds for i in 1:dad.n
        p = dad.Nodes[i]
        dad.BC[i] = 1
        dad.BV[i] = 0.0
        if abs(p[3]) < atol
            if (p[1] <= hot * Lx && p[2] <= hot * Ly) ||
               (p[1] >= (1 - hot) * Lx && p[2] <= hot * Ly)
                dad.BC[i] = 0
                dad.BV[i] = 1.0
            end
        elseif abs(p[3] - Lz) < atol
            if cold * Lx <= p[1] <= (1 - cold) * Lx &&
               cold * Ly <= p[2] <= (1 - cold) * Ly
                dad.BC[i] = 0
                dad.BV[i] = 0.0
            end
        end
    end
    return dad
end

"""Clamp `x=0`; uniform traction `(0,0,tz)` on `x=Lx`; traction-free elsewhere."""
function apply_cantilever_bcs!(dad::BEMdata{<:Elasticity}; Lx=2.0, tz=-1.0)
    atol = 1e-8 * max(Lx, 1.0)
    dim = 3
    @inbounds for i in 1:dad.n
        p = dad.Nodes[i]
        for d in 1:dim
            dad.BC[dim * (i - 1) + d] = 1
            dad.BV[dim * (i - 1) + d] = 0.0
        end
        if abs(p[1]) < atol
            for d in 1:dim
                dad.BC[dim * (i - 1) + d] = 0
                dad.BV[dim * (i - 1) + d] = 0.0
            end
        elseif abs(p[1] - Lx) < atol
            dad.BC[dim * (i - 1) + 3] = 1
            dad.BV[dim * (i - 1) + 3] = float(tz)
        end
    end
    return dad
end

# =============================================================================
# Problem constructors
# =============================================================================

"""
    heat_cube_3d(; ndiv=2, nint=2, degree=1, k=1.0) -> BEMdata

Unit-cube heat conductor (3-D Pacheco inverted-V analogue): hot patches on
the bottom front corners (`T=1`), cold patch on the top centre (`T=0`),
insulated elsewhere. Interior Cartesian samples for DIBEM-SIMP.
"""
function heat_cube_3d(; ndiv=2, nint=2, degree=1, k=1.0, L=1.0, recombine::Bool=true)
    msh = mesh_box_3d(; Lx=L, Ly=L, Lz=L, ndiv=ndiv, nome="heat_cube_3d",
        bc="1;0", recombine=recombine)
    dad = format3d(msh, Laplace(float(k)); tipo=degree, pontointerno=false)
    set_internal_nodes!(dad, box_interior_grid(L, L, L, nint))
    apply_heat_bridge_bcs!(dad; Lx=L, Ly=L, Lz=L)
    return dad
end

"""
    cantilever_cube_3d(; ndiv=2, nint=2, degree=1, E=1.0, ν=0.3) -> BEMdata

Box cantilever `[0,L]×[0,H]×[0,W]`, clamp at `x=0`, downward traction `tz`
on `x=L`. Default `L=2`, `H=W=1`.
"""
function cantilever_cube_3d(; ndiv=2, nint=2, degree=1, E=1.0, ν=0.3,
        L=2.0, H=1.0, W=1.0, tz=-1.0, recombine::Bool=true)
    nx = ndiv isa Integer ? max(Int(ndiv), 2) : Int(ndiv[1])
    ny = ndiv isa Integer ? max(Int(round(ndiv * H / L)), 1) : Int(ndiv[2])
    nz = ndiv isa Integer ? max(Int(round(ndiv * W / L)), 1) : Int(ndiv[3])
    msh = mesh_box_3d(; Lx=L, Ly=H, Lz=W, ndiv=(nx, ny, nz), nome="cantilever_cube_3d",
        bc="1;0;1;0;1;0", recombine=recombine)
    dad = format3d(msh, Elasticity(E, ν, 1.0; plane_strain=true); tipo=degree,
        pontointerno=false)
    set_internal_nodes!(dad, box_interior_grid(L, H, W, nint))
    apply_cantilever_bcs!(dad; Lx=L, tz=tz)
    return dad
end

function solve_dt_density!(dad::BEMdata, opt::DibemSimpOptions=DibemSimpOptions())
    opt.method = :dt
    return solve_dibem_simp!(dad, opt)
end

# =============================================================================
# Density grid / volume-matched iso-surface (3-D analogue of cut_low_density!)
# =============================================================================

function _bbox_axes3(pts; n=21, pad=0.0)
    xmin = minimum(p[1] for p in pts); xmax = maximum(p[1] for p in pts)
    ymin = minimum(p[2] for p in pts); ymax = maximum(p[2] for p in pts)
    zmin = minimum(p[3] for p in pts); zmax = maximum(p[3] for p in pts)
    dx, dy, dz = xmax - xmin, ymax - ymin, zmax - zmin
    n = max(Int(n), 3)
    return range(xmin - pad * dx, xmax + pad * dx; length=n),
           range(ymin - pad * dy, ymax + pad * dy; length=n),
           range(zmin - pad * dz, zmax + pad * dz; length=n)
end

"""Inverse-distance interpolate nodal `val` onto a Cartesian grid."""
function interpolate_to_grid_3d(pts::AbstractVector{<:Point3D}, val::AbstractVector,
        xs, ys, zs; p::Real=2)
    nx, ny, nz = length(xs), length(ys), length(zs)
    Z = fill(NaN, nx, ny, nz)
    isempty(pts) && return Z
    tree = KDTree(reduce(hcat, pts))
    k = min(8, length(pts))
    @inbounds for kk in 1:nz, jj in 1:ny, ii in 1:nx
        q = SVector(xs[ii], ys[jj], zs[kk])
        idxs, dists = knn(tree, q, k)
        wsum = 0.0
        s = 0.0
        ok = false
        for (id, di) in zip(idxs, dists)
            if di < 1e-14
                s = val[id]
                wsum = 1.0
                ok = true
                break
            end
            w = 1 / (di^p)
            s += w * val[id]
            wsum += w
            ok = true
        end
        Z[ii, jj, kk] = ok ? s / wsum : NaN
    end
    return Z
end

"""Scatter nodal `ρ` onto a 3-D grid. Non-finite samples are void (`0`)."""
function density_grid_3d(dad::BEMdata, ρ::AbstractVector; ngrid::Integer=21)
    dad.dimension == 3 || throw(ArgumentError("density_grid_3d needs a 3-D BEMdata"))
    length(ρ) == dad.nt || throw(DimensionMismatch("ρ vs nt"))
    pts = Point3D[p for p in all_points(dad)]
    xs, ys, zs = _bbox_axes3(pts; n=ngrid, pad=0.0)
    Z = interpolate_to_grid_3d(pts, collect(Float64, ρ), xs, ys, zs)
    @inbounds for k in eachindex(zs), j in eachindex(ys), i in eachindex(xs)
        v = Z[i, j, k]
        if !isfinite(v)
            Z[i, j, k] = 0.0
        end
    end
    return xs, ys, zs, Z
end

"""Voxel volume of `{Z ≥ lev}` (same counting as the 2-D grid-solid area)."""
function grid_solid_volume(xs, ys, zs, Z, lev)
    nx, ny, nz = length(xs), length(ys), length(zs)
    (nx < 2 || ny < 2 || nz < 2) && return 0.0
    cell = abs((xs[2] - xs[1]) * (ys[2] - ys[1]) * (zs[2] - zs[1]))
    v = 0.0
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        Z[i, j, k] >= lev && (v += cell)
    end
    return v
end

function _ρ_cut_for_volume_3d(xs, ys, zs, Z, Vtarget; lo::Float64=0.12, hi::Float64=0.78)
    V_of(lev) = grid_solid_volume(xs, ys, zs, Z, lev)
    V_lo, V_hi = V_of(lo), V_of(hi)
    Vtarget >= V_lo && return lo
    Vtarget <= V_hi && return hi
    lev = lo
    @inbounds for _ in 1:24
        mid = 0.5 * (lo + hi)
        if V_of(mid) > Vtarget
            lo = mid
        else
            hi = mid
        end
        lev = mid
    end
    return lev
end

function _bbox_volume(pts)
    xmin = minimum(p[1] for p in pts); xmax = maximum(p[1] for p in pts)
    ymin = minimum(p[2] for p in pts); ymax = maximum(p[2] for p in pts)
    zmin = minimum(p[3] for p in pts); zmax = maximum(p[3] for p in pts)
    return abs((xmax - xmin) * (ymax - ymin) * (zmax - zmin))
end

"""
    cut_density_3d!(dad, ρ, opt) -> (tris, ρh, lev)

Heaviside-project `ρ`, scatter onto a grid, pick an iso-level whose voxel
solid volume matches `opt.volfrac` of the bounding box (if `match_area`),
and return marching-tetrahedra triangles. Caches `simp_iso` / `simp_ρ_cut`.
Does **not** remesh the BEM surface.
"""
function cut_density_3d!(dad::BEMdata, ρ::AbstractVector, opt::DibemSimpOptions)
    dad.dimension == 3 || throw(ArgumentError("cut_density_3d! is 3-D"))
    pts = all_points(dad)
    ρf = density_filter(pts, ρ, opt.rmin)
    ρh = _heaviside(ρf, opt.β_end)
    ngrid = clamp(Int(opt.ngrid), 5, 41)
    xs, ys, zs, Z = density_grid_3d(dad, ρh; ngrid=ngrid)
    Vtarget = opt.volfrac * _bbox_volume(pts)
    lev = if opt.match_area
        clamp(_ρ_cut_for_volume_3d(xs, ys, zs, Z, Vtarget), 0.12, 0.78)
    else
        opt.ρ_cut
    end
    tris = marching_cubes(xs, ys, zs, Z, lev)
    set_cache!(dad; simp_iso=tris, simp_ρ_cut=lev, simp_ρ=ρh)
    return tris, ρh, lev
end

# =============================================================================
# Closed iso-surface → traction-free cavities → rebuild BEMdata (2-D analogue)
# =============================================================================

"""Merge duplicate vertices of a triangle soup. Returns `(verts, faces)`."""
function stitch_triangles(tris::AbstractVector{<:NTuple{3,Point3D}}; atol::Real=1e-10)
    verts = Point3D[]
    faces = NTuple{3,Int}[]
    function vid!(p)
        @inbounds for i in eachindex(verts)
            sum(abs2, verts[i] - p) < atol && return i
        end
        push!(verts, p)
        return length(verts)
    end
    @inbounds for (a, b, c) in tris
        ia, ib, ic = vid!(a), vid!(b), vid!(c)
        (ia == ib || ib == ic || ic == ia) && continue
        push!(faces, (ia, ib, ic))
    end
    return verts, faces
end

function _edge_faces(faces)
    e2f = Dict{Tuple{Int,Int},Vector{Int}}()
    @inbounds for (t, (a, b, c)) in enumerate(faces)
        for e in ((a, b), (b, c), (c, a))
            key = e[1] < e[2] ? e : (e[2], e[1])
            push!(get!(e2f, key, Int[]), t)
        end
    end
    return e2f
end

function _tri_components(faces)
    n = length(faces)
    n == 0 && return Vector{Int}[]
    adj = [Int[] for _ in 1:n]
    for fs in values(_edge_faces(faces))
        for i in 1:length(fs), j in (i + 1):length(fs)
            push!(adj[fs[i]], fs[j])
            push!(adj[fs[j]], fs[i])
        end
    end
    seen = falses(n)
    comps = Vector{Int}[]
    for s in 1:n
        seen[s] && continue
        stack = Int[s]
        seen[s] = true
        comp = Int[]
        while !isempty(stack)
            t = pop!(stack)
            push!(comp, t)
            for u in adj[t]
                seen[u] && continue
                seen[u] = true
                push!(stack, u)
            end
        end
        push!(comps, comp)
    end
    return comps
end

function _component_closed(faces, idx)
    e2f = _edge_faces(faces[idx])
    for fs in values(e2f)
        length(fs) == 2 || return false
    end
    return !isempty(e2f)
end

function _tri_area_centroid(verts, faces, idx)
    A = 0.0
    c = zero(Point3D)
    @inbounds for t in idx
        a, b, c3 = verts[faces[t][1]], verts[faces[t][2]], verts[faces[t][3]]
        n = cross(b - a, c3 - a)
        at = 0.5 * norm(n)
        A += at
        c += at * ((a + b + c3) / 3)
    end
    return A, (A > 0 ? c / A : zero(Point3D))
end

function _signed_volume(verts, faces, idx)
    V = 0.0
    @inbounds for t in idx
        a, b, c = verts[faces[t][1]], verts[faces[t][2]], verts[faces[t][3]]
        V += dot(a, cross(b, c))
    end
    return V / 6
end

"""Flip winding so the geometric normal points **into** the enclosed volume (BEM hole)."""
function _orient_cavity!(verts, faces, idx)
    if _signed_volume(verts, faces, idx) > 0
        @inbounds for t in idx
            a, b, c = faces[t]
            faces[t] = (a, c, b)
        end
    end
    return faces
end

function _laplacian_smooth!(verts, faces, idx; passes::Integer=1)
    used = falses(length(verts))
    @inbounds for t in idx, k in 1:3
        used[faces[t][k]] = true
    end
    nbr = [Set{Int}() for _ in eachindex(verts)]
    @inbounds for t in idx
        a, b, c = faces[t]
        push!(nbr[a], b); push!(nbr[a], c)
        push!(nbr[b], a); push!(nbr[b], c)
        push!(nbr[c], a); push!(nbr[c], b)
    end
    for _ in 1:passes
        nxt = copy(verts)
        @inbounds for i in eachindex(verts)
            used[i] || continue
            isempty(nbr[i]) && continue
            s = zero(Point3D)
            for j in nbr[i]
                s += verts[j]
            end
            nxt[i] = s / length(nbr[i])
        end
        verts .= nxt
    end
    return verts
end

function _reindex_component(verts, faces, idx)
    old = sort!(unique(v for t in idx for v in faces[t]))
    map = Dict{Int,Int}(old[i] => i for i in eachindex(old))
    v2 = verts[old]
    f2 = NTuple{3,Int}[(map[faces[t][1]], map[faces[t][2]], map[faces[t][3]]) for t in idx]
    return v2, f2
end

function _min_dist_to_nodes(pts, nodes)
    dmin = Inf
    @inbounds for p in pts, q in nodes
        d = sum(abs2, p - q)
        d < dmin && (dmin = d)
    end
    return sqrt(dmin)
end

function _in_bbox(p, xmin, xmax, ymin, ymax, zmin, zmax, m)
    return xmin + m < p[1] < xmax - m &&
           ymin + m < p[2] < ymax - m &&
           zmin + m < p[3] < zmax - m
end

"""Ray–triangle (Möller–Trumbore)."""
function _ray_tri(orig, dir, a, b, c)
    ab = b - a
    ac = c - a
    pvec = cross(dir, ac)
    det = dot(ab, pvec)
    abs(det) < 1e-16 && return false
    invdet = 1 / det
    tvec = orig - a
    u = dot(tvec, pvec) * invdet
    (u < -1e-12 || u > 1 + 1e-12) && return false
    qvec = cross(tvec, ab)
    v = dot(dir, qvec) * invdet
    (v < -1e-12 || u + v > 1 + 1e-12) && return false
    t = dot(ac, qvec) * invdet
    return t > 1e-12
end

"""Odd/even ray test. `faces` index into `verts`."""
function point_in_cavity(p::Point3D, verts, faces)
    dir = Point3D(1.0, 0.013, 0.007)
    hits = 0
    @inbounds for (ia, ib, ic) in faces
        _ray_tri(p, dir, verts[ia], verts[ib], verts[ic]) && (hits += 1)
    end
    return isodd(hits)
end

"""
    extract_closed_cavities(tris, dad; ρ, ρ_cut, min_area, min_dist) -> Vector{NamedTuple}

Connected closed components of the iso-surface that sit inside the original
box, enclose **void** (low `ρ` at the centroid), and stay `min_dist` away
from the current Γ. One Laplacian pass (Chaikin analogue) then reoriented so
`n` points into the cavity.
"""
function extract_closed_cavities(tris::AbstractVector{<:NTuple{3,Point3D}}, dad::BEMdata;
        ρ=nothing, ρ_cut::Float64=0.4, min_area::Float64=4e-4, min_dist::Float64=0.008)
    isempty(tris) && return NamedTuple[]
    outer = dad.Nodes
    xmin = minimum(p[1] for p in outer); xmax = maximum(p[1] for p in outer)
    ymin = minimum(p[2] for p in outer); ymax = maximum(p[2] for p in outer)
    zmin = minimum(p[3] for p in outer); zmax = maximum(p[3] for p in outer)
    Ldiag = hypot(xmax - xmin, hypot(ymax - ymin, zmax - zmin))
    dmin = min_dist * Ldiag
    atol = max(1e-12, 1e-9 * Ldiag)
    verts, faces = stitch_triangles(tris; atol=atol)
    out = NamedTuple[]
    for idx in _tri_components(faces)
        length(idx) < 4 && continue
        _component_closed(faces, idx) || continue
        A, c = _tri_area_centroid(verts, faces, idx)
        A < min_area && continue
        _in_bbox(c, xmin, xmax, ymin, ymax, zmin, zmax, dmin) || continue
        if ρ !== nothing && length(ρ) == dad.nt
            ρc = ρ[argmin(i -> sum(abs2, point(dad, i) - c), 1:dad.nt)]
            ρc > ρ_cut && continue
        end
        v2, f2 = _reindex_component(verts, faces, idx)
        _min_dist_to_nodes(v2, outer) < dmin && continue
        _laplacian_smooth!(v2, f2, 1:length(f2); passes=1)
        _orient_cavity!(v2, f2, 1:length(f2))
        push!(out, (verts=v2, faces=f2, area=A, centroid=c))
    end
    return out
end

function _collapsed_tri_element!(nodes, normal, ELEM, BC, BV, a, b, c, N, dNx, dNy, dof, region)
    X = Point3D[a, b, c, c]
    n_per = size(N, 1)
    idx0 = length(nodes) + 1
    idx = collect(idx0:(idx0 + n_per - 1))
    colloc = N * X
    dx1 = dNx * X
    dx2 = dNy * X
    J = norm.(cross.(dx1, dx2))
    ngeo = cross(b - a, c - a)
    nlen = norm(ngeo)
    ngeo = nlen > 1e-16 ? ngeo / nlen : Point3D(0.0, 0.0, 1.0)
    nrm = Vector{Point3D}(undef, n_per)
    @inbounds for k in 1:n_per
        nk = J[k] > 1e-16 ? cross(dx1[k], dx2[k]) / J[k] : ngeo
        if J[k] > 1e-16 && dot(nk, ngeo) < 0
            nk = -nk
        end
        nrm[k] = nk
    end
    append!(nodes, colloc)
    append!(normal, nrm)
    for _ in 1:n_per
        for _ in 1:dof
            push!(BC, 1)
            push!(BV, 0.0)
        end
    end
    push!(ELEM, Element(idx, collect(Float64, J), max(nlen, 1e-16), region))
    return nothing
end

function _filter_internals_3d(pts, cavities; d_min::Real=0.01)
    out = Point3D[]
    for p in pts
        skip = false
        for cav in cavities
            if point_in_cavity(p, cav.verts, cav.faces)
                skip = true
                break
            end
            if _min_dist_to_nodes((p,), cav.verts) < d_min
                skip = true
                break
            end
        end
        skip || push!(out, p)
    end
    return out
end

"""
    bemdata_from_iso(dad, tris; ρ, ρ_cut, min_area, min_dist) -> (dad, n_cavities)

2-D `bemdata_from_loops` analogue: keep the original outer surface, add each
**closed** interior iso-component as a traction-free cavity, discretized as
collapsed linear triangles (same as `format3d` Gmsh type 2). Requires
`degree(dad.element_type) == 1`. Does not Chaikin-style refine the triangles
beyond one Laplacian pass in [`extract_closed_cavities`](@ref).
"""
function bemdata_from_iso(dad::BEMdata, tris::AbstractVector{<:NTuple{3,Point3D}};
        ρ=nothing, ρ_cut::Float64=0.4, min_area::Float64=4e-4, min_dist::Float64=0.008,
        d_min::Real=0.01)
    dad.dimension == 3 || throw(ArgumentError("bemdata_from_iso is 3-D"))
    degree(dad.element_type) == 1 || throw(ArgumentError(
        "cavity remesh needs degree 1 (collapsed triangles); got $(degree(dad.element_type))"))
    cavs = extract_closed_cavities(tris, dad; ρ=ρ, ρ_cut=ρ_cut, min_area=min_area,
        min_dist=min_dist)
    isempty(cavs) && return dad, 0
    dof = dad.properties isa Vectorial ? 3 : 1
    qsi, wi = gausslegendre(2)
    N, dNx, dNy = shapefun2D(Equispaced(1), qsi)
    nodes = Point3D[dad.Nodes[i] for i in 1:dad.n]
    nrm = Point3D[dad.Normal[i] for i in 1:dad.n]
    ELEM = Element[Element(copy(el.index), copy(el.Jacobian), el.Length, el.Region)
                   for el in dad.elements]
    BC = copy(dad.BC)
    BV = copy(dad.BV)
    for (k, cav) in enumerate(cavs)
        region = 1000 + k
        for (ia, ib, ic) in cav.faces
            _collapsed_tri_element!(nodes, nrm, ELEM, BC, BV,
                cav.verts[ia], cav.verts[ib], cav.verts[ic], N, dNx, dNy, dof, region)
        end
    end
    internals = _filter_internals_3d(collect(dad.internalNodes), cavs; d_min=d_min)
    n = length(nodes)
    ni = length(internals)
    coll = ni == 0 ? nodes : vcat(nodes, internals)
    dad2 = BEMdata(dad.name, 3, ELEM, dad.element_type, dad.elem_weight, coll, nrm,
        dad.properties, BC, BV, n, ni, n + ni, BEMCache())
    set_cache!(dad2; topology_iso=tris, n_cavities=length(cavs), cavities=cavs)
    return dad2, length(cavs)
end

# =============================================================================
# VTK: boundary + interior points as vertices with scalar ρ
# =============================================================================

"""
    export_vtk_density(dad, ρ, path) -> path

ASCII VTK POLYDATA of every collocation node (boundary then interior)
with scalar `ρ` (and `DT` if cached).
"""
function export_vtk_density(dad::BEMdata, ρ::AbstractVector, path::AbstractString;
        DT=nothing)
    length(ρ) == dad.nt || throw(DimensionMismatch("ρ length $(length(ρ)) ≠ nt=$(dad.nt)"))
    nt = dad.nt
    open(path, "w") do io
        println(io, "# vtk DataFile Version 3.0")
        println(io, dad.name, " density")
        println(io, "ASCII")
        println(io, "DATASET POLYDATA")
        println(io, "POINTS ", nt, " float")
        @inbounds for i in 1:nt
            p = point(dad, i)
            if length(p) == 2
                @printf(io, "%.10g %.10g 0\n", p[1], p[2])
            else
                @printf(io, "%.10g %.10g %.10g\n", p[1], p[2], p[3])
            end
        end
        println(io, "VERTICES ", nt, " ", 2nt)
        @inbounds for i in 0:nt-1
            println(io, "1 ", i)
        end
        println(io, "POINT_DATA ", nt)
        println(io, "SCALARS rho float 1")
        println(io, "LOOKUP_TABLE default")
        @inbounds for i in 1:nt
            @printf(io, "%.10g\n", ρ[i])
        end
        if DT !== nothing && length(DT) == nt
            println(io, "SCALARS DT float 1")
            println(io, "LOOKUP_TABLE default")
            @inbounds for i in 1:nt
                @printf(io, "%.10g\n", DT[i])
            end
        end
    end
    return path
end

"""
    export_vtk_isosurface(tris, path) -> path

ASCII VTK POLYDATA of marching-tetrahedra triangles (`NTuple{3,Point3D}`).
"""
function export_vtk_isosurface(tris::AbstractVector{<:NTuple{3,Point3D}}, path::AbstractString)
    ntri = length(tris)
    npts = 3ntri
    open(path, "w") do io
        println(io, "# vtk DataFile Version 3.0")
        println(io, "iso-surface")
        println(io, "ASCII")
        println(io, "DATASET POLYDATA")
        println(io, "POINTS ", npts, " float")
        @inbounds for (a, b, c) in tris
            @printf(io, "%.10g %.10g %.10g\n", a[1], a[2], a[3])
            @printf(io, "%.10g %.10g %.10g\n", b[1], b[2], b[3])
            @printf(io, "%.10g %.10g %.10g\n", c[1], c[2], c[3])
        end
        println(io, "POLYGONS ", ntri, " ", 4ntri)
        @inbounds for t in 0:ntri-1
            println(io, "3 ", 3t, " ", 3t + 1, " ", 3t + 2)
        end
    end
    return path
end
