# Pacheco (2020) BEM + topological-derivative topology optimization.
# Velocity: inward normal only. Nucleation: low-DT iso-curves (iter 1 and later).

export PachecoOptions, TopologyHistory
export solve_topology!, pacheco_step!, match_volume!
export nucleate_holes!, move_boundary!, move_boundary_standin!
export move_boundary_jump!
export dt_on_grid, interpolate_to_grid

@kwdef mutable struct PachecoOptions
    ΔA::Float64 = 0.5
    vmax::Float64 = 0.1
    pct::Float64 = 0.9
    nv::Float64 = 0.04
    λ::Float64 = 4.0
    passos::Int = 6
    ΔAmin::Float64 = 0.01
    nucleate_every::Int = 1          # 1 = every iter; typemax(Int) = only when forced
    nucleate_first::Bool = true
    min_hole_area::Float64 = 1e-3
    min_hole_dist::Float64 = 0.02    # fraction of bbox diagonal
    d_min::Float64 = 0.01            # interior-point clearance, fraction of Lx
    maxiter::Int = 80
    npg::Int = 16
    verbose::Bool = true
    area_rtol::Float64 = 0.08      # stop when A ≤ Ap*(1+area_rtol)
    motion::Symbol = :quantile       # :quantile, :standin, or :jump
    standin_α::Float64 = 0.35        # vn = α (DT − λ)
    standin_inward::Bool = false     # true → vn ≤ 0 (recede only)
    volume_step::Float64 = 0.20      # fraction of (A − Ap) per stand-in/jump iter
    nsearch::Int = 4
    jump_mode::Symbol = :linear      # :linear (JuMP+HiGHS LP) or :mma (JuMP+NLopt)
    jump_maxeval::Int = 8
end

struct TopologyHistory
    area::Vector{Float64}
    J::Vector{Float64}
    maxDT::Vector{Float64}
    n_holes::Vector{Int}
    designs::Vector{TopologyDesign}
end

TopologyHistory() = TopologyHistory(Float64[], Float64[], Float64[], Int[], TopologyDesign[])

_fmt_obj(J) = @sprintf("%.4g", J)

function Base.push!(h::TopologyHistory; area, J, maxDT, n_holes, design)
    push!(h.area, area)
    push!(h.J, J)
    push!(h.maxDT, maxDT)
    push!(h.n_holes, n_holes)
    push!(h.designs, deepcopy(design))
    return h
end

# -----------------------------------------------------------------------------
# DT → Cartesian grid
# -----------------------------------------------------------------------------

function interpolate_to_grid(pts::AbstractVector{<:Point}, val::AbstractVector, xs, ys; p::Real=2)
    nx, ny = length(xs), length(ys)
    Z = fill(NaN, nx, ny)
    isempty(pts) && return Z
    tree = KDTree(reduce(hcat, pts))
    k = min(8, length(pts))
    @inbounds for j in 1:ny, i in 1:nx
        q = SVector(xs[i], ys[j])
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
        Z[i, j] = ok ? s / wsum : NaN
    end
    return Z
end

function dt_on_grid(dad::BEMdata, DTb, DTi; ngrid::Integer=45)
    pts = Point2D[dad.Nodes; dad.internalNodes]
    val = vcat(DTb, DTi)
    xs, ys = _bbox_axes(pts; n=ngrid)
    Z = interpolate_to_grid(pts, val, xs, ys)
    return xs, ys, Z
end

function _bbox_axes(pts; n=45, pad=0.02)
    xmin = minimum(p[1] for p in pts)
    xmax = maximum(p[1] for p in pts)
    ymin = minimum(p[2] for p in pts)
    ymax = maximum(p[2] for p in pts)
    dx = xmax - xmin
    dy = ymax - ymin
    return range(xmin - pad * dx, xmax + pad * dx; length=n),
           range(ymin - pad * dy, ymax + pad * dy; length=n)
end

# -----------------------------------------------------------------------------
# Nucleation
# -----------------------------------------------------------------------------

"""
Insert low-`DT` iso-curves as holes (closed) or as replacements of Neumann
stretches of the outer loop (open). Frozen Dirichlet vertices are never cut.
"""
function nucleate_holes!(d::TopologyDesign, dad::BEMdata, DTb, DTi, opt::PachecoOptions)
    isempty(DTi) && return 0
    level = quantile(DTi, clamp(opt.nv, 0.0, 1.0))
    xs, ys, Z = dt_on_grid(dad, DTb, DTi)
    nx = length(xs)
    # NaNs → treat as high DT (do not nucleate outside)
    zmax = maximum(z for z in Z if isfinite(z); init=level + 1)
    @inbounds for i in eachindex(Z)
        isfinite(Z[i]) || (Z[i] = zmax)
    end
    lines = marching_squares(xs, ys, Z, level)
    nadd = 0
    Ldiag = _design_diag(d)
    dmin = opt.min_hole_dist * Ldiag
    for line in lines
        length(line) < 8 && continue
        closed = norm(line[1] - line[end]) < 1e-8 * (1 + Ldiag)
        if closed
            pts = line[1] ≈ line[end] ? line[1:end-1] : copy(line)
            # keep only if entirely in the current solid
            all(p -> in_design(p, d), pts[1:max(1, length(pts) ÷ 4):end]) || continue
            a = abs(polygon_area(pts))
            a < opt.min_hole_area && continue
            if _min_dist_to_design(pts, d) < dmin
                continue
            end
            # hole must enclose a *low*-DT pocket, not a high-DT core
            c = sum(pts) / length(pts)
            ic = min(nx, max(1, searchsortedfirst(xs, c[1]) - 1))
            jc = min(length(ys), max(1, searchsortedfirst(ys, c[2]) - 1))
            if Z[ic, jc] > 2 * level
                continue
            end
            chaikin_smooth!(pts; closed=true, passes=1)
            nel = max(8, Int(round(length(pts) / 2)))
            pts = resample_polyline(pts, nel; closed=true)
            a = polygon_area(pts)
            a > 0 && reverse!(pts)                 # hole CW
            seg = hole_segment(pts, d, nel)
            push!(d.loops, [seg])
            nadd += 1
        else
            nadd += _splice_open_isocurve!(d, line, dmin)
        end
    end
    return nadd
end

function _design_diag(d::TopologyDesign)
    outer = loop_vertices(d.loops[1])
    xmin = minimum(p[1] for p in outer)
    xmax = maximum(p[1] for p in outer)
    ymin = minimum(p[2] for p in outer)
    ymax = maximum(p[2] for p in outer)
    return hypot(xmax - xmin, ymax - ymin)
end

function _min_dist_to_design(pts, d::TopologyDesign)
    dmin = Inf
    for segs in d.loops
        v = loop_vertices(segs)
        for p in pts
            dmin = min(dmin, _dist_to_polyline(p, v))
        end
    end
    return dmin
end

function _splice_open_isocurve!(d::TopologyDesign, line::Vector{Point2D}, dmin)
    # attach only to the outer loop, and only onto Neumann vertices
    segs = d.loops[1]
    outer = loop_vertices(segs)
    # map vertex index → (iseg, iv)
    map = _vertex_owners(segs)
    i1, d1 = _nearest_neumann(line[1], segs, map)
    i2, d2 = _nearest_neumann(line[end], segs, map)
    (i1 == 0 || i2 == 0 || i1 == i2) && return 0
    d1 > dmin * 2 && return 0
    d2 > dmin * 2 && return 0
    n = length(outer)
    # path along outer from i1 to i2 (forward); skip if it contains frozen verts
    path = _arc_indices(i1, i2, n)
    if any(k -> map[k].frozen, path)
        path = _arc_indices(i2, i1, n)
        reverse!(line)
        i1, i2 = i2, i1
        any(k -> map[k].frozen, path) && return 0
    end
    length(path) < 2 && return 0
    # rebuild outer verts: keep up to i1, insert iso-curve, then from i2
    newverts = Point2D[]
    for k in 1:n
        if k == i1
            push!(newverts, outer[i1])
            append!(newverts, line[2:end-1])
            push!(newverts, outer[i2])
        elseif k in path && k != i2
            continue
        else
            push!(newverts, outer[k])
        end
    end
    # rebuild as a single Neumann/mixed segment sequence, preserving Dirichlet bits
    d.loops[1] = _rebuild_outer_segments(segs, newverts)
    freeze_dirichlet!(d)
    return 1
end

function _vertex_owners(segs::Vector{BoundarySegment})
    # flatten; last of seg i identified with first of seg i+1
    out = NamedTuple{(:frozen, :fixed),Tuple{Bool,Bool}}[]
    for s in segs
        nv = length(s.verts)
        for k in 1:nv
            # skip duplicate closing vertex of each segment except last of loop
            if k == nv && s !== segs[end]
                continue
            end
            push!(out, (frozen=s.frozen[k], fixed=_is_fixed_segment(s)))
        end
    end
    return out
end

function _nearest_neumann(p, segs, map)
    verts = loop_vertices(segs)
    ibest, dbest = 0, Inf
    for (i, v) in enumerate(verts)
        i > length(map) && break
        map[i].frozen && continue
        map[i].fixed && continue
        di = norm(v - p)
        if di < dbest
            dbest = di
            ibest = i
        end
    end
    return ibest, dbest
end

function _arc_indices(i1, i2, n)
    out = Int[i1]
    i = i1
    while i != i2
        i = i == n ? 1 : i + 1
        push!(out, i)
        length(out) > n && break
    end
    return out
end

function _rebuild_outer_segments(old::Vector{BoundarySegment}, newverts::Vector{Point2D})
    # Keep Dirichlet / load patches as straight segments; remaining is free.
    diris = [s for s in old if _is_fixed_segment(s)]
    if isempty(diris)
        nel = max(sum(s.n_el for s in old), 8)
        return [_free_segment(newverts, old, nel)]
    end
    # snap fixed endpoints to nearest newverts, split the loop
    segs = BoundarySegment[]
    used = falses(length(newverts))
    n = length(newverts)
    # locate each fixed chord in newverts
    anchors = NTuple{3,Int}[]   # (i_start, i_end, old_index)
    for (io, s) in enumerate(old)
        _is_fixed_segment(s) || continue
        a = s.verts[1]
        b = s.verts[end]
        ia = argmin(i -> norm(newverts[i] - a), 1:n)
        ib = argmin(i -> norm(newverts[i] - b), 1:n)
        push!(anchors, (ia, ib, io))
    end
    sort!(anchors; by=x -> x[1])
    # walk the loop
    for (k, (ia, ib, io)) in enumerate(anchors)
        s0 = old[io]
        # Neumann chain before this Dirichlet (from previous ib to ia)
        prev_ib = k == 1 ? anchors[end][2] : anchors[k - 1][2]
        chain = _arc_indices(prev_ib, ia, n)
        if length(chain) ≥ 3
            pts = newverts[chain]
            nel = max(length(pts) - 1, 2)
            push!(segs, _free_segment(pts, old, nel))
        end
        push!(segs, BoundarySegment(copy(s0.verts); bc=copy(s0.bc), value=copy(s0.value),
            frozen=copy(s0.frozen), n_el=s0.n_el))
    end
    isempty(segs) && push!(segs, _free_segment(newverts, old, max(length(newverts), 8)))
    return segs
end

# -----------------------------------------------------------------------------
# Inward-normal motion
# -----------------------------------------------------------------------------

"""Insert interior vertices on Neumann segments so frozen Dirichlet junctions are not the only nodes."""
function _densify_neumann!(d::TopologyDesign)
    for segs in d.loops
        closed_loop = length(segs) == 1 && !_is_fixed_segment(segs[1])
        for seg in segs
            _is_fixed_segment(seg) && continue
            target = max(seg.n_el + 1, 4)
            length(seg.verts) >= target && continue
            fr0 = !isempty(seg.frozen) && seg.frozen[1]
            fr1 = !isempty(seg.frozen) && seg.frozen[end]
            seg.verts = resample_polyline(seg.verts, target; closed=closed_loop)
            seg.frozen = BitVector(fill(false, length(seg.verts)))
            if !closed_loop
                seg.frozen[1] = fr0
                seg.frozen[end] = fr1
            end
        end
    end
    freeze_dirichlet!(d)
    return d
end

@inline _orient2(a, b, c) =
    (b[1] - a[1]) * (c[2] - a[2]) - (b[2] - a[2]) * (c[1] - a[1])

"""Proper (non-endpoint) intersection of open segments `ab` and `cd`."""
function _proper_intersect(a, b, c, d)
    o1 = _orient2(a, b, c)
    o2 = _orient2(a, b, d)
    o3 = _orient2(c, d, a)
    o4 = _orient2(c, d, b)
    return (o1 * o2 < 0.0) && (o3 * o4 < 0.0)
end

function _loop_self_intersects(pts::AbstractVector{<:SVector{2}})
    n = length(pts)
    n < 4 && return false
    @inbounds for i in 1:n
        a = pts[i]
        b = pts[i == n ? 1 : i + 1]
        j0 = i + 2
        j1 = i == 1 ? n - 1 : n
        for j in j0:j1
            (i == 1 && j == n) && continue
            c = pts[j]
            d = pts[j == n ? 1 : j + 1]
            _proper_intersect(a, b, c, d) && return true
        end
    end
    return false
end

function _loops_cross(a::AbstractVector{<:SVector{2}}, b::AbstractVector{<:SVector{2}})
    na, nb = length(a), length(b)
    (na < 2 || nb < 2) && return false
    @inbounds for i in 1:na
        p = a[i]
        q = a[i == na ? 1 : i + 1]
        for j in 1:nb
            r = b[j]
            s = b[j == nb ? 1 : j + 1]
            _proper_intersect(p, q, r, s) && return true
        end
    end
    return false
end

"""True if any loop is a bow-tie or a hole crosses the outer contour."""
function _design_folded(d::TopologyDesign)
    for segs in d.loops
        v = loop_vertices(segs)
        length(v) ≥ 4 && _loop_self_intersects(v) && return true
    end
    length(d.loops) < 2 && return false
    outer = loop_vertices(d.loops[1])
    for k in 2:length(d.loops)
        _loops_cross(outer, loop_vertices(d.loops[k])) && return true
    end
    return false
end

"""Max inward step that keeps `p` on the same side of the neighbour chord."""
function _inward_step_limit(pprev, p, pnext, dir)
    Ld = norm(dir)
    Ld < 1e-16 && return 0.0
    e0 = norm(p - pprev)
    e1 = norm(pnext - p)
    lim = 0.35 * min(e0, e1)
    ab = pnext - pprev
    L2 = dot(ab, ab)
    L2 < 1e-30 && return lim
    # signed distance to infinite line through neighbours, along `dir`
    nχ = SVector(-ab[2], ab[1])   # left of chord
    nL = norm(nχ)
    nL < 1e-16 && return lim
    nχ /= nL
    dist = abs(dot(p - pprev, nχ))
    toward = -dot(dir / Ld, nχ)   # >0 if `dir` (already −n̂) goes toward the chord
    if toward > 0.2
        lim = min(lim, 0.40 * dist / toward)
    end
    return max(lim, 0.0)
end

function _snapshot_verts(d::TopologyDesign)
    [[(copy(s.verts), copy(s.frozen)) for s in segs] for segs in d.loops]
end

function _restore_verts!(d::TopologyDesign, snap)
    for (segs, ssnap) in zip(d.loops, snap)
        for (s, (v, fr)) in zip(segs, ssnap)
            s.verts = copy(v)
            s.frozen = copy(fr)
        end
    end
    return d
end

"""Move Neumann vertices with low `DT` along `-n̂`. Dirichlet frozen.

Each step is clamped so the local triangle cannot invert; a pass that
makes a loop self-intersect is reverted (no folded `Γ`).
"""
function move_boundary!(d::TopologyDesign, dad::BEMdata, DTb, opt::PachecoOptions)
    _densify_neumann!(d)
    vmax = opt.vmax
    npass = max(opt.passos, 1)
    hchar = _design_diag(d) / max(sum(s.n_el for segs in d.loops for s in segs), 1)
    vimax = min(vmax / npass, 0.25 * hchar)
    tree = KDTree(reduce(hcat, collect(Point2D, dad.Nodes)))
    DTcut = quantile(DTb, clamp(opt.pct, 0.0, 1.0))
    DTmax = maximum(DTb)
    DTmin = minimum(DTb)
    span = max(DTmax - DTmin, 1e-30)

    for _ in 1:npass
        snap = _snapshot_verts(d)
        for segs in d.loops
            loopv = loop_vertices(segs)
            nrm = vertex_outward_normals(loopv; closed=true)
            nv = length(loopv)
            for seg in segs
                _is_fixed_segment(seg) && continue
                n = length(seg.verts)
                n < 3 && continue
                owners = _closest_indices(seg.verts, loopv)
                for i in 1:n
                    seg.frozen[i] && continue
                    if i > 1 && seg.frozen[i - 1]
                        continue
                    end
                    if i < n && seg.frozen[i + 1]
                        continue
                    end
                    idxs, _ = knn(tree, seg.verts[i], 1)
                    DTi = DTb[idxs[1]]
                    DTi > DTcut && continue
                    α = ((DTmax - DTi) / span)^opt.λ
                    io = owners[i]
                    n̂ = nrm[io]
                    Ln = norm(n̂)
                    Ln < 1e-16 && continue
                    n̂ /= Ln
                    iprev = io == 1 ? nv : io - 1
                    inext = io == nv ? 1 : io + 1
                    dir = -n̂
                    step = vimax * α
                    step = min(step, _inward_step_limit(loopv[iprev], seg.verts[i],
                        loopv[inext], dir))
                    step > 0 || continue
                    seg.verts[i] = seg.verts[i] + step * dir
                end
                if count(!, seg.frozen) ≥ 2
                    fr0, fr1 = seg.frozen[1], seg.frozen[end]
                    v0, v1 = seg.verts[1], seg.verts[end]
                    target = max(seg.n_el + 1, 4)
                    mid = resample_polyline(seg.verts, target; closed=false)
                    if fr0
                        mid[1] = v0
                    end
                    if fr1
                        mid[end] = v1
                    end
                    seg.verts = mid
                    seg.frozen = BitVector(fill(false, length(mid)))
                    seg.frozen[1] = fr0
                    seg.frozen[end] = fr1
                end
            end
        end
        if _design_folded(d)
            _restore_verts!(d, snap)
            break
        end
    end
    freeze_dirichlet!(d)
    return d
end

"""
    move_boundary_standin!(d, dad, DTb, opt; Atarget) -> (d, nmoved)

First-order feasible-direction step with `s = DT` on Γ_d (same kinematics as
Portela): `vn ∝ (DT − λ)`, area projected to `Atarget − A`. Both ways unless
`opt.standin_inward`. Dirichlet / load patches stay frozen.
"""
function move_boundary_standin!(d::TopologyDesign, dad, DTb, opt::PachecoOptions;
        Atarget::Real, α::Real=opt.standin_α, vmax::Real=opt.vmax)
    freeze_dirichlet!(d)
    _densify_neumann!(d)
    A = design_area(d)
    intW, L = _design_WL(d, dad, DTb)
    L < 1e-16 && return d, 0
    meanW = intW / L
    λ = meanW - (Atarget - A) / max(α * L, 1e-16)
    ok, nmov = _move_portela!(d, dad, DTb, λ, α, vmax, :normal, Point2D(NaN, NaN),
        Atarget - A; inward=opt.standin_inward)
    ok || return d, 0
    freeze_dirichlet!(d)
    return d, nmov
end

function _closest_indices(pts, loopv)
    [argmin(j -> norm(p - loopv[j]), eachindex(loopv)) for p in pts]
end

# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------

function _solve_design(d::TopologyDesign, opt::PachecoOptions)
    dad = bemdata_from_loops(d; d_min=opt.d_min)
    H_G_full_direct(dad; npg=opt.npg, threaded=true)
    solve(dad)
    DTb, DTi = topological_derivative(dad)
    if opt.motion === :quantile && 0 < opt.pct < 1 && !isempty(DTb)
        cut = quantile(DTb, opt.pct)
        DTb = copy(DTb); DTb[DTb .> cut] .= cut
        if !isempty(DTi)
            DTi = copy(DTi); DTi[DTi .> cut] .= cut
        end
    end
    J = design_objective(dad)
    A = design_area(d)
    return dad, DTb, DTi, J, A
end

"""
    solve_topology!(design, opt=PachecoOptions(); history=true) -> (design, dad, hist)

Pacheco loop: BEM → DT → (optional iso-cut) → node motion,
until `A ≤ (1-ΔA) A0` or `maxiter`.

`opt.motion = :quantile` (default) is the original inward-only low-DT
recession. `:standin` uses `vn ∝ (DT − λ)` with area projection (MMFD-lite).
`:jump` solves the same first-order step with JuMP (`jump_mode=:linear` HiGHS
LP, or `:mma` NLopt MMA).
"""
function solve_topology!(d::TopologyDesign, opt::PachecoOptions=PachecoOptions(); history::Bool=true)
    freeze_dirichlet!(d)
    _densify_neumann!(d)
    dad, DTb, DTi, J, A = _solve_design(d, opt)
    A0 = A
    Ap = (1 - opt.ΔA) * A0
    hist = TopologyHistory()
    history && push!(hist; area=A, J=J, maxDT=maximum(DTb), n_holes=n_holes(d), design=d)
    mot = opt.motion
    mot in (:quantile, :standin, :jump) ||
        throw(ArgumentError("PachecoOptions.motion must be :quantile, :standin, or :jump, got $mot"))
    opt.verbose && println("Pacheco($(mot)) iter 0  A=$(round(A; digits=4))  J=$(_fmt_obj(J))  holes=$(n_holes(d))")

    iter = 0
    Atol = Ap * (1 + max(opt.area_rtol, 0.0))
    Amin = 0.97 * Ap
    pct0 = opt.pct
    vmax_base = opt.vmax
    while A > Atol && iter < opt.maxiter
        iter += 1
        Aprev = A
        remain_frac = (A - Ap) / max(A0, 1e-12)
        if mot === :quantile
            # far from Ap: recede more of Γ (lower pct) with a larger step
            opt.pct = remain_frac > 0.08 ? min(pct0, 0.48) :
                      remain_frac > 0.04 ? min(pct0, 0.68) : pct0
            opt.vmax = vmax_base * clamp(remain_frac / 0.10, 0.40, 1.7)
        else
            opt.vmax = vmax_base
        end
        d_prev = copy_design(d)
        allow_nuc = opt.nucleate_every < typemax(Int)
        nuc_every = if !allow_nuc
            typemax(Int)
        elseif remain_frac > 0.12
            min(opt.nucleate_every, 2)
        else
            opt.nucleate_every
        end
        do_nuc = allow_nuc && ((iter == 1 && opt.nucleate_first) ||
                 (iter > 1 && nuc_every < typemax(Int) && (iter - 1) % nuc_every == 0))
        if allow_nuc && d.properties isa Elasticity && !isempty(DTi) && !isempty(DTb)
            do_nuc = do_nuc || (minimum(DTb) * 20 > minimum(DTi))
        end
        nadd = 0
        if do_nuc
            nadd = nucleate_holes!(d, dad, DTb, DTi, opt)
            opt.verbose && nadd > 0 && println("  nucleated $nadd curve(s)")
        end
        if mot === :standin || mot === :jump
            d_nuced = copy_design(d)
            Atarget = A > Ap ? max(Ap, A - opt.volume_step * (A - Ap)) : Ap
            Ψ = _Ψ0(dad)
            area_viol = A > Ap * (1 + opt.area_rtol)
            accepted = false
            αtry, vtry = opt.standin_α, vmax_base
            local dad_try = dad
            local DTb_try = DTb
            local DTi_try = DTi
            local J_try = J
            local A_try = A
            for _ in 1:max(opt.nsearch, 1)
                d.loops = deepcopy(d_nuced.loops)
                nmov = if mot === :jump
                    _, nm = move_boundary_jump!(d, dad, DTb, opt; Atarget, vmax=vtry)
                    nm
                else
                    _, nm = move_boundary_standin!(d, dad, DTb, opt; Atarget,
                        α=αtry, vmax=vtry)
                    nm
                end
                if nmov == 0 || _design_folded(d)
                    d.loops = deepcopy(d_nuced.loops)
                    αtry *= 0.5
                    vtry *= 0.5
                    continue
                end
                dad_try, DTb_try, DTi_try, J_try, A_try = _solve_design(d, opt)
                folded = _design_folded(d) || !isfinite(J_try) || J_try <= 0
                Ψ_try = _Ψ0(dad_try)
                Ψ_ok = Ψ_try <= Ψ + 1e-3 * max(abs(Ψ), 1e-12)
                area_ok = A_try > Amin && A_try < 1.15 * A0
                progress = area_viol ? (A_try < A) : Ψ_ok
                if folded || !area_ok || !progress
                    d.loops = deepcopy(d_nuced.loops)
                    αtry *= 0.5
                    vtry *= 0.5
                    continue
                end
                accepted = true
                break
            end
            opt.pct = pct0
            opt.vmax = vmax_base
            if !accepted
                d.loops = deepcopy(d_prev.loops)
                vmax_base *= 0.55
                opt.vmax = vmax_base
                opt.verbose && println("  revert $mot; vmax → $(round(vmax_base; digits=4))")
                A <= Atol && break
                continue
            end
            dad, DTb, DTi, J, A = dad_try, DTb_try, DTi_try, J_try, A_try
        else
            if nadd == 0 || iter > 1
                move_boundary!(d, dad, DTb, opt)
            end
            opt.pct = pct0
            opt.vmax = vmax_base
            dad, DTb, DTi, J, A = _solve_design(d, opt)
            folded = _design_folded(d) || !isfinite(J) || J <= 0
            if A < Amin || folded
                d.loops = deepcopy(d_prev.loops)
                dad, DTb, DTi, J, A = _solve_design(d, opt)
                vmax_base *= 0.55
                opt.vmax = vmax_base
                why = folded ? "folded Γ" : "A < Ap"
                opt.verbose && println("  revert $why; vmax → $(round(vmax_base; digits=4))")
                A <= Atol && break
                continue
            end
            dA = Aprev - A
            if dA < opt.ΔAmin
                vmax_base *= 1.18
                opt.vmax = vmax_base
                opt.verbose && println("  vmax → $(round(vmax_base; digits=4))")
            end
        end
        history && push!(hist; area=A, J=J, maxDT=maximum(DTb), n_holes=n_holes(d), design=d)
        opt.verbose && println("Pacheco($(mot)) iter $iter  A=$(round(A; digits=4))  J=$(_fmt_obj(J))  holes=$(n_holes(d))")
    end
    opt.pct = pct0
    opt.vmax = vmax_base
    return d, dad, hist
end

# Filled in Topology.__init__ from JumpShape.jl (JuMP is not a precompile dep).
function move_boundary_jump! end

function pacheco_step!(d, dad, DTb, DTi, opt)
    nucleate_holes!(d, dad, DTb, DTi, opt)
    if opt.motion === :standin
        A = design_area(d)
        Atarget = A * (1 - max(opt.volume_step, 0.02))
        move_boundary_standin!(d, dad, DTb, opt; Atarget)
    elseif opt.motion === :jump
        A = design_area(d)
        Atarget = A * (1 - max(opt.volume_step, 0.02))
        move_boundary_jump!(d, dad, DTb, opt; Atarget)
    else
        move_boundary!(d, dad, DTb, opt)
    end
    return d
end

"""
    match_volume!(design, Ap; opt=PachecoOptions(), rtol=opt.area_rtol)

Pacheco motion until `design_area` is in `[0.97 Ap, Ap*(1+rtol)]`. A second
more aggressive pass runs if the first stop is still above the band.
"""
function match_volume!(d::TopologyDesign, Ap::Real;
        opt::PachecoOptions=PachecoOptions(), rtol::Float64=NaN)
    rtol = isfinite(rtol) ? rtol : opt.area_rtol
    freeze_dirichlet!(d)
    A = design_area(d)
    if A <= Ap * (1 + rtol)
        dad, _, _, J, A = _solve_design(d, opt)
        return d, dad, J, A
    end
    ΔA0, rtol0, max0, verb0, pct0, vmax0 = opt.ΔA, opt.area_rtol, opt.maxiter, opt.verbose, opt.pct, opt.vmax
    nf0, ne0 = opt.nucleate_first, opt.nucleate_every
    opt.ΔA = clamp(1 - Ap / max(A, 1e-12), 0.02, 0.95)
    opt.area_rtol = rtol
    d, dad, _ = solve_topology!(d, opt; history=false)
    A = design_area(d)
    if A > Ap * (1 + rtol)
        opt.pct = min(pct0, 0.45)
        opt.vmax = vmax0 * 1.45
        if ne0 < typemax(Int)
            opt.nucleate_first = true
            opt.nucleate_every = min(ne0, 2)
        end
        opt.maxiter = max(opt.maxiter, 12)
        opt.ΔA = clamp(1 - Ap / max(A, 1e-12), 0.02, 0.95)
        opt.verbose && println("  volume pass 2  A=$(round(A; digits=3)) → Ap=$(round(Ap; digits=3))")
        d, dad, _ = solve_topology!(d, opt; history=false)
        A = design_area(d)
    end
    J = design_objective(dad)
    opt.ΔA, opt.area_rtol, opt.maxiter, opt.verbose, opt.pct, opt.vmax =
        ΔA0, rtol0, max0, verb0, pct0, vmax0
    opt.nucleate_first, opt.nucleate_every = nf0, ne0
    return d, dad, J, A
end
