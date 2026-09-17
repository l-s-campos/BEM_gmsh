# Level-set topology optimization (Laplace heat conductor or plane-stress elasticity).
# Design variable φ on a fixed Cartesian grid; BEM on {φ=0} via bemdata_from_loops.
#
# Amstutz: φ ← DT − τ, τ bisected so |Ω| meets the area target.
# Hamilton–Jacobi: φ_t + v_n |∇φ| = 0 with v_n = DT − λ, plus a weak DT source
# for nucleation, and Russo–Smereka reinitialization.

export LevelSetGrid, LevelSetOptions
export amstutz_step!, hj_step!, solve_levelset!
export signed_distance!, reinit_russo_smereka!
export phi_to_design, design_to_phi!

mutable struct LevelSetGrid
    xs::Vector{Float64}
    ys::Vector{Float64}
    φ::Matrix{Float64}
    dx::Float64
    dy::Float64
end

function LevelSetGrid(xs::AbstractVector, ys::AbstractVector, φ::AbstractMatrix)
    dx = length(xs) > 1 ? abs(xs[2] - xs[1]) : 1.0
    dy = length(ys) > 1 ? abs(ys[2] - ys[1]) : 1.0
    return LevelSetGrid(collect(Float64, xs), collect(Float64, ys), Matrix{Float64}(φ), dx, dy)
end

function LevelSetGrid(d::TopologyDesign; ngrid::Integer=64, pad::Float64=0.08)
    outer = loop_vertices(d.loops[1])
    xmin = minimum(p[1] for p in outer)
    xmax = maximum(p[1] for p in outer)
    ymin = minimum(p[2] for p in outer)
    ymax = maximum(p[2] for p in outer)
    dx0 = xmax - xmin
    dy0 = ymax - ymin
    xs = collect(range(xmin - pad * dx0, xmax + pad * dx0; length=ngrid))
    ys = collect(range(ymin - pad * dy0, ymax + pad * dy0; length=ngrid))
    φ = zeros(length(xs), length(ys))
    g = LevelSetGrid(xs, ys, φ)
    design_to_phi!(g, d)
    return g
end

@kwdef mutable struct LevelSetOptions
    ΔA::Float64 = 0.5
    method::Symbol = :amstutz          # :amstutz or :hj
    ngrid::Int = 64
    maxiter::Int = 40
    dt_cfl::Float64 = 0.4
    n_hj::Int = 8                      # HJ substeps per BEM solve
    n_reinit::Int = 5
    nucleate_weight::Float64 = 0.15    # interior DT source for HJ
    volume_step::Float64 = 0.20        # fraction of (A0-Ap) per Amstutz iter
    min_loop_area::Float64 = 1e-3
    d_min::Float64 = 0.01
    npg::Int = 16
    verbose::Bool = true
    nel_per_loop::Int = 24
end

# -----------------------------------------------------------------------------
# φ ↔ design
# -----------------------------------------------------------------------------

function design_to_phi!(g::LevelSetGrid, d::TopologyDesign)
    nx, ny = length(g.xs), length(g.ys)
    @inbounds for j in 1:ny, i in 1:nx
        p = Point2D(g.xs[i], g.ys[j])
        s = _signed_dist_design(p, d)
        g.φ[i, j] = s
    end
    return g
end

function _signed_dist_design(p, d::TopologyDesign)
    outer = loop_vertices(d.loops[1])
    s = _signed_dist_poly(p, outer)          # >0 inside outer
    for k in 2:length(d.loops)
        hv = loop_vertices(d.loops[k])
        sh = _signed_dist_poly(p, hv)        # >0 inside hole polygon
        s = min(s, -sh)                      # solid = outside holes
    end
    return s
end

function _signed_dist_poly(p, verts)
    d = _dist_to_polyline(p, verts)
    return in_polygon(p, verts) ? d : -d
end

"""Extract `{φ=0}` as `TopologyDesign` loops. Dirichlet segments of `proto` are frozen in."""
function phi_to_design(g::LevelSetGrid, proto::TopologyDesign; min_area::Float64=1e-3, nel::Int=24)
    lines = marching_squares(g.xs, g.ys, g.φ, 0.0)
    outer0 = loop_vertices(proto.loops[1])
    Aref = _area_from_phi(g)
    loops = Vector{BoundarySegment}[]
    open_lines = Vector{Point2D}[]
    for line in lines
        length(line) < 8 && continue
        if norm(line[1] - line[end]) < 1e-3
            pts = line[1] ≈ line[end] ? copy(line[1:end-1]) : copy(line)
            chaikin_smooth!(pts; closed=true, passes=1)
            abs(polygon_area(pts)) < min_area && continue
            pts = resample_polyline(pts, nel; closed=true)
            push!(loops, [hole_segment(pts, proto, nel)])
        else
            push!(open_lines, line)
        end
    end
    if !isempty(open_lines)
        A0outer = abs(polygon_area(outer0))
        pts = copy(outer0)
        Acur = A0outer
        for line in sort(open_lines; by=length, rev=true)
            cand = _close_isocurve_on_outer(line, pts, Aref)
            length(cand) < 8 && continue
            Ac = abs(polygon_area(cand))
            # accept a bite that moves toward Aref without collapsing
            if Ac < Acur && Ac ≥ 0.7 * max(Aref, min_area) && Ac ≤ 1.05 * A0outer
                pts = cand
                Acur = Ac
            end
        end
        if Acur ≥ min_area && Acur < 0.999 * A0outer
            pts = resample_polyline(pts, nel; closed=true)
            pushfirst!(loops, [hole_segment(pts, proto, nel)])
        end
    end
    isempty(loops) && return deepcopy(proto)
    areas = [abs(polygon_area(loop_vertices(s))) for s in loops]
    # one outer (largest) + holes strictly inside it
    perm = sortperm(areas; rev=true)
    loops = loops[perm]
    outer_pts = loop_vertices(loops[1])
    kept = Vector{Vector{BoundarySegment}}()
    push!(kept, loops[1])
    for k in 2:length(loops)
        hv = loop_vertices(loops[k])
        c = sum(hv) / length(hv)
        in_polygon(c, outer_pts) && push!(kept, loops[k])
    end
    loops = kept
    _ensure_loop_orientation!(loops[1]; hole=false)
    for k in 2:length(loops)
        _ensure_loop_orientation!(loops[k]; hole=true)
    end
    d = TopologyDesign(loops; degree=proto.degree, k=proto.k,
        properties=proto.properties,
        nx_int=proto.nx_int, ny_int=proto.ny_int, name=proto.name)
    _paint_dirichlet!(d, proto)
    freeze_dirichlet!(d)
    return d
end

"""Close an open iso-curve by the original-outer arc that keeps the loop inside the solid."""
function _close_isocurve_on_outer(line::Vector{Point2D}, outer::Vector{Point2D})
    n = length(outer)
    n < 3 && return Point2D[]
    i1 = argmin(i -> norm(line[1] - outer[i]), 1:n)
    i2 = argmin(i -> norm(line[end] - outer[i]), 1:n)
    i1 == i2 && return Point2D[]
    arc12 = outer[_arc_indices(i1, i2, n)]
    arc21 = outer[_arc_indices(i2, i1, n)]
    cand1 = [line; arc21[2:end-1]]
    cand2 = [reverse(line); arc12[2:end-1]]
    function _ok(pts)
        length(pts) < 8 && return 0.0
        a = abs(polygon_area(pts))
        a <= 0 && return 0.0
        polygon_area(pts) < 0 && reverse!(pts)
        return a
    end
    a1, a2 = _ok(cand1), _ok(cand2)
    (a1 == 0 && a2 == 0) && return Point2D[]
    return a1 >= a2 ? cand1 : cand2
end

"""Close an open iso-curve picking the candidate whose area is closest to `Aref`."""
function _close_isocurve_on_outer(line::Vector{Point2D}, outer::Vector{Point2D}, Aref::Float64)
    n = length(outer)
    n < 3 && return Point2D[]
    i1 = argmin(i -> norm(line[1] - outer[i]), 1:n)
    i2 = argmin(i -> norm(line[end] - outer[i]), 1:n)
    i1 == i2 && return Point2D[]
    arc12 = outer[_arc_indices(i1, i2, n)]
    arc21 = outer[_arc_indices(i2, i1, n)]
    cand1 = [line; arc21[2:end-1]]
    cand2 = [reverse(line); arc12[2:end-1]]
    function _area(pts)
        length(pts) < 8 && return 0.0
        a = polygon_area(pts)
        a < 0 && reverse!(pts)
        return abs(a)
    end
    a1, a2 = _area(cand1), _area(cand2)
    (a1 == 0 && a2 == 0) && return Point2D[]
    a1 == 0 && return cand2
    a2 == 0 && return cand1
    return abs(a1 - Aref) <= abs(a2 - Aref) ? cand1 : cand2
end

"""
Walk the original outer loop, jumping along open iso-curves whenever an
endpoint sits on Γ. Dirichlet stretches between jumps stay on the original
boundary so Γ_D is preserved.
"""
function _rebuild_outer_from_isocurves(outer::Vector{Point2D}, lines::Vector{Vector{Point2D}})
    n = length(outer)
    n < 3 && return Point2D[]
    hits = NamedTuple{(:i1, :i2, :line),Tuple{Int,Int,Vector{Point2D}}}[]
    for line in lines
        length(line) < 4 && continue
        i1 = argmin(i -> norm(line[1] - outer[i]), 1:n)
        i2 = argmin(i -> norm(line[end] - outer[i]), 1:n)
        i1 == i2 && continue
        # require endpoints actually near the outer (not a floating fragment)
        (norm(line[1] - outer[i1]) < 0.15 && norm(line[end] - outer[i2]) < 0.15) || continue
        push!(hits, (i1=i1, i2=i2, line=line))
    end
    isempty(hits) && return Point2D[]
    used = falses(length(hits))
    pts = Point2D[]
    i = 1
    start = 1
    guard = 0
    jumped = false
    while guard < n + 2 * length(hits) + 2
        guard += 1
        hit = 0
        fwd = true
        for k in eachindex(hits)
            used[k] && continue
            if hits[k].i1 == i
                hit = k; fwd = true; break
            elseif hits[k].i2 == i
                hit = k; fwd = false; break
            end
        end
        if hit != 0
            ln = hits[hit].line
            if fwd
                append!(pts, ln)
                i = hits[hit].i2
            else
                append!(pts, reverse(ln))
                i = hits[hit].i1
            end
            used[hit] = true
            jumped = true
        else
            push!(pts, outer[i])
            i = i == n ? 1 : i + 1
        end
        if jumped && i == start && length(pts) ≥ 8
            break
        end
        if !jumped && i == start && guard > n
            break
        end
    end
    jumped || return Point2D[]
    length(pts) < 8 && return Point2D[]
    a = polygon_area(pts)
    a < 0 && reverse!(pts)
    return pts
end

"""Keep φ ≥ +dx in a tube around each Dirichlet / load patch so Γ_D is never cut."""
function _protect_dirichlet!(g::LevelSetGrid, proto::TopologyDesign)
    r = 2.5 * max(g.dx, g.dy)
    diris = [s for segs in proto.loops for s in segs if _is_fixed_segment(s)]
    isempty(diris) && return g
    @inbounds for j in eachindex(g.ys), i in eachindex(g.xs)
        p = Point2D(g.xs[i], g.ys[j])
        for s in diris
            a, b = s.verts[1], s.verts[end]
            if _dist_point_seg(p, a, b) < r
                g.φ[i, j] = max(g.φ[i, j], r)
                break
            end
        end
    end
    return g
end

"""Tag extracted-loop vertices that sit on a proto Dirichlet / load patch."""
function _paint_dirichlet!(d::TopologyDesign, proto::TopologyDesign)
    fixed = [s for segs in proto.loops for s in segs if _is_fixed_segment(s)]
    isempty(fixed) && return d
    outer = loop_vertices(proto.loops[1])
    Lx = maximum(p[1] for p in outer) - minimum(p[1] for p in outer)
    Ly = maximum(p[2] for p in outer) - minimum(p[2] for p in outer)
    tol = max(0.12, 0.12 * max(Lx, Ly))
    newloops = Vector{Vector{BoundarySegment}}()
    for segs in d.loops
        verts = loop_vertices(segs)
        n = length(verts)
        n < 3 && continue
        lab = zeros(Int, n)
        for i in 1:n
            for (k, s) in enumerate(fixed)
                if _dist_point_seg(verts[i], s.verts[1], s.verts[end]) < tol
                    lab[i] = k
                    break
                end
            end
        end
        push!(newloops, _labeled_loop_segments(verts, lab, fixed, segs[1].n_el, proto))
    end
    isempty(newloops) || (d.loops = newloops)
    return d
end

function _segment_from_label(pts, lab, fixed, nel, proto)
    lab == 0 && return hole_segment(pts, proto, nel)
    s = fixed[lab]
    return BoundarySegment(pts; bc=copy(s.bc), value=copy(s.value), n_el=nel)
end

function _labeled_loop_segments(verts, lab, fixed, n_el, proto)
    n = length(verts)
    segs = BoundarySegment[]
    i = 1
    while i <= n
        j = i
        while j < n && lab[j + 1] == lab[i]
            j += 1
        end
        j2 = j == n ? 1 : j + 1
        pts = j == n ? verts[i:n] : verts[i:j2]
        if j == n && lab[1] == lab[i] && !isempty(segs) &&
                ((lab[i] == 0 && !_is_fixed_segment(segs[1])) ||
                 (lab[i] > 0 && segs[1].bc == fixed[lab[i]].bc && segs[1].value == fixed[lab[i]].value))
            segs[1].verts = vcat(pts, segs[1].verts)
            segs[1].n_el += max(length(pts) - 1, 1)
            segs[1].frozen = BitVector(fill(_is_fixed_segment(segs[1]), length(segs[1].verts)))
            break
        end
        nel = max(length(pts) - 1, 2)
        push!(segs, _segment_from_label(pts, lab[i], fixed, nel, proto))
        i = j + 1
    end
    isempty(segs) && push!(segs, hole_segment(verts, proto, n_el))
    return segs
end

# -----------------------------------------------------------------------------
# Reinitialization
# -----------------------------------------------------------------------------

function signed_distance!(g::LevelSetGrid; niter::Integer=20, dt=nothing)
    Δ = min(g.dx, g.dy)
    τ = dt === nothing ? 0.4 * Δ : dt
    for _ in 1:niter
        reinit_russo_smereka!(g, τ)
    end
    return g
end

"""One Russo–Smereka step: φ_t + S(φ0)(|∇φ| − 1) = 0."""
function reinit_russo_smereka!(g::LevelSetGrid, dt::Float64)
    φ = g.φ
    nx, ny = size(φ)
    φn = copy(φ)
    dx, dy = g.dx, g.dy
    @inbounds for j in 2:(ny - 1), i in 2:(nx - 1)
        s = φn[i, j] / sqrt(φn[i, j]^2 + dx^2)
        φx_m = (φn[i, j] - φn[i - 1, j]) / dx
        φx_p = (φn[i + 1, j] - φn[i, j]) / dx
        φy_m = (φn[i, j] - φn[i, j - 1]) / dy
        φy_p = (φn[i, j + 1] - φn[i, j]) / dy
        if s > 0
            gx = _gplus(φx_m, φx_p)
            gy = _gplus(φy_m, φy_p)
        else
            gx = _gminus(φx_m, φx_p)
            gy = _gminus(φy_m, φy_p)
        end
        mag = sqrt(gx + gy)
        φ[i, j] = φn[i, j] - dt * s * (mag - 1)
    end
    # Neumann copy on the frame
    φ[1, :] .= φ[2, :]
    φ[end, :] .= φ[end - 1, :]
    φ[:, 1] .= φ[:, 2]
    φ[:, end] .= φ[:, end - 1]
    return g
end

@inline _gplus(a, b) = max(max(a, 0)^2, max(-b, 0)^2)
@inline _gminus(a, b) = max(max(-a, 0)^2, max(b, 0)^2)

# -----------------------------------------------------------------------------
# HJ / Amstutz
# -----------------------------------------------------------------------------

function _dt_field(g::LevelSetGrid, dad, DTb, DTi)
    pts = Point2D[dad.Nodes; dad.internalNodes]
    val = vcat(DTb, DTi)
    return interpolate_to_grid(pts, val, g.xs, g.ys)
end

function _area_from_phi(g::LevelSetGrid)
    A = 0.0
    nx, ny = size(g.φ)
    @inbounds for j in 1:(ny - 1), i in 1:(nx - 1)
        μ = 0.25 * (max(g.φ[i, j], 0) + max(g.φ[i+1, j], 0) +
                    max(g.φ[i, j+1], 0) + max(g.φ[i+1, j+1], 0))
        # Heaviside ≈ fraction of positive corners
        npos = (g.φ[i, j] > 0) + (g.φ[i+1, j] > 0) + (g.φ[i, j+1] > 0) + (g.φ[i+1, j+1] > 0)
        A += (npos / 4) * g.dx * g.dy
    end
    return A
end

"""Amstutz update: φ = DT − τ with τ so the superlevel has area `Atarget`."""
function amstutz_step!(g::LevelSetGrid, dad, DTb, DTi, Atarget::Float64)
    DT = _dt_field(g, dad, DTb, DTi)
    zmax = maximum(z for z in DT if isfinite(z); init=0.0)
    @inbounds for i in eachindex(DT)
        isfinite(DT[i]) || (DT[i] = -1e30)
    end
    lo = minimum(DT)
    hi = zmax
    φbest = copy(g.φ)
    for _ in 1:24
        τ = 0.5 * (lo + hi)
        @inbounds for i in eachindex(g.φ)
            g.φ[i] = DT[i] - τ
        end
        A = _area_from_phi(g)
        if A > Atarget
            lo = τ          # raise threshold → less solid
            φbest .= g.φ
        else
            hi = τ
        end
    end
    g.φ .= φbest
    return g
end

"""
Shift ``φ ← φ - c`` so ``|{φ>0}|`` matches `Atarget` (Allaire volume projection).
"""
function _shift_phi_to_area!(g::LevelSetGrid, Atarget::Float64)
    lo = minimum(g.φ)
    hi = maximum(g.φ)
    φ0 = copy(g.φ)
    best = copy(g.φ)
    for _ in 1:20
        c = 0.5 * (lo + hi)
        @inbounds for i in eachindex(g.φ)
            g.φ[i] = φ0[i] - c
        end
        A = _area_from_phi(g)
        if A > Atarget
            lo = c          # raise iso → less solid
            best .= g.φ
        else
            hi = c
        end
    end
    g.φ .= best
    return g
end

"""
HJ substeps for Ω = {φ > 0}.

Outward speed of the solid is v_n = DT - λ (recede where DT is low).
Because ∇φ points into the solid, the consistent scheme is

    φ_t - v_n |∇φ| = 0    ⇒    φ ← φ + Δt v_n |∇φ|

Void is masked (DT = 0) so it cannot nucleate spurious solid. After advection
and a few reinit steps, φ is shifted so the area matches `Atarget`.
"""
function hj_step!(g::LevelSetGrid, dad, DTb, DTi, Atarget::Float64, opt::LevelSetOptions)
    DT = _dt_field(g, dad, DTb, DTi)
    nx, ny = size(g.φ)
    DTsol = Float64[]
    @inbounds for j in 1:ny, i in 1:nx
        if !isfinite(DT[i, j]) || g.φ[i, j] < 0
            DT[i, j] = 0.0          # void: v_n = -λ < 0 → φ decreases
        else
            push!(DTsol, DT[i, j])
        end
    end
    A = _area_from_phi(g)
    μ = isempty(DTsol) ? 0.0 : mean(DTsol)
    σ = isempty(DTsol) ? 1.0 : max(std(DTsol), 1e-8)
    # raise λ when the solid is too large so more of the front recedes
    λ = μ + 1.5 * σ * (A - Atarget) / max(abs(Atarget), abs(A), 1e-8)
    vmax = 1e-8
    @inbounds for z in DT
        vmax = max(vmax, abs(z - λ))
    end
    Δ = min(g.dx, g.dy)
    dt = opt.dt_cfl * Δ / vmax
    band = 4 * Δ
    for _ in 1:opt.n_hj
        φn = copy(g.φ)
        @inbounds for j in 2:(ny - 1), i in 2:(nx - 1)
            # narrow band: freeze far-field SDF
            abs(φn[i, j]) > band && continue
            vn = DT[i, j] - λ
            φx_m = (φn[i, j] - φn[i - 1, j]) / g.dx
            φx_p = (φn[i + 1, j] - φn[i, j]) / g.dx
            φy_m = (φn[i, j] - φn[i, j - 1]) / g.dy
            φy_p = (φn[i, j + 1] - φn[i, j]) / g.dy
            # φ_t + c|∇φ|=0 with c = -v_n
            if vn > 0
                mag = sqrt(_gminus(φx_m, φx_p) + _gminus(φy_m, φy_p))
            else
                mag = sqrt(_gplus(φx_m, φx_p) + _gplus(φy_m, φy_p))
            end
            src = 0.0
            if φn[i, j] > 0
                src = opt.nucleate_weight * min(DT[i, j] - λ, 0.0)   # holes only
            end
            g.φ[i, j] = φn[i, j] + dt * vn * mag + dt * src
        end
        g.φ[1, :] .= g.φ[2, :]
        g.φ[end, :] .= g.φ[end - 1, :]
        g.φ[:, 1] .= g.φ[:, 2]
        g.φ[:, end] .= g.φ[:, end - 1]
    end
    nre = max(opt.n_reinit, 0)
    nre > 0 && signed_distance!(g; niter=nre)
    _shift_phi_to_area!(g, Atarget)
    return g
end

function solve_levelset!(d::TopologyDesign, opt::LevelSetOptions=LevelSetOptions(); history::Bool=true)
    freeze_dirichlet!(d)
    proto = deepcopy(d)
    g = LevelSetGrid(d; ngrid=opt.ngrid)
    dad = bemdata_from_loops(d; d_min=opt.d_min)
    H_G_full_direct(dad; npg=opt.npg)
    solve(dad)
    DTb, DTi = topological_derivative(dad)
    J = design_objective(dad)
    A = design_area(d)
    A0 = A
    Ap = (1 - opt.ΔA) * A0
    hist = TopologyHistory()
    history && push!(hist; area=A, J=J, maxDT=maximum(DTb), n_holes=n_holes(d), design=d)
    opt.verbose && println("LSM($(opt.method)) iter 0  A=$(round(A; digits=4))  J=$(_fmt_obj(J))")
    φ_keep = copy(g.φ)

    iter = 0
    Atol = Ap + 0.02 * A0
    while A > Atol && iter < opt.maxiter
        iter += 1
        Atarget = max(Ap, A - opt.volume_step * (A0 - Ap))
        if opt.method === :amstutz
            amstutz_step!(g, dad, DTb, DTi, Atarget)
        elseif opt.method === :hj
            hj_step!(g, dad, DTb, DTi, Atarget, opt)
        else
            throw(ArgumentError("LevelSetOptions.method must be :amstutz or :hj"))
        end
        _protect_dirichlet!(g, proto)
        dtry = phi_to_design(g, proto; min_area=opt.min_loop_area, nel=opt.nel_per_loop)
        Atry = design_area(dtry)
        if Atry > 1.05 * A0 || Atry > A * 1.02
            opt.verbose && println("  reject A=$(round(Atry; digits=4)); keep previous")
            g.φ .= φ_keep
            history && push!(hist; area=A, J=J, maxDT=maximum(DTb), n_holes=n_holes(d), design=d)
            continue
        end
        dadtry = bemdata_from_loops(dtry; d_min=opt.d_min)
        H_G_full_direct(dadtry; npg=opt.npg)
        solve(dadtry)
        Jtry = design_objective(dadtry)
        if !(Jtry > 0)
            opt.verbose && println("  reject J=$(_fmt_obj(Jtry)); keep previous")
            g.φ .= φ_keep
            continue
        end
        d = dtry
        dad = dadtry
        DTb, DTi = topological_derivative(dad)
        J = Jtry
        A = Atry
        design_to_phi!(g, d)          # sync SDF to the extracted BEM contour
        _protect_dirichlet!(g, proto)
        φ_keep = copy(g.φ)
        history && push!(hist; area=A, J=J, maxDT=maximum(DTb), n_holes=n_holes(d), design=d)
        opt.verbose && println("LSM($(opt.method)) iter $iter  A=$(round(A; digits=4))  J=$(_fmt_obj(J))  holes=$(n_holes(d))")
    end
    return d, dad, hist, g
end
