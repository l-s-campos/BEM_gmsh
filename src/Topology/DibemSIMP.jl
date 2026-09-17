# Nodal SIMP on DIBEM (fixed Γ) then iso-cut of low ρ into explicit holes.
# Laplace:  K_i = Kmin + (K0-Kmin) ρ_i^p ; state = solve_heterogeneous!.
# Elasticity: E_i = Emin + (E0-Emin) ρ_i^p with Emin = Kmin*E0, E0 = dad.E;
#   maximize stiffness (≡ minimize compliance) at fixed volume.
# After n_simp steps, ρ=ρ_cut iso-curves become traction-free holes.

export DibemSimpOptions, simp_conductivity, simp_young, dibem_volume_weights
export density_filter, dibem_simp_step!, cut_low_density!, solve_dibem_simp!
export nodal_DT, density_from_dt, solve_dt_density!

"""Options for [`solve_dibem_simp!`](@ref): penalization `p`, void ratio `Kmin`,
volume fraction of the **projected** density `H_β(filter(ρ))`, hat-filter
`rmin`, OC with `p`/`β` continuation (`method=:simp`) or a DT threshold
(`method=:dt`), volume-matched iso-cut, then **shape-only** Pacheco to `Ap`.
Laplace uses `K0`; elasticity uses `dad.properties.E` with `Emin = Kmin E0`."""
@kwdef mutable struct DibemSimpOptions
    p::Float64 = 3.0
    p_start::Float64 = 1.0          # continuation: p_start → p
    K0::Float64 = 1.0
    Kmin::Float64 = 1e-3
    ρmin::Float64 = 1e-3
    volfrac::Float64 = 0.5
    rmin::Float64 = 0.08
    n_simp::Int = 40
    η::Float64 = 0.5
    move::Float64 = 0.15
    ρ_cut::Float64 = 0.4
    rbf = PHS(1; poly_deg=-1)
    npg::Int = 12
    ngrid::Int = 61
    β_start::Float64 = 1.0          # Heaviside continuation β_start → β_end
    β_end::Float64 = 8.0
    min_hole_dist::Float64 = 0.008
    min_hole_area::Float64 = 4e-4
    method::Symbol = :simp          # :simp (energy) or :dt (topological derivative)
    match_area::Bool = true         # pick ρ_cut so grid solid ≈ volfrac
    cut::Bool = true
    pacheco::Bool = true            # polish: a few DT motion steps to Ap
    pacheco_opt::PachecoOptions = PachecoOptions(maxiter=40, verbose=false,
        nucleate_first=false, nucleate_every=typemax(Int), area_rtol=0.08,
        min_hole_dist=0.008, min_hole_area=4e-4)
    verbose::Bool = true
end

_heaviside(ρ, β, η=0.5) = β <= 0 ? ρ :
    @. (tanh(β * η) + tanh(β * (ρ - η))) / (2 * tanh(β * η) + 1e-30)

function _heaviside_deriv(ρ::AbstractVector, β::Real, η=0.5)
    n = length(ρ)
    out = ones(n)
    β <= 0 && return out
    th = tanh(β * η)
    @inbounds for i in 1:n
        t = tanh(β * (ρ[i] - η))
        out[i] = (β / 2) * (1 - t * t) / (th + 1e-30)
    end
    return out
end

_simp_cont(it, n, a, b) = n <= 1 ? a : a + (b - a) * ((it - 1) / (n - 1))

"""`K = Kmin + (K0-Kmin) ρ^p` at each node."""
function simp_conductivity(ρ::AbstractVector, opt::DibemSimpOptions, p::Real=opt.p)
    K = similar(ρ)
    @inbounds for i in eachindex(ρ)
        K[i] = opt.Kmin + (opt.K0 - opt.Kmin) * ρ[i]^p
    end
    return K
end

"""`E = Emin + (E0-Emin) ρ^p` with `Emin = Kmin * E0` (`Kmin` = void/solid ratio)."""
function simp_young(ρ::AbstractVector, opt::DibemSimpOptions, E0::Real, p::Real=opt.p)
    Emin = opt.Kmin * E0
    E = similar(ρ)
    @inbounds for i in eachindex(ρ)
        E[i] = Emin + (E0 - Emin) * ρ[i]^p
    end
    return E
end

"""RIM nodal volumes `IF`; negatives clipped so OC stays well-posed."""
function dibem_volume_weights(dad::BEMdata, rbf)
    IF = if dad.properties isa Laplace
        IF0, _ = _dibem_rbf_volume(dad, rbf)
        IF0
    else
        _dibem_rbf_IF(dad, rbf, all_points(dad))
    end
    w = max.(IF, 0.0)
    s = sum(w)
    if s < 1e-14
        fill!(w, 1.0)
    end
    return w
end

"""Linear hat filter `W_ij = max(0, rmin − |x_i−x_j|)`."""
function density_filter(pts::AbstractVector, val::AbstractVector, rmin::Real)
    n = length(val)
    n == length(pts) || throw(DimensionMismatch("filter: pts and val length"))
    rmin = float(rmin)
    out = zeros(n)
    @inbounds for i in 1:n
        wsum = 0.0
        s = 0.0
        for j in 1:n
            w = rmin - norm(pts[i] - pts[j])
            w <= 0 && continue
            s += w * val[j]
            wsum += w
        end
        out[i] = wsum > 0 ? s / wsum : val[i]
    end
    return out
end

function _freeze_dirichlet(dad::BEMdata)
    return _freeze_design_nodes(dad)
end

"""Solid nodes: any Dirichlet DOF (Laplace or elasticity) plus loaded traction patches."""
function _freeze_design_nodes(dad::BEMdata{<:Laplace})
    fr = falses(dad.nt)
    @inbounds for i in 1:dad.n
        dad.BC[i] == 0 && (fr[i] = true)
    end
    return fr
end

function _freeze_design_nodes(dad::BEMdata{<:Elasticity})
    fr = falses(dad.nt)
    dim = dad.dimension
    @inbounds for i in 1:dad.n
        for d in 1:dim
            dof = dim * (i - 1) + d
            dof > length(dad.BC) && break
            if dad.BC[dof] == 0 || abs(dad.BV[dof]) > 0
                fr[i] = true
                break
            end
        end
    end
    return fr
end

"""Self-adjoint FGM indicator: `|∇T|² ∂K/∂ρ` from the heterogeneous Laplace solve."""
function dibem_simp_sensitivity(dad::BEMdata{<:Laplace}, ρ, w, opt::DibemSimpOptions, p::Real=opt.p)
    pts = all_points(dad)
    T = dad.T
    length(T) >= dad.nt || error("need dad.T from solve_heterogeneous!")
    gT = _grad_field(pts, view(T, 1:dad.nt), opt.rbf)
    dc = zeros(length(ρ))
    pm = max(p - 1, 0.0)
    dK0 = (opt.K0 - opt.Kmin) * p
    dim = length(gT)
    @inbounds for i in eachindex(ρ)
        g2 = 0.0
        for α in 1:dim
            g2 += gT[α][i]^2
        end
        dc[i] = g2 * dK0 * ρ[i]^pm * w[i]
    end
    return dc
end

"""Self-adjoint FGM indicator: `(σ_ref:ε) ∂E/∂ρ` from the heterogeneous elasticity solve."""
function dibem_simp_sensitivity(dad::BEMdata{<:Elasticity}, ρ, w, opt::DibemSimpOptions, p::Real=opt.p)
    sed = _elastic_sed_ref(dad, opt.rbf)
    dc = zeros(length(ρ))
    pm = max(p - 1, 0.0)
    scale = (1 - opt.Kmin) * p
    @inbounds for i in eachindex(ρ)
        dc[i] = max(sed[i], 0.0) * scale * ρ[i]^pm * w[i]
    end
    return dc
end

"""Optimality-criteria update that **maximizes** C at fixed volume fraction.

`volume(ρn)` defaults to `dot(ρn,w)/sum(w)`; pass the projected volume
`∫ H_β(filter(ρ))` (Xu 2010 / Wang–Lazarov–Sigmund 2011).
"""
function _oc_update(ρ, dc, w, volfrac, frozen; η=0.5, move=0.2, ρmin=1e-3, volume=nothing)
    ρn = copy(ρ)
    @inbounds for i in eachindex(ρ)
        frozen[i] && (ρn[i] = 1.0)
    end
    l1, l2 = 0.0, 1.0e9
    sw = sum(w)
    volof = volume === nothing ? (ρc -> dot(ρc, w) / sw) : volume
    @inbounds for _ in 1:50
        lmid = 0.5 * (l1 + l2)
        for i in eachindex(ρ)
            frozen[i] && continue
            be = dc[i] / (lmid * w[i] + 1e-30)
            be = max(be, 0.0)
            ρn[i] = clamp(ρ[i] * be^η, max(ρmin, ρ[i] - move), min(1.0, ρ[i] + move))
        end
        if volof(ρn) > volfrac
            l1 = lmid
        else
            l2 = lmid
        end
    end
    return ρn
end

"""Nodal topological derivative `DT` (boundary then interior), clipped at 0."""
function nodal_DT(dad::BEMdata)
    DTb, DTi = topological_derivative(dad)
    DT = zeros(dad.nt)
    n = min(length(DTb), dad.n)
    DT[1:n] .= DTb[1:n]
    if !isempty(DTi)
        ni = min(length(DTi), dad.nt - dad.n)
        DT[dad.n+1:dad.n+ni] .= DTi[1:ni]
    end
    @inbounds for i in eachindex(DT)
        DT[i] = max(DT[i], 0.0)
    end
    return DT
end

"""Volume-constrained `ρ` from a DT field (high DT → solid). Hard 0–1 threshold."""
function density_from_dt(DT::AbstractVector, w, volfrac, frozen; ρmin=1e-3)
    n = length(DT)
    n == length(w) || throw(DimensionMismatch("DT vs w"))
    ρ = fill(ρmin, n)
    @inbounds for i in eachindex(ρ)
        frozen[i] && (ρ[i] = 1.0)
    end
    sw = sum(w)
    sw < 1e-14 && return ρ
    l1, l2 = 0.0, maximum(DT; init=0.0) + 1.0
    @inbounds for _ in 1:40
        λ = 0.5 * (l1 + l2)
        for i in eachindex(ρ)
            frozen[i] && continue
            ρ[i] = DT[i] >= λ ? 1.0 : ρmin
        end
        if dot(ρ, w) / sw > volfrac
            l1 = λ
        else
            l2 = λ
        end
    end
    @inbounds for i in eachindex(ρ)
        frozen[i] && (ρ[i] = 1.0)
    end
    return ρ
end

function _state_objective!(dad::BEMdata{<:Laplace}, ρphys, opt::DibemSimpOptions, p::Real)
    K = simp_conductivity(ρphys, opt, p)
    kfl = 0.05 * opt.K0
    @inbounds for i in eachindex(K)
        K[i] = max(K[i], kfl)
    end
    solve_heterogeneous!(dad, K; rbf=opt.rbf)
    return thermal_conductance(dad)
end

function _state_objective!(dad::BEMdata{<:Elasticity}, ρphys, opt::DibemSimpOptions, p::Real)
    E0 = float(dad.properties.E)
    E = simp_young(ρphys, opt, E0, p)
    solve_heterogeneous!(dad, E; rbf=opt.rbf)
    return elastic_compliance(dad)
end

"""One SIMP iteration: filter → Heaviside → heterogeneous solve → OC. Mutates `ρ`."""
function dibem_simp_step!(dad::BEMdata, ρ::AbstractVector, w, opt::DibemSimpOptions; it::Int=1)
    length(ρ) == dad.nt || throw(DimensionMismatch("ρ length $(length(ρ)) ≠ nt=$(dad.nt)"))
    dad.properties isa Union{Laplace,Elasticity} ||
        error("DIBEM-SIMP supports Laplace and isotropic Elasticity")
    pts = all_points(dad)
    frozen = _freeze_design_nodes(dad)
    p = _simp_cont(it, opt.n_simp, opt.p_start, opt.p)
    β = min(_simp_cont(it, opt.n_simp, opt.β_start, opt.β_end), 5.0)
    ρf = density_filter(pts, ρ, opt.rmin)
    @inbounds for i in eachindex(ρ)
        frozen[i] && (ρf[i] = 1.0)
    end
    ρh = _heaviside(ρf, β)
    @inbounds for i in eachindex(ρ)
        frozen[i] && (ρh[i] = 1.0)
    end
    C = _state_objective!(dad, ρh, opt, p)
    if !isfinite(C) || C <= 0
        V = dot(ρh, w) / sum(w)
        return (; C, V, ρf, ρh, dc=zeros(length(ρ)), p, β)
    end
    dH = _heaviside_deriv(ρf, β)
    dc = dibem_simp_sensitivity(dad, ρh, w, opt, p)
    @inbounds for i in eachindex(dc)
        dc[i] *= dH[i]
    end
    dc = density_filter(pts, dc, opt.rmin)
    sw = sum(w)
    volof = function (ρn)
        ρfn = density_filter(pts, ρn, opt.rmin)
        @inbounds for i in eachindex(ρn)
            frozen[i] && (ρfn[i] = 1.0)
        end
        ρhn = _heaviside(ρfn, β)
        @inbounds for i in eachindex(ρn)
            frozen[i] && (ρhn[i] = 1.0)
        end
        return dot(ρhn, w) / sw
    end
    ρ .= _oc_update(ρ, dc, w, opt.volfrac, frozen; η=opt.η, move=opt.move, ρmin=opt.ρmin,
        volume=volof)
    V = volof(ρ)
    return (; C, V, ρf, ρh, dc, p, β)
end

"""Scatter nodal `ρ` onto a grid; outside the current solid is void (`0`)."""
function _density_grid(d::TopologyDesign, dad::BEMdata, ρ; ngrid::Integer=61)
    pts = Point2D[dad.Nodes; dad.internalNodes]
    xs, ys = _bbox_axes(pts; n=ngrid, pad=0.0)
    Z = interpolate_to_grid(pts, collect(Float64, ρ), xs, ys)
    @inbounds for j in eachindex(ys), i in eachindex(xs)
        p = Point2D(xs[i], ys[j])
        if !in_design(p, d) || !isfinite(Z[i, j])
            Z[i, j] = 0.0
        end
    end
    return xs, ys, Z
end

function _grid_solid_area(d::TopologyDesign, xs, ys, Z, lev)
    nx, ny = length(xs), length(ys)
    (nx < 2 || ny < 2) && return 0.0
    dx = xs[2] - xs[1]
    dy = ys[2] - ys[1]
    cell = abs(dx * dy)
    a = 0.0
    @inbounds for j in 1:ny, i in 1:nx
        Z[i, j] >= lev || continue
        a += cell
    end
    return a
end

"""Bisection: raise `lev` until grid solid area drops to `Atarget`."""
function _ρ_cut_for_volume(d::TopologyDesign, xs, ys, Z, Atarget;
        lo::Float64=0.12, hi::Float64=0.78)
    A_of(lev) = _grid_solid_area(d, xs, ys, Z, lev)
    A_lo = A_of(lo)
    A_hi = A_of(hi)
    Atarget >= A_lo && return lo
    Atarget <= A_hi && return hi
    lev = lo
    @inbounds for _ in 1:24
        mid = 0.5 * (lo + hi)
        if A_of(mid) > Atarget
            lo = mid
        else
            hi = mid
        end
        lev = mid
    end
    return lev
end

function _try_closed_hole!(d, ptsl, Z, xs, ys, ρ_cut, dmin, amin, Ldiag)
    all(p -> in_design(p, d), ptsl[1:max(1, length(ptsl) ÷ 4):end]) || return 0
    a = abs(polygon_area(ptsl))
    a < amin && return 0
    c = sum(ptsl) / length(ptsl)
    in_design(c, d) || return 0
    ic = min(length(xs), max(1, searchsortedfirst(xs, c[1]) - 1))
    jc = min(length(ys), max(1, searchsortedfirst(ys, c[2]) - 1))
    Z[ic, jc] > ρ_cut && return 0
    # large interior voids may run close to Γ; require only a thin wall
    Aouter = design_area(d)
    wall = a > 0.08 * max(Aouter, 1e-12) ? max(0.003 * Ldiag, 0.15 * dmin) : dmin
    _min_dist_to_design(ptsl, d) < wall && return 0
    chaikin_smooth!(ptsl; closed=true, passes=1)
    nel = max(8, Int(round(length(ptsl) / 2)))
    ptsl = resample_polyline(ptsl, nel; closed=true)
    polygon_area(ptsl) > 0 && reverse!(ptsl)
    push!(d.loops, [hole_segment(ptsl, d, nel)])
    return 1
end

"""Split an open iso-curve into interior bays between Neumann landings on Γ."""
function _density_bays(line::Vector{Point2D}, d::TopologyDesign, snap::Float64)
    segs = d.loops[1]
    mapv = _vertex_owners(segs)
    n = length(line)
    near = falses(n)
    @inbounds for k in 1:n
        i, di = _nearest_neumann(line[k], segs, mapv)
        near[k] = i > 0 && di <= snap
    end
    bays = Vector{Vector{Point2D}}()
    k = 1
    while k <= n
        near[k] && (k += 1; continue)
        k1 = k
        while k <= n && !near[k]
            k += 1
        end
        k2 = k - 1
        left = k1 > 1 && near[k1 - 1]
        right = k2 < n && near[k2 + 1]
        if left && right && (k2 - k1 + 1) >= 6
            push!(bays, line[k1-1:k2+1])
        end
    end
    return bays
end

"""Replace one Neumann chain `a → b` by the iso-curve subpath nearest those joints."""
function _iso_subpath(line::Vector{Point2D}, a::Point2D, b::Point2D, maxdist::Float64)
    isempty(line) && return nothing
    i = argmin(k -> norm(line[k] - a), eachindex(line))
    j = argmin(k -> norm(line[k] - b), eachindex(line))
    (norm(line[i] - a) > maxdist || norm(line[j] - b) > maxdist) && return nothing
    i == j && return nothing
    if i < j
        return line[i:j]
    else
        return reverse(line[j:i])
    end
end

"""Pull insulated (Neumann) stretches of the outer loop onto the density iso-curve; Dirichlet patches stay."""
function _apply_iso_to_neumann!(d::TopologyDesign, line::Vector{Point2D}, maxdist::Float64)
    segs = d.loops[1]
    n = length(segs)
    nadd = 0
    newsegs = BoundarySegment[]
    i = 1
    while i <= n
        s = segs[i]
        if _is_fixed_segment(s)
            push!(newsegs, s)
            i += 1
            continue
        end
        # prefer one segment (corners split chains); then a merged Neumann run
        replaced = false
        sub = _iso_subpath(line, s.verts[1], s.verts[end], maxdist)
        if sub !== nothing && length(sub) >= 3
            nel = max(s.n_el, 8)
            sub = resample_polyline(sub, nel + 1; closed=false)
            push!(newsegs, _free_segment(sub, segs, nel))
            nadd += 1
            replaced = true
            i += 1
        end
        if replaced
            continue
        end
        j = i
        while j <= n && !_is_fixed_segment(segs[j])
            j += 1
        end
        a = segs[i].verts[1]
        b = segs[j - 1].verts[end]
        sub = _iso_subpath(line, a, b, maxdist)
        if sub === nothing || length(sub) < 3
            append!(newsegs, segs[i:j-1])
        else
            nel = max(sum(s.n_el for s in segs[i:j-1]), 8)
            sub = resample_polyline(sub, nel + 1; closed=false)
            push!(newsegs, _free_segment(sub, segs, nel))
            nadd += 1
        end
        i = j
    end
    nadd == 0 && return 0
    d.loops[1] = newsegs
    freeze_dirichlet!(d)
    return nadd
end

function _splice_density_bays!(d::TopologyDesign, line::Vector{Point2D}, snap::Float64)
    nadd = 0
    nouter = length(loop_vertices(d.loops[1]))
    for bay in _density_bays(line, d, snap)
        i1, _ = _nearest_neumann(bay[1], d.loops[1], _vertex_owners(d.loops[1]))
        i2, _ = _nearest_neumann(bay[end], d.loops[1], _vertex_owners(d.loops[1]))
        (i1 == 0 || i2 == 0 || i1 == i2) && continue
        path = _arc_indices(i1, i2, nouter)
        path2 = _arc_indices(i2, i1, nouter)
        plen = min(length(path), length(path2))
        plen > 0.4 * nouter && continue          # would replace too much of Γ
        nadd += _splice_open_isocurve!(d, bay, snap)
        nadd > 0 && (nouter = length(loop_vertices(d.loops[1])))
    end
    return nadd
end

"""Insert `ρ = ρ_cut` iso-curves as holes or as bays on insulated outer edges.

If `Atarget` is set, skip an edit that would drop `design_area` below it.
"""
function cut_low_density!(d::TopologyDesign, dad::BEMdata, ρ::AbstractVector;
        ρ_cut::Float64=0.4, opt::PachecoOptions=PachecoOptions(), ngrid::Int=61,
        Atarget::Union{Nothing,Float64}=nothing)
    length(ρ) == dad.nt || throw(DimensionMismatch("ρ vs nt"))
    xs, ys, Z = _density_grid(d, dad, ρ; ngrid=ngrid)
    lines = marching_squares(xs, ys, Z, ρ_cut)
    nadd = 0
    Ldiag = _design_diag(d)
    dmin = opt.min_hole_dist * Ldiag
    amin = opt.min_hole_area
    snap = max(dmin * 8, 0.08 * Ldiag)
    joint = max(snap, 0.35 * Ldiag)
    Amin = Atarget === nothing ? 0.0 : 0.97 * Atarget
    closed = Vector{Vector{Point2D}}()
    openl = Vector{Vector{Point2D}}()
    for line in lines
        length(line) < 5 && continue
        if norm(line[1] - line[end]) < 1e-8 * (1 + Ldiag)
            ptsl = line[1] ≈ line[end] ? line[1:end-1] : copy(line)
            push!(closed, ptsl)
        else
            push!(openl, line)
        end
    end
    sort!(closed; by=p -> abs(polygon_area(p)), rev=true)
    for ptsl in closed
        a_hole = abs(polygon_area(ptsl))
        Amin > 0 && design_area(d) - a_hole < Amin && continue
        d_save = Amin > 0 ? copy_design(d) : d
        nh = _try_closed_hole!(d, ptsl, Z, xs, ys, ρ_cut, dmin, amin, Ldiag)
        if nh == 0
            n = _apply_iso_to_neumann!(d, ptsl, joint)
            if n > 0 && Amin > 0 && design_area(d) < Amin
                d.loops = deepcopy(d_save.loops)
            else
                nadd += n
            end
        elseif Amin > 0 && design_area(d) < Amin
            d.loops = deepcopy(d_save.loops)
        else
            nadd += nh
        end
    end
    sort!(openl; by=length, rev=true)
    for line in openl
        d_save = Amin > 0 ? copy_design(d) : d
        n = _apply_iso_to_neumann!(d, line, joint)
        n == 0 && (n = _splice_density_bays!(d, line, snap))
        if n > 0 && Amin > 0 && design_area(d) < Amin
            d.loops = deepcopy(d_save.loops)
        else
            nadd += n
        end
    end
    return nadd
end

function _cut_with_levels!(d, dad, ρ, opt::DibemSimpOptions)
    pts = all_points(dad)
    ρf = density_filter(pts, ρ, opt.rmin)
    ρh = _heaviside(ρf, opt.β_end)
    xs, ys, Z = _density_grid(d, dad, ρh; ngrid=opt.ngrid)
    A0 = design_area(d)
    Atarget = opt.volfrac * A0
    # stay above Ap; Pacheco polish removes the rest
    Acut = clamp(Atarget * 1.08, Atarget, 0.92 * A0)
    lev = if opt.match_area
        _ρ_cut_for_volume(d, xs, ys, Z, Acut)
    else
        opt.ρ_cut
    end
    lev = clamp(lev, 0.12, 0.70)
    po = PachecoOptions(min_hole_dist=opt.min_hole_dist, min_hole_area=opt.min_hole_area)
    d_before = copy_design(d)
    nadd = cut_low_density!(d, dad, ρh; ρ_cut=lev, opt=po, ngrid=opt.ngrid, Atarget=Atarget)
    if nadd > 0 && design_area(d) < 0.95 * Atarget
        d.loops = deepcopy(d_before.loops)
        nadd = 0
    end
    return nadd, ρh, lev
end

"""Phase 1: nodal SIMP or DT-ρ on a fixed mesh. Mutates `dad` (state / ρ cache)."""
function _dibem_simp_density!(dad::BEMdata, opt::DibemSimpOptions)
    dad.properties isa Union{Laplace,Elasticity} ||
        error("DIBEM-SIMP supports Laplace and isotropic Elasticity")
    has_cache(dad, :H) || H_G_full_direct(dad; npg=opt.npg, threaded=false)
    w = dibem_volume_weights(dad, opt.rbf)
    frozen = _freeze_design_nodes(dad)
    ρ = fill(opt.volfrac, dad.nt)
    @inbounds for i in eachindex(ρ)
        frozen[i] && (ρ[i] = 1.0)
    end
    hist = NamedTuple{(:C, :V, :gray),Tuple{Float64,Float64,Float64}}[]
    C = 0.0
    V = opt.volfrac
    pts = all_points(dad)
    if opt.method === :dt
        solve(dad)
        C = design_objective(dad)
        DT = density_filter(pts, nodal_DT(dad), opt.rmin)
        ρ .= density_from_dt(DT, w, opt.volfrac, frozen; ρmin=opt.ρmin)
        V = dot(ρ, w) / sum(w)
        gray = count(x -> opt.ρmin + 0.05 < x < 0.95, ρ) / length(ρ)
        push!(hist, (C=C, V=V, gray=gray))
        opt.verbose && println("DT-ρ  C=$(round(C; sigdigits=4))  V=$(round(V; digits=3))  gray=$(round(gray; digits=3))")
    else
        for it in 1:opt.n_simp
            st = dibem_simp_step!(dad, ρ, w, opt; it=it)
            C, V = st.C, st.V
            gray = count(x -> opt.ρmin + 0.05 < x < 0.95, ρ) / length(ρ)
            push!(hist, (C=C, V=V, gray=gray))
            opt.verbose && println("DIBEM-SIMP $it  p=$(round(st.p; digits=2))  β=$(round(st.β; digits=2))  C=$(round(C; sigdigits=4))  V=$(round(V; digits=3))  gray=$(round(gray; digits=3))")
        end
    end
    set_cache!(dad; simp_ρ=ρ, simp_C=C, simp_method=opt.method)
    return ρ, hist, C, V
end

"""
    solve_dibem_simp!(dad::BEMdata, opt=DibemSimpOptions()) -> (dad, ρ, hist)

Density SIMP / DT-ρ on a **fixed** mesh (2-D or 3-D). In 3-D, `opt.cut`
extracts a volume-matched iso-surface ([`cut_density_3d!`](@ref)); closed
interior components are inserted as traction-free cavities and the BEM
mesh is rebuilt ([`bemdata_from_iso`](@ref)), like 2-D `bemdata_from_loops`.
Pacheco node motion stays 2-D.
"""
function solve_dibem_simp!(dad::BEMdata, opt::DibemSimpOptions=DibemSimpOptions())
    ρ, hist, C, _ = _dibem_simp_density!(dad, opt)
    lev = opt.ρ_cut
    if opt.cut && dad.dimension == 3
        tris, ρh, lev = cut_density_3d!(dad, ρ, opt)
        ρ = ρh
        opt.verbose && println("  3-D iso lev=$(round(lev; digits=3))  tris=$(length(tris))")
        if !isempty(tris) && degree(dad.element_type) == 1
            dad2, nadd = bemdata_from_iso(dad, tris; ρ=ρh, ρ_cut=lev,
                min_area=opt.min_hole_area, min_dist=opt.min_hole_dist)
            if nadd > 0
                dad = dad2
                H_G_full_direct(dad; npg=opt.npg, threaded=false)
                solve(dad)
                C = design_objective(dad)
                ρ = ones(dad.nt)
                opt.verbose && println("  3-D cavities=$nadd  n=$(dad.n)  J=$(round(C; sigdigits=4))")
            end
        end
    elseif dad.dimension == 3 && opt.verbose && opt.pacheco
        println("  3-D: skipping Pacheco (needs 2-D loops)")
    end
    set_cache!(dad; simp_ρ=ρ, simp_C=C, simp_ρ_cut=lev, simp_method=opt.method)
    return dad, ρ, hist
end

"""
    solve_dibem_simp!(design, opt=DibemSimpOptions()) -> (design, dad, ρ, hist)

Phase 1: nodal SIMP on a fixed outer loop (`H,G` assembled once).
Laplace maximizes conductance; elasticity minimizes compliance.
Phase 2 (if `opt.cut`): iso-cut at a `ρ` level whose grid solid area matches
`volfrac`, then (if `opt.pacheco`) a few Pacheco DT steps to the same `Ap`.
"""
function solve_dibem_simp!(d::TopologyDesign, opt::DibemSimpOptions=DibemSimpOptions())
    d.properties isa Union{Laplace,Elasticity} ||
        error("DIBEM-SIMP supports Laplace and isotropic Elasticity")
    freeze_dirichlet!(d)
    A0 = design_area(d)
    dad = bemdata_from_loops(d)
    ρ, hist, C, _ = _dibem_simp_density!(dad, opt)
    nadd = 0
    lev = opt.ρ_cut
    if opt.cut
        nadd, ρh, lev = _cut_with_levels!(d, dad, ρ, opt)
        opt.verbose && println("  cut lev=$(round(lev; digits=3))  $nadd edit(s)  holes=$(n_holes(d))  area=$(round(design_area(d); digits=3))")
        if nadd > 0
            dad = bemdata_from_loops(d)
            H_G_full_direct(dad; npg=opt.npg, threaded=false)
            solve(dad)
            C = design_objective(dad)
        end
        ρ = ρh
    end
    if opt.pacheco
        Atarget = opt.volfrac * A0
        po = opt.pacheco_opt
        po.verbose = opt.verbose
        po.min_hole_dist = opt.min_hole_dist
        po.min_hole_area = opt.min_hole_area
        po.nucleate_first = false
        po.nucleate_every = typemax(Int)
        if d.properties isa Elasticity
            po.vmax = min(po.vmax, 0.035)
        end
        A = design_area(d)
        opt.verbose && println("  match volume  A=$(round(A; digits=3)) → Ap=$(round(Atarget; digits=3))")
        d, dad, C, A = match_volume!(d, Atarget; opt=po, rtol=po.area_rtol)
        opt.verbose && println("  A=$(round(A; digits=3))  holes=$(n_holes(d))  J=$(round(C; sigdigits=4))")
    end
    set_cache!(dad; simp_ρ=ρ, simp_C=C, simp_ρ_cut=lev, simp_method=opt.method)
    return d, dad, ρ, hist
end

"""
    solve_dt_density!(design, opt=DibemSimpOptions(method=:dt))

Same as [`solve_dibem_simp!`](@ref) but `ρ` is built from the topological
derivative (`DT = k|∇T|²` or plane-stress DT), not SIMP strain energy.
"""
function solve_dt_density!(d::TopologyDesign, opt::DibemSimpOptions=DibemSimpOptions())
    opt.method = :dt
    return solve_dibem_simp!(d, opt)
end
