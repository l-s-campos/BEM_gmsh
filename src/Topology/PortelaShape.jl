# Portela (2012) dual-BEM shape optimal design.
# Traction-free / insulated Γ_d; min compliance (elasticity) or max conductance
# (Laplace) subject to area. No nucleation.
#
# State: mixed CBIE / HBIE on Γ_d (`state=:dual`) or standard CBIE (`:cbie`).
# Velocity: vn ∝ (W − λ) both ways (constant strain-energy free boundary).

export PortelaOptions, solve_portela!, prepare_design_dual!
export strain_energy_density, portela_element_grads

@kwdef mutable struct PortelaOptions
    ΔA::Float64 = 0.0                 # Amax = (1 − ΔA) A0; 0 → reshape at A0
    Amax::Float64 = NaN               # if finite, overrides (1 − ΔA) A0
    vmax::Float64 = 0.04
    maxiter::Int = 40
    npg::Int = 16
    state::Symbol = :dual             # :dual (HBIE on Γ_d) or :cbie
    param::Symbol = :normal           # :normal or :radial
    origin::SVector{2,Float64} = SVector(NaN, NaN)
    α::Float64 = 0.35                 # vn = α (W − λ)
    volume_step::Float64 = 0.20       # fraction of (A − Amax) per iter
    area_rtol::Float64 = 0.04
    d_min::Float64 = 0.01
    verbose::Bool = true
    nsearch::Int = 4                  # inner line-search tries
end

"""
    portela_element_grads(W, ξ, J, w) -> (g0, g1)

Portela (2012) element integrals with linear `vn` on a straight element.
`g0[k] = ∂Ψ0/∂vn_k = −∫ W N_k dΓ`, `g1[k] = ∂Ψ1/∂vn_k = ∫ N_k dΓ`,
`N_1 = (1−ξ)/2`, `N_2 = (1+ξ)/2`. Collocation `ξ`, Jacobian `J` and
Gauss weights `w` are the discontinuous nodes of the state element
(not the paper’s ξ = ±2/3).
"""
function portela_element_grads(W::AbstractVector, ξ::AbstractVector,
        J::AbstractVector, w::AbstractVector)
    length(W) == length(ξ) == length(J) == length(w) ||
        throw(ArgumentError("portela_element_grads: length mismatch"))
    g01 = g02 = g11 = g12 = 0.0
    @inbounds for a in eachindex(W)
        N1 = (1 - ξ[a]) / 2
        N2 = (1 + ξ[a]) / 2
        dΓ = J[a] * w[a]
        g01 -= W[a] * N1 * dΓ
        g02 -= W[a] * N2 * dΓ
        g11 += N1 * dΓ
        g12 += N2 * dΓ
    end
    return (g01, g02), (g11, g12)
end

"""Hoop strain-energy density `W` at every boundary collocation node."""
function strain_energy_density(dad::BEMdata{<:Elasticity})
    σ, _ = boundary_stress_strain(dad)
    E = float(dad.properties.E)
    ν = float(dad.properties.nu)
    Ep = dad.properties.plane_strain ? E / (1 - ν^2) : E
    n = dad.n
    W = zeros(n)
    @inbounds for i in 1:n
        tr = σ[i, 1] + σ[i, 2]
        W[i] = tr * tr / (2 * Ep)
    end
    return W
end

function strain_energy_density(dad::BEMdata{<:Laplace})
    g = boundary_grad_T(dad)
    k = float(dad.properties.k)
    return [0.5 * k * (gi[1]^2 + gi[2]^2) for gi in g]
end

"""
    prepare_design_dual!(dad, design)

Tag collocation on unfrozen zero-Neumann segments as hypersingular
(`eq_type = 3`); all other boundary nodes stay collocation BIE (`eq_type = 1`).
No crack twins (`twin = 0`).
"""
function prepare_design_dual!(dad, d::TopologyDesign)
    n = dad.n
    eq_type = ones(Int, n)
    i = 0
    n_per = d.degree + 1
    for segs in d.loops
        for seg in segs
            length(seg.verts) < 2 && continue
            is_des = !_is_fixed_segment(seg)
            for _ in 1:seg.n_el, _ in 1:n_per
                i += 1
                i > n && break
                is_des && (eq_type[i] = 3)
            end
        end
    end
    set_cache!(dad; eq_type=eq_type, twin=zeros(Int, n))
    return dad
end

_Ψ0(dad::BEMdata{<:Elasticity}) = elastic_compliance(dad)
_Ψ0(dad::BEMdata{<:Laplace}) = -thermal_conductance(dad)

function _assemble_design_dual!(dad, npg::Int)
    B = parentmodule(@__MODULE__)
    if dad.properties isa Elasticity
        B.Crack.assemble_dual_elasticity!(dad; npg=npg, threaded=true)
    else
        B.Crack.assemble_dual_laplace!(dad; npg=npg, threaded=true)
    end
    return dad
end

function _state_ok(dad)
    v = dad.properties isa Elasticity ? dad.u : dad.T
    return all(isfinite, v) && isfinite(_Ψ0(dad))
end

function _solve_portela_state(d::TopologyDesign, opt::PortelaOptions)
    dad = bemdata_from_loops(d; d_min=opt.d_min)
    used = :cbie
    if opt.state === :dual
        prepare_design_dual!(dad, d)
        nd = count(==(3), dad.eq_type)
        if nd > 0
            try
                _assemble_design_dual!(dad, opt.npg)
                solve(dad)
                if _state_ok(dad)
                    used = :dual
                else
                    error("nonfinite dual state")
                end
            catch err
                opt.verbose && println("  dual fallback ($(typeof(err))): CBIE")
                dad = bemdata_from_loops(d; d_min=opt.d_min)
                H_G_full_direct(dad; npg=opt.npg, threaded=true)
                solve(dad)
                used = :cbie
            end
        else
            H_G_full_direct(dad; npg=opt.npg, threaded=true)
            solve(dad)
        end
    elseif opt.state === :cbie
        H_G_full_direct(dad; npg=opt.npg, threaded=true)
        solve(dad)
    else
        throw(ArgumentError("PortelaOptions.state must be :dual or :cbie, got $(opt.state)"))
    end
    return dad, used
end

function _origin(d::TopologyDesign, opt::PortelaOptions)
    o = opt.origin
    all(isfinite, o) && return Point2D(o[1], o[2])
    if length(d.loops) ≥ 2
        v = loop_vertices(d.loops[2])
        isempty(v) || return sum(v) / length(v)
    end
    pts = Point2D[]
    for segs in d.loops, s in segs
        _is_fixed_segment(s) && continue
        append!(pts, s.verts)
    end
    isempty(pts) && return Point2D(0.0, 0.0)
    return sum(pts) / length(pts)
end

"""Weighted ∫ W dΓ and length of Γ_d (unfrozen zero-Neumann vertices)."""
function _design_WL(d::TopologyDesign, dad, W::AbstractVector)
    tree = KDTree(reduce(hcat, collect(Point2D, dad.Nodes)))
    intW = 0.0
    L = 0.0
    nW = length(W)
    for segs in d.loops
        loopv = loop_vertices(segs)
        nv = length(loopv)
        nv < 2 && continue
        own = _vertex_owners(segs)
        for i in 1:nv
            i > length(own) && break
            own[i].frozen && continue
            own[i].fixed && continue
            im = i == 1 ? nv : i - 1
            ip = i == nv ? 1 : i + 1
            ℓ = 0.5 * (norm(loopv[i] - loopv[im]) + norm(loopv[ip] - loopv[i]))
            idxs, _ = knn(tree, loopv[i], 1)
            iw = idxs[1]
            iw > nW && continue
            intW += W[iw] * ℓ
            L += ℓ
        end
    end
    return intW, L
end

"""Like `_inward_step_limit`, but collinear verts may leave the neighbour chord
(that is the Portela design motion on a straight quadratic element)."""
function _shape_step_limit(pprev, p, pnext, dir)
    Ld = norm(dir)
    Ld < 1e-16 && return 0.0
    e0 = norm(p - pprev)
    e1 = norm(pnext - p)
    lim = 0.35 * min(e0, e1)
    ab = pnext - pprev
    L2 = dot(ab, ab)
    L2 < 1e-30 && return lim
    nχ = SVector(-ab[2], ab[1])
    nL = norm(nχ)
    nL < 1e-16 && return lim
    nχ /= nL
    dist = abs(dot(p - pprev, nχ))
    dist < 1e-10 && return lim
    toward = -dot(dir / Ld, nχ)
    if toward > 0.2
        lim = min(lim, 0.40 * dist / toward)
    end
    return max(lim, 0.0)
end

function _move_portela!(d::TopologyDesign, dad, W, λ, α, vmax, param, origin, dA::Real=0.0;
        inward::Bool=false)
    snap = _snapshot_verts(d)
    tree = KDTree(reduce(hcat, collect(Point2D, dad.Nodes)))
    nW = length(W)
    props = NamedTuple{(:seg, :i, :n̂, :r̂, :cosn, :ℓ, :lim, :vn0),
        Tuple{BoundarySegment,Int,Point2D,Point2D,Float64,Float64,Float64,Float64}}[]
    for segs in d.loops
        loopv = loop_vertices(segs)
        nrm = vertex_outward_normals(loopv; closed=true)
        nv = length(loopv)
        nv < 2 && continue
        for seg in segs
            _is_fixed_segment(seg) && continue
            n = length(seg.verts)
            n < 2 && continue
            owners = _closest_indices(seg.verts, loopv)
            for i in 1:n
                seg.frozen[i] && continue
                p = seg.verts[i]
                idxs, _ = knn(tree, p, 1)
                iw = idxs[1]
                iw > nW && continue
                io = owners[i]
                n̂ = nrm[io]
                Ln = norm(n̂)
                Ln < 1e-16 && continue
                n̂ /= Ln
                iprev = io == 1 ? nv : io - 1
                inext = io == nv ? 1 : io + 1
                ℓ = 0.5 * (norm(loopv[io] - loopv[iprev]) + norm(loopv[inext] - loopv[io]))
                vn0 = α * (W[iw] - λ)
                inward && (vn0 = min(vn0, 0.0))
                r̂ = n̂
                cosn = 1.0
                if param === :radial
                    r = p - origin
                    b = norm(r)
                    b < 1e-14 && continue
                    r̂ = r / b
                    cosn = dot(r̂, n̂)
                    abs(cosn) < 1e-8 && continue
                end
                dir = vn0 ≥ 0 ? n̂ : -n̂
                lim = _shape_step_limit(loopv[iprev], p, loopv[inext], dir)
                push!(props, (seg=seg, i=i, n̂=n̂, r̂=r̂, cosn=cosn, ℓ=ℓ, lim=lim, vn0=vn0))
            end
        end
    end
    isempty(props) && return false, 0
    L = sum(q.ℓ for q in props)
    L < 1e-16 && return false, 0
    clip(vn, lim) = clamp(clamp(vn, -vmax, vmax), -lim, lim)
    vns = [clip(q.vn0, q.lim) for q in props]
    dA_clip = sum(vns[k] * props[k].ℓ for k in eachindex(props))
    c = (float(dA) - dA_clip) / L
    for k in eachindex(props)
        v = clip(vns[k] + c, props[k].lim)
        inward && (v = min(v, 0.0))
        vns[k] = v
    end
    moved = 0
    for (k, q) in enumerate(props)
        vn = vns[k]
        abs(vn) < 1e-16 && continue
        p = q.seg.verts[q.i]
        if param === :radial
            b = norm(p - origin)
            db = vn / q.cosn
            bnew = max(b + db, 1e-3)
            q.seg.verts[q.i] = origin + bnew * q.r̂
        else
            q.seg.verts[q.i] = p + vn * q.n̂
        end
        moved += 1
    end
    if _design_folded(d)
        _restore_verts!(d, snap)
        return false, 0
    end
    return true, moved
end

"""
    solve_portela!(design, opt=PortelaOptions(); history=true) -> (design, dad, hist)

Portela loop: BEM (dual or CBIE) → hoop energy `W` on Γ_d → `vn ∝ (W − λ)`
with λ from the area constraint. No hole nucleation. Elasticity minimizes
compliance; Laplace maximizes thermal conductance.
"""
function solve_portela!(d::TopologyDesign, opt::PortelaOptions=PortelaOptions();
        history::Bool=true)
    freeze_dirichlet!(d)
    _densify_neumann!(d)
    dad, used = _solve_portela_state(d, opt)
    W = strain_energy_density(dad)
    J = design_objective(dad)
    A = design_area(d)
    A0 = A
    Amax = isfinite(opt.Amax) ? opt.Amax : (1 - opt.ΔA) * A0
    hist = TopologyHistory()
    history && push!(hist; area=A, J=J, maxDT=maximum(W; init=0.0),
        n_holes=n_holes(d), design=d)
    opt.verbose && println("Portela($(used)) iter 0  A=$(round(A; digits=4))  J=$(_fmt_obj(J))  holes=$(n_holes(d))")

    origin = _origin(d, opt)
    α = opt.α
    vmax = opt.vmax
    Ψ = _Ψ0(dad)
    Ψ_init = Ψ
    iter = 0
    while iter < opt.maxiter
        iter += 1
        intW, L = _design_WL(d, dad, W)
        L < 1e-16 && break
        meanW = intW / L
        if A > Amax * (1 + opt.area_rtol)
            Atarget = max(Amax, A - opt.volume_step * (A - Amax))
        elseif A < Amax * (1 - opt.area_rtol)
            Atarget = min(Amax, A + opt.volume_step * (Amax - A))
        else
            Atarget = Amax
        end
        λ = meanW - (Atarget - A) / max(α * L, 1e-16)
        accepted = false
        αtry, vtry = α, vmax
        d_prev = copy_design(d)
        local dad_try = dad
        local W_try = W
        local J_try = J
        local A_try = A
        local Ψ_try = Ψ
        local used_try = used
        for _ in 1:max(opt.nsearch, 1)
            d.loops = deepcopy(d_prev.loops)
            ok, nmov = _move_portela!(d, dad, W, λ, αtry, vtry, opt.param, origin,
                Atarget - A)
            if !ok || nmov == 0
                αtry *= 0.5
                vtry *= 0.5
                continue
            end
            freeze_dirichlet!(d)
            dad_try, used_try = _solve_portela_state(d, opt)
            if !_state_ok(dad_try) || _design_folded(d)
                d.loops = deepcopy(d_prev.loops)
                αtry *= 0.5
                vtry *= 0.5
                continue
            end
            J_try = design_objective(dad_try)
            A_try = design_area(d)
            Ψ_try = _Ψ0(dad_try)
            area_ok = A_try > 0.50 * max(Amax, 1e-12)
            area_viol = A > Amax * (1 + opt.area_rtol)
            Ψ_ok = Ψ_try <= Ψ + 1e-4 * max(abs(Ψ_init), 1e-12) &&
                Ψ_try <= Ψ_init + 1e-3 * max(abs(Ψ_init), 1e-12)
            ok_step = isfinite(J_try) && area_ok &&
                (area_viol ? (A_try < A) :
                    (Ψ_ok && A_try <= Amax * (1 + opt.area_rtol)))
            if !ok_step
                d.loops = deepcopy(d_prev.loops)
                αtry *= 0.5
                vtry *= 0.5
                continue
            end
            accepted = true
            W_try = strain_energy_density(dad_try)
            break
        end
        if !accepted
            d.loops = deepcopy(d_prev.loops)
            opt.verbose && println("  stop: no feasible Portela step")
            break
        end
        dad, W, J, A, Ψ, used = dad_try, W_try, J_try, A_try, Ψ_try, used_try
        α = min(opt.α, 1.15 * αtry)
        vmax = min(opt.vmax, 1.15 * vtry)
        history && push!(hist; area=A, J=J, maxDT=maximum(W; init=0.0),
            n_holes=n_holes(d), design=d)
        opt.verbose && println("Portela($(used)) iter $iter  A=$(round(A; digits=4))  J=$(_fmt_obj(J))  holes=$(n_holes(d))")
        abs(A - Amax) / max(Amax, 1e-12) ≤ opt.area_rtol && iter ≥ 2 &&
            abs(hist.J[end] - hist.J[max(end - 1, 1)]) ≤ 1e-4 * max(abs(J), 1e-12) && break
    end
    return d, dad, hist
end
