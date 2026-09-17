# JuMP first-order shape step on Γ_d (same dofs as the DT stand-in).
# :linear — linearized MMFD LP (JuMP + HiGHS): min −s·ℓ∘vn  s.t. ℓ·vn ≤ ΔA.
# :mma    — nonlinear MMA (JuMP + NLopt LD_MMA) with BEM in f and area.

using JuMP
using HiGHS
import NLopt

export move_boundary_jump!

struct MotionDof
    iloop::Int
    iseg::Int
    ivert::Int
    n̂::Point2D
    ℓ::Float64
    lim::Float64
    s::Float64
end

function _motion_dofs(d::TopologyDesign, dad, sfield::AbstractVector)
    tree = KDTree(reduce(hcat, collect(Point2D, dad.Nodes)))
    nS = length(sfield)
    dofs = MotionDof[]
    for (iloop, segs) in enumerate(d.loops)
        loopv = loop_vertices(segs)
        nrm = vertex_outward_normals(loopv; closed=true)
        nv = length(loopv)
        nv < 2 && continue
        for (iseg, seg) in enumerate(segs)
            _is_fixed_segment(seg) && continue
            n = length(seg.verts)
            n < 2 && continue
            owners = _closest_indices(seg.verts, loopv)
            for i in 1:n
                seg.frozen[i] && continue
                p = seg.verts[i]
                idxs, _ = knn(tree, p, 1)
                iw = idxs[1]
                iw > nS && continue
                io = owners[i]
                n̂ = nrm[io]
                Ln = norm(n̂)
                Ln < 1e-16 && continue
                n̂ /= Ln
                iprev = io == 1 ? nv : io - 1
                inext = io == nv ? 1 : io + 1
                ℓ = 0.5 * (norm(loopv[io] - loopv[iprev]) +
                           norm(loopv[inext] - loopv[io]))
                lim = max(
                    _shape_step_limit(loopv[iprev], p, loopv[inext], n̂),
                    _shape_step_limit(loopv[iprev], p, loopv[inext], -n̂),
                )
                push!(dofs, MotionDof(iloop, iseg, i, n̂, ℓ, lim, sfield[iw]))
            end
        end
    end
    return dofs
end

function _apply_vn!(d::TopologyDesign, dofs::Vector{MotionDof}, vn::AbstractVector)
    snap = _snapshot_verts(d)
    @inbounds for k in eachindex(dofs)
        q = dofs[k]
        abs(vn[k]) < 1e-16 && continue
        d.loops[q.iloop][q.iseg].verts[q.ivert] += vn[k] * q.n̂
    end
    if _design_folded(d)
        _restore_verts!(d, snap)
        return false
    end
    return true
end

function _jump_linear_vn(dofs::Vector{MotionDof}, dA::Real, vmax::Real, inward::Bool)
    n = length(dofs)
    n == 0 && return Float64[]
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, vn[i=1:n])
    for i in 1:n
        set_lower_bound(vn[i], max(-vmax, -dofs[i].lim))
        set_upper_bound(vn[i], inward ? 0.0 : min(vmax, dofs[i].lim))
    end
    # linearized Ψ0: δΨ0 = −∫ s vn dΓ  →  min −s·ℓ∘vn
    @objective(model, Min, sum(-dofs[i].s * dofs[i].ℓ * vn[i] for i in 1:n))
    @constraint(model, sum(dofs[i].ℓ * vn[i] for i in 1:n) <= dA)
    optimize!(model)
    has_values(model) && return value.(vn)
    return zeros(n)
end

function _jump_mma_vn(d0::TopologyDesign, dofs::Vector{MotionDof}, Atarget::Real,
        opt::PachecoOptions, vmax::Real, inward::Bool)
    n = length(dofs)
    n == 0 && return zeros(0)
    cache = Dict{UInt64,NamedTuple{(:Ψ, :A, :gΨ, :gA),
        Tuple{Float64,Float64,Vector{Float64},Vector{Float64}}}}()
    function _eval(z::AbstractVector{<:Real})
        key = hash(round.(Float64.(z); digits=10))
        haskey(cache, key) && return cache[key]
        vn = collect(Float64, z)
        d = copy_design(d0)
        if !_apply_vn!(d, dofs, vn)
            rec = (Ψ=1e6, A=1e6, gΨ=zeros(n), gA=Float64[q.ℓ for q in dofs])
            cache[key] = rec
            return rec
        end
        dad = bemdata_from_loops(d; d_min=opt.d_min)
        H_G_full_direct(dad; npg=opt.npg, threaded=true)
        solve(dad)
        if !_state_ok(dad)
            rec = (Ψ=1e6, A=1e6, gΨ=zeros(n), gA=Float64[q.ℓ for q in dofs])
            cache[key] = rec
            return rec
        end
        DTb, _ = topological_derivative(dad)
        tree = KDTree(reduce(hcat, collect(Point2D, dad.Nodes)))
        gΨ = zeros(n)
        gA = zeros(n)
        @inbounds for i in 1:n
            q = dofs[i]
            p = d.loops[q.iloop][q.iseg].verts[q.ivert]
            idxs, _ = knn(tree, p, 1)
            s = DTb[idxs[1]]
            gΨ[i] = -s * q.ℓ
            gA[i] = q.ℓ
        end
        rec = (Ψ=_Ψ0(dad), A=design_area(d), gΨ=gΨ, gA=gA)
        cache[key] = rec
        return rec
    end
    obj_f(z...) = _eval(collect(Float64, z)).Ψ
    function ∇obj(g, z...)
        g .= _eval(collect(Float64, z)).gΨ
        return
    end
    area_f(z...) = _eval(collect(Float64, z)).A
    function ∇area(g, z...)
        g .= _eval(collect(Float64, z)).gA
        return
    end
    model = Model(NLopt.Optimizer)
    set_attribute(model, "algorithm", :LD_MMA)
    set_attribute(model, "maxeval", opt.jump_maxeval)
    set_attribute(model, "xtol_rel", 1e-3)
    set_silent(model)
    @variable(model, vn[i=1:n], start = 0.0)
    for i in 1:n
        set_lower_bound(vn[i], max(-vmax, -dofs[i].lim))
        set_upper_bound(vn[i], inward ? 0.0 : min(vmax, dofs[i].lim))
    end
    @operator(model, op_obj, n, obj_f, ∇obj)
    @operator(model, op_area, n, area_f, ∇area)
    @objective(model, Min, op_obj(vn...))
    @constraint(model, op_area(vn...) <= Atarget)
    optimize!(model)
    has_values(model) && return value.(vn)
    return zeros(n)
end

"""
    move_boundary_jump!(d, dad, DTb, opt; Atarget) -> (d, nmoved)

JuMP first-order step on `DT` (or any nodal field `DTb`). `opt.jump_mode`
is `:linear` (HiGHS LP on the linearized MMFD subproblem) or `:mma`
(NLopt `LD_MMA` with BEM evaluations). Same free-vertex dofs as
[`move_boundary_standin!`](@ref).
"""
function move_boundary_jump!(d::TopologyDesign, dad, DTb, opt::PachecoOptions;
        Atarget::Real, vmax::Real=opt.vmax)
    freeze_dirichlet!(d)
    _densify_neumann!(d)
    dofs = _motion_dofs(d, dad, DTb)
    isempty(dofs) && return d, 0
    A = design_area(d)
    dA = Atarget - A
    mode = opt.jump_mode
    vn = if mode === :linear
        _jump_linear_vn(dofs, dA, vmax, opt.standin_inward)
    elseif mode === :mma
        _jump_mma_vn(d, dofs, Atarget, opt, vmax, opt.standin_inward)
    else
        throw(ArgumentError("PachecoOptions.jump_mode must be :linear or :mma, got $mode"))
    end
    ok = _apply_vn!(d, dofs, vn)
    ok || return d, 0
    freeze_dirichlet!(d)
    return d, count(v -> abs(v) > 1e-16, vn)
end
