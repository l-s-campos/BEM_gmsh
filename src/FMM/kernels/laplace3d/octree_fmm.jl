# Cubic octree FMM, same tree as H² (`hmatrix_splitter` + `cube=true`).
# 2D keeps Flatiron FMM2D log-multipoles. 3D uses spherical-harmonic
# P2M/M2M/M2L/L2L/L2P (`l3dterms`) and dual-tree lists on that octree
# (FMM3D U/V lists; hanging-node neighbors stay P2P).

"""Cached 3D Laplace FMM on a cubic octree (H² / FMM3D lists)."""
mutable struct Laplace3DFMMPlan
    n::Int
    p::Int
    stree::ClusterTree{3,Float64}
    sl2g::Vector{Int}
    src::Vector{SVector{3,Float64}}
    sdata::Vector{L3Exp}
    eps::Float64
    nmax::Int
    η::Float64
    full_fmm::Bool
    dens_loc::Vector{Float64}
    pot_local::Vector{Float64}
    grad_local::Matrix{Float64}
    Ptab::Matrix{Float64}
    p2p_jobs::Vector{Tuple{ClusterTree{3,Float64},ClusterTree{3,Float64}}}
    m2l_jobs::Vector{Tuple{ClusterTree{3,Float64},ClusterTree{3,Float64}}}
    trans::Laplace3DTransWS
    trans_tls::Vector{Laplace3DTransWS}
    src_x::Vector{Float64}
    src_y::Vector{Float64}
    src_z::Vector{Float64}
    m2l_groups::Vector{Tuple{ClusterTree{3,Float64},Vector{ClusterTree{3,Float64}}}}
    p2p_groups::Vector{Tuple{ClusterTree{3,Float64},Vector{ClusterTree{3,Float64}}}}
    leaf_nodes::Vector{ClusterTree{3,Float64}}
    level_nodes::Vector{Vector{ClusterTree{3,Float64}}}
    m2l_T::Dict{NTuple{4,Int},Matrix{Float64}}
    deg::Vector{Int}
    pw::Laplace3DPWQuad
    pw_il::Vector{Vector{Tuple{ClusterTree{3,Float64},Vector{ClusterTree{3,Float64}}}}}
    es_jobs::Vector{Tuple{ClusterTree{3,Float64},ClusterTree{3,Float64}}}
    es_groups::Vector{Tuple{ClusterTree{3,Float64},Vector{ClusterTree{3,Float64}}}}
    pw_mexp::Matrix{ComplexF64}
    pw_ws_tls::Vector{Laplace3DPWWS}
end

function _p2p_plan!(plan::Laplace3DFMMPlan, ch, g_loc)
    nt = Threads.nthreads()
    threaded = nt > 1 && length(plan.p2p_groups) >= 8
    xs, ys, zs = plan.src_x, plan.src_y, plan.src_z
    if threaded
        Threads.@threads :static for gi in eachindex(plan.p2p_groups)
            tnode, snodes = plan.p2p_groups[gi]
            @inbounds for snode in snodes
                _p2p_laplace3d_soa!(plan.pot_local, xs, ys, zs, ch, tnode, snode, g_loc)
            end
        end
    else
        @inbounds for (tnode, snode) in plan.p2p_jobs
            _p2p_laplace3d_soa!(plan.pot_local, xs, ys, zs, ch, tnode, snode, g_loc)
        end
    end
    return nothing
end

function _ensure_trans_tls!(plan::Laplace3DFMMPlan)
    nt = _fmm_tls_len()
    tls = plan.trans_tls
    if length(tls) < nt
        old = length(tls)
        resize!(plan.trans_tls, nt)
        old == 0 && (plan.trans_tls[1] = plan.trans)
        @inbounds for i in max(old + 1, 2):nt
            plan.trans_tls[i] = Laplace3DTransWS(plan.p)
        end
    end
    return plan.trans_tls
end

function _m2l_groups_chunk_3d!(sdata, groups, p, ws, tid, nt, n)
    @inbounds for gi in tid:nt:n
        tnode, snodes = groups[gi]
        td = _l3get(sdata, tnode)
        for snode in snodes
            sd = _l3get(sdata, snode)
            _m2l_laplace3d!(td.localexp, td.center, sd.multipole, sd.center, sd.R, p, ws)
        end
    end
    return nothing
end

function _ensure_pw_tls!(plan::Laplace3DFMMPlan)
    nt = _fmm_tls_len()
    tls = plan.pw_ws_tls
    if length(tls) < nt
        old = length(tls)
        resize!(plan.pw_ws_tls, nt)
        nλ = length(plan.pw.λ)
        @inbounds for i in (old + 1):nt
            plan.pw_ws_tls[i] = Laplace3DPWWS(plan.p, plan.pw.nexp, nλ)
        end
    end
    return plan.pw_ws_tls
end

function _m2l_pw_dir!(plan::Laplace3DFMMPlan, dir::Int)
    groups = plan.pw_il[dir]
    isempty(groups) && return nothing
    p = plan.p
    quad = plan.pw
    sdata = plan.sdata
    mexp = plan.pw_mexp
    seen = fill(false, size(mexp, 2))
    srcs = ClusterTree{3,Float64}[]
    @inbounds for (_, snodes) in groups
        for snode in snodes
            id = node_id(snode)
            if !seen[id]
                seen[id] = true
                push!(srcs, snode)
            end
        end
    end
    nthr = Threads.nthreads()
    wss = _ensure_pw_tls!(plan)
    ns = length(srcs)
    if nthr > 1 && ns >= 8
        nt = min(nthr, ns)
        @sync for tid in 1:nt
            let tid = tid, ws = wss[tid]
                Threads.@spawn _pw_m2x_chunk!(mexp, sdata, srcs, dir, p, ws, quad, tid, nt, ns)
            end
        end
    else
        _pw_m2x_chunk!(mexp, sdata, srcs, dir, p, wss[1], quad, 1, 1, ns)
    end
    ng = length(groups)
    if nthr > 1 && ng >= 8
        nt = min(nthr, ng)
        @sync for tid in 1:nt
            let tid = tid, ws = wss[tid]
                Threads.@spawn _pw_x2l_chunk!(sdata, groups, mexp, dir, p, ws, quad, tid, nt, ng)
            end
        end
    else
        _pw_x2l_chunk!(sdata, groups, mexp, dir, p, wss[1], quad, 1, 1, ng)
    end
    return nothing
end

function _pw_m2x_chunk!(mexp, sdata, srcs, dir, p, ws, quad, tid, nt, ns)
    @inbounds for si in tid:nt:ns
        snode = srcs[si]
        sd = _l3get(sdata, snode)
        a = _box_side(snode)
        _pw_m2x!(view(mexp, :, node_id(snode)), sd.multipole, a, dir, p, ws, quad)
    end
    return nothing
end

function _pw_x2l_chunk!(sdata, groups, mexp, dir, p, ws, quad, tid, nt, ng)
    acc = ws.acc
    @inbounds for gi in tid:nt:ng
        tnode, snodes = groups[gi]
        td = _l3get(sdata, tnode)
        fill!(acc, 0)
        a = _box_side(tnode)
        inva = 1 / a
        tctr = td.center
        for snode in snodes
            Δ = tctr - _l3get(sdata, snode).center
            ix = round(Int, Δ[1] * inva)
            iy = round(Int, Δ[2] * inva)
            iz = round(Int, Δ[3] * inva)
            _pw_shift_add!(acc, view(mexp, :, node_id(snode)), ix, iy, iz, dir, quad)
        end
        _pw_x2l!(td.localexp, a, dir, p, acc, ws, quad)
    end
    return nothing
end

function _m2l_octree!(plan::Laplace3DFMMPlan)
    p = plan.p
    groups = plan.es_groups
    n = length(groups)
    nthr = Threads.nthreads()
    if nthr > 1 && n >= 8
        tls = _ensure_trans_tls!(plan)
        nt = min(nthr, n)
        sdata = plan.sdata
        @sync for tid in 1:nt
            let tid = tid, ws = tls[tid]
                Threads.@spawn _m2l_groups_chunk_3d!(sdata, groups, p, ws, tid, nt, n)
            end
        end
    elseif !isempty(plan.es_jobs)
        ws = plan.trans
        @inbounds for (tnode, snode) in plan.es_jobs
            td = _l3get(plan.sdata, tnode)
            sd = _l3get(plan.sdata, snode)
            _m2l_laplace3d!(td.localexp, td.center, sd.multipole, sd.center, sd.R, p, ws)
        end
    end
    @inbounds for dir in 1:6
        _m2l_pw_dir!(plan, dir)
    end
    return nothing
end

function _p2m_chunk!(sdata, src, ch, p, leaves, ws, tid, nt, n)
    Ptab = ws.Ptab
    @inbounds for i in tid:nt:n
        node = leaves[i]
        ed = _l3get(sdata, node)
        form_mpole3d!(ed.multipole, ed.center, src, ch, nothing, index_range(node), p;
            reset=true, Ptab=Ptab)
    end
    return nothing
end

function _m2m_chunk!(sdata, parents, p, ws, tid, nt, n)
    @inbounds for i in tid:nt:n
        node = parents[i]
        isempty(children(node)) && continue
        ed = _l3get(sdata, node)
        fill!(ed.multipole, 0)
        Ptab = ws.Ptab
        for c in children(node)
            cd = _l3get(sdata, c)
            R = max(cd.R * 1.05, 1e-12)
            _equiv_charges!(ws.pts, ws.q, ws.b, cd.multipole, R, p)
            @inbounds for j in eachindex(ws.pts)
                ws.pts[j] = cd.center + ws.pts[j]
            end
            form_mpole3d!(ed.multipole, ed.center, ws.pts, ws.q, nothing,
                eachindex(ws.pts), p; reset=false, Ptab=Ptab)
        end
    end
    return nothing
end

function _l2l_chunk!(sdata, kids, p, ws, tid, nt, n)
    @inbounds for i in tid:nt:n
        c = kids[i]
        par = parentnode(c)
        cd = _l3get(sdata, c)
        ed = _l3get(sdata, par)
        _l2l_laplace3d!(cd.localexp, cd.center, ed.localexp, ed.center, cd.R, p, ws)
    end
    return nothing
end

function _l2p_chunk!(sdata, src, pot, leaves, p, ws, tid, nt, n)
    Ptab = ws.Ptab
    @inbounds for i in tid:nt:n
        node = leaves[i]
        ed = _l3get(sdata, node)
        L = ed.localexp
        ctr = ed.center
        for j in index_range(node)
            pot[j] += INV4PI * _local_series(src[j] - ctr, L, p, Ptab)
        end
    end
    return nothing
end

function _l2p_grad_chunk!(sdata, src, g_loc, leaves, p, ws, tid, nt, n)
    Ptab = ws.Ptab
    δ = 1e-7
    e1, e2, e3 = SVector(1.0, 0.0, 0.0), SVector(0.0, 1.0, 0.0), SVector(0.0, 0.0, 1.0)
    @inbounds for i in tid:nt:n
        node = leaves[i]
        ed = _l3get(sdata, node)
        L = ed.localexp
        ctr = ed.center
        for j in index_range(node)
            d = src[j] - ctr
            gx = (_local_series(d + δ * e1, L, p, Ptab) - _local_series(d - δ * e1, L, p, Ptab)) / (2δ)
            gy = (_local_series(d + δ * e2, L, p, Ptab) - _local_series(d - δ * e2, L, p, Ptab)) / (2δ)
            gz = (_local_series(d + δ * e3, L, p, Ptab) - _local_series(d - δ * e3, L, p, Ptab)) / (2δ)
            g_loc[1, j] += gx * INV4PI
            g_loc[2, j] += gy * INV4PI
            g_loc[3, j] += gz * INV4PI
        end
    end
    return nothing
end

function _spawn_trans_chunks!(plan::Laplace3DFMMPlan, n, helper, args...)
    n <= 0 && return nothing
    nthr = Threads.nthreads()
    if nthr > 1 && n >= 8
        tls = _ensure_trans_tls!(plan)
        nt = min(nthr, n)
        @sync for tid in 1:nt
            let tid = tid, ws = tls[tid]
                Threads.@spawn helper(args..., ws, tid, nt, n)
            end
        end
    else
        helper(args..., plan.trans, 1, 1, n)
    end
    return nothing
end

function _upward_octree!(plan::Laplace3DFMMPlan, ch)
    p = plan.p
    sdata = plan.sdata
    src = plan.src
    leaves = plan.leaf_nodes
    _spawn_trans_chunks!(plan, length(leaves), _p2m_chunk!, sdata, src, ch, p, leaves)
    levels = plan.level_nodes
    @inbounds for lev in length(levels):-1:2
        parents = levels[lev - 1]
        _spawn_trans_chunks!(plan, length(parents), _m2m_chunk!, sdata, parents, p)
    end
    return nothing
end

function _downward_octree!(plan::Laplace3DFMMPlan)
    p = plan.p
    sdata = plan.sdata
    levels = plan.level_nodes
    @inbounds for lev in 1:(length(levels) - 1)
        kids = levels[lev + 1]
        _spawn_trans_chunks!(plan, length(kids), _l2l_chunk!, sdata, kids, p)
    end
    return nothing
end

function _l2p_octree!(plan::Laplace3DFMMPlan, g_loc)
    _downward_octree!(plan)
    p = plan.p
    src = plan.src
    leaves = plan.leaf_nodes
    _spawn_trans_chunks!(plan, length(leaves), _l2p_chunk!,
        plan.sdata, src, plan.pot_local, leaves, p)
    g_loc === nothing && return nothing
    _spawn_trans_chunks!(plan, length(leaves), _l2p_grad_chunk!,
        plan.sdata, src, g_loc, leaves, p)
    return nothing
end

"""
    apply_laplace3d!(plan, pot; charges, grad=nothing)

In-place source→source 3D Laplace FMM using a cached [`Laplace3DFMMPlan`](@ref).
`pot` is overwritten in the **global** (input) ordering of `plan.sl2g`.
If `grad` is a `3×n` matrix, Cartesian `∇φ` is written in the same ordering.
"""
function apply_laplace3d!(plan::Laplace3DFMMPlan, pot::AbstractVector{<:Real};
        charges, grad=nothing)
    n = plan.n
    length(pot) == n || throw(DimensionMismatch("pot length $(length(pot)) ≠ $n"))
    length(charges) == n || throw(DimensionMismatch("charges"))
    want_g = grad !== nothing
    if want_g
        size(grad, 1) == 3 && size(grad, 2) == n ||
            throw(DimensionMismatch("grad must be 3×$n"))
    end
    _permute_to_local!(plan.dens_loc, charges, plan.sl2g)
    ch = plan.dens_loc
    fill!(plan.pot_local, 0)
    g_loc = nothing
    if want_g
        fill!(plan.grad_local, 0)
        g_loc = plan.grad_local
    end
    for ed in plan.sdata
        fill!(ed.multipole, 0)
        fill!(ed.localexp, 0)
    end
    _upward_octree!(plan, ch)
    _m2l_octree!(plan)
    _l2p_octree!(plan, g_loc)
    _p2p_plan!(plan, ch, g_loc)
    _unpermute_from_local!(pot, plan.pot_local, plan.sl2g)
    if want_g
        @inbounds for i in 1:n
            g = plan.sl2g[i]
            grad[1, g] = plan.grad_local[1, i]
            grad[2, g] = plan.grad_local[2, i]
            grad[3, g] = plan.grad_local[3, i]
        end
    end
    return pot
end

"""
    build_laplace3d_plan(sources; eps=1e-8, nmax=-1, η=1.0, p=nothing)

Cache a cubic octree, spherical-harmonic expansions, and dual-tree interaction
lists for repeated [`apply_laplace3d!`](@ref). Same-size lattice M2L uses
FMM3D plane-wave translations; mixed-size pairs stay equivalent-sphere.

`nmax < 0` uses `laplace3d_ndiv(eps)` (`200` at `1e-8`, Flatiron `lndiv`). `p=nothing` uses
`laplace3d_nterms(eps)` (Flatiron `l3dterms`, capped at 16). Pass `p=` to
lower the expansion order at the call site.
"""
function build_laplace3d_plan(
    sources::AbstractMatrix{<:Real};
    eps::Float64=1e-8,
    nmax::Int=-1,
    η::Float64=1.0,
    full_fmm::Bool=true,
    tree=nothing,
    p::Union{Nothing,Int}=nothing,
)
    size(sources, 1) == 3 || throw(ArgumentError("sources must be 3×n"))
    ns = size(sources, 2)
    nmax = nmax < 0 ? laplace3d_ndiv(eps) : nmax
    p = p === nothing ? laplace3d_nterms(eps) : Int(p)
    if tree !== nothing
        stree = tree
        sl2g = loc2glob(stree)
        src = root_elements(stree)
    else
        spl = hmatrix_splitter(; nmax=nmax)
        stree, sl2g, src = build_point_tree(sources, spl)
    end
    if node_id(stree) == 0
        assign_node_ids!(stree)
    end
    sdata = _allocate_l3(stree, p)
    p2p = Tuple{ClusterTree{3,Float64},ClusterTree{3,Float64}}[]
    m2l = Tuple{ClusterTree{3,Float64},ClusterTree{3,Float64}}[]
    # Dual-tree on the cubic octree so hanging-node (adaptive) neighbors are
    # P2P, matching FMM3D's U-list. Equal-size well-separated pairs are M2L.
    adm = StrongAdmissibility(η=Float64(η))
    _collect_self_jobs!(m2l, p2p, stree, stree, adm)
    get_spherical_cache(p)
    deg = [_coeff_degree(i) for i in 1:_ncoeff_sph(p)]
    m2l_T = Dict{NTuple{4,Int},Matrix{Float64}}()
    trans = Laplace3DTransWS(p)
    src_x = [pt[1] for pt in src]
    src_y = [pt[2] for pt in src]
    src_z = [pt[3] for pt in src]
    pw = _pw_quad(p)
    pw_il, es_jobs = _split_m2l_planewave(m2l, pw)
    nn = nnodes(stree)
    return Laplace3DFMMPlan(
        ns, p, stree, sl2g, src, sdata, eps, nmax, η, true,
        zeros(Float64, ns), zeros(Float64, ns), zeros(Float64, 3, ns),
        zeros(Float64, p + 1, p + 1),
        p2p, m2l, trans, Laplace3DTransWS[],
        src_x, src_y, src_z,
        _group_jobs_by_target(m2l), _group_jobs_by_target(p2p),
        collect(leaves(stree)), nodes_by_depth(stree), m2l_T, deg,
        pw, pw_il, es_jobs, _group_jobs_by_target(es_jobs),
        zeros(ComplexF64, pw.nexp, nn), Laplace3DPWWS[],
    )
end
