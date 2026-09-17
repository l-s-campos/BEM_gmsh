# =============================================================================
# Dual-tree 2D Laplace FMM on ClusterTree
#
# Expansions are the complex log multipoles of Flatiron l2d*. Physical potential
# is Re(φ). Densities are therefore processed as real channels; complex charges
# run two real FMMs and recombine.
# =============================================================================

"""Output container (Flatiron-style)."""
mutable struct FMMVals
    pot::Any
    grad::Any
    hess::Any
    pottarg::Any
    gradtarg::Any
    hesstarg::Any
    pre::Any
    pretarg::Any
    ier::Any
end
FMMVals() = FMMVals(
    nothing, nothing, nothing,
    nothing, nothing, nothing,
    nothing, nothing,
    0,
)

"""FMM strong admissibility — alias of `FMMStrongAdmissibility` (HMatrices)."""
const StrongAdmissibility = FMMStrongAdmissibility

struct FMMNodeData
    multipole::Vector{ComplexF64}
    localexp::Vector{ComplexF64}
    rscale::Float64
    center::SVector{2,Float64}
end

function _node_center_scale(node::ClusterTree{2,T}) where {T}
    c = center(container(node))
    hs = max(maximum(high_corner(container(node)) - low_corner(container(node))) / 2, 1e-30)
    return SVector{2,Float64}(c), Float64(hs)
end

function _allocate_expdata(root::ClusterTree{2}, nterms::Int)
    # Prefer node_id-indexed Vector (HMatrices assigns ids at build).
    if node_id(root) == 0
        assign_node_ids!(root)
    end
    nn = nnodes(root)
    data = Vector{FMMNodeData}(undef, nn)
    for node in nodes(root)
        ctr, rsc = _node_center_scale(node)
        data[node_id(node)] = FMMNodeData(
            zeros(ComplexF64, nterms + 1),
            zeros(ComplexF64, nterms + 1),
            rsc,
            ctr,
        )
    end
    return data
end

_get(data::AbstractVector, node) = data[node_id(node)]
_get(data::AbstractDict, node) = data[objectid(node)]

_leafget(leafmap::AbstractVector, node) = leafmap[node_id(node)]
_leafget(leafmap::AbstractDict, node) = leafmap[objectid(node)]

"""Leaf particle buffers indexed by `node_id` (nothing on non-leaves)."""
function _make_leaf_buffers(tree::ClusterTree{2}, pot_local::Vector{Float64}, want_grad::Bool)
    if node_id(tree) == 0
        assign_node_ids!(tree)
    end
    nn = nnodes(tree)
    pot_leaf = Vector{Any}(undef, nn)
    fill!(pot_leaf, nothing)
    grad_local = want_grad ? zeros(ComplexF64, length(pot_local)) : nothing
    grad_leaf = want_grad ? Vector{Any}(undef, nn) : nothing
    want_grad && fill!(grad_leaf, nothing)
    for leaf in leaves(tree)
        r = index_range(leaf)
        id = node_id(leaf)
        pot_leaf[id] = view(pot_local, r)
        if want_grad
            grad_leaf[id] = view(grad_local, r)
        end
    end
    return pot_leaf, grad_leaf, grad_local
end

function _zero_locals!(data)
    itr = data isa AbstractDict ? values(data) : data
    for nd in itr
        fill!(nd.localexp, 0)
    end
end

function _zero_multipoles!(data)
    itr = data isa AbstractDict ? values(data) : data
    for nd in itr
        fill!(nd.multipole, 0)
    end
end

# ---------- upward ----------
function _upward!(node::ClusterTree{2}, data, sources, charges, dipoles, carray,
        ws::Laplace2DTransWS)
    nd = _get(data, node)
    if isleaf(node)
        fill!(nd.multipole, 0)
        if charges !== nothing
            form_multipole_charge!(
                nd.multipole, nd.rscale, nd.center, sources, charges, index_range(node),
                ws.tmp1)
        end
        if dipoles !== nothing
            form_multipole_dipole!(
                nd.multipole, nd.rscale, nd.center, sources, dipoles, index_range(node),
                ws.tmp2)
        end
    else
        fill!(nd.multipole, 0)
        for child in children(node)
            _upward!(child, data, sources, charges, dipoles, carray, ws)
            cd = _get(data, child)
            m2m!(nd.multipole, nd.rscale, nd.center, cd.multipole, cd.rscale, cd.center,
                carray, ws)
        end
    end
    return nothing
end
function _upward!(node::ClusterTree{2}, data, sources, charges, dipoles, carray)
    nterms = length(_get(data, node).multipole) - 1
    return _upward!(node, data, sources, charges, dipoles, carray, Laplace2DTransWS(nterms))
end

# ---------- dual-tree interaction ----------
function _interact!(
    tnode::ClusterTree{2},
    snode::ClusterTree{2},
    tdata,
    sdata,
    sources,
    charges,
    dipstr_loc,
    dipvec_loc,
    targets,
    pot_leaf,
    grad_leaf,
    carray,
    adm,
    thresh::Float64,
    same_tree::Bool,
)
    if adm(tnode, snode)
        td = _get(tdata, tnode)
        sd = _get(sdata, snode)
        m2l!(td.localexp, td.rscale, td.center, sd.multipole, sd.rscale, sd.center, carray)
        return nothing
    end

    if isleaf(tnode) && isleaf(snode)
        g = grad_leaf === nothing ? nothing : grad_leaf
        # pot_leaf / grad_leaf are full-length local buffers (index = tree order)
        direct_laplace_sv!(
            pot_leaf,
            g,
            sources,
            charges,
            dipstr_loc,
            dipvec_loc,
            index_range(snode),
            targets,
            index_range(tnode);
            thresh=thresh,
            exclude_self=same_tree,
        )
        return nothing
    end

    if isleaf(snode) || (!isleaf(tnode) && diameter(tnode) >= diameter(snode))
        for c in children(tnode)
            _interact!(
                c, snode, tdata, sdata, sources, charges, dipstr_loc, dipvec_loc, targets,
                pot_leaf, grad_leaf, carray, adm, thresh, same_tree,
            )
        end
    else
        for c in children(snode)
            _interact!(
                tnode, c, tdata, sdata, sources, charges, dipstr_loc, dipvec_loc, targets,
                pot_leaf, grad_leaf, carray, adm, thresh, same_tree,
            )
        end
    end
    return nothing
end

# ---------- downward ----------
function _downward!(node::ClusterTree{2}, data, targets, pot_leaf, grad_leaf, carray,
        ws::Laplace2DTransWS)
    nd = _get(data, node)
    if isleaf(node)
        g = grad_leaf === nothing ? nothing : grad_leaf
        eval_local!(pot_leaf, g, nd.rscale, nd.center, nd.localexp, targets, index_range(node))
    else
        for child in children(node)
            cd = _get(data, child)
            l2l!(cd.localexp, cd.rscale, cd.center, nd.localexp, nd.rscale, nd.center, carray, ws)
            _downward!(child, data, targets, pot_leaf, grad_leaf, carray, ws)
        end
    end
    return nothing
end
function _downward!(node::ClusterTree{2}, data, targets, pot_leaf, grad_leaf, carray)
    nterms = length(_get(data, node).localexp) - 1
    return _downward!(node, data, targets, pot_leaf, grad_leaf, carray, Laplace2DTransWS(nterms))
end

# ---------- index helpers ----------
function _to_local_real(v::AbstractVector, l2g::Vector{Int})
    out = Vector{Float64}(undef, length(l2g))
    @inbounds for i in eachindex(l2g)
        out[i] = Float64(real(v[l2g[i]]))  # real channel
    end
    return out
end

function _to_local_imag(v::AbstractVector, l2g::Vector{Int})
    out = Vector{Float64}(undef, length(l2g))
    @inbounds for i in eachindex(l2g)
        out[i] = Float64(imag(complex(v[l2g[i]])))
    end
    return out
end

function _local_dipoles_converted(dipstr::AbstractVector, dipvec::AbstractMatrix, l2g, ::Val{:real})
    n = length(l2g)
    out = Vector{ComplexF64}(undef, n)
    @inbounds for i in 1:n
        g = l2g[i]
        s = Float64(real(dipstr[g]))
        out[i] = s * (-complex(dipvec[1, g], dipvec[2, g]))
    end
    return out
end

function _local_dipoles_converted(dipstr::AbstractVector, dipvec::AbstractMatrix, l2g, ::Val{:imag})
    n = length(l2g)
    out = Vector{ComplexF64}(undef, n)
    @inbounds for i in 1:n
        g = l2g[i]
        s = Float64(imag(complex(dipstr[g])))
        out[i] = s * (-complex(dipvec[1, g], dipvec[2, g]))
    end
    return out
end

function _local_dipstr_real(dipstr::AbstractVector, l2g)
    out = Vector{Float64}(undef, length(l2g))
    @inbounds for i in eachindex(l2g)
        out[i] = Float64(real(dipstr[l2g[i]]))
    end
    return out
end

function _local_dipstr_imag(dipstr::AbstractVector, l2g)
    out = Vector{Float64}(undef, length(l2g))
    @inbounds for i in eachindex(l2g)
        out[i] = Float64(imag(complex(dipstr[l2g[i]])))
    end
    return out
end

function _local_dipvec(dipvec::AbstractMatrix, l2g)
    out = Vector{SVector{2,Float64}}(undef, length(l2g))
    @inbounds for i in eachindex(l2g)
        g = l2g[i]
        out[i] = SVector{2,Float64}(dipvec[1, g], dipvec[2, g])
    end
    return out
end

"""
    _fmm_real_channel(...) -> (pot_src, grad_src_dz, pot_trg, grad_trg_dz)

Single real-valued density channel. Potentials are `Float64`; gradients are
packed as complex `d/dz` (convert with `dz_to_physical`).
"""
function _fmm_real_channel(
    stree,
    sdata,
    src_local,
    sl2g,
    charges_loc::Union{Nothing,Vector{Float64}},
    dipoles_mp::Union{Nothing,Vector{ComplexF64}},
    dipstr_loc::Union{Nothing,Vector{Float64}},
    dipvec_loc::Union{Nothing,Vector{SVector{2,Float64}}},
    targets_mat,
    pg::Int,
    pgt::Int,
    nterms::Int,
    carray,
    adm,
    thresh::Float64,
    spl,
)
    ns = length(sl2g)
    _zero_multipoles!(sdata)
    _upward!(stree, sdata, src_local, charges_loc, dipoles_mp, carray)

    pot_src = nothing
    grad_src = nothing
    pot_trg = nothing
    grad_trg = nothing

    if pg > 0
        pot_local = zeros(Float64, ns)
        pot_leaf, grad_leaf, grad_local = _make_leaf_buffers(stree, pot_local, pg >= 2)
        _zero_locals!(sdata)
        _interact!(
            stree, stree, sdata, sdata, src_local, charges_loc, dipstr_loc, dipvec_loc,
            src_local, pot_local, grad_local, carray, adm, thresh, true,
        )
        _downward!(stree, sdata, src_local, pot_local, grad_local, carray)

        pot_src = zeros(Float64, ns)
        @inbounds for i in 1:ns
            pot_src[sl2g[i]] = pot_local[i]
        end
        if pg >= 2
            grad_src = zeros(ComplexF64, ns)  # d/dz in global order
            @inbounds for i in 1:ns
                grad_src[sl2g[i]] = grad_local[i]
            end
        end
    end

    if pgt > 0 && targets_mat !== nothing
        nt = size(targets_mat, 2)
        trg_pts = [SVector{2,Float64}(targets_mat[1, i], targets_mat[2, i]) for i in 1:nt]
        ttree = ClusterTree(trg_pts, spl; copy_elements=false)
        tl2g = loc2glob(ttree)
        trg_local = root_elements(ttree)
        tdata = _allocate_expdata(ttree, nterms)
        _zero_locals!(tdata)

        pot_tlocal = zeros(Float64, nt)
        pot_leaf, grad_leaf, grad_tlocal = _make_leaf_buffers(ttree, pot_tlocal, pgt >= 2)

        _interact!(
            ttree, stree, tdata, sdata, src_local, charges_loc, dipstr_loc, dipvec_loc,
            trg_local, pot_tlocal, grad_tlocal, carray, adm, thresh, false,
        )
        _downward!(ttree, tdata, trg_local, pot_tlocal, grad_tlocal, carray)

        pot_trg = zeros(Float64, nt)
        @inbounds for i in 1:nt
            pot_trg[tl2g[i]] = pot_tlocal[i]
        end
        if pgt >= 2
            grad_trg = zeros(ComplexF64, nt)
            @inbounds for i in 1:nt
                grad_trg[tl2g[i]] = grad_tlocal[i]
            end
        end
    end

    return pot_src, grad_src, pot_trg, grad_trg
end

function _pack_grad(gdz::Vector{ComplexF64})
    n = length(gdz)
    g = zeros(Float64, 2, n)
    @inbounds for i in 1:n
        gx, gy = dz_to_physical(gdz[i])
        # For real potential, gradient components are Re(φ_x)-style:
        # ∂x u = Re(φ'), ∂y u = -Im(φ') when u = Re(φ)
        g[1, i] = gx
        g[2, i] = gy
    end
    return g
end

function _densities_are_complex(charges, dipstr)
    if charges !== nothing
        eltype(charges) <: Complex && return true
        any(x -> imag(complex(x)) != 0, charges) && return true
    end
    if dipstr !== nothing
        eltype(dipstr) <: Complex && return true
        any(x -> imag(complex(x)) != 0, dipstr) && return true
    end
    return false
end

function _fmm2d_laplace(
    eps::Float64,
    sources::AbstractMatrix{<:Real};
    charges=nothing,
    dipstr=nothing,
    dipvec=nothing,
    targets=nothing,
    pg::Int=0,
    pgt::Int=0,
    nmax::Int=-1,
    η::Float64=1.0,
    splitter=nothing,
)
    @assert size(sources, 1) == 2 "sources must be 2×n"
    ns = size(sources, 2)
    has_charge = charges !== nothing
    has_dipole = dipstr !== nothing
    @assert has_charge || has_dipole "provide charges and/or dipoles"
    @assert pg > 0 || pgt > 0 "set pg and/or pgt to 1 or 2"
    if has_dipole
        @assert dipvec !== nothing "dipvec required with dipstr"
        @assert size(dipvec, 1) == 2 && size(dipvec, 2) == ns
    end
    if targets !== nothing
        @assert size(targets, 1) == 2
    end

    nterms = laplace_nterms(eps)
    carray = binomial_table(2 * nterms + 2)
    adm = StrongAdmissibility(η=η)
    nmax = nmax < 0 ? laplace2d_ndiv(eps) : nmax
    spl = splitter === nothing ? hmatrix_splitter(; nmax=nmax) : splitter

    src_pts = [SVector{2,Float64}(sources[1, i], sources[2, i]) for i in 1:ns]
    cube = spl isa DyadicSplitter && !spl.tight
    stree = ClusterTree(src_pts, spl; copy_elements=false, cube=cube)
    sl2g = loc2glob(stree)
    src_local = root_elements(stree)
    sdata = _allocate_expdata(stree, nterms)

    L = diameter(container(stree))
    thresh = L * Base.eps(Float64)

    chg_vec = has_charge ? (charges isa AbstractMatrix ? vec(charges) : charges) : nothing
    dstr_vec = has_dipole ? (dipstr isa AbstractMatrix ? vec(dipstr) : dipstr) : nothing
    dipvec_loc = has_dipole ? _local_dipvec(dipvec, sl2g) : nothing

    complex_dens = _densities_are_complex(chg_vec, dstr_vec)

    # --- real channel ---
    ch_r = has_charge ? _to_local_real(chg_vec, sl2g) : nothing
    dp_r = has_dipole ? _local_dipoles_converted(dstr_vec, dipvec, sl2g, Val(:real)) : nothing
    ds_r = has_dipole ? _local_dipstr_real(dstr_vec, sl2g) : nothing

    pot_r, gdz_r, pt_r, gt_r = _fmm_real_channel(
        stree, sdata, src_local, sl2g, ch_r, dp_r, ds_r, dipvec_loc,
        targets, pg, pgt, nterms, carray, adm, thresh, spl,
    )

    if !complex_dens
        pot_src = pot_r === nothing ? nothing : complex.(pot_r)
        grad_src = gdz_r === nothing ? nothing : complex.(_pack_grad(gdz_r))
        pot_trg = pt_r === nothing ? nothing : complex.(pt_r)
        grad_trg = gt_r === nothing ? nothing : complex.(_pack_grad(gt_r))
        return pot_src, grad_src, pot_trg, grad_trg
    end

    # --- imag channel ---
    ch_i = has_charge ? _to_local_imag(chg_vec, sl2g) : nothing
    dp_i = has_dipole ? _local_dipoles_converted(dstr_vec, dipvec, sl2g, Val(:imag)) : nothing
    ds_i = has_dipole ? _local_dipstr_imag(dstr_vec, sl2g) : nothing

    pot_i, gdz_i, pt_i, gt_i = _fmm_real_channel(
        stree, sdata, src_local, sl2g, ch_i, dp_i, ds_i, dipvec_loc,
        targets, pg, pgt, nterms, carray, adm, thresh, spl,
    )

    pot_src =
        pot_r === nothing ? nothing : (complex.(pot_r) .+ im .* pot_i)
    pot_trg =
        pt_r === nothing ? nothing : (complex.(pt_r) .+ im .* pt_i)

    grad_src = nothing
    if gdz_r !== nothing
        gr = _pack_grad(gdz_r)
        gi = _pack_grad(gdz_i)
        grad_src = complex.(gr) .+ im .* gi
    end
    grad_trg = nothing
    if gt_r !== nothing
        gr = _pack_grad(gt_r)
        gi = _pack_grad(gt_i)
        grad_trg = complex.(gr) .+ im .* gi
    end

    return pot_src, grad_src, pot_trg, grad_trg
end

# =============================================================================
# Cached plan (reuse tree + expansions across matvecs)
# =============================================================================

"""
Precomputed 2D Laplace FMM geometry and workspaces.

Build once with [`build_laplace2d_plan`](@ref), then call
[`apply_laplace2d!`](@ref) for each density. Avoids rebuilding the cluster tree,
binomial table, and expansion buffers on every matvec (the previous bottleneck
in `FMMKernelMatrix` / Kelvin).
"""
mutable struct Laplace2DFMMPlan
    n::Int
    tree::ClusterTree{2,Float64}
    sl2g::Vector{Int}
    src_local::Vector{SVector{2,Float64}}
    sdata::Vector{FMMNodeData}
    carray::Matrix{Float64}
    nterms::Int
    adm::StrongAdmissibility
    thresh::Float64
    dens_loc::Vector{Float64}
    dipoles_mp::Vector{ComplexF64}
    dipstr_loc::Vector{Float64}
    dipvec_loc::Union{Nothing,Vector{SVector{2,Float64}}}
    pot_local::Vector{Float64}
    pot_leaf::Vector{Any}
    grad_local::Union{Nothing,Vector{ComplexF64}}
    grad_leaf::Union{Nothing,Vector{Any}}
    want_grad::Bool
    # Cached dual-tree jobs (self interaction): far M2L and near P2P
    m2l_jobs::Vector{Tuple{ClusterTree{2,Float64},ClusterTree{2,Float64}}}
    p2p_jobs::Vector{Tuple{ClusterTree{2,Float64},ClusterTree{2,Float64}}}
    trans::Laplace2DTransWS
    src_x::Vector{Float64}
    src_y::Vector{Float64}
    m2l_groups::Vector{Tuple{ClusterTree{2,Float64},Vector{ClusterTree{2,Float64}}}}
    p2p_groups::Vector{Tuple{ClusterTree{2,Float64},Vector{ClusterTree{2,Float64}}}}
    leaf_nodes::Vector{ClusterTree{2,Float64}}
    trans_tls::Vector{Laplace2DTransWS}
end

"""Group `(tnode, snode)` jobs by target so writes to one box stay sequential
and different targets can run under `Threads.@threads` (Flatiron OpenMP)."""
function _group_jobs_by_target(jobs::Vector{Tuple{T,T}}) where T
    map = Dict{Int,Int}()
    groups = Vector{Tuple{T,Vector{T}}}()
    sizehint!(groups, length(jobs))
    for (t, s) in jobs
        id = node_id(t)
        i = get(map, id, 0)
        if i == 0
            push!(groups, (t, T[s]))
            map[id] = length(groups)
        else
            push!(groups[i][2], s)
        end
    end
    return groups
end

@inline _fmm_tls_len() = max(Int(Threads.nthreads()), Int(Threads.maxthreadid()))

function _ensure_trans_tls!(plan::Laplace2DFMMPlan)
    nt = _fmm_tls_len()
    tls = plan.trans_tls
    if length(tls) < nt
        old = length(tls)
        resize!(plan.trans_tls, nt)
        old == 0 && (plan.trans_tls[1] = plan.trans)
        @inbounds for i in max(old + 1, 2):nt
            plan.trans_tls[i] = Laplace2DTransWS(plan.nterms)
        end
    end
    return plan.trans_tls
end

function _upward_m2m_only!(node::ClusterTree{2}, data, carray, ws::Laplace2DTransWS)
    isleaf(node) && return nothing
    for child in children(node)
        _upward_m2m_only!(child, data, carray, ws)
    end
    nd = _get(data, node)
    fill!(nd.multipole, 0)
    for child in children(node)
        cd = _get(data, child)
        m2m!(nd.multipole, nd.rscale, nd.center, cd.multipole, cd.rscale, cd.center, carray, ws)
    end
    return nothing
end

function _downward_l2l_only!(node::ClusterTree{2}, data, carray, ws::Laplace2DTransWS)
    isleaf(node) && return nothing
    nd = _get(data, node)
    for child in children(node)
        cd = _get(data, child)
        l2l!(cd.localexp, cd.rscale, cd.center, nd.localexp, nd.rscale, nd.center, carray, ws)
        _downward_l2l_only!(child, data, carray, ws)
    end
    return nothing
end

function _upward_plan!(plan::Laplace2DFMMPlan, ch_loc, dp_mp)
    lvs = plan.leaf_nodes
    nt = Threads.nthreads()
    src = plan.src_local
    if nt > 1 && length(lvs) >= 8
        tls = _ensure_trans_tls!(plan)
        Threads.@threads :static for i in eachindex(lvs)
            leaf = lvs[i]
            nd = _get(plan.sdata, leaf)
            fill!(nd.multipole, 0)
            ws = tls[Threads.threadid()]
            if ch_loc !== nothing
                form_multipole_charge!(
                    nd.multipole, nd.rscale, nd.center, src, ch_loc, index_range(leaf), ws.tmp1)
            end
            if dp_mp !== nothing
                form_multipole_dipole!(
                    nd.multipole, nd.rscale, nd.center, src, dp_mp, index_range(leaf), ws.tmp2)
            end
        end
        _upward_m2m_only!(plan.tree, plan.sdata, plan.carray, plan.trans)
    else
        _upward!(plan.tree, plan.sdata, src, ch_loc, dp_mp, plan.carray, plan.trans)
    end
    return nothing
end

function _downward_plan!(plan::Laplace2DFMMPlan)
    _downward_l2l_only!(plan.tree, plan.sdata, plan.carray, plan.trans)
    gfull = plan.want_grad ? plan.grad_local : nothing
    lvs = plan.leaf_nodes
    src = plan.src_local
    nt = Threads.nthreads()
    if nt > 1 && length(lvs) >= 8
        Threads.@threads :static for i in eachindex(lvs)
            leaf = lvs[i]
            nd = _get(plan.sdata, leaf)
            eval_local!(plan.pot_local, gfull, nd.rscale, nd.center, nd.localexp, src, index_range(leaf))
        end
    else
        for leaf in lvs
            nd = _get(plan.sdata, leaf)
            eval_local!(plan.pot_local, gfull, nd.rscale, nd.center, nd.localexp, src, index_range(leaf))
        end
    end
    return nothing
end

"""Collect self-interaction M2L / P2P lists once (same logic as `_interact!`)."""
function _collect_self_jobs!(m2l, p2p, tnode, snode, adm)
    if adm(tnode, snode)
        push!(m2l, (tnode, snode))
        return nothing
    end
    if isleaf(tnode) && isleaf(snode)
        push!(p2p, (tnode, snode))
        return nothing
    end
    if isleaf(snode) || (!isleaf(tnode) && diameter(tnode) >= diameter(snode))
        for c in children(tnode)
            _collect_self_jobs!(m2l, p2p, c, snode, adm)
        end
    else
        for c in children(snode)
            _collect_self_jobs!(m2l, p2p, tnode, c, adm)
        end
    end
    return nothing
end

function _m2l_groups_chunk_2d!(sdata, carray, groups, ws, tid, nt, n)
    @inbounds for gi in tid:nt:n
        tnode, snodes = groups[gi]
        td = _get(sdata, tnode)
        for snode in snodes
            sd = _get(sdata, snode)
            m2l!(td.localexp, td.rscale, td.center, sd.multipole, sd.rscale, sd.center,
                carray, ws)
        end
    end
    return nothing
end

function _m2l_plan_2d!(plan::Laplace2DFMMPlan)
    sdata = plan.sdata
    carray = plan.carray
    groups = plan.m2l_groups
    n = length(groups)
    nthr = Threads.nthreads()
    # Spawn-indexed TLS, not `@threads` + `threadid()`: on Julia 1.13 the
    # default pool uses ids 2:nthreads+1 and concurrent `m2l!` with
    # `threadid()` workspaces disagrees with dense. `let tid=tid` so the
    # spawn closure does not capture the loop variable.
    if nthr > 1 && n >= 8
        tls = _ensure_trans_tls!(plan)
        nt = min(nthr, n)
        @sync for tid in 1:nt
            let tid = tid, ws = tls[tid]
                Threads.@spawn _m2l_groups_chunk_2d!(sdata, carray, groups, ws, tid, nt, n)
            end
        end
        return nothing
    end
    ws = plan.trans
    @inbounds for (tnode, snode) in plan.m2l_jobs
        td = _get(sdata, tnode)
        sd = _get(sdata, snode)
        m2l!(td.localexp, td.rscale, td.center, sd.multipole, sd.rscale, sd.center, carray, ws)
    end
    return nothing
end

function _p2p_plan_2d!(plan::Laplace2DFMMPlan, ch_loc, ds_loc, dv_loc, gfull)
    src = plan.src_local
    thresh = plan.thresh
    charge_only = ds_loc === nothing && gfull === nothing && ch_loc !== nothing
    nt = Threads.nthreads()
    threaded = nt > 1 && length(plan.p2p_groups) >= 8
    if threaded
        Threads.@threads :static for gi in eachindex(plan.p2p_groups)
            tnode, snodes = plan.p2p_groups[gi]
            @inbounds for snode in snodes
                same = tnode === snode
                if charge_only
                    direct_laplace_soa!(
                        plan.pot_local, plan.src_x, plan.src_y, ch_loc,
                        index_range(snode), index_range(tnode); exclude_self=same,
                    )
                else
                    direct_laplace_sv!(
                        plan.pot_local, gfull, src, ch_loc, ds_loc, dv_loc,
                        index_range(snode), src, index_range(tnode);
                        thresh=thresh, exclude_self=same,
                    )
                end
            end
        end
        return nothing
    end
    @inbounds for (tnode, snode) in plan.p2p_jobs
        same = tnode === snode
        if charge_only
            direct_laplace_soa!(
                plan.pot_local, plan.src_x, plan.src_y, ch_loc,
                index_range(snode), index_range(tnode); exclude_self=same,
            )
        else
            direct_laplace_sv!(
                plan.pot_local, gfull, src, ch_loc, ds_loc, dv_loc,
                index_range(snode), src, index_range(tnode);
                thresh=thresh, exclude_self=same,
            )
        end
    end
    return nothing
end

function _run_cached_interact!(plan::Laplace2DFMMPlan, ch_loc, ds_loc, dv_loc, gleaf)
    gfull = gleaf === nothing ? nothing : plan.grad_local
    _m2l_plan_2d!(plan)
    _p2p_plan_2d!(plan, ch_loc, ds_loc, dv_loc, gfull)
    return nothing
end

"""
    build_laplace2d_plan(sources; eps=1e-8, nmax=50, η=1.0, pg=1,
                         tree=nothing, dipvec=nothing, splitter=nothing)

Cache tree + expansions + interaction lists for repeated source→source Laplace
matvecs on `sources` (`2 × n`).

# Keywords
- `tree` — optional existing [`ClusterTree`](@ref) on the **same** point set
  (global order = columns of `sources`).
- `pg=2` — allocate gradient workspaces
- `dipvec` — fixed dipole directions (`2 × n`) for double-layer plans
"""
function build_laplace2d_plan(
    sources::AbstractMatrix{<:Real};
    eps::Float64=1e-8,
    nmax::Int=-1,
    η::Float64=1.0,
    pg::Int=1,
    splitter=nothing,
    dipvec=nothing,
    tree=nothing,
)
    size(sources, 1) == 2 || throw(ArgumentError("sources must be 2×n"))
    ns = size(sources, 2)
    nmax = nmax < 0 ? laplace2d_ndiv(eps) : nmax
    nterms = laplace_nterms(eps)
    carray = binomial_table(2 * nterms + 2)
    adm = StrongAdmissibility(η=η)

    stree = if tree !== nothing
        tree isa ClusterTree || throw(ArgumentError("tree must be a ClusterTree"))
        length(tree) == ns || throw(DimensionMismatch(
            "tree length $(length(tree)) ≠ n=$ns — build ClusterTree on the same points"))
        if node_id(tree) == 0
            assign_node_ids!(tree)
        end
        tree
    else
        spl = splitter === nothing ? hmatrix_splitter(; nmax=nmax) : splitter
        src_pts = [SVector{2,Float64}(sources[1, i], sources[2, i]) for i in 1:ns]
        cube = spl isa DyadicSplitter && !spl.tight
        ClusterTree(src_pts, spl; copy_elements=false, cube=cube)
    end

    sl2g = loc2glob(stree)
    src_local = root_elements(stree)
    # Sanity: local points should match sources[sl2g] when tree was built from sources
    sdata = _allocate_expdata(stree, nterms)
    L = diameter(container(stree))
    thresh = L * Base.eps(Float64)

    pot_local = zeros(Float64, ns)
    want_grad = pg >= 2
    pot_leaf, grad_leaf, grad_local = _make_leaf_buffers(stree, pot_local, want_grad)

    dvec_loc = dipvec === nothing ? nothing : _local_dipvec(dipvec, sl2g)

    m2l = Tuple{ClusterTree{2,Float64},ClusterTree{2,Float64}}[]
    p2p = Tuple{ClusterTree{2,Float64},ClusterTree{2,Float64}}[]
    sizehint!(m2l, 256); sizehint!(p2p, 256)
    _collect_self_jobs!(m2l, p2p, stree, stree, adm)
    src_x = [p[1] for p in src_local]
    src_y = [p[2] for p in src_local]
    leaf_nodes = collect(leaves(stree))
    m2l_groups = _group_jobs_by_target(m2l)
    p2p_groups = _group_jobs_by_target(p2p)

    return Laplace2DFMMPlan(
        ns, stree, sl2g, src_local, sdata, carray, nterms, adm, thresh,
        zeros(Float64, ns), zeros(ComplexF64, ns), zeros(Float64, ns), dvec_loc,
        pot_local, pot_leaf, grad_local, grad_leaf, want_grad, m2l, p2p,
        Laplace2DTransWS(nterms),
        src_x, src_y,
        m2l_groups, p2p_groups,
        leaf_nodes, Laplace2DTransWS[],
    )
end

function _permute_to_local!(out::Vector{Float64}, v::AbstractVector, sl2g::Vector{Int})
    @inbounds for i in eachindex(sl2g)
        out[i] = Float64(v[sl2g[i]])
    end
    return out
end

function _unpermute_from_local!(out::AbstractVector, loc::Vector{Float64}, sl2g::Vector{Int})
    @inbounds for i in eachindex(sl2g)
        out[sl2g[i]] = loc[i]
    end
    return out
end

"""Fill multipole packing for real dipoles: `mp_i = s_i * (-(v_x + i v_y))`."""
function _fill_dipole_mp!(mp::Vector{ComplexF64}, dipstr_g::AbstractVector,
        dipvec::AbstractMatrix, sl2g::Vector{Int})
    @inbounds for i in eachindex(sl2g)
        g = sl2g[i]
        s = Float64(dipstr_g[g])
        mp[i] = s * (-complex(dipvec[1, g], dipvec[2, g]))
    end
    return mp
end

"""
    apply_laplace2d!(plan, pot; charges=nothing, dipstr=nothing, dipvec=nothing, grad=nothing)

In-place source→source Laplace FMM using a cached [`Laplace2DFMMPlan`](@ref).
Provide `charges` and/or `dipstr` (+ `dipvec` or plan's fixed dipvec).
If `plan.want_grad` and `grad` is a `2×n` matrix, write physical gradients.
"""
function apply_laplace2d!(
    plan::Laplace2DFMMPlan,
    pot::AbstractVector{<:Real};
    charges=nothing,
    dipstr=nothing,
    dipvec=nothing,
    grad=nothing,
)
    ns = plan.n
    length(pot) == ns || throw(DimensionMismatch("pot length $(length(pot)) ≠ $(ns)"))
    has_c = charges !== nothing
    has_d = dipstr !== nothing
    has_c || has_d || throw(ArgumentError("provide charges and/or dipstr"))

    ch_loc = nothing
    if has_c
        length(charges) == ns || throw(DimensionMismatch("charges"))
        _permute_to_local!(plan.dens_loc, charges, plan.sl2g)
        ch_loc = plan.dens_loc
    end

    dp_mp = nothing
    ds_loc = nothing
    dv_loc = plan.dipvec_loc
    if has_d
        length(dipstr) == ns || throw(DimensionMismatch("dipstr"))
        dmat = dipvec === nothing ? nothing : dipvec
        if dmat === nothing && dv_loc === nothing
            throw(ArgumentError("dipvec required with dipstr (or bake into plan)"))
        end
        if dmat !== nothing
            # refresh plan local dipvec if caller passes a matrix each time
            dv_loc = _local_dipvec(dmat, plan.sl2g)
            plan.dipvec_loc = dv_loc
            _fill_dipole_mp!(plan.dipoles_mp, dipstr, dmat, plan.sl2g)
        else
            # fixed dipvec_loc on plan: rebuild mp from global dipstr + local vec
            @inbounds for i in eachindex(plan.sl2g)
                g = plan.sl2g[i]
                s = Float64(dipstr[g])
                v = plan.dipvec_loc[i]
                plan.dipoles_mp[i] = s * (-complex(v[1], v[2]))
            end
        end
        _permute_to_local!(plan.dipstr_loc, dipstr, plan.sl2g)
        dp_mp = plan.dipoles_mp
        ds_loc = plan.dipstr_loc
    end

    fill!(plan.pot_local, 0)
    if plan.want_grad
        fill!(plan.grad_local, 0)
    end
    _zero_multipoles!(plan.sdata)
    _zero_locals!(plan.sdata)

    _upward_plan!(plan, ch_loc, dp_mp)
    gleaf = plan.want_grad ? plan.grad_leaf : nothing
    _run_cached_interact!(plan, ch_loc, ds_loc, dv_loc, gleaf)
    _downward_plan!(plan)

    _unpermute_from_local!(pot, plan.pot_local, plan.sl2g)

    if grad !== nothing
        plan.want_grad || throw(ArgumentError("plan built with pg=1; rebuild with pg=2 for gradients"))
        size(grad, 1) == 2 && size(grad, 2) == ns || throw(DimensionMismatch("grad must be 2×n"))
        @inbounds for i in 1:ns
            gx, gy = dz_to_physical(plan.grad_local[i])
            g = plan.sl2g[i]
            grad[1, g] = gx
            grad[2, g] = gy
        end
    end
    return pot
end

# =============================================================================
# Public API
# =============================================================================

"""
```julia
vals = lfmm2d(eps, sources; charges=nothing, dipstr=nothing, dipvec=nothing,
              targets=nothing, pg=0, pgt=0, nmax=50, η=1.0)
```

Pure-Julia 2D Laplace FMM with complex densities (Flatiron `lfmm2d` kernel):

```math
u(x) = \\sum_j c_j \\log\\|x-x_j\\|
     + d_j\\, v_j\\cdot\\nabla_{x_j}\\log\\|x-x_j\\|
```
"""
function lfmm2d(
    eps::Real,
    sources::AbstractMatrix{<:Real};
    charges=nothing,
    dipstr=nothing,
    dipvec=nothing,
    targets=nothing,
    pg::Integer=0,
    pgt::Integer=0,
    nmax::Integer=-1,
    η::Real=1.0,
    splitter=nothing,
    plan::Union{Nothing,Laplace2DFMMPlan}=nothing,
)
    vals = FMMVals()
    # Fast path: cached plan, real charges, self evaluation, potential only
    if plan !== nothing && targets === nothing && dipstr === nothing &&
            charges !== nothing && Int(pg) == 1 && Int(pgt) == 0 &&
            !_densities_are_complex(charges, nothing)
        pot = Vector{Float64}(undef, plan.n)
        apply_laplace2d!(plan, pot; charges=charges)
        vals.pot = complex.(pot)
        vals.ier = 0
        return vals
    end
    pot, grad, pottarg, gradtarg = _fmm2d_laplace(
        Float64(eps),
        sources;
        charges=charges,
        dipstr=dipstr,
        dipvec=dipvec,
        targets=targets,
        pg=Int(pg),
        pgt=Int(pgt),
        nmax=Int(nmax),
        η=Float64(η),
        splitter=splitter,
    )
    vals.pot = pot
    vals.grad = grad
    vals.pottarg = pottarg
    vals.gradtarg = gradtarg
    vals.ier = 0
    return vals
end

"""
```julia
vals = rfmm2d(eps, sources; charges=nothing, dipstr=nothing, dipvec=nothing,
              targets=nothing, pg=0, pgt=0, nmax=-1, η=1.0)
```

Real-valued 2D Laplace FMM.
"""
function rfmm2d(
    eps::Real,
    sources::AbstractMatrix{<:Real};
    charges=nothing,
    dipstr=nothing,
    dipvec=nothing,
    targets=nothing,
    pg::Integer=0,
    pgt::Integer=0,
    nmax::Integer=-1,
    η::Real=1.0,
    splitter=nothing,
    plan::Union{Nothing,Laplace2DFMMPlan}=nothing,
)
    # Real charge self-eval via plan (avoids complex round-trip)
    if plan !== nothing && targets === nothing && dipstr === nothing &&
            charges !== nothing && Int(pg) == 1 && Int(pgt) == 0
        vals = FMMVals()
        pot = Vector{Float64}(undef, plan.n)
        apply_laplace2d!(plan, pot; charges=charges)
        vals.pot = pot
        vals.ier = 0
        return vals
    elseif plan !== nothing && targets === nothing && dipstr === nothing &&
            charges !== nothing && Int(pg) == 2 && Int(pgt) == 0 && plan.want_grad
        vals = FMMVals()
        pot = Vector{Float64}(undef, plan.n)
        grad = zeros(Float64, 2, plan.n)
        apply_laplace2d!(plan, pot; charges=charges, grad=grad)
        vals.pot = pot
        vals.grad = grad
        vals.ier = 0
        return vals
    end
    vals = lfmm2d(
        eps, sources;
        charges=charges, dipstr=dipstr, dipvec=dipvec, targets=targets,
        pg=pg, pgt=pgt, nmax=nmax, η=η, splitter=splitter, plan=plan,
    )
    vals.pot !== nothing && (vals.pot = real.(vals.pot))
    vals.grad !== nothing && (vals.grad = real.(vals.grad))
    vals.pottarg !== nothing && (vals.pottarg = real.(vals.pottarg))
    vals.gradtarg !== nothing && (vals.gradtarg = real.(vals.gradtarg))
    return vals
end

"""
```julia
vals = l2ddir(sources, targets; charges=nothing, dipstr=nothing, dipvec=nothing,
              pgt=1, thresh=0.0)
```

Direct ``O(N^2)`` complex Laplace evaluation (sources → targets only).
"""
function l2ddir(
    sources::AbstractMatrix{<:Real},
    targets::AbstractMatrix{<:Real};
    charges=nothing,
    dipstr=nothing,
    dipvec=nothing,
    pgt::Integer=1,
    thresh::Float64=0.0,
)
    nt = size(targets, 2)
    vals = FMMVals()
    pot = zeros(ComplexF64, nt)
    grad = pgt >= 2 ? zeros(ComplexF64, 2, nt) : nothing
    direct_laplace!(pot, grad, sources, charges, dipstr, dipvec, targets; thresh=thresh)
    vals.pottarg = pot
    vals.gradtarg = grad
    vals.ier = 0
    return vals
end

"""Direct ``O(N^2)`` real Laplace evaluation."""
function r2ddir(
    sources::AbstractMatrix{<:Real},
    targets::AbstractMatrix{<:Real};
    charges=nothing,
    dipstr=nothing,
    dipvec=nothing,
    pgt::Integer=1,
    thresh::Float64=0.0,
)
    vals = l2ddir(
        sources, targets; charges=charges, dipstr=dipstr, dipvec=dipvec, pgt=pgt, thresh=thresh
    )
    vals.pottarg = real.(vals.pottarg)
    vals.gradtarg !== nothing && (vals.gradtarg = real.(vals.gradtarg))
    return vals
end
