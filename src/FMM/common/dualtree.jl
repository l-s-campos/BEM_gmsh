# =============================================================================
# Generic dual-tree multipole engine
#
# Far-field modes:
#   m2l!  — multipole → local (full FMM)
#   m2p!  — multipole → particle (treecode / leaf multipoles)
#
# Threading: interaction lists are collected then executed in parallel.
# =============================================================================

"""Per-node expansion storage."""
mutable struct ExpData
    multipole::Vector{ComplexF64}
    localexp::Vector{ComplexF64}
    rscale::Float64
    center::SVector
end

function allocate_expdata(root::ClusterTree{N}, ncoeff::Int) where {N}
    data = Dict{UInt,ExpData}()
    for node in nodes(root)
        c = SVector{N,Float64}(center(container(node)))
        hs = max(maximum(high_corner(container(node)) - low_corner(container(node))) / 2, 1e-30)
        data[objectid(node)] = ExpData(
            zeros(ComplexF64, ncoeff), zeros(ComplexF64, ncoeff), Float64(hs), c,
        )
    end
    return data
end

_edata(data, node) = data[objectid(node)]

function zero_expansions!(data; multipole=true, localexp=true)
    for ed in values(data)
        multipole && fill!(ed.multipole, 0)
        localexp && fill!(ed.localexp, 0)
    end
end

# ---------- upward (optionally threaded over siblings) ----------
function dualtree_upward!(node, data, upward_leaf!, m2m!; threaded::Bool=false)
    ed = _edata(data, node)
    if isleaf(node)
        fill!(ed.multipole, 0)
        upward_leaf!(node, ed)
    else
        ch = children(node)
        if threaded && length(ch) > 1
            Threads.@threads for child in ch
                dualtree_upward!(child, data, upward_leaf!, m2m!; threaded=false)
            end
        else
            for child in ch
                dualtree_upward!(child, data, upward_leaf!, m2m!; threaded=threaded)
            end
        end
        fill!(ed.multipole, 0)
        for child in ch
            m2m!(ed, _edata(data, child))
        end
    end
    return nothing
end

# ---------- interaction list collection ----------
const JOB_M2L = UInt8(1)
const JOB_M2P = UInt8(2)
const JOB_P2P = UInt8(3)

struct DualJob
    kind::UInt8
    tnode::Any
    snode::Any
end

function _collect_interact!(
    jobs::Vector{DualJob},
    tnode, snode, adm;
    m2l! = nothing,
    m2p! = nothing,
    leaf_mpole_only::Bool=false,
)
    if adm(tnode, snode)
        if m2l! !== nothing && !leaf_mpole_only
            push!(jobs, DualJob(JOB_M2L, tnode, snode))
            return nothing
        elseif m2p! !== nothing
            if !isleaf(snode)
                for c in children(snode)
                    _collect_interact!(jobs, tnode, c, adm; m2l!, m2p!, leaf_mpole_only)
                end
            elseif !isleaf(tnode)
                for c in children(tnode)
                    _collect_interact!(jobs, c, snode, adm; m2l!, m2p!, leaf_mpole_only)
                end
            else
                push!(jobs, DualJob(JOB_M2P, tnode, snode))
            end
            return nothing
        end
        return nothing
    end
    if isleaf(tnode) && isleaf(snode)
        push!(jobs, DualJob(JOB_P2P, tnode, snode))
        return nothing
    end
    if isleaf(snode) || (!isleaf(tnode) && diameter(tnode) >= diameter(snode))
        for c in children(tnode)
            _collect_interact!(jobs, c, snode, adm; m2l!, m2p!, leaf_mpole_only)
        end
    else
        for c in children(snode)
            _collect_interact!(jobs, tnode, c, adm; m2l!, m2p!, leaf_mpole_only)
        end
    end
    return nothing
end

"""
Dual-tree interaction with optional threading.

When `threaded=true`, builds an interaction list then runs independent
M2L / M2P / P2P jobs with `Threads.@threads`.
"""
function dualtree_interact!(
    tnode, snode, tdata, sdata, adm;
    m2l! = nothing,
    m2p! = nothing,
    p2p!,
    leaf_mpole_only::Bool=false,
    threaded::Bool=false,
)
    if !threaded
        return _dualtree_interact_serial!(
            tnode, snode, tdata, sdata, adm; m2l!, m2p!, p2p!, leaf_mpole_only,
        )
    end

    jobs = DualJob[]
    sizehint!(jobs, 256)
    _collect_interact!(jobs, tnode, snode, adm; m2l!, m2p!, leaf_mpole_only)

    # Partition by type so M2L (writes target locals) can be careful about races.
    # M2L jobs that write to the same target node must not run concurrently.
    # Group M2L by target node id; run groups in parallel across different targets.
    m2l_jobs = DualJob[]
    other_jobs = DualJob[]
    for j in jobs
        if j.kind == JOB_M2L
            push!(m2l_jobs, j)
        else
            push!(other_jobs, j)
        end
    end

    # P2P / M2P write to disjoint target-leaf particle buffers if each leaf has
    # its own buffer — still safe if two jobs share a target leaf (atomic-add race).
    # We bin other_jobs by target node and run bins in parallel.
    if !isempty(other_jobs)
        bins = Dict{UInt,Vector{DualJob}}()
        for j in other_jobs
            id = objectid(j.tnode)
            list = get!(bins, id) do
                DualJob[]
            end
            push!(list, j)
        end
        bin_list = collect(values(bins))
        Threads.@threads for b in bin_list
            for j in b
                if j.kind == JOB_M2P
                    m2p!(j.tnode, j.snode, tdata, sdata)
                else
                    p2p!(j.tnode, j.snode)
                end
            end
        end
    end

    if !isempty(m2l_jobs) && m2l! !== nothing
        bins = Dict{UInt,Vector{DualJob}}()
        for j in m2l_jobs
            id = objectid(j.tnode)
            list = get!(bins, id) do
                DualJob[]
            end
            push!(list, j)
        end
        bin_list = collect(values(bins))
        Threads.@threads for b in bin_list
            for j in b
                m2l!(j.tnode, j.snode, tdata, sdata)
            end
        end
    end
    return nothing
end

function _dualtree_interact_serial!(
    tnode, snode, tdata, sdata, adm;
    m2l! = nothing,
    m2p! = nothing,
    p2p!,
    leaf_mpole_only::Bool=false,
)
    if adm(tnode, snode)
        if m2l! !== nothing && !leaf_mpole_only
            m2l!(tnode, snode, tdata, sdata)
            return nothing
        elseif m2p! !== nothing
            if !isleaf(snode)
                for c in children(snode)
                    _dualtree_interact_serial!(tnode, c, tdata, sdata, adm; m2l!, m2p!, p2p!, leaf_mpole_only)
                end
            elseif !isleaf(tnode)
                for c in children(tnode)
                    _dualtree_interact_serial!(c, snode, tdata, sdata, adm; m2l!, m2p!, p2p!, leaf_mpole_only)
                end
            else
                m2p!(tnode, snode, tdata, sdata)
            end
            return nothing
        end
        return nothing
    end
    if isleaf(tnode) && isleaf(snode)
        p2p!(tnode, snode)
        return nothing
    end
    if isleaf(snode) || (!isleaf(tnode) && diameter(tnode) >= diameter(snode))
        for c in children(tnode)
            _dualtree_interact_serial!(c, snode, tdata, sdata, adm; m2l!, m2p!, p2p!, leaf_mpole_only)
        end
    else
        for c in children(snode)
            _dualtree_interact_serial!(tnode, c, tdata, sdata, adm; m2l!, m2p!, p2p!, leaf_mpole_only)
        end
    end
    return nothing
end

function dualtree_downward!(node, data, l2l!, l2p!; threaded::Bool=false)
    ed = _edata(data, node)
    if isleaf(node)
        l2p!(node, ed)
    else
        ch = children(node)
        for child in ch
            cd = _edata(data, child)
            l2l!(cd, ed)
        end
        if threaded && length(ch) > 1
            Threads.@threads for child in ch
                dualtree_downward!(child, data, l2l!, l2p!; threaded=false)
            end
        else
            for child in ch
                dualtree_downward!(child, data, l2l!, l2p!; threaded=threaded)
            end
        end
    end
    return nothing
end

"""Build points as `SVector{N}` list and a cluster tree."""
function build_point_tree(pts::AbstractMatrix{<:Real}, spl)
    N = size(pts, 1)
    n = size(pts, 2)
    sv = [SVector{N,Float64}(ntuple(d -> Float64(pts[d, i]), N)) for i in 1:n]
    tree = ClusterTree(sv, spl; copy_elements=false)
    return tree, loc2glob(tree), root_elements(tree)
end

function permute_to_local(v::AbstractVector, l2g)
    out = similar(v, Float64)
    @inbounds for i in eachindex(l2g)
        out[i] = Float64(real(v[l2g[i]]))
    end
    return out
end

function permute_to_local_complex(v::AbstractVector, l2g)
    out = Vector{ComplexF64}(undef, length(l2g))
    @inbounds for i in eachindex(l2g)
        out[i] = complex(v[l2g[i]])
    end
    return out
end

function unpermute!(dest::AbstractVector, src::AbstractVector, l2g)
    @inbounds for i in eachindex(l2g)
        dest[l2g[i]] = src[i]
    end
    return dest
end

# ---------- spherical helpers for 3D translations ----------

"""Fibonacci sphere points on sphere of radius `R` about `center`."""
function fibonacci_sphere(center::SVector{3,Float64}, R::Float64, n::Int)
    pts = Vector{SVector{3,Float64}}(undef, n)
    φ = π * (√5 - 1)
    @inbounds for i in 0:(n - 1)
        y = 1 - 2(i + 0.5) / n
        r = sqrt(max(0.0, 1 - y * y))
        θ = φ * i
        pts[i + 1] = center + SVector(R * r * cos(θ), R * r * sin(θ), R * y)
    end
    return pts
end
