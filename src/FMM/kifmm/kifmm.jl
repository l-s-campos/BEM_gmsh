# =============================================================================
# Kernel-Independent FMM (Ying–Biros–Zorin / exafmm-t style)
#
# Each box carries:
#   up_equiv  — upward equivalent densities on INNER surface (α=1.05)
#   dn_check  — downward check potentials on INNER surface (α=1.05)
#
# Operators:
#   P2M: sources → up_check pots → pinv(K) → up_equiv
#   M2M: child up_equiv → parent up_check pots → pinv → parent up_equiv
#   M2L: source up_equiv → target dn_check pots
#   L2L: parent dn_equiv (outer) → child dn_check
#   L2P: dn_check → pinv → dn_equiv (outer) → targets
#   P2P: near-field direct
# =============================================================================

const α_INNER = 1.05
const α_OUTER = 2.95

"""Pseudo-inverse via truncated SVD (exafmm-t style)."""
function pinv_svd(A::AbstractMatrix{T}; rtol=nothing) where {T}
    U, S, V = svd(A; full=false)
    ε = rtol === nothing ? (eps(real(T)) * maximum(S) * 4) : rtol * maximum(S)
    Sp = [s > ε ? inv(s) : zero(s) for s in S]
    return V * Diagonal(Sp) * U'
end

"""
Precomputed check↔equivalent maps for one box size (half-width `r`).
"""
struct KIPrecomp{T}
    p::Int
    r::Float64
    nsurf::Int
    # relative surfaces about origin
    up_check::Vector{SVector{3,Float64}}   # outer
    up_equiv::Vector{SVector{3,Float64}}   # inner
    dn_check::Vector{SVector{3,Float64}}   # inner
    dn_equiv::Vector{SVector{3,Float64}}   # outer
    # pinv maps: equiv = UC2E * check_pot
    UC2E::Matrix{T}
    DC2E::Matrix{T}
end

function precompute_ki(ker::KIKernel, p::Int, r::Float64)
    origin = SVector(0.0, 0.0, 0.0)
    up_check = surface_points(p, r, origin; α=α_OUTER)
    up_equiv = surface_points(p, r, origin; α=α_INNER)
    dn_check = surface_points(p, r, origin; α=α_INNER)
    dn_equiv = surface_points(p, r, origin; α=α_OUTER)
    # B[j,i] = K(check_j, equiv_i): pot at check from unit charge at equiv
    B = kernel_matrix(up_equiv, up_check, ker)  # ncheck × nequiv
    UC2E = pinv_svd(B)  # nequiv × ncheck: σ = UC2E * φ
    # downward: same geometry for scale-invariant kernels; reuse
    Bd = kernel_matrix(dn_equiv, dn_check, ker)
    DC2E = pinv_svd(Bd)
    T = eltype(B)
    return KIPrecomp{T}(p, r, nsurf(p), up_check, up_equiv, dn_check, dn_equiv, UC2E, DC2E)
end

"""Cache precomputes by rounded half-width."""
const _KI_CACHE = Dict{UInt64,Any}()

function _ker_key(ker::KILaplace3D, p, r)
    return ( :laplace, p, round(r; sigdigits=6) )
end
function _ker_key(ker::KIYukawa3D, p, r)
    return ( :yukawa, p, round(r; sigdigits=6), ker.κ )
end
function _ker_key(ker::KIHelmholtz3D, p, r)
    return ( :helmholtz, p, round(r; sigdigits=6), ker.zk )
end

function get_ki_precomp(ker::KIKernel, p::Int, r::Float64)
    key = hash(_ker_key(ker, p, r))
    return get!(_KI_CACHE, key) do
        precompute_ki(ker, p, r)
    end
end

# shift relative surface to box center
@inline function _shift_surf(rel::Vector{SVector{3,Float64}}, ctr::SVector{3,Float64})
    return [ctr + rel[i] for i in eachindex(rel)]
end

mutable struct KINodeData{T}
    up_equiv::Vector{T}
    dn_check::Vector{T}
    center::SVector{3,Float64}
    r::Float64  # half-width proxy
end

function _allocate_ki(root::ClusterTree{3}, n::Int, ::Type{T}) where {T}
    data = Dict{UInt,KINodeData{T}}()
    for node in nodes(root)
        ctr = SVector{3,Float64}(center(container(node)))
        r = max(diameter(node) / 2, 1e-30)
        data[objectid(node)] = KINodeData{T}(zeros(T, n), zeros(T, n), ctr, r)
    end
    return data
end

"""
```julia
vals = kifmm3d(sources, charges; kernel=KILaplace3D(), targets=nothing,
               p=6, nmax=40, η=1.0, pg=0, pgt=1)
```

Kernel-independent FMM (exafmm-t style) in 3D.

# Arguments
- `p`: surface discretization order (nsurf = 6(p-1)²+2). Typical 4–10.
- `kernel`: [`KILaplace3D`](@ref), [`KIYukawa3D`](@ref), or [`KIHelmholtz3D`](@ref)
"""
function kifmm3d(
    sources::AbstractMatrix{<:Real},
    charges::AbstractVector;
    kernel::KIKernel=KILaplace3D(),
    targets=nothing,
    p::Int=6,
    nmax::Int=40,
    η::Real=1.0,
    pg::Integer=0,
    pgt::Integer=1,
    threaded::Bool=false,
)
    @assert size(sources, 1) == 3
    @assert p >= 2
    ns = size(sources, 2)
    @assert pg > 0 || pgt > 0

    T = kernel isa KIHelmholtz3D ? ComplexF64 : Float64
    nsf = nsurf(p)
    adm = StrongAdmissibility(η=Float64(η))
    spl = GeometricSplitter(nmax=nmax)
    stree, sl2g, src = build_point_tree(sources, spl)
    ch = if T == Float64
        permute_to_local(vec(real.(charges)), sl2g)
    else
        permute_to_local_complex(vec(charges), sl2g)
    end

    sdata = _allocate_ki(stree, nsf, T)

    # ---- upward: P2M + M2M ----
    function upward!(node)
        ed = sdata[objectid(node)]
        pre = get_ki_precomp(kernel, p, ed.r)
        if isleaf(node)
            # P2M
            check = _shift_surf(pre.up_check, ed.center)
            φ = zeros(T, nsf)
            pot_p2p!(φ, src, ch, check, kernel; src_range=index_range(node))
            ed.up_equiv .= pre.UC2E * φ
        else
            chs = children(node)
            if threaded && length(chs) > 1
                Threads.@threads for c in chs
                    upward!(c)
                end
            else
                for c in chs
                    upward!(c)
                end
            end
            # M2M: children equiv → parent check → parent equiv
            φ = zeros(T, nsf)
            parent_check = _shift_surf(pre.up_check, ed.center)
            for c in chs
                cd = sdata[objectid(c)]
                pc = get_ki_precomp(kernel, p, cd.r)
                child_equiv_pts = _shift_surf(pc.up_equiv, cd.center)
                pot_p2p!(φ, child_equiv_pts, cd.up_equiv, parent_check, kernel)
            end
            ed.up_equiv .= pre.UC2E * φ
        end
    end
    upward!(stree)

    vals = FMMVals()

    function run(tmat, is_src)
        nt = size(tmat, 2)
        if is_src
            ttree, tl2g, tpts = stree, sl2g, src
            same = true
            tdata = sdata
        else
            ttree, tl2g, tpts = build_point_tree(tmat, spl)
            same = false
            tdata = _allocate_ki(ttree, nsf, T)
        end
        # zero dn_check on target tree
        for ed in values(tdata)
            fill!(ed.dn_check, 0)
        end

        pot_loc = zeros(T, nt)

        # ---- interaction: M2L + P2P ----
        function m2l_job!(tnode, snode, _, __)
            td = tdata[objectid(tnode)]
            sd = sdata[objectid(snode)]
            pt = get_ki_precomp(kernel, p, td.r)
            ps = get_ki_precomp(kernel, p, sd.r)
            trg_check = _shift_surf(pt.dn_check, td.center)
            src_equiv = _shift_surf(ps.up_equiv, sd.center)
            pot_p2p!(td.dn_check, src_equiv, sd.up_equiv, trg_check, kernel)
        end

        function p2p_job!(tnode, snode)
            for (jt, j) in enumerate(index_range(tnode))
                # accumulate into pot_loc directly
            end
            # use shared buffer approach
            idx = collect(index_range(tnode))
            buf = zeros(T, length(idx))
            pot_p2p!(
                buf, src, ch, tpts, kernel;
                exclude_self=same, src_range=index_range(snode), trg_range=idx,
            )
            for (jt, j) in enumerate(idx)
                pot_loc[j] += buf[jt]
            end
        end

        dummy_s = allocate_expdata(stree, 1)
        dummy_t = same ? dummy_s : allocate_expdata(ttree, 1)
        dualtree_interact!(
            ttree, stree, dummy_t, dummy_s, adm;
            m2l! = m2l_job!,
            p2p! = p2p_job!,
            leaf_mpole_only=false,
            threaded=threaded,
        )

        # ---- downward: L2L + L2P ----
        function downward!(node)
            ed = tdata[objectid(node)]
            pre = get_ki_precomp(kernel, p, ed.r)
            if isleaf(node)
                # L2P: dn_check → dn_equiv → targets
                σ = pre.DC2E * ed.dn_check
                equiv_pts = _shift_surf(pre.dn_equiv, ed.center)
                idx = collect(index_range(node))
                buf = zeros(T, length(idx))
                pot_p2p!(buf, equiv_pts, σ, tpts, kernel; trg_range=idx)
                for (jt, j) in enumerate(idx)
                    pot_loc[j] += buf[jt]
                end
            else
                # L2L to children
                # convert parent check → parent outer equiv, evaluate on child check
                σ_parent = pre.DC2E * ed.dn_check
                parent_equiv_pts = _shift_surf(pre.dn_equiv, ed.center)
                for c in children(node)
                    cd = tdata[objectid(c)]
                    pc = get_ki_precomp(kernel, p, cd.r)
                    child_check = _shift_surf(pc.dn_check, cd.center)
                    pot_p2p!(cd.dn_check, parent_equiv_pts, σ_parent, child_check, kernel)
                end
                chs = children(node)
                if threaded && length(chs) > 1
                    Threads.@threads for c in chs
                        downward!(c)
                    end
                else
                    for c in chs
                        downward!(c)
                    end
                end
            end
        end
        downward!(ttree)

        pot_g = zeros(T, nt)
        unpermute!(pot_g, pot_loc, tl2g)
        return pot_g
    end

    if pg > 0
        vals.pot = run(sources, true)
    end
    if pgt > 0 && targets !== nothing
        vals.pottarg = run(targets, false)
    end
    vals.ier = 0
    return vals
end
