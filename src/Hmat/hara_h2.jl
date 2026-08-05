# Nested H² HARA: build H2Matrix from matvecs only (no kernel entries / proxies)

"""
    hara_h2(S, tree; kwargs...) -> H2Matrix

**Nested-H² HARA**: construct an [`H2Matrix`](@ref) using only multi-RHS matvecs
of the sampler `S` (no entry evaluation, no geometric proxies).

# Algorithm (MVP)
1. Cluster tree → [`H2TreeIndex`](@ref) + box far/near lists (`H2BoxAdmissibility`).
2. **Nested bases (bottom-up):** sample `Y = A*Ω` once; for each cluster take
   rows on candidate indices (full leaf range or stacked child skeletons),
   row-ID ([`_row_id`](@ref)) → generator `U` + skeleton.
3. **Nearfield:** dense blocks by sampling identity columns on the cluster.
4. **Far couplings:** skeleton–skeleton (or mixed) blocks by sampling identity
   on column skeletons / full ranges (same layout as [`assemble_h2`](@ref)).
5. Optional [`h2_orthog!`](@ref) / [`h2_compress!`](@ref).

# Keywords
- `alpha=0.5` — box admissibility parameter
- `rtol`, `rank` — ID / rank caps for nested bases
- `nsample=64` — number of global random probes for basis construction
- `global_index=true` — `S` in exterior ordering; trees define permutation
- `symmetric=true` — symmetric far/near lists (square geometric H²)
- `orthog=true`, `compress=true` — post-process bases / far blocks
- `adm` — override admissibility predicate

See also [`hara`](@ref) (classic H leaves) and [`assemble_h2`](@ref) (proxy/entry build).
"""
function hara_h2(
        S::AbstractMatvecSampler,
        tree::R;
        alpha = 0.5,
        rtol = 1e-4,
        rank = typemax(Int),
        nsample = 64,
        global_index = use_global_index(),
        symmetric = true,
        orthog = true,
        compress = true,
        adm = nothing,
    ) where {R}
    T = Float64
    α = Float64(alpha)
    adm_fun = isnothing(adm) ? H2BoxAdmissibility(α) : adm
    n = length(tree)
    size(S, 1) == n && size(S, 2) == n ||
        throw(DimensionMismatch("hara_h2 expects square sampler matching tree length $n"))

    rp = loc2glob(tree)
    Sloc = _hara_local_sampler(S, rp, rp, global_index)
    tidx = H2TreeIndex(tree)
    nnode = length(tidx.nodes)

    # --- nested bases from one multi-RHS sample Y = A*Ω ---
    ns = Int(min(n, max(Int(nsample), 16)))
    rmax = min(Int(rank), n)
    Ω = randn(T, n, ns)
    Yfull = zeros(T, n, ns)
    mul!(Yfull, Sloc, Ω)

    U = [zeros(T, 0, 0) for _ in 1:nnode]
    skeleton = [Int[] for _ in 1:nnode]
    for i in tidx.leafnodes
        skeleton[i] = collect(index_range(tidx.nodes[i]))
    end

    nlevel = length(tidx.levels)
    for lvl in nlevel:-1:1
        for node in tidx.levels[lvl]
            if isempty(tidx.children[node])
                cand = skeleton[node]
            else
                cand = reduce(vcat, (skeleton[c] for c in tidx.children[node]); init = Int[])
            end
            isempty(cand) && continue
            A_sam = Yfull[cand, :]
            rcap = min(rmax, size(A_sam, 1), size(A_sam, 2))
            if rcap == 0
                U[node] = zeros(T, length(cand), 0)
                skeleton[node] = Int[]
                continue
            end
            basis, Jloc = _row_id(A_sam, float(rtol), rcap)
            U[node] = Matrix(basis)
            skeleton[node] = isempty(Jloc) ? Int[] : cand[Jloc]
        end
    end

    near, far = _h2_block_partition(tidx, adm_fun; symmetric)
    minlvl = isempty(far) ? nlevel :
             minimum(min(tidx.depth_of[i], tidx.depth_of[j]) for (i, j) in far) + 1

    # --- nearfield dense ---
    Ddiag = Dict{Int, Matrix{T}}()
    Dnear = Dict{Tuple{Int, Int}, Matrix{T}}()
    for i in tidx.leafnodes
        ir = index_range(tidx.nodes[i])
        Ddiag[i] = _dense_block_from_sampler(Sloc, ir, ir)
    end
    for (i, j) in near
        ir = index_range(tidx.nodes[i])
        jr = index_range(tidx.nodes[j])
        Dnear[(i, j)] = _dense_block_from_sampler(Sloc, ir, jr)
    end

    # --- far couplings (skeleton layout matches assemble_h2) ---
    Bfar = Dict{Tuple{Int, Int}, Any}()
    for (i, j) in far
        di, dj = tidx.depth_of[i], tidx.depth_of[j]
        if di == dj
            Ii, Ij = skeleton[i], skeleton[j]
        elseif di > dj
            Ii = skeleton[i]
            Ij = collect(index_range(tidx.nodes[j]))
        else
            Ii = collect(index_range(tidx.nodes[i]))
            Ij = skeleton[j]
        end
        if isempty(Ii) || isempty(Ij)
            Bfar[(i, j)] = zeros(T, length(Ii), length(Ij))
        else
            Bfar[(i, j)] = _sample_index_block(Sloc, Ii, Ij)
        end
    end

    H2 = H2Matrix{R, T}(
        tidx, U, skeleton, near, far, Ddiag, Dnear, Bfar,
        copy(rp), copy(rp), minlvl, α, n, nothing,
    )
    orthog && h2_orthog!(H2)
    compress && h2_compress!(H2; rtol=float(rtol), rank=rmax)
    return H2
end

function hara_h2(K::AbstractMatrix, tree; kwargs...)
    return hara_h2(KernelMatvecSampler(K), tree; kwargs...)
end

"""Wrap global sampler into tree-local ordering (same as classic `hara`)."""
function _hara_local_sampler(S::AbstractMatvecSampler, rp::Vector{Int}, cp::Vector{Int}, global_index::Bool)
    global_index || return S
    f! = (Y, X) -> begin
        Xg = zeros(eltype(X), size(S, 2), size(X, 2))
        Xg[cp, :] .= X
        Yg = zeros(eltype(Y), size(S, 1), size(X, 2))
        mul!(Yg, S, Xg)
        Y .= Yg[rp, :]
        return Y
    end
    f_adj! = (Y, X) -> begin
        Xg = zeros(eltype(X), size(S, 1), size(X, 2))
        Xg[rp, :] .= X
        Yg = zeros(eltype(Y), size(S, 2), size(X, 2))
        mul!(Yg, adjoint(S), Xg)
        Y .= Yg[cp, :]
        return Y
    end
    return FunctionSampler(f!, size(S, 2); m = size(S, 1), f_adj! = f_adj!)
end

"""`A[Irows, Jcols]` via multi-RHS matvec with identity on `Jcols` (local order)."""
function _sample_index_block(S::AbstractMatvecSampler, Irows::Vector{Int}, Jcols::Vector{Int})
    T = Float64
    n = size(S, 2)
    m = size(S, 1)
    k = length(Jcols)
    X = zeros(T, n, k)
    @inbounds for (t, j) in enumerate(Jcols)
        X[j, t] = one(T)
    end
    Y = zeros(T, m, k)
    mul!(Y, S, X)
    return Y[Irows, :]
end

"""
    hara(S, tree; format=:H2, kwargs...)

Convenience: `format=:H` → classic [`hara`](@ref) (needs `coltree=tree`);
`format=:H2` → [`hara_h2`](@ref).
"""
function hara(
        S::AbstractMatvecSampler,
        tree::R;
        format::Symbol = :H2,
        kwargs...,
    ) where {R}
    if format === :H2 || format === :h2
        return hara_h2(S, tree; kwargs...)
    elseif format === :H || format === :h
        return hara(S, tree, tree; kwargs...)
    else
        throw(ArgumentError("hara format must be :H or :H2, got $(repr(format))"))
    end
end
