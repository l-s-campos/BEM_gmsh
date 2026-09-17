# Nested H² low-rank update (H2Lib rkupdate / Börm–Reimer R2–R3)
#
# G ← G + X Y′
#   R2: expand nested pack.U so leaves span range([X Y]), rebuild transfers,
#       rewrite far couplings S = Vi′ (Mold + Xi Yj′) Vj
#   R3: weighted recompress of bases (clusteroperator) + project S

"""
    struct H2ClusterOperator{R,T}

Weight tree for nested recompression (H2Lib `clusteroperator`).
`C[i]` acts on the rank of cluster `i`.
"""
struct H2ClusterOperator{R, T}
    tidx::H2TreeIndex{R}
    C::Vector{Matrix{T}}
end

"""
    prepare_h2_weights(pack; side=:row) -> H2ClusterOperator

Frobenius-normalized weights from nested far couplings (H2Lib
`prepare_row/col_clusteroperator` MVP).
"""
function prepare_h2_weights(pack::H2Pack{R, T}; side::Symbol = :row) where {R, T}
    side in (:row, :col) || throw(ArgumentError("side must be :row or :col"))
    tidx = pack.tidx
    nnode = length(tidx.nodes)
    C = [Matrix{T}(I, max(size(pack.U[i], 2), 0), max(size(pack.U[i], 2), 0)) for i in 1:nnode]
    touches = [Matrix{T}[] for _ in 1:nnode]

    for ((i, j), S) in pack.Bfar
        Sm = _h2_as_matrix(S)
        isempty(Sm) && continue
        nrm = norm(Sm)
        α = nrm > 0 ? (one(T) / nrm) : one(T)
        ri, rj = size(pack.U[i], 2), size(pack.U[j], 2)
        if side === :row
            size(Sm, 1) == ri && push!(touches[i], α .* Sm)
            size(Sm, 2) == rj && push!(touches[j], α .* transpose(Sm))
        else
            size(Sm, 2) == rj && push!(touches[j], α .* transpose(Sm))
            size(Sm, 1) == ri && push!(touches[i], α .* Sm)
        end
    end

    for i in 1:nnode
        r = size(pack.U[i], 2)
        r == 0 && continue
        mats = Matrix{T}[]
        for P in touches[i]
            size(P, 1) == r && push!(mats, transpose(P))
            size(P, 2) == r && push!(mats, P)
        end
        isempty(mats) && continue
        Yw = reduce(vcat, mats)
        size(Yw, 1) == 0 && continue
        F = qr(Yw)
        rr = min(size(F.R, 1), r)
        C[i] = Matrix(F.R[1:rr, 1:r])
    end
    return H2ClusterOperator{R, T}(tidx, C)
end

_h2_as_matrix(S::RkMatrix) = Matrix(S)
_h2_as_matrix(S::AbstractMatrix) = Matrix(S)
_h2_as_matrix(_) = zeros(Float64, 0, 0)

"""
    h2_rkupdate_nested!(G, X, Y; rtol=1e-6, recompress=true)

Nested-basis ``G ← G + XY'``: expand bases (R2) then optional weighted
recompress (R3). `X`,`Y` are `pack.n × k` in tree-local order.
"""
function h2_rkupdate_nested!(
        G::H2Node{R, T},
        X::AbstractMatrix,
        Y::AbstractMatrix;
        rtol = 1e-6,
        atol = 0.0,
        rank = typemax(Int),
        recompress::Bool = true,
    ) where {R, T}
    pack = G.pack
    n = pack.n
    size(X, 1) == n && size(Y, 1) == n || throw(DimensionMismatch("X,Y rows"))
    size(X, 2) == size(Y, 2) || throw(DimensionMismatch("X,Y cols"))
    size(X, 2) == 0 && return G

    old_blocks = Dict{Tuple{Int, Int}, Matrix{T}}()
    h2_foreach(G) do node
        isleaf_h2(node) || return
        old_blocks[(node.row_id, node.col_id)] = h2_block_matrix(node)
    end

    Z = X === Y ? Matrix{T}(X) : hcat(Matrix{T}(X), Matrix{T}(Y))
    _h2_expand_bases_xy!(pack, Z)
    h2_clear_basis_cache!(pack)

    h2_foreach(G) do node
        isleaf_h2(node) || return
        i, j = node.row_id, node.col_id
        Ir = rowrange(node)
        Jr = colrange(node)
        Mold = get(old_blocks, (i, j)) do
            zeros(T, length(Ir), length(Jr))
        end
        mul!(Mold, view(X, Ir, :), adjoint(view(Y, Jr, :)), true, true)

        if isdense_h2(node) || (i == j && haskey(pack.Ddiag, i))
            node.kind = H2DenseLeaf
            node.F = Mold
            node.S = nothing
            node.s_full = false
            if i == j
                pack.Ddiag[i] = Mold
            else
                pack.Dnear[(i, j)] = Mold
            end
            return
        end

        Vi = h2_basis_matrix(pack, i)
        Vj = h2_basis_matrix(pack, j)
        if size(Vi, 2) > 0 && size(Vj, 2) > 0 &&
                size(Vi, 1) == size(Mold, 1) && size(Vj, 1) == size(Mold, 2)
            Snew = adjoint(Vi) * (Mold * Vj)
            recon = Vi * (Snew * adjoint(Vj))
            rel = norm(recon - Mold) / max(norm(Mold), eps(real(T)))
            if rel <= max(10 * float(rtol), 1e-8)
                node.kind = H2UniformLeaf
                node.S = Snew
                node.F = nothing
                node.s_full = false
                pack.Bfar[(i, j)] = Snew
                return
            end
        end

        m, nblk = size(Mold)
        comp = TSVD(; rtol = max(float(rtol), 1e-14), atol = float(atol), rank = rank)
        Rk = compress!(copy(Mold), comp)
        if size(Rk.A, 2) * (m + nblk) < m * nblk
            node.kind = H2UniformLeaf
            node.S = Rk
            node.F = nothing
            node.s_full = true
            pack.Bfar[(i, j)] = Rk
        else
            node.kind = H2DenseLeaf
            node.F = Mold
            node.S = nothing
            node.s_full = false
            pack.Dnear[(i, j)] = Mold
        end
    end

    if recompress && float(rtol) > 0
        rw = prepare_h2_weights(pack; side = :row)
        Rproj = _h2_weighted_recompress!(pack, rw; rtol = float(rtol), atol = float(atol), rank = rank)
        h2_foreach(G) do node
            isuniform(node) && !node.s_full || return
            i, j = node.row_id, node.col_id
            haskey(pack.Bfar, (i, j)) || return
            pack.Bfar[(i, j)] = _h2_proj_S(pack.Bfar[(i, j)], Rproj[i], Rproj[j])
            node.S = pack.Bfar[(i, j)]
        end
        _h2_rebuild_transfers!(pack)
        h2_clear_basis_cache!(pack)
        h2_foreach(G) do node
            isuniform(node) && !node.s_full || return
            i, j = node.row_id, node.col_id
            haskey(pack.Bfar, (i, j)) || return
            node.S = pack.Bfar[(i, j)]
        end
    end
    return G
end

function _h2_proj_S(S, Ri::AbstractMatrix, Rj::AbstractMatrix)
    Sm = _h2_as_matrix(S)
    if size(Ri, 2) == size(Sm, 1) && size(Rj, 2) == size(Sm, 2)
        return Ri * Sm * adjoint(Rj)
    end
    return Sm
end

function _h2_expand_bases_xy!(pack::H2Pack{R, T}, Z::AbstractMatrix{T}) where {R, T}
    tidx = pack.tidx
    for node in tidx.leafnodes
        Ir = index_range(tidx.nodes[node])
        U = pack.U[node]
        A = size(U, 2) == 0 ? Z[Ir, :] : hcat(U, Z[Ir, :])
        A = _h2_drop_zero_cols(Matrix{T}(A))
        if size(A, 2) == 0
            pack.U[node] = zeros(T, length(Ir), 0)
            continue
        end
        F = qr(A)
        rn = max(_h2_qr_rank(F.R), 0)
        pack.U[node] = Matrix(F.Q)[:, 1:rn]
    end
    _h2_rebuild_transfers!(pack)
    return pack
end

function _h2_rebuild_transfers!(pack::H2Pack{R, T}) where {R, T}
    tidx = pack.tidx
    nlevel = length(tidx.levels)
    for lvl in nlevel:-1:1
        for node in tidx.levels[lvl]
            ch = tidx.children[node]
            isempty(ch) && continue
            Vch = Matrix{T}[h2_basis_matrix(pack, c) for c in ch]
            stacked = _h2_stack_bases(Vch)
            if size(stacked, 2) == 0
                pack.U[node] = zeros(T, 0, 0)
                continue
            end
            F = qr(stacked)
            rn = max(_h2_qr_rank(F.R), 0)
            Vp = Matrix(F.Q)[:, 1:rn]
            pack.U[node] = _h2_fit_transfer(Vch, Vp)
        end
    end
    return pack
end

function _h2_drop_zero_cols(A::Matrix{T}; tol = 1e-14) where {T}
    size(A, 2) == 0 && return A
    keep = Int[j for j in 1:size(A, 2) if norm(view(A, :, j)) > tol]
    isempty(keep) && return zeros(T, size(A, 1), 0)
    return A[:, keep]
end

function _h2_qr_rank(R::AbstractMatrix; tol = 1e-12)
    m, n = size(R)
    scale = zero(float(real(eltype(R))))
    @inbounds for j in 1:min(m, n)
        scale = max(scale, abs(R[j, j]))
    end
    τ = tol * max(scale, eps(typeof(scale)))
    rn = 0
    @inbounds for j in 1:min(m, n)
        abs(R[j, j]) > τ || break
        rn = j
    end
    return rn
end

function _h2_stack_bases(Vch::Vector{<:AbstractMatrix{T}}) where {T}
    isempty(Vch) && return zeros(T, 0, 0)
    return reduce(vcat, Vch)
end

function _h2_fit_transfer(Vch::Vector{<:AbstractMatrix{T}}, Vp::AbstractMatrix{T}) where {T}
    rs = Int[size(V, 2) for V in Vch]
    ms = Int[size(V, 1) for V in Vch]
    E = zeros(T, sum(rs; init = 0), size(Vp, 2))
    off_m = 0
    off_r = 0
    for (Vc, r, m) in zip(Vch, rs, ms)
        if r > 0 && m > 0 && size(Vp, 2) > 0
            E[(off_r + 1):(off_r + r), :] = adjoint(Vc) * Vp[(off_m + 1):(off_m + m), :]
        end
        off_m += m
        off_r += r
    end
    return E
end

function _h2_full_basis(pack::H2Pack{R, T}, i::Int) where {R, T}
    r = size(pack.U[i], 2)
    m = length(index_range(pack.tidx.nodes[i]))
    r == 0 && return zeros(T, m, 0)
    return h2_basis_matrix(pack, i)
end

function _h2_weighted_recompress!(
        pack::H2Pack{R, T},
        rw::H2ClusterOperator;
        rtol = 1e-6,
        atol = 0.0,
        rank = typemax(Int),
    ) where {R, T}
    tidx = pack.tidx
    nnode = length(tidx.nodes)
    Rproj = Matrix{T}[Matrix{T}(I, size(pack.U[i], 2), size(pack.U[i], 2)) for i in 1:nnode]
    Vmid = [_h2_full_basis(pack, i) for i in 1:nnode]

    for node in tidx.leafnodes
        U = pack.U[node]
        r0 = size(U, 2)
        r0 == 0 && continue
        W = rw.C[node]
        A = if size(W, 2) == r0 && size(W, 1) > 0
            U * adjoint(W)
        elseif size(W, 1) == r0 && size(W, 2) > 0
            U * W
        else
            U
        end
        size(A, 2) == 0 && continue
        F = svd(A; full = false)
        σ1 = isempty(F.S) ? zero(real(T)) : real(F.S[1])
        τ = max(float(atol), float(rtol) * σ1)
        rkeep = count(s -> s > τ, F.S)
        rkeep = min(max(rkeep, 1), Int(rank), length(F.S), r0)
        Q = F.U[:, 1:rkeep]
        B = Q' * U
        pack.U[node] = Q
        Rproj[node] = B
    end

    _h2_rebuild_transfers!(pack)

    for i in 1:nnode
        isempty(tidx.children[i]) && continue
        Vn = _h2_full_basis(pack, i)
        if size(Vmid[i], 2) > 0 && size(Vn, 2) > 0 && size(Vmid[i], 1) == size(Vn, 1)
            Rproj[i] = adjoint(Vn) * Vmid[i]
        else
            Rproj[i] = Matrix{T}(I, size(Vn, 2), size(Vn, 2))
        end
    end
    return Rproj
end
