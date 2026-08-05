# Nested basis orthogonalization and recompression for H2Matrix

"""
    h2_orthog!(H::H2Matrix)

Orthogonalize nested bases `U` bottom-up and project far couplings.

For each node (fine → coarse):
1. If non-leaf, left-multiply the transfer by `blkdiag(R_children)`.
2. Thin QR: `U = Q R`, store `Q` as the new generator and keep `R`.
3. Transform far blocks so `Q_i S_new Q_j' ≈ U_i_old S_old U_j_old'`.
"""
function h2_orthog!(H::H2Matrix{R, T}) where {R, T}
    tidx = H.tidx
    nnode = length(tidx.nodes)
    Rfact = [Matrix{T}(undef, 0, 0) for _ in 1:nnode]

    nlevel = length(tidx.levels)
    for lvl in nlevel:-1:1
        for node in tidx.levels[lvl]
            Un = H.U[node]
            (isempty(Un) || size(Un, 2) == 0) && continue
            ch = tidx.children[node]
            if !isempty(ch)
                Un = _h2_apply_child_R_left(Un, ch, Rfact, T)
            end
            m, r = size(Un)
            r == 0 && continue
            if m >= r
                F = qr(Un)
                H.U[node] = Matrix(F.Q)          # m×r
                Rfact[node] = Matrix(F.R)        # r×r
            else
                F = svd!(Matrix(Un))
                rr = count(s -> s > zero(T), F.S)
                rr = max(rr, 0)
                H.U[node] = F.U[:, 1:rr]
                Rfact[node] = Diagonal(F.S[1:rr]) * F.Vt[1:rr, :]
            end
        end
    end

    for (i, j) in H.far
        haskey(H.Bfar, (i, j)) || continue
        H.Bfar[(i, j)] = _h2_project_coupling(H.Bfar[(i, j)], Rfact[i], Rfact[j], tidx, i, j)
    end
    return H
end

function _h2_apply_child_R_left(Un::Matrix{T}, ch, Rfact, ::Type{T}) where {T}
    row_blocks = Matrix{T}[]
    off = 1
    for c in ch
        Rc = Rfact[c]
        rc = size(Rc, 1) == 0 ? 0 : size(Rc, 2)  # old child rank = cols of R before replace
        # After child processing, Rfact[c] is r×r with r = size(U[c],2) new = old
        # Row block height equals child's generator column count before parent update,
        # which equals size(Rfact[c], 2) if R was r_old×r_old and U_old was m×r_old.
        rc = isempty(Rc) ? 0 : size(Rc, 2)
        rc == 0 && continue
        off + rc - 1 > size(Un, 1) && break
        block = Un[off:(off + rc - 1), :]
        off += rc
        push!(row_blocks, isempty(Rc) ? block : Rc * block)
    end
    isempty(row_blocks) && return Un
    return reduce(vcat, row_blocks)
end

function _h2_project_coupling(B, Ri::AbstractMatrix, Rj::AbstractMatrix, tidx, i, j)
    B = B isa AbstractMatrix ? Matrix(B) : Matrix(B)
    di, dj = tidx.depth_of[i], tidx.depth_of[j]
    if di == dj
        if !isempty(Ri) && size(Ri, 2) == size(B, 1)
            B = Ri * B
        end
        if !isempty(Rj) && size(Rj, 2) == size(B, 2)
            B = B * adjoint(Rj)
        end
    elseif di > dj
        if !isempty(Ri) && size(Ri, 2) == size(B, 1)
            B = Ri * B
        end
    else
        if !isempty(Rj) && size(Rj, 2) == size(B, 2)
            B = B * adjoint(Rj)
        end
    end
    return B
end

"""
    h2_compress!(H::H2Matrix; rtol=1e-6, atol=0, rank=typemax(Int))

Recompress far couplings (dense → truncated SVD / Rk, or `compress!` on Rk).
Prefer running [`h2_orthog!`](@ref) first.

MVP: does not shrink nested `U` ranks (safe for matvec nesting); reduces far
block cost, which often dominates.
"""
function h2_compress!(
        H::H2Matrix{R, T};
        rtol = 1e-6,
        atol = 0.0,
        rank = typemax(Int),
    ) where {R, T}
    comp = TSVD(; rtol=float(rtol), atol=float(atol), rank=rank)
    for (key, B) in collect(H.Bfar)
        if B isa RkMatrix
            compress!(B, comp)
            H.Bfar[key] = B
        elseif B isa Matrix
            mb, nb = size(B)
            min(mb, nb) == 0 && continue
            # only recompress if clearly low-rank beneficial
            if min(mb, nb) >= 4
                Rk = compress!(copy(B), comp)
                if LinearAlgebra.rank(Rk) * (mb + nb) < mb * nb
                    H.Bfar[key] = Rk
                end
            end
        end
    end
    return H
end
