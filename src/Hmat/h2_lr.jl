# Nested H² LR factorization on recursive [`H2Node`](@ref) (H2Lib `lrdecomp_h2matrix`)
#
# Algorithm (same recursion as H2Lib / classic H-LU):
#
#   for i = 1:sons
#       LR(Xᵢᵢ)
#       for j > i
#           Xᵢⱼ ← Lᵢᵢ \ Xᵢⱼ          (block forward substitution)
#           Xⱼᵢ ← Xⱼᵢ / Uᵢᵢ          (block back substitution)
#       for j > i, k > i
#           Xⱼₖ ← Xⱼₖ − Xⱼᵢ Xᵢₖ     (Schur / addmul)
#
# Dense diagonal leaves store standard unit-lower + upper (no pivoting).
# Off-diagonal blocks are densified when algebra requires it (uniform→dense),
# which is exact for the nested block pattern; recompression is optional.

const _H2_NOPIVOT = VERSION >= v"1.7" ? NoPivot : Val{false}

# ---- deep copy of factor tree ------------------------------------------------

"""
    h2_clone(N::H2Node) -> H2Node

Deep copy of a recursive H² tree (new `H2Pack` + copied leaf data). Used before
in-place [`lrdecomp_h2node!`](@ref).
"""
function h2_clone(N::H2Node{R, T}) where {R, T}
    p0 = N.pack
    pack = H2Pack{R, T}(
        p0.src,
        p0.tidx,
        [copy(U) for U in p0.U],
        Dict{Tuple{Int, Int}, Any}(k => (v isa AbstractMatrix ? copy(v) : deepcopy(v)) for (k, v) in p0.Bfar),
        Dict(k => copy(v) for (k, v) in p0.Ddiag),
        Dict(k => copy(v) for (k, v) in p0.Dnear),
        p0.n,
        copy(p0.rowperm),
        copy(p0.colperm),
    )
    return _h2_clone_node(N, pack)
end

function _h2_clone_node(N::H2Node{R, T}, pack::H2Pack{R, T}) where {R, T}
    if isdense_h2(N)
        return _h2node_dense(N.row_id, N.col_id, copy(N.F), pack)
    elseif isuniform(N)
        S = N.S isa RkMatrix ? RkMatrix(copy(N.S.A), copy(N.S.B)) :
            N.S isa AbstractMatrix ? copy(N.S) : deepcopy(N.S)
        return _h2node_uniform(N.row_id, N.col_id, S, pack; s_full = N.s_full)
    else
        rs, cs = size(N.sons)
        sons = Matrix{H2Node{R, T}}(undef, rs, cs)
        @inbounds for j in 1:cs, i in 1:rs
            sons[i, j] = _h2_clone_node(N.sons[i, j], pack)
        end
        return _h2node_split(N.row_id, N.col_id, sons, pack)
    end
end

# ---- materialize / densify ----------------------------------------------------

"""Dense matrix of a node in its local row/col index order (contiguous)."""
function h2_block_matrix(N::H2Node{R, T}) where {R, T}
    m, n = size(N)
    if isdense_h2(N)
        return copy(N.F)
    elseif isuniform(N)
        return _h2_uniform_dense(N)
    else
        M = zeros(T, m, n)
        roffs = _h2_son_row_offsets(N)
        coffs = _h2_son_col_offsets(N)
        rs, cs = size(N.sons)
        @inbounds for j in 1:cs, i in 1:rs
            Bij = h2_block_matrix(N.sons[i, j])
            mi, nj = size(Bij)
            ro, co = roffs[i], coffs[j]
            M[(ro + 1):(ro + mi), (co + 1):(co + nj)] .= Bij
        end
        return M
    end
end

function _h2_uniform_dense(N::H2Node{R, T}) where {R, T}
    Ir = rowrange(N)
    Jr = colrange(N)
    m, n = length(Ir), length(Jr)
    S = N.S
    if N.s_full
        return S isa RkMatrix ? Matrix(S) : Matrix{T}(S)
    end
    Vr = h2_basis_matrix(N.pack, N.row_id)
    Vc = h2_basis_matrix(N.pack, N.col_id)
    Sm = S isa RkMatrix ? Matrix(S) : Matrix{T}(S)
    if size(Sm, 1) == size(Vr, 2) && size(Sm, 2) == size(Vc, 2)
        return Vr * (Sm * adjoint(Vc))
    elseif size(Sm, 1) == size(Vr, 2) && size(Sm, 2) == n
        return Vr * Sm
    elseif size(Sm, 1) == m && size(Sm, 2) == size(Vc, 2)
        return Sm * adjoint(Vc)
    elseif size(Sm, 1) == m && size(Sm, 2) == n
        return Sm
    else
        return _h2_extract_dense(N.pack, N.row_id, N.col_id)
    end
end

function _h2_son_row_offsets(N::H2Node)
    rs = size(N.sons, 1)
    offs = Vector{Int}(undef, rs)
    o = 0
    @inbounds for i in 1:rs
        offs[i] = o
        o += size(N.sons[i, 1], 1)
    end
    return offs
end

function _h2_son_col_offsets(N::H2Node)
    cs = size(N.sons, 2)
    offs = Vector{Int}(undef, cs)
    o = 0
    @inbounds for j in 1:cs
        offs[j] = o
        o += size(N.sons[1, j], 2)
    end
    return offs
end

"""Replace node by a dense leaf with matrix `F` (same clusters)."""
function h2_replace_dense!(N::H2Node{R, T}, F::Matrix{T}) where {R, T}
    size(F) == size(N) || throw(DimensionMismatch("h2_replace_dense! size"))
    N.kind = H2DenseLeaf
    N.S = nothing
    N.F = F
    N.sons = Matrix{H2Node{R, T}}(undef, 0, 0)
    N.s_full = false
    return N
end

"""Densify any node to a dense leaf (in place)."""
function h2_densify!(N::H2Node)
    isdense_h2(N) && return N
    return h2_replace_dense!(N, h2_block_matrix(N))
end

# ---- block apply y += α A x on full vectors (ranges from A) -------------------

function h2_block_apply!(
        y::AbstractVector{T},
        A::H2Node{<:Any, T},
        x::AbstractVector{T},
        α::Number = one(T),
    ) where {T}
    if isdense_h2(A)
        mul!(view(y, rowrange(A)), A.F, view(x, colrange(A)), α, true)
    elseif isuniform(A)
        # temporary full apply via densify path for α ≠ 1
        if α == 1 || α == true
            _h2_apply_uniform!(y, A, x)
        else
            Fd = _h2_uniform_dense(A)
            mul!(view(y, rowrange(A)), Fd, view(x, colrange(A)), α, true)
        end
    else
        for s in A.sons
            h2_block_apply!(y, s, x, α)
        end
    end
    return y
end

# ---- nested LR ----------------------------------------------------------------

"""
    lrdecomp_h2node!(X::H2Node; rtol=1e-6) -> X

In-place nested LR factorization of a square recursive H² tree (H2Lib
`lrdecomp_h2matrix`). Overwrites `X` with unit-lower + upper factors in the
dense diagonal leaves (same convention as hierarchical `lu!` on `HMatrix`).

Prefer [`lrdecomp_h2node`](@ref) to clone first.
"""
function lrdecomp_h2node!(X::H2Node; rtol = 1e-6)
    size(X, 1) == size(X, 2) || throw(DimensionMismatch("lrdecomp_h2node! needs square block"))
    if isdense_h2(X)
        lu!(X.F, _H2_NOPIVOT())
        return X
    elseif isuniform(X)
        # Diagonal must not be uniform under strong admissibility
        throw(ArgumentError("lrdecomp_h2node!: uniform leaf on diagonal (row=$(X.row_id))"))
    else
        rs, cs = size(X.sons)
        rs == cs || throw(DimensionMismatch("lrdecomp_h2node!: non-square sons $(rs)×$(cs)"))
        for i in 1:rs
            lrdecomp_h2node!(X.sons[i, i]; rtol = rtol)
            for j in (i + 1):rs
                # Xᵢⱼ ← Lᵢᵢ \ Xᵢⱼ
                h2_ldiv_left!(X.sons[i, i], X.sons[i, j]; unit_diag = true)
                # Xⱼᵢ ← Xⱼᵢ / Uᵢᵢ
                h2_rdiv_right!(X.sons[j, i], X.sons[i, i]; unit_diag = false)
            end
            for j in (i + 1):rs, k in (i + 1):rs
                # Xⱼₖ ← Xⱼₖ − Xⱼᵢ Xᵢₖ
                h2_addmul!(X.sons[j, k], X.sons[j, i], X.sons[i, k], -1; rtol = rtol)
            end
        end
        return X
    end
end

"""
    lrdecomp_h2node(X::H2Node; kwargs...) -> H2NodeLU

Clone `X`, run [`lrdecomp_h2node!`](@ref), wrap as [`H2NodeLU`](@ref).
"""
function lrdecomp_h2node(X::H2Node; kwargs...)
    F = h2_clone(X)
    lrdecomp_h2node!(F; kwargs...)
    return H2NodeLU(F)
end

"""
    struct H2NodeLU

Nested LR factors stored as one recursive [`H2Node`](@ref) (unit lower + upper
in dense diagonal leaves), analogous to `LU` of an `HMatrix`.
"""
struct H2NodeLU{R, T}
    factors::H2Node{R, T}
end

Base.size(F::H2NodeLU) = size(F.factors)
Base.size(F::H2NodeLU, d::Integer) = size(F.factors, d)
Base.eltype(F::H2NodeLU) = eltype(F.factors)

function Base.show(io::IO, ::MIME"text/plain", F::H2NodeLU)
    return print(io, "H2NodeLU nested LR of $(F.factors)")
end

# ---- left solve L \\ B (B overwritten) ----------------------------------------

"""
    h2_ldiv_left!(L, B; unit_diag=true)

Overwrite `B` with `L \\ B`. `L` holds nested LR factors; only the unit-lower
(or lower) triangle is used.
"""
function h2_ldiv_left!(L::H2Node{R, T}, B::H2Node{R, T}; unit_diag::Bool = true) where {R, T}
    size(L, 1) == size(L, 2) || throw(DimensionMismatch())
    size(L, 1) == size(B, 1) || throw(DimensionMismatch("h2_ldiv_left! row mismatch"))

    if isdense_h2(L) && isdense_h2(B)
        if unit_diag
            ldiv!(UnitLowerTriangular(L.F), B.F)
        else
            ldiv!(LowerTriangular(L.F), B.F)
        end
        return B
    end

    if issplit(L) && issplit(B) && size(L.sons, 1) == size(B.sons, 1) &&
            size(L.sons, 2) == size(L.sons, 1)
        r = size(L.sons, 1)
        cB = size(B.sons, 2)
        for col in 1:cB
            for i in 1:r
                h2_ldiv_left!(L.sons[i, i], B.sons[i, col]; unit_diag = unit_diag)
                for j in (i + 1):r
                    h2_addmul!(B.sons[j, col], L.sons[j, i], B.sons[i, col], -one(T); rtol = 0.0)
                end
            end
        end
        return B
    end

    # densify fallback
    Ld = h2_block_matrix(L)
    Bd = h2_block_matrix(B)
    if unit_diag
        ldiv!(UnitLowerTriangular(Ld), Bd)
    else
        ldiv!(LowerTriangular(Ld), Bd)
    end
    return h2_replace_dense!(B, Bd)
end

# ---- right solve B / U (B overwritten) ----------------------------------------

"""
    h2_rdiv_right!(B, U; unit_diag=false)

Overwrite `B` with `B / U` (i.e. `rdiv!(B, UpperTriangular(U))`).
`U` is nested LR factors; upper triangle is used (`unit_diag` rare).
"""
function h2_rdiv_right!(B::H2Node{R, T}, U::H2Node{R, T}; unit_diag::Bool = false) where {R, T}
    size(U, 1) == size(U, 2) || throw(DimensionMismatch())
    size(B, 2) == size(U, 1) || throw(DimensionMismatch("h2_rdiv_right! col mismatch"))

    if isdense_h2(B) && isdense_h2(U)
        if unit_diag
            rdiv!(B.F, UnitUpperTriangular(U.F))
        else
            rdiv!(B.F, UpperTriangular(U.F))
        end
        return B
    end

    if issplit(B) && issplit(U) && size(U.sons, 1) == size(U.sons, 2) &&
            size(B.sons, 2) == size(U.sons, 1)
        r = size(U.sons, 1)
        rB = size(B.sons, 1)
        for row in 1:rB
            for i in 1:r
                h2_rdiv_right!(B.sons[row, i], U.sons[i, i]; unit_diag = unit_diag)
                for j in (i + 1):r
                    h2_addmul!(B.sons[row, j], B.sons[row, i], U.sons[i, j], -one(T); rtol = 0.0)
                end
            end
        end
        return B
    end

    Bd = h2_block_matrix(B)
    Ud = h2_block_matrix(U)
    if unit_diag
        rdiv!(Bd, UnitUpperTriangular(Ud))
    else
        rdiv!(Bd, UpperTriangular(Ud))
    end
    return h2_replace_dense!(B, Bd)
end

# ---- rkupdate: G ← G + X Y'  (H2Lib rkupdate_h2matrix MVP / Slice R1) ---------

"""
    h2_rkupdate!(G::H2Node, X, Y; rtol=1e-6, atol=0, rank=typemax(Int))

Low-rank update ``G ← G + X Y'`` staying in the recursive H² tree (H2Lib
`rkupdate_h2matrix` MVP).

# Arguments
- `X`: `pack.n × k` in **tree-local** ordering (same as `global_index=false` matvec)
- `Y`: `pack.n × k` likewise

# Leaf behaviour (R1)
- **dense**: `F += X[I,:] Y[J,:]'`
- **uniform**: form block dense sum, recompress with [`TSVD`](@ref); keep as
  full-block [`RkMatrix`](@ref) in `S` when cheaper than dense, else densify
- **split**: recurse on sons with the same global `X`,`Y`

Weighted nested-basis expansion (full Börm–Reimer) is Slice R2–R3; this MVP
already avoids always densifying the whole of `G` and preserves low-rank far
leaves when the update is compressible.
"""
function h2_rkupdate!(
        G::H2Node{R, T},
        X::AbstractMatrix,
        Y::AbstractMatrix;
        rtol = 1e-6,
        atol = 0.0,
        rank = typemax(Int),
    ) where {R, T}
    size(X, 2) == size(Y, 2) || throw(DimensionMismatch("h2_rkupdate!: X,Y ranks"))
    n = G.pack.n
    size(X, 1) == n && size(Y, 1) == n || throw(DimensionMismatch(
        "h2_rkupdate!: X,Y must be pack.n×k (got $(size(X)), $(size(Y)), n=$n)"))
    k = size(X, 2)
    k == 0 && return G

    if isdense_h2(G)
        Ir, Jr = rowrange(G), colrange(G)
        mul!(G.F, view(X, Ir, :), adjoint(view(Y, Jr, :)), true, true)
        return G
    elseif isuniform(G)
        return _h2_rkupdate_uniform!(G, X, Y; rtol, atol, rank)
    else
        for s in G.sons
            h2_rkupdate!(s, X, Y; rtol, atol, rank)
        end
        return G
    end
end

function h2_rkupdate!(
        G::H2Node,
        x::AbstractVector,
        y::AbstractVector;
        kwargs...,
    )
    return h2_rkupdate!(G, reshape(x, :, 1), reshape(y, :, 1); kwargs...)
end

function _h2_rkupdate_uniform!(
        G::H2Node{R, T},
        X::AbstractMatrix,
        Y::AbstractMatrix;
        rtol,
        atol,
        rank,
    ) where {R, T}
    Ir, Jr = rowrange(G), colrange(G)
    m, nblk = length(Ir), length(Jr)
    M = h2_block_matrix(G)
    mul!(M, view(X, Ir, :), adjoint(view(Y, Jr, :)), true, true)
    # Recompress when a tolerance is requested
    frtol = float(rtol)
    if frtol > 0 || atol > 0 || rank < typemax(Int)
        comp = TSVD(; rtol = frtol > 0 ? frtol : 0.0, atol = float(atol), rank = rank)
        Rk = compress!(copy(M), comp)
        rkeep = size(Rk.A, 2)
        if rkeep * (m + nblk) < m * nblk
            G.kind = H2UniformLeaf
            G.S = Rk
            G.F = nothing
            G.sons = Matrix{H2Node{R, T}}(undef, 0, 0)
            G.s_full = true   # full-block Rk, not nested coupling
            return G
        end
    end
    return h2_replace_dense!(G, M)
end

# ---- thin factors / apply for product → Rk ------------------------------------

"""Economy factors `M ≈ Lf Rf'` with `Lf::m×r`, `Rf::n×r` (local block)."""
function h2_node_lr_factors(N::H2Node{R, T}; rtol = 1e-12, rank = typemax(Int)) where {R, T}
    m, n = size(N)
    if isdense_h2(N)
        return _dense_lr_factors(N.F; rtol, rank)
    elseif isuniform(N)
        return _uniform_lr_factors(N; rtol, rank)
    else
        return _dense_lr_factors(h2_block_matrix(N); rtol, rank)
    end
end

function _dense_lr_factors(M::Matrix{T}; rtol = 1e-12, rank = typemax(Int)) where {T}
    m, n = size(M)
    (m == 0 || n == 0) && return (zeros(T, m, 0), zeros(T, n, 0))
    F = svd(M; full = false)
    τ = max(float(rtol) * (isempty(F.S) ? zero(T) : F.S[1]), zero(real(T)))
    r = count(s -> s > τ, F.S)
    r = max(r, 0)
    r = min(r, Int(rank), length(F.S))
    r == 0 && return (zeros(T, m, 0), zeros(T, n, 0))
    Lf = F.U[:, 1:r] * Diagonal(F.S[1:r])
    Rf = F.V[:, 1:r]
    return Lf, Rf
end

function _uniform_lr_factors(N::H2Node{R, T}; rtol = 1e-12, rank = typemax(Int)) where {R, T}
    m, n = size(N)
    S = N.S
    if N.s_full && S isa RkMatrix
        return copy(S.A), copy(S.B)
    elseif N.s_full
        return _dense_lr_factors(Matrix{T}(S); rtol, rank)
    end
    Vr = h2_basis_matrix(N.pack, N.row_id)
    Vc = h2_basis_matrix(N.pack, N.col_id)
    Sm = S isa RkMatrix ? Matrix(S) : Matrix{T}(S)
    if size(Sm, 1) == size(Vr, 2) && size(Sm, 2) == size(Vc, 2)
        # M = Vr Sm Vc' — factor via small SVD of Sm when possible
        if min(size(Sm)...) > 0
            Fs = svd(Sm; full = false)
            τ = max(float(rtol) * (isempty(Fs.S) ? zero(real(T)) : Fs.S[1]), zero(real(T)))
            r = count(s -> s > τ, Fs.S)
            r = min(max(r, 0), Int(rank), length(Fs.S))
            if r == 0
                return zeros(T, m, 0), zeros(T, n, 0)
            end
            Lf = Vr * (Fs.U[:, 1:r] * Diagonal(Fs.S[1:r]))
            Rf = Vc * Fs.V[:, 1:r]
            return Lf, Rf
        end
        return zeros(T, m, 0), zeros(T, n, 0)
    end
    return _dense_lr_factors(h2_block_matrix(N); rtol, rank)
end

"""Compute `N * M` for thin `M` with `size(M,1) == size(N,2)`."""
function h2_node_mul_thin(N::H2Node{R, T}, M::AbstractMatrix{T}) where {R, T}
    size(M, 1) == size(N, 2) || throw(DimensionMismatch("h2_node_mul_thin"))
    k = size(M, 2)
    k == 0 && return zeros(T, size(N, 1), 0)
    if isdense_h2(N)
        return N.F * M
    elseif isuniform(N) && N.s_full
        return N.S isa RkMatrix ? Matrix(N.S) * M : Matrix{T}(N.S) * M
    elseif isuniform(N)
        return h2_block_matrix(N) * M
    else
        out = zeros(T, size(N, 1), k)
        packn = N.pack.n
        for j in 1:k
            x = zeros(T, packn)
            x[colrange(N)] = view(M, :, j)
            y = zeros(T, packn)
            h2_block_apply!(y, N, x, one(T))
            out[:, j] = view(y, rowrange(N))
        end
        return out
    end
end

"""Compute `N' * M` for thin `M` with `size(M,1) == size(N,1)`."""
function h2_node_mul_thin_adj(N::H2Node{R, T}, M::AbstractMatrix{T}) where {R, T}
    size(M, 1) == size(N, 1) || throw(DimensionMismatch("h2_node_mul_thin_adj"))
    k = size(M, 2)
    k == 0 && return zeros(T, size(N, 2), 0)
    if isdense_h2(N)
        return adjoint(N.F) * M
    else
        # densify path (safe); could transpose-apply later
        return adjoint(h2_block_matrix(N)) * M
    end
end

"""
Build pack-global factors `X,Y` (`n×r`) with
`(A B) ≈ X[row(A),:] * Y[col(B),:]'` (scale `α`) for [`h2_rkupdate!`](@ref).
"""
function h2_product_to_global_rk(
        A::H2Node{R, T},
        B::H2Node{R, T},
        α::Number;
        rtol = 1e-12,
        rank = typemax(Int),
    ) where {R, T}
    pack = A.pack
    n = pack.n
    size(A, 2) == size(B, 1) || throw(DimensionMismatch("h2_product_to_global_rk middle"))

    # Prefer factoring the side that is already low-rank / thinner
    prefer_A = isuniform(A) || (isdense_h2(A) && size(A, 1) <= size(A, 2))
    prefer_B = isuniform(B) || (isdense_h2(B) && size(B, 2) <= size(B, 1))
    use_A = prefer_A || (!prefer_B && length(A) <= length(B))

    if use_A
        # A ≈ Lf Rf'  (mA×r, p×r).  A B = Lf (B' Rf)'
        Lf, Rf = h2_node_lr_factors(A; rtol = rtol, rank = rank)
        size(Rf, 1) == size(B, 1) || throw(DimensionMismatch("A factors vs B rows"))
        Zloc = h2_node_mul_thin_adj(B, Rf)   # nB × r
        r = size(Lf, 2)
        X = zeros(T, n, r)
        Y = zeros(T, n, r)
        if r > 0
            X[rowrange(A), :] = α .* Lf
            Y[colrange(B), :] = Zloc
        end
        return X, Y
    else
        # B ≈ Lf Rf'  (p×r, nB×r).  A B = (A Lf) Rf'
        Lf, Rf = h2_node_lr_factors(B; rtol = rtol, rank = rank)
        size(Lf, 1) == size(A, 2) || throw(DimensionMismatch("B factors vs A cols"))
        Wloc = h2_node_mul_thin(A, Lf)       # mA × r
        r = size(Rf, 2)
        X = zeros(T, n, r)
        Y = zeros(T, n, r)
        if r > 0
            X[rowrange(A), :] = α .* Wloc
            Y[colrange(B), :] = Rf
        end
        return X, Y
    end
end

# ---- addmul: C ← C + α A B ----------------------------------------------------

"""
    h2_addmul!(C, A, B, α=1; rtol=1e-6)

`C ← C + α*A*B` on recursive H² nodes (H2Lib `addmul_h2matrix`).

# Strategy
1. All dense → GEMM
2. Compatible splits → recurse on sons
3. Else form low-rank `X Y' ≈ α A B` and [`h2_rkupdate!`](@ref) into `C`
4. Last resort: densify GEMM into `C`
"""
function h2_addmul!(
        C::H2Node{R, T},
        A::H2Node{R, T},
        B::H2Node{R, T},
        α::Number = one(T);
        rtol = 1e-6,
        rank = typemax(Int),
    ) where {R, T}
    size(A, 2) == size(B, 1) || throw(DimensionMismatch("h2_addmul! inner"))
    size(C, 1) == size(A, 1) && size(C, 2) == size(B, 2) ||
        throw(DimensionMismatch("h2_addmul! outer"))

    if isdense_h2(C) && isdense_h2(A) && isdense_h2(B)
        mul!(C.F, A.F, B.F, α, true)
        return C
    end

    # compatible 2-level split: Cᵢₖ += α ∑ⱼ Aᵢⱼ Bⱼₖ
    if issplit(C) && issplit(A) && issplit(B) &&
            size(A.sons, 1) == size(C.sons, 1) &&
            size(B.sons, 2) == size(C.sons, 2) &&
            size(A.sons, 2) == size(B.sons, 1)
        rC, cC = size(C.sons)
        mid = size(A.sons, 2)
        for i in 1:rC, k in 1:cC
            for j in 1:mid
                h2_addmul!(C.sons[i, k], A.sons[i, j], B.sons[j, k], α; rtol = rtol, rank = rank)
            end
        end
        return C
    end

    # Low-rank-preserving path (H2Lib cases 1–5, 8)
    if isuniform(A) || isuniform(B) || isdense_h2(A) || isdense_h2(B) || isuniform(C)
        try
            X, Y = h2_product_to_global_rk(A, B, α; rtol = max(float(rtol), 1e-14), rank = rank)
            if size(X, 2) > 0
                return h2_rkupdate!(C, X, Y; rtol = rtol, rank = rank)
            end
        catch
            # fall through to densify
        end
    end

    Cd = h2_block_matrix(C)
    Ad = h2_block_matrix(A)
    Bd = h2_block_matrix(B)
    mul!(Cd, Ad, Bd, α, true)
    # If C was (or should stay) compressible, try rkwrite of the residual update only
    if float(rtol) > 0 && (isuniform(C) || !isdense_h2(C))
        # write full result via replace + optional compress as uniform
        m, nblk = size(Cd)
        comp = TSVD(; rtol = float(rtol), rank = rank)
        Rk = compress!(copy(Cd), comp)
        rkeep = size(Rk.A, 2)
        if rkeep * (m + nblk) < m * nblk
            C.kind = H2UniformLeaf
            C.S = Rk
            C.F = nothing
            C.sons = Matrix{H2Node{R, T}}(undef, 0, 0)
            C.s_full = true
            return C
        end
    end
    return h2_replace_dense!(C, Cd)
end

# ---- vector triangular solves on factors --------------------------------------

function h2_lsolve_vec!(LU::H2Node{R, T}, x::AbstractVector{T}; unit_diag::Bool = true) where {R, T}
    if isdense_h2(LU)
        ir = rowrange(LU)
        if unit_diag
            ldiv!(UnitLowerTriangular(LU.F), view(x, ir))
        else
            ldiv!(LowerTriangular(LU.F), view(x, ir))
        end
    elseif issplit(LU)
        n = size(LU.sons, 1)
        for i in 1:n
            h2_lsolve_vec!(LU.sons[i, i], x; unit_diag = unit_diag)
            for j in (i + 1):n
                # xⱼ -= Lⱼᵢ xᵢ
                h2_block_apply!(x, LU.sons[j, i], x, -one(T))
            end
        end
    else
        throw(ArgumentError("h2_lsolve_vec!: unexpected uniform on factor diagonal"))
    end
    return x
end

function h2_usolve_vec!(LU::H2Node{R, T}, x::AbstractVector{T}; unit_diag::Bool = false) where {R, T}
    if isdense_h2(LU)
        ir = rowrange(LU)
        if unit_diag
            ldiv!(UnitUpperTriangular(LU.F), view(x, ir))
        else
            ldiv!(UpperTriangular(LU.F), view(x, ir))
        end
    elseif issplit(LU)
        n = size(LU.sons, 1)
        for i in n:-1:1
            for j in (i + 1):n
                # xᵢ -= Uᵢⱼ xⱼ
                h2_block_apply!(x, LU.sons[i, j], x, -one(T))
            end
            h2_usolve_vec!(LU.sons[i, i], x; unit_diag = unit_diag)
        end
    else
        throw(ArgumentError("h2_usolve_vec!: unexpected uniform on factor diagonal"))
    end
    return x
end

function LinearAlgebra.ldiv!(F::H2NodeLU, y::AbstractVector; global_index = use_global_index())
    pack = F.factors.pack
    T = eltype(F)
    length(y) == pack.n || throw(DimensionMismatch())
    if global_index
        # cluster-local solve, then unpermute like H-LU: permute by col, inv by row
        permute!(y, pack.colperm)
        h2_lsolve_vec!(F.factors, y; unit_diag = true)
        h2_usolve_vec!(F.factors, y; unit_diag = false)
        invpermute!(y, pack.rowperm)
    else
        h2_lsolve_vec!(F.factors, y; unit_diag = true)
        h2_usolve_vec!(F.factors, y; unit_diag = false)
    end
    return y
end

function Base.:\(F::H2NodeLU, b::AbstractVector)
    x = copy(b)
    ldiv!(F, x)
    return x
end

function lrsolve_h2matrix(F::H2NodeLU, b::AbstractVector; global_index = true)
    x = copy(b)
    ldiv!(F, x; global_index = global_index)
    return x
end

# ---- public entry from flat H2Matrix ------------------------------------------

"""
    lrdecomp_h2matrix(H2; method=:nested, ...) 

When `method=:nested`, repackage to [`H2Node`](@ref) and run true nested LR
([`lrdecomp_h2node`](@ref)). Other methods keep the H²→H path.
"""
function lrdecomp_h2matrix_nested(H2::H2Matrix; rtol = 1e-6, kwargs...)
    root = h2_repackage(H2)
    F = lrdecomp_h2node(root; rtol = rtol)
    return (; L = nothing, U = nothing, F = F, node = F.factors, method = :nested)
end
