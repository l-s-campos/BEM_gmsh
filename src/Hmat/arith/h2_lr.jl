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
# Dense diagonal / near leaves store unit-lower + upper (no pivoting).
# Far (uniform) blocks stay nested: L\B and B/U Galerkin-project the solved
# cluster basis back onto pack.U (formatted H²). Nested uniform Schur is a
# k×k Gram update. Otherwise form a structured Rk of αAB and inject into C
# by the same projection (else local rkupdate). PartialACA of C+αAB is last
# resort only.

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
        Vector{Union{Nothing, Matrix{T}}}(nothing, length(p0.tidx.nodes)),
        Vector{Union{Nothing, Matrix{T}}}(nothing, length(p0.tidx.nodes)),
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
        size(N.F) == (m, n) || throw(DimensionMismatch(
            "dense leaf F $(size(N.F)) vs node $(m)×$(n) (row=$(N.row_id), col=$(N.col_id))"))
        return copy(N.F)
    elseif isuniform(N)
        return _h2_uniform_dense(N)
    else
        M = zeros(T, m, n)
        Ir0 = first(rowrange(N))
        Jr0 = first(colrange(N))
        rs, cs = size(N.sons)
        @inbounds for j in 1:cs, i in 1:rs
            s = N.sons[i, j]
            Bij = h2_block_matrix(s)
            ir, jr = rowrange(s), colrange(s)
            rows = (first(ir) - Ir0 + 1):(last(ir) - Ir0 + 1)
            cols = (first(jr) - Jr0 + 1):(last(jr) - Jr0 + 1)
            size(Bij) == (length(rows), length(cols)) || continue
            M[rows, cols] .= Bij
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
        # Do not pull the original src matrix during factorization (stale).
        return zeros(T, m, n)
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
        _h2_apply_uniform!(y, A, x, α)
    else
        for s in A.sons
            h2_block_apply!(y, s, x, α)
        end
    end
    return y
end

# ---- local (block-sized) H² algebra ------------------------------------------

function _h2_rel_rows(parent::H2Node, son::H2Node)
    i0 = first(rowrange(parent))
    ir = rowrange(son)
    return (first(ir) - i0 + 1):(last(ir) - i0 + 1)
end

function _h2_rel_cols(parent::H2Node, son::H2Node)
    j0 = first(colrange(parent))
    jr = colrange(son)
    return (first(jr) - j0 + 1):(last(jr) - j0 + 1)
end

"""Y += α A X with X,Y local to A (`size(X,1)==size(A,2)`, `size(Y,1)==size(A,1)`)."""
function h2_mul_local!(
        Y::AbstractMatrix{T},
        A::H2Node{<:Any, T},
        X::AbstractMatrix{T},
        α::Number = one(T),
    ) where {T}
    size(X, 1) == size(A, 2) || throw(DimensionMismatch("h2_mul_local! X"))
    size(Y, 1) == size(A, 1) && size(Y, 2) == size(X, 2) ||
        throw(DimensionMismatch("h2_mul_local! Y"))
    size(X, 2) == 0 && return Y
    if isdense_h2(A)
        mul!(Y, A.F, X, α, true)
    elseif isuniform(A)
        if A.s_full
            mul!(Y, A.S, X, α, true)
        else
            Vr = h2_basis_matrix(A.pack, A.row_id)
            Vc = h2_basis_matrix(A.pack, A.col_id)
            S = A.S
            Sm = S isa RkMatrix ? Matrix(S) : S
            if size(Sm, 1) == size(Vr, 2) && size(Sm, 2) == size(Vc, 2)
                mul!(Y, Vr, Sm * (Vc' * X), α, true)
            elseif size(Sm, 1) == size(Y, 1) && size(Sm, 2) == size(X, 1)
                mul!(Y, Sm, X, α, true)
            else
                mul!(Y, _h2_uniform_dense(A), X, α, true)
            end
        end
    else
        @inbounds for s in A.sons
            h2_mul_local!(
                view(Y, _h2_rel_rows(A, s), :),
                s,
                view(X, _h2_rel_cols(A, s), :),
                α,
            )
        end
    end
    return Y
end

"""Y += α A' X with X,Y local (`size(X,1)==size(A,1)`, `size(Y,1)==size(A,2)`)."""
function h2_mul_local_adj!(
        Y::AbstractMatrix{T},
        A::H2Node{<:Any, T},
        X::AbstractMatrix{T},
        α::Number = one(T),
    ) where {T}
    size(X, 1) == size(A, 1) || throw(DimensionMismatch("h2_mul_local_adj! X"))
    size(Y, 1) == size(A, 2) && size(Y, 2) == size(X, 2) ||
        throw(DimensionMismatch("h2_mul_local_adj! Y"))
    size(X, 2) == 0 && return Y
    if isdense_h2(A)
        mul!(Y, adjoint(A.F), X, α, true)
    elseif isuniform(A)
        if A.s_full
            mul!(Y, adjoint(A.S), X, α, true)
        else
            Vr = h2_basis_matrix(A.pack, A.row_id)
            Vc = h2_basis_matrix(A.pack, A.col_id)
            S = A.S
            Sm = S isa RkMatrix ? Matrix(S) : S
            if size(Sm, 1) == size(Vr, 2) && size(Sm, 2) == size(Vc, 2)
                mul!(Y, Vc, adjoint(Sm) * (Vr' * X), α, true)
            elseif size(Sm, 1) == size(X, 1) && size(Sm, 2) == size(Y, 1)
                mul!(Y, adjoint(Sm), X, α, true)
            else
                mul!(Y, adjoint(_h2_uniform_dense(A)), X, α, true)
            end
        end
    else
        @inbounds for s in A.sons
            h2_mul_local_adj!(
                view(Y, _h2_rel_cols(A, s), :),
                s,
                view(X, _h2_rel_rows(A, s), :),
                α,
            )
        end
    end
    return Y
end

"""In-place `L \\ X` for thin `X` with `size(X,1)==size(L,1)` (block rows)."""
function h2_ldiv_thin!(
        L::H2Node{R, T},
        X::AbstractMatrix{T};
        unit_diag::Bool = true,
    ) where {R, T}
    size(X, 1) == size(L, 1) || throw(DimensionMismatch("h2_ldiv_thin!"))
    size(X, 2) == 0 && return X
    if isdense_h2(L)
        if unit_diag
            ldiv!(UnitLowerTriangular(L.F), X)
        else
            ldiv!(LowerTriangular(L.F), X)
        end
    elseif issplit(L)
        n = size(L.sons, 1)
        for i in 1:n
            sii = L.sons[i, i]
            Xi = view(X, _h2_rel_rows(L, sii), :)
            h2_ldiv_thin!(sii, Xi; unit_diag = unit_diag)
            for j in (i + 1):n
                sji = L.sons[j, i]
                h2_mul_local!(view(X, _h2_rel_rows(L, sji), :), sji, Xi, -one(T))
            end
        end
    else
        throw(ArgumentError("h2_ldiv_thin!: uniform on factor diagonal"))
    end
    return X
end

"""In-place `Y ← U^{-T} Y` for thin `Y` with `size(Y,1)==size(U,1)`."""
function h2_usolve_thin_adj!(
        U::H2Node{R, T},
        Y::AbstractMatrix{T};
        unit_diag::Bool = false,
    ) where {R, T}
    size(Y, 1) == size(U, 1) || throw(DimensionMismatch("h2_usolve_thin_adj!"))
    size(Y, 2) == 0 && return Y
    if isdense_h2(U)
        if unit_diag
            ldiv!(UnitLowerTriangular(adjoint(U.F)), Y)
        else
            ldiv!(LowerTriangular(adjoint(U.F)), Y)
        end
    elseif issplit(U)
        n = size(U.sons, 1)
        for i in 1:n
            sii = U.sons[i, i]
            Yi = view(Y, _h2_rel_rows(U, sii), :)
            h2_usolve_thin_adj!(sii, Yi; unit_diag = unit_diag)
            for j in (i + 1):n
                sij = U.sons[i, j]
                h2_mul_local_adj!(view(Y, _h2_rel_cols(U, sij), :), sij, Yi, -one(T))
            end
        end
    else
        throw(ArgumentError("h2_usolve_thin_adj!: uniform on factor diagonal"))
    end
    return Y
end

_h2_as_coupling(S::RkMatrix) = Matrix(S)
_h2_as_coupling(S::AbstractMatrix) = Matrix(S)
_h2_as_coupling(_) = zeros(Float64, 0, 0)

_h2_span_tol(rtol) = max(10 * float(rtol), 1e-8)

"""Least-squares coeffs `V \\ X`. If `residual=true`, also return
`‖X - V Coef‖/‖X‖`."""
function _h2_span_coeffs(V::AbstractMatrix{T}, X::AbstractMatrix{T}; residual::Bool = false) where {T}
    size(V, 1) == size(X, 1) || return residual ? (nothing, Inf) : nothing
    k, r = size(V, 2), size(X, 2)
    if r == 0
        Z = zeros(T, k, 0)
        return residual ? (Z, 0.0) : Z
    end
    k == 0 && return residual ? (nothing, Inf) : nothing
    Coef = V \ X
    residual || return Coef
    nrm = norm(X)
    rel = nrm <= 0 ? 0.0 : norm(V * Coef - X) / nrm
    return Coef, rel
end

"""Replace uniform `G` by `A B'` projected onto nested `V` only when the
factors already live in those bases (keeps `ldiv` accurate)."""
function _h2_try_store_nested!(
        G::H2Node{R, T},
        A::AbstractMatrix,
        B::AbstractMatrix;
        rtol = 1e-6,
    ) where {R, T}
    isuniform(G) || return false
    Vr = h2_basis_matrix(G.pack, G.row_id)
    Vc = h2_basis_matrix(G.pack, G.col_id)
    size(A, 1) == size(Vr, 1) == size(G, 1) || return false
    size(B, 1) == size(Vc, 1) == size(G, 2) || return false
    size(A, 2) == size(B, 2) || return false
    Xr, relX = _h2_span_coeffs(Vr, Matrix{T}(A); residual = true)
    Yc, relY = _h2_span_coeffs(Vc, Matrix{T}(B); residual = true)
    Xr === nothing && return false
    τ = _h2_span_tol(rtol)
    (relX <= τ && relY <= τ) || return false
    G.S = Xr * adjoint(Yc)
    G.F = nothing
    G.sons = Matrix{H2Node{R, T}}(undef, 0, 0)
    G.s_full = false
    return true
end

"""`C.S += (V_row \\ X)(V_col \\ Y)'` when `XY'` already lives in the nested
bases. Returns `false` if the span residual exceeds the formatted tolerance."""
function _h2_add_rk_nested!(
        C::H2Node{R, T},
        X::AbstractMatrix{T},
        Y::AbstractMatrix{T};
        rtol = 1e-6,
    ) where {R, T}
    isuniform(C) && !C.s_full || return false
    size(X, 2) == size(Y, 2) || return false
    size(X, 2) == 0 && return true
    Vr = h2_basis_matrix(C.pack, C.row_id)
    Vc = h2_basis_matrix(C.pack, C.col_id)
    size(X, 1) == size(Vr, 1) == size(C, 1) || return false
    size(Y, 1) == size(Vc, 1) == size(C, 2) || return false
    Xr, relX = _h2_span_coeffs(Vr, X; residual = true)
    Yc, relY = _h2_span_coeffs(Vc, Y; residual = true)
    (Xr === nothing || Yc === nothing) && return false
    τ = _h2_span_tol(rtol)
    (relX <= τ && relY <= τ) || return false
    dS = Xr * adjoint(Yc)
    Sc = _h2_as_coupling(C.S)
    if size(Sc) == size(dS)
        C.S = Sc + dS
    elseif isempty(Sc)
        C.S = dS
    else
        return false
    end
    C.F = nothing
    C.s_full = false
    return true
end

"""Inject `XY'` into `C`: nested project if it fits, else local rkupdate."""
function _h2_inject_rk!(
        C::H2Node{R, T},
        X::AbstractMatrix{T},
        Y::AbstractMatrix{T};
        rtol = 1e-6,
        rank = typemax(Int),
    ) where {R, T}
    size(X, 2) == 0 && return C
    if _h2_add_rk_nested!(C, X, Y; rtol = rtol)
        return C
    end
    return h2_rkupdate_local!(C, X, Y; rtol = rtol, rank = rank)
end

function _h2_store_rk!(
        G::H2Node{R, T},
        A::AbstractMatrix{T},
        B::AbstractMatrix{T};
        rtol = 1e-6,
        rank = typemax(Int),
    ) where {R, T}
    m, n = size(G)
    r0 = size(A, 2)
    if r0 == 0
        return h2_replace_dense!(G, zeros(T, m, n))
    end
    if _h2_try_store_nested!(G, A, B; rtol = rtol)
        return G
    end
    Af = Matrix{T}(A)
    Bf = Matrix{T}(B)
    frtol = float(rtol)
    if (frtol > 0 || rank < typemax(Int)) && r0 > 0
        Rk = RkMatrix(Af, Bf)
        compress!(Rk, TSVD(; rtol = frtol > 0 ? frtol : 0.0, rank = Int(rank)))
        Af, Bf = Rk.A, Rk.B
    end
    r = size(Af, 2)
    if r > 0 && r * (m + n) < m * n
        G.kind = H2UniformLeaf
        G.S = RkMatrix(Af, Bf)
        G.F = nothing
        G.sons = Matrix{H2Node{R, T}}(undef, 0, 0)
        G.s_full = true
        return G
    end
    return h2_replace_dense!(G, Af * adjoint(Bf))
end

"""`G ← G + X Y'` with `X,Y` local to `G` (not pack.n). Does not touch `pack.U`."""
function h2_rkupdate_local!(
        G::H2Node{R, T},
        X::AbstractMatrix{T},
        Y::AbstractMatrix{T};
        rtol = 1e-6,
        rank = typemax(Int),
    ) where {R, T}
    size(X, 1) == size(G, 1) || throw(DimensionMismatch("h2_rkupdate_local! X rows"))
    size(Y, 1) == size(G, 2) || throw(DimensionMismatch("h2_rkupdate_local! Y rows"))
    size(X, 2) == size(Y, 2) || throw(DimensionMismatch("h2_rkupdate_local! rank"))
    size(X, 2) == 0 && return G
    if isdense_h2(G)
        mul!(G.F, X, adjoint(Y), true, true)
        return G
    elseif issplit(G)
        @inbounds for s in G.sons
            h2_rkupdate_local!(
                s,
                view(X, _h2_rel_rows(G, s), :),
                view(Y, _h2_rel_cols(G, s), :);
                rtol = rtol,
                rank = rank,
            )
        end
        return G
    else
        if G.s_full && G.S isa RkMatrix
            A = hcat(G.S.A, X)
            B = hcat(G.S.B, Y)
        else
            Lf, Rf = _uniform_lr_factors(G; rtol = 0.0)
            A = size(Lf, 2) == 0 ? Matrix{T}(X) : hcat(Lf, X)
            B = size(Rf, 2) == 0 ? Matrix{T}(Y) : hcat(Rf, Y)
        end
        return _h2_store_rk!(G, A, B; rtol = rtol, rank = rank)
    end
end

function h2_add_dense!(
        C::H2Node{R, T},
        P::AbstractMatrix{T};
        rtol = 1e-6,
        rank = typemax(Int),
    ) where {R, T}
    size(P) == size(C) || throw(DimensionMismatch("h2_add_dense!"))
    if isdense_h2(C)
        C.F .+= P
        return C
    elseif issplit(C)
        @inbounds for s in C.sons
            h2_add_dense!(
                s,
                view(P, _h2_rel_rows(C, s), _h2_rel_cols(C, s));
                rtol = rtol,
                rank = rank,
            )
        end
        return C
    else
        M = _h2_uniform_dense(C)
        M .+= P
        m, n = size(M)
        frtol = float(rtol)
        if frtol > 0 || rank < typemax(Int)
            Rk = compress!(copy(M), TSVD(; rtol = frtol > 0 ? frtol : 0.0, rank = Int(rank)))
            if size(Rk.A, 2) * (m + n) < m * n
                C.kind = H2UniformLeaf
                C.S = Rk
                C.F = nothing
                C.sons = Matrix{H2Node{R, T}}(undef, 0, 0)
                C.s_full = true
                return C
            end
        end
        return h2_replace_dense!(C, M)
    end
end

"""Thin Rk factors `N ≈ Lf Rf'` without densifying a split node."""
function _h2_rk_factors(N::H2Node{R, T}) where {R, T}
    if isdense_h2(N)
        m, n = size(N.F)
        if n <= m
            return N.F, Matrix{T}(I, n, n)
        else
            return Matrix{T}(I, m, m), collect(adjoint(N.F))
        end
    elseif isuniform(N) && N.s_full && N.S isa RkMatrix
        return N.S.A, N.S.B
    elseif isuniform(N)
        return _uniform_lr_factors(N; rtol = 0.0)
    else
        throw(ArgumentError("_h2_rk_factors: split node"))
    end
end

"""QR-SVD truncate of thin factors `X Y'`."""
function _h2_compress_factors(
        X::AbstractMatrix{T},
        Y::AbstractMatrix{T},
        rtol,
        rank,
    ) where {T}
    r = size(X, 2)
    r == 0 && return Matrix{T}(X), Matrix{T}(Y)
    frtol = float(rtol)
    (frtol > 0 || Int(rank) < r) || return Matrix{T}(X), Matrix{T}(Y)
    Rk = RkMatrix(Matrix{T}(X), Matrix{T}(Y))
    compress!(Rk, TSVD(; rtol = frtol > 0 ? frtol : 0.0, rank = Int(rank)))
    return Rk.A, Rk.B
end

"""Local `α A B ≈ X Y'` with `X::size(A,1)×r`, `Y::size(B,2)×r`."""
function h2_product_local_rk(
        A::H2Node{R, T},
        B::H2Node{R, T},
        α::Number;
        rtol = 0.0,
        rank = typemax(Int),
    ) where {R, T}
    if isuniform(A) && A.s_full && A.S isa RkMatrix &&
            isuniform(B) && B.s_full && B.S isa RkMatrix &&
            size(A.S.B, 1) == size(B.S.A, 1)
        Mid = adjoint(A.S.B) * B.S.A
        return A.S.A * (α * Mid), B.S.B
    end
    # Nested uniforms, matching mid cluster: AB = V_r (S_A G S_B) V_c'
    if isuniform(A) && !A.s_full && isuniform(B) && !B.s_full && A.col_id == B.row_id
        Vr = h2_basis_matrix(A.pack, A.row_id)
        Vc = h2_basis_matrix(B.pack, B.col_id)
        Gmid = h2_gram_matrix(A.pack, A.col_id)
        Sa = _h2_as_coupling(A.S)
        Sb = _h2_as_coupling(B.S)
        if size(Sa, 2) == size(Gmid, 1) && size(Sb, 1) == size(Gmid, 2) &&
                size(Sa, 1) == size(Vr, 2) && size(Sb, 2) == size(Vc, 2)
            Mid = Sa * (Gmid * Sb)
            return Vr * (α * Mid), copy(Vc)
        end
    end
    if issplit(A) && issplit(B) && size(A.sons, 2) == size(B.sons, 1)
        return _h2_product_split_rk(A, B, α; rtol = rtol, rank = rank)
    end
    if issplit(A) && !issplit(B)
        Xb, Yb = _h2_rk_factors(B)
        r = size(Xb, 2)
        X = zeros(T, size(A, 1), r)
        r > 0 && h2_mul_local!(X, A, Xb, α)
        return X, Yb
    elseif !issplit(A)
        Xa, Ya = _h2_rk_factors(A)
        r = size(Ya, 2)
        Y = zeros(T, size(B, 2), r)
        r > 0 && h2_mul_local_adj!(Y, B, Ya, one(T))
        return α == 1 || α == true ? Xa : α .* Xa, Y
    else
        # A split, B leaf-like already handled; densify B's left factors
        Xb, Yb = _dense_lr_factors(h2_block_matrix(B); rtol = 0.0)
        r = size(Xb, 2)
        X = zeros(T, size(A, 1), r)
        r > 0 && h2_mul_local!(X, A, Xb, α)
        return X, Yb
    end
end

"""`α A B ≈ X Y'` when both `A` and `B` are split (H2Lib `mul_h2matrix_rkmatrix`)."""
function _h2_product_split_rk(
        A::H2Node{R, T},
        B::H2Node{R, T},
        α::Number;
        rtol = 0.0,
        rank = typemax(Int),
    ) where {R, T}
    mid = size(A.sons, 2)
    rA, cB = size(A.sons, 1), size(B.sons, 2)
    m, n = size(A, 1), size(B, 2)
    i0, j0 = first(rowrange(A)), first(colrange(B))
    Xacc = zeros(T, m, 0)
    Yacc = zeros(T, n, 0)
    cap = 64
    @inbounds for j in 1:mid, ia in 1:rA, ib in 1:cB
        X, Y = h2_product_local_rk(A.sons[ia, j], B.sons[j, ib], α; rtol = rtol, rank = rank)
        r = size(X, 2)
        r == 0 && continue
        Xf = zeros(T, m, r)
        Yf = zeros(T, n, r)
        ir, jr = rowrange(A.sons[ia, j]), colrange(B.sons[j, ib])
        rows = (first(ir) - i0 + 1):(last(ir) - i0 + 1)
        cols = (first(jr) - j0 + 1):(last(jr) - j0 + 1)
        size(X, 1) == length(rows) && size(Y, 1) == length(cols) || continue
        Xf[rows, :] = X
        Yf[cols, :] = Y
        Xacc = hcat(Xacc, Xf)
        Yacc = hcat(Yacc, Yf)
        if size(Xacc, 2) > cap
            Xacc, Yacc = _h2_compress_factors(Xacc, Yacc, rtol, rank)
        end
    end
    return _h2_compress_factors(Xacc, Yacc, rtol, rank)
end

# ---- ACA of virtual C + αAB (H-matrix MulLinearOp analogue) -------------------

"""Column `j` of a local H² block, written into `col` (size `size(N,1)`)."""
function h2_getcol!(col::AbstractVector{T}, N::H2Node{<:Any, T}, j::Int) where {T}
    length(col) == size(N, 1) || throw(DimensionMismatch("h2_getcol!"))
    fill!(col, zero(T))
    if isdense_h2(N)
        @inbounds col .= view(N.F, :, j)
    elseif isuniform(N)
        if N.s_full
            getcol!(col, N.S, j)
        else
            Vr = h2_basis_matrix(N.pack, N.row_id)
            Vc = h2_basis_matrix(N.pack, N.col_id)
            S = N.S
            Sm = S isa RkMatrix ? Matrix(S) : S
            if size(Sm, 1) == size(Vr, 2) && size(Sm, 2) == size(Vc, 2) &&
                    1 <= j <= size(Vc, 1)
                mul!(col, Vr, Sm * view(Vc, j, :))
            else
                x = zeros(T, size(N, 2))
                x[j] = one(T)
                h2_mul_local!(reshape(col, :, 1), N, reshape(x, :, 1), one(T))
            end
        end
    else
        @inbounds for s in N.sons
            cols = _h2_rel_cols(N, s)
            j in cols || continue
            jl = j - first(cols) + 1
            rows = _h2_rel_rows(N, s)
            h2_getcol!(view(col, rows), s, jl)
        end
    end
    return col
end

"""Row `i` of a local H² block, written into `row` (size `size(N,2)`)."""
function h2_getrow!(row::AbstractVector{T}, N::H2Node{<:Any, T}, i::Int) where {T}
    length(row) == size(N, 2) || throw(DimensionMismatch("h2_getrow!"))
    fill!(row, zero(T))
    if isdense_h2(N)
        @inbounds row .= view(N.F, i, :)
    elseif isuniform(N)
        if N.s_full
            getcol!(row, adjoint(N.S), i)
        else
            y = zeros(T, size(N, 1))
            y[i] = one(T)
            h2_mul_local_adj!(reshape(row, :, 1), N, reshape(y, :, 1), one(T))
        end
    else
        @inbounds for s in N.sons
            rows = _h2_rel_rows(N, s)
            i in rows || continue
            il = i - first(rows) + 1
            cols = _h2_rel_cols(N, s)
            h2_getrow!(view(row, cols), s, il)
        end
    end
    return row
end

"""
Virtual `C + α A B + X Y'` for PartialACA (H-matrix `MulLinearOp`).
`getindex` is disabled; ACA uses [`getblock!`](@ref) / columns.
"""
struct H2MulOp{T} <: AbstractMatrix{T}
    C::Union{H2Node, Nothing}
    A::Union{H2Node, Nothing}
    B::Union{H2Node, Nothing}
    α::T
    X::Union{AbstractMatrix{T}, Nothing}
    Y::Union{AbstractMatrix{T}, Nothing}
    m::Int
    n::Int
end

function H2MulOp(C::H2Node{R, T}, A::H2Node{R, T}, B::H2Node{R, T}, α::Number) where {R, T}
    return H2MulOp{T}(C, A, B, T(α), nothing, nothing, size(C, 1), size(C, 2))
end

Base.size(L::H2MulOp) = (L.m, L.n)
Base.getindex(::H2MulOp, args...) = error("getindex(::H2MulOp) is disabled; use getblock!")

function _h2mulop_col!(col::AbstractVector{T}, L::H2MulOp{T}, j::Int) where {T}
    fill!(col, zero(T))
    if L.C !== nothing
        h2_getcol!(col, L.C, j)
    end
    if L.A !== nothing && L.B !== nothing
        tmp = zeros(T, size(L.B, 1))
        h2_getcol!(tmp, L.B, j)
        h2_mul_local!(reshape(col, :, 1), L.A, reshape(tmp, :, 1), L.α)
    end
    if L.X !== nothing && L.Y !== nothing
        mul!(col, L.X, view(L.Y, j, :), true, true)
    end
    return col
end

function _h2mulop_row!(row::AbstractVector{T}, L::H2MulOp{T}, i::Int) where {T}
    fill!(row, zero(T))
    if L.C !== nothing
        h2_getrow!(row, L.C, i)
    end
    if L.A !== nothing && L.B !== nothing
        tmp = zeros(T, size(L.A, 2))
        h2_getrow!(tmp, L.A, i)
        h2_mul_local_adj!(reshape(row, :, 1), L.B, reshape(tmp, :, 1), L.α)
    end
    if L.X !== nothing && L.Y !== nothing
        mul!(row, L.Y, view(L.X, i, :), true, true)
    end
    return row
end

function getblock!(out, L::H2MulOp, irange, j::Int)
    if irange == 1:size(L, 1) || irange == axes(L, 1)
        return _h2mulop_col!(out, L, j)
    end
    tmp = zeros(eltype(L), size(L, 1))
    _h2mulop_col!(tmp, L, j)
    @inbounds for (iloc, i) in enumerate(irange)
        out[iloc] = tmp[i]
    end
    return out
end

function getblock!(out, Lt::Adjoint{<:Any, <:H2MulOp}, irange, j::Int)
    L = parent(Lt)
    if irange == 1:size(L, 2) || irange == axes(L, 2)
        return _h2mulop_row!(out, L, j)
    end
    tmp = zeros(eltype(L), size(L, 2))
    _h2mulop_row!(tmp, L, j)
    @inbounds for (iloc, i) in enumerate(irange)
        out[iloc] = tmp[i]
    end
    return out
end

function _h2_default_compressor(rtol, rank)
    return PartialACA(; rtol = float(rtol) > 0 ? float(rtol) : sqrt(eps(Float64)),
        rank = Int(rank))
end

"""Overwrite `C` by PartialACA of the virtual `C + α A B`."""
function _h2_aca_addmul!(
        C::H2Node{R, T},
        A::H2Node{R, T},
        B::H2Node{R, T},
        α::Number;
        rtol = 1e-6,
        rank = typemax(Int),
        compressor = nothing,
    ) where {R, T}
    m, n = size(C)
    (m == 0 || n == 0) && return C
    comp = compressor === nothing ? _h2_default_compressor(rtol, rank) : compressor
    op = H2MulOp(C, A, B, α)
    Rk = comp(op)
    r = size(Rk.A, 2)
    if r > 0 && r * (m + n) < m * n
        C.kind = H2UniformLeaf
        C.S = Rk
        C.F = nothing
        C.sons = Matrix{H2Node{R, T}}(undef, 0, 0)
        C.s_full = true
        return C
    end
    return h2_replace_dense!(C, Matrix(Rk))
end

"""Exact nested Schur: `C += α A B` with all three nested uniforms, k×k Gram."""
function _h2_addmul_nested_uniform!(
        C::H2Node{R, T},
        A::H2Node{R, T},
        B::H2Node{R, T},
        α::Number,
    ) where {R, T}
    (isuniform(A) && isuniform(B) && isuniform(C)) || return false
    (A.s_full || B.s_full || C.s_full) && return false
    A.col_id == B.row_id || return false
    Gmid = h2_gram_matrix(A.pack, A.col_id)
    Sa = _h2_as_coupling(A.S)
    Sb = _h2_as_coupling(B.S)
    Sc = _h2_as_coupling(C.S)
    (size(Sa, 2) == size(Gmid, 1) && size(Sb, 1) == size(Gmid, 2)) || return false
    Mid = Sa * (Gmid * Sb)
    if C.row_id == A.row_id && C.col_id == B.col_id &&
            size(Sc, 1) == size(Mid, 1) && size(Sc, 2) == size(Mid, 2)
        C.S = Sc + α * Mid
        return true
    end
    # Different nested bases, same index range: S_C += E_r (α Mid) E_c'
    VrC = h2_basis_matrix(C.pack, C.row_id)
    VcC = h2_basis_matrix(C.pack, C.col_id)
    VrA = C.row_id == A.row_id ? VrC : h2_basis_matrix(A.pack, A.row_id)
    VcB = C.col_id == B.col_id ? VcC : h2_basis_matrix(B.pack, B.col_id)
    size(VrA, 2) == size(Mid, 1) && size(VcB, 2) == size(Mid, 2) || return false
    size(VrC, 1) == size(VrA, 1) && size(VcC, 1) == size(VcB, 1) || return false
    Er = _h2_span_coeffs(VrC, VrA)
    Ec = _h2_span_coeffs(VcC, VcB)
    (Er === nothing || Ec === nothing) && return false
    dS = Er * (α * Mid) * adjoint(Ec)
    size(Sc) == size(dS) || return false
    C.S = Sc + dS
    return true
end

# ---- nested LR ----------------------------------------------------------------

"""
    lrdecomp_h2node!(X::H2Node; rtol=1e-6) -> X

In-place nested LR factorization of a square recursive H² tree (H2Lib
`lrdecomp_h2matrix`). Overwrites `X` with unit-lower + upper factors in the
dense diagonal leaves (same convention as hierarchical `lu!` on `HMatrix`).

Prefer [`lrdecomp_h2node`](@ref) to clone first.
"""
function lrdecomp_h2node!(X::H2Node; rtol = 1e-6, compressor = nothing)
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
            lrdecomp_h2node!(X.sons[i, i]; rtol = rtol, compressor = compressor)
            for j in (i + 1):rs
                h2_ldiv_left!(
                    X.sons[i, i], X.sons[i, j];
                    unit_diag = true, rtol = rtol, compressor = compressor,
                )
                h2_rdiv_right!(
                    X.sons[j, i], X.sons[i, i];
                    unit_diag = false, rtol = rtol, compressor = compressor,
                )
            end
            for j in (i + 1):rs, k in (i + 1):rs
                h2_addmul!(
                    X.sons[j, k], X.sons[j, i], X.sons[i, k], -1;
                    rtol = rtol, schur_rk = true, compressor = compressor,
                )
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

"""`B ← L \\ B` for nested uniform `B = Vr S Vc'`: solve `L \\ Vr`, Galerkin
project onto `Vr`, `S ← T S`. Returns `false` if `B` is not nested uniform."""
function _h2_ldiv_nested_uniform!(
        L::H2Node{R, T},
        B::H2Node{R, T};
        unit_diag::Bool,
        rtol = 1e-6,
    ) where {R, T}
    isuniform(B) && !B.s_full || return false
    Vr = h2_basis_matrix(B.pack, B.row_id)
    Vc = h2_basis_matrix(B.pack, B.col_id)
    S = _h2_as_coupling(B.S)
    size(Vr, 1) == size(B, 1) == size(L, 1) || return false
    size(S, 1) == size(Vr, 2) && size(S, 2) == size(Vc, 2) || return false
    size(Vr, 2) == 0 && return false
    Xf = copy(Vr)
    h2_ldiv_thin!(L, Xf; unit_diag = unit_diag)
    Tr, rel = _h2_span_coeffs(Vr, Xf; residual = true)
    B.F = nothing
    B.sons = Matrix{H2Node{R, T}}(undef, 0, 0)
    if Tr !== nothing && size(Tr, 2) == size(S, 1) && rel <= _h2_span_tol(rtol)
        B.S = Tr * S
        B.s_full = false
        return true
    end
    # Exact thin Rk: (L\Vr) S Vc' — no SVD of S, column basis stays nested.
    B.S = RkMatrix(Xf * S, Matrix{T}(Vc))
    B.s_full = true
    return true
end

"""`B ← B / U` for nested uniform `B = Vr S Vc'`: solve `U^{-T} Vc`, Galerkin
project onto `Vc`, `S ← S T'`."""
function _h2_rdiv_nested_uniform!(
        B::H2Node{R, T},
        U::H2Node{R, T};
        unit_diag::Bool,
        rtol = 1e-6,
    ) where {R, T}
    isuniform(B) && !B.s_full || return false
    Vr = h2_basis_matrix(B.pack, B.row_id)
    Vc = h2_basis_matrix(B.pack, B.col_id)
    S = _h2_as_coupling(B.S)
    size(Vc, 1) == size(B, 2) == size(U, 1) || return false
    size(S, 1) == size(Vr, 2) && size(S, 2) == size(Vc, 2) || return false
    size(Vc, 2) == 0 && return false
    Yf = copy(Vc)
    h2_usolve_thin_adj!(U, Yf; unit_diag = unit_diag)
    Tr, rel = _h2_span_coeffs(Vc, Yf; residual = true)
    B.F = nothing
    B.sons = Matrix{H2Node{R, T}}(undef, 0, 0)
    if Tr !== nothing && size(Tr, 1) == size(S, 2) && rel <= _h2_span_tol(rtol)
        B.S = S * adjoint(Tr)
        B.s_full = false
        return true
    end
    # Exact thin Rk: Vr S (U^{-T} Vc)'
    B.S = RkMatrix(Vr * S, Matrix{T}(Yf))
    B.s_full = true
    return true
end

# ---- left solve L \\ B (B overwritten) ----------------------------------------

"""
    h2_ldiv_left!(L, B; unit_diag=true)

Overwrite `B` with `L \\ B`. `L` holds nested LR factors; only the unit-lower
(or lower) triangle is used.
"""
function h2_ldiv_left!(
        L::H2Node{R, T},
        B::H2Node{R, T};
        unit_diag::Bool = true,
        rtol = 1e-6,
        compressor = nothing,
    ) where {R, T}
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
                h2_ldiv_left!(
                    L.sons[i, i], B.sons[i, col];
                    unit_diag = unit_diag, rtol = rtol, compressor = compressor,
                )
                for j in (i + 1):r
                    h2_addmul!(
                        B.sons[j, col], L.sons[j, i], B.sons[i, col], -one(T);
                        rtol = rtol, schur_rk = true, compressor = compressor,
                    )
                end
            end
        end
        return B
    end

    # Nested uniform: L \ (Vr S Vc') = (L \ Vr) S Vc', then Galerkin
    # project L\Vr back onto Vr so S stays a nested coupling.
    if _h2_ldiv_nested_uniform!(L, B; unit_diag = unit_diag, rtol = rtol)
        return B
    end

    # Thin factor path: L \ (Lf Rf') = (L \ Lf) Rf' — never densify L
    if isuniform(B) || isdense_h2(B)
        Lf, Rf = _h2_rk_factors(B)
        Xf = copy(Lf)
        h2_ldiv_thin!(L, Xf; unit_diag = unit_diag)
        return _h2_store_rk!(B, Xf, Rf; rtol = rtol)
    end

    Bd = h2_block_matrix(B)
    h2_ldiv_thin!(L, Bd; unit_diag = unit_diag)
    return h2_replace_dense!(B, Bd)
end

# ---- right solve B / U (B overwritten) ----------------------------------------

"""
    h2_rdiv_right!(B, U; unit_diag=false)

Overwrite `B` with `B / U` (i.e. `rdiv!(B, UpperTriangular(U))`).
`U` is nested LR factors; upper triangle is used (`unit_diag` rare).
"""
function h2_rdiv_right!(
        B::H2Node{R, T},
        U::H2Node{R, T};
        unit_diag::Bool = false,
        rtol = 1e-6,
        compressor = nothing,
    ) where {R, T}
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
                h2_rdiv_right!(
                    B.sons[row, i], U.sons[i, i];
                    unit_diag = unit_diag, rtol = rtol, compressor = compressor,
                )
                for j in (i + 1):r
                    h2_addmul!(
                        B.sons[row, j], B.sons[row, i], U.sons[i, j], -one(T);
                        rtol = rtol, schur_rk = true, compressor = compressor,
                    )
                end
            end
        end
        return B
    end

    # Nested uniform: (Vr S Vc') / U = Vr S (U^{-T} Vc)', Galerkin onto Vc.
    if _h2_rdiv_nested_uniform!(B, U; unit_diag = unit_diag, rtol = rtol)
        return B
    end

    # Thin factor path: (Lf Rf') / U = Lf (U^{-T} Rf)' — never densify U
    if isuniform(B) || isdense_h2(B)
        Lf, Rf = _h2_rk_factors(B)
        Yf = copy(Rf)
        h2_usolve_thin_adj!(U, Yf; unit_diag = unit_diag)
        return _h2_store_rk!(B, Lf, Yf; rtol = rtol)
    end

    Bd = h2_block_matrix(B)
    Yf = collect(adjoint(Bd))
    h2_usolve_thin_adj!(U, Yf; unit_diag = unit_diag)
    return h2_replace_dense!(B, collect(adjoint(Yf)))
end

# ---- rkupdate: G ← G + X Y'  (H2Lib rkupdate_h2matrix MVP / Slice R1) ---------

"""
    h2_rkupdate!(G::H2Node, X, Y; rtol=1e-6, method=:nested, ...)

Low-rank update ``G ← G + X Y'`` on a recursive H² tree (H2Lib `rkupdate_h2matrix`).

# Arguments
- `X`,`Y`: `pack.n × k` in **tree-local** ordering (`global_index=false`)

# Methods
- `:nested` (default) — expand nested `pack.U`, rewrite couplings, weighted
  recompress ([`h2_rkupdate_nested!`](@ref), Börm–Reimer R2–R3)
- `:block` — leaf-wise densify + TSVD (R1)
"""
function h2_rkupdate!(
        G::H2Node{R, T},
        X::AbstractMatrix,
        Y::AbstractMatrix;
        rtol = 1e-6,
        atol = 0.0,
        rank = typemax(Int),
        method::Symbol = :nested,
    ) where {R, T}
    size(X, 2) == size(Y, 2) || throw(DimensionMismatch("h2_rkupdate!: X,Y ranks"))
    n = G.pack.n
    size(X, 1) == n && size(Y, 1) == n || throw(DimensionMismatch(
        "h2_rkupdate!: X,Y must be pack.n×k (got $(size(X)), $(size(Y)), n=$n)"))
    k = size(X, 2)
    k == 0 && return G

    if method === :nested
        try
            return h2_rkupdate_nested!(G, X, Y; rtol, atol, rank)
        catch
            # fall back to block path
            method = :block
        end
    end
    method === :block || throw(ArgumentError("h2_rkupdate! method must be :nested or :block"))

    if isdense_h2(G)
        Ir, Jr = rowrange(G), colrange(G)
        mul!(G.F, view(X, Ir, :), adjoint(view(Y, Jr, :)), true, true)
        return G
    elseif isuniform(G)
        return _h2_rkupdate_uniform!(G, X, Y; rtol, atol, rank)
    else
        for s in G.sons
            h2_rkupdate!(s, X, Y; rtol, atol, rank, method = :block)
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
        rtol = 0.0,
        rank = typemax(Int),
    ) where {R, T}
    pack = A.pack
    n = pack.n
    size(A, 2) == size(B, 1) || throw(DimensionMismatch("h2_product_to_global_rk middle"))
    p = size(A, 2)
    p == 0 && return zeros(T, n, 0), zeros(T, n, 0)

    # Exact rank-p product A*B without SVD of the operands (SVD round-trip on
    # large dense blocks is the rtol=0 residual floor). Truncation happens later
    # in h2_rkupdate! when rtol>0.
    prefer_A = isuniform(A) || (isdense_h2(A) && size(A, 1) <= p) ||
        (!isuniform(B) && length(A) <= length(B))
    if prefer_A
        Lf, Rf = _h2_exact_left_factors(A)
        size(Rf, 1) == size(B, 1) || throw(DimensionMismatch("A factors vs B rows"))
        Zloc = h2_node_mul_thin_adj(B, Rf)
        r = min(size(Lf, 2), Int(rank))
        X = zeros(T, n, r)
        Y = zeros(T, n, r)
        if r > 0
            X[rowrange(A), :] = α .* view(Lf, :, 1:r)
            Y[colrange(B), :] = view(Zloc, :, 1:r)
        end
        return X, Y
    else
        Lf, Rf = _h2_exact_left_factors(B)
        size(Lf, 1) == size(A, 2) || throw(DimensionMismatch("B factors vs A cols"))
        Wloc = h2_node_mul_thin(A, Lf)
        r = min(size(Rf, 2), Int(rank))
        X = zeros(T, n, r)
        Y = zeros(T, n, r)
        if r > 0
            X[rowrange(A), :] = α .* view(Wloc, :, 1:r)
            Y[colrange(B), :] = view(Rf, :, 1:r)
        end
        return X, Y
    end
end

"""Exact `M ≈ Lf Rf'` with no truncation (`Rf = I` or existing Rk factors)."""
function _h2_exact_left_factors(N::H2Node{R, T}) where {R, T}
    if isdense_h2(N)
        m, p = size(N.F)
        return N.F, Matrix{T}(I, p, p)
    elseif isuniform(N) && N.s_full && N.S isa RkMatrix
        return copy(N.S.A), copy(N.S.B)
    elseif isuniform(N) && N.s_full
        M = Matrix{T}(N.S)
        return M, Matrix{T}(I, size(M, 2), size(M, 2))
    elseif isuniform(N)
        return _uniform_lr_factors(N; rtol = 0.0)
    else
        M = h2_block_matrix(N)
        return M, Matrix{T}(I, size(M, 2), size(M, 2))
    end
end

# ---- addmul: C ← C + α A B ----------------------------------------------------

"""
    h2_addmul!(C, A, B, α=1; rtol=1e-6)

`C ← C + α*A*B` on recursive H² nodes (H2Lib `addmul_h2matrix`).

# Strategy
1. All dense → GEMM
2. Nested uniforms with matching clusters → k×k Gram (`S_C += α S_A G S_B`)
3. Compatible splits → recurse on sons
4. Structured `X Y' ≈ α A B` (nested apply / thin H²×k) then inject:
   project onto `C`'s nested bases if the product lives there, else local
   rkupdate (H2Lib `convert_uniform_rkmatrix` + `rkupdate_h2matrix`)
5. Last resort: PartialACA of the virtual `C+αAB`, or densify GEMM
"""
function h2_addmul!(
        C::H2Node{R, T},
        A::H2Node{R, T},
        B::H2Node{R, T},
        α::Number = one(T);
        rtol = 1e-6,
        rank = typemax(Int),
        schur_rk::Bool = true,
        compressor = nothing,
    ) where {R, T}
    size(A, 2) == size(B, 1) || throw(DimensionMismatch("h2_addmul! inner"))
    size(C, 1) == size(A, 1) && size(C, 2) == size(B, 2) ||
        throw(DimensionMismatch("h2_addmul! outer"))

    if isdense_h2(C) && isdense_h2(A) && isdense_h2(B)
        mul!(C.F, A.F, B.F, α, true)
        return C
    end

    # Nested uniforms: C.S += α Sa G Sb  (k×k, exact)
    if schur_rk && _h2_addmul_nested_uniform!(C, A, B, α)
        return C
    end

    # Compatible splits: Cᵢₖ += α ∑ⱼ Aᵢⱼ Bⱼₖ
    if issplit(C) && issplit(A) && issplit(B) &&
            size(A.sons, 1) == size(C.sons, 1) &&
            size(B.sons, 2) == size(C.sons, 2) &&
            size(A.sons, 2) == size(B.sons, 1)
        rC, cC = size(C.sons)
        mid = size(A.sons, 2)
        for i in 1:rC, k in 1:cC
            for j in 1:mid
                h2_addmul!(
                    C.sons[i, k], A.sons[i, j], B.sons[j, k], α;
                    rtol = rtol, rank = rank, schur_rk = schur_rk,
                    compressor = compressor,
                )
            end
        end
        return C
    end

    # Structured Rk of αAB + inject (H2Lib cases 1–2, 5, 8). ACA is last resort.
    if schur_rk
        X, Y = h2_product_local_rk(A, B, α; rtol = rtol, rank = rank)
        if size(X, 1) == size(C, 1) && size(Y, 1) == size(C, 2)
            return _h2_inject_rk!(C, X, Y; rtol = rtol, rank = rank)
        end
        if float(rtol) > 0 && isuniform(C)
            return _h2_aca_addmul!(C, A, B, α; rtol = rtol, rank = rank, compressor = compressor)
        end
    end

    if isdense_h2(A) && isdense_h2(B)
        return h2_add_dense!(C, α * (A.F * B.F); rtol = rtol, rank = rank)
    end

    P = α * (h2_block_matrix(A) * h2_block_matrix(B))
    return h2_add_dense!(C, P; rtol = rtol, rank = rank)
end

function _h2_addmul_split_split!(
        C::H2Node{R, T},
        A::H2Node{R, T},
        B::H2Node{R, T},
        α::Number;
        rtol,
        rank,
    ) where {R, T}
    X, Y = _h2_product_split_rk(A, B, α; rtol = rtol, rank = rank)
    return _h2_inject_rk!(C, X, Y; rtol = rtol, rank = rank)
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
        xtmp = similar(x)
        for i in 1:n
            h2_lsolve_vec!(LU.sons[i, i], x; unit_diag = unit_diag)
            # Snapshot xᵢ so later Lⱼᵢ applies do not alias dest/src.
            fill!(xtmp, zero(T))
            iri = colrange(LU.sons[i, i])
            copyto!(view(xtmp, iri), view(x, iri))
            for j in (i + 1):n
                h2_block_apply!(x, LU.sons[j, i], xtmp, -one(T))
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
        xtmp = similar(x)
        for i in n:-1:1
            fill!(xtmp, zero(T))
            for j in (i + 1):n
                jr = colrange(LU.sons[i, j])
                copyto!(view(xtmp, jr), view(x, jr))
            end
            for j in (i + 1):n
                h2_block_apply!(x, LU.sons[i, j], xtmp, -one(T))
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

function LinearAlgebra.ldiv!(y::AbstractVector, F::H2NodeLU, x::AbstractVector;
        global_index = use_global_index())
    y === x || copyto!(y, x)
    return ldiv!(F, y; global_index = global_index)
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

"""
    lu(A::NNCAMatrix; method=:nested, rtol=1e-6)

Nested H² LR (H2Lib `lrdecomp_h2matrix`) on a square scalar NNCA operator.
Keeps the recursive block tree; does **not** convert to an H-matrix.

Triangular solves on nested uniforms apply `L` to the cluster basis `V`
(k columns). If `L\\V` still lives in `span(V)`, the coupling stays nested
and Schur is a k×k Gram update; otherwise the exact thin Rk `(L\\V) S V_c'`
is stored (`s_full`). Other products form a structured Rk and inject by
nested project or local rkupdate. PartialACA of `C+αAB` is last resort.
`method=:hmatrix` is the expanded H-LU fallback.
Pass `compressor=PartialACA(; rtol=...)` or `TSVD(; rtol=...)` to override.
"""
function LinearAlgebra.lu(
        A::NNCAMatrix;
        method::Symbol = :nested,
        rtol = 1e-6,
        compressor = nothing,
        kwargs...,
    )
    if method === :hmatrix
        return lu(hmatrix(A); kwargs...)
    end
    method === :nested || throw(ArgumentError("lu(::NNCAMatrix) method must be :nested or :hmatrix"))
    return lrdecomp_h2node(h2node(A); rtol = rtol, compressor = compressor)
end
