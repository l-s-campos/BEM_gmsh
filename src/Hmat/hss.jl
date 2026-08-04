# =============================================================================
# HSSBasisID
# =============================================================================

"""
    struct HSSBasisID{T}

Interpolative-decomposition basis: tall `m × r` factor with skeleton rows
`P[1:r]` and extension `E` (`(m-r) × r`), i.e. `U = P * [I; E]`.
"""
struct HSSBasisID{T}
    P::Vector{Int}
    E::Matrix{T}
end

HSSBasisID{T}(m::Integer, r::Integer) where {T} =
    HSSBasisID{T}(collect(1:Int(m)), zeros(T, max(Int(m) - Int(r), 0), Int(r)))

Base.eltype(::HSSBasisID{T}) where {T} = T
nrows(B::HSSBasisID) = length(B.P)
ncols(B::HSSBasisID) = size(B.E, 2)
LinearAlgebra.rank(B::HSSBasisID) = ncols(B)
skeleton_rows(B::HSSBasisID) = view(B.P, 1:ncols(B))
remainder_rows(B::HSSBasisID) = view(B.P, (ncols(B) + 1):nrows(B))

function apply!(Y::AbstractMatrix, B::HSSBasisID, X::AbstractMatrix)
    Y[skeleton_rows(B), :] .= X
    ncols(B) < nrows(B) && mul!(view(Y, remainder_rows(B), :), B.E, X)
    return Y
end
apply(B::HSSBasisID{T}, X::AbstractMatrix) where {T} =
    apply!(zeros(T, nrows(B), size(X, 2)), B, X)
apply(B::HSSBasisID, x::AbstractVector) = vec(apply(B, reshape(x, :, 1)))

function applyC(B::HSSBasisID{T}, X::AbstractMatrix) where {T}
    Y = Matrix{T}(X[skeleton_rows(B), :])
    ncols(B) < nrows(B) && mul!(Y, B.E', view(X, remainder_rows(B), :), true, true)
    return Y
end
applyC(B::HSSBasisID, x::AbstractVector) = vec(applyC(B, reshape(x, :, 1)))

Base.Matrix(B::HSSBasisID{T}) where {T} = apply(B, Matrix{T}(I, ncols(B), ncols(B)))

function _row_id(A::AbstractMatrix{T}, rtol::Real, rmax::Int) where {T}
    m, n = size(A)
    rmax = min(rmax, m, n)
    if m == 0 || n == 0 || rmax == 0
        return HSSBasisID{T}(m, 0), Int[]
    end
    F = qr(A', ColumnNorm())
    Rdiag = abs.(diag(F.R))
    thr = rtol * (isempty(Rdiag) ? zero(eltype(Rdiag)) : first(Rdiag))
    r = 0
    @inbounds for k in 1:min(length(Rdiag), rmax)
        Rdiag[k] > thr || break
        r = k
    end
    r = max(r, min(1, rmax))
    J = Vector{Int}(F.p[1:r])
    rest = setdiff(collect(1:m), J)
    isempty(rest) && return HSSBasisID{T}(J, zeros(T, 0, r)), J
    E = Matrix{T}(A[rest, :] / A[J, :])
    return HSSBasisID{T}(vcat(J, rest), E), J
end

# =============================================================================
# HSSMatrix
# =============================================================================

"""
    mutable struct HSSScatteringNode{T}

Per-node factors for the discrete-scattering direct solver (Martinsson Ch. 18).

- leaf: `Dlu` factors `D = A(I,I)`; `S = V† D^{-1} U`
- parent: `Z = [I S₀B₀₁; S₁B₁₀ I]^{-1}`, `S = V† Z blkdiag(S₀,S₁) U`
- root: only `Z` for the two children (incoming field is zero)
"""
mutable struct HSSScatteringNode{T}
    S::Matrix{T}
    Z::Matrix{T}
    Dlu::Any
end

"""
    mutable struct HSSMatrix{R,T} <: AbstractStructuredMatrix{T}

Hierarchically Semi-Separable / Hierarchically Block Separable (HSS ≡ HBS)
matrix with nested [`HSSBasisID`](@ref) generators on a binary
[`ClusterTree`](@ref).

In Martinsson's *Fast Direct Solvers* the nested weak-admissibility format is
called **HBS**; the literature name **HSS** (Xia/Gu et al.) is algebraically the
same class (nested bases, sibling couplings only). This type implements that
shared format.

# Generators
- leaves: dense `D` and bases `U,V`
- non-leaves: couplings `B01,B10` and nested translators `U,V`

Assemble with [`assemble_hss`](@ref) / [`assemble_hbs`](@ref).

# Factorization
- `lu!(H; method=:scattering)` — native Ch. 18 discrete scattering matrices
  (O(Nk²), fewer inversions, preferred)
- `lu!(H; method=:hodlr)` — expand nested generators to HODLR and reuse HODLR ULV
"""
mutable struct HSSMatrix{R, T} <: AbstractStructuredMatrix{T}
    tree::R
    U::HSSBasisID{T}
    V::HSSBasisID{T}
    D::Union{Matrix{T}, Nothing}
    B01::Union{Matrix{T}, Nothing}
    B10::Union{Matrix{T}, Nothing}
    children::Vector{HSSMatrix{R, T}}
    parentnode::HSSMatrix{R, T}
    """HODLR mirror holding ULV factors after `lu!(…; method=:hodlr)`."""
    hodlr_factors::Union{HODLRMatrix{R, T}, Nothing}
    """Scattering-matrix factors after `lu!(…; method=:scattering)`."""
    scattering::Union{HSSScatteringNode{T}, Nothing}
    function HSSMatrix{R, T}(tree, parent = nothing) where {R, T}
        H = new{R, T}(tree)
        m = length(tree)
        H.U = HSSBasisID{T}(m, 0)
        H.V = HSSBasisID{T}(m, 0)
        H.D = nothing
        H.B01 = nothing
        H.B10 = nothing
        H.children = HSSMatrix{R, T}[]
        H.parentnode = isnothing(parent) ? H : parent
        H.hodlr_factors = nothing
        H.scattering = nothing
        return H
    end
end

"""Alias: HBS (Martinsson) and HSS (Xia/Gu) share the same nested format."""
const HBSMatrix = HSSMatrix
const HBSBasisID = HSSBasisID
const HBSScatteringNode = HSSScatteringNode

isleaf(H::HSSMatrix) = isempty(H.children)
isroot(H::HSSMatrix) = H.parentnode === H
children(H::HSSMatrix) = H.children
parentnode(H::HSSMatrix) = H.parentnode
index_range(H::HSSMatrix) = index_range(H.tree)
Base.size(H::HSSMatrix) = (length(H.tree), length(H.tree))
Base.eltype(::HSSMatrix{<:Any, T}) where {T} = T
rowperm(H::HSSMatrix) = loc2glob(H.tree)
colperm(H::HSSMatrix) = loc2glob(H.tree)
U_rank(H::HSSMatrix) = ncols(H.U)
V_rank(H::HSSMatrix) = ncols(H.V)

Base.show(io::IO, H::HSSMatrix) =
    print(io, "HSSMatrix{$(eltype(H))} of size $(size(H,1)) × $(size(H,2))")
Base.show(io::IO, ::MIME"text/plain", H::HSSMatrix) = show(io, H)

# =============================================================================
# Assembly
# =============================================================================

"""
    assemble_hss([T,], K, tree; rtol=1e-6, rank=typemax(Int),
                 method=:dense, global_index=true, oversampling=10)

Assemble a square [`HSSMatrix`](@ref).

# Keywords
- `method=:dense` — form dense `A`, bottom-up row-ID (O(N²))
- `method=:randomized` — Gaussian sampling + skeleton element extraction
- `rtol`, `rank` — ID tolerance / max rank
- `oversampling` — extra sample columns for `:randomized`
"""
function assemble_hss(
        ::Type{T},
        K,
        tree::R;
        rtol = sqrt(eps(Float64)),
        rank = typemax(Int),
        method = :dense,
        global_index = use_global_index(),
        oversampling = 10,
    ) where {T, R}
    rp = loc2glob(tree)
    global_index && (K = PermutedMatrix(K, rp, rp))
    root = _build_hss_tree(HSSMatrix{R, T}, tree)
    if method === :dense
        n = length(tree)
        A = Matrix{T}(undef, n, n)
        getblock!(A, K, 1:n, 1:n)
        _hss_compress_dense!(root, A, float(rtol), Int(rank))
    elseif method === :randomized
        _hss_compress_randomized!(root, K, float(rtol), Int(rank), Int(oversampling))
    else
        throw(ArgumentError("unknown HSS method $(repr(method)); use :dense or :randomized"))
    end
    return root
end

assemble_hss(K::AbstractMatrix, tree; kwargs...) =
    assemble_hss(eltype(K), K, tree; kwargs...)

"""
    assemble_hbs([T,], K, tree; kwargs...)

Assemble a hierarchically block separable (HBS) matrix. Alias of
[`assemble_hss`](@ref): HBS (Martinsson Ch. 14–18) and HSS use the same nested
weak-admissibility generators; Chapter 16's ID skeletonization is the default
basis representation (`HSSBasisID`).
"""
const assemble_hbs = assemble_hss

function _build_hss_tree(::Type{HSSMatrix{R, T}}, tree, parent = nothing) where {R, T}
    H = HSSMatrix{R, T}(tree, parent)
    if !isleaf(tree)
        ch = children(tree)
        length(ch) == 2 || error("HSS requires a binary ClusterTree")
        H.children = [
            _build_hss_tree(HSSMatrix{R, T}, ch[1], H),
            _build_hss_tree(HSSMatrix{R, T}, ch[2], H),
        ]
    end
    return H
end

function _hss_compress_dense!(H::HSSMatrix{R, T}, A::Matrix{T}, rtol, rmax) where {R, T}
    I = index_range(H)
    n = size(A, 1)
    Ic = vcat(1:(first(I) - 1), (last(I) + 1):n)
    if isleaf(H)
        H.D = A[I, I]
        Brow = isempty(Ic) ? zeros(T, length(I), 0) : A[I, Ic]
        Bcol = isempty(Ic) ? zeros(T, length(I), 0) : Matrix(A[Ic, I]')
        rB = min(rmax, size(Brow, 1), size(Brow, 2))
        rC = min(rmax, size(Bcol, 1), size(Bcol, 2))
        U, Jr_loc = _row_id(Brow, rtol, rB)
        V, Jc_loc = _row_id(Bcol, rtol, rC)
        H.U, H.V = U, V
        Jr = isempty(Jr_loc) ? Int[] : collect(I[Jr_loc])
        Jc = isempty(Jc_loc) ? Int[] : collect(I[Jc_loc])
        return Jr, Jc
    else
        c0, c1 = H.children[1], H.children[2]
        Jr0, Jc0 = _hss_compress_dense!(c0, A, rtol, rmax)
        Jr1, Jc1 = _hss_compress_dense!(c1, A, rtol, rmax)
        H.B01 = isempty(Jr0) || isempty(Jc1) ? zeros(T, length(Jr0), length(Jc1)) : A[Jr0, Jc1]
        H.B10 = isempty(Jr1) || isempty(Jc0) ? zeros(T, length(Jr1), length(Jc0)) : A[Jr1, Jc0]
        Jrows, Jcols = vcat(Jr0, Jr1), vcat(Jc0, Jc1)
        Brow = isempty(Ic) || isempty(Jrows) ? zeros(T, length(Jrows), 0) : A[Jrows, Ic]
        Bcol = isempty(Ic) || isempty(Jcols) ? zeros(T, length(Jcols), 0) : Matrix(A[Ic, Jcols]')
        U, Jr_loc = _row_id(Brow, rtol, min(rmax, size(Brow, 1), size(Brow, 2)))
        V, Jc_loc = _row_id(Bcol, rtol, min(rmax, size(Bcol, 1), size(Bcol, 2)))
        H.U, H.V = U, V
        Jr = isempty(Jr_loc) ? Int[] : Jrows[Jr_loc]
        Jc = isempty(Jc_loc) ? Int[] : Jcols[Jc_loc]
        return Jr, Jc
    end
end

function _hss_compress_randomized!(H::HSSMatrix{R, T}, K, rtol, rmax, ov) where {R, T}
    n = size(H, 1)
    rt = rmax >= typemax(Int) ÷ 2 ? min(n, 48) : rmax
    rs = min(n, max(rt + ov, ov + 1))
    Ωr, Ωc = randn(T, n, rs), randn(T, n, rs)
    Sr = _sample_A_times(K, Ωr)
    Sc = _sample_At_times(K, Ωc)
    _hss_compress_from_samples!(H, K, Sr, Sc, Ωr, Ωc, rtol, rmax)
    return H
end

function _sample_A_times(K, Ω::Matrix{T}) where {T}
    n, rs = size(Ω)
    S = zeros(T, n, rs)
    col = Vector{T}(undef, n)
    for j in 1:n
        getblock!(col, K, 1:n, j)
        @inbounds for k in 1:rs
            axpy!(Ω[j, k], col, view(S, :, k))
        end
    end
    return S
end

function _sample_At_times(K, Ω::Matrix{T}) where {T}
    n, rs = size(Ω)
    S = Matrix{T}(undef, n, rs)
    col = Vector{T}(undef, n)
    for j in 1:n
        getblock!(col, K, 1:n, j)
        @inbounds for k in 1:rs
            S[j, k] = dot(col, view(Ω, :, k))
        end
    end
    return S
end

function _hss_compress_from_samples!(
        H::HSSMatrix{R, T}, K, Sr, Sc, Ωr, Ωc, rtol, rmax,
    ) where {R, T}
    I = index_range(H)
    if isleaf(H)
        m = length(I)
        D = Matrix{T}(undef, m, m)
        getblock!(D, K, I, I)
        H.D = D
        # peel diagonal contribution so ID sees off-diagonal range
        Brow = Sr[I, :] - D * Ωr[I, :]
        Bcol = Sc[I, :] - D' * Ωc[I, :]
        U, Jr_loc = _row_id(Brow, rtol, min(rmax, m, size(Brow, 2)))
        V, Jc_loc = _row_id(Bcol, rtol, min(rmax, m, size(Bcol, 2)))
        H.U, H.V = U, V
        Jr = isempty(Jr_loc) ? Int[] : collect(I[Jr_loc])
        Jc = isempty(Jc_loc) ? Int[] : collect(I[Jc_loc])
        return Jr, Jc
    else
        c0, c1 = H.children[1], H.children[2]
        Jr0, Jc0 = _hss_compress_from_samples!(c0, K, Sr, Sc, Ωr, Ωc, rtol, rmax)
        Jr1, Jc1 = _hss_compress_from_samples!(c1, K, Sr, Sc, Ωr, Ωc, rtol, rmax)
        H.B01 = _extract_elems(K, Jr0, Jc1, T)
        H.B10 = _extract_elems(K, Jr1, Jc0, T)
        Jrows, Jcols = vcat(Jr0, Jr1), vcat(Jc0, Jc1)
        # samples restricted to skeletons (diagonal already peeled at leaves)
        Brow = isempty(Jrows) ? zeros(T, 0, size(Sr, 2)) : Sr[Jrows, :]
        Bcol = isempty(Jcols) ? zeros(T, 0, size(Sc, 2)) : Sc[Jcols, :]
        U, Jr_loc = _row_id(Brow, rtol, min(rmax, max(length(Jrows), 1), size(Brow, 2)))
        V, Jc_loc = _row_id(Bcol, rtol, min(rmax, max(length(Jcols), 1), size(Bcol, 2)))
        H.U, H.V = U, V
        Jr = isempty(Jr_loc) ? Int[] : Jrows[Jr_loc]
        Jc = isempty(Jc_loc) ? Int[] : Jcols[Jc_loc]
        return Jr, Jc
    end
end

function _extract_elems(K, I::Vector{Int}, J::Vector{Int}, ::Type{T}) where {T}
    B = zeros(T, length(I), length(J))
    @inbounds for jj in eachindex(J), ii in eachindex(I)
        B[ii, jj] = K[I[ii], J[jj]]
    end
    return B
end

# =============================================================================
# Matvec (nested up/down sweeps)
# =============================================================================

function LinearAlgebra.mul!(
        y::AbstractVector, H::HSSMatrix, x::AbstractVector,
        a::Number = 1, b::Number = 0; global_index = use_global_index(),
    )
    if global_index
        x = x[colperm(H)]
        y = permute!(y, rowperm(H))
        rmul!(x, a)
    elseif a != 1
        x = a * x
    end
    iszero(b) ? fill!(y, 0) : rmul!(y, b)
    y .+= _hss_matvec(H, collect(x))
    global_index && invpermute!(y, rowperm(H))
    return y
end

function LinearAlgebra.mul!(
        Y::AbstractMatrix, H::HSSMatrix, X::AbstractMatrix,
        a::Number = 1, b::Number = 0; kwargs...,
    )
    size(Y, 2) == size(X, 2) || throw(DimensionMismatch())
    @inbounds for k in 1:size(Y, 2)
        mul!(view(Y, :, k), H, view(X, :, k), a, b; kwargs...)
    end
    return Y
end

function _hss_matvec(H::HSSMatrix{R, T}, x::Vector{T}) where {R, T}
    y = zeros(T, length(x))
    zmap = IdDict{Any, Vector{T}}()
    _hss_up!(H, x, zmap)
    _hss_down!(H, y, x, zmap, T[])
    return y
end

function _hss_up!(H::HSSMatrix{R, T}, x, zmap) where {R, T}
    if isleaf(H)
        z = applyC(H.V, collect(view(x, index_range(H))))
    else
        z0 = _hss_up!(H.children[1], x, zmap)
        z1 = _hss_up!(H.children[2], x, zmap)
        z = applyC(H.V, vcat(z0, z1))
    end
    zmap[H] = z
    return z
end

function _hss_down!(H::HSSMatrix{R, T}, y, x, zmap, gin::Vector{T}) where {R, T}
    if isleaf(H)
        I = index_range(H)
        yloc = H.D * view(x, I)
        isempty(gin) || (yloc = yloc + apply(H.U, gin))
        view(y, I) .+= yloc
    else
        c0, c1 = H.children[1], H.children[2]
        z0, z1 = zmap[c0]::Vector{T}, zmap[c1]::Vector{T}
        # B01: U_rank(c0) × V_rank(c1); B10: U_rank(c1) × V_rank(c0)
        g0 = isempty(H.B01) ? zeros(T, size(H.B01, 1)) : H.B01 * z1
        g1 = isempty(H.B10) ? zeros(T, size(H.B10, 1)) : H.B10 * z0
        if !isempty(gin)
            uin = apply(H.U, gin)
            # U translator rows are stacked U-ranks of children (not V-ranks)
            r0u = size(H.B01, 1)
            @assert length(uin) == r0u + size(H.B10, 1)
            g0 = g0 + uin[1:r0u]
            g1 = g1 + uin[(r0u + 1):end]
        end
        _hss_down!(c0, y, x, zmap, g0)
        _hss_down!(c1, y, x, zmap, g1)
    end
    return y
end

function Base.Matrix(H::HSSMatrix{R, T}; global_index = true) where {R, T}
    n = size(H, 1)
    M, ej = zeros(T, n, n), zeros(T, n)
    @inbounds for j in 1:n
        fill!(ej, 0); ej[j] = 1
        M[:, j] = _hss_matvec(H, ej)
    end
    global_index || return M
    p = rowperm(H)
    return Matrix(PermutedMatrix(M, invperm(p), invperm(p)))
end

compression_ratio(H::HSSMatrix) =
    (length(H) * sizeof(eltype(H))) / Base.summarysize(H)

function maxrank(H::HSSMatrix)
    r = max(U_rank(H), V_rank(H))
    for c in children(H)
        r = max(r, maxrank(c))
    end
    return r
end

function Base.deepcopy_internal(H::HSSMatrix{R, T}, sd::IdDict) where {R, T}
    haskey(sd, H) && return sd[H]
    H2 = HSSMatrix{R, T}(H.tree, nothing)
    sd[H] = H2
    H2.U = HSSBasisID(copy(H.U.P), copy(H.U.E))
    H2.V = HSSBasisID(copy(H.V.P), copy(H.V.E))
    H2.D = isnothing(H.D) ? nothing : deepcopy(H.D)
    H2.B01 = isnothing(H.B01) ? nothing : deepcopy(H.B01)
    H2.B10 = isnothing(H.B10) ? nothing : deepcopy(H.B10)
    H2.hodlr_factors = nothing
    H2.scattering = nothing
    if !isempty(H.children)
        H2.children = [Base.deepcopy_internal(c, sd) for c in H.children]
        for c in H2.children
            c.parentnode = H2
        end
    end
    return H2
end

# =============================================================================
# Expanded bases + HODLR conversion for ULV
# =============================================================================

"""Full expanded `length(H) × r` U basis (nested generators → dense)."""
function _full_U(H::HSSMatrix{R, T}) where {R, T}
    if isleaf(H)
        return Matrix(H.U)
    else
        U0, U1 = _full_U(H.children[1]), _full_U(H.children[2])
        r0 = size(U0, 2)
        Un = Matrix(H.U)   # (r0+r1) × r
        r = size(Un, 2)
        r == 0 && return zeros(T, size(H, 1), 0)
        return vcat(U0 * Un[1:r0, :], U1 * Un[(r0 + 1):end, :])
    end
end

function _full_V(H::HSSMatrix{R, T}) where {R, T}
    if isleaf(H)
        return Matrix(H.V)
    else
        V0, V1 = _full_V(H.children[1]), _full_V(H.children[2])
        r0 = size(V0, 2)
        Vn = Matrix(H.V)
        r = size(Vn, 2)
        r == 0 && return zeros(T, size(H, 1), 0)
        return vcat(V0 * Vn[1:r0, :], V1 * Vn[(r0 + 1):end, :])
    end
end

"""
Convert nested HSS generators to an equivalent [`HODLRMatrix`](@ref) with
explicit `RkMatrix` off-diagonals (used for ULV factorization).
"""
function hss_to_hodlr(H::HSSMatrix{R, T}, comp = PartialACA(; rtol = 1e-12)) where {R, T}
    return _hss_to_hodlr_rec(H, nothing, comp)
end

function _hss_to_hodlr_rec(H::HSSMatrix{R, T}, parent, comp) where {R, T}
    if isleaf(H)
        return HODLRMatrix{R, T}(H.tree, copy(H.D), nothing, nothing, nothing, parent)
    else
        Hh = HODLRMatrix{R, T}(H.tree, nothing, nothing, nothing, nothing, parent)
        c0 = _hss_to_hodlr_rec(H.children[1], Hh, comp)
        c1 = _hss_to_hodlr_rec(H.children[2], Hh, comp)
        Hh.children = [c0, c1]
        U0, V1 = _full_U(H.children[1]), _full_V(H.children[2])
        U1, V0 = _full_U(H.children[2]), _full_V(H.children[1])
        # A01 ≈ U0 * B01 * V1' ; recompress to drop dependent generators
        Hh.B01 = _recompress_rk(RkMatrix(Matrix(U0 * H.B01), Matrix(V1)), comp)
        Hh.B10 = _recompress_rk(RkMatrix(Matrix(U1 * H.B10), Matrix(V0)), comp)
        return Hh
    end
end

# =============================================================================
# Factorization: scattering (Ch. 18) and HODLR-ULV fallback
# =============================================================================

const HSS_LU = LU{<:Any, <:HSSMatrix}

"""
    lu!(H::HSSMatrix; method=:scattering, kwargs...)
    lu!(H::HSSMatrix, compressor; method=:hodlr, kwargs...)

Factor an HSS/HBS matrix for subsequent `\\` / `ldiv!`.

# Methods
- `:scattering` (default) — Martinsson Ch. 18 discrete scattering matrices.
  Works directly on nested generators; cost O(Nk²); fewer inversions than the
  Woodbury form in Ch. 14 (S = D̂⁻¹ is the fundamental object).
- `:hodlr` — expand to an equivalent [`HODLRMatrix`](@ref) and run recursive
  HODLR ULV (legacy path; usually slower and more memory).
"""
function LinearAlgebra.lu!(
        H::HSSMatrix,
        compressor;
        method::Symbol = :hodlr,
        kwargs...,
    )
    method === :hodlr || throw(ArgumentError(
        "compressor argument is only valid with method=:hodlr (got method=$(repr(method)))",
    ))
    return _hss_factor_hodlr!(H, compressor; kwargs...)
end

function LinearAlgebra.lu!(
        H::HSSMatrix;
        method::Symbol = :scattering,
        atol = 0,
        rank = typemax(Int),
        rtol = atol > 0 || rank < typemax(Int) ? 0 : sqrt(eps(Float64)),
        kwargs...,
    )
    if method === :scattering || method === :hbs || method === :chapter18
        return _hss_factor_scattering!(H)
    elseif method === :hodlr || method === :ulv
        return _hss_factor_hodlr!(H, PartialACA(; atol, rank, rtol); kwargs...)
    else
        throw(ArgumentError(
            "unknown HSS/HBS lu method $(repr(method)); use :scattering or :hodlr",
        ))
    end
end

LinearAlgebra.lu(H::HSSMatrix, args...; kwargs...) = lu!(deepcopy(H), args...; kwargs...)

function _hss_factor_hodlr!(H::HSSMatrix, compressor; kwargs...)
    Hod = hss_to_hodlr(H, compressor)
    lu!(Hod, compressor; kwargs...)
    H.hodlr_factors = Hod
    H.scattering = nothing
    return LU(H, Int[], 0)
end

# ---- Chapter 18: discrete scattering matrices --------------------------------

"""Left-apply `V†` to a matrix (dual of the ID/`HSSBasisID` basis)."""
_hss_Vdual(V::HSSBasisID, M::AbstractMatrix) = applyC(V, M)

"""Build `blkdiag(S0, S1)` acting from stacked U-ranks to stacked V-ranks."""
function _blkdiag_S(S0::AbstractMatrix{T}, S1::AbstractMatrix{T}) where {T}
    r0u, r0v = size(S0, 2), size(S0, 1)
    r1u, r1v = size(S1, 2), size(S1, 1)
    S = zeros(T, r0v + r1v, r0u + r1u)
    S[1:r0v, 1:r0u] = S0
    S[(r0v + 1):end, (r0u + 1):end] = S1
    return S
end

"""Sibling exchange block `[0 B01; B10 0]` (U-stacked rows × V-stacked cols)."""
function _sibling_B(B01::AbstractMatrix{T}, B10::AbstractMatrix{T}) where {T}
    r0u, r1v = size(B01)
    r1u, r0v = size(B10)
    B = zeros(T, r0u + r1u, r0v + r1v)
    B[1:r0u, (r0v + 1):end] = B01
    B[(r0u + 1):end, 1:r0v] = B10
    return B
end

"""Z = [I, S0*B01; S1*B10, I]^{-1} on the stacked outgoing (V-rank) space."""
function _scattering_Z(
        S0::AbstractMatrix{T},
        S1::AbstractMatrix{T},
        B01::AbstractMatrix{T},
        B10::AbstractMatrix{T},
    ) where {T}
    r0v = size(S0, 1)
    r1v = size(S1, 1)
    n = r0v + r1v
    if n == 0
        return zeros(T, 0, 0)
    end
    M = Matrix{T}(I, n, n)
    # S0*B01 : (r0v × r0u)*(r0u × r1v) → r0v × r1v
    if r0v > 0 && r1v > 0 && size(B01, 1) > 0 && size(B01, 2) > 0
        M[1:r0v, (r0v + 1):end] += S0 * B01
    end
    if r1v > 0 && r0v > 0 && size(B10, 1) > 0 && size(B10, 2) > 0
        M[(r0v + 1):end, 1:r0v] += S1 * B10
    end
    return inv(M)
end

function _hss_factor_scattering!(H::HSSMatrix{R, T}) where {R, T}
    _hss_build_scattering!(H)
    H.hodlr_factors = nothing
    return LU(H, Int[], 0)
end

function _hss_build_scattering!(H::HSSMatrix{R, T}) where {R, T}
    if isleaf(H)
        D = H.D
        isnothing(D) && error("leaf HSS node missing diagonal block D")
        Dlu = lu(D)
        # S = V† D^{-1} U
        Um = Matrix(H.U)          # n × ku
        if size(Um, 2) == 0 && V_rank(H) == 0
            S = zeros(T, 0, 0)
        else
            DinvU = Dlu \ Um
            S = _hss_Vdual(H.V, DinvU)
        end
        H.scattering = HSSScatteringNode{T}(S, zeros(T, 0, 0), Dlu)
        return S
    end

    c0, c1 = H.children[1], H.children[2]
    S0 = _hss_build_scattering!(c0)
    S1 = _hss_build_scattering!(c1)
    B01 = isnothing(H.B01) ? zeros(T, size(S0, 2), size(S1, 1)) : H.B01
    B10 = isnothing(H.B10) ? zeros(T, size(S1, 2), size(S0, 1)) : H.B10
    Z = _scattering_Z(S0, S1, B01, B10)

    if isroot(H)
        # root: only Z for the top-level solve; S is unused (incoming = 0)
        H.scattering = HSSScatteringNode{T}(zeros(T, 0, 0), Z, nothing)
        return zeros(T, 0, 0)
    end

    # S = V† Z blkdiag(S0,S1) U
    Sd = _blkdiag_S(S0, S1)
    Um = Matrix(H.U)   # (u0+u1) × ku
    if size(Um, 2) == 0 && V_rank(H) == 0
        S = zeros(T, 0, 0)
    else
        tmp = Z * (Sd * Um)
        S = _hss_Vdual(H.V, tmp)
    end
    H.scattering = HSSScatteringNode{T}(S, Z, nothing)
    return S
end

function LinearAlgebra.ldiv!(F::HSS_LU, y::AbstractVector; global_index = true)
    H = F.factors
    if !isnothing(H.scattering)
        return _hss_solve_scattering!(H, y; global_index)
    elseif !isnothing(H.hodlr_factors)
        return ldiv!(LU(H.hodlr_factors, Int[], 0), y; global_index)
    else
        error("HSSMatrix has not been factored; call lu! first")
    end
end

function _hss_solve_scattering!(
        H::HSSMatrix{R, T},
        y::AbstractVector;
        global_index = true,
    ) where {R, T}
    @assert !isnothing(H.scattering) "missing scattering factors"
    if global_index
        permute!(y, colperm(H))
    end
    f = collect(y)
    # upwards: effective loads ỹ
    ytilde = IdDict{Any, Vector{T}}()
    _hss_scattering_up!(H, f, ytilde)
    # downwards: recover solution into y
    fill!(y, zero(T))
    _hss_scattering_down!(H, f, y, ytilde, T[])
    if global_index
        invpermute!(y, rowperm(H))
    end
    return y
end

function _hss_scattering_up!(H::HSSMatrix{R, T}, f, ytilde) where {R, T}
    fac = H.scattering::HSSScatteringNode{T}
    if isleaf(H)
        I = index_range(H)
        # ỹ = V† D^{-1} f(I)
        Dinvf = fac.Dlu \ collect(view(f, I))
        yt = applyC(H.V, Dinvf)
        ytilde[H] = yt
        return yt
    end
    c0, c1 = H.children[1], H.children[2]
    yt0 = _hss_scattering_up!(c0, f, ytilde)
    yt1 = _hss_scattering_up!(c1, f, ytilde)
    stacked = vcat(yt0, yt1)
    if isroot(H)
        # root has no S/V; store stacked loads for the top solve
        ytilde[H] = stacked
        return stacked
    end
    # ỹ = V† Z [ỹ0; ỹ1]
    yt = applyC(H.V, fac.Z * stacked)
    ytilde[H] = yt
    return yt
end

function _hss_scattering_down!(
        H::HSSMatrix{R, T},
        f,
        q::AbstractVector,
        ytilde,
        cin::Vector{T},
    ) where {R, T}
    fac = H.scattering::HSSScatteringNode{T}
    if isleaf(H)
        I = index_range(H)
        rhs = collect(view(f, I))
        if !isempty(cin)
            rhs .-= apply(H.U, cin)
        end
        view(q, I) .= fac.Dlu \ rhs
        return q
    end

    c0, c1 = H.children[1], H.children[2]
    fac0 = c0.scattering::HSSScatteringNode{T}
    fac1 = c1.scattering::HSSScatteringNode{T}
    S0, S1 = fac0.S, fac1.S
    B01 = isnothing(H.B01) ? zeros(T, size(S0, 2), size(S1, 1)) : H.B01
    B10 = isnothing(H.B10) ? zeros(T, size(S1, 2), size(S0, 1)) : H.B10
    yt0 = ytilde[c0]::Vector{T}
    yt1 = ytilde[c1]::Vector{T}
    stacked_y = vcat(yt0, yt1)

    # rhs = [ỹ0; ỹ1] - blkdiag(S0,S1) U cin   (cin empty at root)
    rhs = copy(stacked_y)
    if !isempty(cin)
        uin = apply(H.U, cin)                 # length u0+u1
        Sd = _blkdiag_S(S0, S1)
        rhs .-= Sd * uin
    end
    qtilde = fac.Z * rhs                      # stacked outgoing of children

    # incoming to children: sibling exchange + parent translation
    r0v = length(yt0)
    q0 = qtilde[1:r0v]
    q1 = qtilde[(r0v + 1):end]
    g0 = isempty(B01) ? zeros(T, size(B01, 1)) : B01 * q1
    g1 = isempty(B10) ? zeros(T, size(B10, 1)) : B10 * q0
    if !isempty(cin)
        uin = apply(H.U, cin)
        r0u = size(B01, 1)
        g0 = g0 + uin[1:r0u]
        g1 = g1 + uin[(r0u + 1):end]
    end
    _hss_scattering_down!(c0, f, q, ytilde, g0)
    _hss_scattering_down!(c1, f, q, ytilde, g1)
    return q
end
