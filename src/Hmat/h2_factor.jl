# H² factorization (H2Lib-style API)
#
# H2Lib `lrdecomp_h2matrix` recurses on a son-block H² tree with nested
# `addmul_h2matrix` / `lowersolve_h2matrix` and clusteroperator weights.
# Our `H2Matrix` is a flat proxy/ID layout (H2Pack-style), so the practical
# port is:
#
#   1. convert H² → classic `HMatrix` (same cluster tree / box admissibility)
#   2. run existing hierarchical LU / Cholesky (same block recursion as H2Lib)
#
# This matches Börm–Reimer “LR via H-arithmetic after representation change”
# and the BEM plan MVP path, while exposing the H2Lib names.

"""
    h2_to_hmatrix(H2::H2Matrix; rtol=1e-6, atol=0, rank=typemax(Int),
                  method=:block, batch=16, threads=false) -> HMatrix

Convert a square [`H2Matrix`](@ref) to a classic [`HMatrix`](@ref) on the same
cluster tree, using H² matvecs only (no dense ``n×n`` materialization).

# Methods
- `:block` (default) — for each H leaf, inject identity columns on the leaf
  column range and read the row block from multi-RHS H² applies; admissible
  leaves are recompressed with [`TSVD`](@ref).
- `:hara` — black-box [`hara`](@ref) from an H² [`FunctionSampler`](@ref)
  (needs an adjoint matvec; uses the same apply when the operator is treated
  as self-adjoint, which matches default symmetric H² assembly).

The H block tree uses [`H2BoxAdmissibility`](@ref)`(H2.alpha)`.
"""
function h2_to_hmatrix(
        H2::H2Matrix{R, T};
        rtol = 1e-6,
        atol = 0.0,
        rank = typemax(Int),
        method::Symbol = :block,
        batch::Int = 16,
        threads::Bool = false,
        adm = nothing,
    ) where {R, T}
    tree = H2.tidx.nodes[1]
    adm_fun = adm === nothing ? H2BoxAdmissibility(H2.alpha) : adm
    method in (:block, :hara) || throw(ArgumentError(
        "h2_to_hmatrix method must be :block or :hara; got $method"))

    if method === :hara
        return _h2_to_hmatrix_hara(H2, tree, adm_fun; rtol, atol, rank, batch, threads)
    end
    return _h2_to_hmatrix_block(H2, tree, adm_fun; rtol, atol, rank)
end

function _h2_to_hmatrix_hara(
        H2::H2Matrix{R, T}, tree, adm;
        rtol, atol, rank, batch, threads,
    ) where {R, T}
    n = size(H2, 1)
    # Default H² assembly stores a symmetric operator (far B + B', near D + D').
    # Use the same apply for the adjoint sampler (SPD / symmetric BEM kernels).
    S = FunctionSampler(
        (Y, X) -> mul!(Y, H2, X);
        m = n,
        f_adj! = (Y, X) -> mul!(Y, H2, X),
    )
    return hara(
        S, tree, tree;
        adm = adm,
        rtol = float(rtol),
        atol = float(atol),
        rank = rank,
        batch = batch,
        threads = threads,
        global_index = true,
    )
end

function _h2_to_hmatrix_block(
        H2::H2Matrix{R, T}, tree, adm;
        rtol, atol, rank,
    ) where {R, T}
    H = HMatrix{T}(tree, tree, adm)
    n = size(H2, 1)
    comp = TSVD(; rtol = float(rtol), atol = float(atol), rank = rank)
    # H ranges are in tree-local order; apply H² without global permutation.
    for leaf in leaves(H)
        isleaf(leaf) || continue
        hasdata(leaf) && continue  # should be empty
        Ir = rowrange(leaf)
        Jr = colrange(leaf)
        mblk, nblk = length(Ir), length(Jr)
        (mblk == 0 || nblk == 0) && continue
        Blk = Matrix{T}(undef, mblk, nblk)
        # multi-RHS identity injection on Jr (local)
        X = zeros(T, n, nblk)
        @inbounds for (k, j) in enumerate(Jr)
            X[j, k] = one(T)
        end
        Y = zeros(T, n, nblk)
        mul!(Y, H2, X, one(T), zero(T); global_index = false)
        @inbounds Blk .= view(Y, Ir, :)
        if isadmissible(leaf)
            setdata!(leaf, compress!(copy(Blk), comp))
        else
            setdata!(leaf, Blk)
        end
    end
    return H
end

# ---------------------------------------------------------------------------
# H2Lib-named factorizations
# ---------------------------------------------------------------------------

"""
    H2LU{TH}

Approximate LU factorization of an [`H2Matrix`](@ref), stored as a classic
hierarchical [`LU`](@ref) of the converted [`HMatrix`](@ref).

Solves use `F \\ b` / [`ldiv!`](@ref) like H-matrix LU.
"""
struct H2LU{TH}
    lu::LU{<:Any, TH}
    H::TH
end

Base.size(F::H2LU, d::Integer) = size(F.H, d)
Base.size(F::H2LU) = size(F.H)
Base.eltype(F::H2LU) = eltype(F.H)

function Base.getproperty(F::H2LU, s::Symbol)
    if s === :L
        return getproperty(getfield(F, :lu), :L)
    elseif s === :U
        return getproperty(getfield(F, :lu), :U)
    elseif s === :factors
        return getproperty(getfield(F, :lu), :factors)
    else
        return getfield(F, s)
    end
end

function Base.show(io::IO, ::MIME"text/plain", F::H2LU)
    return print(io, "H2LU factorization via H-matrix LU of $(F.H)")
end

"""
    lrdecomp_h2matrix(H2::H2Matrix; rtol=1e-6, atol=0, rank=typemax(Int),
                      method=:block, threads=false, lu_rtol=rtol, kwargs...)

Port of H2Lib `lrdecomp_h2matrix`: approximate LR factorization of an H² matrix.

# Returns
Named tuple `(; L, U, F, H)` where
- `H` is the converted [`HMatrix`](@ref)
- `F` is `LinearAlgebra.LU` of `H` (in-place factors)
- `L`, `U` are `UnitLowerTriangular` / `UpperTriangular` views

# Notes
Unlike the full H2Lib path (nested H² `addmul` + clusteroperator recompression),
this MVP converts H²→H then runs the existing hierarchical block LU — the same
recursive pattern as H2Lib once the matrix lives on an H block tree:

```text
for i
  LR(Xᵢᵢ) → Lᵢᵢ, Rᵢᵢ
  Lᵢᵢ \\ Xᵢⱼ → Rᵢⱼ
  Xⱼᵢ / Rᵢᵢ → Lⱼᵢ
  Xⱼₖ ← Xⱼₖ − Lⱼᵢ Rᵢₖ
```
"""
function lrdecomp_h2matrix(
        H2::H2Matrix;
        rtol = 1e-6,
        atol = 0.0,
        rank = typemax(Int),
        method::Symbol = :block,
        batch::Int = 16,
        threads::Bool = false,
        lu_rtol = nothing,
        lu_atol = nothing,
        lu_rank = nothing,
        adm = nothing,
    )
    # True nested LR on recursive H2Node (H2Lib lrdecomp_h2matrix)
    if method === :nested
        return lrdecomp_h2matrix_nested(H2; rtol = rtol)
    end
    # Practical path: H² → H → hierarchical LU
    conv = method === :hara ? :hara : :block
    H = h2_to_hmatrix(
        H2;
        rtol = rtol,
        atol = atol,
        rank = rank,
        method = conv,
        batch = batch,
        threads = threads,
        adm = adm,
    )
    frtol = something(lu_rtol, rtol)
    fatol = something(lu_atol, atol)
    frank = something(lu_rank, rank)
    F = lu!(H; rtol = frtol, atol = fatol, rank = frank, threads = threads)
    return (; L = F.L, U = F.U, F = F, H = H, method = conv)
end

"""
    lrsolve_h2matrix(F, b; global_index=true) -> x

Solve `A x ≈ b` given `F` from [`lrdecomp_h2matrix`](@ref) or [`lu`](@ref) on H².
"""
function lrsolve_h2matrix(F::NamedTuple, b::AbstractVector; global_index = true)
    x = copy(b)
    return lrsolve_h2matrix(F.F, x; global_index = global_index)
end

function lrsolve_h2matrix(F::H2LU, b::AbstractVector; global_index = true)
    x = copy(b)
    ldiv!(F.lu, x; global_index = global_index)
    return x
end

function lrsolve_h2matrix(F::LU, b::AbstractVector; global_index = true)
    x = copy(b)
    ldiv!(F, x; global_index = global_index)
    return x
end

"""
    LinearAlgebra.lu(H2::H2Matrix; kwargs...) -> H2LU

Approximate LU of an H² matrix (H2Lib `lrdecomp_h2matrix` practical port).
Keywords forwarded to [`lrdecomp_h2matrix`](@ref).
"""
function LinearAlgebra.lu(H2::H2Matrix; method::Symbol = :block, kwargs...)
    out = lrdecomp_h2matrix(H2; method = method, kwargs...)
    if out.F isa H2NodeLU
        return out.F
    end
    return H2LU(out.F, out.H)
end

function LinearAlgebra.ldiv!(F::H2LU, y::AbstractVector; global_index = true)
    return ldiv!(F.lu, y; global_index = global_index)
end

function Base.:\(F::H2LU, b::AbstractVector)
    x = copy(b)
    ldiv!(F, x)
    return x
end

# ---------------------------------------------------------------------------
# Cholesky (H2Lib choldecomp_h2matrix)
# ---------------------------------------------------------------------------

"""
    choldecomp_h2matrix(H2::H2Matrix; ridge=0, rtol=1e-6, ...) -> NamedTuple

Port of H2Lib `choldecomp_h2matrix` via H²→H then hierarchical Cholesky.
Returns `(; L, F, H)` with `F = cholesky(H; ridge, ...)`.
"""
function choldecomp_h2matrix(
        H2::H2Matrix;
        ridge = 0.0,
        rtol = 1e-6,
        atol = 0.0,
        rank = typemax(Int),
        method::Symbol = :block,
        batch::Int = 16,
        threads::Bool = false,
        adm = nothing,
        kwargs...,
    )
    H = h2_to_hmatrix(
        H2;
        rtol = rtol,
        atol = atol,
        rank = rank,
        method = method,
        batch = batch,
        threads = threads,
        adm = adm,
    )
    F = cholesky(H; ridge = ridge, rtol = rtol, atol = atol, rank = rank, threads = threads, kwargs...)
    return (; L = F.L, F = F, H = parent(F) isa Hermitian ? parent(parent(F)) : F.factors)
end

"""
    LinearAlgebra.cholesky(H2::H2Matrix; ridge=0, kwargs...)

Approximate Cholesky of a symmetric positive-(semi)definite H² matrix.
"""
function LinearAlgebra.cholesky(H2::H2Matrix; ridge = 0.0, kwargs...)
    out = choldecomp_h2matrix(H2; ridge = ridge, kwargs...)
    return out.F
end
