# Diagonal ridge, Cholesky convenience, GMRES + hierarchical preconditioner

"""
    add_diag_ridge!(H::HMatrix, ε)

Add `ε` to the **dense diagonal leaves** of a square `HMatrix` (in local tree
ordering). Useful before Cholesky when the compressed operator is nearly SPD
but slightly indefinite due to truncation.
"""
function add_diag_ridge!(H::HMatrix, ε::Real)
    size(H, 1) == size(H, 2) || throw(DimensionMismatch("add_diag_ridge! needs square H"))
    ε = float(ε)
    iszero(ε) && return H
    for leaf in leaves(H)
        isleaf(leaf) || continue
        hasdata(leaf) || continue
        rowrange(leaf) == colrange(leaf) || continue
        d = data(leaf)
        if d isa Matrix
            m = size(d, 1)
            @inbounds for i in 1:m
                d[i, i] += ε
            end
        end
    end
    return H
end

"""
    cholesky(H::HMatrix; ridge=0, kwargs...)

Hierarchical Cholesky of a square H-matrix via `Hermitian(H)`.
Optional `ridge` is added on dense diagonal leaves before factorization.
"""
function LinearAlgebra.cholesky(H::HMatrix; ridge::Real = 0, kwargs...)
    size(H, 1) == size(H, 2) || throw(DimensionMismatch("cholesky needs square H"))
    Hc = deepcopy(H)
    ridge != 0 && add_diag_ridge!(Hc, ridge)
    return cholesky(Hermitian(Hc); kwargs...)
end

function LinearAlgebra.cholesky!(H::HMatrix; ridge::Real = 0, kwargs...)
    ridge != 0 && add_diag_ridge!(H, ridge)
    return cholesky!(Hermitian(H); kwargs...)
end

"""
    gmres_h(A, b; Pl=nothing, atol=1e-10, rtol=1e-8, itmax=0, kwargs...)

GMRES solve with optional left preconditioner `Pl` (supports `ldiv!`), e.g.
hierarchical `lu(H)` or `cholesky(H)` factors.

Returns `(x, stats)` like `Krylov.gmres`.
"""
function gmres_h(
        A,
        b::AbstractVector;
        Pl = nothing,
        atol = 1e-10,
        rtol = 1e-8,
        itmax::Integer = 0,
        history::Bool = true,
        kwargs...,
    )
    n = length(b)
    itm = itmax > 0 ? Int(itmax) : max(4 * n, 200)
    if Pl === nothing
        return Krylov.gmres(A, b; atol=atol, rtol=rtol, itmax=itm, history=history, kwargs...)
    end
    return Krylov.gmres(A, b; M=Pl, ldiv=true, atol=atol, rtol=rtol, itmax=itm,
        history=history, kwargs...)
end

"""
    h2_lu_prec(A::NNCAMatrix; rank=2, rtol=0, method=:hmatrix)

LU of a rank-`rank` truncation of the H² matrix `A`, for use as `Pl` in
[`gmres_h`](@ref). Weighted recompress (H2Lib-style) caps every nested
basis and M2L at `rank`; **near-field leaves stay dense**. `method=:hmatrix`
expands to H-LU; `:nested` keeps H² LR.

Assemble a rank-`k` Chebyshev operator directly with
`assemble_h2(K, tree; method=:cheb, rank=k)` if you do not already have `A`.
"""
function h2_lu_prec(
        A::NNCAMatrix;
        rank::Integer = 2,
        rtol::Real = 0,
        method::Symbol = :hmatrix,
        kwargs...,
    )
    Int(rank) >= 1 || throw(ArgumentError("h2_lu_prec rank must be ≥ 1"))
    B = deepcopy(A)
    τ = float(rtol) > 0 ? float(rtol) : 1e-16
    _h2_cheb_recompress!(B, τ; rank = Int(rank))
    return lu(B; method = method, kwargs...)
end

