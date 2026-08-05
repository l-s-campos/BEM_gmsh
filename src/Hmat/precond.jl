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
    hara_product(A, B, rowtree, coltree; kwargs...) -> HMatrix

Build hierarchical approximation of `C ≈ A*B` by HARA sampling

```text
v ↦ A*(B*v)
```

without forming the dense product. `A` and `B` need only support `mul!` /
`adjoint` matvecs (e.g. two `HMatrix` operators).
"""
function hara_product(A, B, rowtree, coltree; kwargs...)
    n = size(B, 2)
    size(A, 2) == size(B, 1) || throw(DimensionMismatch("hara_product: A,B incompatible"))
    m = size(A, 1)
    f! = function (Y, X)
        Z = B * X
        return mul!(Y, A, Z)
    end
    f_adj! = function (Y, X)
        Z = A' * X
        return mul!(Y, B', Z)
    end
    S = FunctionSampler(f!, n; m=m, f_adj! = f_adj!)
    return hara(S, rowtree, coltree; kwargs...)
end
