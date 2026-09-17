# Hierarchical low-rank update (HLRU) and structured add for classic H-matrices

"""
    hlru!(H::HMatrix, X, Y; rtol=1e-6, atol=0, rank=typemax(Int),
          global_index=true, tsvd=nothing)

In-place hierarchical low-rank update

```text
H ← H + X * Y'
```

where `X` is `n×k` and `Y` is `m×k` (for square H, `n=m=size(H,1)`).

Each leaf block receives the restricted update and is recompressed with
[`TSVD`](@ref) when low-rank. Dense nearfield blocks are updated in place.

# Keywords
- `global_index`: if `true` (default), `X`/`Y` use the exterior (unpermuted) index
- `rtol`, `atol`, `rank`: recompression controls
"""
function hlru!(
        H::HMatrix,
        X::AbstractMatrix,
        Y::AbstractMatrix;
        rtol = 1e-6,
        atol = 0.0,
        rank = typemax(Int),
        global_index = use_global_index(),
        tsvd = nothing,
    )
    size(X, 2) == size(Y, 2) || throw(DimensionMismatch("X and Y must have same column count"))
    size(X, 1) == size(H, 1) || throw(DimensionMismatch("size(X,1) != size(H,1)"))
    size(Y, 1) == size(H, 2) || throw(DimensionMismatch("size(Y,1) != size(H,2)"))
    comp = tsvd === nothing ? TSVD(; rtol=float(rtol), atol=float(atol), rank=rank) : tsvd

    # Work in local (tree) ordering
    Xl = global_index ? X[rowperm(H), :] : X
    Yl = global_index ? Y[colperm(H), :] : Y
    Xl = Matrix{eltype(H)}(Xl)
    Yl = Matrix{eltype(H)}(Yl)

    piv = pivot(H)
    for leaf in leaves(H)
        isleaf(leaf) || continue
        hasdata(leaf) || continue
        Ir = rowrange(leaf)
        Jr = colrange(leaf)
        # local block indices into root-local arrays
        Ii = (Ir.start - piv[1] + 1):(Ir.stop - piv[1] + 1)
        Jj = (Jr.start - piv[2] + 1):(Jr.stop - piv[2] + 1)
        Xi = Xl[Ii, :]
        Yj = Yl[Jj, :]
        _hlru_leaf!(leaf, Xi, Yj, comp)
    end
    return H
end

function _hlru_leaf!(leaf::HMatrix, Xi::Matrix, Yj::Matrix, comp::TSVD)
    d = data(leaf)
    if d isa Matrix
        mul!(d, Xi, adjoint(Yj), true, true)
        # optional: if admissible-style storage desired later, could recompress
        return leaf
    elseif d isa RkMatrix
        # [U Xi] [V Yj]'
        Anew = hcat(d.A, Xi)
        Bnew = hcat(d.B, Yj)
        R = RkMatrix(Anew, Bnew)
        compress!(R, comp)
        setdata!(leaf, R)
        return leaf
    else
        return leaf
    end
end

"""
    hlru!(H, x::AbstractVector, y::AbstractVector; kwargs...)

Rank-1 update `H ← H + x*y'`.
"""
function hlru!(H::HMatrix, x::AbstractVector, y::AbstractVector; kwargs...)
    return hlru!(H, reshape(x, :, 1), reshape(y, :, 1); kwargs...)
end

"""
    hadd!(C::HMatrix, A::HMatrix, B::HMatrix, α=1, β=1; rtol=1e-6, atol=0, rank=typemax(Int))

Structured add on **compatible** H-matrices (same block tree):

```text
C ← α*A + β*B
```

`C` must already have the same block structure as `A` and `B` (e.g. assembled
skeleton with the same trees). Leaf data is combined and low-rank leaves are
recompressed.
"""
function hadd!(
        C::HMatrix,
        A::HMatrix,
        B::HMatrix,
        α::Number = 1,
        β::Number = 1;
        rtol = 1e-6,
        atol = 0.0,
        rank = typemax(Int),
    )
    size(C) == size(A) == size(B) || throw(DimensionMismatch("hadd! size mismatch"))
    comp = TSVD(; rtol=float(rtol), atol=float(atol), rank=rank)
    _hadd_walk!(C, A, B, α, β, comp)
    return C
end

function _hadd_walk!(C::HMatrix, A::HMatrix, B::HMatrix, α, β, comp)
    if isleaf(C)
        hasdata(A) && hasdata(B) && hasdata(C) || return C
        da, db = data(A), data(B)
        if da isa Matrix && db isa Matrix
            Dc = data(C)
            Dc isa Matrix || setdata!(C, zeros(eltype(C), size(da)))
            Dc = data(C)::Matrix
            # C = α A + β B
            @. Dc = β * db
            axpy!(α, da, Dc)
        else
            # promote to dense small blocks or merge Rk
            Ma = da isa RkMatrix ? Matrix(da) : da
            Mb = db isa RkMatrix ? Matrix(db) : db
            M = α * Ma + β * Mb
            if isadmissible(C)
                setdata!(C, compress!(copy(M), comp))
            else
                setdata!(C, Matrix(M))
            end
        end
        return C
    end
    chC, chA, chB = children(C), children(A), children(B)
    size(chC) == size(chA) == size(chB) ||
        throw(ArgumentError("hadd!: incompatible H-matrix block structure"))
    for i in eachindex(chC)
        _hadd_walk!(chC[i], chA[i], chB[i], α, β, comp)
    end
    return C
end
