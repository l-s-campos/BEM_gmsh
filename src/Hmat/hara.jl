# Sampler interface + HARA-style hierarchical assembly from matvecs (classic H)

"""
    abstract type AbstractMatvecSampler

Black-box linear operator accessible only through multi-RHS matvecs.
Implement [`LinearAlgebra.mul!`](@ref) as `mul!(Y, S, X)` for `Y = A*X`.
Optional adjoint: `mul!(Y, adjoint(S), X)`.
"""
abstract type AbstractMatvecSampler end

Base.size(S::AbstractMatvecSampler, d::Int) = size(S)[d]

"""
    FunctionSampler(f!, n; m=n, f_adj!=nothing)

Wraps `f!(Y, X)` computing `Y = A*X` for an `m×n` operator.
"""
struct FunctionSampler{F, Fa} <: AbstractMatvecSampler
    f!::F
    f_adj!::Fa
    m::Int
    n::Int
end

function FunctionSampler(f!, n::Integer; m::Integer = n, f_adj! = nothing)
    return FunctionSampler{typeof(f!), typeof(f_adj!)}(f!, f_adj!, Int(m), Int(n))
end

Base.size(S::FunctionSampler) = (S.m, S.n)

function LinearAlgebra.mul!(Y::AbstractMatrix, S::FunctionSampler, X::AbstractMatrix)
    size(X, 1) == S.n || throw(DimensionMismatch())
    size(Y, 1) == S.m || throw(DimensionMismatch())
    S.f!(Y, X)
    return Y
end

function LinearAlgebra.mul!(Y::AbstractMatrix, St::Adjoint{<:Any, <:FunctionSampler}, X::AbstractMatrix)
    S = parent(St)
    S.f_adj! === nothing && throw(ArgumentError("FunctionSampler has no adjoint matvec"))
    size(X, 1) == S.m || throw(DimensionMismatch())
    size(Y, 1) == S.n || throw(DimensionMismatch())
    S.f_adj!(Y, X)
    return Y
end

"""
    KernelMatvecSampler(K::AbstractMatrix)

Sampler that applies a dense/abstract matrix (e.g. [`KernelMatrix`](@ref)) via `mul!`.
"""
struct KernelMatvecSampler{K} <: AbstractMatvecSampler
    K::K
end

Base.size(S::KernelMatvecSampler) = size(S.K)

function LinearAlgebra.mul!(Y::AbstractMatrix, S::KernelMatvecSampler, X::AbstractMatrix)
    return mul!(Y, S.K, X)
end

function LinearAlgebra.mul!(Y::AbstractMatrix, St::Adjoint{<:Any, <:KernelMatvecSampler}, X::AbstractMatrix)
    return mul!(Y, adjoint(parent(St).K), X)
end

# ---------------------------------------------------------------------------
# Randomized range finder for one block via global sampler
# ---------------------------------------------------------------------------

"""
Sample `A[I,J] * Ω` using a global matvec: inject `Ω` into rows `J`, read `I`.
`I`,`J` are index ranges in the **local** (tree) ordering; sampler is local too.
"""
function _sample_block_action!(
        YI::AbstractMatrix,
        S::AbstractMatvecSampler,
        I::UnitRange{Int},
        J::UnitRange{Int},
        Ω::AbstractMatrix,
        Xbuf::AbstractMatrix,
        Ybuf::AbstractMatrix,
    )
    fill!(Xbuf, 0)
    Xbuf[J, :] .= Ω
    mul!(Ybuf, S, Xbuf)
    YI .= view(Ybuf, I, :)
    return YI
end

"""
Adaptive randomized low-rank approximation of block `(I,J)` of sampler `S`
(local ordering). Returns an [`RkMatrix`](@ref).
"""
function _rand_lr_block(
        S::AbstractMatvecSampler,
        I::UnitRange{Int},
        J::UnitRange{Int};
        rtol = 1e-4,
        atol = 0.0,
        rank = typemax(Int),
        batch = 8,
        ntest = 3,
        max_passes = 20,
    )
    T = Float64
    m, n = length(I), length(J)
    N = size(S, 2)
    M = size(S, 1)
    rmax = min(Int(rank), m, n)
    rmax <= 0 && return RkMatrix(zeros(T, m, 0), zeros(T, n, 0))

    Xbuf = zeros(T, N, batch + ntest)
    Ybuf = zeros(T, M, batch + ntest)
    Q = zeros(T, m, 0)
    YI = zeros(T, m, batch)

    r = 0
    streak = 0
    pass = 0
    while r < rmax && pass < max_passes
        pass += 1
        bs = min(batch, rmax - r)
        Ω = randn(T, n, bs)
        _sample_block_action!(view(YI, :, 1:bs), S, I, J, Ω, view(Xbuf, :, 1:bs), view(Ybuf, :, 1:bs))
        # remove existing range
        if r > 0
            YI[:, 1:bs] .-= Q * (Q' * view(YI, :, 1:bs))
        end
        F = qr!(YI[:, 1:bs])
        Qnew = Matrix(F.Q)
        # orthonormalize against Q
        if r > 0
            Qnew .-= Q * (Q' * Qnew)
            F2 = qr!(Qnew)
            Qnew = Matrix(F2.Q)
        end
        Q = r == 0 ? Qnew : hcat(Q, Qnew)
        r = size(Q, 2)

        # error estimate on fresh test vectors
        Ωt = randn(T, n, ntest)
        Yt = zeros(T, m, ntest)
        _sample_block_action!(Yt, S, I, J, Ωt, view(Xbuf, :, 1:ntest), view(Ybuf, :, 1:ntest))
        resid = Yt .- Q * (Q' * Yt)
        nA = norm(Yt) + eps(T)
        nR = norm(resid)
        tol = max(float(atol), float(rtol) * nA)
        if nR <= tol
            streak += 1
            streak >= 2 && break
        else
            streak = 0
        end
    end

    # B' ≈ Q' * A_IJ  via A_IJ' * Q  (need adjoint) or sample e_j
    # Use adjoint sampler if available; else column probing of J
    k = size(Q, 2)
    if k == 0
        return RkMatrix(zeros(T, m, 0), zeros(T, n, 0))
    end
    Bt = _sample_block_row_action(S, I, J, Q, Xbuf, Ybuf)
    # A ≈ Q * Bt, RkMatrix stores A_fac * B_fac' so B_fac = Bt'
    Bfac = collect(Bt')
    R = RkMatrix(Matrix(Q), Bfac)
    return compress!(R, TSVD(; rtol=float(rtol), atol=float(atol), rank=rmax))
end

function _sample_block_row_action(
        S::AbstractMatvecSampler,
        I::UnitRange{Int},
        J::UnitRange{Int},
        Q::AbstractMatrix,
        Xbuf::AbstractMatrix,
        Ybuf::AbstractMatrix,
    )
    T = eltype(Q)
    m, k = size(Q)
    n = length(J)
    N = size(S, 2)
    M = size(S, 1)
    # Prefer adjoint: Z = A' * E_I * Q  => Z[J,:] = A_IJ' * Q
    try
        X = zeros(T, M, k)
        X[I, :] .= Q
        Z = zeros(T, N, k)
        mul!(Z, adjoint(S), X)
        return collect(Z[J, :]')  # k×n = Q' A_IJ
    catch
        # fallback: probe each column of J (slow)
        Bt = zeros(T, k, n)
        x = zeros(T, N)
        y = zeros(T, M)
        for (jj, j) in enumerate(J)
            fill!(x, 0)
            x[j] = one(T)
            fill!(y, 0)
            # single column via thin matrix
            X1 = reshape(x, :, 1)
            Y1 = reshape(y, :, 1)
            mul!(Y1, S, X1)
            Bt[:, jj] = Q' * view(y, I)
        end
        return Bt
    end
end

function _dense_block_from_sampler(
        S::AbstractMatvecSampler,
        I::UnitRange{Int},
        J::UnitRange{Int},
    )
    T = Float64
    m, n = length(I), length(J)
    N = size(S, 2)
    M = size(S, 1)
    # A_IJ = A * E_J restricted to I (Identity — not the range `I`)
    Ω = Matrix{T}(LinearAlgebra.I, n, n)
    Xbuf = zeros(T, N, n)
    Ybuf = zeros(T, M, n)
    YI = zeros(T, m, n)
    _sample_block_action!(YI, S, I, J, Ω, Xbuf, Ybuf)
    return YI
end

# ---------------------------------------------------------------------------
# HARA: fill classic H skeleton from sampler
# ---------------------------------------------------------------------------

"""
    hara(S::AbstractMatvecSampler, rowtree, coltree; kwargs...) -> HMatrix

**HARA-style** assembly of a classic [`HMatrix`](@ref) using only matvecs of `S`.

Builds the usual admissible/dense block structure, then:
- admissible leaves → adaptive randomized low-rank ([`RkMatrix`](@ref));
- dense leaves → block recovered by sampling the identity on columns.

# Keywords
- `adm`, `rtol`, `atol`, `rank`, `batch` — structure / accuracy
- `global_index=true` — trees define a permutation; sampler is assumed in
  **global** order and is wrapped with [`PermutedMatrix`](@ref)-style applies

This is an MVP (classic H, not nested H²). Sufficient for black-box recompression
and product operators `v ↦ A(Bv)`.
"""
function hara(
        S::AbstractMatvecSampler,
        rowtree::R,
        coltree::R;
        adm = StrongAdmissibilityStd(3),
        rtol = 1e-4,
        atol = 0.0,
        rank = typemax(Int),
        batch = 8,
        global_index = use_global_index(),
        threads = false,
    ) where {R}
    T = Float64
    H = HMatrix{T}(rowtree, coltree, adm)
    # local sampler: A_loc = P_r * A_glob * P_c'
    Sloc = if global_index
        rp, cp = loc2glob(rowtree), loc2glob(coltree)
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
        FunctionSampler(f!, size(S, 2); m = size(S, 1), f_adj! = f_adj!)
    else
        S
    end
    _hara_fill!(H, Sloc; rtol=float(rtol), atol=float(atol), rank=rank, batch=batch)
    return H
end

function hara(K::AbstractMatrix, rowtree, coltree; kwargs...)
    return hara(KernelMatvecSampler(K), rowtree, coltree; kwargs...)
end

function _hara_fill!(H::HMatrix, S::AbstractMatvecSampler; rtol, atol, rank, batch)
    if isleaf(H)
        I = rowrange(H)
        J = colrange(H)
        if isadmissible(H)
            R = _rand_lr_block(S, I, J; rtol=rtol, atol=atol, rank=rank, batch=batch)
            setdata!(H, R)
        else
            D = _dense_block_from_sampler(S, I, J)
            setdata!(H, D)
        end
        return H
    end
    for ch in children(H)
        _hara_fill!(ch, S; rtol=rtol, atol=atol, rank=rank, batch=batch)
    end
    return H
end
