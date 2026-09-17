# Column interpolative decomposition (Cheng–Gimbutas–Martinsson–Rokhlin)
# with Gu–Eisenstat RRQR so max|T| ≤ Tmax without inflating rank.
# A[:, rd] ≈ A[:, sk] * T  with T of size |sk| × |rd|.
# Large Kid uses a Gaussian row-sketch then the same ID (Liberty–Woolfe–
# Martinsson–Rokhlin–Tygert), matching LowRankApprox `id(; sketch=:randn)`.

const _ID_SKETCH_MN = 80_000
const _ID_SKETCH_MIN = 64
const _ID_SKETCH_OVER = 8

"""
    interpolative_decomp(A; rtol=1e-6, rank=typemax(Int), Tmax=2, rrqr_iter=64, sketch=:auto)

Column ID of `A`. Returns `(sk, rd, T)` such that `A[:, rd] ≈ A[:, sk] * T`.
Rank is `min(count(σ > rtol*σ₁), rank)`. If `Tmax < Inf`, Gu–Eisenstat
column swaps on the truncated `k×n` factor keep `max|T| ≤ Tmax` (FLAM
default `Tmax=2`) and may drop rank.

`sketch=:auto` (default) uses a Gaussian row-sketch when `A` is tall and
large (`m ≥ 2n` and `m*n ≥ 80_000`); `sketch=false` disables it.
Wide blocks keep early-terminating GEQP3 so HSS proxy ranks are not cut.
"""
function interpolative_decomp(
        A::AbstractMatrix{T};
        rtol = 1e-6,
        rank = typemax(Int),
        Tmax = 2,
        rrqr_iter::Integer = 64,
        sketch = :auto,
    ) where {T}
    m, n = size(A)
    n == 0 && return Int[], Int[], zeros(T, 0, 0)
    m == 0 && return Int[], collect(1:n), zeros(T, 0, n)
    if _id_use_sketch(sketch, m, n, rank)
        return _id_sketch(A; rtol = rtol, rank = rank, Tmax = Tmax,
            rrqr_iter = rrqr_iter)
    end
    return _id_rrqr(A; rtol = rtol, rank = rank, Tmax = Tmax, rrqr_iter = rrqr_iter)
end

function _id_use_sketch(sketch, m::Int, n::Int, rank)
    (sketch === false || sketch === :none) && return false
    sketch === true && return min(m, n) >= 4
    # Row-sketch helps when A is tall. Wide proxy Kid is already cheap
    # for early GEQP3; sketching it can drop kernel rank.
    m < 2 * n && return false
    m * n < _ID_SKETCH_MN && return false
    min(m, n) < _ID_SKETCH_MIN && return false
    r = Int(rank)
    r < typemax(Int) && r >= min(m, n) - 1 && return false
    return true
end

function _id_sketch(A::AbstractMatrix{T}; rtol, rank, Tmax, rrqr_iter) where {T}
    m, n = size(A)
    over = _ID_SKETCH_OVER
    rmax = min(m, n, Int(rank))
    s = min(m, max(over + 8, 32))
    if Int(rank) < typemax(Int)
        s = min(m, max(s, Int(rank) + over))
    end
    local sk, rd, Tm
    while true
        Ω = randn(T, s, m)
        B = Ω * A
        sk, rd, Tm = _id_rrqr(B; rtol = rtol, rank = rank, Tmax = Tmax,
            rrqr_iter = rrqr_iter)
        k = length(sk)
        (k + over <= s || s >= m || k >= rmax) && return sk, rd, Tm
        s = min(m, max(2s, k + over + 8))
    end
end

function _id_rank_R(R::AbstractMatrix, rtol, rank, n::Int)
    mk = min(size(R, 1), n)
    mk < 1 && return 0
    σ1 = abs(R[1, 1])
    τ = float(rtol) * σ1
    k = 0
    @inbounds for i in 1:mk
        abs(R[i, i]) > τ || break
        k = i
    end
    return clamp(k, 1, min(mk, Int(rank)))
end

function _id_gu_eisenstat!(R::Matrix{T}, p::Vector{Int}, k::Int, n::Int,
        tmax, τ, rrqr_iter) where {T}
    if isfinite(tmax) && k < n && Int(rrqr_iter) > 0
        for _ in 1:Int(rrqr_iter)
            Tm = _id_interp(R, k)
            isempty(Tm) && break
            amax, bi, bj = _id_tmax_worst(Tm)
            amax <= tmax && break
            @inbounds for row in 1:k
                R[row, bi], R[row, k + bj] = R[row, k + bj], R[row, bi]
            end
            p[bi], p[k + bj] = p[k + bj], p[bi]
            Fk = qr(view(R, :, 1:k))
            R[:, 1:k] = Fk.R
            R[:, (k + 1):n] = Fk.Q' * R[:, (k + 1):n]
            while k > 1 && abs(R[k, k]) <= τ
                k -= 1
                R = R[1:k, :]
            end
        end
    end
    return R, p, k
end

function _id_rrqr(A::AbstractMatrix{T}; rtol, rank, Tmax, rrqr_iter) where {T}
    m, n = size(A)
    B = Matrix{T}(A)
    if m > 8 * n
        B = Matrix{T}(qr(B).R)
        m = size(B, 1)
    end
    tmax = float(Tmax)
    if min(m, n) <= 64 && m * n <= 16_000
        F = qr(B, ColumnNorm())
        p = Vector{Int}(F.p)
        k = _id_rank_R(F.R, rtol, rank, n)
        k < 1 && return Int[], collect(1:n), zeros(T, 0, n)
        R = Matrix{T}(F.R[1:k, :])
        τ = float(rtol) * abs(R[1, 1])
        R, p, k = _id_gu_eisenstat!(R, p, k, n, tmax, τ, rrqr_iter)
        return p[1:k], p[(k + 1):n], _id_interp(R, k)
    end
    k, p, B = _id_geqp3_early!(B, float(rtol), Int(rank))
    k < 1 && return Int[], collect(1:n), zeros(T, 0, n)
    R = triu(B[1:k, :])
    τ = float(rtol) * abs(R[1, 1])
    R, p, k = _id_gu_eisenstat!(R, p, k, n, tmax, τ, rrqr_iter)
    return p[1:k], p[(k + 1):n], _id_interp(R, k)
end

function _id_geqp3_early!(A::Matrix{T}, rtol, rank) where {T}
    m, n = size(A)
    jpvt = collect(1:n)
    (m < 1 || n < 1) && return 0, jpvt, A
    RT = real(T)
    colnrm = Vector{RT}(undef, n)
    @inbounds for j in 1:n
        s = zero(RT)
        for i in 1:m
            s += abs2(A[i, j])
        end
        colnrm[j] = s
    end
    kmax = min(m, n, Int(rank))
    k = 0
    σ1 = zero(RT)
    for j in 1:kmax
        p = j
        best = colnrm[j]
        @inbounds for q in (j + 1):n
            if colnrm[q] > best
                best = colnrm[q]
                p = q
            end
        end
        if p != j
            @inbounds for i in 1:m
                A[i, j], A[i, p] = A[i, p], A[i, j]
            end
            jpvt[j], jpvt[p] = jpvt[p], jpvt[j]
            colnrm[j], colnrm[p] = colnrm[p], colnrm[j]
        end
        x = view(A, j:m, j)
        τh = LinearAlgebra.reflector!(x)
        if j < n
            LinearAlgebra.reflectorApply!(x, τh, view(A, j:m, (j + 1):n))
        end
        diag = abs(A[j, j])
        if j == 1
            σ1 = diag
        end
        if diag <= rtol * σ1
            break
        end
        k = j
        @inbounds for q in (j + 1):n
            colnrm[q] -= abs2(A[j, q])
            colnrm[q] < 0 && (colnrm[q] = zero(RT))
        end
        if j % 32 == 0
            @inbounds for q in (j + 1):n
                s = zero(RT)
                for i in (j + 1):m
                    s += abs2(A[i, q])
                end
                colnrm[q] = s
            end
        end
    end
    return k, jpvt, A
end

function _id_interp(R::AbstractMatrix{T}, k::Int) where {T}
    n = size(R, 2)
    k >= n && return zeros(T, k, 0)
    k < 1 && return zeros(T, 0, n)
    kk = min(k, size(R, 1))
    return Matrix{T}(UpperTriangular(view(R, 1:kk, 1:kk)) \ R[1:kk, (kk + 1):n])
end

function _id_tmax_worst(Tm::AbstractMatrix)
    best = 0.0
    bi, bj = 1, 1
    @inbounds for j in axes(Tm, 2), i in axes(Tm, 1)
        a = abs(Tm[i, j])
        if a > best
            best = a
            bi, bj = i, j
        end
    end
    return best, bi, bj
end

"""Row ID: `A[rd, :] ≈ T * A[sk, :]` with `T` of size `|rd| × |sk|`."""
function interpolative_decomp_rows(A::AbstractMatrix; kwargs...)
    sk, rd, T = interpolative_decomp(transpose(A); kwargs...)
    return rd, sk, Matrix(transpose(T))
end
