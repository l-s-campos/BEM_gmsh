# =============================================================================
# HSS basis (interpolative decomposition) — compatible with BEM Hmat HSSBasisID
# =============================================================================

"""
    struct HSSBasisID{T}

Interpolative decomposition: `U = P * [I; E]` with skeleton rows `P[1:r]`.
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

"""Row interpolative decomposition of `A` (m × n), tolerance `rtol`."""
function row_id(A::AbstractMatrix{T}, rtol::Real, rmax::Int) where {T}
    m, n = size(A)
    rmax = min(rmax, m, n)
    if m == 0 || n == 0 || rmax == 0
        return HSSBasisID{T}(m, 0), Int[]
    end
    F = qr(Matrix(A'), ColumnNorm())
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
