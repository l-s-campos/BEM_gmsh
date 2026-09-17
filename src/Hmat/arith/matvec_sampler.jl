# Black-box operators accessible only through multi-RHS matvecs.

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
