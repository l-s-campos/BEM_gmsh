# =============================================================================
# Tensor kernel helpers (SMatrix entries, flat ↔ blocked vectors)
# =============================================================================

"""
    tensor_blocksize(T)

For a scalar type return `(1,1)`. For `SMatrix{p,q}` return `(p,q)`.
"""
tensor_blocksize(::Type{<:Number}) = (1, 1)
tensor_blocksize(::Type{<:SMatrix{p, q}}) where {p, q} = (p, q)
tensor_blocksize(::T) where {T} = tensor_blocksize(T)

"""
    is_tensor_eltype(T) -> Bool

`true` when `T` is a static matrix used as a tensor-valued kernel entry.
"""
is_tensor_eltype(::Type{<:Number}) = false
is_tensor_eltype(::Type{<:SMatrix}) = true
is_tensor_eltype(::Type) = false
is_tensor_eltype(::T) where {T} = is_tensor_eltype(T)

# ---- range / index maps -----------------------------------------------------

"""
    expand_range(r::UnitRange, p::Int) -> UnitRange

Map a block-index range `r` to the corresponding scalar-index range when each
block contributes `p` scalar rows/columns: `i ↦ (p*(i-1)+1):(p*i)`.
"""
function expand_range(r::UnitRange{<:Integer}, p::Integer)
    p == 1 && return Int(first(r)):Int(last(r))
    a, b = Int(first(r)), Int(last(r))
    return (p * (a - 1) + 1):(p * b)
end

"""
    block_index(i_scalar, p) -> (i_block, i_local)

Convert a 1-based scalar index to `(block index, local index in 1:p)`.
"""
function block_index(i::Integer, p::Integer)
    p == 1 && return Int(i), 1
    i0 = Int(i) - 1
    return div(i0, p) + 1, (i0 % p) + 1
end

"""
    scalar_index(i_block, i_local, p) -> Int
"""
scalar_index(i_block::Integer, i_local::Integer, p::Integer) =
    p * (Int(i_block) - 1) + Int(i_local)

function _expand_perm(perm::Vector{Int}, p::Int)
    n = length(perm)
    out = Vector{Int}(undef, p * n)
    @inbounds for (i, g) in enumerate(perm)
        for a in 1:p
            out[p * (i - 1) + a] = p * (g - 1) + a
        end
    end
    return out
end

# ---- vector flatten / unflatten --------------------------------------------

"""
    scalarize(v)

Flatten a vector of tensor entries into a contiguous scalar vector.
- `Vector{<:Number}` → copy
- `Vector{SVector{p,T}}` → `Vector{T}` of length `p*n`
"""
scalarize(v::AbstractVector{<:Number}) = collect(v)

function scalarize(v::AbstractVector{SVector{p, T}}) where {p, T}
    n = length(v)
    out = Vector{T}(undef, p * n)
    @inbounds for i in 1:n
        base = p * (i - 1)
        vi = v[i]
        for a in 1:p
            out[base + a] = vi[a]
        end
    end
    return out
end

"""
    descalarize(v, ::Type{SVector{p,T}})

Inverse of [`scalarize`](@ref) for `SVector{p}` densities.
"""
function descalarize(v::AbstractVector{T}, ::Type{SVector{p, T}}) where {p, T}
    length(v) % p == 0 || throw(DimensionMismatch("length $(length(v)) not divisible by $p"))
    n = length(v) ÷ p
    out = Vector{SVector{p, T}}(undef, n)
    @inbounds for i in 1:n
        base = p * (i - 1)
        out[i] = SVector{p, T}(ntuple(a -> v[base + a], p))
    end
    return out
end

descalarize(v::AbstractVector{T}, ::Val{p}) where {T, p} = descalarize(v, SVector{p, T})

@inline function _svload(::Val{d}, x::AbstractVector{T}, j::Integer) where {d, T}
    o = d * (Int(j) - 1)
    return SVector{d, T}(ntuple(Val(d)) do a
        @inbounds x[o + a]
    end)
end

@inline function _svadd!(y::AbstractVector{T}, j::Integer, v::SVector{d, T}) where {d, T}
    o = d * (Int(j) - 1)
    @inbounds for a in 1:d
        y[o + a] += v[a]
    end
    return y
end

# Matrix{SMatrix} × Vector{SVector} at H-matrix dense leaves.
function LinearAlgebra.mul!(
        C::AbstractVector{SVector{p, T}},
        A::Matrix{S},
        x::AbstractVector{SVector{q, T}},
        α::Number = true,
        β::Number = false,
    ) where {p, q, T, S <: SMatrix{p, q, T}}
    m, n = size(A)
    length(C) == m && length(x) == n || throw(DimensionMismatch())
    if iszero(β)
        @inbounds for i in 1:m
            s = zero(SVector{p, T})
            for j in 1:n
                s += A[i, j] * x[j]
            end
            C[i] = α * s
        end
    else
        @inbounds for i in 1:m
            s = β * C[i]
            for j in 1:n
                s += α * (A[i, j] * x[j])
            end
            C[i] = s
        end
    end
    return C
end

# Matrix{SMatrix} × flat node-major vector (length q n → p m).
function LinearAlgebra.mul!(
        C::AbstractVector{T},
        A::Matrix{S},
        x::AbstractVector{T},
        α::Number = true,
        β::Number = false,
    ) where {T <: Number, p, q, S <: SMatrix{p, q, T}}
    m, n = size(A)
    length(C) == p * m && length(x) == q * n || throw(DimensionMismatch())
    if iszero(β)
        fill!(C, zero(T))
    elseif β != 1
        rmul!(C, β)
    end
    @inbounds for j in 1:n
        xj = α * _svload(Val(q), x, j)
        for i in 1:m
            _svadd!(C, i, A[i, j] * xj)
        end
    end
    return C
end

function LinearAlgebra.mul!(
        C::AbstractVector{T},
        At::Adjoint{<:Any, Matrix{S}},
        x::AbstractVector{T},
        α::Number = true,
        β::Number = false,
    ) where {T <: Number, p, q, S <: SMatrix{p, q, T}}
    A = parent(At)
    m, n = size(A)
    length(C) == q * n && length(x) == p * m || throw(DimensionMismatch())
    if iszero(β)
        fill!(C, zero(T))
    elseif β != 1
        rmul!(C, β)
    end
    @inbounds for j in 1:n
        s = zero(SVector{q, T})
        for i in 1:m
            s += adjoint(A[i, j]) * _svload(Val(p), x, i)
        end
        _svadd!(C, j, α * s)
    end
    return C
end
