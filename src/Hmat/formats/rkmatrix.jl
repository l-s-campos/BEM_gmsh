"""
    mutable struct RkMatrix{T}

Representation of a rank `r` matrix `M` in outer product format `M =
A*adjoint(B)` where `A` has size `m × r` and `B` has size `n × r`.

The internal representation stores `A` and `B`, but `R.Bt` or `R.At` can be used
to get the respective adjoints.
"""
mutable struct RkMatrix{T} <: AbstractMatrix{T}
    A::Matrix{T}
    B::Matrix{T}
    function RkMatrix(A::Matrix{T}, B::Matrix{T}) where {T}
        @assert size(A, 2) == size(B, 2) "second dimension of `A` and `B` must match"
        m, r = size(A)
        n = size(B, 1)
        if r * (m + n) >= m * n
            @debug "Inefficient RkMatrix:" size(A) size(B)
        end
        return new{T}(A, B)
    end
end

function Base.setproperty!(R::RkMatrix, s::Symbol, mat::Base.Matrix)
    setfield!(R, s, mat)
    return R
end

function Base.show(io::IO, ::MIME"text/plain", R::RkMatrix)
    return print(
        io,
        size(R, 1),
        "×",
        size(R, 2),
        " RkMatrix{",
        eltype(R),
        "}",
        " of rank ",
        rank(R),
    )
end

function Base.getindex(::RkMatrix, args...)
    msg = """method `getindex(::RkMatrix,args...)` has been disabled to avoid
    performance pitfalls. Unless you made an explicit call to `getindex`, this
    error usually means that a linear algebra routine involving an
    `RkMatrix` has fallen back to a generic implementation."""
    return error(msg)
end

"""
    getcol!(col,M::AbstractMatrix,j)

Fill the entries of `col` with column `j` of `M`.
"""
function getcol!(col, R::RkMatrix, j::Int, ::Val{T} = Val(false)) where {T}
    # NOTE: using a `Val` argument to dispatch on the type of `T` is important
    # for performance
    return if T
        mul!(col, R.A, view(adjoint(R.B), :, j), true, true)
    else
        mul!(col, R.A, view(adjoint(R.B), :, j))
    end
end

const AdjRk = Adjoint{<:Any, <:RkMatrix}
const AdjRkOrRk = Union{AdjRk, RkMatrix}

function getcol!(
        col,
        Ra::Adjoint{<:Any, <:RkMatrix},
        j::Int,
        ::Val{T} = Val(false),
    ) where {T}
    # NOTE: using a `Val` argument to dispatch on the type of `T` is important
    # for performance
    R = parent(Ra)
    return if T
        mul!(col, R.B, view(adjoint(R.A), :, j), true, true)
    else
        mul!(col, R.B, view(adjoint(R.A), :, j))
    end
end

Base.copy(R::RkMatrix) = RkMatrix(copy(R.A), copy(R.B))

Base.eltype(::RkMatrix{T}) where {T} = T
Base.size(rmat::RkMatrix) = (size(rmat.A, 1), size(rmat.B, 1))
Base.length(rmat::RkMatrix) = prod(size(rmat))
function Base.isapprox(rmat::RkMatrix, B::AbstractArray, args...; kwargs...)
    return isapprox(Matrix(rmat), B, args...; kwargs...)
end

LinearAlgebra.rank(M::RkMatrix) = size(M.A, 2)

function Base.getproperty(R::RkMatrix, s::Symbol)
    if s == :Bt
        return adjoint(R.B)
    elseif s == :At
        return adjoint(R.A)
    else
        return getfield(R, s)
    end
end

function Base.Matrix(R::RkMatrix{<:Number})
    return R.A * R.Bt
end
function Base.Matrix(adjR::Adjoint{<:Any, <:RkMatrix})
    R = parent(adjR)
    return R.B * R.At
end
function Base.Matrix(R::RkMatrix{<:SMatrix})
    # collect must be used when we have a matrix of `SMatrix` because of this issue:
    # https://github.com/JuliaArrays/StaticArrays.jl/issues/966#issuecomment-943679214
    return R.A * collect(R.Bt)
end

function LinearAlgebra.mul!(
        C::AbstractVector{SVector{p, T}},
        Rk::RkMatrix{S},
        F::AbstractVector{SVector{q, T}},
        a::Number,
        b::Number,
    ) where {p, q, T, S <: SMatrix{p, q, T}}
    m, n = size(Rk)
    r = rank(Rk)
    tmp = Rk.Bt * F
    rmul!(C, b)
    for k in 1:r
        tmp[k] *= a
        for i in 1:m
            C[i] += Rk.A[i, k] * tmp[k]
        end
    end
    return C
end

# RkMatrix{SMatrix} × flat node-major vector. `tmp` is length-r of SVector{p}.
function LinearAlgebra.mul!(
        C::AbstractVector{T},
        Rk::RkMatrix{S},
        F::AbstractVector{T},
        a::Number = true,
        b::Number = false,
    ) where {T <: Number, p, q, S <: SMatrix{p, q, T}}
    m, n = size(Rk)
    r = rank(Rk)
    length(C) == p * m && length(F) == q * n || throw(DimensionMismatch())
    if iszero(b)
        fill!(C, zero(T))
    elseif b != 1
        rmul!(C, b)
    end
    tmp = Vector{SVector{p, T}}(undef, r)
    @inbounds for k in 1:r
        s = zero(SVector{p, T})
        for j in 1:n
            s += adjoint(Rk.B[j, k]) * _svload(Val(q), F, j)
        end
        tmp[k] = a * s
    end
    @inbounds for k in 1:r
        tk = tmp[k]
        for i in 1:m
            _svadd!(C, i, Rk.A[i, k] * tk)
        end
    end
    return C
end

function LinearAlgebra.mul!(
        C::AbstractVector{T},
        Rt::Adjoint{<:Any, RkMatrix{S}},
        F::AbstractVector{T},
        a::Number = true,
        b::Number = false,
    ) where {T <: Number, p, q, S <: SMatrix{p, q, T}}
    Rk = parent(Rt)
    m, n = size(Rk)
    r = rank(Rk)
    length(C) == q * n && length(F) == p * m || throw(DimensionMismatch())
    if iszero(b)
        fill!(C, zero(T))
    elseif b != 1
        rmul!(C, b)
    end
    tmp = Vector{SVector{p, T}}(undef, r)
    @inbounds for k in 1:r
        s = zero(SVector{p, T})
        for i in 1:m
            s += adjoint(Rk.A[i, k]) * _svload(Val(p), F, i)
        end
        tmp[k] = a * s
    end
    @inbounds for k in 1:r
        tk = tmp[k]
        for j in 1:n
            _svadd!(C, j, Rk.B[j, k] * tk)
        end
    end
    return C
end

# scalar multiplication
Base.:*(a::Number, R::RkMatrix) = (A = a * R.A; B = copy(R.B); RkMatrix(A, B))

# RkMatrix × vector / matrix — avoid getindex fallback.
# Small task-local scratch avoids allocating B'*X on every far interaction.
const _RK_TMP = [Matrix{Float64}(undef, 0, 0) for _ in 1:Threads.nthreads()]
const _RK_TMP_LOCK = ReentrantLock()

function _rk_tmp!(::Type{T}, r::Int, s::Int) where {T}
    tid = Threads.threadid()
    if tid > length(_RK_TMP)
        lock(_RK_TMP_LOCK) do
            n = tid
            isdefined(Threads, :maxthreadid) && (n = max(n, Threads.maxthreadid()))
            while length(_RK_TMP) < n
                push!(_RK_TMP, Matrix{Float64}(undef, 0, 0))
            end
        end
    end
    buf = _RK_TMP[tid]
    if eltype(buf) != T || size(buf, 1) < r || size(buf, 2) < s
        buf = Matrix{T}(undef, max(r, size(buf, 1)), max(s, size(buf, 2)))
        _RK_TMP[tid] = buf
    end
    return view(buf, 1:r, 1:s)
end

function LinearAlgebra.mul!(
        C::AbstractVector,
        R::RkMatrix{T},
        x::AbstractVector,
        α::Number = true,
        β::Number = false,
    ) where {T <: Number}
    r = size(R.A, 2)
    tmp = _rk_tmp!(T, r, 1)
    tv = view(tmp, :, 1)
    mul!(tv, adjoint(R.B), x)
    return mul!(C, R.A, tv, α, β)
end
function LinearAlgebra.mul!(
        C::AbstractMatrix,
        R::RkMatrix{T},
        X::AbstractMatrix,
        α::Number = true,
        β::Number = false,
    ) where {T <: Number}
    r = size(R.A, 2)
    s = size(X, 2)
    tmp = _rk_tmp!(T, r, s)
    mul!(tmp, adjoint(R.B), X)
    return mul!(C, R.A, tmp, α, β)
end
function LinearAlgebra.mul!(
        C::AbstractVector,
        Rt::Adjoint{T, <:RkMatrix{T}},
        x::AbstractVector,
        α::Number = true,
        β::Number = false,
    ) where {T <: Number}
    R = parent(Rt)
    r = size(R.A, 2)
    tmp = _rk_tmp!(T, r, 1)
    tv = view(tmp, :, 1)
    mul!(tv, adjoint(R.A), x)
    return mul!(C, R.B, tv, α, β)
end
function LinearAlgebra.mul!(
        C::AbstractMatrix,
        Rt::Adjoint{T, <:RkMatrix{T}},
        X::AbstractMatrix,
        α::Number = true,
        β::Number = false,
    ) where {T <: Number}
    R = parent(Rt)
    r = size(R.A, 2)
    s = size(X, 2)
    tmp = _rk_tmp!(T, r, s)
    mul!(tmp, adjoint(R.A), X)
    return mul!(C, R.B, tmp, α, β)
end

Base.:*(R::RkMatrix{<:Number}, x::AbstractVector) =
    mul!(similar(x, eltype(R), size(R, 1)), R, x)
Base.:*(R::RkMatrix{<:Number}, X::AbstractMatrix) =
    mul!(similar(X, eltype(R), size(R, 1), size(X, 2)), R, X)
Base.:*(Rt::Adjoint{<:Number, <:RkMatrix{<:Number}}, x::AbstractVector) =
    mul!(similar(x, eltype(Rt), size(Rt, 1)), Rt, x)
Base.:*(Rt::Adjoint{<:Number, <:RkMatrix{<:Number}}, X::AbstractMatrix) =
    mul!(similar(X, eltype(Rt), size(Rt, 1), size(X, 2)), Rt, X)
