# =============================================================================
# Body-style API (inspired by exafmm-t)
#
# Bodies carry coordinates, charge, and output potential / force.
# =============================================================================

"""
    mutable struct Body{D,T}

N-body particle with coordinates `X`, charge `q`, and output fields
`p` (potential) and `F` (force = −∇p, optional).

# Example
```julia
bodies = [Body(SVector(rand(),rand(),rand()), randn()) for _ in 1:1000]
evaluate!(bodies, Laplace3D(eps=1e-8))
```
"""
mutable struct Body{D,T}
    X::SVector{D,Float64}
    q::T
    p::T
    F::SVector{D,T}
end

function Body(X::SVector{D,<:Real}, q::T) where {D,T}
    z = zero(T)
    return Body{D,T}(SVector{D,Float64}(X), q, z, ntuple(_ -> z, D) |> SVector{D,T})
end
Body(x::NTuple{D,<:Real}, q) where {D} = Body(SVector{D,Float64}(x...), q)
Body(x::AbstractVector{<:Real}, q) = Body(SVector{length(x),Float64}(x...), q)

"""Abstract kernel descriptor for [`evaluate!`](@ref)."""
abstract type AbstractKernel end

Base.@kwdef struct Laplace2D <: AbstractKernel
    eps::Float64 = 1e-8
    nmax::Int = 50
    η::Float64 = 1.0
end

Base.@kwdef struct Laplace3D <: AbstractKernel
    eps::Float64 = 1e-8
    nmax::Int = -1
    η::Float64 = 1.0
    threaded::Bool = false
end

Base.@kwdef struct Helmholtz2D <: AbstractKernel
    eps::Float64 = 1e-8
    zk::ComplexF64 = 1.0 + 0.0im
    nmax::Int = 40
    η::Float64 = 1.2
end

Base.@kwdef struct Helmholtz3D <: AbstractKernel
    eps::Float64 = 1e-8
    zk::ComplexF64 = 1.0 + 0.0im
    nmax::Int = 40
    η::Float64 = 1.2
    threaded::Bool = false
end

Base.@kwdef struct Yukawa2D <: AbstractKernel
    eps::Float64 = 1e-8
    κ::Float64 = 1.0
    nmax::Int = 40
    η::Float64 = 1.2
end

Base.@kwdef struct Yukawa3D <: AbstractKernel
    eps::Float64 = 1e-8
    κ::Float64 = 1.0
    nmax::Int = 40
    η::Float64 = 1.2
    threaded::Bool = false
end

function _bodies_to_arrays(bodies::Vector{Body{D,T}}) where {D,T}
    n = length(bodies)
    X = Matrix{Float64}(undef, D, n)
    q = Vector{T}(undef, n)
    @inbounds for i in 1:n
        for d in 1:D
            X[d, i] = bodies[i].X[d]
        end
        q[i] = bodies[i].q
    end
    return X, q
end

function _write_pot!(bodies::Vector{Body{D,T}}, pot::AbstractVector) where {D,T}
    @inbounds for i in eachindex(bodies)
        bodies[i].p = pot[i]
    end
    return bodies
end

function _write_grad!(bodies::Vector{Body{D,T}}, grad::AbstractMatrix) where {D,T}
    # F = -∇p
    @inbounds for i in eachindex(bodies)
        bodies[i].F = SVector{D,T}(ntuple(d -> -grad[d, i], D))
    end
    return bodies
end

"""
    evaluate!(bodies, kernel; force=false)

Compute N-body potentials (and optionally forces) for `bodies` with the given
kernel, writing results into each `Body`'s `p` (and `F`) fields.

# Kernels
- [`Laplace2D`](@ref) / [`Laplace3D`](@ref)
- [`Helmholtz2D`](@ref) / [`Helmholtz3D`](@ref)
- [`Yukawa2D`](@ref) / [`Yukawa3D`](@ref)  (modified Helmholtz, from exafmm-t)
"""
function evaluate! end

function evaluate!(bodies::Vector{Body{2,T}}, k::Laplace2D; force::Bool=false) where {T<:Real}
    X, q = _bodies_to_arrays(bodies)
    pg = force ? 2 : 1
    vals = rfmm2d(k.eps, X; charges=q, pg=pg, nmax=k.nmax, η=k.η)
    _write_pot!(bodies, vals.pot)
    force && _write_grad!(bodies, vals.grad)
    return bodies
end

function evaluate!(bodies::Vector{Body{2,T}}, k::Laplace2D; force::Bool=false) where {T<:Complex}
    X, q = _bodies_to_arrays(bodies)
    pg = force ? 2 : 1
    vals = lfmm2d(k.eps, X; charges=q, pg=pg, nmax=k.nmax, η=k.η)
    _write_pot!(bodies, vals.pot)
    force && _write_grad!(bodies, vals.grad)
    return bodies
end

function evaluate!(bodies::Vector{Body{3,T}}, k::Laplace3D; force::Bool=false) where {T<:Real}
    X, q = _bodies_to_arrays(bodies)
    pg = force ? 2 : 1
    vals = lfmm3d(k.eps, X; charges=q, pg=pg, nmax=k.nmax, η=k.η, threaded=k.threaded)
    _write_pot!(bodies, vals.pot)
    force && vals.grad !== nothing && _write_grad!(bodies, vals.grad)
    return bodies
end

function evaluate!(bodies::Vector{Body{2,T}}, k::Helmholtz2D; force::Bool=false) where {T}
    X, q = _bodies_to_arrays(bodies)
    vals = hfmm2d(k.eps, k.zk, X; charges=complex.(q), pg=1, nmax=k.nmax, η=k.η)
    _write_pot!(bodies, T.(vals.pot))
    force && @warn "Helmholtz2D force via evaluate! not yet implemented"
    return bodies
end

function evaluate!(bodies::Vector{Body{3,T}}, k::Helmholtz3D; force::Bool=false) where {T}
    X, q = _bodies_to_arrays(bodies)
    vals = hfmm3d(k.eps, k.zk, X; charges=complex.(q), pg=1, nmax=k.nmax, η=k.η, threaded=k.threaded)
    _write_pot!(bodies, T.(vals.pot))
    force && @warn "Helmholtz3D force via evaluate! not yet implemented"
    return bodies
end

function evaluate!(bodies::Vector{Body{2,T}}, k::Yukawa2D; force::Bool=false) where {T<:Real}
    X, q = _bodies_to_arrays(bodies)
    vals = yfmm2d(k.eps, k.κ, X; charges=q, pg=1, nmax=k.nmax, η=k.η)
    _write_pot!(bodies, vals.pot)
    force && @warn "Yukawa2D force via evaluate! not yet implemented"
    return bodies
end

function evaluate!(bodies::Vector{Body{3,T}}, k::Yukawa3D; force::Bool=false) where {T<:Real}
    X, q = _bodies_to_arrays(bodies)
    vals = yfmm3d(k.eps, k.κ, X; charges=q, pg=1, nmax=k.nmax, η=k.η, threaded=k.threaded)
    _write_pot!(bodies, vals.pot)
    force && @warn "Yukawa3D force via evaluate! not yet implemented"
    return bodies
end
