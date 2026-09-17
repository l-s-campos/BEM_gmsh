# Shared kernel containers and helpers for all problem types
export fundamental, fundamental_U, fundamental_T, fundamental_hyper, fundamental_stress, fundamental_grad
export lekhnitskii_params, lekhnitskii_engineering, lekhnitskii_rotate, wavenumber
export KernelPair, StressKernels

using Tensorial: Mat, Vec, Tensor

const _I2 = one(Mat{2,2})
const _I3 = one(Mat{3,3})
@inline _otimes(a::Vec, b::Vec) = a * b'
@inline _dot(a::Vec, b::Vec) = sum(a[i] * b[i] for i in eachindex(a))
@inline _R2(r) = LinearAlgebra.dot(r, r)
@inline _R(r) = sqrt(_R2(r))
@inline _to_vec(r::SVector{N,T}) where {N,T} = Vec{N,T}(ntuple(i -> r[i], N))
@inline _to_vec(r::Vec) = r
@inline _to_smat(m::Mat{M,N,T}) where {M,N,T} = SMatrix{M,N,T}(ntuple(i -> m[i], M * N))

"""
    KernelPair{U,T}

Single-layer (`U`/`G`) and double-layer (`T`/`H`) kernels.
"""
struct KernelPair{U,T}
    U::U
    T::T
end
Base.iterate(k::KernelPair, s=1) = s == 1 ? (k.U, 2) : s == 2 ? (k.T, 3) : nothing
Base.length(::KernelPair) = 2

"""Third-order stress kernels `D`, `S` for interior recovery."""
struct StressKernels{D,S}
    D::D
    S::S
end
Base.iterate(k::StressKernels, s=1) = s == 1 ? (k.D, 2) : s == 2 ? (k.S, 3) : nothing

function Base.show(io::IO, ::MIME"text/plain", kp::KernelPair)
    println(io, "KernelPair{$(typeof(kp.U)), $(typeof(kp.T))}")
    println(io, "  U (single-layer / G) = ", kp.U)
    print(io, "  T (double-layer / H) = ", kp.T)
end
