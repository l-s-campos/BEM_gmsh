# Anisotropic / orthotropic Laplace (steady heat): ∇·(K ∇u) = 0
# Φ, q* = −K∇Φ·n  (Chang–Tadeu / standard anisotropic Green's function).
# BIE flux q is the physical flux n·K∇u.

export OrthotropicLaplace, AnisotropicLaplace, conductivity_K

"""
    AnisotropicLaplace(K)

Steady anisotropic heat conduction ``∇·(K ∇u)=0`` with SPD ``K`` (2×2 or 3×3).
BIE flux follows the package convention ``q = -n·K∇u`` (same as `Laplace`).
``q^* = -K∇Φ·n = (r·n) / (c ρ^d)`` because ``K K^{-1} r = r``.
"""
struct AnisotropicLaplace{D,T,L} <: Scalar
    K::SMatrix{D,D,T,L}
    Kinv::SMatrix{D,D,T,L}
    sdet::T
end

function AnisotropicLaplace(K::AbstractMatrix)
    D = size(K, 1)
    size(K, 2) == D || throw(ArgumentError("K must be square"))
    D == 2 || D == 3 || throw(ArgumentError("AnisotropicLaplace is 2D or 3D"))
    Kd = SMatrix{D,D,Float64}(Float64.(K))
    det(Kd) > 0 || throw(ArgumentError("K must be SPD (det>0)"))
    return AnisotropicLaplace{D,Float64,D * D}(Kd, inv(Kd), sqrt(det(Kd)))
end

AnisotropicLaplace(; kx::Real=1.0, ky::Real=1.0, kz::Union{Real,Nothing}=nothing) =
    kz === nothing ? AnisotropicLaplace(@SMatrix [kx 0; 0 ky]) :
                    AnisotropicLaplace(@SMatrix [kx 0 0; 0 ky 0; 0 0 kz])

"""
    OrthotropicLaplace(; k1=1.0, k2=1.0)

2D diagonal conductivity ``K=\\mathrm{diag}(k_1,k_2)``. Same kernels as
[`AnisotropicLaplace`](@ref).
"""
@kwdef mutable struct OrthotropicLaplace{T} <: Scalar
    k1::T = 1.0
    k2::T = 1.0
end

conductivity_K(p::Laplace) = @SMatrix [p.k 0; 0 p.k]
conductivity_K(p::OrthotropicLaplace) = @SMatrix [p.k1 0; 0 p.k2]
conductivity_K(p::AnisotropicLaplace) = p.K

function _aniso_kernels(Kinv, sdet, r::SVector{2}, n::SVector{2})
    ρ2 = dot(r, Kinv * r)
    ρ2 = max(ρ2, 1e-30)
    G = -log(sqrt(ρ2)) / (2π * sdet)
    # q* = −K∇Φ·n = (r·n) / (2π √detK ρ²)   (K K⁻¹ r = r)
    H = dot(r, n) / (2π * sdet * ρ2)
    return KernelPair(G, H)
end

function _aniso_kernels(Kinv, sdet, r::SVector{3}, n::SVector{3})
    ρ2 = dot(r, Kinv * r)
    ρ = sqrt(max(ρ2, 1e-30))
    G = 1 / (4π * sdet * ρ)
    H = dot(r, n) / (4π * sdet * ρ2 * ρ)
    return KernelPair(G, H)
end

function fundamental(props::OrthotropicLaplace, r::SVector{2}, n::SVector{2})
    k1, k2 = float(props.k1), float(props.k2)
    Kinv = @SMatrix [1/k1 0; 0 1/k2]
    return _aniso_kernels(Kinv, sqrt(k1 * k2), r, n)
end

function fundamental(props::AnisotropicLaplace{2}, r::SVector{2}, n::SVector{2})
    return _aniso_kernels(props.Kinv, props.sdet, r, n)
end

function fundamental(props::AnisotropicLaplace{3}, r::SVector{3}, n::SVector{3})
    return _aniso_kernels(props.Kinv, props.sdet, r, n)
end

function fundamental(dad::BEMdata{<:OrthotropicLaplace}, r::Point2D, n::Point2D)
    kp = fundamental(dad.properties, r, n)
    return kp.U, kp.T
end
function fundamental(dad::BEMdata{<:AnisotropicLaplace{2}}, r::Point2D, n::Point2D)
    kp = fundamental(dad.properties, r, n)
    return kp.U, kp.T
end
function fundamental(dad::BEMdata{<:AnisotropicLaplace{3}}, r::Point3D, n::Point3D)
    kp = fundamental(dad.properties, r, n)
    return kp.U, kp.T
end

is_isotropic(p::OrthotropicLaplace; tol=1e-12) = abs(p.k1 - p.k2) < tol * (1 + abs(p.k1))
function is_isotropic(p::AnisotropicLaplace; tol=1e-12)
    K = p.K
    k = K[1, 1]
    return norm(K - k * I) < tol * (1 + abs(k))
end

const LaplaceLike = Union{Laplace,OrthotropicLaplace,AnisotropicLaplace,Helmholtz}
