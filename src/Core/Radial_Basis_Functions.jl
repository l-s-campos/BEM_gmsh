# Radial basis functions, monomials, interpolants, and PDE techniques
# Core API kept for DiBFM / thermo / visualization; extensions in RBF_Extensions.jl

export AbstractBasis, AbstractRadialBasis, AbstractPHS
export PHS, PHS1, PHS2, PHS3, PHS4, PHS5, PHS6, PHS7
export FundamentalRBF, fundamental_rbf
export IMQ, Gaussian
export MonomialBasis, degree, dim
export RBF, rbf_weights, rbf_evaluate, rbf_partial_weights, rbf_partial, rbf_cardinal
export rbf_npoly, rbf_length_scale
export ∂
export int, poly_deg

# Own RBF/monomial partial operator. `using Tensorial` brings a *type* `∂` into
# the parent module; without an explicit new generic, method definitions would
# extend that constructor (Julia ≥1.12 warning). This shadows Tensorial.∂ here.
function ∂ end

abstract type AbstractBasis end
abstract type AbstractRadialBasis <: AbstractBasis end
abstract type AbstractPHS <: AbstractRadialBasis end

# =============================================================================
# Helpers
# =============================================================================

function check_poly_deg(poly_deg)
    if poly_deg < -1
        throw(ArgumentError(
            "poly_deg must be >= -1 (got $poly_deg). Use 2 for quadratic, 0 for constant, -1 to disable.",
        ))
    end
    return nothing
end

function rbf_npoly(dim::Integer, deg::Integer)
    deg < 0 && return 0
    return binomial(dim + deg, dim)
end

function rbf_length_scale(xs::AbstractVector)
    n = length(xs)
    n < 2 && return 1.0
    step = max(1, n ÷ 64)
    ds = Float64[]
    @inbounds for i in 1:step:n
        dmin = Inf
        for j in 1:n
            j == i && continue
            d = norm(xs[i] - xs[j])
            d < dmin && (dmin = d)
        end
        isfinite(dmin) && dmin > 0 && push!(ds, dmin)
    end
    isempty(ds) && return 1.0
    return median(ds)
end

_scale_r(r, h) = r / h

# =============================================================================
# PHS kernels
# =============================================================================

function PHS(n::T = 3; poly_deg::T = 2) where {T <: Int}
    check_poly_deg(poly_deg)
    (1 <= n <= 7) || throw(ArgumentError("PHS order n must be in 1:7 (got $n)"))
    n == 1 && return PHS1(poly_deg)
    n == 2 && return PHS2(poly_deg)
    n == 3 && return PHS3(poly_deg)
    n == 4 && return PHS4(poly_deg)
    n == 5 && return PHS5(poly_deg)
    n == 6 && return PHS6(poly_deg)
    return PHS7(poly_deg)
end

struct PHS1{T <: Int} <: AbstractPHS
    poly_deg::T
    function PHS1(poly_deg::T) where {T <: Int}
        check_poly_deg(poly_deg)
        return new{T}(poly_deg)
    end
end
(phs::PHS1)(r::Number) = float(r)
(phs::PHS1)(x::Point, xᵢ::Point) = phs(norm(x - xᵢ))
function ∂(::PHS1, dim::Int, x::Point, xᵢ::Point)
    r = norm(x - xᵢ)
    return (x[dim] - xᵢ[dim]) / (r + AVOID_INF)
end
int(::PHS1, x::Point2D, xᵢ::Point2D) = (r = norm(x - xᵢ); r^3 / 3)
int(::PHS1, x::Point3D, xᵢ::Point3D) = (r = norm(x - xᵢ); r^4 / 4)

struct PHS2{T <: Int} <: AbstractPHS
    poly_deg::T
    function PHS2(poly_deg::T) where {T <: Int}
        check_poly_deg(poly_deg)
        return new{T}(poly_deg)
    end
end
(phs::PHS2)(r::Number) = (r = float(r); r * r * log(r + AVOID_INF))
(phs::PHS2)(x, xᵢ) = phs(norm(x - xᵢ))
function ∂(::PHS2, dim::Int, x::Point, xᵢ::Point)
    r = norm(x - xᵢ)
    return (x[dim] - xᵢ[dim]) * (2 * log(r + AVOID_INF) + 1)
end
int(::PHS2, x::Point2D, xᵢ::Point2D) = (r = norm(x - xᵢ); (4 * r^4 * log(r + AVOID_INF) - r^4) / 16)
int(::PHS2, x::Point3D, xᵢ::Point3D) = (r = norm(x - xᵢ); (5 * r^5 * log(r + AVOID_INF) - r^5) / 25)

struct PHS3{T <: Int} <: AbstractPHS
    poly_deg::T
    function PHS3(poly_deg::T) where {T <: Int}
        check_poly_deg(poly_deg)
        return new{T}(poly_deg)
    end
end
(phs::PHS3)(r::Number) = float(r)^3
(phs::PHS3)(x, xᵢ) = phs(norm(x - xᵢ))
function ∂(::PHS3, dim::Int, x::Point, xᵢ::Point)
    r = norm(x - xᵢ)
    return 3 * (x[dim] - xᵢ[dim]) * r
end
int(::PHS3, x::Point2D, xᵢ::Point2D) = (r = norm(x - xᵢ); r^5 / 5)
int(::PHS3, x::Point3D, xᵢ::Point3D) = (r = norm(x - xᵢ); r^6 / 6)

struct PHS4{T <: Int} <: AbstractPHS
    poly_deg::T
    function PHS4(poly_deg::T) where {T <: Int}
        check_poly_deg(poly_deg)
        return new{T}(poly_deg)
    end
end
(phs::PHS4)(r::Number) = (r = float(r); r^4 * log(r + AVOID_INF))
(phs::PHS4)(x, xᵢ) = phs(norm(x - xᵢ))
function ∂(::PHS4, dim::Int, x::Point, xᵢ::Point)
    r = norm(x - xᵢ)
    return (x[dim] - xᵢ[dim]) * r^2 * (4 * log(r + AVOID_INF) + 1)
end
int(::PHS4, x::Point2D, xᵢ::Point2D) = (r = norm(x - xᵢ); (r^6 * log(r + AVOID_INF)) / 6 - r^6 / 36)
int(::PHS4, x::Point3D, xᵢ::Point3D) = (r = norm(x - xᵢ); (r^7 * log(r + AVOID_INF)) / 7 - r^7 / 49)

struct PHS5{T <: Int} <: AbstractPHS
    poly_deg::T
    function PHS5(poly_deg::T) where {T <: Int}
        check_poly_deg(poly_deg)
        return new{T}(poly_deg)
    end
end
(phs::PHS5)(r::Number) = float(r)^5
(phs::PHS5)(x, xᵢ) = phs(norm(x - xᵢ))
function ∂(::PHS5, dim::Int, x::Point, xᵢ::Point)
    r = norm(x - xᵢ)
    return 5 * (x[dim] - xᵢ[dim]) * r^3
end
int(::PHS5, x::Point2D, xᵢ::Point2D) = (r = norm(x - xᵢ); r^7 / 7)
int(::PHS5, x::Point3D, xᵢ::Point3D) = (r = norm(x - xᵢ); r^8 / 8)

struct PHS6{T <: Int} <: AbstractPHS
    poly_deg::T
    function PHS6(poly_deg::T) where {T <: Int}
        check_poly_deg(poly_deg)
        return new{T}(poly_deg)
    end
end
(phs::PHS6)(r::Number) = (r = float(r); r^6 * log(r + AVOID_INF))
(phs::PHS6)(x, xᵢ) = phs(norm(x - xᵢ))
function ∂(::PHS6, dim::Int, x::Point, xᵢ::Point)
    r = norm(x - xᵢ)
    return (x[dim] - xᵢ[dim]) * r^4 * (6 * log(r + AVOID_INF) + 1)
end
int(::PHS6, x::Point2D, xᵢ::Point2D) = (r = norm(x - xᵢ); (r^8 * log(r + AVOID_INF)) / 8 - r^8 / 64)
int(::PHS6, x::Point3D, xᵢ::Point3D) = (r = norm(x - xᵢ); (r^9 * log(r + AVOID_INF)) / 9 - r^9 / 81)

struct PHS7{T <: Int} <: AbstractPHS
    poly_deg::T
    function PHS7(poly_deg::T) where {T <: Int}
        check_poly_deg(poly_deg)
        return new{T}(poly_deg)
    end
end
(phs::PHS7)(r::Number) = float(r)^7
(phs::PHS7)(x, xᵢ) = phs(norm(x - xᵢ))
function ∂(::PHS7, dim::Int, x::Point, xᵢ::Point)
    r = norm(x - xᵢ)
    return 7 * (x[dim] - xᵢ[dim]) * r^5
end
int(::PHS7, x::Point2D, xᵢ::Point2D) = (r = norm(x - xᵢ); r^9 / 9)
int(::PHS7, x::Point3D, xᵢ::Point3D) = (r = norm(x - xᵢ); r^10 / 10)

for phs in (:PHS1, :PHS2, :PHS3, :PHS4, :PHS5, :PHS6, :PHS7)
    @eval $phs(; poly_deg::Int = 2) = $phs(poly_deg)
end

function Base.show(io::IO, rbf::R) where {R <: AbstractPHS}
    print(io, print_basis(rbf))
    print(io, "\n└─Polynomial augmentation: degree $(rbf.poly_deg)")
    return nothing
end
print_basis(::PHS1) = "Polyharmonic spline (r)"
print_basis(::PHS2) = "Polyharmonic spline (r² log r)"
print_basis(::PHS3) = "Polyharmonic spline (r³)"
print_basis(::PHS4) = "Polyharmonic spline (r⁴ log r)"
print_basis(::PHS5) = "Polyharmonic spline (r⁵)"
print_basis(::PHS6) = "Polyharmonic spline (r⁶ log r)"
print_basis(::PHS7) = "Polyharmonic spline (r⁷)"

# =============================================================================
# IMQ / Gaussian
# =============================================================================

struct IMQ{T <: Real} <: AbstractRadialBasis
    ε::T
    poly_deg::Int
    function IMQ(ε::T = 1.0; poly_deg::Int = 1) where {T <: Real}
        check_poly_deg(poly_deg)
        ε > 0 || throw(ArgumentError("IMQ ε must be > 0"))
        return new{T}(ε, poly_deg)
    end
end
(b::IMQ)(r::Number) = 1 / sqrt(1 + (b.ε * float(r))^2)
(b::IMQ)(x::Point, xᵢ::Point) = b(norm(x - xᵢ))
function ∂(b::IMQ, dim::Int, x::Point, xᵢ::Point)
    r2 = norm(x - xᵢ)^2
    s = 1 + b.ε^2 * r2
    return -(b.ε^2) * (x[dim] - xᵢ[dim]) * s^(-1.5)
end

struct Gaussian{T <: Real} <: AbstractRadialBasis
    ε::T
    poly_deg::Int
    function Gaussian(ε::T = 1.0; poly_deg::Int = 1) where {T <: Real}
        check_poly_deg(poly_deg)
        ε > 0 || throw(ArgumentError("Gaussian ε must be > 0"))
        return new{T}(ε, poly_deg)
    end
end
(b::Gaussian)(r::Number) = exp(-(b.ε * float(r))^2)
(b::Gaussian)(x::Point, xᵢ::Point) = b(norm(x - xᵢ))
function ∂(b::Gaussian, dim::Int, x::Point, xᵢ::Point)
    r2 = norm(x - xᵢ)^2
    return -2 * b.ε^2 * (x[dim] - xᵢ[dim]) * exp(-(b.ε^2) * r2)
end

poly_deg(b::AbstractPHS) = b.poly_deg
poly_deg(b::IMQ) = b.poly_deg
poly_deg(b::Gaussian) = b.poly_deg

# =============================================================================
# Fundamental solution as DIBEM RBF (same u* as single-layer G)
# =============================================================================
#
# φ(R) = u*(R) with F_ii set by [`_zero_rowsum_diag!`](@ref):
#   F_ii = 1 − ∑_{j≠i} F_ij  ⇒  F 1 = 1  (constants in the range of F).
#
# Radial particular for IF: ∫_0^R φ(ρ) ρ^{d-1} dρ  (same as DIBEM ID for FS).
# =============================================================================

"""
    FundamentalRBF(; k=1.0, dim=2, poly_deg=-1)

Use the Laplace fundamental solution as the DIBEM radial basis:

- 2D: ``φ = −\\log(R)/(2πk)``
- 3D: ``φ = 1/(4πk R)``

Callable as `φ(r)` (Euclidean distance) or `φ(x, xᵢ)`. Default `poly_deg=-1`
(no polynomial tail). Pair with [`_zero_rowsum_diag!`](@ref) on the Gram matrix
so each row of `F` sums to one.
"""
struct FundamentalRBF{T<:Real} <: AbstractRadialBasis
    k::T
    dim::Int
    poly_deg::Int
end

function FundamentalRBF(; k::Real=1.0, dim::Int=2, poly_deg::Int=-1)
    dim in (2, 3) || throw(ArgumentError("FundamentalRBF dim must be 2 or 3"))
    check_poly_deg(poly_deg)
    return FundamentalRBF(float(k), dim, poly_deg)
end

"""Convenience: pull `k` from a Laplace problem / BEMdata."""
fundamental_rbf(props::Laplace; dim::Int=2, poly_deg::Int=-1) =
    FundamentalRBF(; k=float(props.k), dim=dim, poly_deg=poly_deg)
fundamental_rbf(dad::BEMdata{<:Laplace}; poly_deg::Int=-1) =
    fundamental_rbf(dad.properties; dim=dad.dimension, poly_deg=poly_deg)

function (b::FundamentalRBF)(r::Number)
    R = max(float(r), 0.0)
    R < 1e-15 && return 0.0          # diagonal filled later by zero-row-sum
    if b.dim == 2
        return -log(R) / (2π * b.k)
    else
        return 1 / (4π * b.k * R)
    end
end
(b::FundamentalRBF)(x::Point, xᵢ::Point) = b(norm(x - xᵢ))

poly_deg(b::FundamentalRBF) = b.poly_deg

# ∫_0^R φ(ρ) ρ dρ  (2D) / ∫_0^R φ(ρ) ρ² dρ (3D) — matches Domain.jl radial_integral(Laplace)
function int(b::FundamentalRBF, x::Point2D, xᵢ::Point2D)
    R = norm(x - xᵢ)
    R < 1e-30 && return 0.0
    # −(2 R² log R − R²) / (8π k)
    return -(2 * R^2 * log(R) - R^2) / (8π * b.k)
end
function int(b::FundamentalRBF, x::Point3D, xᵢ::Point3D)
    R = norm(x - xᵢ)
    R < 1e-30 && return 0.0
    return R^2 / (8π * b.k)   # 1/(4πk) * R²/2
end

# =============================================================================
# Monomial basis
# =============================================================================

struct MonomialBasis{Dim, Deg, F <: Function} <: AbstractBasis
    f::F
end

function MonomialBasis(dim::Int, deg::Int)
    deg < 0 && throw(ArgumentError("Monomial basis degree must be ≥ 0 (got $deg)"))
    f = _get_monomial_basis(Val(dim), Val(deg))
    return MonomialBasis{dim, deg, typeof(f)}(f)
end

function (m::MonomialBasis{Dim, Deg})(x) where {Dim, Deg}
    b = ones(typeof(float(x[1])), binomial(Dim + Deg, Dim))
    m.f(b, x)
    return b
end

degree(::MonomialBasis{D, Deg}) where {D, Deg} = Deg
dim(::MonomialBasis{D, Deg}) where {D, Deg} = D

# monomial fill functions
function _get_monomial_basis(::Val{1}, ::Val{0})
    return (b, x) -> (b[1] = 1; b)
end
function _get_monomial_basis(::Val{1}, ::Val{1})
    return (b, x) -> (b[1] = 1; b[2] = x[1]; b)
end
function _get_monomial_basis(::Val{1}, ::Val{2})
    return (b, x) -> (b[1] = 1; b[2] = x[1]; b[3] = x[1]^2; b)
end
function _get_monomial_basis(::Val{2}, ::Val{0})
    return (b, x) -> (b[1] = 1; b)
end
function _get_monomial_basis(::Val{2}, ::Val{1})
    return (b, x) -> (b[1] = 1; b[2] = x[1]; b[3] = x[2]; b)
end
function _get_monomial_basis(::Val{2}, ::Val{2})
    return (b, x) -> begin
        b[1] = 1
        b[2] = x[1]; b[3] = x[2]
        b[4] = x[1] * x[2]; b[5] = x[1]^2; b[6] = x[2]^2
        b
    end
end
function _get_monomial_basis(::Val{3}, ::Val{0})
    return (b, x) -> (b[1] = 1; b)
end
function _get_monomial_basis(::Val{3}, ::Val{1})
    return (b, x) -> (b[1] = 1; b[2] = x[1]; b[3] = x[2]; b[4] = x[3]; b)
end
function _get_monomial_basis(::Val{3}, ::Val{2})
    return (b, x) -> begin
        b[1] = 1
        b[2] = x[1]; b[3] = x[2]; b[4] = x[3]
        b[5] = x[1]*x[2]; b[6] = x[1]*x[3]; b[7] = x[2]*x[3]
        b[8] = x[1]^2; b[9] = x[2]^2; b[10] = x[3]^2
        b
    end
end
function _get_monomial_basis(::Val{D}, ::Val{Deg}) where {D, Deg}
    # generic fallback: only constant
    Deg > 2 && @warn "MonomialBasis dim=$D deg=$Deg using constant only fallback"
    return (b, x) -> (b[1] = 1; b)
end

function ∂(m::MonomialBasis{2, 0}, dim::Int, x)
    return [0.0]
end
function ∂(m::MonomialBasis{2, 1}, dim::Int, x)
    d = zeros(3)
    dim == 1 && (d[2] = 1)
    dim == 2 && (d[3] = 1)
    return d
end
function ∂(m::MonomialBasis{2, 2}, dim::Int, x)
    d = zeros(6)
    if dim == 1
        d[2] = 1; d[4] = x[2]; d[5] = 2x[1]
    else
        d[3] = 1; d[4] = x[1]; d[6] = 2x[2]
    end
    return d
end
function ∂(::MonomialBasis{3, 0}, dim::Int, x)
    return [0.0]
end
function ∂(::MonomialBasis{3, 1}, dim::Int, x)
    d = zeros(4)
    dim == 1 && (d[2] = 1)
    dim == 2 && (d[3] = 1)
    dim == 3 && (d[4] = 1)
    return d
end
function ∂(::MonomialBasis{3, 2}, dim::Int, x)
    d = zeros(10)
    if dim == 1
        d[2] = 1; d[5] = x[2]; d[6] = x[3]; d[8] = 2x[1]
    elseif dim == 2
        d[3] = 1; d[5] = x[1]; d[7] = x[3]; d[9] = 2x[2]
    else
        d[4] = 1; d[6] = x[1]; d[7] = x[2]; d[10] = 2x[3]
    end
    return d
end
function ∂(::MonomialBasis{3, Deg}, dim::Int, x) where {Deg}
    n = binomial(3 + Deg, 3)
    return zeros(n)
end
function ∂(m::MonomialBasis{1, Deg}, dim::Int, x) where {Deg}
    n = Deg + 1
    d = zeros(n)
    if Deg >= 1 && dim == 1
        d[2] = 1
    end
    if Deg >= 2 && dim == 1
        d[3] = 2x[1]
    end
    return d
end

# radial integrals of monomials (DIBEM geo props) — simplified
function int(::MonomialBasis{2, 0}, pf::Point, x::Point)
    R = norm(pf - x)
    return [R^2 / 2]
end
function int(::MonomialBasis{2, 1}, pf::Point, x::Point)
    R = norm(pf - x)
    r = x - pf
    return [R^2 / 2; R^2 / 3 * r + R^2 / 2 * SVector(pf[1], pf[2])]
end
function int(m::MonomialBasis{2, 2}, pf::Point, x::Point)
    R = norm(pf - x)
    r = x - pf
    return [
        R^2 / 2
        R^2 / 3 * r[1] + R^2 / 2 * pf[1]
        R^2 / 3 * r[2] + R^2 / 2 * pf[2]
        R^2 / 4 * r[1] * r[2] + R^2 / 3 * (r[1] * pf[2] + r[2] * pf[1]) + R^2 / 2 * pf[1] * pf[2]
        R^2 / 4 * r[1]^2 + 2 * R^2 / 3 * r[1] * pf[1] + R^2 / 2 * pf[1]^2
        R^2 / 4 * r[2]^2 + 2 * R^2 / 3 * r[2] * pf[2] + R^2 / 2 * pf[2]^2
    ]
end
function int(::MonomialBasis{3, 0}, pf::Point, x::Point)
    R = norm(pf - x)
    return [R^3 / 3]
end
function int(::MonomialBasis{3, 1}, pf::Point, x::Point)
    R = norm(pf - x)
    e = (x - pf) / (R + eps(R))
    R3 = R^3 / 3
    R4 = R^4 / 4
    return [R3
            pf[1] * R3 + e[1] * R4
            pf[2] * R3 + e[2] * R4
            pf[3] * R3 + e[3] * R4]
end
function int(::MonomialBasis{3, 2}, pf::Point, x::Point)
    R = norm(pf - x)
    e = (x - pf) / (R + eps(R))
    R3 = R^3 / 3
    R4 = R^4 / 4
    R5 = R^5 / 5
    x0, y0, z0 = pf[1], pf[2], pf[3]
    ex, ey, ez = e[1], e[2], e[3]
    return [
        R3
        x0 * R3 + ex * R4
        y0 * R3 + ey * R4
        z0 * R3 + ez * R4
        x0 * y0 * R3 + (x0 * ey + y0 * ex) * R4 + ex * ey * R5
        x0 * z0 * R3 + (x0 * ez + z0 * ex) * R4 + ex * ez * R5
        y0 * z0 * R3 + (y0 * ez + z0 * ey) * R4 + ey * ez * R5
        x0^2 * R3 + 2 * x0 * ex * R4 + ex^2 * R5
        y0^2 * R3 + 2 * y0 * ey * R4 + ey^2 * R5
        z0^2 * R3 + 2 * z0 * ez * R4 + ez^2 * R5
    ]
end
function int(m::MonomialBasis{3, Deg}, pf::Point, x::Point) where {Deg}
    R = norm(pf - x)
    n = binomial(3 + Deg, 3)
    v = zeros(n)
    v[1] = R^3 / 3
    return v
end

# =============================================================================
# Cardinal weights
# =============================================================================

function rbf_cardinal(
        xstar::Point,
        xs::AbstractVector{<:Point};
        basis::AbstractRadialBasis = PHS(3; poly_deg = 1),
        ridge::Real = 1e-12,
        h::Union{Nothing, Real} = nothing,
    )
    n = length(xs)
    n == 0 && return Float64[]
    @inbounds for j in 1:n
        if norm(xstar - xs[j])^2 < 1e-28
            w = zeros(n)
            w[j] = 1.0
            return w
        end
    end
    dim = length(xs[1])
    deg = poly_deg(basis)
    while deg >= 0 && rbf_npoly(dim, deg) > n
        deg -= 1
    end
    npoly = rbf_npoly(dim, deg)
    hh = h === nothing ? rbf_length_scale(xs) : float(h)
    hh = max(hh, 1e-14)
    A = zeros(n, n)
    @inbounds for j in 1:n, i in 1:j
        aij = basis(_scale_r(norm(xs[i] - xs[j]), hh))
        A[i, j] = aij
        A[j, i] = aij
    end
    ε = float(ridge) * (tr(A) / n + 1)
    @inbounds for i in 1:n
        A[i, i] += ε
    end
    if npoly == 0
        ψ = [basis(_scale_r(norm(xstar - xs[i]), hh)) for i in 1:n]
        return A \ ψ
    end
    mon = MonomialBasis(dim, deg)
    P = zeros(n, npoly)
    @inbounds for i in 1:n
        P[i, :] = mon(xs[i])
    end
    K = [A P; P' zeros(npoly, npoly)]
    ψ = [basis(_scale_r(norm(xstar - xs[i]), hh)) for i in 1:n]
    rhs = vcat(ψ, mon(xstar))
    coef = try
        K \ rhs
    catch
        w = zeros(n)
        s = 0.0
        @inbounds for i in 1:n
            wi = 1 / (norm(xstar - xs[i]) + 1e-14)
            w[i] = wi
            s += wi
        end
        return w ./ s
    end
    return coef[1:n]
end

function rbf_cardinal(
        xstar::Point,
        xs_global::AbstractVector{<:Point},
        ids::AbstractVector{<:Integer};
        kwargs...,
    )
    n = length(xs_global)
    row = zeros(n)
    isempty(ids) && return row
    pts = [xs_global[j] for j in ids]
    wloc = rbf_cardinal(xstar, pts; kwargs...)
    @inbounds for (k, j) in enumerate(ids)
        row[j] = wloc[k]
    end
    return row
end

# =============================================================================
# Global RBF interpolant
# =============================================================================

struct RBF{B <: AbstractRadialBasis, M, L}
    x::Vector{Point}
    rbf_basis::B
    monomial_basis::M
    npoly::Int
    h::Float64
    F::Matrix{Float64}
    P::Matrix{Float64}
    fat::L
    ridge::Float64
end

function Base.show(io::IO, rbf::RBF)
    println(io, "RBF interpolant")
    println(io, "  centres: $(length(rbf.x)), h=$(rbf.h), ridge=$(rbf.ridge), npoly=$(rbf.npoly)")
    print(io, "  "); show(io, rbf.rbf_basis)
    return nothing
end

function RBF(
        x::AbstractVector{<:Point},
        basis::B = PHS();
        ridge::Real = 1e-12,
        h::Union{Nothing, Real} = nothing,
    ) where {B <: AbstractRadialBasis}
    x = collect(Point, x)
    k = length(x)
    k == 0 && error("RBF: empty centre set")
    dim = length(x[1])
    deg = poly_deg(basis)
    while deg >= 0 && rbf_npoly(dim, deg) > k
        deg -= 1
    end
    npoly = rbf_npoly(dim, deg)
    hh = h === nothing ? rbf_length_scale(x) : float(h)
    hh = max(hh, 1e-14)
    F = zeros(k, k)
    @inbounds for j in 1:k, i in 1:j
        fij = basis(_scale_r(norm(x[i] - x[j]), hh))
        F[i, j] = fij
        F[j, i] = fij
    end
    ε = float(ridge) * (tr(F) / max(k, 1) + 1)
    @inbounds for i in 1:k
        F[i, i] += ε
    end
    mon = npoly > 0 ? MonomialBasis(dim, deg) : nothing
    P = zeros(npoly, k)
    if npoly > 0
        @inbounds for i in 1:k
            P[:, i] = mon(x[i])
        end
    end
    if npoly == 0
        fat = lu(F)
        return RBF{B, Nothing, typeof(fat)}(x, basis, nothing, 0, hh, F, P, fat, float(ridge))
    end
    Z = zeros(npoly, npoly)
    aux = Matrix{Float64}([F P'; P Z])
    fat = try
        lu(aux)
    catch
        @inbounds for i in 1:k
            aux[i, i] += ε * 1e3
        end
        lu(aux)
    end
    return RBF{B, typeof(mon), typeof(fat)}(x, basis, mon, npoly, hh, F, P, fat, float(ridge))
end

function rbf_weights(rbf::RBF, x::Point)
    k = length(rbf.x)
    ψ = zeros(k)
    @inbounds for i in 1:k
        ψ[i] = rbf.rbf_basis(_scale_r(norm(x - rbf.x[i]), rbf.h))
    end
    if rbf.npoly == 0
        return rbf.fat \ ψ
    end
    p = rbf.monomial_basis(x)
    return (rbf.fat \ vcat(ψ, p))[1:k]
end

function rbf_partial_weights(rbf::RBF, dim::Int, x::Point)
    k = length(rbf.x)
    h = rbf.h
    xh = x / h
    ψ = zeros(k)
    @inbounds for i in 1:k
        ψ[i] = ∂(rbf.rbf_basis, dim, xh, rbf.x[i] / h) / h
    end
    if rbf.npoly == 0
        return rbf.fat \ ψ
    end
    p = ∂(rbf.monomial_basis, dim, x)
    return (rbf.fat \ vcat(ψ, p))[1:k]
end

function rbf_evaluate(rbf::RBF, xt::AbstractVector{<:Point}, y::AbstractVector{<:Real})
    k = length(rbf.x)
    length(y) == k || throw(DimensionMismatch("y length $(length(y)) ≠ $(k) centres"))
    rhs = rbf.npoly == 0 ? collect(Float64, y) : vcat(Float64.(y), zeros(rbf.npoly))
    coef = rbf.fat \ rhs
    nt = length(xt)
    out = zeros(nt)
    @inbounds for j in 1:nt
        s = 0.0
        for i in 1:k
            s += coef[i] * rbf.rbf_basis(_scale_r(norm(xt[j] - rbf.x[i]), rbf.h))
        end
        if rbf.npoly > 0
            p = rbf.monomial_basis(xt[j])
            for α in 1:rbf.npoly
                s += coef[k + α] * p[α]
            end
        end
        out[j] = s
    end
    return out
end
rbf_evaluate(rbf::RBF, x::Point, y::AbstractVector{<:Real}) = only(rbf_evaluate(rbf, [x], y))

function rbf_partial(rbf::RBF, dim::Int, xt::AbstractVector{<:Point}, y::AbstractVector{<:Real})
    k = length(rbf.x)
    h = rbf.h
    rhs = rbf.npoly == 0 ? collect(Float64, y) : vcat(Float64.(y), zeros(rbf.npoly))
    coef = rbf.fat \ rhs
    nt = length(xt)
    out = zeros(nt)
    @inbounds for j in 1:nt
        xh = xt[j] / h
        s = 0.0
        for i in 1:k
            s += coef[i] * ∂(rbf.rbf_basis, dim, xh, rbf.x[i] / h) / h
        end
        if rbf.npoly > 0
            p = ∂(rbf.monomial_basis, dim, xt[j])
            for α in 1:rbf.npoly
                s += coef[k + α] * p[α]
            end
        end
        out[j] = s
    end
    return out
end

(rbf::RBF)(x::Point) = rbf_weights(rbf, x)
function (rbf::RBF)(x::Vector{<:Point})
    Y = zeros(length(x), length(rbf.x))
    for (j, xi) in enumerate(x)
        Y[j, :] .= rbf_weights(rbf, xi)
    end
    return Y
end
(rbf::RBF)(x::Vector{<:Point}, y::Vector{Float64}) = rbf_evaluate(rbf, x, y)
∂(rbf::RBF, dim::Int, x::Point) = rbf_partial_weights(rbf, dim, x)
function ∂(rbf::RBF, dim::Int, x::Vector{<:Point})
    Y = zeros(length(x), length(rbf.x))
    for (j, xi) in enumerate(x)
        Y[j, :] .= rbf_partial_weights(rbf, dim, xi)
    end
    return Y
end
∂(rbf::RBF, dim::Int, x::Vector{<:Point}, y::Vector{Float64}) = rbf_partial(rbf, dim, x, y)

# Extensions: Wendland, MQ, radial integrals, local/PU/rational, Kansa, compare
include(joinpath(@__DIR__, "RBF_Extensions.jl"))
