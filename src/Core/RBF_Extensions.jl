# =============================================================================
# RBF extensions: compact/MQ kernels, radial integrals, local/PU/rational,
# Kansa Poisson, method comparison  (PRBFT / RadialBasisFunctions.jl / FastRVSK)
# =============================================================================

export MQ, WendlandC2, WendlandC4, WendlandC6
export radial_integral, laplacian_phi, laplace_particular
export LocalRBF, local_rbf_fit, local_rbf_eval
export PURBF, pu_rbf_fit, pu_rbf_eval
export RationalRBF, rational_rbf_fit, rational_rbf_eval
export kansa_poisson, kansa_eval, compare_rbf_methods
export rbf_neighbors, rbf_laplacian_weights, _rbf_kdtree

# -----------------------------------------------------------------------------
# MQ + Wendland
# -----------------------------------------------------------------------------

struct MQ{T <: Real} <: AbstractRadialBasis
    ε::T
    C::T
    poly_deg::Int
    paper::Bool
    function MQ{T}(ε::T, C::T, poly_deg::Int, paper::Bool) where {T <: Real}
        check_poly_deg(poly_deg)
        return new{T}(ε, C, poly_deg, paper)
    end
end

"""
    MQ(ε=1.0; poly_deg=1)
    MQ(; C=0.01, poly_deg=1)

Multiquadric.

- `MQ(ε)`: ``φ = √(1+(ε r)²)``
- `MQ(; C)`: ``φ = √(r²+C²)`` (Samaan & Rashed 2007; typical `C=0.01`)
"""
function MQ(ε::T = 1.0; poly_deg::Int = 1,
        C::Union{Nothing,Real} = nothing) where {T <: Real}
    if C === nothing
        ε > 0 || throw(ArgumentError("MQ ε must be > 0"))
        return MQ{T}(ε, zero(ε), poly_deg, false)
    else
        Cc = float(C)
        Cc > 0 || throw(ArgumentError("MQ C must be > 0"))
        TT = typeof(Cc)
        return MQ{TT}(one(TT), Cc, poly_deg, true)
    end
end
(b::MQ)(r::Number) = b.paper ? sqrt(float(r)^2 + b.C^2) : sqrt(1 + (b.ε * float(r))^2)
(b::MQ)(x::Point, xᵢ::Point) = b(norm(x - xᵢ))
function ∂(b::MQ, dim::Int, x::Point, xᵢ::Point)
    dx = x[dim] - xᵢ[dim]
    r2 = norm(x - xᵢ)^2
    if b.paper
        return dx / sqrt(r2 + b.C^2)
    else
        return (b.ε^2) * dx / sqrt(1 + b.ε^2 * r2)
    end
end
poly_deg(b::MQ) = b.poly_deg
print_basis(b::MQ) = b.paper ? "Multiquadric √(r²+C²), C=$(b.C)" :
    "Multiquadric √(1+(ε r)²), ε=$(b.ε)"

abstract type AbstractWendland <: AbstractRadialBasis end

struct WendlandC2{T <: Real} <: AbstractWendland
    δ::T
    poly_deg::Int
    function WendlandC2(δ::T = 1.0; poly_deg::Int = 0) where {T <: Real}
        check_poly_deg(poly_deg)
        δ > 0 || throw(ArgumentError("Wendland δ must be > 0"))
        return new{T}(δ, poly_deg)
    end
end
function (b::WendlandC2)(r::Number)
    ρ = float(r) / b.δ
    ρ >= 1 && return zero(float(r))
    s = 1 - ρ
    return s^4 * (4ρ + 1)
end
(b::WendlandC2)(x::Point, xᵢ::Point) = b(norm(x - xᵢ))
function ∂(b::WendlandC2, dim::Int, x::Point, xᵢ::Point)
    dx = x[dim] - xᵢ[dim]
    rr = norm(x - xᵢ)
    r = rr / b.δ
    r >= 1 && return 0.0
    dφdr = -20 * r * (1 - r)^3
    return dφdr / b.δ * dx / (rr + AVOID_INF)
end
poly_deg(b::WendlandC2) = b.poly_deg

struct WendlandC4{T <: Real} <: AbstractWendland
    δ::T
    poly_deg::Int
    function WendlandC4(δ::T = 1.0; poly_deg::Int = 0) where {T <: Real}
        check_poly_deg(poly_deg)
        δ > 0 || throw(ArgumentError("Wendland δ must be > 0"))
        return new{T}(δ, poly_deg)
    end
end
function (b::WendlandC4)(r::Number)
    ρ = float(r) / b.δ
    ρ >= 1 && return zero(float(r))
    return (1 - ρ)^6 * (3 + 18ρ + 35 * ρ^2)
end
(b::WendlandC4)(x::Point, xᵢ::Point) = b(norm(x - xᵢ))
function ∂(b::WendlandC4, dim::Int, x::Point, xᵢ::Point)
    dx = x[dim] - xᵢ[dim]
    rr = norm(x - xᵢ)
    r = rr / b.δ
    r >= 1 - 1e-15 && return 0.0
    dφdr = -56 * r * (1 - r)^5 * (1 + 5r)
    return dφdr / b.δ * dx / (rr + AVOID_INF)
end
poly_deg(b::WendlandC4) = b.poly_deg

struct WendlandC6{T <: Real} <: AbstractWendland
    δ::T
    poly_deg::Int
    function WendlandC6(δ::T = 1.0; poly_deg::Int = 0) where {T <: Real}
        check_poly_deg(poly_deg)
        δ > 0 || throw(ArgumentError("Wendland δ must be > 0"))
        return new{T}(δ, poly_deg)
    end
end
function (b::WendlandC6)(r::Number)
    ρ = float(r) / b.δ
    ρ >= 1 && return zero(float(r))
    return (1 - ρ)^8 * (1 + 8ρ + 25ρ^2 + 32ρ^3)
end
(b::WendlandC6)(x::Point, xᵢ::Point) = b(norm(x - xᵢ))
function ∂(b::WendlandC6, dim::Int, x::Point, xᵢ::Point)
    dx = x[dim] - xᵢ[dim]
    rr = norm(x - xᵢ)
    r = rr / b.δ
    r >= 1 - 1e-15 && return 0.0
    ε = 1e-8
    φp = b((r + ε) * b.δ)
    φm = b(max(r - ε, 0.0) * b.δ)
    dφdr = (φp - φm) / (2ε)
    return dφdr / b.δ * dx / (rr + AVOID_INF)
end
poly_deg(b::WendlandC6) = b.poly_deg

# -----------------------------------------------------------------------------
# Radial integration ∫_0^R φ(ρ) ρ^{d-1} dρ  — closed forms only (CAS)
# -----------------------------------------------------------------------------
# Antiderivatives from computer algebra (Maxima/SymPy):
#   integrate(phi(rho)*rho^(d-1), rho, 0, R)
# Verified against high-order Gauss in tests (`radial_integral_gauss`).

"""
    radial_integral(basis, R; dim=2)

- 2D: ``∫_0^R φ(ρ)\\,ρ\\,dρ``
- 3D: ``∫_0^R φ(ρ)\\,ρ²\\,dρ``

Fully **analytical** for PHS, IMQ, MQ, Gaussian, Wendland C²/C⁴/C⁶.
No numerical quadrature in production code.
"""
function radial_integral(basis::AbstractRadialBasis, R::Real; dim::Int = 2)
    R = float(R)
    R <= 0 && return 0.0
    dim in (2, 3) || throw(ArgumentError("dim must be 2 or 3"))
    return _radial_integral_impl(basis, R, dim)
end

# ---- PHS --------------------------------------------------------------------
_radial_integral_impl(::PHS1, R::Real, dim::Int) = dim == 2 ? float(R)^3 / 3 : float(R)^4 / 4
_radial_integral_impl(::PHS2, R::Real, dim::Int) =
    (R = float(R); dim == 2 ? (4 * R^4 * log(R + AVOID_INF) - R^4) / 16 :
    (5 * R^5 * log(R + AVOID_INF) - R^5) / 25)
_radial_integral_impl(::PHS3, R::Real, dim::Int) = dim == 2 ? float(R)^5 / 5 : float(R)^6 / 6
_radial_integral_impl(::PHS4, R::Real, dim::Int) =
    (R = float(R); dim == 2 ? R^6 * log(R + AVOID_INF) / 6 - R^6 / 36 :
    R^7 * log(R + AVOID_INF) / 7 - R^7 / 49)
_radial_integral_impl(::PHS5, R::Real, dim::Int) = dim == 2 ? float(R)^7 / 7 : float(R)^8 / 8
_radial_integral_impl(::PHS6, R::Real, dim::Int) =
    (R = float(R); dim == 2 ? R^8 * log(R + AVOID_INF) / 8 - R^8 / 64 :
    R^9 * log(R + AVOID_INF) / 9 - R^9 / 81)
_radial_integral_impl(::PHS7, R::Real, dim::Int) = dim == 2 ? float(R)^9 / 9 : float(R)^10 / 10

# ---- IMQ  φ = 1/√(1+(εr)²) -------------------------------------------------
function _radial_integral_impl(b::IMQ, R::Real, dim::Int)
    R = float(R)
    ε = float(b.ε)
    ε2 = ε * ε
    s = sqrt(1 + ε2 * R * R)
    dim == 2 && return (s - 1) / ε2
    return (ε * R * s - asinh(ε * R)) / (2 * ε^3)
end

# ---- MQ ---------------------------------------------------------------------
# Paper form φ = √(r²+C²): 2D uses ∫ ρ √(ρ²+C²) dρ = (1/3)(ρ²+C²)^{3/2}
# (user / RIM). Scaled Hardy φ = √(1+(ε r)²) is the same with C = 1/ε.
function _radial_integral_impl(b::MQ, R::Real, dim::Int)
    R = float(R)
    if b.paper
        C = float(b.C)
        C2 = C * C
        s = sqrt(R * R + C2)
        dim == 2 && return (s^3 - C^3) / 3
        # ∫_0^R ρ² √(ρ²+C²) dρ
        return (R * (2 * R * R + C2) * s - C2 * C2 * log((R + s) / C)) / 8
    end
    ε = float(b.ε)
    ε2 = ε * ε
    s = sqrt(1 + ε2 * R * R)
    dim == 2 && return (ε2 * R * R * s + s - 1) / (3 * ε2)
    return (2 * ε^3 * R^3 * s + ε * R * s - asinh(ε * R)) / (8 * ε^3)
end

# ---- Gaussian  φ = exp(-(εr)²) ----------------------------------------------
function _radial_integral_impl(b::Gaussian, R::Real, dim::Int)
    R = float(R)
    ε = float(b.ε)
    ε2 = ε * ε
    e = exp(-ε2 * R * R)
    dim == 2 && return (1 - e) / (2 * ε2)
    return -R * e / (2 * ε2) + sqrt(π) * erf(ε * R) / (4 * ε^3)
end

# ---- Wendland C2 ------------------------------------------------------------
function _radial_integral_impl(b::WendlandC2, R::Real, dim::Int)
    R = float(R)
    s = float(b.δ)
    R >= s && return dim == 2 ? s^2 / 14 : s^3 / 42
    if dim == 2
        return R^2 * (8R^5 - 35R^4 * s + 56R^3 * s^2 - 35R^2 * s^3 + 7s^5) / (14 * s^5)
    else
        return R^3 * (21R^5 - 90R^4 * s + 140R^3 * s^2 - 84R^2 * s^3 + 14s^5) / (42 * s^5)
    end
end

# ---- Wendland C4 ------------------------------------------------------------
function _radial_integral_impl(b::WendlandC4, R::Real, dim::Int)
    R = float(R)
    s = float(b.δ)
    R >= s && return dim == 2 ? s^2 / 6 : 8 * s^3 / 165
    if dim == 2
        return R^2 * (21R^8 - 128R^7 * s + 315R^6 * s^2 - 384R^5 * s^3 +
                      210R^4 * s^4 - 42R^2 * s^6 + 9s^8) / (6 * s^8)
    else
        return R^3 * (525R^8 - 3168R^7 * s + 7700R^6 * s^2 - 9240R^5 * s^3 +
                      4950R^4 * s^4 - 924R^2 * s^6 + 165s^8) / (165 * s^8)
    end
end

# ---- Wendland C6 ------------------------------------------------------------
function _radial_integral_impl(b::WendlandC6, R::Real, dim::Int)
    R = float(R)
    s = float(b.δ)
    R >= s && return dim == 2 ? 7 * s^2 / 156 : 16 * s^3 / 1365
    if dim == 2
        return R^2 * (384R^11 - 3003R^10 * s + 9984R^9 * s^2 - 18018R^8 * s^3 +
                      18304R^7 * s^4 - 9009R^6 * s^5 + 1716R^4 * s^7 -
                      429R^2 * s^9 + 78s^11) / (156 * s^11)
    else
        return R^3 * (3120R^11 - 24255R^10 * s + 80080R^9 * s^2 - 143325R^8 * s^3 +
                      144144R^7 * s^4 - 70070R^6 * s^5 + 12870R^4 * s^7 -
                      3003R^2 * s^9 + 455s^11) / (1365 * s^11)
    end
end

function _radial_integral_impl(basis::AbstractRadialBasis, R::Real, dim::Int)
    throw(ArgumentError(
        "no closed-form radial_integral for $(typeof(basis)); add a CAS-derived method",
    ))
end

"""
    radial_integral_gauss(basis, R; dim=2, n=96)

High-order Gauss–Legendre **reference** for verifying closed forms (tests only).
Not used by production assembly paths.
"""
function radial_integral_gauss(basis::AbstractRadialBasis, R::Real; dim::Int = 2, n::Int = 96)
    R = float(R)
    R <= 0 && return 0.0
    # Compact kernels vanish for ρ > support — integrate only where φ ≠ 0
    Rmax = R
    if basis isa AbstractWendland
        Rmax = min(R, float(basis.δ))
    end
    ξ, w = gausslegendre(n)
    s = 0.0
    @inbounds for i in 1:n
        ρ = (ξ[i] + 1) / 2 * Rmax
        s += basis(ρ) * ρ^(dim - 1) * w[i] * (Rmax / 2)
    end
    return s
end
export radial_integral_gauss
for B in (:IMQ, :Gaussian, :MQ, :WendlandC2, :WendlandC4, :WendlandC6)
    @eval begin
        int(b::$B, x::Point2D, xᵢ::Point2D) = radial_integral(b, norm(x - xᵢ); dim = 2)
        int(b::$B, x::Point3D, xᵢ::Point3D) = radial_integral(b, norm(x - xᵢ); dim = 3)
    end
end

# -----------------------------------------------------------------------------
# Laplacian φ and Laplace particular Ψ (∇²Ψ = φ)
# -----------------------------------------------------------------------------

function laplacian_phi(basis::AbstractRadialBasis, r::Real; dim::Int = 2)
    return _lap_phi(basis, float(r), dim)
end

_lap_phi(::PHS1, r, dim) = dim == 2 ? 1 / (r + AVOID_INF) : 2 / (r + AVOID_INF)
_lap_phi(::PHS2, r, dim) =
    dim == 2 ? 4 * log(r + AVOID_INF) + 4 : 6 * log(r + AVOID_INF) + 5
_lap_phi(::PHS3, r, dim) = dim == 2 ? 9r : 12r
_lap_phi(::PHS4, r, dim) =
    dim == 2 ? r^2 * (16 * log(r + AVOID_INF) + 12) : r^2 * (20 * log(r + AVOID_INF) + 16)
_lap_phi(::PHS5, r, dim) = dim == 2 ? 25 * r^3 : 30 * r^3
_lap_phi(::PHS6, r, dim) =
    dim == 2 ? r^4 * (36 * log(r + AVOID_INF) + 24) : r^4 * (42 * log(r + AVOID_INF) + 30)
_lap_phi(::PHS7, r, dim) = dim == 2 ? 49 * r^5 : 56 * r^5

function _lap_phi(b::Gaussian, r, dim)
    ε2 = b.ε^2
    return (-2 * dim * ε2 + 4 * ε2^2 * r^2) * exp(-ε2 * r^2)
end

function _lap_phi(b::MQ, r, dim)
    if b.paper
        s3 = (r * r + b.C^2)^(1.5)
        return dim == 2 ? (r * r + 2 * b.C^2) / s3 : (2 * r * r + 3 * b.C^2) / s3
    end
    ε2 = b.ε^2
    s3 = (1 + ε2 * r * r)^(1.5)
    return dim == 2 ? ε2 * (2 + ε2 * r * r) / s3 : ε2 * (3 + 2 * ε2 * r * r) / s3
end

function _lap_phi(b::AbstractRadialBasis, r, dim)
    ε = 1e-7
    rp, rm = r + ε, max(r - ε, 0.0)
    φp, φ0, φm = b(rp), b(r), b(rm)
    φ′ = (φp - φm) / (rp - rm + 1e-30)
    φ′′ = (φp - 2φ0 + φm) / (ε * ε)
    return φ′′ + (dim - 1) / (r + AVOID_INF) * φ′
end

function laplace_particular(basis::AbstractRadialBasis, r::Real; dim::Int = 2)
    return _lap_part(basis, float(r), dim)
end

_lap_part(::PHS1, r, dim) = r^3 / (3 * (1 + dim))
function _lap_part(::PHS2, r, dim)
    dim == 2 && return r^4 / 16 * (log(r + AVOID_INF) - 0.25)
    return r^4 / 20 * (log(r + AVOID_INF) - 0.2)
end
_lap_part(::PHS3, r, dim) = r^5 / (5 * (3 + dim))
function _lap_part(::PHS4, r, dim)
    dim == 2 && return r^6 / 36 * (log(r + AVOID_INF) - 1 / 6)
    return r^6 / 42 * (log(r + AVOID_INF) - 1 / 7)
end
_lap_part(::PHS5, r, dim) = r^7 / (7 * (5 + dim))
function _lap_part(::PHS6, r, dim)
    dim == 2 && return r^8 / 64 * (log(r + AVOID_INF) - 1 / 8)
    return r^8 / 72 * (log(r + AVOID_INF) - 1 / 9)
end
_lap_part(::PHS7, r, dim) = r^9 / (9 * (7 + dim))

# Non-PHS Laplace particular (closed form; no quadrature)
function _lap_part(b::IMQ, r, dim)
    dim == 2 || throw(ArgumentError("IMQ laplace_particular closed-form in 2D only"))
    ε = float(b.ε)
    s = sqrt(1 + ε^2 * r^2)
    return (s + log(2 / (s + 1)) - 1) / ε^2
end
function _lap_part(b::MQ, r, dim)
    dim == 2 || throw(ArgumentError("MQ laplace_particular closed-form in 2D only"))
    if b.paper
        C = float(b.C)
        s = sqrt(r * r + C * C)
        p = s^3 / 9 + (C * C / 3) * s - (C^3 / 3) * log(C + s)
        p0 = C^3 * (4 / 9 - log(2C) / 3)
        return p - p0
    end
    ε = float(b.ε)
    s = sqrt(1 + ε^2 * r^2)
    return ((2 + ε^2 * r^2) * s - 2 - 3 * log((1 + s) / 2)) / (9 * ε^2)
end
function _lap_part(b::Gaussian, r, dim)
    dim == 2 || throw(ArgumentError("Gaussian laplace_particular closed-form in 2D only"))
    ε = float(b.ε)
    a = ε^2
    # (γ + log(a R^2) + Ei(-a R^2)) / (4 a)
    return (Base.MathConstants.eulergamma + log(max(a * r^2, AVOID_INF)) + expinti(-a * r^2)) / (4 * a)
end
function _lap_part(b::AbstractWendland, r, dim)
    throw(ArgumentError("laplace_particular for Wendland not implemented (use PHS for DRM)"))
end
function _lap_part(b::AbstractRadialBasis, r, dim)
    throw(ArgumentError("no closed-form laplace_particular for $(typeof(b)) dim=$dim"))
end

# -----------------------------------------------------------------------------
# Local RBF-FD
# -----------------------------------------------------------------------------

"""
    rbf_neighbors(x, pts, k) -> Vector{Int}
    rbf_neighbors(tree, x, k) -> Vector{Int}

k-nearest neighbour indices using NearestNeighbors.jl
(https://github.com/KristofferC/NearestNeighbors.jl) via `KDTree` + `knn`.
Prefer the tree form for repeated queries.
"""
function rbf_neighbors(x::Point, pts::AbstractVector{<:Point}, k::Int)
    n = length(pts)
    k = min(k, n)
    n == 0 && return Int[]
    tree = _rbf_kdtree(pts)
    return rbf_neighbors(tree, x, k)
end

function rbf_neighbors(tree::KDTree, x::Point, k::Int)
    k = min(k, length(tree.data))
    # pass StaticVector matching tree point type (avoids slow conversions)
    idxs, = knn(tree, x, k, true)
    return idxs
end

function _rbf_kdtree(pts::AbstractVector{<:Point})
    # NearestNeighbors needs a concrete Vector{<:StaticVector}
    isempty(pts) && throw(ArgumentError("empty point set"))
    D = length(pts[1])
    data = [SVector{D, Float64}(p) for p in pts]
    return KDTree(data)
end

struct LocalRBF{B,T}
    pts::Vector{Point}
    basis::B
    k::Int
    ridge::Float64
    tree::T
end

function local_rbf_fit(
        pts::AbstractVector{<:Point},
        basis::AbstractRadialBasis = PHS(3; poly_deg = 1);
        k::Int = 15,
        ridge::Real = 1e-12,
    )
    pts = collect(Point, pts)
    tree = _rbf_kdtree(pts)
    return LocalRBF(pts, basis, min(k, length(pts)), float(ridge), tree)
end

function local_rbf_eval(lr::LocalRBF, x::Point, y::AbstractVector{<:Real})
    ids = rbf_neighbors(lr.tree, x, lr.k)
    w = rbf_cardinal(x, lr.pts, ids; basis = lr.basis, ridge = lr.ridge)
    s = 0.0
    @inbounds for j in eachindex(lr.pts)
        s += w[j] * y[j]
    end
    return s
end

function rbf_laplacian_weights(
        x::Point,
        pts::AbstractVector{<:Point};
        basis::AbstractRadialBasis = PHS(3; poly_deg = 1),
        k::Int = 15,
        ridge::Real = 1e-12,
        dim::Int = length(x),
    )
    ids = collect(rbf_neighbors(x, pts, min(k, length(pts))))
    m = length(ids)
    xs = [pts[j] for j in ids]
    deg = poly_deg(basis)
    while deg >= 0 && rbf_npoly(dim, deg) > m
        deg -= 1
    end
    npoly = rbf_npoly(dim, deg)
    A = zeros(m + npoly, m + npoly)
    hh = max(rbf_length_scale(xs), 1e-14)
    @inbounds for j in 1:m, i in 1:m
        A[i, j] = basis(_scale_r(norm(xs[i] - xs[j]), hh))
    end
    ε = float(ridge) * (tr(view(A, 1:m, 1:m)) / m + 1)
    @inbounds for i in 1:m
        A[i, i] += ε
    end
    if npoly > 0
        mon = MonomialBasis(dim, deg)
        @inbounds for i in 1:m
            p = mon(xs[i])
            for α in 1:npoly
                A[i, m + α] = p[α]
                A[m + α, i] = p[α]
            end
        end
    end
    rhs = zeros(m + npoly)
    @inbounds for j in 1:m
        rhs[j] = laplacian_phi(basis, norm(x - xs[j]) / hh; dim = dim) / hh^2
    end
    coef = A \ rhs
    w = zeros(length(pts))
    @inbounds for (t, j) in enumerate(ids)
        w[j] = coef[t]
    end
    return w
end

# -----------------------------------------------------------------------------
# Partition of Unity
# -----------------------------------------------------------------------------

struct PURBF{B,T}
    pts::Vector{Point}
    basis::B
    centers::Vector{Point}
    radii::Vector{Float64}
    k_local::Int
    ridge::Float64
    tree::T
end

function pu_rbf_fit(
        pts::AbstractVector{<:Point},
        basis::AbstractRadialBasis = PHS(3; poly_deg = 1);
        n_patches::Int = 0,
        overlap::Float64 = 1.5,
        k_local::Int = 20,
        ridge::Real = 1e-12,
    )
    pts = collect(Point, pts)
    n = length(pts)
    np = n_patches > 0 ? n_patches : max(1, Int(ceil(sqrt(n) / 2)))
    dim = length(pts[1])
    lo = [minimum(p[d] for p in pts) for d in 1:dim]
    hi = [maximum(p[d] for p in pts) for d in 1:dim]
    centers = Point[]
    if dim == 2
        g = max(2, Int(ceil(sqrt(np))))
        gx = range(lo[1], hi[1]; length = g)
        gy = range(lo[2], hi[2]; length = g)
        for x in gx, y in gy
            push!(centers, Point2D(x, y))
        end
    else
        g = max(2, Int(ceil(np^(1 / 3))))
        tt = range(0, 1; length = g)
        for a in tt, b in tt, c in tt
            push!(centers, Point3D(
                lo[1] + a * (hi[1] - lo[1]),
                lo[2] + b * (hi[2] - lo[2]),
                lo[3] + c * (hi[3] - lo[3]),
            ))
        end
    end
    sp = length(centers) == 1 ? (norm(hi .- lo) + 1.0) : rbf_length_scale(centers)
    radii = fill(overlap * max(sp, 1e-6) * 3, length(centers))
    return PURBF(pts, basis, centers, radii, min(k_local, n), float(ridge), _rbf_kdtree(pts))
end

_pu_weight(r, R) = (ρ = r / R; ρ >= 1 ? 0.0 : (1 - ρ)^4 * (4ρ + 1))

function pu_rbf_eval(pu::PURBF, x::Point, y::AbstractVector{<:Real})
    num = 0.0
    den = 0.0
    @inbounds for (ic, c) in enumerate(pu.centers)
        R = pu.radii[ic]
        w = _pu_weight(norm(x - c), R)
        w <= 0 && continue
        ids = Int[]
        for j in eachindex(pu.pts)
            norm(pu.pts[j] - c) <= R && push!(ids, j)
        end
        length(ids) < 3 && continue
        if length(ids) > pu.k_local
            idxs = rbf_neighbors(pu.tree, x, min(pu.k_local * 4, length(pu.pts)))
            idset = Set(ids)
            ids = [j for j in idxs if j in idset]
            ids = ids[1:min(length(ids), pu.k_local)]
            isempty(ids) && continue
        end
        ww = rbf_cardinal(x, pu.pts, ids; basis = pu.basis, ridge = pu.ridge)
        val = dot(ww, y)
        num += w * val
        den += w
    end
    den < 1e-30 && return local_rbf_eval(LocalRBF(pu.pts, pu.basis, pu.k_local, pu.ridge), x, y)
    return num / den
end

# -----------------------------------------------------------------------------
# Rational RBF
# -----------------------------------------------------------------------------

struct RationalRBF{B}
    pts::Vector{Point}
    basis::B
    α::Vector{Float64}
    β::Vector{Float64}
    h::Float64
end

function rational_rbf_fit(
        pts::AbstractVector{<:Point},
        y::AbstractVector{<:Real},
        basis::AbstractRadialBasis = IMQ(1.0; poly_deg = -1);
        ridge::Real = 1e-10,
    )
    pts = collect(Point, pts)
    n = length(pts)
    length(y) == n || throw(DimensionMismatch())
    hh = max(rbf_length_scale(pts), 1e-14)
    A = zeros(n, n)
    @inbounds for j in 1:n, i in 1:j
        a = basis(_scale_r(norm(pts[i] - pts[j]), hh))
        A[i, j] = a
        A[j, i] = a
    end
    ε = float(ridge) * (tr(A) / n + 1)
    @inbounds for i in 1:n
        A[i, i] += ε
    end
    β = A \ ones(n)
    β ./= (sum(abs, β) / n + 1e-14)
    qv = A * β
    qv = qv .+ 0.1 .* sign.(qv .+ 0.1)
    α = A \ (float.(y) .* qv)
    return RationalRBF(pts, basis, α, β, hh)
end

function rational_rbf_eval(rr::RationalRBF, x::Point)
    p = 0.0
    q = 0.0
    @inbounds for i in eachindex(rr.pts)
        φ = rr.basis(_scale_r(norm(x - rr.pts[i]), rr.h))
        p += rr.α[i] * φ
        q += rr.β[i] * φ
    end
    return p / (q + 1e-14 * sign(q + 1e-14))
end
rational_rbf_eval(rr::RationalRBF, xs::AbstractVector{<:Point}) =
    [rational_rbf_eval(rr, x) for x in xs]

# -----------------------------------------------------------------------------
# Kansa Poisson
# -----------------------------------------------------------------------------

function kansa_poisson(
        interior::AbstractVector{<:Point},
        boundary::AbstractVector{<:Point},
        f,
        g;
        basis::AbstractRadialBasis = PHS(3; poly_deg = 1),
        ridge::Real = 1e-12,
    )
    ni = length(interior)
    nb = length(boundary)
    centres = vcat(collect(Point, interior), collect(Point, boundary))
    n = ni + nb
    dim = length(centres[1])
    deg = poly_deg(basis)
    while deg >= 0 && rbf_npoly(dim, deg) > n
        deg -= 1
    end
    npoly = rbf_npoly(dim, deg)
    hh = max(rbf_length_scale(centres), 1e-14)
    N = n + npoly
    A = zeros(N, N)
    rhs = zeros(N)
    mon = npoly > 0 ? MonomialBasis(dim, deg) : nothing

    @inbounds for i in 1:ni
        xi = interior[i]
        for j in 1:n
            rij = norm(xi - centres[j]) / hh
            A[i, j] = laplacian_phi(basis, rij; dim = dim) / hh^2
        end
        rhs[i] = f isa Function ? float(f(xi)) : float(f[i])
    end
    @inbounds for k in 1:nb
        i = ni + k
        xb = boundary[k]
        for j in 1:n
            A[i, j] = basis(_scale_r(norm(xb - centres[j]), hh))
        end
        if npoly > 0
            p = mon(xb)
            for α in 1:npoly
                A[i, n + α] = p[α]
            end
        end
        rhs[i] = g isa Function ? float(g(xb)) : float(g[k])
    end
    if npoly > 0
        @inbounds for α in 1:npoly, j in 1:n
            A[n + α, j] = mon(centres[j])[α]
        end
    end
    ε = float(ridge) * (sum(abs, A) / (N * N) + 1)
    @inbounds for i in 1:n
        A[i, i] += ε
    end
    coef = A \ rhs
    β = npoly > 0 ? coef[(n + 1):end] : Float64[]
    return (; α = coef[1:n], centres, β, basis, h = hh, deg)
end

function kansa_eval(sol, x::Point)
    s = 0.0
    @inbounds for j in eachindex(sol.centres)
        s += sol.α[j] * sol.basis(_scale_r(norm(x - sol.centres[j]), sol.h))
    end
    if !isempty(sol.β) && sol.deg >= 0
        mon = MonomialBasis(length(x), sol.deg)
        p = mon(x)
        for αi in eachindex(sol.β)
            s += sol.β[αi] * p[αi]
        end
    end
    return s
end

# -----------------------------------------------------------------------------
# Compare methods
# -----------------------------------------------------------------------------

function compare_rbf_methods(
        pts::AbstractVector{<:Point},
        y::AbstractVector{<:Real},
        xtest::AbstractVector{<:Point},
        ytrue::AbstractVector{<:Real};
        bases = nothing,
        k_local::Int = 15,
    )
    if bases === nothing
        # scale Wendland support to data
        h = rbf_length_scale(pts)
        bases = [
            PHS(3; poly_deg = 1),
            IMQ(1.0; poly_deg = 1),
            Gaussian(1.0; poly_deg = 1),
            MQ(1.0; poly_deg = 1),
            WendlandC2(5h; poly_deg = 0),
            WendlandC4(5h; poly_deg = 0),
        ]
    end
    results = NamedTuple[]
    for b in bases
        bname = string(typeof(b).name.name)
        for (meth, evalf) in (
                (:global, (x, y) -> begin
                        rbf = RBF(pts, b)
                        rbf_evaluate(rbf, x, y)
                    end),
                (:local, (x, y) -> begin
                        lr = local_rbf_fit(pts, b; k = k_local)
                        [local_rbf_eval(lr, xi, y) for xi in x]
                    end),
                (:pu, (x, y) -> begin
                        pu = pu_rbf_fit(pts, b; k_local = k_local)
                        [pu_rbf_eval(pu, xi, y) for xi in x]
                    end),
            )
            t0 = time()
            try
                ŷ = evalf(xtest, y)
                dt = time() - t0
                err = ŷ .- ytrue
                push!(results, (; method = meth, basis = bname,
                    rmse = sqrt(mean(abs2, err)), maxerr = maximum(abs, err), time = dt))
            catch
                push!(results, (; method = meth, basis = bname, rmse = Inf, maxerr = Inf, time = NaN))
            end
        end
    end
    t0 = time()
    try
        rr = rational_rbf_fit(pts, y, IMQ(1.0; poly_deg = -1))
        ŷ = rational_rbf_eval(rr, xtest)
        dt = time() - t0
        err = ŷ .- ytrue
        push!(results, (; method = :rational, basis = "IMQ",
            rmse = sqrt(mean(abs2, err)), maxerr = maximum(abs, err), time = dt))
    catch
        push!(results, (; method = :rational, basis = "IMQ", rmse = Inf, maxerr = Inf, time = NaN))
    end
    return results
end
