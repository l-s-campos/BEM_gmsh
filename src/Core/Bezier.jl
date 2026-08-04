# =============================================================================
# Isogeometric / Bézier extraction utilities
# =============================================================================
# Bernstein basis on the reference interval [-1, 1], Bézier extraction of open
# B-splines (Borden et al. 2011), NURBS rationalization, and an
# `AbstractPolynomial` type with the same shapefun interface as Legendre /
# Equispaced (Lagrange) elements.
#
# Geometry on a Bézier element:
#   x(ξ) = Σ_a R_a(ξ) P_a ,   R = rationalize(B · C, w)
# where B are Bernstein polynomials, C is the extraction operator, w NURBS
# weights, and P_a the control points stored in `dad.Nodes[elem.index]`.

export Bernstein, BezierElement
export bernstein, dbernstein, bernstein_all, dbernstein_all
export bezier_extraction, open_knot_vector, greville_abscissae
export lagrange_to_bezier_matrix, bezier_controls_from_lagrange
export element_shapefun, apply_extraction
export is_bezier_element, bernstein_degree

# ---------------------------------------------------------------------------
# Bernstein polynomials on [-1, 1]
# ---------------------------------------------------------------------------

"""Map ξ ∈ [-1,1] → u ∈ [0,1]."""
@inline _u01(ξ::Real) = (float(ξ) + 1) / 2
@inline _du01_dξ() = 0.5

"""
    bernstein(i, p, ξ)

Bernstein basis polynomial ``B_i^p`` of degree `p`, index `i ∈ 0:p`, evaluated
at reference coordinate `ξ ∈ [-1,1]`.
"""
function bernstein(i::Integer, p::Integer, ξ::Real)
    0 <= i <= p || throw(ArgumentError("bernstein index i=$i out of 0:$p"))
    u = _u01(ξ)
    # B_i^p(u) = C(p,i) u^i (1-u)^{p-i}
    return binomial(p, i) * u^i * (1 - u)^(p - i)
end

"""Derivative dB_i^p / dξ on [-1,1]."""
function dbernstein(i::Integer, p::Integer, ξ::Real)
    0 <= i <= p || throw(ArgumentError("bernstein index i=$i out of 0:$p"))
    p == 0 && return 0.0
    u = _u01(ξ)
    # dB/du = p (B_{i-1}^{p-1} - B_i^{p-1})
    Bm = i == 0 ? 0.0 : binomial(p - 1, i - 1) * u^(i - 1) * (1 - u)^(p - i)
    Bp = i == p ? 0.0 : binomial(p - 1, i) * u^i * (1 - u)^(p - 1 - i)
    return p * (Bm - Bp) * _du01_dξ()
end

"""All Bernstein basis values at ξ as a length-(p+1) vector (i = 0…p)."""
function bernstein_all(p::Integer, ξ::Real)
    B = Vector{Float64}(undef, p + 1)
    @inbounds for i in 0:p
        B[i + 1] = bernstein(i, p, ξ)
    end
    return B
end

"""All Bernstein derivatives dB/dξ at ξ."""
function dbernstein_all(p::Integer, ξ::Real)
    dB = Vector{Float64}(undef, p + 1)
    @inbounds for i in 0:p
        dB[i + 1] = dbernstein(i, p, ξ)
    end
    return dB
end

function bernstein_matrix(p::Integer, ξs::AbstractVector)
    nq = length(ξs)
    N = Matrix{Float64}(undef, nq, p + 1)
    @inbounds for (k, ξ) in enumerate(ξs)
        for i in 0:p
            N[k, i + 1] = bernstein(i, p, ξ)
        end
    end
    return N
end

function dbernstein_matrix(p::Integer, ξs::AbstractVector)
    nq = length(ξs)
    dN = Matrix{Float64}(undef, nq, p + 1)
    @inbounds for (k, ξ) in enumerate(ξs)
        for i in 0:p
            dN[k, i + 1] = dbernstein(i, p, ξ)
        end
    end
    return dN
end

# ---------------------------------------------------------------------------
# Bernstein as AbstractPolynomial  (Lagrange-like interface)
# ---------------------------------------------------------------------------

"""
    Bernstein(p) <: AbstractPolynomial

Polynomial space of degree `p` with **Bernstein / Bézier** basis on [-1,1].

`shapefun(::Bernstein, ξ)` returns Bernstein values and ξ-derivatives — the
same `(N, dN)` contract as [`Legendre`](@ref) / [`Equispaced`](@ref).

Per-element Bézier extraction and NURBS weights live on [`Element`](@ref)
(`extraction`, `nurbs_weights`) and are applied by [`element_shapefun`](@ref).
"""
struct Bernstein{T<:Number, X<:AbstractVector{T}, W<:AbstractVector, D<:AbstractMatrix} <: AbstractPolynomial{T}
    shift::T
    scale::T
    nodes::X
    weights::W
    Dmat::D
    degree::Int
end

function Bernstein{T}(p::Integer) where {T<:Number}
    p >= 0 || throw(ArgumentError("Bernstein degree must be ≥ 0"))
    # Greville nodes of a single Bezier span ≡ Bernstein Greville = i/p mapped to [-1,1]
    nodes = if p == 0
        T[0]
    else
        T[(2 * i / p - 1) for i in 0:p]
    end
    # analytic derivative matrix at Bernstein Greville nodes
    Dmat = Matrix{T}(undef, p + 1, p + 1)
    @inbounds for j in 0:p
        ξ = nodes[j + 1]
        dB = dbernstein_all(p, ξ)
        for i in 0:p
            Dmat[j + 1, i + 1] = T(dB[i + 1])  # rows = eval pts, cols = basis
        end
    end
    # dummy barycentric weights (unused — shapefun is specialized)
    w = ones(T, p + 1)
    return Bernstein{T, typeof(nodes), typeof(w), typeof(Dmat)}(
        zero(T), one(T), nodes, w, Dmat, Int(p),
    )
end

Bernstein(p::Integer) = Bernstein{Float64}(p)
bernstein_degree(poly::Bernstein) = poly.degree
degree(poly::Bernstein) = poly.degree

function shapefun(poly::Bernstein, x::AbstractVector)
    N = bernstein_matrix(poly.degree, x)
    dN = dbernstein_matrix(poly.degree, x)
    return N, dN
end

function shapefun(poly::Bernstein, x::Number)
    N = reshape(bernstein_all(poly.degree, x), 1, :)
    dN = reshape(dbernstein_all(poly.degree, x), 1, :)
    return N, dN
end

# ---------------------------------------------------------------------------
# Extraction + rational weights
# ---------------------------------------------------------------------------

"""
    apply_extraction(N, dN, elem) -> (Ñ, dÑ)

Apply Bézier extraction `Ñ = N * C` and optional NURBS weights on `elem`.
If `elem.extraction === nothing`, returns `(N, dN)` unchanged.
"""
function apply_extraction(N::AbstractMatrix, dN::AbstractMatrix, elem::Element)
    C = elem.extraction
    if C === nothing
        return N, dN
    end
    size(N, 2) == size(C, 1) || throw(DimensionMismatch(
        "Bernstein width $(size(N,2)) ≠ extraction rows $(size(C,1))",
    ))
    Ne = N * C
    dNe = dN * C
    w = elem.nurbs_weights
    if w === nothing
        return Ne, dNe
    end
    length(w) == size(C, 2) || throw(DimensionMismatch(
        "nurbs_weights length $(length(w)) ≠ extraction cols $(size(C,2))",
    ))
    return rationalize_shape(Ne, dNe, w)
end

"""Projective weights → rational NURBS basis and derivatives."""
function rationalize_shape(N::AbstractMatrix, dN::AbstractMatrix, w::AbstractVector)
    nq, n = size(N)
    length(w) == n || throw(DimensionMismatch())
    Nr = similar(N)
    dNr = similar(dN)
    @inbounds for k in 1:nq
        W = zero(eltype(N))
        dW = zero(eltype(N))
        for j in 1:n
            W += N[k, j] * w[j]
            dW += dN[k, j] * w[j]
        end
        invW = 1 / W
        invW2 = invW * invW
        for j in 1:n
            nj = N[k, j] * w[j]
            dnj = dN[k, j] * w[j]
            Nr[k, j] = nj * invW
            dNr[k, j] = (dnj * W - nj * dW) * invW2
        end
    end
    return Nr, dNr
end

"""
    element_shapefun(poly, elem, ξ) -> (N, dN)

Shape functions for one element: polynomial basis of `poly` (Lagrange or
Bernstein), then Bézier extraction + NURBS weights stored on `elem`.

Drop-in replacement for `shapefun(poly, ξ)` wherever an `Element` is available.
"""
function element_shapefun(poly::AbstractPolynomial, elem::Element, ξ)
    N, dN = shapefun(poly, ξ)
    return apply_extraction(N, dN, elem)
end

is_bezier_element(elem::Element) = elem.extraction !== nothing

# ---------------------------------------------------------------------------
# Open knot vectors, Greville, Bézier extraction (Borden et al. 2011)
# ---------------------------------------------------------------------------

"""
    open_knot_vector(n_el, p; a=0, b=1)

Open (clamped) knot vector for `n_el` equal spans of degree `p` on `[a,b]`.
Length = `n_el + 2p + 1` → `n_el + p` control points.
"""
function open_knot_vector(n_el::Integer, p::Integer; a::Real = 0.0, b::Real = 1.0)
    n_el >= 1 || throw(ArgumentError("n_el ≥ 1"))
    p >= 1 || throw(ArgumentError("p ≥ 1"))
    interior = n_el > 1 ? collect(range(a, b; length = n_el + 1))[2:(end - 1)] : Float64[]
    return vcat(fill(float(a), p + 1), interior, fill(float(b), p + 1))
end

"""Greville abscissae for open knot vector `Ξ` of degree `p`."""
function greville_abscissae(Ξ::AbstractVector{<:Real}, p::Integer)
    ncp = length(Ξ) - p - 1
    γ = Vector{Float64}(undef, ncp)
    @inbounds for i in 1:ncp
        s = 0.0
        for j in 1:p
            s += Ξ[i + j]
        end
        γ[i] = s / p
    end
    return γ
end

"""
    bezier_extraction(Ξ, p) -> (Cs, spans)

Bézier extraction operators for an open knot vector `Ξ` of degree `p`.

Returns
- `Cs::Vector{Matrix{Float64}}` — each `C^e` is `(p+1) × (p+1)` with
  ``N^e(ξ) = B(ξ)\\, C^e`` (Bernstein row × extraction),
- `spans::Vector{Tuple{Int,Float64,Float64}}` — `(global_first_control, u_left, u_right)`
  for every non-empty knot span (1-based control index of the first local basis).
"""
function bezier_extraction(Ξ::AbstractVector{<:Real}, p::Integer)
    knots = collect(Float64, Ξ)
    m = length(knots)
    nb = m - p - 1
    nb >= p + 1 || throw(ArgumentError("knot vector too short for degree $p"))
    # Local span-wise solve B·C = N (stable for open/uniform and non-uniform knots)
    return _bezier_extraction_local(knots, p)
end

"""
Reliable Bézier extraction: for each non-empty span, the local B-spline basis
restricted to that span equals Bernstein × C.  C is obtained by evaluating both
bases at p+1 distinct points in the span and solving B_mat * C = N_mat.
"""
function _bezier_extraction_local(knots::Vector{Float64}, p::Integer)
    m = length(knots)
    Cs = Matrix{Float64}[]
    spans = Tuple{Int, Float64, Float64}[]
    # find unique span starts
    i = p + 1
    while i < m - p
        u0 = knots[i]
        u1 = knots[i + 1]
        if u1 > u0 + 1e-15
            first_cp = i - p
            C = _extraction_operator_span(knots, p, i)
            push!(Cs, C)
            push!(spans, (first_cp, u0, u1))
        end
        i += 1
    end
    isempty(Cs) && error("no non-empty knot spans in knot vector")
    return Cs, spans
end

function _extraction_span(knots, p, a)
    return _extraction_operator_span(knots, p, a)
end

function _extraction_operator_span(knots::Vector{Float64}, p::Integer, span_idx::Integer)
    # span_idx is the left-knot index (1-based) of [Ξ_i, Ξ_{i+1})
    u0, u1 = knots[span_idx], knots[span_idx + 1]
    # sample p+1 points in open interval mapped to [-1,1]
    ξs = p == 0 ? [0.0] : collect(range(-1.0, 1.0; length = p + 1))
    Bmat = bernstein_matrix(p, ξs)                      # (p+1)×(p+1)
    Nmat = Matrix{Float64}(undef, p + 1, p + 1)
    first_cp = span_idx - p
    @inbounds for (k, ξ) in enumerate(ξs)
        u = (ξ + 1) / 2 * (u1 - u0) + u0
        for j in 0:p
            Nmat[k, j + 1] = bspline_basis(first_cp + j, p, knots, u)
        end
    end
    # B * C = N  ⇒  C = B \\ N
    return Bmat \ Nmat
end

"""Cox–de Boor B-spline basis N_{i,p}(u), 1-based control index i."""
function bspline_basis(i::Integer, p::Integer, Ξ::AbstractVector{<:Real}, u::Real)
    n = length(Ξ) - 1
    # handle right endpoint
    if u >= Ξ[end] - 1e-14
        # only last basis nonzero
        nb = length(Ξ) - p - 1
        return i == nb ? 1.0 : 0.0
    end
    # degree 0
    if p == 0
        return (Ξ[i] <= u < Ξ[i + 1]) ? 1.0 : 0.0
    end
    # recursive
    d = zeros(Float64, p + 1)
    @inbounds for j in 0:p
        d[j + 1] = (Ξ[i + j] <= u < Ξ[i + j + 1]) ? 1.0 : 0.0
    end
    @inbounds for k in 1:p
        for j in 0:(p - k)
            left = Ξ[i + j + k] - Ξ[i + j]
            right = Ξ[i + j + k + 1] - Ξ[i + j + 1]
            term1 = left > 0 ? (u - Ξ[i + j]) / left * d[j + 1] : 0.0
            term2 = right > 0 ? (Ξ[i + j + k + 1] - u) / right * d[j + 2] : 0.0
            d[j + 1] = term1 + term2
        end
    end
    return d[1]
end

"""Evaluate all p+1 local B-spline bases for span starting at `first_cp`."""
function bspline_bases_local(first_cp::Integer, p::Integer, Ξ, u::Real)
    N = Vector{Float64}(undef, p + 1)
    @inbounds for j in 0:p
        N[j + 1] = bspline_basis(first_cp + j, p, Ξ, u)
    end
    return N
end

# ---------------------------------------------------------------------------
# Lagrange (Equispaced) ↔ Bézier control conversion
# ---------------------------------------------------------------------------

"""
    lagrange_to_bezier_matrix(p) -> E

Matrix `E` such that Bezier controls ``c = E * x_L`` where `x_L` are values at
Equispaced Lagrange nodes on [-1,1] of degree `p`, and
``Σ L_i(ξ) x_i = Σ B_j(ξ) c_j``.
"""
function lagrange_to_bezier_matrix(p::Integer)
    poly = Equispaced(p)
    ξs = collect(nodes(poly))                 # equispaced including ends
    Lmat = Matrix(interpolation_matrix(poly, ξs))  # identity at nodes
    # At nodes: Lmat = I, Bmat * E = I ⇒ E = Bmat \\ I = inv(Bmat)
    Bmat = bernstein_matrix(p, ξs)
    return Bmat \ Matrix{Float64}(I, p + 1, p + 1)
end

"""Convert equispaced Lagrange nodal coordinates to Bézier controls."""
function bezier_controls_from_lagrange(xL::AbstractVector{<:SVector{D, T}}, p::Integer) where {D, T}
    E = lagrange_to_bezier_matrix(p)
    length(xL) == p + 1 || throw(DimensionMismatch("expected $(p+1) Lagrange nodes, got $(length(xL))"))
    Cpts = Vector{SVector{D, Float64}}(undef, p + 1)
    @inbounds for j in 1:(p + 1)
        acc = zero(SVector{D, Float64})
        for i in 1:(p + 1)
            acc += E[j, i] * SVector{D, Float64}(xL[i])
        end
        Cpts[j] = acc
    end
    return Cpts
end

# ---------------------------------------------------------------------------
# BezierElement alias documentation helper
# ---------------------------------------------------------------------------

"""
    BezierElement(index, jacobian, length, region, C; weights=nothing)

Construct an [`Element`](@ref) carrying Bézier extraction `C` (and optional
NURBS `weights`). Same fields as a Lagrange element plus IGA data — assembly
code that only uses `index / Jacobian / Length / Region` keeps working; shape
functions go through [`element_shapefun`](@ref).
"""
function BezierElement(
        index::Vector{Int},
        jacobian::Vector{Float64},
        length::Float64,
        region::Int,
        C::AbstractMatrix;
        weights = nothing,
        controls = nothing,
    )
    w = weights === nothing ? nothing : collect(Float64, weights)
    return Element(
        index = index,
        Jacobian = jacobian,
        Length = length,
        Region = region,
        extraction = Matrix{Float64}(C),
        nurbs_weights = w,
        controls = controls,
    )
end

"""Geometry nodes for integration: explicit controls if present, else `dad.Nodes[index]`."""
element_geometry(dad, elem::Element) =
    elem.controls === nothing ? dad.Nodes[elem.index] : elem.controls
