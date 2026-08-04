
abstract type AbstractPolynomial{T<:Number} end

for name in [:Chebyshev1, :Chebyshev2, :Legendre, :Equispaced]
    @eval struct $name{T<:Number,X<:AbstractVector{T},W<:AbstractVector,D<:AbstractMatrix} <: AbstractPolynomial{T}
        shift::T
        scale::T
        nodes::X
        weights::W
        Dmat::D
        function $name{T,X,W,D}(shift::T, scale::T, nodes::X, weights::W, Dmat::D) where {T<:Number,X<:AbstractVector{T},W<:AbstractVector,D<:AbstractMatrix}
            length(nodes) == length(weights) || throw(DimensionMismatch("nodes and weights have different lengths"))
            new{T,X,W,D}(shift, scale, nodes, weights, Dmat)
        end
    end
    @eval function $name{T}(N::Integer, start::T, stop::T) where {T<:Number}
        shift = T(stop + start) / 2
        scale = T(stop - start) / 2
        (nodes, weights) = nodes_weights($name{T}, Int(N), shift, scale)
        Dmat = diff_matrix(weights, nodes, N)
        $name{T,typeof(nodes),typeof(weights),typeof(Dmat)}(shift, scale, nodes, weights, Dmat)
    end
    @eval $name{T}(N::Integer, start::Number, stop::Number) where {T} = $name{T}(N, convert(T, start), convert(T, stop))
    @eval function $name(N::Integer, start::Number, stop::Number)
        start, stop = float.(promote(start, stop))
        $name{typeof(start)}(N, start, stop)
    end
    @eval $name(N::Integer) = $name(N, -1, 1)
    @eval $name{T}(N::Integer) where {T} = $name(N, T(-1), T(1))
end

struct ArbitraryPolynomial{T<:Number,X<:AbstractVector{T},W<:AbstractVector} <: AbstractPolynomial{T}
    nodes::X
    weights::W
    function ArbitraryPolynomial(nodes::AbstractVector{T}) where {T<:Number}
        _weights = weights(ArbitraryPolynomial{T}, nodes)
        new{T,typeof(nodes),typeof(_weights)}(nodes, _weights)
    end
end


"""
    degree(poly)

Return the degree of the polynomial specified.
"""
degree(poly::AbstractPolynomial) = length(poly.nodes) - 1 # assumes every poly has a nodes field

"""
    nodes_weights(poly)

Return the nodes and weights of the polynomial specified.
"""
function nodes_weights(::Type{P}, N::Integer, shift=0, scale=1) where {P<:AbstractPolynomial}
    return (nodes(P, N, shift, scale), weights(P, N))
end

nodes_weights(poly::AbstractPolynomial) = (poly.nodes, poly.weights)

function nodes_weights(::Type{<:Legendre{T}}, N::Integer, shift=0, scale=1) where {T}
    if precision(Float64) < precision(T)
        # to do: BigFloat Legendre points are implemented in QuadGK.jl
        error("high-precision $T Legendre support is unimplemented")
    end
    (x, w) = gausslegendre(N + 1) # computes in Float64 precision
    nodes = map(xᵢ -> T(xᵢ * scale + shift), x)
    weights = map(i -> T((2 * isodd(i) - 1) * sqrt((1 - x[i]^2) * w[i])), eachindex(x))
    return (nodes, weights)
end

"""
    weights(poly)
    weights(polytype, N)

Return the Barycentric weights for the specified orthogonal polynomials.  If
an `AbstractPolynomial` type is passed, one must also pass the degree `N``.
"""
function weights end

function weights(::Type{P}, N::Integer) where {P<:AbstractPolynomial}
    _N = Int(N)
    return [_weight(P, _N, j) for j = 0:_N]
end

# Eq. (5.1)
@inline _weight(poly::Type{<:Equispaced{T}}, N::Integer, j::Integer) where {T} = T((2 * xor(isodd(N), iseven(j)) - 1) * binomial(N, j))

# Eq. (5.3)
@inline _weight(poly::Type{<:Chebyshev1{T}}, N::Integer, j::Integer) where {T} = -(2 * iseven(j) - 1) * sinpi(T(2j + 1) / (2N + 2))

# Eq. (5.4)
@inline _weight(poly::Type{<:Chebyshev2{T}}, N::Integer, j::Integer) where {T} = -T((1.0 - 0.5 * ((j == 0) || (j == N))) * (2 * iseven(j) - 1))

function weights(poly::Type{<:ArbitraryPolynomial{T}}, x::AbstractVector{T}) where {T}
    return map(eachindex(x)) do i
        xᵢ = x[i]
        wᵢ = one(T)
        for j = firstindex(x):i-1
            wᵢ *= xᵢ - x[j]
        end
        for j = i+1:lastindex(x)
            wᵢ *= xᵢ - x[j]
        end
        inv(wᵢ)
    end
end

weights(poly::AbstractPolynomial) = poly.weights

"""
    nodes(poly)
    nodes(polytype, N)

Return the nodes for the specified orthogonal polynomials.   If
an `AbstractPolynomial` type is passed, one must also pass the degree `N``.
"""
function nodes end

nodes(poly::Type{<:AbstractPolynomial{T}}, N::Integer) where {T} = nodes(poly, N, zero(T), one(T))

function nodes(::Type{P}, N::Integer, shift::Number, scale::Number) where {P<:AbstractPolynomial}
    _N = Int(N)
    return [_node(P, _N, j) * scale + shift for j = 0:_N]
end

nodes(poly::Type{<:Equispaced}, N::Integer, shift::Number, scale::Number) = range(shift - scale, stop=shift + scale, length=Int(N) + 1)

@inline _node(poly::Type{<:Chebyshev1{T}}, N::Integer, j::Integer) where {T} = -cospi(T(2j + 1) / (2N + 2))

@inline _node(poly::Type{<:Chebyshev2{T}}, N::Integer, j::Integer) where {T} = -cospi(T(j) / N)

nodes(poly::AbstractPolynomial) = poly.nodes

"""
    interpolation_matrix(poly::AbstractPolynomial, x)

Return the interpolation matrix from the nodes of `poly` to the point(s) `x`.
For example :

    P = Chebyshev2{5}()
    x = range(-1, stop=1, length=10)
    M = interpolation_matrix(P, x)

Now `y(x) ≈ M*y₀` given that `y(nodes(poly)) = y₀.
"""
function interpolation_matrix(poly::AbstractPolynomial{T}, x::Union{Number,AbstractVector}) where {T}
    w = weights(poly)
    x₀ = nodes(poly)
    N = degree(poly)
    # M = Matrix{T}(undef, length(x), N+1)
    M = MMatrix{length(x),N + 1,T}(undef)
    # Eq. (4.2)
    for j = eachindex(x)
        xx = convert(T, x[j])
        Msum = zero(T)
        exact = 0
        for i = Base.OneTo(N + 1)
            exact = ifelse(xx == x₀[i], i, exact)
            M[j, i] = w[i] / (xx - x₀[i])
            Msum += M[j, i]
        end
        if Msum == 0
            for i = Base.OneTo(N + 1)
                M[j, i] = zero(T)
            end
        elseif exact > 0
            for i = Base.OneTo(N + 1)
                M[j, i] = zero(T)
            end
            M[j, exact] = one(T)
        else
            for i = Base.OneTo(N + 1)
                M[j, i] /= Msum
            end
        end
    end
    return M
end

"""
    differentiation_matrix(poly::AbstractPolynomial)

Return the differentiation matrix at the nodes of the polynomial specified.

    P = Chebyshev2{5}()
    D = differentiation_matrix(P)

Now dy/dx ≈ `D*y` at the nodes of the polynomial.
"""
function differentiation_matrix(poly::AbstractPolynomial{T}) where {T}
    # Eqs. (9.4) and (9.5)
    w = weights(poly)
    x = nodes(poly)
    N = degree(poly)
    # D = Matrix{T}(undef, N+1, N+1)
    D = MMatrix{N + 1,N + 1,T}(undef)

    for i = Base.OneTo(N + 1)
        Dsum = zero(T)
        for j = Base.OneTo(i - 1)
            temp = (w[j] / w[i]) / (x[i] - x[j])
            D[i, j] = temp
            Dsum += temp
        end
        for j = i+1:N+1
            temp = (w[j] / w[i]) / (x[i] - x[j])
            D[i, j] = temp
            Dsum += temp
        end
        D[i, i] = -Dsum
    end
    return D
end



function diff_matrix(w::AbstractVector, x::Union{Number,AbstractVector}, N::Integer)
    # w = weights(poly)
    # x = nodes(poly)
    # N = degree(poly)
    D = Matrix{Float64}(undef, N + 1, N + 1)
    for i = Base.OneTo(N + 1)
        Dsum = zero(Float64)
        for j = Base.OneTo(i - 1)
            temp = (w[j] / w[i]) / (x[i] - x[j])
            D[i, j] = temp
            Dsum += temp
        end
        for j = i+1:N+1
            temp = (w[j] / w[i]) / (x[i] - x[j])
            D[i, j] = temp
            Dsum += temp
        end
        D[i, i] = -Dsum
    end
    return D
end

function shapefun(poly::AbstractPolynomial, x)
    L = interpolation_matrix(poly, x)
    L, L * poly.Dmat
end


"""
    shapefun2D(poly_x::Polynomial, poly_y::Polynomial, x_eval, y_eval)

2D tensor-product version: returns interpolation_matrix and gradient matrices.
- L: interpolation matrix (value)
- Lx, Ly: ∂/∂x and ∂/∂y gradient matrices at (x_eval, y_eval)
Assumes same order nodes in ξ,η → [-1,1]² reference quad.
"""
function shapefun2D(poly_x::AbstractPolynomial, poly_y::AbstractPolynomial, x_eval, y_eval)
    L_xi = interpolation_matrix(poly_x, x_eval)
    L_eta = interpolation_matrix(poly_y, y_eval)
    L = kron(L_eta, L_xi)  # Value: kron(η, ξ)
    N_xi = length(poly_x.nodes)
    N_eta = length(poly_y.nodes)

    Dx = kron(Matrix(I, N_eta, N_eta), poly_x.Dmat)
    Dy = kron(poly_y.Dmat, Matrix(I, N_xi, N_xi))

    Lx = L * Dx  # ∂/∂x ∝ kron(η, Dξ Lξ); scale by Jacobian if physical
    Ly = L * Dy # ∂/∂y ∝ kron(Dη Lη, ξ)

    return L, Lx, Ly
end


"""
    shapefun2D(poly_x::Polynomial, poly_y::Polynomial, x_eval, y_eval)

2D tensor-product version: returns interpolation_matrix and gradient matrices.
- L: interpolation matrix (value)
- Lx, Ly: ∂/∂x and ∂/∂y gradient matrices at (x_eval, y_eval)
Assumes same order nodes in ξ,η → [-1,1]² reference quad.
"""
function shapefun2D(poly_x::AbstractPolynomial, x_eval)
    L_xi = interpolation_matrix(poly_x, x_eval)
    L = kron(L_xi, L_xi)  # Value: kron(η, ξ)
    N_xi = length(poly_x.nodes)

    Dx = kron(Matrix(I, N_xi, N_xi), poly_x.Dmat)
    Dy = kron(poly_x.Dmat, Matrix(I, N_xi, N_xi))

    Lx = L * Dx  # ∂/∂x ∝ kron(η, Dξ Lξ); scale by Jacobian if physical
    Ly = L * Dy # ∂/∂y ∝ kron(Dη Lη, ξ)

    return L, Lx, Ly
end


"""
    shapefun3d(poly_x::AbstractPolynomial, poly_y::AbstractPolynomial, poly_z::AbstractPolynomial,
             x_eval, y_eval, z_eval)

3D tensor-product for hexahedron [-1,1]³: interpolation + gradients.
Node ordering: vec(ζ-slices × η-rows × ξ-cols).
"""
function shapefun3d(poly_x::AbstractPolynomial, poly_y::AbstractPolynomial, poly_z::AbstractPolynomial,
    x_eval, y_eval, z_eval)
    L_xi = interpolation_matrix(poly_x, x_eval)
    L_eta = interpolation_matrix(poly_y, y_eval)
    L_zeta = interpolation_matrix(poly_z, z_eval)
    L = kron(kron(L_zeta, L_eta), L_xi)  # Value: kron(ζ, η, ξ)

    N_xi = length(poly_x.nodes)
    N_eta = length(poly_y.nodes)
    N_zeta = length(poly_z.nodes)

    # Full diff operators at nodes (ref coords)
    Dx = kron(kron(Matrix(I, N_zeta, N_zeta), Matrix(I, N_eta, N_eta)), poly_x.Dmat)
    Dy = kron(kron(Matrix(I, N_zeta, N_zeta), poly_y.Dmat), Matrix(I, N_xi, N_xi))
    Dz = kron(kron(poly_z.Dmat, Matrix(I, N_eta, N_eta)), Matrix(I, N_xi, N_xi))

    # Gradient matrices: eval shape * nodal diff
    Lx = L * Dx
    Ly = L * Dy
    Lz = L * Dz

    return L, Lx, Ly, Lz
end
