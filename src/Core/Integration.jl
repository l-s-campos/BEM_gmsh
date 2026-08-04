# Singular / nearly-singular integration utilities
# -----------------------------------------------------------------------------
# Policy (from validation vs Dumont / sinh / Sidi / transforms):
#
#  2D curve elements
#    far            → plain GL (caller)
#    nearly         → sinh(a, b=d/L); compose iterate if b ≪ 1
#    on-element     → singular(order=0/1/2)  [Dumont lives outside this file]
#
#  3D surface elements
#    far            → plain tensor GL
#    nearly mild    → tensor sinh
#    nearly strong  → polar + radial sinh   (default when b small / hypersing)
#    on-element     → Duffy (GeometricProperties) / singular radial — not here
#
# Deprecated: BEM.jl-style iterated sinh on already-mapped x (i=1 old).
# Prefer :compose double map on the GL variable u.
# -----------------------------------------------------------------------------

export closest_point_1d, closest_point_2d
export transform, sinhtrans, sinhtrans_J, divide, singular
export nearfield_1d, nearfield_2d, polar_sinh_2d
export nquad_nearfield

using NonlinearSolve
using ADTypes: AutoFiniteDiff
using StaticArrays: @SVector
using FastGaussQuadrature: gausslegendre

# =============================================================================
# Adaptive quadrature size
# =============================================================================

"""Suggested 1D near-field order from scaled distance `b = d/L`."""
function nquad_nearfield(b; nmin=8, nmax=64, nbase=nothing)
    b = max(float(b), 1e-16)
    n = something(nbase, nmin)
    n = max(n, nmin, min(nmax, 8 + 10 * ceil(Int, max(0.0, -log10(b)))))
    return n
end

# =============================================================================
# Sinh maps
# =============================================================================

"""
    sinhtrans(u, w, a, b; iterated=false)

Sinh clustering about peak `a ∈ [-1,1]` with width `b = d/L`.
Returns transformed nodes and weights `w .* J`.

`iterated=true` applies a **compose** double map on the GL variable:
`u → sinh(0,√b) → t → sinh(a,b) → x` (recommended for tiny `b`).

Legacy `sinhtrans(u,w,a,b,i)` with `i==1` now uses compose (not the old
broken second map on already-mapped `x`).
"""
function sinhtrans(u, w, a, b; iterated::Bool=false)
    T = float(eltype(w))
    b = max(T(b), T(1e-14))
    a = T(a)
    if iterated
        t, J1 = _sinh_map(u, zero(T), sqrt(b))
        x, J2 = _sinh_map(t, a, b)
        return x, w .* J1 .* J2
    else
        x, J = _sinh_map(u, a, b)
        return x, w .* J
    end
end

# legacy positional iterated flag (i≠0 → compose)
sinhtrans(u, w, a, b, i::Integer) = sinhtrans(u, w, a, b; iterated=(i != 0))

function sinhtrans_J(u, a, b; iterated::Bool=false)
    w1 = ones(eltype(u), length(u))
    x, wJ = sinhtrans(u, w1, a, b; iterated=iterated)
    return x, wJ
end

function _sinh_map(u, a::T, b::T) where {T<:Real}
    μ = T(0.5) * (asinh((1 + a) / b) + asinh((1 - a) / b))
    η = T(0.5) * (asinh((1 + a) / b) - asinh((1 - a) / b))
    x = @. a + b * sinh(μ * u - η)
    J = @. b * μ * cosh(μ * u - η)
    return x, J
end

"""
    nearfield_1d(a, b; n=nothing, qsi=nothing, w=nothing)

Build 1D near-field rule on [-1,1].
Chooses single vs compose sinh from `b`; densifies `n` if not provided.
"""
function nearfield_1d(a, b; n=nothing, qsi=nothing, w=nothing)
    b = max(float(b), 1e-14)
    a = float(a)
    if qsi === nothing || w === nothing
        nn = something(n, nquad_nearfield(b))
        qsi, w = gausslegendre(nn)
    elseif n !== nothing && n > length(qsi)
        qsi, w = gausslegendre(n)
    end
    iterated = b < 1e-4
    return sinhtrans(qsi, w, a, b; iterated=iterated)
end

# =============================================================================
# Closest-point projection via NonlinearSolve.jl
# =============================================================================

function _proj1d_residual(ξ, p)
    poly, nodes, pf = p
    ξs = ξ isa Number ? ξ : ξ[1]
    N, dN = shapefun(poly, ξs)
    xξ = (N * nodes)[1]
    dx = (dN * nodes)[1]
    return dot(xξ - pf, dx)
end

"""
    closest_point_1d(poly, nodes, pf; ξ0=0, maxiter=25, tol=1e-14) -> (ξ*, x*, dist)
"""
function closest_point_1d(poly, nodes::AbstractVector{<:Point}, pf::Point;
    ξ0=0.0, maxiter=25, tol=1e-14, alg=nothing)
    p = (poly, nodes, pf)
    u0 = float(ξ0)
    nlprob = NonlinearProblem{false}(_proj1d_residual, u0, p)
    solver = alg === nothing ? NewtonRaphson(; autodiff=AutoFiniteDiff()) : alg
    sol = NonlinearSolve.solve(nlprob, solver; abstol=tol, reltol=tol, maxiters=maxiter)
    ξ = sol.u isa Number ? float(sol.u) : float(sol.u[1])

    if ξ < -1 || ξ > 1
        N0, _ = shapefun(poly, -1.0)
        N1, _ = shapefun(poly, 1.0)
        x0 = (N0 * nodes)[1]
        x1 = (N1 * nodes)[1]
        if norm(x0 - pf) <= norm(x1 - pf)
            ξ, xξ = -1.0, x0
        else
            ξ, xξ = 1.0, x1
        end
    else
        N, _ = shapefun(poly, ξ)
        xξ = (N * nodes)[1]
    end
    return ξ, xξ, norm(xξ - pf)
end

function _proj2d_residual(u, p)
    poly, nodes, pf = p
    ξ, η = u[1], u[2]
    N, dNξ, dNη = shapefun2D(poly, poly, ξ, η)
    x = (N * nodes)[1]
    x_ξ = (dNξ * nodes)[1]
    x_η = (dNη * nodes)[1]
    r = x - pf
    return @SVector [dot(r, x_ξ), dot(r, x_η)]
end

"""
    closest_point_2d(poly, nodes, pf; ξ0=(0,0), ...) -> (ξ, η, x*, dist)
"""
function closest_point_2d(poly, nodes::AbstractVector{<:Point}, pf::Point;
    ξ0=(0.0, 0.0), maxiter=25, tol=1e-14, alg=nothing)
    p = (poly, nodes, pf)
    u0 = @SVector [float(ξ0[1]), float(ξ0[2])]
    nlprob = NonlinearProblem{false}(_proj2d_residual, u0, p)
    solver = alg === nothing ? NewtonRaphson(; autodiff=AutoFiniteDiff()) : alg
    sol = NonlinearSolve.solve(nlprob, solver; abstol=tol, reltol=tol, maxiters=maxiter)
    ξ = clamp(float(sol.u[1]), -1.0, 1.0)
    η = clamp(float(sol.u[2]), -1.0, 1.0)
    N, _, _ = shapefun2D(poly, poly, ξ, η)
    x = (N * nodes)[1]
    return ξ, η, x, norm(x - pf)
end

# =============================================================================
# 2D parameter-domain (surface) near-field
# =============================================================================

"""Ray length from (aξ,aη) to ∂[-1,1]² along angle θ."""
function _ray_to_square(aξ, aη, θ)
    c, s = cos(θ), sin(θ)
    tmin = Inf
    if abs(c) > 1e-14
        for edge in (-1.0, 1.0)
            t = (edge - aξ) / c
            if t > 1e-14
                ηh = aη + t * s
                -1 - 1e-12 <= ηh <= 1 + 1e-12 && (tmin = min(tmin, t))
            end
        end
    end
    if abs(s) > 1e-14
        for edge in (-1.0, 1.0)
            t = (edge - aη) / s
            if t > 1e-14
                ξh = aξ + t * c
                -1 - 1e-12 <= ξh <= 1 + 1e-12 && (tmin = min(tmin, t))
            end
        end
    end
    return isfinite(tmin) ? tmin : 0.0
end

function _sinh_radial(u, w, ρmax, b)
    b = max(b, 1e-14)
    tmax = asinh(ρmax / b)
    t = @. tmax * (u + 1) / 2
    ρ = @. b * sinh(t)
    J = @. b * cosh(t) * (tmax / 2)
    return ρ, w .* J
end

"""
    polar_sinh_2d(nρ, nθ, aξ, aη, b) -> (ξ, η, w)

Four triangles from closest point; sinh in ρ, GL in θ. Polar Jacobian ρ included in `w`.
"""
function polar_sinh_2d(nρ::Int, nθ::Int, aξ::Float64, aη::Float64, b::Float64)
    b = max(b, 1e-14)
    aξ = clamp(aξ, -1.0, 1.0)
    aη = clamp(aη, -1.0, 1.0)
    corners = ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0))
    uρ, wρ0 = gausslegendre(nρ)
    uθ, wθ = gausslegendre(nθ)
    ξs = Float64[]; ηs = Float64[]; ws = Float64[]
    sizehint!(ξs, 4 * nρ * nθ); sizehint!(ηs, 4 * nρ * nθ); sizehint!(ws, 4 * nρ * nθ)
    for k in 1:4
        c1 = corners[k]; c2 = corners[mod1(k + 1, 4)]
        θ1 = atan(c1[2] - aη, c1[1] - aξ)
        θ2 = atan(c2[2] - aη, c2[1] - aξ)
        dθ = θ2 - θ1; dθ <= 0 && (dθ += 2π)
        θmid = θ1 + 0.5 * dθ; θhalf = 0.5 * dθ
        for j in eachindex(uθ)
            θ = θmid + θhalf * uθ[j]
            wθj = wθ[j] * θhalf
            ρmax = _ray_to_square(aξ, aη, θ)
            ρmax <= 0 && continue
            ρ, wρ = _sinh_radial(uρ, wρ0, ρmax, b)
            cθ, sθ = cos(θ), sin(θ)
            @inbounds for i in eachindex(ρ)
                push!(ξs, aξ + ρ[i] * cθ)
                push!(ηs, aη + ρ[i] * sθ)
                push!(ws, wρ[i] * wθj * ρ[i])
            end
        end
    end
    return ξs, ηs, ws
end

function _tensor_sinh_2d(n, aξ, aη, b; iterated=false)
    u, w = gausslegendre(n)
    x1, w1 = sinhtrans(u, w, aξ, b; iterated=iterated)
    x2, w2 = sinhtrans(u, w, aη, b; iterated=iterated)
    N = n * n
    ξ = Vector{Float64}(undef, N)
    η = Vector{Float64}(undef, N)
    ww = Vector{Float64}(undef, N)
    k = 1
    @inbounds for j in 1:n, i in 1:n
        ξ[k] = x1[i]; η[k] = x2[j]; ww[k] = w1[i] * w2[j]; k += 1
    end
    return ξ, η, ww
end

"""
    nearfield_2d(aξ, aη, b; n=nothing, mode=:auto) -> (ξ, η, w)

Flattened surface near-field rule on [-1,1]².
`mode = :auto | :tensor | :polar`
"""
function nearfield_2d(aξ, aη, b; n=nothing, mode::Symbol=:auto)
    b = max(float(b), 1e-14)
    nn = something(n, nquad_nearfield(b; nmin=6, nmax=32))
    use_polar = mode === :polar || (mode === :auto && b < 5e-2)
    if use_polar
        m = max(4, cld(nn, 2))
        return polar_sinh_2d(m, m, float(aξ), float(aη), b)
    else
        return _tensor_sinh_2d(nn, float(aξ), float(aη), b; iterated=(b < 1e-4))
    end
end

# =============================================================================
# Element transforms (public API used by assembly)
# =============================================================================

"""
    transform(dad, elem, pf::Point2D) -> (η, w)

2D curve near-field: closest-point + adaptive sinh (compose if b < 1e-4).
"""
function transform(dad, elem, pf::Point2D)
    nodes = element_geometry(dad, elem)
    poly = dad.element_type
    # geometry polynomial must match length(nodes) (Bernstein for IGA controls)
    if length(nodes) != length(poly.nodes)
        poly = poly isa Bernstein ? Bernstein(length(nodes) - 1) : Legendre(length(nodes) - 1)
    end
    Δelem = nodes[end] - nodes[1]
    ξs = poly.nodes
    ξ0 = (ξs[end] - ξs[1]) * dot(Δelem, pf - nodes[1]) / (norm(Δelem)^2 + eps()) + ξs[1]
    eet, _, dist = closest_point_1d(poly, nodes, pf; ξ0=clamp(ξ0, -1.0, 1.0))
    b = dist / max(elem.Length, eps())
    nbase = length(dad.qsi)
    n = nquad_nearfield(b; nmin=nbase, nbase=nbase)
    # reuse dad rule when n unchanged; else denser GL
    if n == nbase
        return nearfield_1d(eet, b; qsi=dad.qsi, w=dad.w)
    else
        return nearfield_1d(eet, b; n=n)
    end
end

"""
    transform(dad, qsi2, elem, pf::Point3D) -> (η1, η2, J1, J2)

Legacy tensor-sinh API (compatible with older 3D loops).
For strong near-field prefer `transform_surface` / updated `integraelem`.
"""
function transform(dad, qsi2, elem, pf::Point3D)
    nodes = dad.Nodes[elem.index]
    poly = dad.element_type
    ξs = poly.nodes
    nξ = length(ξs)
    Δ1 = nodes[min(2, end)] - nodes[1]
    Δ2 = nodes[min(end, nξ)] - nodes[1]
    eet1 = clamp((ξs[min(2, end)] - ξs[1]) * dot(Δ1, pf - nodes[1]) / (norm(Δ1)^2 + eps()) + ξs[1], -1.0, 1.0)
    eet2 = clamp((ξs[end] - ξs[1]) * dot(Δ2, pf - nodes[1]) / (norm(Δ2)^2 + eps()) + ξs[1], -1.0, 1.0)
    ξ, η, _, dist = closest_point_2d(poly, nodes, pf; ξ0=(eet1, eet2))
    b = dist / max(elem.Length, eps())
    n = nquad_nearfield(b; nmin=length(qsi2), nbase=length(qsi2), nmax=48)
    u = n == length(qsi2) ? qsi2 : first(gausslegendre(n))
    iterated = b < 1e-4
    eta1, J1 = sinhtrans_J(u, ξ, b; iterated=iterated)
    eta2, J2 = sinhtrans_J(u, η, b; iterated=iterated)
    return eta1, eta2, J1, J2
end

"""
    transform_surface(dad, elem, pf::Point3D; qsi2=nothing) -> (ξ, η, w)

Flattened surface rule: polar+sinh when `b` small, else tensor sinh.
"""
function transform_surface(dad, elem, pf::Point3D; qsi2=nothing)
    nodes = dad.Nodes[elem.index]
    poly = dad.element_type
    ξs = poly.nodes
    nξ = length(ξs)
    Δ1 = nodes[min(2, end)] - nodes[1]
    Δ2 = nodes[min(end, nξ)] - nodes[1]
    eet1 = clamp((ξs[min(2, end)] - ξs[1]) * dot(Δ1, pf - nodes[1]) / (norm(Δ1)^2 + eps()) + ξs[1], -1.0, 1.0)
    eet2 = clamp((ξs[end] - ξs[1]) * dot(Δ2, pf - nodes[1]) / (norm(Δ2)^2 + eps()) + ξs[1], -1.0, 1.0)
    aξ, aη, _, dist = closest_point_2d(poly, nodes, pf; ξ0=(eet1, eet2))
    b = dist / max(elem.Length, eps())
    n0 = qsi2 === nothing ? 8 : length(qsi2)
    n = nquad_nearfield(b; nmin=n0, nbase=n0, nmax=32)
    return nearfield_2d(aξ, aη, b; n=n, mode=:auto)
end

export transform_surface

function divide(qsi, w, a)
    if -1 < a < 1
        J1 = (1 - a) / 2
        J2 = (1 + a) / 2
        qsi1 = qsi * J1 .+ (a + 1) / 2
        qsi2 = qsi * J2 .+ (a - 1) / 2
        w1 = w * J1
        w2 = w * J2
        return SVector{2 * length(qsi)}([qsi2; qsi1]), SVector{2 * length(w)}([w2; w1])
    end
    return SVector{length(qsi)}(qsi), SVector{length(w)}(w)
end

# =============================================================================
# On-element modified weights (log / CPV / HFP)
# =============================================================================

"""
    singular(qsi, w, order=0, eet=0.0; poly=nothing)

Modified weights for on-element singular integrals with Gauss-node Lagrange
shapes. Prefer Dumont when available; this is the on-element fallback.

| order | K                |
|------:|------------------|
| 0     | log\\|ξ-a\\|       |
| 1     | 1/(ξ-a) CPV      |
| 2     | 1/(ξ-a)² HFP     |
"""
function singular(qsi, w, order::Integer=0, eet::Real=0.0; poly=nothing)
    ngp = length(qsi)
    T = float(eltype(w))
    p = poly === nothing ? Legendre(ngp - 1) : poly
    nN = length(p.nodes)
    nN == ngp || throw(ArgumentError(
        "singular: poly must have $ngp nodes (got $nN) so Nⱼ are cardinal at qsi; " *
        "default Legendre(ngp-1) satisfies this by design"))

    a = T(eet)
    if a <= -1 || a >= 1
        a = clamp(a, nextfloat(T(-1)), prevfloat(T(1)))
    end

    N_a, dN_a = shapefun(p, a)
    N_a = vec(N_a)
    dN_a = vec(dN_a)

    q2, w2 = divide(qsi, w, a)
    N2, _ = shapefun(p, q2)
    n2 = length(q2)
    I = zeros(T, nN)

    if order == 0
        cte = (1 - a) * log(abs(1 - a)) + (1 + a) * log(abs(1 + a)) - T(2)
        @inbounds for j in 1:nN
            s = zero(T)
            for i in 1:n2
                δ = abs(q2[i] - a)
                δ < eps(T) && continue
                s += w2[i] * (N2[i, j] - N_a[j]) * log(δ)
            end
            I[j] = s + N_a[j] * cte
        end
        wn = similar(I)
        @inbounds for j in 1:nN
            Kj = log(abs(qsi[j] - a))
            wn[j] = abs(Kj) > eps(T) ? I[j] / Kj : zero(T)
        end
        return wn

    elseif order == 1
        cte = log(abs((1 - a) / (1 + a)))
        @inbounds for j in 1:nN
            s = zero(T)
            for i in 1:n2
                δ = q2[i] - a
                abs(δ) < eps(T) && continue
                s += w2[i] * (N2[i, j] - N_a[j]) / δ
            end
            I[j] = s + N_a[j] * cte
        end
        wn = similar(I)
        @inbounds for j in 1:nN
            wn[j] = I[j] * (qsi[j] - a)
        end
        return wn

    elseif order == 2
        cte = -T(2) / (1 - a^2)
        ctel = log(abs((1 - a) / (1 + a)))
        @inbounds for j in 1:nN
            s = zero(T)
            for i in 1:n2
                δ = q2[i] - a
                abs(δ) < eps(T) && continue
                s += w2[i] * (N2[i, j] - N_a[j] - dN_a[j] * δ) / δ^2
            end
            I[j] = s + N_a[j] * cte + dN_a[j] * ctel
        end
        wn = similar(I)
        @inbounds for j in 1:nN
            wn[j] = I[j] * (qsi[j] - a)^2
        end
        return wn
    else
        throw(ArgumentError("singular: order must be 0, 1, or 2 (got $order)"))
    end
end

singular(qsi, w, f, order::Integer, eet::Real; poly=nothing) =
    singular(qsi, w, order, eet; poly=poly)

# =============================================================================
# Shape helpers for flattened surface quadrature
# =============================================================================

"""Evaluate 2D Lagrange basis at scattered (ξ[k], η[k]) — not a tensor grid."""
function shapefun2D_points(poly, ξ::AbstractVector, η::AbstractVector)
    n = length(ξ)
    n == length(η) || throw(DimensionMismatch("ξ and η length"))
    nN = length(poly.nodes)^2
    N = zeros(n, nN)
    dNξ = zeros(n, nN)
    dNη = zeros(n, nN)
    @inbounds for k in 1:n
        Lk, Lx, Ly = shapefun2D(poly, poly, ξ[k], η[k])
        N[k, :] .= vec(Lk)
        dNξ[k, :] .= vec(Lx)
        dNη[k, :] .= vec(Ly)
    end
    return N, dNξ, dNη
end
