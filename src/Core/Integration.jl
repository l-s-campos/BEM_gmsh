# Singular / nearly-singular integration
# -----------------------------------------------------------------------------
# 2D curve:  far → plain GL (caller) | nearly → sinh | on-element → Guiggiani
# 3D surface: far → tensor GL | nearly → polar/tensor sinh
#             on-element → polar Guiggiani (radial Laurent subtraction)
# -----------------------------------------------------------------------------

export closest_point_1d, closest_point_2d
export transform, transform_surface, nearfield_1d, default_nearfield
export laurent_coefficients, laurent_coefficients_interp
export guiggiani_integral, guiggiani_GH, guiggiani_GH_surface, sst_GH

using StaticArrays: @SVector, @SMatrix, SVector
using FastGaussQuadrature: gausslegendre
using LinearAlgebra: norm, dot, I, cross
try
    using Richardson: extrapolate
catch
    extrapolate(args...; kwargs...) = error("Richardson.jl is not in this environment")
end

# =============================================================================
# Public: element near-field transforms (assembly)
# =============================================================================

"""
    nearfield_1d(a, b; qsi, w)

1D near-field rule on [-1,1]: sinh map of the supplied GL nodes about peak `a`
with width `b = d/L`.
"""
function nearfield_1d(a, b; qsi, w)
    return _sinhtrans(qsi, w, float(a), max(float(b), 1e-14))
end

"""
    transform(dad, elem, nodes, pf::Point2D; poly=dad.element_type) -> (η, w)

2D curve near-field: closest-point peak + sinh on `dad.qsi` / `dad.w`.

`dad.nearfield` selects the off-element map (default
[`default_nearfield`](@ref): `:tanp3c` isotropic, `:zsinh` anisotropic).

Granados & Gallego, EABE 189 (2026) eqs. (40)–(41) are piecewise in the
complex pole ``ξ_0=ζ_0+iη_0`` of ``z(ξ)=x_1+ix_2``:

- **eq. (40)** natural for ``1/r``: sinh if ``η_0≠0``; exponential
  ``ξ=ζ_0+\\mathrm{sign}(ξ-ζ_0)\\,\\exp((ξ̃-B)/A)`` if ``η_0=0`` and ``|ζ_0|>1``
  (collinear real pole).
- **eq. (41)** natural for ``1/r^2``: tangent if ``η_0≠0``; Möbius
  ``ξ=ζ_0-A/(ξ̃-B)`` if ``η_0=0`` and ``|ζ_0|>1``.

- `:plain` — uniform Gauss
- `:euclid` — sinh about the Euclidean closest point (`b=d/L`)
- `:csinh` — eq. (40)
- `:sinhsinh` — iterated sinh–sinh on that pole when ``η_0≠0``; eq. (40)
  exponential when ``η_0=0``
- `:tangent` — eq. (41)
- `:p3c` — complete cubic of Granados & Gallego, JACM 12 (2026)
- `:tanp3c` — tangent then p3c on the tangent ``±π/2`` poles (EABE 189 §4.1);
  Möbius only when ``η_0=0`` (no ``±π/2`` poles)
- `:zsinh` — sinh at min ``|z_k|=|r_1+μ_k r_2|`` for `AnisotropicElasticity`
"""
default_nearfield(::Problem) = :tanp3c
default_nearfield(::AnisotropicElasticity) = :zsinh
default_nearfield(::AbstractThinPlate) = :euclid

function transform(dad, elem, nodes, pf::Point2D; poly=dad.element_type)
    nf = has_cache(dad, :nearfield) ? dad.nearfield : default_nearfield(dad.properties)
    nf === :plain && return dad.qsi, dad.w
    qsi, w = dad.qsi, dad.w
    if nf === :csinh || nf === :sinhsinh || nf === :tangent ||
            nf === :p3c || nf === :tanp3c
        ζ0, η0 = _complex_pole_1d(poly, nodes, pf)
        nf === :tangent && return _tangenttrans(qsi, w, ζ0, η0)
        nf === :p3c && return _p3ctrans(qsi, w, ζ0, η0)
        nf === :tanp3c && return _tanp3ctrans(qsi, w, ζ0, η0)
        niter = nf === :sinhsinh ? 2 : 1
        return _sinhtrans_iterated(qsi, w, ζ0, η0; niter=niter)
    end
    if nf === :zsinh && dad.properties isa AnisotropicElasticity
        ξz, zmin, bz = _aniso_z_peak(poly, nodes, pf, dad.properties.params.mi)
        a, _, dist = closest_point_1d(poly, nodes, pf; ξ0=_seed_1d(poly, nodes, pf))
        if zmin ≤ dist
            return nearfield_1d(ξz, max(bz, 1e-12); qsi=qsi, w=w)
        end
        b = dist / max(elem.Length, eps())
        return nearfield_1d(a, b; qsi=qsi, w=w)
    end
    a, _, dist = closest_point_1d(poly, nodes, pf; ξ0=_seed_1d(poly, nodes, pf))
    b = dist / max(elem.Length, eps())
    return nearfield_1d(a, b; qsi=qsi, w=w)
end

"""Parent ``ξ`` and width where ``min_k|z_k|`` is smallest on the element."""
function _aniso_z_peak(poly, nodes, pf, μ)
    nN = length(nodes)
    function at(ξ)
        N, dN = shapefun(poly, ξ)
        pg = zero(eltype(nodes))
        dx = zero(eltype(nodes))
        @inbounds for k in 1:nN
            pg += N[1, k] * nodes[k]
            dx += dN[1, k] * nodes[k]
        end
        r = pg - pf
        d1 = abs(r[1] + μ[1] * r[2])
        d2 = abs(r[1] + μ[2] * r[2])
        kmin = d1 < d2 ? 1 : 2
        return min(d1, d2), kmin, dx
    end
    ξb = 0.0
    db = Inf
    kb = 1
    dxb = zero(eltype(nodes))
    @inbounds for ξ in range(-1.0, 1.0; length=81)
        d, k, dx = at(ξ)
        if d < db
            db = d
            ξb = ξ
            kb = k
            dxb = dx
        end
    end
    h = 2 / 80
    lo = max(-1.0, ξb - h)
    hi = min(1.0, ξb + h)
    for _ in 1:18
        m1 = (2lo + hi) / 3
        m2 = (lo + 2hi) / 3
        d1, _, _ = at(m1)
        d2, _, _ = at(m2)
        if d1 < d2
            hi = m2
        else
            lo = m1
        end
    end
    ξb = (lo + hi) / 2
    db, kb, dxb = at(ξb)
    J = norm(dxb)
    t1 = dxb[1] / max(J, 1e-30)
    t2 = dxb[2] / max(J, 1e-30)
    dz = abs(t1 + μ[kb] * t2) * J
    b = db / max(dz, 1e-30)
    return ξb, db, b
end

"""Sinh on the subinterval that contains `a`; plain GL on the others."""
function _subdiv_sinh(a, b; nsub::Int, qsi, w)
    nsub <= 1 && return nearfield_1d(a, b; qsi=qsi, w=w)
    T = float(eltype(w))
    h = T(2) / nsub
    η = T[]
    ww = T[]
    sizehint!(η, nsub * length(qsi))
    sizehint!(ww, nsub * length(w))
    @inbounds for s in 0:(nsub - 1)
        lo = T(-1) + s * h
        hi = lo + h
        if lo - 10 * eps(T) <= a <= hi + 10 * eps(T)
            a_sub = clamp((a - lo) * (2 / h) - 1, nextfloat(T(-1)), prevfloat(T(1)))
            b_sub = max(b * (2 / h), T(1e-12))
            xs, ws = nearfield_1d(a_sub, b_sub; qsi=qsi, w=w)
            append!(η, (xs .+ 1) .* (h / 2) .+ lo)
            append!(ww, ws .* (h / 2))
        else
            append!(η, (qsi .+ 1) .* (h / 2) .+ lo)
            append!(ww, w .* (h / 2))
        end
    end
    return η, ww
end

"""
    transform_surface(dad, elem, pf::Point3D; qsi2=nothing) -> (ξ, η, w)

Flattened surface rule. `dad.nearfield` selects the 3-D map:

- `:tanp3c` / `:tangent` — polar about the closest point, Granados
  eq. (41) on each radial ray (pole at ``ρ=0``, parent ``ζ_0=-1``)
- `:polar` / `:auto` — polar + sinh radial (``:auto`` uses tensor sinh
  when ``b=d/L\\ge 0.05``)
- `:tensor` — tensor-product sinh
- `:plain` — uniform tensor Gauss
- `:dibem` — nearly-singular faces (`d/L < 0.05`): PHS3+poly of the
  full integrand at vertices+edge Gauss, spike recovered by analytic
  radial ``ID``. Far faces fall through to `:auto` (tensor sinh).
"""
function transform_surface(dad, elem, pf::Point3D; qsi2=nothing)
    nodes = dad.Nodes[elem.index]
    poly = dad.element_type
    aξ0, aη0 = _seed_2d(poly, nodes, pf)
    aξ, aη, _, dist = closest_point_2d(poly, nodes, pf; ξ0=(aξ0, aη0))
    b = dist / max(elem.Length, eps())
    n = qsi2 === nothing ? (has_cache(dad, :qsi) ? length(dad.qsi) : 8) : length(qsi2)
    return _nearfield_2d(aξ, aη, b; n=n, mode=_surface_mode(dad))
end

function _surface_mode(dad)
    nf = has_cache(dad, :nearfield) ? dad.nearfield : default_nearfield(dad.properties)
    (nf === :plain || nf === :polar || nf === :tensor ||
        nf === :tanp3c || nf === :tangent || nf === :dibem) && return nf
    return :auto
end

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

# =============================================================================
# Cheap distance to an element (no Newton)
# =============================================================================

@inline function _minmax_nodes(xg)
    lo = hi = xg[1]
    @inbounds for k in 2:length(xg)
        q = xg[k]
        lo = min.(lo, q)
        hi = max.(hi, q)
    end
    return lo, hi
end

@inline function _dist_aabb(p, lo, hi)
    d2 = zero(eltype(p))
    @inbounds for a in eachindex(p)
        v = p[a]
        if v < lo[a]
            Δ = lo[a] - v
            d2 += Δ * Δ
        elseif v > hi[a]
            Δ = v - hi[a]
            d2 += Δ * Δ
        end
    end
    return sqrt(d2)
end

@inline function _dist_chord(p, a, b)
    ab = b - a
    t = clamp(dot(p - a, ab) / (dot(ab, ab) + eps(Float64)), 0.0, 1.0)
    return norm(p - (a + t * ab))
end

"""Closest-point distance to triangle `a,b,c` (Ericson)."""
function _dist_triangle(p, a, b, c)
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = dot(ab, ap)
    d2 = dot(ac, ap)
    d1 <= 0 && d2 <= 0 && return norm(p - a)
    bp = p - b
    d3 = dot(ab, bp)
    d4 = dot(ac, bp)
    d3 >= 0 && d4 <= d3 && return norm(p - b)
    vc = d1 * d4 - d3 * d2
    if vc <= 0 && d1 >= 0 && d3 <= 0
        v = d1 / (d1 - d3)
        return norm(p - (a + v * ab))
    end
    cp = p - c
    d5 = dot(ab, cp)
    d6 = dot(ac, cp)
    d6 >= 0 && d5 <= d6 && return norm(p - c)
    vb = d5 * d2 - d1 * d6
    if vb <= 0 && d2 >= 0 && d6 <= 0
        w = d2 / (d2 - d6)
        return norm(p - (a + w * ac))
    end
    va = d3 * d6 - d5 * d4
    if va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return norm(p - (b + w * (c - b)))
    end
    denom = 1 / (va + vb + vc)
    return norm(p - (a + ab * (vb * denom) + ac * (vc * denom)))
end

"""Distance to the piecewise-linear interpolant through `xg` (exact for linear elements)."""
function _dist_element(pf, xg)
    n = length(xg)
    n == 0 && return Inf
    n == 1 && return norm(pf - xg[1])
    n == 2 && return _dist_chord(pf, xg[1], xg[2])
    if length(pf) == 2
        d = Inf
        @inbounds for k in 1:n-1
            d = min(d, _dist_chord(pf, xg[k], xg[k+1]))
        end
        return d
    end
    n == 3 && return _dist_triangle(pf, xg[1], xg[2], xg[3])
    # Linear or quadratic quad: Gmsh corners are index 1:4. Two triangles
    # of the corners (planar interpolant). Do not fan through mid-edge nodes.
    return min(_dist_triangle(pf, xg[1], xg[2], xg[3]),
               _dist_triangle(pf, xg[1], xg[3], xg[4]))
end

# =============================================================================
# Public: closest-point projection
# =============================================================================

"""
    closest_point_1d(poly, nodes, pf; ξ0=0, maxiter=25, tol=1e-14) -> (ξ*, x*, dist)

Newton on (x-pf)·x' = 0 with analytic Jacobian. Clamps to endpoints if needed.
"""
function closest_point_1d(poly, nodes::AbstractVector{<:Point}, pf::Point;
        ξ0=0.0, maxiter=25, tol=1e-14)
    ξ = float(ξ0)
    xξ = nodes[1]
    @inbounds for _ in 1:maxiter
        res, J, xξ = _proj1d_rj(poly, nodes, pf, ξ)
        abs(J) < 1e-30 && break
        Δ = res / J
        ξ -= Δ
        abs(Δ) < tol && break
    end
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

"""
    closest_point_2d(poly, nodes, pf; ξ0=(0,0), ...) -> (ξ, η, x*, dist)

Newton on [(x-pf)·x_ξ, (x-pf)·x_η] = 0 with analytic Jacobian.
"""
function closest_point_2d(poly, nodes::AbstractVector{<:Point}, pf::Point;
        ξ0=(0.0, 0.0), maxiter=25, tol=1e-14)
    ξ = float(ξ0[1])
    η = float(ξ0[2])
    x = nodes[1]
    @inbounds for _ in 1:maxiter
        F, J, x = _proj2d_rj(poly, nodes, pf, ξ, η)
        detJ = J[1, 1] * J[2, 2] - J[1, 2] * J[2, 1]
        abs(detJ) < 1e-30 && break
        Δ1 = (J[2, 2] * F[1] - J[1, 2] * F[2]) / detJ
        Δ2 = (-J[2, 1] * F[1] + J[1, 1] * F[2]) / detJ
        ξ -= Δ1
        η -= Δ2
        hypot(Δ1, Δ2) < tol && break
    end
    ξ = clamp(ξ, -1.0, 1.0)
    η = clamp(η, -1.0, 1.0)
    N, _, _ = shapefun2D(poly, poly, ξ, η)
    x = (N * nodes)[1]
    return ξ, η, x, norm(x - pf)
end

# =============================================================================
# Public: Guiggiani on-element integration
# =============================================================================

"""
    guiggiani_integral(f, a, order; qsi, w, h=1e-3, kwargs...) -> I

`I = ∫_{-1}^{1} f(ξ) dξ` with Laurent singularity order
`order ∈ {-4,-3,-2,-1,0}` at `ξ = a`.

Default `laurent=:interp`: sample `f` at `ninterp` Gauss–Legendre nodes
(default 20; nodes/weights/barycentric `wᵢ` cached per `n`), extract
``F_{-k}`` from that interpolant, and Gauss-integrate the remainder on
the same nodes. Analytic CPV/HFP of ``(ξ-a)^{-k}`` is added back. Log
(`order=0`) strips ``F_{-1}`` from ``f(ξ-a)`` first, then least-squares
fits ``f-F_{-1}/(ξ-a)\\approx F_0\\log|ξ-a|+b``.

`laurent=:auto` is closed-form tensors then Richardson on Guiggiani rays;
`:richardson` always extrapolates on rays.
"""
function guiggiani_integral(f, a::Real, order::Integer;
        qsi=nothing, w=nothing,
        h::Real=1e-3, kwargs...)
    method, ninterp, rest = _strip_laurent_kwargs(kwargs)
    o = _normalize_guiggiani_order(order)
    if method === :interp
        return _guiggiani_interp(f, a, o, ninterp; h=h, rest=rest)
    end
    a, qsi, w, hh = _guiggiani_rule(a, qsi, w, h)
    probe = f(a + max(hh, 1e-6))
    acc = zero(probe)
    kw = _laurent_extrapolate_kwargs(rest)

    for (s, ρmax) in _guiggiani_rays(a)
        ρmax ≤ eps(typeof(a)) && continue
        Fr = let f = f, a = a, s = s
            ρ -> f(a + s * ρ)
        end
        C = _laurent_coeffs_Fr(Fr, hh, o, probe, kw; rest...)
        Iρ = zero(probe)
        @inbounds for i in eachindex(qsi)
            ρ = (qsi[i] + 1) * ρmax / 2
            ρ ≤ eps(typeof(a)) && continue
            dρ = w[i] * ρmax / 2
            Iρ += _guiggiani_reg(Fr(ρ), ρ, o, C) * dρ
        end
        acc += Iρ + _guiggiani_analytic(ρmax, o, C)
    end
    return acc
end

"""
    guiggiani_GH(f, a; order_G=0, order_H=-1, qsi, w, h=1e-3, kwargs...) -> (I_G, I_H)

Fused on-element pair `f(ξ) → (F_G, F_H)`. Default `laurent=:interp` samples
once at `ninterp` Gauss–Legendre nodes, extracts every ``F_{-k}`` from that
interpolant, and Gauss-integrates the remainder on the same nodes.
`laurent=:auto` uses closed-form / SST tensors then Richardson on Guiggiani
rays; `:richardson` always extrapolates on rays. Log (`order=0`) strips
the 1/ρ term first, then interpolates the remainder over ``\\log|ξ-a|``.
"""
function guiggiani_GH(f, a::Real;
        order_G::Integer=0,
        order_H::Integer=-1,
        qsi=nothing, w=nothing,
        h::Real=1e-3,
        props=nothing, poly=nothing, nodes=nothing, nf=nothing, kwargs...)
    method, ninterp, rest = _strip_laurent_kwargs(kwargs)
    og = _normalize_guiggiani_order(order_G)
    oh = _normalize_guiggiani_order(order_H)
    geom = _guiggiani_geom(props, poly, nodes)
    if method === :interp
        return _guiggiani_interp_pair(f, a, og, oh, ninterp; h=h, rest=rest,
            geom=geom, nf=nf)
    end
    a, qsi, w, hh = _guiggiani_rule(a, qsi, w, h)
    probe_g, probe_h = f(a + max(hh, 1e-6))
    acc_g = zero(probe_g)
    acc_h = zero(probe_h)
    kw = _laurent_extrapolate_kwargs(rest)
    geom = method === :richardson ? nothing : geom

    for (s, ρmax) in _guiggiani_rays(a)
        ρmax ≤ eps(typeof(a)) && continue
        Fr = let f = f, a = a, s = s
            ρ -> f(a + s * ρ)
        end
        Cg = _laurent_pair_coeffs(Fr, hh, og, :G, probe_g, kw, geom, a, s; nf=nf, rest...)
        Ch = _laurent_pair_coeffs(Fr, hh, oh, :H, probe_h, kw, geom, a, s; nf=nf, rest...)
        Ig = zero(probe_g)
        Ih = zero(probe_h)
        @inbounds for i in eachindex(qsi)
            ρ = (qsi[i] + 1) * ρmax / 2
            ρ ≤ eps(typeof(a)) && continue
            dρ = w[i] * ρmax / 2
            Fg, Fh = Fr(ρ)
            Ig += _guiggiani_reg(Fg, ρ, og, Cg) * dρ
            Ih += _guiggiani_reg(Fh, ρ, oh, Ch) * dρ
        end
        acc_g += Ig + _guiggiani_analytic(ρmax, og, Cg)
        acc_h += Ih + _guiggiani_analytic(ρmax, oh, Ch)
    end
    return acc_g, acc_h
end

"""
    sst_GH(f, a; qsi, w, props, poly, nodes) -> (I_G, I_H)

Cordeiro & Leonel (2020) SST **factors** (`_aniso_sst_star`) as Guiggiani
Laurent tensors, orders `(-1, -2)`. Same integral as [`guiggiani_GH`](@ref)
with those tensors: ``ξ-a = s ρ``.
"""
function sst_GH(f, a::Real; qsi=nothing, w=nothing, h::Real=1e-3,
        props, poly, nodes, nf=nothing, kwargs...)
    _, ninterp, rest = _strip_laurent_kwargs(kwargs)
    return guiggiani_GH(f, a; order_G=-1, order_H=-2, qsi=qsi, w=w, h=h,
        props=props, poly=poly, nodes=nodes, nf=nf, laurent=:auto,
        ninterp=ninterp, rest...)
end

"""
    laurent_coefficients(f, h, order; kwargs...) -> (f₋₂, f₋₁, f₀)
    laurent_coefficients(props, poly, nodes, collocation, dir, which, order)

Leading Laurent coefficients of `f(ρ) = f₋₂/ρ² + f₋₁/ρ + f₀ + O(ρ)` as `ρ→0⁺`.
`order`: `-2` hypersingular, `-1` CPV, `0` log (2-D) / bounded polar (3-D).
`which` is `:G` (single-layer) or `:H` (double-layer).

The geometry method returns a closed form for Laplace / Kelvin CBIE
(Marczak 2002; Guiggiani 1998; CILAMCE 2016 cap-iso), 2-D Helmholtz
(Laplace ``k=1`` leading terms, ``H`` flipped), and 2-D Lekhnitskii
anisotropic CBIE **and** HBIE (Cordeiro & Leonel 2020). Returns `nothing`
when no formula applies (3-D Helmholtz, isotropic hypersingular, degenerate
map); the integrand method then uses Richardson.

2-D Kelvin / Laplace HBIE have a closed-form ``F_{-2}`` (the
``ρ^2 T^h φ J`` limit with frozen ``J,n``): Laplace
``-φ/(2π J)``; Kelvin ``\\dfrac{\\mu}{2π(1-ν)} I\\,φ/J``. That is the
term Guiggiani must subtract. Richardson of ``ρ^2 f`` at a fixed parent
``h`` mixes ``F_{-1}ρ`` into ``F_{-2}`` on long or curved elements;
the geometry form is the exact leading coefficient.
"""
function laurent_coefficients(f, h, order::Integer; kwargs...)
    return laurent_coefficients(f, h, Val(Int(_normalize_guiggiani_order(order))); kwargs...)
end

function laurent_coefficients(f, h, ::Val{-4}; kwargs...)
    kw = _laurent_extrapolate_kwargs(kwargs)
    f₋₄, _ = extrapolate(h; x0=zero(h), kw...) do x
        return x^4 * f(x)
    end
    f₋₃, _ = extrapolate(h; x0=zero(h), kw...) do x
        return x^3 * f(x) - f₋₄ / x
    end
    f₋₂, _ = extrapolate(h; x0=zero(h), kw...) do x
        return x^2 * f(x) - f₋₄ / x^2 - f₋₃ / x
    end
    f₋₁, _ = extrapolate(h; x0=zero(h), kw...) do x
        return x * f(x) - f₋₄ / x^3 - f₋₃ / x^2 - f₋₂ / x
    end
    f₀, _ = extrapolate(h; x0=zero(h), kw...) do x
        return f(x) - f₋₄ / x^4 - f₋₃ / x^3 - f₋₂ / x^2 - f₋₁ / x
    end
    return f₋₄, f₋₃, f₋₂, f₋₁, f₀
end

function laurent_coefficients(f, h, ::Val{-3}; kwargs...)
    kw = _laurent_extrapolate_kwargs(kwargs)
    f₋₃, _ = extrapolate(h; x0=zero(h), kw...) do x
        return x^3 * f(x)
    end
    z = zero(f₋₃)
    f₋₂, _ = extrapolate(h; x0=zero(h), kw...) do x
        return x^2 * f(x) - f₋₃ / x
    end
    f₋₁, _ = extrapolate(h; x0=zero(h), kw...) do x
        return x * f(x) - f₋₃ / x^2 - f₋₂ / x
    end
    f₀, _ = extrapolate(h; x0=zero(h), kw...) do x
        return f(x) - f₋₃ / x^3 - f₋₂ / x^2 - f₋₁ / x
    end
    return z, f₋₃, f₋₂, f₋₁, f₀
end

function laurent_coefficients(f, h, ::Val{-2}; kwargs...)
    kw = _laurent_extrapolate_kwargs(kwargs)
    g = x -> x^2 * f(x)
    f₋₂, _ = extrapolate(h; x0=zero(h), kw...) do x
        return g(x)
    end
    f₋₁, _ = extrapolate(h; x0=zero(h), kw...) do x
        return x * f(x) - f₋₂ / x
    end
    f₀, _ = extrapolate(h; x0=zero(h), kw...) do x
        return f(x) - f₋₂ / x^2 - f₋₁ / x
    end
    return f₋₂, f₋₁, f₀
end

function laurent_coefficients(f, h, ::Val{-1}; kwargs...)
    kw = _laurent_extrapolate_kwargs(kwargs)
    f₋₁, _ = extrapolate(h; x0=zero(h), kw...) do x
        return x * f(x)
    end
    f₀, _ = extrapolate(h; x0=zero(h), kw...) do x
        return f(x) - f₋₁ / x
    end
    z = zero(f₀)
    return z, f₋₁, f₀
end

function laurent_coefficients(f, h, ::Val{0}; kwargs...)
    kw = _laurent_extrapolate_kwargs(kwargs)
    f₀, _ = extrapolate(h; x0=zero(h), kw...) do x
        return f(x)
    end
    z = zero(f₀)
    return z, z, f₀
end

function laurent_coefficients(f, h, ::Val{N}; kwargs...) where {N}
    throw(ArgumentError(
        "laurent_coefficients: order must be in {-4,-3,-2,-1,0} (got $N)"))
end

"""
    laurent_coefficients_interp(f, a, order; n=20) -> (F₋₂, F₋₁, F₀) or 5-tuple

Laurent coefficients of parent-coordinate `f(ξ)` about `ξ = a`, from a
barycentric interpolant of the regularized integrand at `n` Gauss–Legendre
nodes on `[-1,1]`. Same sequential cascade as Richardson, but global:

```
f(ξ)(ξ-a)ᵖ = Σᵢ Nᵢ(ξ) [f(ξᵢ)(ξᵢ-a)ᵖ]
F₋ₚ = Σᵢ Nᵢ(a) f(ξᵢ)(ξᵢ-a)ᵖ
F₋ₚ₊₁ = Σᵢ Nᵢ(a) (f(ξᵢ) - F₋ₚ/(ξᵢ-a)ᵖ) (ξᵢ-a)ᵖ⁻¹
…
```

`Nᵢ` are Lagrange basis functions on the Gauss nodes (not the element
shape functions). `order ∈ {-4,-3,-2,-1,0}`. Log first subtracts the 1/ρ term
``F_{-1}=\\sum_i N_i(a)\\,f(ξ_i)(ξ_i-a)``, then least-squares fits
``f-F_{-1}/(ξ-a)\\approx F_0\\log|ξ-a|+b`` on the sample nodes.
"""
function laurent_coefficients_interp(f, a::Real, order::Integer; n::Integer=20)
    o = _normalize_guiggiani_order(order)
    n >= 2 || throw(ArgumentError("ninterp must be ≥ 2 (got $n)"))
    ξ, Ni, δ = _interp_nodes(a, n)
    fi = [f(ξi) for ξi in ξ]
    return o == 0 ? _interp_log_coeff(fi, Ni, δ) : _interp_cascade(fi, Ni, δ, o)
end

function laurent_coefficients_interp(f, a::Real, ::Val{O}; n::Integer=20) where {O}
    return laurent_coefficients_interp(f, a, Int(O); n=n)
end

# Geometry form: analytic if a formula exists, else `nothing` (caller extrapolates).
laurent_coefficients(::Problem, poly, nodes, collocation, dir, ::Symbol, ::Integer) = nothing

function laurent_coefficients(props::Laplace, poly, nodes, a::Real, s::Real,
        which::Symbol, order::Integer)
    order = _normalize_guiggiani_order(order)
    g = _geom_1d(poly, nodes, a)
    g === nothing && return nothing
    Nrow, J, _, _ = g
    nN = length(Nrow)
    if which === :G && order == 0
        cU = -J / (2 * π * float(props.k))
        F0 = [cU * Nrow[j] for j in 1:nN]
        z = zero(F0)
        return (z, z, F0)
    elseif which === :H && order == -1
        z = zeros(nN)
        return (z, z, z)   # T regular: F_{-1}=F_{-2}=0
    elseif which === :G && order == -1
        # ∂U/∂nξ ~ (e·nξ)/(2π k R) → 0 on the tangent (e ⟂ nξ)
        z = zeros(nN)
        return (z, z, z)
    elseif which === :H && order == -2
        # ∂T/∂nξ → −1/(2π R²) on the element; F = H_hyper φ J
        _, dN = shapefun(poly, a)
        dNrow = view(dN, 1, :)
        s = float(s)
        c = -1 / (2 * π * J)
        Fm2 = [c * Nrow[j] for j in 1:nN]
        Fm1 = [c * s * dNrow[j] for j in 1:nN]
        z = zero(Fm2)
        return (Fm2, Fm1, z)
    end
    return nothing
end

function laurent_coefficients(props::Helmholtz, poly, nodes, a::Real, s::Real,
        which::Symbol, order::Integer)
    order = _normalize_guiggiani_order(order)
    g = _geom_1d(poly, nodes, a)
    g === nothing && return nothing
    Nrow, J, _, _ = g
    nN = length(Nrow)
    z = zeros(ComplexF64, nN)
    # Leading terms match Laplace k=1 with H flipped (acoustic q=∂u/∂n).
    if which === :G && order == 0
        cU = -J / (2 * π)
        F0 = ComplexF64[cU * Nrow[j] for j in 1:nN]
        return (z, z, F0)
    elseif which === :H && order == -1
        return (z, z, z)
    elseif which === :G && order == -1
        return (z, z, z)
    elseif which === :H && order == -2
        _, dN = shapefun(poly, a)
        dNrow = view(dN, 1, :)
        s = float(s)
        c = 1 / (2 * π * J)
        Fm2 = ComplexF64[c * Nrow[j] for j in 1:nN]
        Fm1 = ComplexF64[c * s * dNrow[j] for j in 1:nN]
        return (Fm2, Fm1, z)
    end
    return nothing
end

function laurent_coefficients(props::Elasticity, poly, nodes, a::Real, s::Real,
        which::Symbol, order::Integer)
    order = _normalize_guiggiani_order(order)
    g = _geom_1d(poly, nodes, a)
    g === nothing && return nothing
    Nrow, J, t, n = g
    nN = length(Nrow)
    ν = float(effective_nu(props))
    den = 4 * π * (1 - ν)
    if which === :G && order == 0
        cU = -(3 - 4ν) * J / (2 * den * float(props.mu))
        Tlog = @SMatrix [cU 0.0; 0.0 cU]
        F0 = zeros(2, 2 * nN)
        _pack_nodal_tensor!(F0, Tlog, Nrow)
        z = zero(F0)
        return (z, z, F0)
    elseif which === :H && order == -1
        # F_{-1}_{αβ} = -((1-2ν)/den) (n_α t_β − n_β t_α) φ ; ray factor s
        cT = -(1 - 2ν) / den
        Tant = @SMatrix [
            0.0  cT*(n[1]*t[2]-n[2]*t[1])
            cT*(n[2]*t[1]-n[1]*t[2])  0.0
        ]
        Fm1 = zeros(2, 2 * nN)
        _pack_nodal_tensor!(Fm1, Tant, Nrow)
        z = zero(Fm1)
        return (z, float(s) * Fm1, z)
    elseif which === :G && order == -1
        Uh, _ = _kelvin_Uh_Th_lead(ν, float(props.mu), t, n)
        Fm1 = zeros(2, 2 * nN)
        _pack_nodal_tensor!(Fm1, Uh, Nrow, 1 / float(s))
        z = zero(Fm1)
        return (z, Fm1, z)
    elseif which === :H && order == -2
        _, Th = _kelvin_Uh_Th_lead(ν, float(props.mu), t, n)
        _, dN = shapefun(poly, a)
        dNrow = view(dN, 1, :)
        s = float(s)
        Fm2 = zeros(2, 2 * nN)
        Fm1 = zeros(2, 2 * nN)
        _pack_nodal_tensor!(Fm2, Th, Nrow, 1 / (s * s * J))
        _pack_nodal_tensor!(Fm1, Th, dNrow, 1 / (s * J))
        z = zero(Fm2)
        return (Fm2, Fm1, z)
    end
    return nothing
end

"""
2-D Kelvin HBIE leading tensors (Guiggiani / frozen geometry).

On the element ``e=s t``, ``e·n=0`` and ``tt^{\\mathsf T}+nn^{\\mathsf T}=I``, so
``T^h=\\dfrac{\\mu}{2π(1-ν)}\\,I/R^2`` and
``U^h=\\dfrac{1-2ν}{4π(1-ν)}\\,s(t\\otimes n-n\\otimes t)/R``.

``U^h=U_\\mathrm{lead}/(sρ J)``, ``T^h=T_\\mathrm{lead}/(s^2 ρ^2 J^2)`` with
``s=\\pm 1`` so ``s^2=1``. ``F_{-2}=T_\\mathrm{lead}\\,φ/J`` is exact at
leading order; it does not involve curvature.
"""
function _kelvin_Uh_Th_lead(ν::Real, μ::Real, t::SVector{2}, n::SVector{2})
    den = 4 * π * (1 - ν)
    cD = (1 - 2ν) / den
    Uh = @SMatrix [
        0.0                  cD*(t[1]*n[2]-n[1]*t[2])
        cD*(t[2]*n[1]-n[2]*t[1])  0.0
    ]
    cS = μ / (2 * π * (1 - ν))
    Th = @SMatrix [cS 0.0; 0.0 cS]
    return Uh, Th
end

"""
Cordeiro & Leonel, *Eng. Anal. Bound. Elem.* 119 (2020) 214–224, eqs. 26–37.

On a 2-D parent curve, ``z_k^*-z_k^0=(ξ-ξ_0)J_0(μ_k n_1-n_2)`` (linear
auxiliary, Fig. 2). `conj(X)'` in `fundamental` is `transpose(X)`.

CBIE (`order` 0 / −1), eqs. 28–32:
- ``U_{ijl}=q_{il}A_{jl}`` so the log tensor is ``2\\operatorname{Re}(A q^{\\mathsf T})``
- ``T_{ijl}=g_{jl}(μ_l n_1-n_2)A_{il}`` so ``F_{-1}=s\\cdot 2\\operatorname{Re}(A g^{\\mathsf T})φ``

HBIE (`order` −1 / −2), eqs. 34–37. Non-singular constants ``D_{ijkm}``,
``S_{ijkm}`` of ``n_ξ·D`` and ``n_ξ·S`` are [`_lekh_Uh_Th_lead`](@ref).
Frozen ``J_0,n`` and ``φ^*=φ_0`` (``U^h``), ``φ^{**}=φ_0+φ_{,ξ}(ξ-ξ_0)``
(``T^h``):
- ``U^h=U_\\mathrm{lead}/((ξ-ξ_0)J_0)``
- ``T^h=T_\\mathrm{lead}/((ξ-ξ_0)^2 J_0^2)``

Guiggiani on-element uses these tensors when `props, poly, nodes` are passed
([`_aniso_sst_star`](@ref) for the traction BIE).
"""
function laurent_coefficients(props::AnisotropicElasticity, poly, nodes, a::Real, s::Real,
        which::Symbol, order::Integer)
    order = _normalize_guiggiani_order(order)
    g = _geom_1d(poly, nodes, a)
    g === nothing && return nothing
    Nrow, J, _, _ = g
    nN = length(Nrow)
    p = props.params
    s = float(s)
    # `conj(X)'` in Fundamental.jl is `transpose(X)` (not `adjoint`).
    if which === :G && order == 0
        Ulog = 2 * real(p.A * transpose(p.q))
        F0 = zeros(2, 2 * nN)
        _pack_nodal_tensor!(F0, Ulog, Nrow, J)
        z = zero(F0)
        return (z, z, F0)
    elseif which === :H && order == -1
        Tant = 2 * real(p.A * transpose(p.g))
        Fm1 = zeros(2, 2 * nN)
        _pack_nodal_tensor!(Fm1, Tant, Nrow)
        z = zero(Fm1)
        return (z, s * Fm1, z)
    elseif which === :G && order == -1
        star = _aniso_sst_star(props, poly, nodes, a)
        star === nothing && return nothing
        Uh, _, Nrow, _, _, _ = star
        Fm1 = zeros(2, 2 * nN)
        _pack_nodal_tensor!(Fm1, Uh, Nrow, 1 / s)
        z = zero(Fm1)
        return (z, Fm1, z)
    elseif which === :H && order == -2
        star = _aniso_sst_star(props, poly, nodes, a)
        star === nothing && return nothing
        _, Th, Nrow, dNrow, J, _ = star
        Fm2 = zeros(2, 2 * nN)
        Fm1 = zeros(2, 2 * nN)
        _pack_nodal_tensor!(Fm2, Th, Nrow, 1 / (s * s * J))
        _pack_nodal_tensor!(Fm1, Th, dNrow, 1 / (s * J))
        z = zero(Fm2)
        return (Fm2, Fm1, z)
    end
    return nothing
end

"""
Paper eqs. 23–24, 34–37. ``R=[1 1; μ_1 μ_2]``, ``α_m=μ_m n_1-n_2``.

Field derivatives (eqs. 23–24) use ``1/(z-z_0)`` and ``-1/(z-z_0)^2``;
the traction BIE wants ``∇_ξ=-∇_y``, which is `fundamental_stress`.
On the auxiliary line ``z_m^*-z_m^0=(ξ-ξ_0)J_0 α_m``:

- eq. 35: ``U^h=U_\\mathrm{lead}/((ξ-ξ_0)J_0)`` so ``F_G^*=U_\\mathrm{lead}φ_0/(ξ-ξ_0)``
  (``J_0`` cancels)
- eq. 37: ``T^h=T_\\mathrm{lead}/((ξ-ξ_0)^2 J_0^2)`` so
  ``F_H^*=T_\\mathrm{lead}(φ_0/(ξ-ξ_0)^2+φ_{,ξ}/(ξ-ξ_0))/J_0``
  (one ``J_0`` remains). A reading ``(J_0 α_m)^2`` in the denominator is
  ruled out by ``ρ^2 T^h φ J → T_\\mathrm{lead} φ/J_0``, not ``/J_0^2``.

``U^h_{ik}=n_j D_{kij}`` (traction ``i`` due to force ``k``), not the
``C_{iklm}T_{lj,m}`` index order of eq. 21, which is the transpose.
"""
function _lekh_Uh_Th_lead(p::LekhnitskiiParams, n::SVector{2})
    μ = p.mi
    α1 = μ[1] * n[1] - n[2]
    α2 = μ[2] * n[1] - n[2]
    (abs(α1) < 1e-30 || abs(α2) < 1e-30) && return nothing
    invα = @SMatrix [1 / α1  0; 0  1 / α2]
    μinvα = @SMatrix [μ[1] / α1  0; 0  μ[2] / α2]
    A, q, gmat, C = p.A, p.q, p.g, p.C
    # match `conj(X)'` = `transpose(X)` in `fundamental` / `fundamental_stress`
    XT = transpose
    ux = -2 * real(A * invα * XT(q))
    uy = -2 * real(A * μinvα * XT(q))
    px = 2 * real(A * invα * XT(gmat))
    py = 2 * real(A * μinvα * XT(gmat))
    D1 = C * SVector(ux[1, 1], uy[2, 1], uy[1, 1] + ux[2, 1])
    D2 = C * SVector(ux[1, 2], uy[2, 2], uy[1, 2] + ux[2, 2])
    S1 = C * SVector(px[1, 1], py[2, 1], py[1, 1] + px[2, 1])
    S2 = C * SVector(px[1, 2], py[2, 2], py[1, 2] + px[2, 2])
    n1, n2 = n[1], n[2]
    Uh = @SMatrix [
        n1 * D1[1] + n2 * D1[3]   n1 * D2[1] + n2 * D2[3]
        n1 * D1[3] + n2 * D1[2]   n1 * D2[3] + n2 * D2[2]
    ]
    Th = @SMatrix [
        n1 * S1[1] + n2 * S1[3]   n1 * S2[1] + n2 * S2[3]
        n1 * S1[3] + n2 * S1[2]   n1 * S2[3] + n2 * S2[2]
    ]
    return Uh, Th
end

"""Frozen ``J_0,n,φ_0,φ_{,ξ}`` SST factors for Guiggiani Laurent tensors."""
function _aniso_sst_star(props::AnisotropicElasticity, poly, nodes, a::Real)
    g = _geom_1d(poly, nodes, a)
    g === nothing && return nothing
    Nrow, J, _, n = g
    lead = _lekh_Uh_Th_lead(props.params, n)
    lead === nothing && return nothing
    Uh, Th = lead
    _, dN = shapefun(poly, a)
    return Uh, Th, Nrow, view(dN, 1, :), J, n
end
_aniso_sst_star(::Problem, poly, nodes, a) = nothing

function laurent_coefficients(props::Laplace, poly, nodes, collocation::Tuple, θ::Real,
        which::Symbol, order::Integer)
    order = _normalize_guiggiani_order(order)
    g = _geom_2d(poly, nodes, collocation...)
    g === nothing && return nothing
    Nrow, J, xξ, xη, _ = g
    nN = length(Nrow)
    z = zeros(nN)
    if which === :H && order == -1
        return (z, z, z)
    elseif which === :G && order == 0
        cθ, sθ = cos(θ), sin(θ)
        A = hypot(xξ[1]*cθ + xη[1]*sθ, xξ[2]*cθ + xη[2]*sθ, xξ[3]*cθ + xη[3]*sθ)
        A < 1e-30 && return (z, z, z)
        F0 = [Nrow[j] * J / (4 * π * float(props.k) * A) for j in 1:nN]
        return (z, z, F0)
    end
    return nothing
end

function laurent_coefficients(props::Elasticity, poly, nodes, collocation::Tuple, θ::Real,
        which::Symbol, order::Integer)
    order = _normalize_guiggiani_order(order)
    g = _geom_2d(poly, nodes, collocation...)
    g === nothing && return nothing
    Nrow, J, xξ, xη, n = g
    nN = length(Nrow)
    ν = float(props.nu)
    z = zeros(3, 3 * nN)
    cθ, sθ = cos(θ), sin(θ)
    Avec = xξ * cθ + xη * sθ
    A = norm(Avec)
    A < 1e-30 && return (z, z, z)
    if which === :G && order == 0
        Â = Avec / A
        c = J / (16 * π * float(props.mu) * (1 - ν) * A)
        δ = @SMatrix [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
        F0 = zeros(3, 3 * nN)
        _pack_nodal_tensor!(F0, c * ((3 - 4ν) * δ + Â * Â'), Nrow)
        return (z, z, F0)
    elseif which === :H && order == -1
        c = -(1 - 2ν) * J / (8 * π * (1 - ν) * A^3)
        Tant = @SMatrix [
            0.0  c*(n[1]*Avec[2]-n[2]*Avec[1])  c*(n[1]*Avec[3]-n[3]*Avec[1])
            c*(n[2]*Avec[1]-n[1]*Avec[2])  0.0  c*(n[2]*Avec[3]-n[3]*Avec[2])
            c*(n[3]*Avec[1]-n[1]*Avec[3])  c*(n[3]*Avec[2]-n[2]*Avec[3])  0.0
        ]
        Fm1 = zeros(3, 3 * nN)
        _pack_nodal_tensor!(Fm1, Tant, Nrow)
        return (z, Fm1, z)
    end
    return nothing
end

# =============================================================================
# Analytic geometry helpers (Laplace / Kelvin Guiggiani)
# =============================================================================

"""Shape + tangent/normal + Jacobian of a 2-D parent curve at `a`."""
function _geom_1d(poly, nodes, a)
    N, dN = shapefun(poly, a)
    nN = size(N, 2)
    x = zero(eltype(nodes))
    dx = zero(eltype(nodes))
    @inbounds for k in 1:nN
        x += N[1, k] * nodes[k]
        dx += dN[1, k] * nodes[k]
    end
    J = norm(dx)
    J < 1e-30 && return nothing
    t = dx / J
    n = SVector{2,Float64}(t[2], -t[1])   # same as tan2normal (Input.jl is later)
    return view(N, 1, :), J, t, n
end

"""Shape + parametric derivatives + surface Jacobian at parent `(aξ, aη)`."""
function _geom_2d(poly, nodes, aξ, aη)
    L, Lξ, Lη = shapefun2D(poly, poly, aξ, aη)
    nN = size(L, 2)
    xξ = zero(eltype(nodes))
    xη = zero(eltype(nodes))
    @inbounds for k in 1:nN
        xξ += Lξ[1, k] * nodes[k]
        xη += Lη[1, k] * nodes[k]
    end
    Jv = cross(xξ, xη)
    J = norm(Jv)
    J < 1e-30 && return nothing
    n = Jv / J
    return view(L, 1, :), J, xξ, xη, n
end

@inline function _pack_nodal_tensor!(out::AbstractMatrix, Tαβ, Nrow, scale=1)
    dim = size(Tαβ, 1)
    nN = length(Nrow)
    fill!(out, 0)
    @inbounds for j in 1:nN
        c0 = dim * (j - 1)
        Nj = Nrow[j] * scale
        for β in 1:dim, α in 1:dim
            out[α, c0 + β] = Tαβ[α, β] * Nj
        end
    end
    return out
end

# =============================================================================
# Helpers: seeds & projection residuals
# =============================================================================

"""Linear seed ξ₀ ∈ [-1,1] from the chord projection of `pf` onto the element."""
function _seed_1d(poly, nodes, pf)
    Δ = nodes[end] - nodes[1]
    ξs = poly.nodes
    ξ0 = (ξs[end] - ξs[1]) * dot(Δ, pf - nodes[1]) / (norm(Δ)^2 + eps()) + ξs[1]
    return clamp(ξ0, -1.0, 1.0)
end

"""Bilinear seed (ξ₀, η₀) from edge chords on a surface patch."""
function _seed_2d(poly, nodes, pf)
    ξs = poly.nodes
    nξ = length(ξs)
    Δ1 = nodes[min(2, end)] - nodes[1]
    Δ2 = nodes[min(end, nξ)] - nodes[1]
    eet1 = clamp((ξs[min(2, end)] - ξs[1]) * dot(Δ1, pf - nodes[1]) / (norm(Δ1)^2 + eps()) + ξs[1], -1.0, 1.0)
    eet2 = clamp((ξs[end] - ξs[1]) * dot(Δ2, pf - nodes[1]) / (norm(Δ2)^2 + eps()) + ξs[1], -1.0, 1.0)
    return eet1, eet2
end

@inline function _proj1d_rj(poly, nodes, pf, ξ::Float64)
    N, dN = shapefun(poly, ξ)
    d2N = dN * poly.Dmat
    x = (N * nodes)[1]
    dx = (dN * nodes)[1]
    d2x = (d2N * nodes)[1]
    r = x - pf
    res = dot(r, dx)
    J = dot(dx, dx) + dot(r, d2x)
    return res, J, x
end

function _proj2d_rj(poly, nodes, pf, ξ::Float64, η::Float64)
    N, dNξ, dNη = shapefun2D(poly, poly, ξ, η)
    nξ = length(poly.nodes)
    Dx = kron(Matrix(I, nξ, nξ), poly.Dmat)
    Dy = kron(poly.Dmat, Matrix(I, nξ, nξ))
    d2Nξξ = dNξ * Dx
    d2Nηη = dNη * Dy
    d2Nξη = dNξ * Dy
    x = (N * nodes)[1]
    xξ = (dNξ * nodes)[1]
    xη = (dNη * nodes)[1]
    xξξ = (d2Nξξ * nodes)[1]
    xηη = (d2Nηη * nodes)[1]
    xξη = (d2Nξη * nodes)[1]
    r = x - pf
    F = @SVector [dot(r, xξ), dot(r, xη)]
    J = @SMatrix [
        dot(xξ, xξ)+dot(r, xξξ)  dot(xη, xξ)+dot(r, xξη)
        dot(xξ, xη)+dot(r, xξη)  dot(xη, xη)+dot(r, xηη)
    ]
    return F, J, x
end

# =============================================================================
# Helpers: sinh maps & surface near-field
# =============================================================================

function _sinhtrans(u, w, a, b)
    T = float(eltype(w))
    a = T(a)
    b = T(b)
    # Eq. (40): η0=0 and |ζ0|>1 → exponential (real pole outside [-1,1])
    if abs(b) < T(1e-14) && abs(a) > 1
        return _eq40_real_pole(u, w, a)
    end
    b = max(b, T(1e-14))
    x, J = _sinh_map(u, a, b)
    return x, w .* J
end

"""Granados eq. (40) ``η0=0``: ``ξ=ζ0+sign(ξ-ζ0)\\,exp((ξ̃-B)/A)``.

On ``[-1,1]`` with ``|ζ0|>1``, ``sign(ξ-ζ0)=-sign(ζ0)`` is constant.
`A,B` from ``ξ(±1)=±1``:
``log[s(±1-ζ0)]=(±1-B)/A``, so ``A`` is negative when the pole is to the
right (``ζ0>1``) and the map still increases.
"""
function _eq40_AB(ζ0::T) where {T<:Real}
    s = -sign(ζ0)                       # sign(ξ-ζ0) on [-1,1]
    # s exp((±1-B)/A) = ±1 - ζ0  > 0
    lm = log(s * (-1 - ζ0))
    lp = log(s * (1 - ζ0))
    A = T(2) / (lp - lm)
    B = -A / 2 * (lp + lm)
    return A, B, s
end

function _eq40_real_pole(u, w, ζ0)
    T = float(eltype(w))
    ζ0 = T(ζ0)
    abs(ζ0) <= 1 && return collect(T, u), collect(T, w)
    A, B, s = _eq40_AB(ζ0)
    abs(A) < T(1e-30) && return collect(T, u), collect(T, w)
    x = similar(u, T)
    J = similar(w, T)
    @inbounds for i in eachindex(u)
        e = exp((T(u[i]) - B) / A)
        x[i] = ζ0 + s * e               # paper: ζ0 + sign(ξ-ζ0) exp((ξ̃-B)/A)
        J[i] = s * e / A
    end
    return x, collect(T, w) .* J
end

function _sinh_map(u, a::T, b::T) where {T<:Real}
    μ = T(0.5) * (asinh((1 + a) / b) + asinh((1 - a) / b))
    η = T(0.5) * (asinh((1 + a) / b) - asinh((1 - a) / b))
    x = @. a + b * sinh(μ * u - η)
    J = @. b * μ * cosh(μ * u - η)
    return x, J
end

"""Complex pole ``ξ_0=ζ_0+iη_0`` of ``z(ξ)=x_1+ix_2`` with ``z(ξ_0)=y`` (Granados 2026)."""
function _complex_pole_1d(poly, nodes, pf)
    nN = length(nodes)
    y = complex(float(pf[1]), float(pf[2]))
    if nN == 2 || nN == 3
        ξs = float.(poly.nodes)
        length(ξs) == nN || (ξs = collect(range(-1.0, 1.0; length=nN)))
        Z = [complex(float(nodes[k][1]), float(nodes[k][2])) for k in 1:nN]
        if nN == 2
            # linear: z = B ξ + C
            B = (Z[2] - Z[1]) / (ξs[2] - ξs[1] + 1e-30)
            C = Z[1] - B * ξs[1]
            abs(B) < 1e-30 && return 0.0, 1.0
            r = (y - C) / B
            return real(r), abs(imag(r))
        end
        # unique quadratic through 3 interpolation nodes
        A, B, C = _fit_quad_complex(ξs, Z)
        if abs(A) < 1e-14
            abs(B) < 1e-30 && return 0.0, 1.0
            r = (y - C) / B
            return real(r), abs(imag(r))
        end
        disc = sqrt(B * B - 4 * A * (C - y))
        r1 = (-B + disc) / (2 * A)
        r2 = (-B - disc) / (2 * A)
        r = abs(imag(r1)) <= abs(imag(r2)) ? r1 : r2
        return real(r), abs(imag(r))
    end
    a, _, dist = closest_point_1d(poly, nodes, pf; ξ0=_seed_1d(poly, nodes, pf))
    N, dN = shapefun(poly, a)
    dx = (dN * nodes)[1]
    return a, dist / max(norm(dx), 1e-14)
end

function _fit_quad_complex(ξ, Z)
    # Vandermonde ξk², ξk, 1
    M = @SMatrix [
        ξ[1]^2  ξ[1]  1
        ξ[2]^2  ξ[2]  1
        ξ[3]^2  ξ[3]  1
    ]
    c = M \ SVector(Z[1], Z[2], Z[3])
    return c[1], c[2], c[3]
end

"""True when eq. (41) should use the collinear Möbius (atan saturates)."""
@inline function _eq41_collinear(ζ0, η0, den=nothing)
    abs(ζ0) <= 1 && return false
    abs(η0) < oftype(float(η0), 1e-8) && return true
    den !== nothing && abs(den) < oftype(float(den), 1e-10) && return true
    return false
end

"""Tangent map, natural for ``1/r^2`` (Granados 2026 eq. 41).

``η0≠0``: ``ξ=ζ0+η0\\tan(η0(ξ̃-B)/A)``. ``η0=0`` and ``|ζ0|>1``: Möbius
``ξ=ζ0-A/(ξ̃-B)`` with ``A=ζ0^2-1``, ``B=-ζ0`` (collinear real pole).
`A,B` are fixed by ``ξ(±1)=±1``. Tiny ``η0`` with ``|ζ0|>1`` also uses
Möbius: ``\\mathrm{atan}`` saturates and would send nodes off ``[-1,1]``.
"""
function _tangenttrans(u, w, ζ0, η0)
    T = float(eltype(w))
    ζ0 = T(ζ0)
    η0 = T(η0)
    _eq41_collinear(ζ0, η0) && return _eq41_real_pole(u, w, ζ0)
    η0 = max(η0, T(1e-14))
    # Eq. (41) η0≠0: ξ = ζ0 + η0 tan(η0 (ξ̃-B)/A)
    θp = atan((1 - ζ0) / η0)
    θm = atan((-1 - ζ0) / η0)
    den = θp - θm
    _eq41_collinear(ζ0, η0, den) && return _eq41_real_pole(u, w, ζ0)
    A = T(2) * η0 / den
    B = -(θp + θm) / den
    halfπ = T(π) / 2 - T(1e-8)
    x = similar(u, T)
    J = similar(w, T)
    @inbounds for i in eachindex(u)
        φ = clamp(η0 * (T(u[i]) - B) / A, -halfπ, halfπ)
        tn = tan(φ)
        x[i] = ζ0 + η0 * tn
        J[i] = (η0 * η0 / A) * (1 + tn * tn)
    end
    return x, collect(T, w) .* J
end

"""Granados eq. (41) ``η0=0``: ``ξ=ζ0-A/(ξ̃-B)``, ``A=ζ0^2-1``, ``B=-ζ0``."""
function _eq41_real_pole(u, w, ζ0)
    T = float(eltype(w))
    ζ0 = T(ζ0)
    abs(ζ0) <= 1 && return collect(T, u), collect(T, w)
    A = ζ0 * ζ0 - 1
    B = -ζ0
    x = similar(u, T)
    J = similar(w, T)
    @inbounds for i in eachindex(u)
        d = T(u[i]) - B
        x[i] = ζ0 - A / d
        J[i] = A / (d * d)
    end
    return x, collect(T, w) .* J
end

"""Solve (65) of Granados–Gallego JACM 2026 for the p3c pole ``(ζ̃_0,η̃_0)``."""
function _p3c_inv_pole(ζ0::Real, η0::Real)
    T = float(typeof(ζ0))
    ζ0 = T(ζ0)
    η0 = T(η0)
    if abs(η0) < T(1e-14)
        # cubic ζ̃³ − 3ζ0 ζ̃² + 3ζ̃ − ζ0 = 0
        x = ζ0
        for _ in 1:25
            g = x^3 - 3 * ζ0 * x^2 + 3 * x - ζ0
            gp = 3 * x^2 - 6 * ζ0 * x + 3
            abs(gp) < T(1e-30) && break
            x -= g / gp
            abs(g) < T(1e-14) && break
        end
        return x, zero(T)
    end
    # seed: η̃ from 2η̃³ − 3η0 η̃² − η0 = 0 (ζ0=0) then (86)
    η̃ = cbrt(abs(η0)) * sign(η0)
    for _ in 1:20
        g = 2 * η̃^3 - 3 * η0 * η̃^2 - η0
        gp = 6 * η̃^2 - 6 * η0 * η̃
        abs(gp) < T(1e-30) && break
        η̃ -= g / gp
    end
    η̃ += cbrt(η0) * sqrt(ζ0^2 + η0^2)   # (86)
    ζ̃ = ζ0
    for _ in 1:30
        den = 1 + 3 * ζ̃^2 + 3 * η̃^2
        nζ = (3 + ζ̃^2 + 3 * η̃^2) * ζ̃
        nη = 2 * η̃^3
        f1 = nζ / den - ζ0
        f2 = nη / den - η0
        abs(f1) + abs(f2) < T(1e-14) && break
        denζ = 6 * ζ̃
        denη = 6 * η̃
        nζ_ζ = 3 + 3 * ζ̃^2 + 3 * η̃^2
        nζ_η = 6 * η̃ * ζ̃
        nη_ζ = zero(T)
        nη_η = 6 * η̃^2
        den2 = den * den
        j11 = (nζ_ζ * den - nζ * denζ) / den2
        j12 = (nζ_η * den - nζ * denη) / den2
        j21 = (nη_ζ * den - nη * denζ) / den2
        j22 = (nη_η * den - nη * denη) / den2
        detJ = j11 * j22 - j12 * j21
        abs(detJ) < T(1e-30) && break
        ζ̃ -= (j22 * f1 - j12 * f2) / detJ
        η̃ -= (-j21 * f1 + j11 * f2) / detJ
    end
    return ζ̃, η̃
end

"""Complete cubic (p3c), Granados & Gallego JACM 12 (2026) eqs. 53–60."""
function _p3ctrans(u, w, ζ0, η0)
    T = float(eltype(w))
    ζ̃, η̃ = _p3c_inv_pole(T(ζ0), T(η0))
    den = 1 + 3 * ζ̃^2 + 3 * η̃^2
    A = T(3) / den
    B = (3 + ζ̃^2) * ζ̃ / den
    x = similar(u, T)
    J = similar(w, T)
    @inbounds for i in eachindex(u)
        s = T(u[i])
        ds = s - ζ̃
        x[i] = A * (ds^3 / 3 + η̃^2 * s) + B
        J[i] = A * (ds^2 + η̃^2)
    end
    return x, collect(T, w) .* J
end

"""Tangent then p3c on the tangent's ``±π/2`` poles (EABE 189 §4.1)."""
function _tanp3ctrans(u, w, ζ0, η0)
    T = float(eltype(w))
    ζ0 = T(ζ0)
    η0 = T(η0)
    # Eq. (41) collinear: Möbius; no ±π/2 poles, so no extra p3c
    _eq41_collinear(ζ0, η0) && return _eq41_real_pole(u, w, ζ0)
    η0 = max(η0, T(1e-14))
    ap = atan((1 - ζ0) / η0)
    am = atan((1 + ζ0) / η0)
    den = ap + am
    if _eq41_collinear(ζ0, η0, den)
        return _eq41_real_pole(u, w, ζ0)
    end
    if abs(den) < T(1e-30)
        return _p3ctrans(u, w, ζ0, η0)
    end
    At = T(2) / den
    Bt = -At * (ap - am) / T(2)
    uL = Bt - At * T(π) / 2
    uR = Bt + At * T(π) / 2
    if abs(ζ0) < 1 && -1 < Bt < 1
        return _tanp3c_split(u, w, ζ0, η0, At, Bt, uL, uR)
    elseif ζ0 >= 1
        # left tangent pole can hit −1
        return _tanp3c_compose(u, w, ζ0, η0, At, Bt, uL, T(0))
    else
        return _tanp3c_compose(u, w, ζ0, η0, At, Bt, uR, T(0))
    end
end

function _tanp3c_compose(u, w, ζ0, η0, At, Bt, u_pole, η_pole)
    # p3c in ũ-plane about the tangent singularity, then tangent
    ũ, w1 = _p3ctrans(u, w, u_pole, max(abs(η_pole), 1e-14))
    halfπ = oftype(At, π) / 2 - oftype(At, 1e-8)
    x = similar(ũ)
    Jt = similar(ũ)
    @inbounds for i in eachindex(ũ)
        θ = clamp((ũ[i] - Bt) / At, -halfπ, halfπ)
        tn = tan(θ)
        x[i] = ζ0 + η0 * tn
        Jt[i] = (η0 / At) * (1 + tn * tn)
    end
    return x, w1 .* Jt
end

function _tanp3c_split(u, w, ζ0, η0, At, Bt, uL, uR)
    T = float(eltype(w))
    # two subintervals of the tangent parent: [−1,Bt] and [Bt,1]
    xl, wl = _tanp3c_half(u, w, ζ0, η0, At, Bt, T(-1), Bt, uL)
    xr, wr = _tanp3c_half(u, w, ζ0, η0, At, Bt, Bt, T(1), uR)
    return vcat(xl, xr), vcat(wl, wr)
end

function _tanp3c_half(u, w, ζ0, η0, At, Bt, lo, hi, u_pole)
    T = float(eltype(w))
    # affine [-1,1] → [lo,hi], then p3c about the pole in that local chart, then tangent
    mid = (hi + lo) / 2
    h = (hi - lo) / 2
    h <= T(1e-14) && return _tangenttrans(u, w, ζ0, η0)
    # pole in local s: ũ = mid + h*s, s = (ũ-mid)/h
    s_pole = (u_pole - mid) / h
    s, ws = _p3ctrans(u, w, s_pole, T(1e-14))
    ũ = @. mid + h * s
    ws = ws .* h
    halfπ = T(π) / 2 - T(1e-8)
    x = similar(ũ)
    Jt = similar(ũ)
    @inbounds for i in eachindex(ũ)
        θ = clamp((ũ[i] - Bt) / At, -halfπ, halfπ)
        tn = tan(θ)
        x[i] = ζ0 + η0 * tn
        Jt[i] = (η0 / At) * (1 + tn * tn)
    end
    return x, ws .* Jt
end

"""`niter=1` is `_sinhtrans`; `niter=2` is Granados sinh–sinh (pole ``B_*±i(π/2)A_*``)."""
function _sinhtrans_iterated(u, w, a, b; niter::Int=1)
    T = float(eltype(w))
    a = T(a)
    b = T(b)
    # Eq. (40) η0=0: one exponential is the natural 1/r map; do not iterate
    if abs(b) < T(1e-14) && abs(a) > 1
        return _eq40_real_pole(u, w, a)
    end
    b = max(b, T(1e-14))
    maps = NTuple{2,T}[]
    aa, bb = a, b
    for _ in 1:niter
        push!(maps, (aa, bb))
        μ = T(0.5) * (asinh((1 + aa) / bb) + asinh((1 - aa) / bb))
        η = T(0.5) * (asinh((1 + aa) / bb) - asinh((1 - aa) / bb))
        μ < T(1e-14) && break
        aa = clamp(η / μ, nextfloat(T(-1)), prevfloat(T(1)))
        bb = max(T(π) / (2 * μ), T(1e-14))
    end
    x = collect(T, u)
    J = ones(T, length(u))
    for (aa, bb) in Iterators.reverse(maps)
        xs, dJ = _sinh_map(x, aa, bb)
        x = collect(xs)
        J .*= dJ
    end
    return x, collect(T, w) .* J
end

function _nearfield_2d(aξ, aη, b; n::Int=8, mode::Symbol=:auto)
    mode === :dibem && (mode = :auto)  # far-face fallback from integrate_element
    mode === :plain && return _tensor_plain_2d(n)
    b = max(float(b), 1e-14)
    radial = mode === :tanp3c ? :tanp3c : mode === :tangent ? :tangent : :sinh
    use_polar = mode === :polar || mode === :tanp3c || mode === :tangent ||
                (mode === :auto && b < 5e-2)
    if use_polar
        # cld(n,2) left ∑w short by 5–20% (nρ=4); need ≥n per ray
        m = max(n, 8)
        return _polar_2d(m, m, float(aξ), float(aη), b; radial=radial)
    end
    return _tensor_sinh_2d(n, float(aξ), float(aη), b)
end

function _tensor_plain_2d(n)
    u, w = gausslegendre(n)
    N = n * n
    ξ = Vector{Float64}(undef, N)
    η = Vector{Float64}(undef, N)
    ww = Vector{Float64}(undef, N)
    k = 1
    @inbounds for j in 1:n, i in 1:n
        ξ[k] = u[i]
        η[k] = u[j]
        ww[k] = w[i] * w[j]
        k += 1
    end
    return ξ, η, ww
end

function _tensor_sinh_2d(n, aξ, aη, b)
    u, w = gausslegendre(n)
    x1, w1 = _sinhtrans(u, w, aξ, b)
    x2, w2 = _sinhtrans(u, w, aη, b)
    N = n * n
    ξ = Vector{Float64}(undef, N)
    η = Vector{Float64}(undef, N)
    ww = Vector{Float64}(undef, N)
    k = 1
    @inbounds for j in 1:n, i in 1:n
        ξ[k] = x1[i]
        η[k] = x2[j]
        ww[k] = w1[i] * w2[j]
        k += 1
    end
    return ξ, η, ww
end

_polar_sinh_2d(nρ, nθ, aξ, aη, b) = _polar_2d(nρ, nθ, aξ, aη, b; radial=:sinh)

function _polar_2d(nρ::Int, nθ::Int, aξ::Float64, aη::Float64, b::Float64; radial::Symbol=:sinh)
    b = max(b, 1e-14)
    aξ = clamp(aξ, -1.0, 1.0)
    aη = clamp(aη, -1.0, 1.0)
    corners = ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0))
    θs = Float64[]
    for c in corners
        dx = c[1] - aξ
        dy = c[2] - aη
        hypot(dx, dy) < 1e-14 && continue
        push!(θs, atan(dy, dx))
    end
    if isempty(θs)
        return _tensor_plain_2d(nρ)
    end
    sort!(θs)
    push!(θs, θs[1] + 2π)
    uρ, wρ0 = gausslegendre(nρ)
    uθ, wθ = gausslegendre(nθ)
    ξs = Float64[]; ηs = Float64[]; ws = Float64[]
    sizehint!(ξs, (length(θs) - 1) * nρ * nθ)
    sizehint!(ηs, (length(θs) - 1) * nρ * nθ)
    sizehint!(ws, (length(θs) - 1) * nρ * nθ)
    for k in 1:(length(θs) - 1)
        dθ = θs[k + 1] - θs[k]
        dθ < 1e-14 && continue
        θmid = θs[k] + 0.5 * dθ
        _ray_to_square(aξ, aη, θmid) <= 1e-14 && continue
        θhalf = 0.5 * dθ
        for j in eachindex(uθ)
            θ = θmid + θhalf * uθ[j]
            wθj = wθ[j] * θhalf
            ρmax = _ray_to_square(aξ, aη, θ)
            ρmax <= 1e-14 && continue
            ρ, wρ = _radial_quad(uρ, wρ0, ρmax, b, radial)
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

function _radial_quad(u, w, ρmax, b, radial::Symbol)
    radial === :tanp3c && return _parent_radial(u, w, ρmax, b, :tanp3c)
    radial === :tangent && return _parent_radial(u, w, ρmax, b, :tangent)
    return _sinh_radial(u, w, ρmax, b)
end

"""Sinh on ``[0,ρmax]`` with peak width `b` at ``ρ=0``."""
function _sinh_radial(u, w, ρmax, b)
    b = max(b, 1e-14)
    tmax = asinh(ρmax / b)
    t = @. tmax * (u + 1) / 2
    ρ = @. b * sinh(t)
    J = @. b * cosh(t) * (tmax / 2)
    return ρ, w .* J
end

"""Map ``[0,ρmax]→[-1,1]``, then Granados tangent/tan-p3c with pole at ``ζ_0=-1``.

Parent distance ``η_0=2b/ρmax`` so ``R=\\sqrt{ρ^2+b^2}`` is ``(ρ_{max}/2)\\sqrt{(ξ+1)^2+η_0^2}``.
"""
function _parent_radial(u, w, ρmax, b, kind::Symbol)
    T = float(eltype(w))
    ρmax <= 0 && return T[], T[]
    η0 = 2 * max(T(b), T(1e-14)) / T(ρmax)
    ξ, wξ = kind === :tanp3c ? _tanp3ctrans(u, w, T(-1), η0) :
                               _tangenttrans(u, w, T(-1), η0)
    ρ = similar(ξ, T)
    wρ = similar(wξ, T)
    h = T(ρmax) / 2
    @inbounds for i in eachindex(ξ)
        s = clamp(ξ[i], T(-1), T(1))
        ρ[i] = h * (s + 1)
        wρ[i] = wξ[i] * h
    end
    return ρ, wρ
end

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

# =============================================================================
# Helpers: Guiggiani Laurent / rays
# =============================================================================

function _guiggiani_rule(a, qsi, w, h)
    T = Float64
    a = T(a)
    if qsi === nothing || w === nothing
        qsi, w = gausslegendre(16)
    end
    return a, collect(T, qsi), collect(T, w), T(h)
end

@inline _guiggiani_rays(a::T) where {T<:Real} =
    ((T(-1), a - T(-1)), (T(1), T(1) - a))

function _normalize_guiggiani_order(o::Integer)
    o = Int(o)
    o == 1 && return -1
    o == 2 && return -2
    o == 3 && return -3
    o == 4 && return -4
    o in (-4, -3, -2, -1, 0) || throw(ArgumentError(
        "Guiggiani order must be in {-4,-3,-2,-1,0} (aliases 1→-1, …, 4→-4; got $o)"))
    return o
end

function _laurent_extrapolate_kwargs(kwargs)
    defaults = (; contract=1 / 2, atol=1.0e-12, rtol=1.0e-10)
    return (; defaults..., kwargs...)
end

function _strip_laurent_kwargs(kwargs)
    nt = values(kwargs)
    method = get(nt, :laurent, :interp)
    ninterp = Int(get(nt, :ninterp, 20))
    (method === :auto || method === :richardson || method === :interp) || throw(ArgumentError(
        "laurent must be :auto, :richardson, or :interp (got $(repr(method)))"))
    rest = Base.structdiff(nt, NamedTuple{(:laurent, :ninterp)})
    return method, ninterp, rest
end

"""Cached Gauss–Legendre nodes, quadrature weights, barycentric weights."""
function _build_interp_gp(n::Int)
    ξ, w = gausslegendre(n)
    ξv = collect(Float64, ξ)
    wv = collect(Float64, w)
    bw = collect(Float64, weights(ArbitraryPolynomial{Float64}, ξv))
    return (ξv, wv, bw)
end

const _INTERP_GP = Dict{Int,NTuple{3,Vector{Float64}}}()
const _INTERP_GP_LOCK = ReentrantLock()

"""Gauss–`n` nodes/weights and barycentric `wᵢ` (independent of collocation `a`)."""
function _interp_gp(n::Integer)
    n = Int(n)
    n < 2 && throw(ArgumentError("ninterp must be ≥ 2 (got $n)"))
    lock(_INTERP_GP_LOCK) do
        get!(_INTERP_GP, n) do
            _build_interp_gp(n)
        end
    end
end

"""Barycentric Lagrange `Nᵢ(a)` from cached weights (Berrut–Trefethen)."""
function _barycentric_eval(ξ::AbstractVector, bw::AbstractVector, a::Real)
    T = promote_type(eltype(ξ), eltype(bw), typeof(float(a)))
    aa = convert(T, a)
    n = length(ξ)
    Ni = Vector{T}(undef, n)
    Msum = zero(T)
    exact = 0
    @inbounds for i in 1:n
        d = aa - convert(T, ξ[i])
        d == 0 && (exact = i)
        Ni[i] = convert(T, bw[i]) / d
        Msum += Ni[i]
    end
    if exact > 0
        fill!(Ni, zero(T))
        Ni[exact] = one(T)
        return Ni
    end
    invs = inv(Msum)
    @inbounds for i in 1:n
        Ni[i] *= invs
    end
    return Ni
end

"""Gauss–Legendre nodes/weights, barycentric `Nᵢ(a)`, and `ξᵢ - a`.

Nodes, quadrature weights, and barycentric `wᵢ` are cached per `n`
(default 20). Only `Nᵢ(a)` and `ξᵢ-a` depend on collocation. A node
with `|ξᵢ-a|≤1e-12` is dropped and subset weights are rebuilt; the
usual collocation (element Gauss-2/3) never hits interpolant Gauss-20.
"""
function _interp_rule(a::Real, n::Integer)
    ξ, w, bw = _interp_gp(n)
    a = float(a)
    ndrop = 0
    @inbounds for i in eachindex(ξ)
        abs(ξ[i] - a) ≤ 1e-12 && (ndrop += 1)
    end
    if ndrop == 0
        return ξ, w, _barycentric_eval(ξ, bw, a), ξ .- a
    end
    keep = [abs(ξ[i] - a) > 1e-12 for i in eachindex(ξ)]
    count(keep) < 2 && throw(ArgumentError(
        "laurent_coefficients_interp: fewer than 2 Gauss nodes away from ξ=$a"))
    ξk = ξ[keep]
    wk = w[keep]
    bwk = collect(eltype(ξk), weights(ArbitraryPolynomial{eltype(ξk)}, ξk))
    return ξk, wk, _barycentric_eval(ξk, bwk, a), ξk .- a
end

function _interp_nodes(a::Real, n::Integer)
    ξ, _, Ni, δ = _interp_rule(a, n)
    return ξ, Ni, δ
end

function _barycentric_sum(Ni, vals)
    acc = Ni[1] * vals[1]
    @inbounds for i in 2:length(Ni)
        acc += Ni[i] * vals[i]
    end
    return acc
end

"""Sequential interpolation cascade matching the Richardson Laurent subtraction."""
function _interp_cascade(fi, Ni, δ, order::Int)
    p = -order
    work = [fi[i] for i in eachindex(fi)]
    F = Vector{typeof(work[1])}(undef, p + 1)
    @inbounds for (j, k) in enumerate(p:-1:0)
        acc = _barycentric_sum(Ni, [work[i] * (δ[i]^k) for i in eachindex(work)])
        F[j] = acc
        if k > 0
            for i in eachindex(work)
                work[i] = work[i] - acc / (δ[i]^k)
            end
        end
    end
    return _pack_interp_coeffs(F, order)
end

function _pack_interp_coeffs(F, order::Int)
    if order == -4
        return (F[1], F[2], F[3], F[4], F[5])
    elseif order == -3
        z = zero(F[1])
        return (z, F[1], F[2], F[3], F[4])
    elseif order == -2
        return (F[1], F[2], F[3])
    else # -1
        z = zero(F[1])
        return (z, F[1], F[2])
    end
end

"""Signed `(ξ-a)` Laurent tuple → Guiggiani ray coefficients `F_{-k}(s)=A_{-k} s^k`."""
function _signed_laurent_to_ray(C, s, order::Int)
    if order <= -3
        F₋₄, F₋₃, F₋₂, F₋₁, F₀ = C
        return (F₋₄, F₋₃ * s, F₋₂, F₋₁ * s, F₀)
    end
    F₋₂, F₋₁, F₀ = C
    return (F₋₂, F₋₁ * s, F₀)
end

"""Hadamard / CPV ``∫_{-1}^{1} (ξ-a)^{-k} dξ``."""
function _hfp_parent(a, k::Integer)
    a = float(a)
    k == 1 && return log(abs((1 - a) / (1 + a)))
    k == 2 && return -2 / (1 - a * a)
    k == 3 && return -1 / (2 * (1 - a)^2) + 1 / (2 * (1 + a)^2)
    k == 4 && return -1 / (3 * (1 - a)^3) - 1 / (3 * (1 + a)^3)
    throw(ArgumentError("HFP power must be in {1,2,3,4} (got $k)"))
end

function _log_parent(a)
    a = float(a)
    return (1 - a) * log(abs(1 - a)) + (1 + a) * log(abs(1 + a)) - 2
end

function _interp_reg_node(fi, δ, C, order::Int)
    if order <= -3
        F4, F3, F2, F1, _ = C
        val = fi
        order <= -4 && (val -= F4 / δ^4)
        val -= F3 / δ^3
        val -= F2 / δ^2
        val -= F1 / δ
        return val
    end
    F2, F1, F0 = C
    if order == 0
        return fi - F1 / δ - F0 * log(abs(δ))
    elseif order == -1
        return fi - F1 / δ
    else
        return fi - F2 / δ^2 - F1 / δ
    end
end

function _interp_analytic_parent(a, C, order::Int)
    if order <= -3
        F4, F3, F2, F1, _ = C
        ana = F1 * _hfp_parent(a, 1) + F2 * _hfp_parent(a, 2) + F3 * _hfp_parent(a, 3)
        order <= -4 && (ana += F4 * _hfp_parent(a, 4))
        return ana
    end
    F2, F1, F0 = C
    if order == 0
        return F1 * _hfp_parent(a, 1) + F0 * _log_parent(a)
    elseif order == -1
        return F1 * _hfp_parent(a, 1)
    else
        return F2 * _hfp_parent(a, 2) + F1 * _hfp_parent(a, 1)
    end
end

function _interp_integrate(fi, w, δ, C, order, a)
    acc = w[1] * _interp_reg_node(fi[1], δ[1], C, order)
    @inbounds for i in 2:length(w)
        acc += w[i] * _interp_reg_node(fi[i], δ[i], C, order)
    end
    return acc + _interp_analytic_parent(a, C, order)
end

"""Log coefficient after stripping 1/ρ.

`F₋₁` is still the interpolant of `f(ξ)(ξ-a)` at `a`. `F₀` is **not** the
interpolant of `(f-F₋₁/(ξ-a))/log|ξ-a|` at `a`: `1/log` has a weak
singularity, and high-order Lagrange at the collocation (especially off-
centre Gauss nodes) is unstable — Kirchhoff `G₂₂∼log` was 27% off, even
the wrong sign on the self term.

Fit `f - F₋₁/(ξ-a) ≈ F₀ log|ξ-a| + b` in the least-squares sense on the
sample nodes (same `F₀` for every matrix entry; `log` is scalar).
"""
function _interp_log_coeff(fi, Ni, δ)
    Fm1 = _barycentric_sum(Ni, [fi[i] * δ[i] for i in eachindex(fi)])
    sL = 0.0
    sL2 = 0.0
    nused = 0
    sW = zero(Fm1)
    sWL = zero(Fm1)
    @inbounds for i in eachindex(δ)
        adi = abs(δ[i])
        adi < 1e-14 && continue
        lg = log(adi)
        wi = fi[i] - Fm1 / δ[i]
        sL += lg
        sL2 += lg * lg
        sW += wi
        sWL += wi * lg
        nused += 1
    end
    z = zero(Fm1)
    nused < 2 && return (z, Fm1, z)
    detA = sL2 * nused - sL * sL
    abs(detA) < 1e-30 && return (z, Fm1, z)
    F0 = (nused * sWL - sL * sW) / detA
    return (z, Fm1, F0)
end

"""Sample `f` once at Gauss nodes; coefficients + remainder from those values."""
function _guiggiani_interp(f, a, order::Int, n::Integer; h=1e-3, rest=(;),
        geom=nothing, which=:H, nf=nothing)
    a = float(a)
    ξ, w, Ni, δ = _interp_rule(a, n)
    fi = [f(ξi) for ξi in ξ]
    C = order == 0 ? _interp_log_coeff(fi, Ni, δ) : _interp_cascade(fi, Ni, δ, order)
    return _interp_integrate(fi, w, δ, C, order, a)
end

function _guiggiani_interp_pair(f, a, og::Int, oh::Int, n::Integer; h=1e-3,
        rest=(;), geom=nothing, nf=nothing)
    a = float(a)
    ξ, w, Ni, δ = _interp_rule(a, n)
    samples = [f(ξi) for ξi in ξ]
    Fg = [s[1] for s in samples]
    Fh = [s[2] for s in samples]
    Cg = og == 0 ? _interp_log_coeff(Fg, Ni, δ) : _interp_cascade(Fg, Ni, δ, og)
    Ch = oh == 0 ? _interp_log_coeff(Fh, Ni, δ) : _interp_cascade(Fh, Ni, δ, oh)
    Ig = _interp_integrate(Fg, w, δ, Cg, og, a)
    Ih = _interp_integrate(Fh, w, δ, Ch, oh, a)
    return Ig, Ih
end

@inline _guiggiani_geom(props, poly, nodes) =
    (props === nothing || poly === nothing || nodes === nothing) ? nothing : (props, poly, nodes)

"""Laurent triple for a single radial integrand `Fr(ρ)` (Richardson)."""
function _laurent_coeffs_Fr(Fr, h, order::Int, probe, kw; kwargs...)
    if order == 0
        F0, _ = extrapolate(h; x0=zero(h), kw...) do ρ
            ρ ≤ eps(h) && return zero(probe)
            return Fr(ρ) / log(ρ)
        end
        z = zero(probe)
        return (z, z, F0)
    end
    return laurent_coefficients(Fr, h, Val(order); kwargs...)
end

"""Fused-pair Laurent triple: SST / closed-form [`laurent_coefficients`](@ref)
if `geom` is set and a formula exists, otherwise Richardson on `Fr`.

Kelvin / Lekhnitskii HBIE tensors use the **element** normal. Dual-BEM twins
contract ``T^h=n_ξ·S`` with the opposite collocation normal; flip when
``n_ξ·n_{el}<0``.
"""
function _laurent_pair_coeffs(Fr, h, order::Int, which::Symbol, probe, kw, geom,
        collocation, dir; nf=nothing, kwargs...)
    pick = which === :G ? (t -> t[1]) : (t -> t[2])
    if geom !== nothing
        C = laurent_coefficients(geom[1], geom[2], geom[3], collocation, dir, which, order)
        if C !== nothing
            return _align_twin_normal(C, geom, collocation, nf)
        end
    end
    return _laurent_coeffs_Fr(ρ -> pick(Fr(ρ)), h, order, probe, kw; kwargs...)
end

"""Flip a closed-form Laurent triple when ``n_ξ·n_{el}<0`` (dual-BEM twin).

Tensors are derived with the integration-element normal. Samples use that
same field normal (`nref = dad.Normal[elem]`); `nf` is the collocation
``n_ξ``. Flipping here must not be combined with also aligning `n` to
``n_ξ`` in `_sample_kernel`.
"""
function _align_twin_normal(C, geom, a, nf)
    nf === nothing && return C
    g = _geom_1d(geom[2], geom[3], a)
    g === nothing && return C
    _, _, _, n = g
    return dot(nf, n) < 0 ? (-C[1], -C[2], -C[3]) : C
end

@inline function _guiggiani_reg(F, ρ, order::Int, C)
    if order <= -3
        F₋₄, F₋₃, F₋₂, F₋₁, _ = C
        val = F
        order <= -4 && (val -= F₋₄ / ρ^4)
        val -= F₋₃ / ρ^3
        val -= F₋₂ / ρ^2
        val -= F₋₁ / ρ
        return val
    end
    F₋₂, F₋₁, F₀ = C
    if order == 0
        return F - F₀ * log(ρ)
    elseif order == -1
        return F - F₋₁ / ρ
    else # -2
        return F - F₋₂ / ρ^2 - F₋₁ / ρ
    end
end

@inline function _guiggiani_analytic(ρmax, order::Int, C)
    if order <= -3
        F₋₄, F₋₃, F₋₂, F₋₁, _ = C
        ana = F₋₁ * log(ρmax) - F₋₂ / ρmax - F₋₃ / (2 * ρmax^2)
        order <= -4 && (ana -= F₋₄ / (3 * ρmax^3))
        return ana
    end
    F₋₂, F₋₁, F₀ = C
    if order == 0
        return F₀ * (ρmax * log(ρmax) - ρmax)
    elseif order == -1
        return F₋₁ * log(ρmax)
    else
        return F₋₁ * log(ρmax) - F₋₂ / ρmax
    end
end

# =============================================================================
# 3D surface Guiggiani (polar + radial Laurent subtraction)
# =============================================================================

"""Map 1-D Laurent order to the radial integrand *after* the polar jacobian `ρ`.

3-D CBIE: single-layer ``U∼1/R`` becomes **bounded** (`ρ/R → const`);
double-layer ``T∼1/R²`` becomes **CPV** (`ρ/R² ∼ 1/ρ`). 2-D `order=0` is
`log ρ` on a curve; here it means the bounded 3-D polar integrand.
"""
function _surface_kind(order::Integer)
    o = _normalize_guiggiani_order(order)
    o == 0 && return :bounded
    o == -1 && return :cpv
    return :hfp
end

function _surface_laurent(Fr, h, kind::Symbol, probe, kw, geom, collocation, θ, which; kwargs...)
    order = kind === :bounded ? 0 : kind === :cpv ? -1 : -2
    if geom !== nothing
        C = laurent_coefficients(geom[1], geom[2], geom[3], collocation, θ, which, order)
        C !== nothing && return C
    end
    if kind === :bounded
        f0, _ = extrapolate(h; x0=zero(h), kw...) do ρ
            ρ ≤ eps(h) && return zero(probe)
            return Fr(ρ)
        end
        z = zero(probe)
        return (z, z, f0)
    elseif kind === :cpv
        return laurent_coefficients(Fr, h, Val(-1); kwargs...)
    else
        return laurent_coefficients(Fr, h, Val(-2); kwargs...)
    end
end

"""Ray Laurent of `F(ρ)=φρ` from a parent interpolant at `ρ=0` (`ξ=-1`)."""
function _surface_interp_C(fi, Ni, ρ, kind::Symbol)
    if kind === :bounded
        F0 = _barycentric_sum(Ni, fi)
        z = zero(F0)
        return (z, z, F0)
    elseif kind === :cpv
        return _interp_cascade(fi, Ni, ρ, -1)
    else
        return _interp_cascade(fi, Ni, ρ, -2)
    end
end

"""Radial integral of a fused pair along one polar ray, `laurent=:interp`."""
function _surface_interp_pair(Fr, ρmax, kg::Symbol, kh::Symbol, n::Integer)
    α = ρmax / 2
    ξ, w, Ni, δξ = _interp_rule(-1.0, n)
    ρ = α .* δξ
    samples = [Fr(ρi) for ρi in ρ]
    Fg = [s[1] for s in samples]
    Fh = [s[2] for s in samples]
    Cg = _surface_interp_C(Fg, Ni, ρ, kg)
    Ch = _surface_interp_C(Fh, Ni, ρ, kh)
    Ig = w[1] * _surface_reg(Fg[1], ρ[1], kg, Cg) * α
    Ih = w[1] * _surface_reg(Fh[1], ρ[1], kh, Ch) * α
    @inbounds for i in 2:length(w)
        dρ = w[i] * α
        Ig += _surface_reg(Fg[i], ρ[i], kg, Cg) * dρ
        Ih += _surface_reg(Fh[i], ρ[i], kh, Ch) * dρ
    end
    return Ig + _surface_analytic(ρmax, kg, Cg), Ih + _surface_analytic(ρmax, kh, Ch)
end

@inline function _surface_reg(F, ρ, kind::Symbol, C)
    F₋₂, F₋₁, F₀ = C
    if kind === :bounded
        return F .- F₀
    elseif kind === :cpv
        return F .- F₋₁ ./ ρ
    else
        return F .- F₋₂ ./ ρ^2 .- F₋₁ ./ ρ
    end
end

@inline function _surface_analytic(ρmax, kind::Symbol, C)
    F₋₂, F₋₁, F₀ = C
    if kind === :bounded
        return F₀ .* ρmax
    elseif kind === :cpv
        return F₋₁ .* log(ρmax)
    else
        return F₋₁ .* log(ρmax) .- F₋₂ ./ ρmax
    end
end

"""
    guiggiani_GH_surface(f, aξ, aη; order_G=0, order_H=-1, ...) -> (I_G, I_H)

On-element surface integral of a fused pair `f(ξ,η) → (φ_G, φ_H)` (kernel ×
shape × parent Jacobian, **no** polar `ρ`). Polar coordinates are centred at
`(aξ, aη)` on the parent square `[-1,1]²`; along each ray the radial
integrand is `F(ρ)=φ ρ`, whose Laurent tail is subtracted and restored
analytically (Guiggiani 1992):

| kernel | `φ` | `F=φρ` | `order` |
|--------|-----|--------|---------|
| 3-D `U∼1/R` | `∼1/ρ` | bounded | `0` |
| 3-D `T∼1/R²` | `∼1/ρ²` | CPV `1/ρ` | `-1` |
| hypersingular | `∼1/ρ³` | HFP `1/ρ²` | `-2` |

`laurent=:interp` (default) interpolates `F(ρ)` on each ray; `:richardson`
always extrapolates; `:auto` uses closed-form tensors when they exist.
"""
function guiggiani_GH_surface(f, aξ::Real, aη::Real;
        order_G::Integer=0,
        order_H::Integer=-1,
        qsi=nothing, w=nothing,
        nθ::Union{Int,Nothing}=nothing,
        h::Real=1e-3,
        props=nothing, poly=nothing, nodes=nothing, kwargs...)
    method, ninterp, rest = _strip_laurent_kwargs(kwargs)
    _, qsi, w, hh = _guiggiani_rule(0.0, qsi, w, h)
    aξ = float(aξ)
    aη = float(aη)
    kg = _surface_kind(order_G)
    kh = _surface_kind(order_H)
    kw = _laurent_extrapolate_kwargs(rest)
    geom = _guiggiani_geom(props, poly, nodes)
    method === :richardson && (geom = nothing)
    collocation = (aξ, aη)
    nθ = nθ === nothing ? max(8, length(qsi)) : Int(nθ)
    uθ, wθ = gausslegendre(nθ)

    ξp = aξ + max(hh, 1e-6)
    ηp = aη
    probe_g, probe_h = f(clamp(ξp, -1.0, 1.0), clamp(ηp, -1.0, 1.0))
    acc_g = zero(probe_g)
    acc_h = zero(probe_h)

    corners = ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0))
    for k in 1:4
        c1 = corners[k]
        c2 = corners[mod1(k + 1, 4)]
        θ1 = atan(c1[2] - aη, c1[1] - aξ)
        θ2 = atan(c2[2] - aη, c2[1] - aξ)
        dθ = θ2 - θ1
        dθ <= 0 && (dθ += 2π)
        θmid = θ1 + 0.5 * dθ
        θhalf = 0.5 * dθ
        for j in eachindex(uθ)
            θ = θmid + θhalf * uθ[j]
            wθj = wθ[j] * θhalf
            ρmax = _ray_to_square(aξ, aη, θ)
            ρmax ≤ eps(Float64) && continue
            cθ, sθ = cos(θ), sin(θ)
            Fr = let f = f, aξ = aξ, aη = aη, cθ = cθ, sθ = sθ
                ρ -> begin
                    ξ = clamp(aξ + ρ * cθ, -1.0, 1.0)
                    η = clamp(aη + ρ * sθ, -1.0, 1.0)
                    Fg, Fh = f(ξ, η)
                    return Fg .* ρ, Fh .* ρ
                end
            end
            if method === :interp
                Ig, Ih = _surface_interp_pair(Fr, ρmax, kg, kh, ninterp)
            else
                Cg = _surface_laurent(ρ -> Fr(ρ)[1], hh, kg, probe_g, kw, geom,
                    collocation, θ, :G; rest...)
                Ch = _surface_laurent(ρ -> Fr(ρ)[2], hh, kh, probe_h, kw, geom,
                    collocation, θ, :H; rest...)
                Ig = zero(probe_g)
                Ih = zero(probe_h)
                @inbounds for i in eachindex(qsi)
                    ρ = (qsi[i] + 1) * ρmax / 2
                    ρ ≤ eps(Float64) && continue
                    dρ = w[i] * ρmax / 2
                    Fg, Fh = Fr(ρ)
                    Ig += _surface_reg(Fg, ρ, kg, Cg) * dρ
                    Ih += _surface_reg(Fh, ρ, kh, Ch) * dρ
                end
                Ig += _surface_analytic(ρmax, kg, Cg)
                Ih += _surface_analytic(ρmax, kh, Ch)
            end
            acc_g += Ig * wθj
            acc_h += Ih * wθj
        end
    end
    return acc_g, acc_h
end
