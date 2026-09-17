# Local BEM (compact C¹ fundamental solution) with DIBEM domain integrals.
#
# Poisson  ∇²u = f  on Ω,  u = h₁ on Γ₁,  ∂n u = h₂ on Γ₂.
# Support of u_i* is B(y_i, r_i). Artificial circle/sphere integrals vanish.
# Volume terms: DIBEM RIM on ∂(Ω ∩ B) = clipped Γ plus interior circular arcs.
# ∫_{Ω^i} u and ∫_{Ω^i} f u_i* are local CPD (default PHS(3)+poly) on B(y_i, r_i).
# source=:global keeps DIBEM lumping of f.

export local_u_star, local_du_dr, local_du_dn, local_ball_volume
export radial_integral_local_ustar, radial_integral_local_one
export clip_element_to_ball, assemble_local_bem!, solve_local_bem!

# =============================================================================
# Compact kernel
# =============================================================================

"""Volume of the *full* ball of radius `r_i` (coefficient of the quadratic companion)."""
function local_ball_volume(r_i::Real; dim::Int=2)
    r = float(r_i)
    r > 0 || throw(ArgumentError("r_i must be positive"))
    dim == 2 && return π * r * r
    dim == 3 && return (4π / 3) * r^3
    throw(ArgumentError("dim must be 2 or 3"))
end

"""
    local_u_star(r, r_i; dim=2)

Compactly supported regularized Laplace kernel (`0` for `r > r_i`):

- 2D: `(1/2π) [-ln(r/r_i) + r²/(2 r_i²) - 1/2]₊`
- 3D: `(1/4π) [1/r + r²/(2 r_i³) - 3/(2 r_i)]₊`
"""
function local_u_star(r::Real, r_i::Real; dim::Int=2)
    ρ = float(r)
    ri = float(r_i)
    ri > 0 || throw(ArgumentError("r_i must be positive"))
    dim in (2, 3) || throw(ArgumentError("dim must be 2 or 3"))
    ρ > ri && return 0.0
    if dim == 2
        ρ < 1e-30 && return Inf
        return (-log(ρ / ri) + (ρ * ρ) / (2 * ri * ri) - 0.5) / (2π)
    else
        ρ < 1e-30 && return Inf
        return (1 / ρ + (ρ * ρ) / (2 * ri^3) - 1.5 / ri) / (4π)
    end
end

"""Radial derivative `∂u_i*/∂r` (`0` for `r > r_i`)."""
function local_du_dr(r::Real, r_i::Real; dim::Int=2)
    ρ = float(r)
    ri = float(r_i)
    ρ > ri && return 0.0
    if dim == 2
        ρ < 1e-30 && return -Inf
        return (-1 / ρ + ρ / (ri * ri)) / (2π)
    else
        ρ < 1e-30 && return -Inf
        return (-1 / (ρ * ρ) + ρ / (ri^3)) / (4π)
    end
end

"""
    local_du_dn(rvec, n, r_i; dim=length(rvec))

`∂u_i*/∂n = (∂u_i*/∂r) (r · n)/r`. Zero for `‖r‖ > r_i` and at the origin
(Cauchy principal value is taken in Guiggiani).
"""
function local_du_dn(rvec, n, r_i::Real; dim::Int=length(rvec))
    ρ = norm(rvec)
    ρ > float(r_i) && return 0.0
    ρ < 1e-30 && return 0.0
    rn = dot(rvec, n)
    ri = float(r_i)
    if dim == 2
        return (-rn / (ρ * ρ) + rn / (ri * ri)) / (2π)
    else
        return (-rn / (ρ^3) + rn / (ri^3)) / (4π)
    end
end

# =============================================================================
# Compact radial primitives  ∫_0^{min(R,r_i)} ψ(ρ) ρ^{d-1} dρ
# =============================================================================

"""
RIM primitive of the compact kernel `u_i*`.

`power=1` → ``∫_0^{s} u^* ρ^{d-1}\\,dρ`` (volume factor).
`power=2,3` → extra powers of ``ρ`` for moments ``∫ r u^*``, ``∫ r⊗r u^*``.
"""
function radial_integral_local_ustar(R::Real, r_i::Real; dim::Int=2, power::Int=1)
    s = min(float(R), float(r_i))
    s <= 0 && return 0.0
    ri = float(r_i)
    if dim == 3
        # u* = (1/4π)[1/ρ + ρ²/(2 ri³) − 3/(2 ri)]
        if power == 1
            # ∫ u* ρ² dρ = (1/4π)[s²/2 + s⁵/(10 ri³) − s³/(2 ri)]
            return (0.5 * s * s + (s^5) / (10 * ri^3) - (s^3) / (2 * ri)) / (4π)
        elseif power == 2
            # ∫ u* ρ³ dρ
            return (s^3 / 3 + s^6 / (12 * ri^3) - 3 * s^4 / (8 * ri)) / (4π)
        elseif power == 3
            # ∫ u* ρ⁴ dρ
            return (s^4 / 4 + s^7 / (14 * ri^3) - 3 * s^5 / (10 * ri)) / (4π)
        end
        throw(ArgumentError("power must be 1, 2, or 3"))
    end
    dim == 2 || throw(ArgumentError("dim must be 2 or 3"))
    ls = log(s / ri)
    if power == 1
        # (1/2π) [-(s²/2) ln(s/r_i) + s⁴/(8 r_i²)]
        return (-0.5 * s * s * ls + (s^4) / (8 * ri * ri)) / (2π)
    elseif power == 2
        # (1/2π) [-s³/3 ln + s⁵/(10 r_i²) - s³/18]
        return (-s^3 / 3 * ls + s^5 / (10 * ri * ri) - s^3 / 18) / (2π)
    elseif power == 3
        # (1/2π) [-s⁴/4 ln + s⁶/(12 r_i²) - s⁴/16]
        return (-s^4 / 4 * ls + s^6 / (12 * ri * ri) - s^4 / 16) / (2π)
    end
    throw(ArgumentError("power must be 1, 2, or 3"))
end

"""RIM primitive of the ball indicator (`ψ = 1` for `ρ ≤ r_i`)."""
function radial_integral_local_one(R::Real, r_i::Real; dim::Int=2)
    s = min(float(R), float(r_i))
    s <= 0 && return 0.0
    dim == 2 && return s * s / 2
    dim == 3 && return s^3 / 3
    throw(ArgumentError("dim must be 2 or 3"))
end

"""Callable kernel for [`integrate_element`](@ref): returns `(u_i*, ∂n u_i*)`."""
struct LocalKernel
    r_i::Float64
    dim::Int
end
function (K::LocalKernel)(_dad, r, n)
    return local_u_star(norm(r), K.r_i; dim=K.dim), local_du_dn(r, n, K.r_i; dim=K.dim)
end

# =============================================================================
# Support radius
# =============================================================================

"""Fill spacing from interior cloud and element size (skip clustered GL boundary pairs)."""
function _lbem_fill_spacing(dad)
    hs = Float64[]
    if dad.ni >= 2
        push!(hs, rbf_length_scale(collect(dad.internalNodes)))
    end
    if !isempty(dad.elements)
        push!(hs, median(el.Length for el in dad.elements))
    end
    h = isempty(hs) ? rbf_length_scale(all_points(dad)) : maximum(hs)
    h > 0 || throw(ArgumentError("local BEM radius collapsed (check internals)"))
    return h
end

function _lbem_radii(dad; radius=nothing, radius_factor::Real=2.0)
    if radius isa AbstractVector
        length(radius) == dad.nt || throw(DimensionMismatch("radius length"))
        any(r -> r <= 0, radius) && throw(ArgumentError("radii must be positive"))
        return collect(float.(radius))
    elseif radius === nothing
        r = float(radius_factor) * _lbem_fill_spacing(dad)
        return fill(r, dad.nt)
    else
        r = float(radius)
        r > 0 || throw(ArgumentError("radius must be positive"))
        return fill(r, dad.nt)
    end
end

# =============================================================================
# Compact DIBEM RIM on ∂(Ω ∩ B) = (Γ ∩ B) ∪ (Ω ∩ circle)
#
# The volume density is zero outside the ball, so the radial primitive is
# constant on rays that hit Γ beyond r_i. That constant × solid angle is
# exactly Ψ(r_i) Δθ on the interior circular arcs — cheaper and exact
# compared with looping the whole of Γ.
# =============================================================================

function _lbem_winding(dad, p, ηs, ws)
    poly = dad.element_type
    acc = 0.0
    @inbounds for el in dad.elements
        nodes = dad.Nodes[el.index]
        N, dN = shapefun(poly, ηs)
        pg = N * nodes
        dx = dN * nodes
        for q in eachindex(ηs)
            r = pg[q] - p
            R2 = dot(r, r)
            R2 < 1e-20 && continue
            Jv = dx[q]
            acc += (r[1] * Jv[2] - r[2] * Jv[1]) / R2 * ws[q]
        end
    end
    return acc
end

function _lbem_point_in_omega(dad, p, ηs, ws, orient::Float64)
    return orient * _lbem_winding(dad, p, ηs, ws) > π
end

function _unique_angles(θs; atol=1e-8)
    isempty(θs) && return Float64[]
    a = sort!(mod.(θs, 2π))
    out = Float64[a[1]]
    @inbounds for k in 2:length(a)
        a[k] - out[end] > atol && push!(out, a[k])
    end
    if length(out) > 1 && (out[1] + 2π - out[end]) <= atol
        pop!(out)
    end
    return out
end

function _lbem_elem_may_hit_ball(center, nodes, el, ri)
    lim = ri + el.Length
    @inbounds for k in eachindex(nodes)
        norm(center - nodes[k]) <= lim && return true
    end
    return false
end

function _lbem_collect_clips(dad, center, ri)
    poly = dad.element_type
    segs = Tuple{Element,Float64,Float64}[]
    hits = Float64[]
    tol = 1e-8 * max(ri, 1e-16)
    @inbounds for el in dad.elements
        nodes = dad.Nodes[el.index]
        _lbem_elem_may_hit_ball(center, nodes, el, ri) || continue
        for (ξa, ξb) in clip_element_to_ball(poly, nodes, center, ri)
            push!(segs, (el, ξa, ξb))
            for ξ in (ξa, ξb)
                x, _ = _lbem_curve_eval(poly, nodes, ξ)
                if abs(norm(x - center) - ri) <= tol
                    push!(hits, atan(x[2] - center[2], x[1] - center[1]))
                end
            end
        end
    end
    return segs, _unique_angles(hits)
end

function _lbem_interior_arc_contrib(dad, center, ri, hits, ηs, ws, orient)
    n = length(hits)
    z = 0.0
    n == 0 && return z, z, z, z, z, z, z, z, z, z, z
    dθ = Mx = My = Mxx = Mxy = Myy = 0.0
    Ux = Uy = Uxx = Uxy = Uyy = 0.0
    ri3 = ri^3 / 3
    ri4 = ri^4 / 4
    Ψ2 = radial_integral_local_ustar(ri, ri; dim=2, power=2)
    Ψ3 = radial_integral_local_ustar(ri, ri; dim=2, power=3)
    @inbounds for k in 1:n
        a = hits[k]
        b = k == n ? hits[1] + 2π : hits[k + 1]
        mid = 0.5 * (a + b)
        p = Point2D(center[1] + ri * cos(mid), center[2] + ri * sin(mid))
        if _lbem_point_in_omega(dad, p, ηs, ws, orient)
            dθ += b - a
            ds = sin(b) - sin(a)
            dc = -cos(b) + cos(a)
            Mx += ri3 * ds
            My += ri3 * dc
            Ux += Ψ2 * ds
            Uy += Ψ2 * dc
            s2b, s2a = sin(2b), sin(2a)
            c2b, c2a = cos(2b), cos(2a)
            Δ = b - a
            cxx = Δ / 2 + (s2b - s2a) / 4
            cyy = Δ / 2 - (s2b - s2a) / 4
            cxy = -(c2b - c2a) / 4
            Mxx += ri4 * cxx
            Myy += ri4 * cyy
            Mxy += ri4 * cxy
            Uxx += Ψ3 * cxx
            Uyy += Ψ3 * cyy
            Uxy += Ψ3 * cxy
        end
    end
    return dθ, Mx, My, Mxx, Mxy, Myy, Ux, Uy, Uxx, Uxy, Uyy
end

function _lbem_rim_segment!(IF, IDu, ID1, Mx, My, Mxx, Mxy, Myy,
        IDux, IDuy, IDuxx, IDuxy, IDuyy,
        dad, el, x, i, ri, rbf, ξa, ξb, ηs, ws)
    poly = dad.element_type
    nodes = dad.Nodes[el.index]
    nref = dad.Normal[el.index[1]]
    s = (ξb - ξa) / 2
    m = (ξa + ξb) / 2
    s <= 1e-16 && return nothing
    @inbounds for q in eachindex(ηs)
        ξ = s * ηs[q] + m
        N, dN = shapefun(poly, ξ)
        y = (N * nodes)[1]
        Jv = (dN * nodes)[1]
        J = norm(Jv)
        J < 1e-16 && continue
        n = tan2normal(Jv / J)
        n ⋅ nref < 0 && (n = -n)
        r = y - x
        R = norm(r)
        R < 1e-14 && continue
        nr = n ⋅ r
        wJ = ws[q] * s * J
        wJn = wJ * nr / (R * R)
        IF[i] += radial_integral(rbf, min(R, ri); dim=2) * wJn
        IDu[i] += radial_integral_local_ustar(R, ri; dim=2) * wJn
        ID1[i] += radial_integral_local_one(R, ri; dim=2) * wJn
        mom1 = nr * wJ / 3
        mom2 = nr * wJ / 4
        Mx[i] += mom1 * r[1]
        My[i] += mom1 * r[2]
        Mxx[i] += mom2 * r[1] * r[1]
        Mxy[i] += mom2 * r[1] * r[2]
        Myy[i] += mom2 * r[2] * r[2]
        Ψ2 = radial_integral_local_ustar(R, ri; dim=2, power=2)
        Ψ3 = radial_integral_local_ustar(R, ri; dim=2, power=3)
        IDux[i] += r[1] * Ψ2 / R * wJn
        IDuy[i] += r[2] * Ψ2 / R * wJn
        IDuxx[i] += r[1] * r[1] * Ψ3 / (R * R) * wJn
        IDuxy[i] += r[1] * r[2] * Ψ3 / (R * R) * wJn
        IDuyy[i] += r[2] * r[2] * Ψ3 / (R * R) * wJn
    end
    return nothing
end

function _lbem_rim_accumulate!(IF, IDu, ID1, Mx, My, Mxx, Mxy, Myy,
        IDux, IDuy, IDuxx, IDuxy, IDuyy,
        dad, rbf, radii; npg::Int=16)
    has_cache(dad, :qsi) || _init_quadrature!(dad, npg)
    ηs, ws = dad.qsi, dad.w
    dad.dimension == 2 || error("local BEM assembly is 2D only")
    p0 = dad.ni > 0 ? dad.internalNodes[1] : geometric_props(dad).centroid
    w0 = _lbem_winding(dad, p0, ηs, ws)
    orient = w0 < 0 ? -1.0 : 1.0

    @inbounds for i in 1:dad.nt
        x = point(dad, i)
        ri = radii[i]
        segs, hits = _lbem_collect_clips(dad, x, ri)
        for (el, ξa, ξb) in segs
            _lbem_rim_segment!(IF, IDu, ID1, Mx, My, Mxx, Mxy, Myy,
                IDux, IDuy, IDuxx, IDuxy, IDuyy,
                dad, el, x, i, ri, rbf, ξa, ξb, ηs, ws)
        end
        dUx = dUy = dUxx = dUxy = dUyy = 0.0
        if isempty(hits)
            dθ = isempty(segs) ? 2π : 0.0
            dMx = dMy = dMxx = dMxy = dMyy = 0.0
            if dθ == 2π
                dMxx = dMyy = π * ri^4 / 4
                Ψ3 = radial_integral_local_ustar(ri, ri; dim=2, power=3)
                dUxx = dUyy = π * Ψ3
            end
        else
            dθ, dMx, dMy, dMxx, dMxy, dMyy, dUx, dUy, dUxx, dUxy, dUyy =
                _lbem_interior_arc_contrib(dad, x, ri, hits, ηs, ws, orient)
        end
        if dθ != 0
            IF[i] += radial_integral(rbf, ri; dim=2) * dθ
            IDu[i] += radial_integral_local_ustar(ri, ri; dim=2) * dθ
            ID1[i] += radial_integral_local_one(ri, ri; dim=2) * dθ
        end
        Mx[i] += dMx
        My[i] += dMy
        Mxx[i] += dMxx
        Mxy[i] += dMxy
        Myy[i] += dMyy
        IDux[i] += dUx
        IDuy[i] += dUy
        IDuxx[i] += dUxx
        IDuxy[i] += dUxy
        IDuyy[i] += dUyy
    end
    return nothing
end

function _lbem_dibem_weights(dad, rbf, radii; npg::Int=16)
    nt = dad.nt
    pts = all_points(dad)
    IF = zeros(nt)
    IDu = zeros(nt)
    ID1 = zeros(nt)
    Mx = zeros(nt)
    My = zeros(nt)
    Mxx = zeros(nt)
    Mxy = zeros(nt)
    Myy = zeros(nt)
    IDux = zeros(nt)
    IDuy = zeros(nt)
    IDuxx = zeros(nt)
    IDuxy = zeros(nt)
    IDuyy = zeros(nt)
    _lbem_rim_accumulate!(IF, IDu, ID1, Mx, My, Mxx, Mxy, Myy,
        IDux, IDuy, IDuxx, IDuxy, IDuyy, dad, rbf, radii; npg=npg)

    F = zeros(nt, nt)
    @inbounds for j in 1:nt, i in 1:nt
        i == j && continue
        F[i, j] = rbf(norm(pts[i] - pts[j]))
    end
    _dibem_ridge_F!(F)
    c = F \ IF
    return (; F, c, IF, IDu, ID1, Mx, My, Mxx, Mxy, Myy,
        IDux, IDuy, IDuxx, IDuxy, IDuyy, pts)
end

# =============================================================================
# Local CPD (PHS + poly) quadrature of linear functionals on Ω ∩ B(y_i, r_i)
#   [F  P] [w]   [L(φ_j)]
#   [P' 0] [μ] = [L(p_k)]
# Default `PHS(3)` is poly_deg=2 (1, x, y, xy, x², y²), shifted and scaled by r_i.
# =============================================================================

_lbem_default_rbf() = PHS(3)

function _lbem_neighbour_ids(pts, i, ri)
    xi = pts[i]
    ids = Int[]
    @inbounds for j in eachindex(pts)
        norm(pts[j] - xi) <= ri + 1e-12 && push!(ids, j)
    end
    return ids
end

function _lbem_scaled_monomials(ξ, η, pdeg::Int)
    pdeg < 0 && return Float64[]
    pdeg == 0 && return [1.0]
    pdeg == 1 && return [1.0, ξ, η]
    return [1.0, ξ, η, ξ * η, ξ * ξ, η * η]
end

function _lbem_scaled_poly(pts, ids, xi, ri, pdeg::Int)
    n = length(ids)
    npoly = rbf_npoly(2, pdeg)
    P = zeros(n, npoly)
    invr = 1 / max(ri, 1e-14)
    @inbounds for k in 1:n
        d = pts[ids[k]] - xi
        p = _lbem_scaled_monomials(d[1] * invr, d[2] * invr, pdeg)
        for α in 1:npoly
            P[k, α] = p[α]
        end
    end
    return P
end

function _lbem_scaled_IP(pdeg::Int, ID1, Mx, My, Mxx, Mxy, Myy, ri)
    pdeg < 0 && return Float64[]
    invr = 1 / max(float(ri), 1e-14)
    pdeg == 0 && return [float(ID1)]
    pdeg == 1 && return [float(ID1), Mx * invr, My * invr]
    invr2 = invr * invr
    return [float(ID1), Mx * invr, My * invr, Mxy * invr2, Mxx * invr2, Myy * invr2]
end

function _lbem_local_cpd_K(pts, i, ri, rbf)
    ids = _lbem_neighbour_ids(pts, i, ri)
    n = length(ids)
    n < 3 && return nothing
    pdeg = min(max(poly_deg(rbf), -1), 2)
    while pdeg >= 0 && rbf_npoly(2, pdeg) > n
        pdeg -= 1
    end
    npoly = pdeg < 0 ? 0 : rbf_npoly(2, pdeg)
    F = zeros(n, n)
    @inbounds for k in 1:n, j in 1:k
        fij = rbf(norm(pts[ids[j]] - pts[ids[k]]))
        F[j, k] = fij
        F[k, j] = fij
    end
    _dibem_ridge_F!(F)
    if npoly == 0
        return (; ids, K=F, n, npoly, pdeg, P=zeros(n, 0))
    end
    P = _lbem_scaled_poly(pts, ids, pts[i], ri, pdeg)
    K = [F P; P' zeros(npoly, npoly)]
    return (; ids, K, n, npoly, pdeg, P)
end

function _lbem_cpd_quad_weights(sys, IF, IP)
    n, npoly = sys.n, sys.npoly
    length(IF) == n || return nothing
    rhs = npoly == 0 ? collect(float.(IF)) : vcat(float.(IF), float.(IP))
    length(rhs) == size(sys.K, 1) || return nothing
    coef = try
        sys.K \ rhs
    catch
        return nothing
    end
    w = coef[1:n]
    all(isfinite, w) || return nothing
    return w
end

function _lbem_interior_arc_intervals(dad, center, ri, hits, ηs, ws, orient)
    n = length(hits)
    n == 0 && return Tuple{Float64,Float64}[]
    out = Tuple{Float64,Float64}[]
    @inbounds for k in 1:n
        a = hits[k]
        b = k == n ? hits[1] + 2π : hits[k + 1]
        mid = 0.5 * (a + b)
        p = Point2D(center[1] + ri * cos(mid), center[2] + ri * sin(mid))
        if _lbem_point_in_omega(dad, p, ηs, ws, orient)
            push!(out, (a, b))
        end
    end
    return out
end

function _lbem_rim_phi_segment(dad, el, xj, rbf, ξa, ξb, ηs, ws)
    poly = dad.element_type
    nodes = dad.Nodes[el.index]
    nref = dad.Normal[el.index[1]]
    s = (ξb - ξa) / 2
    m = (ξa + ξb) / 2
    s <= 1e-16 && return 0.0
    acc = 0.0
    @inbounds for q in eachindex(ηs)
        ξ = s * ηs[q] + m
        N, dN = shapefun(poly, ξ)
        y = (N * nodes)[1]
        Jv = (dN * nodes)[1]
        J = norm(Jv)
        J < 1e-16 && continue
        n = tan2normal(Jv / J)
        n ⋅ nref < 0 && (n = -n)
        r = y - xj
        R2 = dot(r, r)
        R2 < 1e-20 && continue
        acc += radial_integral(rbf, sqrt(R2); dim=2) * (n ⋅ r) / R2 * ws[q] * s * J
    end
    return acc
end

function _lbem_rim_phi_arc(y, ri, xj, rbf, θa, θb, ηs, ws)
    s = (θb - θa) / 2
    m = (θa + θb) / 2
    s == 0 && return 0.0
    acc = 0.0
    @inbounds for q in eachindex(ηs)
        θ = s * ηs[q] + m
        n1, n2 = cos(θ), sin(θ)
        rj1 = y[1] + ri * n1 - xj[1]
        rj2 = y[2] + ri * n2 - xj[2]
        R2 = rj1 * rj1 + rj2 * rj2
        R2 < 1e-20 && continue
        nr = n1 * rj1 + n2 * rj2
        acc += radial_integral(rbf, sqrt(R2); dim=2) * nr / R2 * ri * ws[q] * s
    end
    return acc
end

"""`∫_{Ω^i} φ(|x − x_j|)` by divergence RIM on `∂(Ω ∩ B)`."""
function _lbem_rim_phi_omega_i(dad, y, ri, xj, rbf, segs, hits, ηs, ws, orient)
    acc = 0.0
    for (el, ξa, ξb) in segs
        acc += _lbem_rim_phi_segment(dad, el, xj, rbf, ξa, ξb, ηs, ws)
    end
    if isempty(hits)
        if isempty(segs)
            acc += _lbem_rim_phi_arc(y, ri, xj, rbf, 0.0, 2π, ηs, ws)
        end
    else
        for (a, b) in _lbem_interior_arc_intervals(dad, y, ri, hits, ηs, ws, orient)
            acc += _lbem_rim_phi_arc(y, ri, xj, rbf, a, b, ηs, ws)
        end
    end
    return acc
end

function _lbem_add_polar_sector!(xs, wts, y, ρmax, θa, θb, ηs, ws)
    sθ = (θb - θa) / 2
    mθ = (θa + θb) / 2
    sρ = ρmax / 2
    mρ = ρmax / 2
    (sθ == 0 || ρmax <= 0) && return nothing
    @inbounds for i in eachindex(ηs), j in eachindex(ηs)
        θ = sθ * ηs[i] + mθ
        ρ = sρ * ηs[j] + mρ
        w = ws[i] * ws[j] * sθ * sρ * ρ
        w < 1e-18 && continue
        push!(xs, Point2D(y[1] + ρ * cos(θ), y[2] + ρ * sin(θ)))
        push!(wts, w)
    end
    return nothing
end

function _lbem_add_seg_cone!(xs, wts, dad, el, y, ξa, ξb, ηs, ws)
    poly = dad.element_type
    nodes = dad.Nodes[el.index]
    s = (ξb - ξa) / 2
    m = (ξa + ξb) / 2
    s <= 1e-16 && return nothing
    @inbounds for i in eachindex(ηs), j in eachindex(ηs)
        ξ = s * ηs[i] + m
        t = (ηs[j] + 1) / 2
        N, dN = shapefun(poly, ξ)
        xel = (N * nodes)[1]
        Jv = (dN * nodes)[1]
        r = xel - y
        cross = abs(r[1] * Jv[2] - r[2] * Jv[1])
        w = ws[i] * ws[j] * s * 0.5 * t * cross
        w < 1e-18 && continue
        push!(xs, Point2D(y[1] + t * r[1], y[2] + t * r[2]))
        push!(wts, w)
    end
    return nothing
end

"""Cubature nodes/weights on `Ω ∩ B(y, r_i)`: polar pies on interior arcs + cones to clipped Γ."""
function _lbem_omega_quad_points(dad, y, ri, segs, hits, ηs, ws, orient)
    xs = Point2D[]
    wts = Float64[]
    if isempty(hits)
        if isempty(segs)
            _lbem_add_polar_sector!(xs, wts, y, ri, 0.0, 2π, ηs, ws)
        else
            for (el, ξa, ξb) in segs
                _lbem_add_seg_cone!(xs, wts, dad, el, y, ξa, ξb, ηs, ws)
            end
        end
    else
        for (a, b) in _lbem_interior_arc_intervals(dad, y, ri, hits, ηs, ws, orient)
            _lbem_add_polar_sector!(xs, wts, y, ri, a, b, ηs, ws)
        end
        for (el, ξa, ξb) in segs
            _lbem_add_seg_cone!(xs, wts, dad, el, y, ξa, ξb, ηs, ws)
        end
    end
    return xs, wts
end

function _lbem_cpd_avg_weights(sys, dad, pts, i, ri, rbf, ID1, Mx, My, Mxx, Mxy, Myy,
        segs, hits, ηs, ws, orient)
    n = sys.n
    IF = zeros(n)
    y = pts[i]
    @inbounds for k in 1:n
        IF[k] = _lbem_rim_phi_omega_i(dad, y, ri, pts[sys.ids[k]], rbf,
            segs, hits, ηs, ws, orient)
    end
    IP = _lbem_scaled_IP(sys.pdeg, ID1, Mx, My, Mxx, Mxy, Myy, ri)
    return _lbem_cpd_quad_weights(sys, IF, IP)
end

"""Local CPD weights for ``L(f)=∫_{Ω^i} f u_i^*``. `∫ φ_j u_i^*` by cubature; `∫ p u_i^*` by RIM."""
function _lbem_cpd_source_weights(sys, dad, pts, i, ri, rbf,
        IDu, IDux, IDuy, IDuxx, IDuxy, IDuyy,
        segs, hits, ηs, ws, orient)
    qx, qw = _lbem_omega_quad_points(dad, pts[i], ri, segs, hits, ηs, ws, orient)
    isempty(qx) && return nothing
    n = sys.n
    IF = zeros(n)
    y = pts[i]
    ids = sys.ids
    @inbounds for q in eachindex(qx)
        x = qx[q]
        ρ = norm(x - y)
        us = local_u_star(ρ, ri; dim=2)
        isfinite(us) || continue
        wv = qw[q] * us
        for k in 1:n
            IF[k] += wv * rbf(norm(x - pts[ids[k]]))
        end
    end
    IP = _lbem_scaled_IP(sys.pdeg, IDu, IDux, IDuy, IDuxx, IDuxy, IDuyy, ri)
    return _lbem_cpd_quad_weights(sys, IF, IP)
end

function _lbem_push_lumped!(I, J, V, i, pts, xi, ri, c, diag)
    rsum = 0.0
    @inbounds for j in eachindex(pts)
        j == i && continue
        norm(pts[j] - xi) > ri && continue
        push!(I, i); push!(J, j); push!(V, c[j])
        rsum += c[j]
    end
    push!(I, i); push!(J, i); push!(V, diag - rsum)
    return nothing
end

function _lbem_push_source_lumped!(I, J, V, i, pts, xi, ri, c, IDu; dim::Int=2)
    rf = 0.0
    @inbounds for j in eachindex(pts)
        j == i && continue
        rij = norm(pts[j] - xi)
        rij > ri && continue
        mf = local_u_star(rij, ri; dim=dim) * c[j]
        push!(I, i); push!(J, j); push!(V, mf)
        rf += mf
    end
    push!(I, i); push!(J, i); push!(V, IDu - rf)
    return nothing
end

function _lbem_sparse_M(dad, w, rbf, radii; dim::Int=2, npg::Int=16)
    pts, c = w.pts, w.c
    nt = length(pts)
    has_cache(dad, :qsi) || _init_quadrature!(dad, npg)
    ηs, ws = dad.qsi, dad.w
    p0 = dad.ni > 0 ? dad.internalNodes[1] : geometric_props(dad).centroid
    orient = (_lbem_winding(dad, p0, ηs, ws) < 0) ? -1.0 : 1.0
    Ig = Int[]; Jg = Int[]; Vg = Float64[]
    Il = Int[]; Jl = Int[]; Vl = Float64[]
    I1 = Int[]; J1 = Int[]; V1 = Float64[]
    sizehint!(Ig, 8nt); sizehint!(Il, 8nt); sizehint!(I1, 8nt)
    @inbounds for i in 1:nt
        ri = radii[i]
        xi = pts[i]
        _lbem_push_source_lumped!(Ig, Jg, Vg, i, pts, xi, ri, c, w.IDu[i]; dim=dim)

        segs, hits = _lbem_collect_clips(dad, xi, ri)
        sys = _lbem_local_cpd_K(pts, i, ri, rbf)
        if sys === nothing || sys.npoly < 1
            _lbem_push_lumped!(I1, J1, V1, i, pts, xi, ri, c, w.ID1[i])
            _lbem_push_source_lumped!(Il, Jl, Vl, i, pts, xi, ri, c, w.IDu[i]; dim=dim)
        else
            w1 = _lbem_cpd_avg_weights(sys, dad, pts, i, ri, rbf,
                w.ID1[i], w.Mx[i], w.My[i], w.Mxx[i], w.Mxy[i], w.Myy[i],
                segs, hits, ηs, ws, orient)
            if w1 === nothing
                _lbem_push_lumped!(I1, J1, V1, i, pts, xi, ri, c, w.ID1[i])
            else
                for (k, j) in enumerate(sys.ids)
                    push!(I1, i); push!(J1, j); push!(V1, w1[k])
                end
            end
            wf = _lbem_cpd_source_weights(sys, dad, pts, i, ri, rbf,
                w.IDu[i], w.IDux[i], w.IDuy[i], w.IDuxx[i], w.IDuxy[i], w.IDuyy[i],
                segs, hits, ηs, ws, orient)
            if wf === nothing
                _lbem_push_source_lumped!(Il, Jl, Vl, i, pts, xi, ri, c, w.IDu[i]; dim=dim)
            else
                for (k, j) in enumerate(sys.ids)
                    push!(Il, i); push!(Jl, j); push!(Vl, wf[k])
                end
            end
        end
    end
    Mg = sparse(Ig, Jg, Vg, nt, nt)
    Ml = sparse(Il, Jl, Vl, nt, nt)
    M1 = sparse(I1, J1, V1, nt, nt)
    return Mg, Ml, M1
end

# =============================================================================
# Γ ∩ B(y_i, r_i): clip each 2D edge to the local disk
# =============================================================================

@inline function _lbem_curve_eval(poly, nodes, ξ::Real)
    N, dN = shapefun(poly, ξ)
    return (N * nodes)[1], (dN * nodes)[1]
end

"""
    clip_element_to_ball(poly, nodes, center, r_i) -> Vector{NTuple{2,Float64}}

Parametric pieces of the 2D element (`ξ ∈ [-1,1]`) that lie in the closed
disk ``\\|x(ξ) - \\mathrm{center}\\| ≤ r_i``. Empty if the edge misses the ball.

Linear (2-node) edges use the exact chord–circle quadratic. Higher-order
curves: sample + Newton on ``\\|x(ξ)-c\\|^2 = r_i^2``.
"""
function clip_element_to_ball(poly, nodes, center, r_i::Real)
    ri = float(r_i)
    ri > 0 || return NTuple{2,Float64}[]
    if length(nodes) == 2
        return _clip_linear_to_ball(poly, nodes, center, ri)
    end
    return _clip_curve_to_ball(poly, nodes, center, ri)
end

function _clip_linear_to_ball(poly, nodes, center, ri::Float64)
    x0, _ = _lbem_curve_eval(poly, nodes, -1.0)
    x1, _ = _lbem_curve_eval(poly, nodes, 1.0)
    p0 = 0.5 * (x0 + x1)
    p1 = 0.5 * (x1 - x0)
    q = p0 - center
    A = dot(p1, p1)
    r2 = ri * ri
    if A < 1e-30
        return dot(q, q) <= r2 + 1e-14 ? [(-1.0, 1.0)] : NTuple{2,Float64}[]
    end
    B = 2 * dot(q, p1)
    C = dot(q, q) - r2
    Δ = B * B - 4 * A * C
    # A>0: inside is the interval between roots (if any)
    if Δ < -1e-16 * max(A, 1.0)^2
        return C <= 0 ? [(-1.0, 1.0)] : NTuple{2,Float64}[]
    end
    sΔ = sqrt(max(Δ, 0.0))
    inv2A = 0.5 / A
    ξm = (-B - sΔ) * inv2A
    ξp = (-B + sΔ) * inv2A
    if ξm > ξp
        ξm, ξp = ξp, ξm
    end
    a = max(ξm, -1.0)
    b = min(ξp, 1.0)
    return b > a + 1e-14 ? [(a, b)] : NTuple{2,Float64}[]
end

function _clip_curve_to_ball(poly, nodes, center, ri::Float64)
    r2 = ri * ri
    nsample = max(24, 8 * (degree(poly) + 1))
    ξg = range(-1.0, 1.0; length=nsample)
    roots = Float64[]
    function fval(ξ)
        x, _ = _lbem_curve_eval(poly, nodes, ξ)
        d = x - center
        return dot(d, d) - r2
    end
    fa = fval(ξg[1])
    abs(fa) <= 1e-14 * max(r2, 1.0) && push!(roots, ξg[1])
    @inbounds for k in 1:(nsample - 1)
        a = ξg[k]
        b = ξg[k + 1]
        fb = fval(b)
        if abs(fb) <= 1e-14 * max(r2, 1.0)
            push!(roots, b)
        elseif fa * fb < 0
            push!(roots, _lbem_refine_root(poly, nodes, center, r2, a, b, fa, fb))
        end
        fa = fb
    end
    isempty(roots) && return fa <= 0 ? [(-1.0, 1.0)] : NTuple{2,Float64}[]
    sort!(roots)
    uniq = Float64[-1.0]
    @inbounds for ξ in roots
        if ξ - uniq[end] > 1e-12
            push!(uniq, ξ)
        end
    end
    uniq[end] < 1.0 - 1e-12 && push!(uniq, 1.0)
    uniq[end] < 1.0 && (uniq[end] = 1.0)
    segs = NTuple{2,Float64}[]
    @inbounds for k in 1:(length(uniq) - 1)
        a, b = uniq[k], uniq[k + 1]
        b - a < 1e-12 && continue
        fval(0.5 * (a + b)) <= 0 && push!(segs, (a, b))
    end
    return segs
end

function _lbem_refine_root(poly, nodes, center, r2, a, b, fa, fb)
    lo, hi, flo, fhi = a, b, fa, fb
    ξ = lo - flo * (hi - lo) / (fhi - flo + eps())
    ξ = clamp(ξ, lo, hi)
    @inbounds for _ in 1:50
        x, dx = _lbem_curve_eval(poly, nodes, ξ)
        d = x - center
        f = dot(d, d) - r2
        abs(f) <= 1e-14 * max(r2, 1.0) && return ξ
        if flo * f <= 0
            hi, fhi = ξ, f
        else
            lo, flo = ξ, f
        end
        abs(hi - lo) < 1e-14 && return 0.5 * (lo + hi)
        fp = 2 * dot(d, dx)
        if abs(fp) > 1e-14
            ξn = ξ - f / fp
            if lo < ξn < hi
                ξ = ξn
                continue
            end
        end
        ξ = 0.5 * (lo + hi)
    end
    return 0.5 * (lo + hi)
end

function _lbem_integrate_segment!(h, g, dad, el, nodes, pf, K, ξa, ξb, qsi, w, nref, aξ, dist)
    poly = dad.element_type
    s = (ξb - ξa) / 2
    m = (ξa + ξb) / 2
    s <= 1e-16 && return nothing
    use_sinh = dist < 2 * el.Length && (ξa - 1e-10) <= aξ <= (ξb + 1e-10)
    if use_sinh
        a_loc = clamp((aξ - m) / s, -0.999999, 0.999999)
        Lseg = max(el.Length * s, eps())
        η, ww = nearfield_1d(a_loc, dist / Lseg; qsi=qsi, w=w)
    else
        η, ww = qsi, w
    end
    nn = length(nodes)
    @inbounds for q in eachindex(η)
        ξ = s * η[q] + m
        N, dN = shapefun(poly, ξ)
        x = (N * nodes)[1]
        dx = (dN * nodes)[1]
        J = norm(dx)
        J < 1e-16 && continue
        n = tan2normal(dx / J)
        n ⋅ nref < 0 && (n = -n)
        U, T = K(dad, x - pf, n)
        wJ = ww[q] * s * J
        for j in 1:nn
            Nj = N[1, j] * wJ
            h[j] += T * Nj
            g[j] += U * Nj
        end
    end
    return nothing
end

function _assemble_local_HG!(dad, radii; npg::Int=16)
    _init_quadrature!(dad, npg)
    nt, n = dad.nt, dad.n
    dim = dad.dimension
    H = zeros(nt, nt)
    G = zeros(nt, n)
    elems = dad.elements
    poly = dad.element_type
    qsi, wgt = dad.qsi, dad.w
    for i in 1:nt
        pf = point(dad, i)
        ri = radii[i]
        K = LocalKernel(ri, dim)
        @inbounds for el in elems
            xj = dad.Nodes[el.index]
            _lbem_elem_may_hit_ball(pf, xj, el, ri) || continue
            segs = clip_element_to_ball(poly, xj, pf, ri)
            isempty(segs) && continue
            nn = length(el)
            hloc = zeros(nn)
            gloc = zeros(nn)
            full = length(segs) == 1 && segs[1][1] <= -1 + 1e-12 && segs[1][2] >= 1 - 1e-12
            if full
                integrate_element(dad, el, xj, pf, hloc, gloc, K)
            else
                aξ, _, dist = closest_point_1d(poly, xj, pf; ξ0=_seed_1d(poly, xj, pf))
                nref = dad.Normal[el.index[1]]
                for (ξa, ξb) in segs
                    _lbem_integrate_segment!(hloc, gloc, dad, el, xj, pf, K, ξa, ξb,
                        qsi, wgt, nref, aξ, dist)
                end
            end
            for (k, j) in enumerate(el.index)
                H[i, j] += hloc[k]
                G[i, j] += gloc[k]
            end
        end
    end
    return H, G
end

# =============================================================================
# Mixed BC (paper variables: unknown ∂n u on Dirichlet, u elsewhere)
# =============================================================================

function _lbem_mixed_system(dad, Au, G, rhs_f)
    n, nt = dad.n, dad.nt
    k = float(dad.properties.k)
    A = Matrix{Float64}(Au)
    b = Vector{Float64}(rhs_f)
    @inbounds for j in 1:n
        if dad.BC[j] == 0
            # Dirichlet: u known, unknown is ∂n u. A_q = -G.
            uknown = dad.BV[j]
            b .-= view(Au, :, j) .* uknown
            A[:, j] .= .-view(G, :, j)
        else
            # Neumann: package q = -k ∂n u is known
            dun = -dad.BV[j] / k
            b .+= view(G, :, j) .* dun
        end
    end
    return A, b
end

function _lbem_scatter_sol!(dad, x)
    n, nt = dad.n, dad.nt
    k = float(dad.properties.k)
    T = zeros(nt)
    qn = zeros(n)   # ∂n u
    @inbounds for j in 1:n
        if dad.BC[j] == 0
            T[j] = dad.BV[j]
            qn[j] = x[j]
        else
            T[j] = x[j]
            qn[j] = -dad.BV[j] / k
        end
    end
    @inbounds for j in (n + 1):nt
        T[j] = x[j]
    end
    q = -k .* qn
    set_cache!(dad; T=T, q=q)
    return T
end

# =============================================================================
# 3D: compact RIM on full Γ (primitive min(R,r_i)) + cone cubature of Ω ∩ B
# =============================================================================
#
# Volume density is zero outside the ball, so ∫_{Ω∩B} ψ = ∫_Γ Ψ(min(R,r_i)) (n·r)/R³ dΓ
# (no spherical-cap geometry). Cubature of Ω ∩ B: radial cones from y_i through
# every surface Gauss point, ρ ∈ [0, min(R, r_i)] (convex / star-shaped Ω).

function _lbem_aabb_hits_ball(nodes, center, ri)
    dim = length(center)
    d2 = 0.0
    @inbounds for d in 1:dim
        lo = hi = nodes[1][d]
        for k in 2:length(nodes)
            v = nodes[k][d]
            lo = min(lo, v)
            hi = max(hi, v)
        end
        c = center[d]
        if c < lo
            d2 += (lo - c)^2
        elseif c > hi
            d2 += (c - hi)^2
        end
    end
    return d2 <= (float(ri) + 1e-12)^2
end

function _lbem_scaled_monomials3(ξ, η, ζ, pdeg::Int)
    pdeg < 0 && return Float64[]
    pdeg == 0 && return [1.0]
    pdeg == 1 && return [1.0, ξ, η, ζ]
    return [1.0, ξ, η, ζ, ξ * ξ, ξ * η, ξ * ζ, η * η, η * ζ, ζ * ζ]
end

function _lbem_scaled_poly3(pts, ids, xi, ri, pdeg::Int)
    n = length(ids)
    npoly = rbf_npoly(3, pdeg)
    P = zeros(n, npoly)
    invr = 1 / max(ri, 1e-14)
    @inbounds for k in 1:n
        d = pts[ids[k]] - xi
        p = _lbem_scaled_monomials3(d[1] * invr, d[2] * invr, d[3] * invr, pdeg)
        for α in 1:npoly
            P[k, α] = p[α]
        end
    end
    return P
end

function _lbem_local_cpd_K3(pts, i, ri, rbf)
    ids = _lbem_neighbour_ids(pts, i, ri)
    n = length(ids)
    n < 4 && return nothing
    pdeg = min(max(poly_deg(rbf), -1), 2)
    while pdeg >= 0 && rbf_npoly(3, pdeg) > n
        pdeg -= 1
    end
    npoly = pdeg < 0 ? 0 : rbf_npoly(3, pdeg)
    F = zeros(n, n)
    @inbounds for k in 1:n, j in 1:k
        fij = rbf(norm(pts[ids[j]] - pts[ids[k]]))
        F[j, k] = fij
        F[k, j] = fij
    end
    _dibem_ridge_F!(F)
    if npoly == 0
        return (; ids, K=F, n, npoly, pdeg, P=zeros(n, 0))
    end
    P = _lbem_scaled_poly3(pts, ids, pts[i], ri, pdeg)
    K = [F P; P' zeros(npoly, npoly)]
    return (; ids, K, n, npoly, pdeg, P)
end

function _lbem_scaled_IP3(pdeg::Int, ID1, Mx, My, Mz, Mxx, Mxy, Mxz, Myy, Myz, Mzz, ri)
    pdeg < 0 && return Float64[]
    invr = 1 / max(float(ri), 1e-14)
    pdeg == 0 && return [float(ID1)]
    pdeg == 1 && return [float(ID1), Mx * invr, My * invr, Mz * invr]
    invr2 = invr * invr
    return [float(ID1), Mx * invr, My * invr, Mz * invr,
        Mxx * invr2, Mxy * invr2, Mxz * invr2, Myy * invr2, Myz * invr2, Mzz * invr2]
end

"""Cone cubature of `Ω ∩ B(y, r_i)`: surface Gauss × radial GL on `[0, min(R,r_i)]`."""
function _lbem_omega_quad_points_3d(dad, y, ri, ηs, ws; nρ::Int=6)
    xs = Point3D[]
    wts = Float64[]
    ρη, ρw = gausslegendre(nρ)
    _rim_foreach(dad, y; ηs=ηs, ws=ws) do wJn, R, e, _
        s = min(R, ri)
        s <= 1e-16 && return
        @inbounds for k in eachindex(ρη)
            ρ = (ρη[k] + 1) / 2 * s
            wρ = ρw[k] * s / 2
            wvol = wJn * wρ * ρ * ρ
            abs(wvol) < 1e-18 && continue
            push!(xs, y + ρ * e)
            push!(wts, wvol)
        end
    end
    return xs, wts
end

function _lbem_dibem_weights_3d(dad, rbf, radii; npg::Int=12)
    nt = dad.nt
    pts = all_points(dad)
    IF = zeros(nt)
    IDu = zeros(nt)
    ID1 = zeros(nt)
    Mx = zeros(nt); My = zeros(nt); Mz = zeros(nt)
    Mxx = zeros(nt); Myy = zeros(nt); Mzz = zeros(nt)
    Mxy = zeros(nt); Mxz = zeros(nt); Myz = zeros(nt)
    IDux = zeros(nt); IDuy = zeros(nt); IDuz = zeros(nt)
    IDuxx = zeros(nt); IDuyy = zeros(nt); IDuzz = zeros(nt)
    IDuxy = zeros(nt); IDuxz = zeros(nt); IDuyz = zeros(nt)
    has_cache(dad, :qsi) || _init_quadrature!(dad, npg)
    ηs, ws = dad.qsi, dad.w
    geos = _rim_build_elements(dad, ηs, ws)
    @inbounds for i in 1:nt
        x = point(dad, i)
        ri = radii[i]
        _rim_foreach(dad, x; geos=geos) do wJn, R, e, _
            s = min(R, ri)
            s <= 0 && return
            IF[i] += radial_integral(rbf, s; dim=3) * wJn
            IDu[i] += radial_integral_local_ustar(R, ri; dim=3, power=1) * wJn
            ID1[i] += (s^3 / 3) * wJn
            s4 = s^4 / 4
            s5 = s^5 / 5
            Mx[i] += wJn * e[1] * s4
            My[i] += wJn * e[2] * s4
            Mz[i] += wJn * e[3] * s4
            Mxx[i] += wJn * e[1] * e[1] * s5
            Myy[i] += wJn * e[2] * e[2] * s5
            Mzz[i] += wJn * e[3] * e[3] * s5
            Mxy[i] += wJn * e[1] * e[2] * s5
            Mxz[i] += wJn * e[1] * e[3] * s5
            Myz[i] += wJn * e[2] * e[3] * s5
            Ψ2 = radial_integral_local_ustar(R, ri; dim=3, power=2)
            Ψ3 = radial_integral_local_ustar(R, ri; dim=3, power=3)
            IDux[i] += wJn * e[1] * Ψ2
            IDuy[i] += wJn * e[2] * Ψ2
            IDuz[i] += wJn * e[3] * Ψ2
            IDuxx[i] += wJn * e[1] * e[1] * Ψ3
            IDuyy[i] += wJn * e[2] * e[2] * Ψ3
            IDuzz[i] += wJn * e[3] * e[3] * Ψ3
            IDuxy[i] += wJn * e[1] * e[2] * Ψ3
            IDuxz[i] += wJn * e[1] * e[3] * Ψ3
            IDuyz[i] += wJn * e[2] * e[3] * Ψ3
        end
    end
    F = zeros(nt, nt)
    @inbounds for j in 1:nt, i in 1:nt
        i == j && continue
        F[i, j] = rbf(norm(pts[i] - pts[j]))
    end
    _dibem_ridge_F!(F)
    c = F \ IF
    return (; F, c, IF, IDu, ID1, Mx, My, Mz, Mxx, Myy, Mzz, Mxy, Mxz, Myz,
        IDux, IDuy, IDuz, IDuxx, IDuyy, IDuzz, IDuxy, IDuxz, IDuyz, pts)
end

function _lbem_cpd_from_quad_3d(sys, qx, qw, pts, i, ri, rbf, IP; uweight::Bool=false)
    n = sys.n
    IF = zeros(n)
    y = pts[i]
    ids = sys.ids
    @inbounds for q in eachindex(qx)
        x = qx[q]
        ρ = norm(x - y)
        wv = qw[q]
        if uweight
            us = local_u_star(ρ, ri; dim=3)
            isfinite(us) || continue
            wv *= us
        end
        abs(wv) < 1e-18 && continue
        for k in 1:n
            IF[k] += wv * rbf(norm(x - pts[ids[k]]))
        end
    end
    return _lbem_cpd_quad_weights(sys, IF, IP)
end

function _lbem_sparse_M_3d(dad, w, rbf, radii; npg::Int=12)
    pts, c = w.pts, w.c
    nt = length(pts)
    has_cache(dad, :qsi) || _init_quadrature!(dad, npg)
    ηs, ws = dad.qsi, dad.w
    Ig = Int[]; Jg = Int[]; Vg = Float64[]
    Il = Int[]; Jl = Int[]; Vl = Float64[]
    I1 = Int[]; J1 = Int[]; V1 = Float64[]
    sizehint!(Ig, 8nt); sizehint!(Il, 8nt); sizehint!(I1, 8nt)
    @inbounds for i in 1:nt
        ri = radii[i]
        xi = pts[i]
        _lbem_push_source_lumped!(Ig, Jg, Vg, i, pts, xi, ri, c, w.IDu[i]; dim=3)
        sys = _lbem_local_cpd_K3(pts, i, ri, rbf)
        if sys === nothing || sys.npoly < 1
            _lbem_push_lumped!(I1, J1, V1, i, pts, xi, ri, c, w.ID1[i])
            _lbem_push_source_lumped!(Il, Jl, Vl, i, pts, xi, ri, c, w.IDu[i]; dim=3)
            continue
        end
        qx, qw = _lbem_omega_quad_points_3d(dad, xi, ri, ηs, ws)
        IP1 = _lbem_scaled_IP3(sys.pdeg, w.ID1[i], w.Mx[i], w.My[i], w.Mz[i],
            w.Mxx[i], w.Mxy[i], w.Mxz[i], w.Myy[i], w.Myz[i], w.Mzz[i], ri)
        w1 = isempty(qx) ? nothing : _lbem_cpd_from_quad_3d(sys, qx, qw, pts, i, ri, rbf, IP1)
        if w1 === nothing
            _lbem_push_lumped!(I1, J1, V1, i, pts, xi, ri, c, w.ID1[i])
        else
            for (k, j) in enumerate(sys.ids)
                push!(I1, i); push!(J1, j); push!(V1, w1[k])
            end
        end
        IPu = _lbem_scaled_IP3(sys.pdeg, w.IDu[i], w.IDux[i], w.IDuy[i], w.IDuz[i],
            w.IDuxx[i], w.IDuxy[i], w.IDuxz[i], w.IDuyy[i], w.IDuyz[i], w.IDuzz[i], ri)
        wf = isempty(qx) ? nothing :
            _lbem_cpd_from_quad_3d(sys, qx, qw, pts, i, ri, rbf, IPu; uweight=true)
        if wf === nothing
            _lbem_push_source_lumped!(Il, Jl, Vl, i, pts, xi, ri, c, w.IDu[i]; dim=3)
        else
            for (k, j) in enumerate(sys.ids)
                push!(Il, i); push!(Jl, j); push!(Vl, wf[k])
            end
        end
    end
    Mg = sparse(Ig, Jg, Vg, nt, nt)
    Ml = sparse(Il, Jl, Vl, nt, nt)
    M1 = sparse(I1, J1, V1, nt, nt)
    return Mg, Ml, M1
end

function _assemble_local_HG_3d!(dad, radii; npg::Int=12)
    _init_quadrature!(dad, npg)
    nt, n = dad.nt, dad.n
    H = zeros(nt, nt)
    G = zeros(nt, n)
    elems = dad.elements
    for i in 1:nt
        pf = point(dad, i)
        ri = radii[i]
        K = LocalKernel(ri, 3)
        @inbounds for el in elems
            xj = dad.Nodes[el.index]
            _lbem_aabb_hits_ball(xj, pf, ri) || continue
            nn = length(el)
            hloc = zeros(nn)
            gloc = zeros(nn)
            integrate_element(dad, el, xj, pf, hloc, gloc, K)
            for (k, j) in enumerate(el.index)
                H[i, j] += hloc[k]
                G[i, j] += gloc[k]
            end
        end
    end
    return H, G
end

function _assemble_local_bem_3d!(dad, radii, basis, npg, source)
    v0 = [local_ball_volume(r; dim=3) for r in radii]
    w = _lbem_dibem_weights_3d(dad, basis, radii; npg=npg)
    Mg, Ml, M1 = _lbem_sparse_M_3d(dad, w, basis, radii; npg=npg)
    Mf = source === :local ? Ml : Mg
    H, G = _assemble_local_HG_3d!(dad, radii; npg=npg)
    Au = Matrix(H)
    M1d = Matrix(M1)
    @inbounds for i in 1:dad.nt
        invv = 1 / v0[i]
        for j in 1:dad.nt
            Au[i, j] -= M1d[i, j] * invv
        end
        Au[i, i] = 0.0
        Au[i, i] = -sum(view(Au, i, :))
    end
    set_cache!(dad; H=sparse(H), G=sparse(G), M=Mf,
        lbem_M1=M1, lbem_M_local=Ml, lbem_M_global=Mg,
        lbem_Au=Au, lbem_radius=radii, lbem_v0=v0,
        dibem_c=w.c, dibem_F=w.F, dibem_IF=w.IF, dibem_ID=w.IDu,
        lbem_ID1=w.ID1, dibem_rbf=basis, lbem_method=:local, lbem_source=source)
    return dad
end

# =============================================================================
# Public API
# =============================================================================

"""
    assemble_local_bem!(dad; radius=nothing, radius_factor=2.0, rbf=nothing,
                        npg=16, source=:local)

Assemble local BEM operators on `dad` (2D or 3D Laplace, internals required).

Caches:
- `H`, `G` — compact double / single layer on `Γ ∩ B(y_i, r_i)`
- `M` — source operator `∫_{Ω^i} f u_i*` (`source=:local` CPD or `:global` DIBEM lumping)
- `lbem_M_local`, `lbem_M_global` — both source operators
- extras `lbem_M1`, `lbem_radius`, `lbem_v0`, `dibem_c`

Default interpolant is `PHS(3)` (poly_deg=2). `M1` and local `M` are per-ball CPD.
"""
function assemble_local_bem!(dad::BEMdata{<:Laplace};
        radius=nothing, radius_factor::Real=2.0,
        rbf=nothing, npg::Int=16, source::Symbol=:local)
    source in (:local, :global) || throw(ArgumentError("source must be :local or :global"))
    dad.dimension in (2, 3) || error("assemble_local_bem! is 2D or 3D")
    dad.ni > 0 || error("local BEM needs internal nodes (pontointerno=true or set_internal_nodes!)")
    radii = _lbem_radii(dad; radius=radius, radius_factor=radius_factor)
    basis = rbf === nothing ? _lbem_default_rbf() : rbf
    if dad.dimension == 3
        return _assemble_local_bem_3d!(dad, radii, basis, npg, source)
    end
    v0 = [local_ball_volume(r; dim=2) for r in radii]

    w = _lbem_dibem_weights(dad, basis, radii; npg=npg)
    Mg, Ml, M1 = _lbem_sparse_M(dad, w, basis, radii; dim=2, npg=npg)
    Mf = source === :local ? Ml : Mg
    H, G = _assemble_local_HG!(dad, radii; npg=npg)

    # A_u u - G (∂n u) = -M f    with A_u = H - M1 ./ v0  and row-sum free term
    # (Green: ∫ f u_i* enters with a minus; Laplace f=0 is unchanged)
    Au = Matrix(H)
    M1d = Matrix(M1)
    @inbounds for i in 1:dad.nt
        invv = 1 / v0[i]
        for j in 1:dad.nt
            Au[i, j] -= M1d[i, j] * invv
        end
        Au[i, i] = 0.0
        Au[i, i] = -sum(view(Au, i, :))
    end

    set_cache!(dad; H=sparse(H), G=sparse(G), M=Mf,
        lbem_M1=M1, lbem_M_local=Ml, lbem_M_global=Mg,
        lbem_Au=Au, lbem_radius=radii, lbem_v0=v0,
        dibem_c=w.c, dibem_F=w.F, dibem_IF=w.IF, dibem_ID=w.IDu,
        lbem_ID1=w.ID1, lbem_Mx=w.Mx, lbem_My=w.My,
        lbem_Mxx=w.Mxx, lbem_Mxy=w.Mxy, lbem_Myy=w.Myy,
        lbem_IDux=w.IDux, lbem_IDuy=w.IDuy,
        dibem_rbf=basis, lbem_method=:local, lbem_source=source)
    return dad
end

"""
    solve_local_bem!(dad, f=0; radius=nothing, radius_factor=2.0, rbf=nothing,
                     npg=16, source=:local, rebuild=false)

Solve the local BEM system. `f` is a number, `f(x)` function, or nodal vector
on `all_points(dad)`. Writes `dad.T` (boundary + interior) and `dad.q`
(`q = -k ∂n u` on the boundary).
"""
function solve_local_bem!(dad::BEMdata{<:Laplace}, f=0;
        radius=nothing, radius_factor::Real=2.0,
        rbf=nothing, npg::Int=16, source::Symbol=:local, rebuild::Bool=false)
    if rebuild || !has_cache(dad, :lbem_Au)
        assemble_local_bem!(dad; radius=radius, radius_factor=radius_factor,
            rbf=rbf, npg=npg, source=source)
    elseif has_cache(dad, :lbem_M_local) && has_cache(dad, :lbem_M_global)
        Mf = source === :local ? dad.lbem_M_local : dad.lbem_M_global
        set_cache!(dad; M=Mf, lbem_source=source)
    end
    fv = _eval_field(f, all_points(dad))
    rhs = Vector(-(dad.M * fv))
    A, b = _lbem_mixed_system(dad, dad.lbem_Au, Matrix(dad.G), rhs)
    set_cache!(dad; A=A, b=b)
    x = bem_linsolve(A, b)
    return _lbem_scatter_sol!(dad, x)
end
