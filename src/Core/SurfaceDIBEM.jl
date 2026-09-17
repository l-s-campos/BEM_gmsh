# Surface DIBEM: 3-D face integrals via PHS3+poly on the parent square.
# Loeffler product interpolant of g = N J K at nodes + edge Gauss;
# spike recovered by analytic radial ID (constant-kernel identity).

"""Parent-edge Gauss + vertices, corners unique. Returns (ξ, η) in [-1,1]²."""
function _surf_dibem_centers(nedge::Int)
    u, _ = gausslegendre(nedge)
    ξ = Float64[]
    η = Float64[]
    seen = Set{Tuple{Int,Int}}()
    function pushc!(a, b)
        key = (round(Int, a * 1e9), round(Int, b * 1e9))
        key in seen && return
        push!(seen, key)
        push!(ξ, a)
        push!(η, b)
        return nothing
    end
    for (a, b) in ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0))
        pushc!(a, b)
    end
    @inbounds for s in u
        abs(s) >= 1 - 1e-14 && continue
        pushc!(s, -1.0)
        pushc!(1.0, s)
        pushc!(s, 1.0)
        pushc!(-1.0, s)
    end
    return ξ, η
end

"""PHS3 `∫_0^R φ ρ dρ = R^5/5`; 2-D RIM on the parent square → four edges."""
function _surf_dibem_IF_phs3(cx::Float64, cy::Float64; n::Int=12)
    u, w = gausslegendre(n)
    acc = 0.0
    # CCW: start, Δ=(end-start), parent outward n
    edges = (
        (-1.0, -1.0, 2.0, 0.0, 0.0, -1.0),
        (1.0, -1.0, 0.0, 2.0, 1.0, 0.0),
        (1.0, 1.0, -2.0, 0.0, 0.0, 1.0),
        (-1.0, 1.0, 0.0, -2.0, -1.0, 0.0),
    )
    @inbounds for (ξ0, η0, dξ, dη, nx, ny) in edges
        ds = hypot(dξ, dη) / 2
        for k in eachindex(u)
            s = u[k]
            ξ = ξ0 + dξ * (s + 1) / 2
            η = η0 + dη * (s + 1) / 2
            dx = ξ - cx
            dy = η - cy
            R = hypot(dx, dy)
            R < 1e-14 && continue
            acc += w[k] * ds * (R^3 / 5) * (nx * dx + ny * dy)
        end
    end
    return acc
end

function _surf_dibem_weights(ξ::Vector{Float64}, η::Vector{Float64};
        nrim::Int=12, ridge::Float64=1e-12)
    n = length(ξ)
    F = zeros(n, n)
    @inbounds for j in 1:n, i in 1:j
        r = hypot(ξ[i] - ξ[j], η[i] - η[j])
        fij = r^3
        F[i, j] = fij
        F[j, i] = fij
    end
    ε = ridge * (sum(abs, F) / max(n * (n - 1), 1) + 1)
    @inbounds for i in 1:n
        F[i, i] += ε
    end
    IF = [_surf_dibem_IF_phs3(ξ[i], η[i]; n=nrim) for i in 1:n]
    # parent poly 1, ξ, η
    P = ones(n, 3)
    @inbounds for i in 1:n
        P[i, 2] = ξ[i]
        P[i, 3] = η[i]
    end
    IP = [4.0, 0.0, 0.0]
    K = [F P; P' zeros(3, 3)]
    coef = K \ [IF; IP]
    return coef[1:n]
end

function _surf_dibem_cache!(dad, nedge::Int)
    has_cache(dad, :surf_dibem) && dad.surf_dibem[1] == nedge && return dad.surf_dibem
    ξ, η = _surf_dibem_centers(nedge)
    c = _surf_dibem_weights(ξ, η; nrim=max(nedge, 12))
    tup = (nedge, ξ, η, c)
    set_cache!(dad; surf_dibem=tup)
    return tup
end

"""Linearized parent distance: ``R = h\\sqrt{(ξ-a)^2+(η-b)^2+d_p^2}``."""
function _id_radial_parent(a, b, d_par, k::Int; nθ::Int=16)
    u, w = gausslegendre(nθ)
    acc = 0.0
    corners = ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0))
    θs = Float64[]
    for cr in corners
        dx = cr[1] - a
        dy = cr[2] - b
        hypot(dx, dy) < 1e-14 && continue
        push!(θs, atan(dy, dx))
    end
    isempty(θs) && return 0.0
    sort!(θs)
    push!(θs, θs[1] + 2π)
    dp = abs(d_par)
    @inbounds for s in 1:(length(θs) - 1)
        dθ = θs[s + 1] - θs[s]
        dθ < 1e-14 && continue
        θmid = θs[s] + 0.5 * dθ
        _ray_to_square(a, b, θmid) <= 1e-14 && continue
        θh = 0.5 * dθ
        for j in eachindex(u)
            θ = θmid + θh * u[j]
            ρm = _ray_to_square(a, b, θ)
            ρm <= 1e-14 && continue
            hyp = hypot(ρm, dp)
            Iρ = if k == 1
                hyp - dp
            elseif k == 3
                dp < 1e-16 ? 0.0 : (1 / dp - 1 / hyp)
            else
                0.0
            end
            acc += w[j] * θh * Iρ
        end
    end
    return acc
end

function _surf_at(poly, nodes, ξ, η)
    L, Lξ, Lη = shapefun2D_points(poly, [ξ], [η])
    pg = (L * nodes)[1]
    tξ = (Lξ * nodes)[1]
    tη = (Lη * nodes)[1]
    Jv = cross(tξ, tη)
    J = norm(Jv)
    nrm = J > 1e-16 ? Jv / J : zero(pg)
    return pg, J, nrm, vec(L)
end

"""Fill local `h,g` for one 3-D face.

Product interpolant of the full integrand `g = N J K` at vertices + edge
Gauss (PHS3 + linear on the parent square). The interior spike is recovered
by the 2-D DIBEM diagonal analog

```
I_j = c · g_j + N_j(a) (ID − ∑_k c · g_k)
```

so `∑_j I_j = ID`, with `ID` the analytic polar integral of the leading
Laplace kernel (`1/R`, `(r·n)/R³`) on the linearized parent metric.
"""
function integrate_element_dibem!(h, g, dad, elem, nodes, pf::Point3D, f=fundamental)
    dad.properties isa Laplace || throw(ArgumentError(
        "surface DIBEM v1 supports Laplace only (got $(typeof(dad.properties)))"))
    h isa AbstractVector || throw(ArgumentError(
        "surface DIBEM v1 is scalar (got $(typeof(h)))"))
    poly = dad.element_type
    nedge = has_cache(dad, :qsi) ? length(dad.qsi) : 8
    _, ξc, ηc, cw = _surf_dibem_cache!(dad, nedge)
    aξ, aη, _, dist = closest_point_2d(poly, nodes, pf; ξ0=_seed_2d(poly, nodes, pf))
    pg0, J0, n0, Na = _surf_at(poly, nodes, aξ, aη)
    hsc = sqrt(max(J0, 1e-30))
    d_par = dist / hsc
    kcond = dad.properties.k
    inv4πk = 1 / (4π * kcond)
    zn = dot(pg0 - pf, n0)
    IDρ_U = _id_radial_parent(aξ, aη, d_par, 1)
    IDρ_T = _id_radial_parent(aξ, aη, d_par, 3)
    ID_U = inv4πk * (J0 / hsc) * IDρ_U
    ID_T = (1 / (4π)) * (J0 / hsc^3) * zn * IDρ_T

    nn = length(Na)
    IU = zeros(nn)
    IT = zeros(nn)
    @inbounds for p in eachindex(cw)
        xp, J, nrm, Nj = _surf_at(poly, nodes, ξc[p], ηc[p])
        U, T = f(dad, xp - pf, nrm)
        wp = cw[p] * J
        for j in 1:nn
            IU[j] += wp * Nj[j] * U
            IT[j] += wp * Nj[j] * T
        end
    end
    dU = ID_U - sum(IU)
    dT = ID_T - sum(IT)
    @inbounds for j in 1:nn
        g[j] += IU[j] + Na[j] * dU
        h[j] += IT[j] + Na[j] * dT
    end
    return nothing
end
