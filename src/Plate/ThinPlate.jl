"""
    ThinPlate

Kirchhoff (thin) plate BEM, ported from `calc_placa.jl`.
Domain load/inertia: `dibem_plate!` (BEM atual `Monta_M_RIMd`).

- Isotropic: `ThinPlateProps` / Shi–Bezine (`placa_fina_isotropica`).
- Anisotropic / laminated: `AnisoThinPlateProps` / Lekhnitskii μ
  (`placa_fina`). The isotropic limit ``μ₁=μ₂=i`` is singular — do not
  smear ``D=√(D₁₁ D₂₂)`` into the isotropic kernels.

Boundary unknowns per collocation node: ``(w, ∂w/∂n)``.
Conjugate tractions: ``(V_n, M_n)``. Corner free terms: concentrated force ``R_c``.

Elements are the shared [`Element`](@ref) (CAD `geo`, collocation `index`).
Field interpolation is `Legendre(p)` at Gauss–Legendre nodes, geometry
`Equispaced(p)` on `el.geo` — the same split as `format2d`. Dense assembly
mirrors Laplace: far nodal lumping, `:tanp3c` near-field, on-element
Guiggiani / analytic CPV.
"""
module ThinPlate

using LinearAlgebra
using Statistics
using StaticArrays
using FastGaussQuadrature
using ProgressMeter
using ForwardDiff
try
    using Richardson: extrapolate
catch
    extrapolate(args...; kwargs...) = error("Richardson.jl is not in this environment")
end

import ..Element, ..shapefun, ..Equispaced, ..Legendre,
    ..discontinuous_nodes_weights, ..tan2normal, ..guiggiani_integral,
    ..guiggiani_GH, ..laurent_coefficients,
    ..closest_point_1d, ..nearfield_1d, .._near_element,
    .._complex_pole_1d, .._tanp3ctrans, .._source_on_element, ..assemble!,
    ..AbstractThinPlate, ..BEMdata, ..BEMCache, ..set_cache!, ..has_cache,
    .._after_vectorial_assemble!, ..fundamental, ..applyBC, ..solve,
    ..integrate_element, .._far_nodal_vec!, ..expand, ..point,
    ..format2d, ..formatdata, ..set_internal_nodes!, ..H_G_full_direct,
    .._collocation_loop!

export AbstractThinPlateProps, ThinPlateProps, AnisoThinPlateProps, PlateCorner, PlateMesh
export bending_stiffness, build_square_plate
export aniso_thin_plate_props, lekhnitskii_roots
export assemble_plate!, apply_bc_plate, solve_plate!, prepare_plate!
export plate_kernels, plate_w, plate_w_int
export analytical_wmax_ss_square, navier_w_ss_square, navier_w_ss_ortho
export build_rect_plate_crack, assemble_plate_dual!, sif_ctod_plate, pin_plate_rbm!
export plate_hbie_kernels

const Point2D = SVector{2,Float64}

# =============================================================================
# Properties & mesh
# =============================================================================

"""Abstract Kirchhoff plate properties (isotropic or anisotropic)."""
const AbstractThinPlateProps = AbstractThinPlate

"""Isotropic Kirchhoff plate — alias of [`BEM.ThinPlate`](@ref)."""
const ThinPlateProps = parentmodule(parentmodule(@__MODULE__)).ThinPlate

bending_stiffness(p::ThinPlateProps) = p.E * p.h^3 / (12 * (1 - p.ν^2))

"""Corner: position, adjacent normals (before/after), element ids, BC."""
mutable struct PlateCorner
    id::Int
    pos::Point2D
    n_prev::Point2D   # normal of arriving edge
    n_next::Point2D   # normal of leaving edge
    el_prev::Int
    el_next::Int
    # bc_type: 0 = w known at corner, 1 = Rc known (free corner)
    bc_type::Int
    bc_val::Float64
end

"""
Kirchhoff plate mesh. Elements are shared [`Element`](@ref).

BCs (`BC`/`BV`) match `BEMdata`: length `2n`, pairs
`(w, Vn)` then `(∂w/∂n, Mn)` per collocation node.
`0` = kinematic known, `1` = static known.
"""
mutable struct PlateMesh{P<:AbstractThinPlateProps,Poly}
    elements::Vector{Element}
    element_type::Poly           # Legendre(p), same as BEMdata
    elem_weight::Vector{Float64} # GL weights on [-1,1] (collocation = Gauss nodes)
    nodes::Vector{Point2D}       # boundary collocation
    Normal::Vector{Point2D}
    BC::Vector{Int}
    BV::Vector{Float64}
    corners::Vector{PlateCorner}
    internal::Vector{Point2D}    # interior collocation (w only)
    props::P
    H::Matrix{Float64}
    G::Matrix{Float64}
    q::Vector{Float64}
    u::Vector{Float64}           # [w; ∂w/∂n]_b + w_int + w_corner
    t::Vector{Float64}           # [Vn; Mn]_b + 0_int + Rc
    eq_type::Vector{Int}         # 1 outer CBIE, 2 crack CBIE, 3 crack HBIE
    twin::Vector{Int}            # coincident crack-face partner (0 if none)
end

function PlateMesh(elems, element_type, elem_weight, nodes, Normal, BC, BV,
        corners, props::AbstractThinPlateProps; internal=Point2D[],
        eq_type=Int[], twin=Int[])
    n = length(nodes)
    ni = length(internal)
    nc = length(corners)
    ndof = 2n + ni + nc
    eq = isempty(eq_type) ? ones(Int, n) : collect(Int, eq_type)
    tw = isempty(twin) ? zeros(Int, n) : collect(Int, twin)
    return PlateMesh{typeof(props),typeof(element_type)}(
        elems, element_type, collect(Float64, elem_weight),
        nodes, Normal, BC, BV, corners, internal, props,
        zeros(ndof, ndof), zeros(ndof, ndof), zeros(ndof),
        zeros(ndof), zeros(ndof), eq, tw)
end

"""DOF layout: 1:2n boundary (w,∂w/∂n), then ni internals (w), then nc corners (w)."""
_ndof(m::PlateMesh) = 2 * length(m.nodes) + length(m.internal) + length(m.corners)
_n(m) = length(m.nodes)
_ni(m) = length(m.internal)
_nc(m) = length(m.corners)

"""Straight CAD edge (collinear `el.geo`) — linear map is exact."""
function _geo_is_straight(geo::AbstractVector)
    n = length(geo)
    n <= 2 && return true
    d = geo[n] - geo[1]
    L2 = d[1] * d[1] + d[2] * d[2]
    L2 < 1e-30 && return false
    @inbounds for k in 2:(n - 1)
        t = geo[k] - geo[1]
        cr = t[1] * d[2] - t[2] * d[1]
        cr * cr > 1e-20 * L2 && return false
    end
    return true
end

"""Linear parent map on a straight edge: `ξ∈[-1,1]` → endpoints `geo[1]`,`geo[end]`."""
function _elem_geom_linear(geo, ξ)
    g0, g1 = geo[1], geo[end]
    N0 = (1 - ξ) * 0.5
    N1 = (1 + ξ) * 0.5
    x = N0 * g0 + N1 * g1
    dx = (g1 - g0) * 0.5
    J = norm(dx)
    n = J > 0 ? Point2D(dx[2] / J, -dx[1] / J) : Point2D(0.0, 0.0)
    return x, J, n
end

# Cached `Equispaced(p)` so curved `elem_geom` does not rebuild Dmat per ξ.
const _EQUISPACED_CACHE = ntuple(i -> Equispaced(i), 8)
function _cached_equispaced(p::Integer)
    (1 <= p <= 8) && return _EQUISPACED_CACHE[p]
    return Equispaced(p)
end

"""Geometry at `ξ` from CAD `el.geo` via `Equispaced` (same as `format2d`).

Straight edges (square/rect plates) use the linear endpoint map — the same
collocation nodes as Laplace (`Legendre` Gauss points, `el.Jacobian` already
stored). Curved CAD uses a cached `Equispaced` interpolant.
"""
function elem_geom(el::Element, ξ)
    geo = el.geo
    isempty(geo) && error("elem_geom: element has no CAD `geo`")
    _geo_is_straight(geo) && return _elem_geom_linear(geo, ξ)
    Ng, dNg = shapefun(_cached_equispaced(length(geo) - 1), ξ)
    x = zero(geo[1])
    dx = zero(geo[1])
    @inbounds for k in eachindex(geo)
        x += Ng[1, k] * geo[k]
        dx += dNg[1, k] * geo[k]
    end
    J = norm(dx)
    n = J > 0 ? Point2D(dx[2] / J, -dx[1] / J) : Point2D(0.0, 0.0)
    return x, J, n
end
elem_geom(::PlateMesh, el::Element, ξ) = elem_geom(el, ξ)

# =============================================================================
# Fundamentals (isotropic Kirchhoff) — calsolfund2.m / calc_placa.jl
# =============================================================================

"""
    plate_kernels(pg, pf, n, nf, D, ν) -> (U, P)

`U` multiplies tractions in G, `P` multiplies kinematics in H (after free term).

```
U = [  w    -∂w/∂n ]
    [ ∂w/∂m -∂²w/∂n∂m ]
P = [  Vn   -Mn ]
    [ ∂Vn/∂m -∂Mn/∂m ]
```
"""
function plate_kernels(pg::Point2D, pf::Point2D, n::Point2D, nf::Point2D, D, ν)
    rs = pg - pf
    r = norm(rs)
    r < 1e-30 && error("plate_kernels: coincident points")
    n1, n2 = n[1], n[2]
    m1, m2 = nf[1], nf[2]
    s1, s2 = -n2, n1
    rd1, rd2 = rs[1] / r, rs[2] / r
    mr = m1 * rd1 + m2 * rd2
    nr = n1 * rd1 + n2 * rd2
    mn = n1 * m1 + n2 * m2
    ms = m1 * s1 + m2 * s2
    sr = s1 * rd1 + s2 * rd2

    w = r^2 / (8π * D) * (log(r) - 0.5)
    dwdn = r / (4π * D) * log(r) * nr
    mnn = -1 / (4π) * ((1 + ν) * log(r) + (1 - ν) * nr^2 + ν)
    vn = nr / (4π * r) * (2 * (1 - ν) * sr^2 - 3 + ν)

    dwdm = -r / (4π * D) * log(r) * mr
    d2wdndm = -1 / (4π * D) * (mr * nr + mn * log(r))
    dvndm = -1 / (4π * r^2) * (
        2 * (1 - ν) * sr * (sr * mn + 2 * nr * ms - 4 * sr * mr * nr) +
        (3 - ν) * (-mn + 2 * mr * nr)
    )
    dmndm = 1 / (4π * r) * ((1 + ν) * mr - 2 * (1 - ν) * nr * (mr * nr - mn))

    U = @SMatrix [w -dwdn; dwdm -d2wdndm]
    P = @SMatrix [vn -mnn; dvndm -dmndm]
    return U, P
end

plate_kernels(pg::Point2D, pf::Point2D, n::Point2D, nf::Point2D, p::ThinPlateProps) =
    plate_kernels(pg, pf, n, nf, bending_stiffness(p), p.ν)

"""`r = x - d`. Returns `(U, P)` as 2×2 `SMatrix` for vectorial `H_G_full_direct`."""
function fundamental(props::ThinPlateProps, r::SVector{2}, n::SVector{2}, nf::SVector{2})
    return plate_kernels(r, zero(r), n, nf, props)
end
function fundamental(dad::BEMdata{<:AbstractThinPlate}, r::SVector{2}, n::SVector{2},
        nf::SVector{2})
    return fundamental(dad.properties, r, n, nf)
end
fundamental(dad::BEMdata{<:AbstractThinPlate}, r::SVector{2}, n::SVector{2}) =
    fundamental(dad, r, n, zero(n))

# =============================================================================
# Singular integrals on a straight element (integraelemsing isotropic)
# =============================================================================

"""∫_0^L u^k log u du (L>0)."""
_int_uk_log(k, L) = L <= 0 ? 0.0 : L^(k + 1) / (k + 1) * (log(L) - 1 / (k + 1))

"""
Moments of the field interpolant `poly` about `ξ0` on ``[-1,1]``:

- `I0 = ∫ N dξ`
- `Ilog = ∫ N log|ξ-ξ0| dξ`
- `Is = CPV ∫ N/(ξ-ξ0) dξ`
- `Is2 = HFP ∫ N/(ξ-ξ0)² dξ` (zero at an endpoint)

Exact via Taylor of `N` at `ξ0` (degree = `length(poly.nodes)-1`).
"""
function _poly_moments(poly, ξ0::Real)
    a = clamp(float(ξ0), -1.0, 1.0)
    N0, _ = shapefun(poly, a)
    nN = size(N0, 2)
    p = nN - 1
    # N^{(k)}(a), k = 0:p
    derivs = Vector{typeof(N0)}(undef, p + 1)
    derivs[1] = N0
    Dk = N0
    for k in 1:p
        Dk = Dk * poly.Dmat
        derivs[k + 1] = Dk
    end
    Lℓ = max(a + 1, 0.0)
    Lr = max(1 - a, 0.0)
    endpoint = Lℓ < 1e-14 || Lr < 1e-14
    Iinv = if Lℓ > 0 && Lr > 0
        log(Lr / Lℓ)
    elseif Lr > 0
        log(Lr)
    elseif Lℓ > 0
        -log(Lℓ)
    else
        0.0
    end
    Iinv2 = endpoint ? 0.0 : -(1 / Lr + 1 / Lℓ)

    # ∫ u^k dξ and ∫ u^k log|u| dξ on [-1,1], u=ξ-a
    Iu = zeros(p + 1)
    Iulog = zeros(p + 1)
    for k in 0:p
        Iu[k + 1] = (Lr > 0 ? Lr^(k + 1) / (k + 1) : 0.0) -
                    (Lℓ > 0 ? (-Lℓ)^(k + 1) / (k + 1) : 0.0)
        Iulog[k + 1] = (Lr > 0 ? _int_uk_log(k, Lr) : 0.0) +
                       (Lℓ > 0 ? ((-1)^k) * _int_uk_log(k, Lℓ) : 0.0)
    end

    I0 = zeros(nN)
    IlogN = zeros(nN)
    Is = zeros(nN)
    Is2 = zeros(nN)
    @inbounds for j in 1:nN
        s0 = 0.0
        slog = 0.0
        ss = 0.0
        ss2 = 0.0
        fact = 1.0
        for k in 0:p
            ck = derivs[k + 1][1, j] / fact   # N^{(k)}/k!
            s0 += ck * Iu[k + 1]
            slog += ck * Iulog[k + 1]
            if k == 0
                ss += ck * Iinv
                ss2 += endpoint ? 0.0 : ck * Iinv2
            elseif k == 1
                ss += ck * Iu[1]
                ss2 += endpoint ? 0.0 : ck * Iinv
            else
                ss += ck * Iu[k]          # ∫ u^{k-1}
                ss2 += ck * Iu[k - 1]     # ∫ u^{k-2}
            end
            fact *= (k + 1)
        end
        I0[j] = s0
        IlogN[j] = slog
        Is[j] = ss
        Is2[j] = ss2
    end
    return I0, IlogN, Is, Is2
end

function integraelemsing(x1::Point2D, x3::Point2D, D, ν, xi0, poly)
    dx = (x3[1] - x1[1])
    dy = (x3[2] - x1[2])
    L = hypot(dx, dy)
    nx, ny = dy / L, -dx / L
    theta = atan(dy, dx)
    D11 = D
    D22 = D
    D12 = ν * D
    D16 = 0.0
    D26 = 0.0
    D66 = (1 - ν) * D / 2

    intN, IlogN, intNsr, intNsr2 = _poly_moments(poly, xi0)
    nN = length(intN)
    Nlog = (L / 2) .* (IlogN .+ log(L / 2) .* intN)
    h_el = zeros(2, 2nN)
    g_el = zeros(nN)

    for j in 1:nN
        d2wdx2 = 1 / (8π * D) * ((1 + cos(2theta)) * intN[j] * L / 2 + 2 * Nlog[j])
        d2wdxdy = 1 / (8π * D) * sin(2theta) * intN[j] * L / 2
        d2wdy2 = -1 / (8π * D) * ((-1 + cos(2theta)) * intN[j] * L - 2 * Nlog[j])

        d3wdx3 = 1 / (4π * D) * cos(theta) * (3 - 2 * cos(theta)^2) * 2 * intNsr[j] / L * L / 2
        d3wdx2dy = 1 / (4π * D) * sin(theta) * (1 - 2 * cos(theta)^2) * 2 * intNsr[j] / L * L / 2
        d3wdxdy2 = 1 / (4π * D) * cos(theta) * (1 - 2 * sin(theta)^2) * 2 * intNsr[j] / L * L / 2
        d3wdy3 = 1 / (4π * D) * sin(theta) * (3 - 2 * sin(theta)^2) * 2 * intNsr[j] / L * L / 2

        d4wdx4 = 1 / (4π * D) * (8 * cos(theta)^4 - 12 * cos(theta)^2 + 3) * (4 * intNsr2[j] / L^2) * L / 2
        d4wdx3dy = 1 / (4π * D) * (-6 * cos(theta) * sin(theta) + 8 * sin(theta) * cos(theta)^3) *
                   (4 * intNsr2[j] / L^2) * L / 2
        d4wdx2dy2 = 1 / (4π * D) * (8 * (cos(theta) * sin(theta))^2 - 1) * (4 * intNsr2[j] / L^2) * L / 2
        d4wdxdy3 = 1 / (4π * D) * (-6 * cos(theta) * sin(theta) + 8 * cos(theta) * sin(theta)^3) *
                   (4 * intNsr2[j] / L^2) * L / 2
        d4wdy4 = 1 / (4π * D) * (8 * sin(theta)^4 - 12 * sin(theta)^2 + 3) * (4 * intNsr2[j] / L^2) * L / 2

        f1 = D11 * nx^2 + 2 * D16 * nx * ny + D12 * ny^2
        f2 = 2 * (D16 * nx^2 + 2 * D66 * nx * ny + D26 * ny^2)
        f3 = D12 * nx^2 + 2 * D26 * nx * ny + D22 * ny^2
        h1 = D11 * nx * (1 + ny^2) + 2 * D16 * ny^3 - D12 * nx * ny^2
        h2 = 4 * D16 * nx + D12 * ny * (1 + nx^2) + 4 * D66 * ny^3 - D11 * nx^2 * ny - 2 * D26 * nx * ny^2
        h3 = 4 * D26 * ny + D12 * nx * (1 + ny^2) + 4 * D66 * nx^3 - D22 * nx * ny^2 - 2 * D16 * nx^2 * ny
        h4 = D22 * ny * (1 + nx^2) + 2 * D26 * nx^3 - D12 * nx^2 * ny

        mn = -(f1 * d2wdx2 + f2 * d2wdxdy + f3 * d2wdy2)
        vn = -(h1 * d3wdx3 + h2 * d3wdx2dy + h3 * d3wdxdy2 + h4 * d3wdy3)
        dmndx = -(f1 * d3wdx3 + f2 * d3wdx2dy + f3 * d3wdxdy2)
        dmndy = -(f1 * d3wdx2dy + f2 * d3wdxdy2 + f3 * d3wdy3)
        dvndx = -(h1 * d4wdx4 + h2 * d4wdx3dy + h3 * d4wdx2dy2 + h4 * d4wdxdy3)
        dvndy = -(h1 * d4wdx3dy + h2 * d4wdx2dy2 + h3 * d4wdxdy3 + h4 * d4wdy4)
        m1, m2 = nx, ny
        d2wdndm = -(d2wdx2 * nx * m1 + d2wdxdy * (nx * m2 + ny * m1) + d2wdy2 * ny * m2)
        dmndm = -(dmndx * m1 + dmndy * m2)
        dvndm = -(dvndx * m1 + dvndy * m2)
        g22 = -d2wdndm
        h11, h12, h21, h22 = vn, -mn, dvndm, -dmndm

        h_el[:, 2j-1:2j] .= [h11 h12; h21 h22]
        g_el[j] = g22
    end
    return h_el, g_el
end

integraelemsing(x1::Point2D, x3::Point2D, p::ThinPlateProps, xi0, poly) =
    integraelemsing(x1, x3, bending_stiffness(p), p.ν, xi0, poly)

"""
On-element Guiggiani / Richardson for the 2×2 plate kernels.

Orders (G=`U`, H=`P`):

```
U:  [ r²log   r log ]     →  0  (log is the strongest; weaker entries have F₀≈0)
    [ r log    log  ]
P:  [  1/r     log  ]     →  -1, 0
    [  1/r²    1/r  ]     →  -2,-1
```

Fused `order_H=-2` leaves the `log` in `M_n` unsubtracted (~1e-4). `G₂₂∼log`
uses interpolant `order_G=0`; log `F₀` is a least-squares fit (not Lagrange
of `f/log` at the collocation — that was 27% off on CCCC). Richardson
rays match `integraelemsing` to ~1e-5 (`scripts/debug/plate_guiggiani_cccc.jl`).
"""
function _plate_singular_guiggiani!(h_el, g_el, el::Element, poly, pf, nf, ξ0,
        props::AbstractThinPlateProps; qsi, w, h::Float64=1e-3, tip=nothing)
    nN = length(el.index)
    a = clamp(float(ξ0), nextfloat(-1.0), prevfloat(1.0))
    zG = zeros(eltype(g_el), 2, 2nN)
    zH = zeros(eltype(h_el), 2, 2nN)
    # One sample pass (Laplace `guiggiani_GH`, `laurent=:interp`). Mixed
    # P orders are recovered from the interpolant (`order_H=-2` is the
    # strongest; log in M_n is not left unsubtracted as with fused Richardson).
    Ig, Ih = guiggiani_GH(a; order_G=0, order_H=-2, qsi=qsi, w=w, h=h,
        laurent=:interp) do ξ
        pg, J, n̂ = elem_geom(el, ξ)
        norm(pg - pf) < 1e-30 && return zG, zH
        U, P = plate_kernels(pg, pf, n̂, nf, props)
        Nw, Nθ = _plate_Nwt(el, poly, ξ, tip)
        Fg = zeros(eltype(g_el), 2, 2nN)
        Fh = zeros(eltype(h_el), 2, 2nN)
        @inbounds for j in 1:nN
            wJ = Nw[j] * J
            tJ = Nθ[j] * J
            Fg[1, 2j - 1] = U[1, 1] * wJ
            Fg[2, 2j - 1] = U[2, 1] * wJ
            Fg[1, 2j] = U[1, 2] * tJ
            Fg[2, 2j] = U[2, 2] * tJ
            Fh[1, 2j - 1] = P[1, 1] * wJ
            Fh[2, 2j - 1] = P[2, 1] * wJ
            Fh[1, 2j] = P[1, 2] * tJ
            Fh[2, 2j] = P[2, 2] * tJ
        end
        return Fg, Fh
    end
    g_el .= Ig
    h_el .= Ih
    return h_el, g_el
end

# =============================================================================
# Corner free terms (compute_Rw)
# =============================================================================

compute_Rw(pf::Point2D, nf::Point2D, corners::Vector{PlateCorner}, p::ThinPlateProps) =
    compute_Rw(pf, nf, corners, bending_stiffness(p), p.ν)

function compute_Rw(pf::Point2D, nf::Point2D, corners::Vector{PlateCorner}, D, ν)
    nc = length(corners)
    RS = zeros(2, nc)
    WS = zeros(2, nc)
    m1, m2 = nf[1], nf[2]
    for (i, c) in enumerate(corners)
        pc = c.pos
        na1, na2 = c.n_prev[1], c.n_prev[2]
        nd1, nd2 = c.n_next[1], c.n_next[2]
        sa1, sa2 = -na2, na1
        sd1, sd2 = -nd2, nd1
        rs = pc - pf
        r = norm(rs)
        if r > 1e-14
            rd1, rd2 = rs[1] / r, rs[2] / r
            rsa = sa1 * rd1 + sa2 * rd2
            rsd = sd1 * rd1 + sd2 * rd2
            rna = na1 * rd1 + na2 * rd2
            rnd = nd1 * rd1 + nd2 * rd2
            mna = m1 * na1 + m2 * na2
            mnd = m1 * nd1 + m2 * nd2
            mr = m1 * rd1 + m2 * rd2
            msa = m1 * sa1 + m2 * sa2
            msd = m1 * sd1 + m2 * sd2
            sen2ba = -2 * rna * rsa
            sen2bd = -2 * rnd * rsd
            mnsa = (1 - ν) / (8π) * sen2ba
            mnsd = (1 - ν) / (8π) * sen2bd
            Rci = mnsd - mnsa
            dmnsa = -(1 - ν) / (4π * r) * (2 * mr * rna * rsa - mna * rsa - msa * rna)
            dmnsd = -(1 - ν) / (4π * r) * (2 * mr * rnd * rsd - mnd * rsd - msd * rnd)
            dRci = dmnsd - dmnsa
            we = 1 / (8π * D) * r^2 * (log(r) - 0.5)
            dwdm = -r / (4π * D) * mr * log(r)
            RS[:, i] .= (Rci, dRci)
            WS[:, i] .= (we, dwdm)
        else
            cbetac = -(sa1 * sd1 + sa2 * sd2)
            sbetac = -(na1 * sd1 + na2 * sd2)
            betac = atan(sbetac, cbetac) / (2π)
            betac < 0 && (betac += 1)
            RS[:, i] .= (betac, 0.0)
            WS[:, i] .= (0.0, 0.0)
        end
    end
    return RS, WS
end

# =============================================================================
# Distributed load particular integral (compute_q)
# =============================================================================

compute_q_el(pf::Point2D, nf::Point2D, mesh::PlateMesh, el::Element, qsi, w) =
    compute_q_el(pf, nf, mesh, el, qsi, w, mesh.props)
compute_q_el(pf::Point2D, nf::Point2D, dad::BEMdata{<:AbstractThinPlate}, el::Element, qsi, w) =
    compute_q_el(pf, nf, dad, el, qsi, w, dad.properties)

function compute_q_el(pf::Point2D, nf::Point2D, mesh, el::Element,
    qsi, w, p::ThinPlateProps)
    D = bending_stiffness(p)
    qa, qb, qc = p.q_a, p.q_b, p.q_c
    q_el = zeros(2)
    m1, m2 = nf[1], nf[2]
    xf, yf = pf[1], pf[2]
    Clocal0 = qa * xf + qb * yf + qc
    for (ig, ξ) in enumerate(qsi)
        pg, J, n = elem_geom(el, ξ)
        rs = pg - pf
        r = norm(rs)
        r < 1e-14 && continue
        rd1, rd2 = rs[1] / r, rs[2] / r
        n1, n2 = n[1], n[2]
        nr = n1 * rd1 + n2 * rd2
        theta = atan(rs[2], rs[1])
        Clocal = Clocal0  # constant part; linear terms neglected in particular as ref. uses Clocal at source
        # include field-point linear load
        Clocal = qa * pg[1] + qb * pg[2] + qc
        intrdwdx = -(r^2 * cos(theta) * (2 - 6 * log(r))) / (72 * D * π)
        intrdwdy = -(r^2 * (2 - 6 * log(r)) * sin(theta)) / (72 * D * π)
        int1 = -Clocal * nr * (r^3 * (3 - 4 * log(r))) / (128 * D * π)
        int2 = -Clocal * nr * (intrdwdx * m1 + intrdwdy * m2)
        q_el .+= (int1, int2) .* (J * w[ig])
    end
    return q_el
end

include("ThinPlateAniso.jl")

# =============================================================================
# Assembly
# =============================================================================

"""Lift `BEMdata` to [`PlateMesh`](@ref) (BEM atual layout: internals are `w` only)."""
function _platemesh_from_dad(dad::BEMdata{<:AbstractThinPlate})
    return PlateMesh(dad.elements, dad.element_type, collect(Float64, dad.elem_weight),
        collect(Point2D, dad.Nodes), collect(Point2D, dad.Normal),
        copy(dad.BC), copy(dad.BV), _plate_corners(dad), dad.properties;
        internal=collect(Point2D, dad.internalNodes))
end

"""Copy PlateMesh `H,G,q` into the 2-DOF + dummy-slope + corner `BEMdata` layout."""
function _import_platemesh_system!(dad::BEMdata{<:AbstractThinPlate}, mesh::PlateMesh)
    n, ni = dad.n, dad.ni
    nc = length(mesh.corners)
    ndd = 2 * (n + ni) + nc
    nGc = 2n + nc
    H = zeros(ndd, ndd)
    G = zeros(ndd, nGc)
    q = zeros(ndd)
    function row_dad(rm)
        rm <= 2n && return rm
        if rm <= 2n + ni
            k = rm - 2n
            return 2 * (n + k) - 1
        end
        return 2 * (n + ni) + (rm - 2n - ni)
    end
    function colG_dad(cm)
        cm <= 2n && return cm
        cm <= 2n + ni && return 0
        return 2n + (cm - 2n - ni)
    end
    Hm, Gm, qm = mesh.H, mesh.G, mesh.q
    ndm = size(Hm, 1)
    @inbounds for j in 1:ndm
        jd = row_dad(j)
        for i in 1:ndm
            H[row_dad(i), jd] = Hm[i, j]
        end
    end
    @inbounds for j in 1:ndm
        gd = colG_dad(j)
        gd == 0 && continue
        for i in 1:ndm
            G[row_dad(i), gd] = Gm[i, j]
        end
    end
    @inbounds for i in 1:ndm
        q[row_dad(i)] = qm[i]
    end
    # PlateMesh already has interior jump c=1 on w. Only pin dummy ∂w/∂n.
    if dad.ni > 0
        @inbounds for k in 1:dad.ni
            r2 = 2 * (dad.n + k)
            fill!(view(H, r2, :), 0.0)
            H[r2, r2] = 1.0
            fill!(view(G, r2, :), 0.0)
        end
    end
    set_cache!(dad; H, G, plate_q=q, plate_corners=mesh.corners)
    return dad
end

"""
    assemble_plate!(dad::BEMdata{<:AbstractThinPlate}; npg=12, kwargs...)

Kirchhoff `H, G`. Default `singular=:auto` uses the BEM atual on-element
analytic CPV (`integraelemsing`) on straight edges — required for clamped
`G₂₂` / `M_n`. `singular=:guiggiani` is the vectorial Guiggiani path
(too stiff on CCCC). Then dummy internal slope, `q`, corner `R_c`.
"""
function assemble!(dad::BEMdata{<:AbstractThinPlate}; method::Symbol=:dense,
        npg=12, nsub::Int=8, threaded::Bool=true, near_factor::Real=1.5,
        singular::Symbol=:auto, kwargs...)
    if has_cache(dad, :eq_type) && any(==(3), dad.eq_type)
        return assemble_plate_dual!(dad; npg=npg, nsub=nsub, threaded=threaded,
            near_factor=near_factor)
    end
    method === :dense || throw(ArgumentError("plate assemble! method must be :dense; got $method"))
    if singular === :guiggiani
        return H_G_full_direct(dad; npg=npg, threaded=threaded, near_factor=near_factor,
            singular=:guiggiani, kwargs...)
    end
    mesh = _platemesh_from_dad(dad)
    assemble_plate!(mesh; npg=npg, singular=singular, threaded=threaded,
        near_factor=near_factor)
    return _import_platemesh_system!(dad, mesh)
end

function assemble_plate!(dad::BEMdata{<:AbstractThinPlate}; npg=12, kwargs...)
    assemble!(dad; npg=npg, kwargs...)
    return dad
end

function _after_vectorial_assemble!(dad::BEMdata{<:AbstractThinPlate}, H, G)
    _plate_pin_internal_slope!(H, G, dad)
    qv = _plate_load_vector(dad)
    corners = _plate_corners(dad)
    if !isempty(corners)
        H, G, qv = _plate_append_corners(dad, H, G, qv, corners)
        set_cache!(dad; H, G)
    end
    set_cache!(dad; plate_q=qv)
    return nothing
end

_plate_corners(dad::BEMdata) =
    has_cache(dad, :plate_corners) ? dad.plate_corners : PlateCorner[]

function _plate_pin_internal_slope!(H, G, dad)
    dad.ni == 0 && return nothing
    @inbounds for k in 1:dad.ni
        i = dad.n + k
        r1 = 2 * i - 1
        r2 = 2 * i
        H[r1, r1] += 1.0          # interior jump c = 1 (not 1/2)
        fill!(view(H, r2, :), 0.0)
        H[r2, r2] = 1.0           # dummy ∂w/∂n at interior
        fill!(view(G, r2, :), 0.0)
    end
    return nothing
end

function _plate_load_vector(dad::BEMdata{<:AbstractThinPlate})
    p = dad.properties
    nd = 2 * dad.nt
    qv = zeros(nd)
    (p.q_a == 0 && p.q_b == 0 && p.q_c == 0) && return qv
    qsi = has_cache(dad, :qsi) ? dad.qsi : gausslegendre(12)[1]
    w = has_cache(dad, :w) ? dad.w : gausslegendre(12)[2]
    if !(has_cache(dad, :qsi) && has_cache(dad, :w))
        qsi, w = gausslegendre(12)
    end
    @inbounds for i in 1:dad.nt
        pf = point(dad, i)
        nf = i <= dad.n ? dad.Normal[i] : zero(pf)
        rows = expand(i, 2)
        for el in dad.elements
            qe = compute_q_el(pf, nf, dad, el, qsi, w)
            qv[rows[1]] += qe[1]
            i <= dad.n && (qv[rows[2]] += qe[2])
        end
    end
    return qv
end

function _plate_append_corners(dad, H, G, qv, corners)
    nc = length(corners)
    nH, nGc = size(H, 1), size(G, 2)
    ndof = nH + nc
    H2 = zeros(ndof, ndof)
    G2 = zeros(ndof, nGc + nc)
    q2 = zeros(ndof)
    H2[1:nH, 1:nH] .= H
    G2[1:nH, 1:nGc] .= G
    q2[1:nH] .= qv
    props = dad.properties
    dim = 2
    @inbounds for i in 1:dad.nt
        pf = point(dad, i)
        nf = i <= dad.n ? dad.Normal[i] : zero(pf)
        rows = expand(i, dim)
        RS, WS = compute_Rw(pf, nf, corners, props)
        H2[rows, nH .+ (1:nc)] .= RS
        G2[rows, nGc .+ (1:nc)] .= WS
        if i > dad.n
            H2[rows[2], nH .+ (1:nc)] .= 0
            G2[rows[2], nGc .+ (1:nc)] .= 0
        end
    end
    nf0 = Point2D(0.0, 0.0)
    f = (d, r, nrm) -> fundamental(d, r, nrm, nf0)
    qsi = has_cache(dad, :qsi) ? dad.qsi : gausslegendre(12)[1]
    w = has_cache(dad, :w) ? dad.w : gausslegendre(length(qsi))[2]
    has_q = !(props.q_a == 0 && props.q_b == 0 && props.q_c == 0)
    maxc = maximum(length(el.index) for el in dad.elements; init=1)
    hloc = zeros(2, 2 * maxc)
    gloc = zeros(2, 2 * maxc)
    nf_cut = has_cache(dad, :near_factor) ? float(dad.near_factor) : 1.5
    for c in 1:nc
        row = nH + c
        corner = corners[c]
        pf = corner.pos
        @inbounds for el in dad.elements
            xj = dad.Nodes[el.index]
            nn = length(el.index)
            if _near_element(pf, xj, el; factor=nf_cut)
                hv = view(hloc, 1:2, 1:(2nn))
                gv = view(gloc, 1:2, 1:(2nn))
                fill!(hv, 0)
                fill!(gv, 0)
                integrate_element(dad, el, xj, pf, hv, gv, f)
                for a in 1:nn
                    ja = el.index[a]
                    c0 = 2 * ja - 1
                    H2[row, c0] += hv[1, 2a - 1]
                    H2[row, c0 + 1] += hv[1, 2a]
                    G2[row, c0] += gv[1, 2a - 1]
                    G2[row, c0 + 1] += gv[1, 2a]
                end
            else
                wts = dad.elem_weight
                for k in eachindex(el.index)
                    node = el.index[k]
                    U, T = f(dad, dad.Nodes[node] - pf, dad.Normal[node])
                    wjk = el.Jacobian[k] * wts[k]
                    c0 = 2 * node - 1
                    H2[row, c0] += T[1, 1] * wjk
                    H2[row, c0 + 1] += T[1, 2] * wjk
                    G2[row, c0] += U[1, 1] * wjk
                    G2[row, c0 + 1] += U[1, 2] * wjk
                end
            end
            if has_q
                q2[row] += compute_q_el(pf, nf0, dad, el, qsi, w)[1]
            end
        end
        RS, WS = compute_Rw(pf, nf0, corners, props)
        H2[row, nH .+ (1:nc)] .+= RS[1, :]
        G2[row, nGc .+ (1:nc)] .+= WS[1, :]
    end
    return H2, G2, q2
end

"""
    assemble_plate!(mesh::PlateMesh; npg=12, singular=:auto, threaded=true, near_factor=1.5)

Legacy `PlateMesh` assembler (Dual BEM / scripts). Square plates from
[`build_square_plate`](@ref) return `BEMdata{<:ThinPlate}` and use
[`assemble!`](@ref).
"""
function assemble_plate!(mesh::PlateMesh; npg=12, singular::Symbol=:auto,
        nsub::Int=8, threaded::Bool=true, near_factor::Real=1.5,
        nearfield::Symbol=:euclid)
    any(==(3), mesh.eq_type) && return assemble_plate_dual!(mesh; npg=npg, nsub=nsub)
    (singular === :guiggiani || singular === :analytic || singular === :auto) ||
        throw(ArgumentError("singular must be :auto, :guiggiani, or :analytic; got $singular"))
    n = _n(mesh)
    ni = _ni(mesh)
    nc = _nc(mesh)
    ndof = 2n + ni + nc
    H = zeros(ndof, ndof)
    G = zeros(ndof, ndof)
    qv = zeros(ndof)
    props = mesh.props
    qsi, w = gausslegendre(npg)
    qsi_f, w_f = gausslegendre(max(npg, 20))
    cols_c0 = 2n + ni
    poly = mesh.element_type
    nf_cut = float(near_factor)
    has_q = _has_distributed_load(props)
    elems = mesh.elements
    nE = length(elems)
    nodes_el = Vector{typeof(mesh.nodes[elems[1].index])}(undef, nE)
    maxcols = 2
    @inbounds for eidx in 1:nE
        el = elems[eidx]
        nodes_el[eidx] = mesh.nodes[el.index]
        maxcols = max(maxcols, 2 * length(el.index))
    end
    nb = (threaded && Threads.nthreads() > 1) ? Threads.maxthreadid() : 1
    hbufs = [zeros(2, maxcols) for _ in 1:nb]
    gbufs = [zeros(2, maxcols) for _ in 1:nb]

    let H = H, G = G, qv = qv
        _plate_rows!(threaded, n, hbufs, gbufs) do i, hbuf, gbuf
            pf = mesh.nodes[i]
            nf = mesh.Normal[i]
            rows = (2i - 1):(2i)
            @inbounds for eidx in 1:nE
                el = elems[eidx]
                _plate_pair!(H, G, qv, mesh, el, nodes_el[eidx], pf, nf, rows,
                    i, eidx, 0, 0, hbuf, gbuf, poly, props, qsi, w, qsi_f, w_f,
                    singular, nf_cut, nearfield, has_q)
            end
            RS, WS = compute_Rw(pf, nf, mesh.corners, props)
            H[rows, cols_c0 .+ (1:nc)] .= RS
            G[rows, cols_c0 .+ (1:nc)] .= WS
        end
    end

    nf0 = Point2D(0.0, 0.0)
    hbuf = hbufs[1]
    gbuf = gbufs[1]
    for k in 1:ni
        row = 2n + k
        rows = row:row
        pf = mesh.internal[k]
        @inbounds for eidx in 1:nE
            el = elems[eidx]
            _plate_pair!(H, G, qv, mesh, el, nodes_el[eidx], pf, nf0, rows,
                0, eidx, 0, 0, hbuf, gbuf, poly, props, qsi, w, qsi_f, w_f,
                singular, nf_cut, nearfield, has_q)
        end
        RS, WS = compute_Rw(pf, nf0, mesh.corners, props)
        H[row, cols_c0 .+ (1:nc)] .+= RS[1, :]
        G[row, cols_c0 .+ (1:nc)] .+= WS[1, :]
        H[row, row] += 1.0
    end
    for c in 1:nc
        corner = mesh.corners[c]
        row = cols_c0 + c
        rows = row:row
        pf = corner.pos
        @inbounds for eidx in 1:nE
            el = elems[eidx]
            _plate_pair!(H, G, qv, mesh, el, nodes_el[eidx], pf, nf0, rows,
                0, eidx, corner.el_prev, corner.el_next, hbuf, gbuf, poly, props,
                qsi, w, qsi_f, w_f, singular, nf_cut, nearfield, has_q)
        end
        RS, WS = compute_Rw(pf, nf0, mesh.corners, props)
        H[row, cols_c0 .+ (1:nc)] .+= RS[1, :]
        G[row, cols_c0 .+ (1:nc)] .+= WS[1, :]
    end

    @inbounds for i in 1:n
        H[2i - 1, 2i - 1] += 0.5
        H[2i, 2i] += 0.5
    end

    mesh.H = H
    mesh.G = G
    mesh.q = qv
    return mesh
end

"""`assemble!(plate::PlateMesh)` — same kwargs as [`assemble_plate!`](@ref)."""
assemble!(mesh::PlateMesh; kwargs...) = assemble_plate!(mesh; kwargs...)

@inline _has_distributed_load(p) = p.q_a != 0 || p.q_b != 0 || p.q_c != 0

"""Static row schedule so thread-local `h`/`g` buffers are not shared (Julia
`:dynamic` `@threads` can migrate tasks onto the same `threadid()`)."""
function _plate_rows!(body, threaded::Bool, n::Int, hbufs, gbufs)
    if threaded && Threads.nthreads() > 1
        Threads.@threads :static for i in 1:n
            tid = Threads.threadid()
            body(i, hbufs[tid], gbufs[tid])
        end
    else
        for i in 1:n
            body(i, hbufs[1], gbufs[1])
        end
    end
    return nothing
end

function _resolve_singular(singular::Symbol, el::Element)
    if singular === :auto
        return (!isempty(el.geo) && _geo_is_straight(el.geo)) ? :analytic : :guiggiani
    end
    return singular
end

"""Near-field parent rule. Default `:euclid` is sinh about the closest point
(`b=d/L`), the classical BEM nearly-singular map. `:tanp3c` is Laplace's
isotropic default; `:plain` is uniform Gauss."""
function _plate_near_rule(el, poly, nodes, pf, qsi, w, nearfield::Symbol)
    if nearfield === :plain
        return qsi, w
    elseif nearfield === :tanp3c
        ζ0, η0 = _complex_pole_1d(poly, nodes, pf)
        return _tanp3ctrans(qsi, w, ζ0, η0)
    else
        a, _, dist = closest_point_1d(poly, nodes, pf)
        b = dist / max(el.Length, eps())
        return nearfield_1d(a, max(b, 1e-14); qsi=qsi, w=w)
    end
end

function _plate_integrate_regular!(h, g, el, pf, nf, poly, props, η, ww)
    nN = length(el.index)
    Nf, _ = shapefun(poly, η)
    @inbounds for i in eachindex(η)
        pg, J, n̂ = elem_geom(el, η[i])
        sum(abs2, pg - pf) < 1e-28 && continue
        U, P = plate_kernels(pg, pf, n̂, nf, props)
        wJ = J * ww[i]
        for a in 1:nN
            Na = Nf[i, a] * wJ
            h[1, 2a - 1] += P[1, 1] * Na
            h[1, 2a] += P[1, 2] * Na
            h[2, 2a - 1] += P[2, 1] * Na
            h[2, 2a] += P[2, 2] * Na
            g[1, 2a - 1] += U[1, 1] * Na
            g[1, 2a] += U[1, 2] * Na
            g[2, 2a - 1] += U[2, 1] * Na
            g[2, 2a] += U[2, 2] * Na
        end
    end
    return nothing
end

function _plate_far_nodal!(H, G, rows, el, pf, nf, props, mesh)
    wts = mesh.elem_weight
    nr = length(rows)
    @inbounds for k in eachindex(el.index)
        node = el.index[k]
        U, P = plate_kernels(mesh.nodes[node], pf, mesh.Normal[node], nf, props)
        wjk = el.Jacobian[k] * wts[k]
        c0 = 2 * node - 1
        for r in 1:nr
            rr = rows[r]
            H[rr, c0] += P[r, 1] * wjk
            H[rr, c0 + 1] += P[r, 2] * wjk
            G[rr, c0] += U[r, 1] * wjk
            G[rr, c0 + 1] += U[r, 2] * wjk
        end
    end
    return nothing
end

function _plate_on_element!(h, g, el, poly, pf, nf, ξ0, props, singular, qsi, w, qsi_f, w_f)
    method = _resolve_singular(singular, el)
    fill!(h, 0)
    fill!(g, 0)
    if method === :analytic && length(el.geo) >= 2
        # G₁₁,G₁₂,G₂₁ from Gauss (skip the coincident point); H and G₂₂ analytic.
        _plate_integrate_regular!(h, g, el, pf, nf, poly, props, qsi_f, w_f)
        hsing, g2sing = integraelemsing(el.geo[1], el.geo[end], props, ξ0, poly)
        h .= hsing
        nN = length(el.index)
        @inbounds for a in 1:nN
            g[2, 2a] = g2sing[a]
        end
    else
        _plate_singular_guiggiani!(h, g, el, poly, pf, nf, ξ0, props; qsi=qsi_f, w=w_f)
    end
    return nothing
end

function _scatter_plate!(H, G, rows, idx, hloc, gloc)
    nN = length(idx)
    nr = length(rows)
    @inbounds for a in 1:nN
        c0 = 2 * idx[a] - 1
        for r in 1:nr
            rr = rows[r]
            H[rr, c0] += hloc[r, 2a - 1]
            H[rr, c0 + 1] += hloc[r, 2a]
            G[rr, c0] += gloc[r, 2a - 1]
            G[rr, c0 + 1] += gloc[r, 2a]
        end
    end
    return nothing
end

"""One source–element pair. `src` is the collocation index (0 if interior).
`sing_prev`/`sing_next` mark corner-adjacent elements (0 if none)."""
function _plate_pair!(H, G, qv, mesh, el, xj, pf, nf, rows, src, eidx, sing_prev, sing_next,
        hbuf, gbuf, poly, props, qsi, w, qsi_f, w_f, singular, near_factor,
        nearfield, has_q)
    idx = el.index
    nN = length(idx)
    nc2 = 2nN
    hloc = view(hbuf, 1:2, 1:nc2)
    gloc = view(gbuf, 1:2, 1:nc2)
    on_el = (src != 0 && _source_on_element(el, src)) ||
            eidx == sing_prev || eidx == sing_next
    if on_el
        ξ0 = if src != 0
            loc = findfirst(==(src), idx)
            poly.nodes[loc]
        elseif eidx == sing_prev
            1.0
        else
            -1.0
        end
        _plate_on_element!(hloc, gloc, el, poly, pf, nf, ξ0, props, singular,
            qsi, w, qsi_f, w_f)
        _scatter_plate!(H, G, rows, idx, hloc, gloc)
    elseif _near_element(pf, xj, el; factor=near_factor)
        fill!(hloc, 0)
        fill!(gloc, 0)
        η, ww = _plate_near_rule(el, poly, xj, pf, qsi, w, nearfield)
        _plate_integrate_regular!(hloc, gloc, el, pf, nf, poly, props, η, ww)
        _scatter_plate!(H, G, rows, idx, hloc, gloc)
    else
        _plate_far_nodal!(H, G, rows, el, pf, nf, props, mesh)
    end
    if has_q
        qe = compute_q_el(pf, nf, mesh, el, qsi, w)
        nr = length(rows)
        @inbounds for r in 1:nr
            qv[rows[r]] += qe[r]
        end
    end
    return nothing
end

# =============================================================================
# BC & solve
# =============================================================================

"""
Apply mixed BCs. Kinematic known → swap columns (scale tractions by `D`).
Returns `(A, b, is_kin, known_val)`.
"""
function apply_bc_plate(mesh::PlateMesh)
    A = copy(mesh.H)
    G = mesh.G
    Dsc = bending_stiffness(mesh.props)
    n = _n(mesh)
    ni = _ni(mesh)
    nc = _nc(mesh)
    ndof = 2n + ni + nc
    is_kin = falses(ndof)
    known = zeros(ndof)
    b = copy(mesh.q)

    @inbounds for dof in 1:2n
        val = mesh.BV[dof]
        known[dof] = val
        if mesh.BC[dof] == 0
            is_kin[dof] = true
            for i in 1:ndof
                b[i] -= A[i, dof] * val
                A[i, dof] = -G[i, dof] * Dsc
            end
        else
            for i in 1:ndof
                b[i] += G[i, dof] * val
            end
        end
    end
    for (c, corner) in enumerate(mesh.corners)
        dof = 2n + ni + c
        val = corner.bc_val
        known[dof] = val
        if corner.bc_type == 0
            is_kin[dof] = true
            @inbounds for i in 1:ndof
                b[i] -= A[i, dof] * val
                A[i, dof] = -G[i, dof] * Dsc
            end
        else
            @inbounds for i in 1:ndof
                b[i] += G[i, dof] * val
            end
        end
    end
    return A, b, is_kin, known
end

"""
    solve_plate!(mesh::PlateMesh) -> (u, t)

Apply mixed Kirchhoff BCs, solve ``H u = q + G t`` for unknown
deflection / slope / corner force, and store `mesh.u`, `mesh.t`.
Tractions are scaled by bending stiffness ``D``.
"""
function solve_plate!(mesh::PlateMesh)
    A, b, is_kin, known = apply_bc_plate(mesh)
    any(==(3), mesh.eq_type) && _equilibrate_rows!(A, b)
    x = A \ b
    n = _n(mesh)
    ni = _ni(mesh)
    nc = _nc(mesh)
    ndof = 2n + ni + nc
    u = zeros(ndof)
    t = zeros(ndof)
    Dsc = bending_stiffness(mesh.props)
    for dof in 1:ndof
        if is_kin[dof]
            u[dof] = known[dof]
            t[dof] = x[dof] * Dsc
        else
            # internal DOFs and Neumann unknowns
            if dof <= 2n || dof > 2n + ni
                t[dof] = known[dof]
            end
            u[dof] = x[dof]
        end
    end
    mesh.u = u
    mesh.t = t
    return u, t
end

"""Boundary deflection `w` at node `i`."""
plate_w(mesh::PlateMesh, i::Int) = mesh.u[2i - 1]
plate_w(dad::BEMdata{<:AbstractThinPlate}, i::Int) = dad.u[2i - 1]

"""Interior deflection at internal point `k`."""
plate_w_int(mesh::PlateMesh, k::Int=1) = mesh.u[2 * _n(mesh) + k]
plate_w_int(dad::BEMdata{<:AbstractThinPlate}, k::Int=1) = dad.u[2 * (dad.n + k) - 1]

function applyBC(dad::BEMdata{<:AbstractThinPlate}; kwargs...)
    H = dad.H
    G = dad.G
    Dsc = bending_stiffness(dad.properties)
    ndof = size(H, 1)
    A = copy(H)
    b = zeros(ndof)
    has_cache(dad, :plate_q) && (b .+= dad.plate_q)
    n2 = 2 * dad.n
    @inbounds for dof in 1:n2
        val = dad.BV[dof]
        if dad.BC[dof] == 0
            for i in 1:ndof
                b[i] -= A[i, dof] * val
                A[i, dof] = -G[i, dof] * Dsc
            end
        else
            for i in 1:ndof
                b[i] += G[i, dof] * val
            end
        end
    end
    corners = _plate_corners(dad)
    nGc = 2 * dad.n
    nH0 = 2 * dad.nt
    for (c, corner) in enumerate(corners)
        dofH = nH0 + c
        dofG = nGc + c
        val = corner.bc_val
        if corner.bc_type == 0
            @inbounds for i in 1:ndof
                b[i] -= A[i, dofH] * val
                A[i, dofH] = -G[i, dofG] * Dsc
            end
        else
            @inbounds for i in 1:ndof
                b[i] += G[i, dofG] * val
            end
        end
    end
    set_cache!(dad; A, b)
    return nothing
end

function solve(dad::BEMdata{<:AbstractThinPlate}; kwargs...)
    applyBC(dad)
    A, b = dad.A, dad.b
    has_cache(dad, :eq_type) && any(==(3), dad.eq_type) && _equilibrate_rows!(A, b)
    x = A \ b
    Dsc = bending_stiffness(dad.properties)
    ndof = length(x)
    u = zeros(ndof)
    t = zeros(ndof)
    n2 = 2 * dad.n
    @inbounds for dof in 1:n2
        if dad.BC[dof] == 0
            u[dof] = dad.BV[dof]
            t[dof] = x[dof] * Dsc
        else
            t[dof] = dad.BV[dof]
            u[dof] = x[dof]
        end
    end
    @inbounds for dof in (n2 + 1):min(2 * dad.nt, ndof)
        u[dof] = x[dof]
    end
    corners = _plate_corners(dad)
    nH0 = 2 * dad.nt
    @inbounds for c in eachindex(corners)
        dof = nH0 + c
        if corners[c].bc_type == 0
            u[dof] = corners[c].bc_val
            t[dof] = x[dof] * Dsc
        else
            t[dof] = corners[c].bc_val
            u[dof] = x[dof]
        end
    end
    set_cache!(dad; u=u, traction=t, T=u)
    return u
end

function solve_plate!(dad::BEMdata{<:AbstractThinPlate})
    solve(dad)
    return dad.u, dad.t
end
"""Collocation points that carry a `w` unknown, and their DOF indices."""
function _plate_w_samples(plate::PlateMesh)
    n = length(plate.nodes)
    ni = length(plate.internal)
    nc = length(plate.corners)
    pts = Point2D[]
    w_index = Int[]
    for i in 1:n
        push!(pts, plate.nodes[i])
        push!(w_index, 2i - 1)
    end
    for k in 1:ni
        push!(pts, plate.internal[k])
        push!(w_index, 2n + k)
    end
    for c in 1:nc
        push!(pts, plate.corners[c].pos)
        push!(w_index, 2n + ni + c)
    end
    return pts, w_index
end
function _plate_w_samples(dad::BEMdata{<:AbstractThinPlate})
    n = dad.n
    pts = Point2D[]
    w_index = Int[]
    for i in 1:n
        push!(pts, dad.Nodes[i])
        push!(w_index, 2i - 1)
    end
    for k in 1:dad.ni
        push!(pts, dad.internalNodes[k])
        push!(w_index, 2 * (n + k) - 1)
    end
    for (c, corner) in enumerate(_plate_corners(dad))
        push!(pts, corner.pos)
        push!(w_index, 2 * dad.nt + c)
    end
    return pts, w_index
end

function apply_bc_plate(dad::BEMdata{<:AbstractThinPlate})
    applyBC(dad)
    ndof = size(dad.A, 1)
    is_kin = falses(ndof)
    known = zeros(ndof)
    n2 = 2 * dad.n
    @inbounds for dof in 1:n2
        known[dof] = dad.BV[dof]
        is_kin[dof] = dad.BC[dof] == 0
    end
    corners = _plate_corners(dad)
    nH0 = 2 * dad.nt
    for (c, corner) in enumerate(corners)
        dof = nH0 + c
        known[dof] = corner.bc_val
        is_kin[dof] = corner.bc_type == 0
    end
    return dad.A, dad.b, is_kin, known
end

# =============================================================================
# Mesh builder — square plate
# =============================================================================

"""
BC presets per edge (string):
- `"C"` clamped: w=0, ∂w/∂n=0
- `"S"` simply supported: w=0, Mn=0
- `"F"` free: Vn=0, Mn=0
"""
function _bc_from_char(c::Char)
    c = uppercase(c)
    if c == 'C'
        return (0, 0.0, 0, 0.0)   # (type_w, val_w, type_m, val_m)
    elseif c == 'S'
        return (0, 0.0, 1, 0.0)
    elseif c == 'F'
        return (1, 0.0, 1, 0.0)
    else
        error("unknown BC '$c' (use C/S/F)")
    end
end

function _make_edge!(elems, nodes, normals, BC, BV, p0, p1, n_el, bc,
        qsi, wi, edge_tag)
    tw, vw, tm, vm = bc
    nN = length(qsi)
    p = nN - 1
    Ngeo, dNgeo = shapefun(Equispaced(p), qsi)
    t_dir = p1 - p0
    Ledge = norm(t_dir)
    n_hat = Ledge > 0 ? Point2D(t_dir[2] / Ledge, -t_dir[1] / Ledge) : Point2D(0.0, 1.0)
    for e in 1:n_el
        s0 = (e - 1) / n_el
        s1 = e / n_el
        g0 = (1 - s0) * p0 + s0 * p1
        g1 = (1 - s1) * p0 + s1 * p1
        X = Point2D[(1 - σ) * g0 + σ * g1 for σ in range(0, 1; length=nN)]
        col = Ngeo * X
        dx = dNgeo * X
        J = norm.(dx)
        idx0 = length(nodes)
        idx = (idx0 + 1):(idx0 + nN)
        append!(nodes, col)
        append!(normals, tan2normal.(dx ./ J))
        for _ in 1:nN
            push!(BC, tw); push!(BV, vw)
            push!(BC, tm); push!(BV, vm)
        end
        L = abs(dot(J, wi))
        push!(elems, Element(;
            index=collect(Int64, idx),
            Jacobian=collect(Float64, J),
            Length=Float64(L),
            Region=Int64(edge_tag),
            geo=X))
    end
    return n_hat
end

"""Infer Kirchhoff corner `R_c` DOFs from CAD endpoints (two incident edges)."""
function _infer_plate_corners(dad::BEMdata, corner_bc::Char)
    hits = Dict{NTuple{2,Float64},Vector{Tuple{Int,Int,Point2D}}}()
    @inbounds for (ie, el) in enumerate(dad.elements)
        length(el.geo) < 2 && continue
        n̂ = dad.Normal[el.index[1]]
        for (endflag, pg) in ((-1, el.geo[1]), (1, el.geo[end]))
            key = (round(pg[1]; digits=12), round(pg[2]; digits=12))
            push!(get!(hits, key, Tuple{Int,Int,Point2D}[]), (ie, endflag, n̂))
        end
    end
    bt = uppercase(corner_bc) == 'F' ? 1 : 0
    corners = PlateCorner[]
    id = 0
    for (key, lst) in hits
        length(lst) == 2 || continue
        (ie_a, end_a, na), (ie_b, end_b, nb) = lst[1], lst[2]
        abs(na[1] * nb[1] + na[2] * nb[2]) > 0.5 && continue  # same-edge joint, not a corner
        # arriving edge ends at +1; leaving edge starts at -1
        if end_a == 1 && end_b == -1
            el_prev, el_next, n_prev, n_next = ie_a, ie_b, na, nb
        elseif end_b == 1 && end_a == -1
            el_prev, el_next, n_prev, n_next = ie_b, ie_a, nb, na
        else
            el_prev, el_next, n_prev, n_next = ie_a, ie_b, na, nb
        end
        id += 1
        push!(corners, PlateCorner(id, Point2D(key[1], key[2]),
            n_prev, n_next, el_prev, el_next, bt, 0.0))
    end
    sort!(corners; by=c -> atan(c.pos[2] - 0.5, c.pos[1] - 0.5))
    for (k, c) in enumerate(corners)
        c.id = k
    end
    return corners
end

"""
    prepare_plate!(dad; corner_bc='F')

Attach Kirchhoff corner reactions on `dad::BEMdata{<:ThinPlate}` after
[`format2d`](@ref) / [`formatdata`](@ref). `'F'` frees ``R_c``, `'C'` clamps corner `w`.
"""
function prepare_plate!(dad::BEMdata{<:AbstractThinPlate}; corner_bc::Char='F')
    set_cache!(dad; plate_corners=_infer_plate_corners(dad, corner_bc))
    return dad
end

"""
    build_square_plate(; a=1.0, n_el=8, bc="SSSS", props=ThinPlateProps(),
                         corner_bc='C', p=2) -> BEMdata{<:ThinPlate}

Square ``[0,a]²`` via [`quadrado_plate`](@ref) then [`formatdata`](@ref)/`format2d`.
`bc` is a 4-char string for edges bottom,right,top,left (`C`/`S`/`F`).
`corner_bc`: `'C'` clamps corner w, `'F'` frees ``R_c``.
"""
function build_square_plate(; a=1.0, n_el=8, bc="SSSS",
    props::AbstractThinPlateProps=ThinPlateProps(), corner_bc='C', n_internal=1,
    p::Integer=2, internal::Union{Nothing,AbstractVector}=nothing)

    length(bc) == 4 || error("bc must have 4 characters")
    p >= 1 || error("degree must be ≥ 1, got $p")
    B = parentmodule(parentmodule(@__MODULE__))
    msh = B.quadrado_plate(; a=a, ndiv=n_el + 1, ordem=p, bc=bc,
        nome="plate_$(n_el)_p$(p)")
    dad = formatdata(msh, props; tipo=p, pontointerno=false)
    if internal !== nothing
        pts = Point2D[Point2D(pt[1], pt[2]) for pt in internal]
    else
        pts = Point2D[]
        if n_internal == 1
            push!(pts, Point2D(a / 2, a / 2))
        elseif n_internal > 1
            g = ceil(Int, sqrt(n_internal))
            xs = range(a / (g + 1), a * g / (g + 1); length=g)
            for y in xs, x in xs
                push!(pts, Point2D(x, y))
                length(pts) >= n_internal && break
            end
        end
    end
    isempty(pts) || set_internal_nodes!(dad, pts)
    prepare_plate!(dad; corner_bc=corner_bc)
    return dad
end

# =============================================================================
# Analytical — simply supported square plate (Navier)
# =============================================================================

"""
    navier_w_ss_square(x, y; a=1, q=1, D=1, nterms=40) -> w

Navier double-sine series for simply supported square plate under uniform ``q``.
"""
function navier_w_ss_square(x, y; a=1.0, q=1.0, D=1.0, nterms=40)
    w = 0.0
    for m in 1:2:nterms, n in 1:2:nterms
        den = (m^2 + n^2)^2
        w += sin(m * π * x / a) * sin(n * π * y / a) / (m * n * den)
    end
    return 16q * a^4 / (π^6 * D) * w
end

"""Maximum deflection at centre: ``≈ 0.004062 q a⁴ / D`` (ν-independent for SS)."""
function analytical_wmax_ss_square(; a=1.0, q=1.0, D=1.0, nterms=80)
    return navier_w_ss_square(a / 2, a / 2; a=a, q=q, D=D, nterms=nterms)
end

include("ThinPlateDual.jl")

end # module
