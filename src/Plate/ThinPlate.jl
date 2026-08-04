"""
    ThinPlate

Isotropic Kirchhoff (thin) plate BEM, ported from `calc_placa.jl`
(`placa_fina_isotropica` / Shi–Bezine fundamentals).

Boundary unknowns per collocation node: ``(w, ∂w/∂n)``.
Conjugate tractions: ``(V_n, M_n)``. Corner free terms: concentrated force ``R_c``.

All elements are quadratic **discontinuous** (collocation at ``ξ=±2/3,0``).
"""
module ThinPlate

using LinearAlgebra
using StaticArrays
using FastGaussQuadrature
using ProgressMeter

export ThinPlateProps, PlateCorner, PlateNode, PlateElement, PlateMesh
export bending_stiffness, build_square_plate
export assemble_plate!, apply_bc_plate, solve_plate!
export plate_w, plate_w_int
export analytical_wmax_ss_square, navier_w_ss_square

const Point2D = SVector{2,Float64}
const XI_DISC = (-2 / 3, 0.0, 2 / 3)

# =============================================================================
# Properties & mesh
# =============================================================================

"""
Isotropic Kirchhoff plate.

``D = E h³ / [12(1-ν²)]``. Distributed load ``q = q_a x + q_b y + q_c``.
"""
@kwdef mutable struct ThinPlateProps
    E::Float64 = 1.0
    ν::Float64 = 0.3
    h::Float64 = 0.01
    q_a::Float64 = 0.0
    q_b::Float64 = 0.0
    q_c::Float64 = 0.0
    ρ::Float64 = 1.0
end

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

mutable struct PlateNode
    id::Int
    pos::Point2D
    normal::Point2D
end

"""
Element BC: for each of 2 conjugate pairs and 3 local nodes
`bc_type[pair, loc]`: 0 = kinematic known (w or ∂w/∂n), 1 = static known (Vn or Mn).
`bc_val[pair, loc]`.
pair 1 → (w, Vn), pair 2 → (∂w/∂n, Mn).
"""
mutable struct PlateElement
    id::Int
    geo::NTuple{3,Int}
    fis::NTuple{3,Int}
    bc_type::Matrix{Int}     # 2×3
    bc_val::Matrix{Float64}  # 2×3
end

mutable struct PlateMesh
    geo_nodes::Vector{Point2D}
    nodes::Vector{PlateNode}
    elements::Vector{PlateElement}
    corners::Vector{PlateCorner}
    internal::Vector{Point2D}   # interior collocation (w only)
    props::ThinPlateProps
    H::Matrix{Float64}
    G::Matrix{Float64}
    q::Vector{Float64}
    u::Vector{Float64}       # [w; ∂w/∂n]_b + w_int + w_corner
    t::Vector{Float64}       # [Vn; Mn]_b + 0_int + Rc
end

function PlateMesh(geo, nodes, elems, corners, props; internal=Point2D[])
    n = length(nodes)
    ni = length(internal)
    nc = length(corners)
    ndof = 2n + ni + nc
    return PlateMesh(geo, nodes, elems, corners, internal, props,
        zeros(ndof, ndof), zeros(ndof, ndof), zeros(ndof),
        zeros(ndof), zeros(ndof))
end

"""DOF layout: 1:2n boundary (w,∂w/∂n), then ni internals (w), then nc corners (w)."""
_ndof(m::PlateMesh) = 2 * length(m.nodes) + length(m.internal) + length(m.corners)
_n(m) = length(m.nodes)
_ni(m) = length(m.internal)
_nc(m) = length(m.corners)

# shape functions
N_cont(ξ) = SVector(0.5ξ * (ξ - 1), 1 - ξ^2, 0.5ξ * (ξ + 1))
dN_cont(ξ) = SVector(ξ - 0.5, -2ξ, ξ + 0.5)
N_disc(ξ) = SVector(ξ * (9 / 8 * ξ - 3 / 4), 1 - 9 / 4 * ξ^2, ξ * (9 / 8 * ξ + 3 / 4))

function elem_geom(mesh::PlateMesh, el::PlateElement, ξ)
    g = el.geo
    X = (mesh.geo_nodes[g[1]], mesh.geo_nodes[g[2]], mesh.geo_nodes[g[3]])
    N = N_cont(ξ)
    dN = dN_cont(ξ)
    x = N[1] * X[1] + N[2] * X[2] + N[3] * X[3]
    dx = dN[1] * X[1] + dN[2] * X[2] + dN[3] * X[3]
    J = norm(dx)
    n = J > 0 ? Point2D(dx[2] / J, -dx[1] / J) : Point2D(0.0, 0.0)
    return x, J, n
end

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

# =============================================================================
# Singular integrals on discontinuous element (integraelemsing isotropic)
# =============================================================================

function integraelemsing(x1::Point2D, x3::Point2D, D, ν, xi0)
    # geometry as in reference: dx from ends with factor for quadratic span
    dx = (x3[1] - x1[1])
    dy = (x3[2] - x1[2])
    L = hypot(dx, dy)
    sx, sy = dx / L, dy / L
    nx, ny = sy, -sx
    theta = atan(dy, dx)
    D11 = D
    D22 = D
    D12 = ν * D
    D16 = 0.0
    D26 = 0.0
    D66 = (1 - ν) * D / 2

    h_el = zeros(2, 6)
    g_el = zeros(3)
    intN = (0.75, 0.5, 0.75)
    if xi0 ≈ 0
        Nlog = (
            -(L * (1 + log(8) - 3 * log(L))) / 8,
            (L * (-3 + log(L / 2))) / 4,
            -(L * (1 + log(8) - 3 * log(L))) / 8,
        )
        intNsr = (-1.5, 0.0, 1.5)
        intNsr2 = (2.25, -6.5, 2.25)
    elseif abs(xi0) ≈ 2 / 3
        Nlog = (
            (L * (-39 + 10 * log(5) - 27 * log(6) + 27 * log(L))) / 72,
            (L * (-5 + (25 * log(5)) / 6 - 3 * log(6) + 3 * log(L))) / 12,
            -(L * (3 + 25 * log(6 / 5) + log(36) - 27 * log(L))) / 72,
        )
        intNsr = ((3 * (-4 + (4 * log(5)) / 3)) / 4, 3.0, 0.0)
        intNsr2 = ((3 * (-9 / 5 - 3 * log(5))) / 4, (-9 + 6 * log(5)) / 2, (3 * (3 - log(5))) / 4)
        # if source at +2/3, reverse orientation of local node assignment later
    else
        Nlog = ((L * (-7 + 3 * log(L))) / 8, (L * log(L)) / 4, (L * (-1 + 3 * log(L))) / 8)
        intNsr = ((3 * (-10 + 5 * log(2))) / 8, 9 / 2 - (5 * log(8)) / 4, (3 * (-2 + log(2))) / 8)
        intNsr2 = (0.0, 0.0, 0.0)
    end

    for j in 1:3
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

        if xi0 ≈ -2 / 3
            h_el[:, 2j-1:2j] .= [h11 h12; h21 h22]
            g_el[j] = g22
        elseif xi0 ≈ 2 / 3 || xi0 ≈ 0
            # reverse local numbering for +2/3 (and use for 0 as reference)
            jj = 4 - j
            h_el[:, 2jj-1:2jj] .= [h11 h12; h21 h22]
            g_el[jj] = g22
        else
            h_el[:, 2j-1:2j] .= [h11 h12; h21 h22]
            g_el[j] = g22
        end
    end
    return h_el, g_el
end

# =============================================================================
# Corner free terms (compute_Rw)
# =============================================================================

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

function compute_q_el(pf::Point2D, nf::Point2D, mesh::PlateMesh, el::PlateElement,
    qsi, w, D)
    qa, qb, qc = mesh.props.q_a, mesh.props.q_b, mesh.props.q_c
    q_el = zeros(2)
    m1, m2 = nf[1], nf[2]
    xf, yf = pf[1], pf[2]
    Clocal0 = qa * xf + qb * yf + qc
    for (ig, ξ) in enumerate(qsi)
        pg, J, n = elem_geom(mesh, el, ξ)
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

# =============================================================================
# Assembly
# =============================================================================

"""
    assemble_plate!(mesh; npg=12)

Build `H`, `G` (size `2n+nc`) and load vector `q` for the Kirchhoff plate BIE.
"""
function assemble_plate!(mesh::PlateMesh; npg=12)
    n = _n(mesh)
    ni = _ni(mesh)
    nc = _nc(mesh)
    ndof = 2n + ni + nc
    H = zeros(ndof, ndof)
    G = zeros(ndof, ndof)
    qv = zeros(ndof)
    D = bending_stiffness(mesh.props)
    ν = mesh.props.ν
    qsi, w = gausslegendre(npg)
    qsi_f, w_f = gausslegendre(max(npg, 20))
    cols_c0 = 2n + ni   # corner columns start after internals

    # ---- boundary collocation (2 equations each) ----
    @showprogress "Plate H,G (boundary)" for i in 1:n
        node = mesh.nodes[i]
        pf = node.pos
        nf = node.normal
        rows = 2i-1:2i
        for el in mesh.elements
            fis = el.fis
            on_el = i in fis
            h_el = zeros(2, 6)
            g_el = zeros(2, 6)
            qs, ws = on_el ? (qsi_f, w_f) : (qsi, w)
            for (ig, ξ) in enumerate(qs)
                pg, J, n̂ = elem_geom(mesh, el, ξ)
                R = norm(pg - pf)
                R < 1e-14 && continue
                U, P = plate_kernels(pg, pf, n̂, nf, D, ν)
                Nf = N_disc(ξ)
                Nm = @SMatrix [
                    Nf[1] 0 Nf[2] 0 Nf[3] 0
                    0 Nf[1] 0 Nf[2] 0 Nf[3]
                ]
                wJ = J * ws[ig]
                # Standard plate BIE: c u + ∫ P u = ∫ U t  →  H←P, G←U
                h_el .+= P * Nm * wJ
                g_el .+= U * Nm * wJ
            end
            if on_el
                loc = findfirst(==(i), fis)
                xi0 = XI_DISC[loc]
                g1 = mesh.geo_nodes[el.geo[1]]
                g3 = mesh.geo_nodes[el.geo[3]]
                hsing, g2sing = integraelemsing(g1, g3, D, ν, xi0)
                # analytical finite-part H + singular G (∂²w/∂n∂m column)
                h_el .= hsing
                for a in 1:3
                    g_el[2, 2a] = g2sing[a]
                end
            end
            for a in 1:3
                ja = fis[a]
                cols = 2ja-1:2ja
                H[rows, cols] .+= h_el[:, 2a-1:2a]
                G[rows, cols] .+= g_el[:, 2a-1:2a]
            end
            qv[rows] .+= compute_q_el(pf, nf, mesh, el, qsi, w, D)
        end
        RS, WS = compute_Rw(pf, nf, mesh.corners, D, ν)
        # reference: H←R (corner w), G←W (corner Rc)
        H[rows, cols_c0 .+ (1:nc)] .= RS
        G[rows, cols_c0 .+ (1:nc)] .= WS
    end

    # ---- internal + corner collocation (w equation = first kernel row) ----
    nf0 = Point2D(0.0, 0.0)
    function assemble_w_row!(row, pf; free_term=1.0, sing_els=Int[])
        for el in mesh.elements
            h_row = zeros(6)
            g_row = zeros(6)
            on = el.id in sing_els
            qs, ws = on ? (qsi_f, w_f) : (qsi, w)
            for (ig, ξ) in enumerate(qs)
                pg, J, n̂ = elem_geom(mesh, el, ξ)
                R = norm(pg - pf)
                R < 1e-14 && continue
                U, P = plate_kernels(pg, pf, n̂, nf0, D, ν)
                Nf = N_disc(ξ)
                wJ = J * ws[ig]
                for a in 1:3
                    h_row[2a-1] += P[1, 1] * Nf[a] * wJ
                    h_row[2a]   += P[1, 2] * Nf[a] * wJ
                    g_row[2a-1] += U[1, 1] * Nf[a] * wJ
                    g_row[2a]   += U[1, 2] * Nf[a] * wJ
                end
            end
            if on
                # source at element end → ξ = ±1 style; use nearest disc node
                xi0 = el.id == sing_els[1] ? -1.0 : 1.0
                g1 = mesh.geo_nodes[el.geo[1]]
                g3 = mesh.geo_nodes[el.geo[3]]
                hsing, _ = integraelemsing(g1, g3, D, ν, xi0)
                # only first row of H
                for a in 1:3
                    h_row[2a-1] = hsing[1, 2a-1]
                    h_row[2a]   = hsing[1, 2a]
                end
            end
            for a in 1:3
                ja = el.fis[a]
                H[row, 2ja-1:2ja] .+= h_row[2a-1:2a]
                G[row, 2ja-1:2ja] .+= g_row[2a-1:2a]
            end
            qe = compute_q_el(pf, nf0, mesh, el, qsi, w, D)
            qv[row] += qe[1]
        end
        RS, WS = compute_Rw(pf, nf0, mesh.corners, D, ν)
        H[row, cols_c0 .+ (1:nc)] .+= RS[1, :]
        G[row, cols_c0 .+ (1:nc)] .+= WS[1, :]
        H[row, row] += free_term
        return nothing
    end

    @showprogress "Plate H,G (internal)" for k in 1:ni
        assemble_w_row!(2n + k, mesh.internal[k]; free_term=1.0)
    end
    @showprogress "Plate H,G (corners)" for c in 1:nc
        corner = mesh.corners[c]
        # collocate at corner; singular on the two adjacent elements
        assemble_w_row!(cols_c0 + c, corner.pos;
            free_term=0.0,  # free term from compute_Rw when r=0 (betac)
            sing_els=[corner.el_prev, corner.el_next])
    end

    # free term on boundary
    @inbounds for i in 1:n
        H[2i-1:2i, 2i-1:2i] .+= (I(2) ./ 2)
    end

    mesh.H = H
    mesh.G = G
    mesh.q = qv
    return mesh
end

# =============================================================================
# BC & solve
# =============================================================================

"""
Apply mixed BCs. Kinematic known → swap columns (scale tractions by `D`).
Returns `(A, b, is_kin, known_val)`.
"""
function apply_bc_plate(mesh::PlateMesh)
    H = copy(mesh.H)
    G = copy(mesh.G)
    Dsc = bending_stiffness(mesh.props)
    n = _n(mesh)
    ni = _ni(mesh)
    nc = _nc(mesh)
    ndof = 2n + ni + nc
    is_kin = falses(ndof)
    known = zeros(ndof)

    for el in mesh.elements
        for loc in 1:3
            j = el.fis[loc]
            for pair in 1:2
                dof = 2(j - 1) + pair
                if el.bc_type[pair, loc] == 0
                    is_kin[dof] = true
                    known[dof] = el.bc_val[pair, loc]
                    colH = H[:, dof]
                    colG = G[:, dof]
                    H[:, dof] = -colG .* Dsc
                    G[:, dof] = -colH
                else
                    known[dof] = el.bc_val[pair, loc]
                end
            end
        end
    end
    # internal w always unknown (kinematic free) — nothing to swap
    for (c, corner) in enumerate(mesh.corners)
        dof = 2n + ni + c
        if corner.bc_type == 0
            is_kin[dof] = true
            known[dof] = corner.bc_val
            colH = H[:, dof]
            colG = G[:, dof]
            H[:, dof] = -colG .* Dsc
            G[:, dof] = -colH
        else
            known[dof] = corner.bc_val
        end
    end

    b = mesh.q .+ G * known
    return H, b, is_kin, known
end

function solve_plate!(mesh::PlateMesh)
    A, b, is_kin, known = apply_bc_plate(mesh)
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

"""Interior deflection at internal point `k`."""
plate_w_int(mesh::PlateMesh, k::Int=1) = mesh.u[2 * _n(mesh) + k]

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

function _push_geo!(geo, p)
    push!(geo, Point2D(p))
    return length(geo)
end

function _make_edge!(geo, nodes, elems, p0, p1, n_el, bc)
    tw, vw, tm, vm = bc
    geo_ids = Int[]
    for k in 0:(2 * n_el)
        s = k / (2 * n_el)
        push!(geo_ids, _push_geo!(geo, (1 - s) * p0 + s * p1))
    end
    t_dir = p1 - p0
    L = norm(t_dir)
    n_hat = L > 0 ? Point2D(t_dir[2] / L, -t_dir[1] / L) : Point2D(0.0, 1.0)
    fis_all = Int[]
    for e in 1:n_el
        g1 = geo_ids[2e - 1]
        g2 = geo_ids[2e]
        g3 = geo_ids[2e + 1]
        Xg = (geo[g1], geo[g2], geo[g3])
        fis = Int[]
        for a in 1:3
            ξ = XI_DISC[a]
            N = N_cont(ξ)
            pos = N[1] * Xg[1] + N[2] * Xg[2] + N[3] * Xg[3]
            id = length(nodes) + 1
            push!(nodes, PlateNode(id, pos, n_hat))
            push!(fis, id)
            push!(fis_all, id)
        end
        bt = fill(1, 2, 3)
        bv = zeros(2, 3)
        for loc in 1:3
            bt[1, loc] = tw; bv[1, loc] = vw
            bt[2, loc] = tm; bv[2, loc] = vm
        end
        id_el = length(elems) + 1
        push!(elems, PlateElement(id_el, (g1, g2, g3), (fis[1], fis[2], fis[3]), bt, bv))
    end
    return fis_all, geo_ids, n_hat
end

"""
    build_square_plate(; a=1.0, n_el=8, bc="SSSS", props=ThinPlateProps(),
                         corner_bc='C') -> PlateMesh

Square ``[0,a]²``. `bc` is a 4-char string for edges bottom,right,top,left
(`C`/`S`/`F`). `corner_bc`: `'C'` clamps corner w, `'F'` frees ``R_c``.
"""
function build_square_plate(; a=1.0, n_el=8, bc="SSSS",
    props=ThinPlateProps(), corner_bc='C', n_internal=1)

    length(bc) == 4 || error("bc must have 4 characters")
    geo = Point2D[]
    nodes = PlateNode[]
    elems = PlateElement[]

    pts = (Point2D(0, 0), Point2D(a, 0), Point2D(a, a), Point2D(0, a))
    edges = ((1, 2), (2, 3), (3, 4), (4, 1))
    edge_normals = Point2D[]
    edge_el_range = Tuple{Int,Int}[]

    for (e, (i0, i1)) in enumerate(edges)
        el0 = length(elems) + 1
        _, _, nh = _make_edge!(geo, nodes, elems, pts[i0], pts[i1], n_el,
            _bc_from_char(bc[e]))
        push!(edge_normals, nh)
        push!(edge_el_range, (el0, length(elems)))
    end

    corners = PlateCorner[]
    cbc = uppercase(corner_bc)
    for k in 1:4
        kprev = mod1(k - 1, 4)
        n_prev = edge_normals[kprev]
        n_next = edge_normals[k]
        el_prev = edge_el_range[kprev][2]
        el_next = edge_el_range[k][1]
        bt = cbc == 'F' ? 1 : 0
        push!(corners, PlateCorner(k, pts[k], n_prev, n_next, el_prev, el_next, bt, 0.0))
    end

    # internal points (grid, excluding boundary)
    internal = Point2D[]
    if n_internal == 1
        push!(internal, Point2D(a / 2, a / 2))
    elseif n_internal > 1
        g = ceil(Int, sqrt(n_internal))
        xs = range(a / (g + 1), a * g / (g + 1); length=g)
        for y in xs, x in xs
            push!(internal, Point2D(x, y))
            length(internal) >= n_internal && break
        end
    end

    return PlateMesh(geo, nodes, elems, corners, props; internal=internal)
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

end # module
