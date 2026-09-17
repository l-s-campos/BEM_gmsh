# Anisotropic Kirchhoff (Shi–Bezine / Lekhnitskii), ported from
# `calc_placa.jl` (`placa_fina` / `calsolfund`). Included by ThinPlate.
# Isotropic μ₁=μ₂=i makes C1,C2 singular — keep ThinPlateProps for that case.

"""
Anisotropic Kirchhoff plate. Bending stiffness ``D_{ij}`` (Voigt) and
Lekhnitskii parameters ``μ_k = d_k + i e_k`` (Im>0, ordered by Re).

``C_1,C_2,C_3`` are the Shi–Bezine combination coefficients.
Traction columns in `apply_bc_plate` are scaled by ``D_{22}``.
"""
@kwdef mutable struct AnisoThinPlateProps <: AbstractThinPlate
    D11::Float64
    D22::Float64
    D12::Float64
    D16::Float64 = 0.0
    D26::Float64 = 0.0
    D66::Float64
    d::SVector{2,Float64}
    e::SVector{2,Float64}
    C1::Float64
    C2::Float64
    C3::Float64
    q_a::Float64 = 0.0
    q_b::Float64 = 0.0
    q_c::Float64 = 0.0
    ρ::Float64 = 1.0
    h::Float64 = 0.01
end

bending_stiffness(p::AnisoThinPlateProps) = p.D22

function fundamental(props::AnisoThinPlateProps, r::SVector{2}, n::SVector{2}, nf::SVector{2})
    return plate_kernels(r, zero(r), n, nf, props)
end

"""
    lekhnitskii_roots(D11, D22, D12, D16, D26, D66) -> (d, e)

Roots of ``D_{22} μ⁴ + 4 D_{26} μ³ + 2(D_{12}+2 D_{66}) μ² + 4 D_{16} μ + D_{11} = 0``
(``w=f(x+μ y)``, same array as `PolynomialRoots` in `placa_fina`) with Im>0,
ordered by Re. Errors on the isotropic repeated-root limit.
"""
function lekhnitskii_roots(D11, D22, D12, D16, D26, D66)
    # Lowest-first [D11, 4 D16, 2(D12+2 D66), 4 D26, D22] as in calc_placa.jl.
    a4 = float(D22)
    a4 == 0 && error("lekhnitskii_roots: D22 must be nonzero")
    a3 = 4 * float(D26)
    a2 = 2 * float(D12) + 4 * float(D66)
    a1 = 4 * float(D16)
    a0 = float(D11)
    # companion of μ⁴ + (a3/a4) μ³ + (a2/a4) μ² + (a1/a4) μ + a0/a4
    C = zeros(4, 4)
    C[2, 1] = 1
    C[3, 2] = 1
    C[4, 3] = 1
    C[1, 4] = -a0 / a4
    C[2, 4] = -a1 / a4
    C[3, 4] = -a2 / a4
    C[4, 4] = -a3 / a4
    rts = eigvals(C)
    pos = [z for z in rts if imag(z) > 0]
    if length(pos) != 2
        pos = sort(collect(rts); by=imag, rev=true)[1:2]
    end
    sort!(pos; by=real)
    μ1, μ2 = pos[1], pos[2]
    if imag(μ1) < 1e-12 || imag(μ2) < 1e-12
        error("lekhnitskii_roots: non-elliptic Dᵢⱼ (Im(μ)≈0)")
    end
    if abs(μ1 - μ2) < 1e-8 * max(abs(μ1), abs(μ2), 1.0)
        error("lekhnitskii_roots: repeated μ (isotropic limit); use ThinPlateProps")
    end
    d = SVector(real(μ1), real(μ2))
    e = SVector(imag(μ1), imag(μ2))
    return d, e
end

function _lekhnitskii_C(d, e)
    Δd = d[1] - d[2]
    G = Δd^2 + (e[1] + e[2])^2
    H = Δd^2 + (e[1] - e[2])^2
    den = G * H
    if den * e[1] * e[2] < 1e-30
        error("Lekhnitskii C coefficients singular (repeated μ); use ThinPlateProps")
    end
    C1 = (Δd^2 - (e[1]^2 - e[2]^2)) / (den * e[1])
    C2 = (Δd^2 + (e[1]^2 - e[2]^2)) / (den * e[2])
    C3 = 4 * Δd / den
    return C1, C2, C3
end

"""
    aniso_thin_plate_props(; D11, D22, D12, D66, D16=0, D26=0, kwargs...)
    aniso_thin_plate_props(D::AbstractMatrix; kwargs...)

Build [`AnisoThinPlateProps`](@ref) from Voigt ``D_{ij}`` (no isotropic smearing).
`D` is ``3×3`` ``[D₁₁ D₁₂ D₁₆; · D₂₂ D₂₆; · · D₆₆]``.
"""
function aniso_thin_plate_props(; D11, D22, D12, D66, D16=0.0, D26=0.0,
        q_a=0.0, q_b=0.0, q_c=0.0, ρ=1.0, h=0.01)
    d, e = lekhnitskii_roots(D11, D22, D12, D16, D26, D66)
    C1, C2, C3 = _lekhnitskii_C(d, e)
    return AnisoThinPlateProps(; D11=float(D11), D22=float(D22), D12=float(D12),
        D16=float(D16), D26=float(D26), D66=float(D66), d=d, e=e,
        C1=C1, C2=C2, C3=C3, q_a=float(q_a), q_b=float(q_b), q_c=float(q_c),
        ρ=float(ρ), h=float(h))
end

function aniso_thin_plate_props(D::AbstractMatrix; kwargs...)
    size(D) == (3, 3) || error("D must be 3×3 Voigt")
    return aniso_thin_plate_props(; D11=D[1, 1], D22=D[2, 2], D12=D[1, 2],
        D16=D[1, 3], D26=D[2, 3], D66=D[3, 3], kwargs...)
end

# -----------------------------------------------------------------------------
# R,S potentials at one Lekhnitskii root (a=1)
# -----------------------------------------------------------------------------

function _RS_at(r, θ, d, e)
    s, c = sincos(θ)
    α = c + d * s
    β = e * s
    ρ2 = α * α + β * β
    L = log(r * r * ρ2)
    At = atan(β, α)
    r2 = r * r
    γ = α * α - e * e * s * s
    R = r2 * γ * (L - 3) - 4 * r2 * e * s * α * At
    S = r2 * e * s * α * (L - 3) + r2 * γ * At
    dRdx = 2 * r * α * (L - 2) - 4 * r * e * s * At
    dRdy = 2 * r * (d * α - e * e * s) * (L - 2) - 4 * r * e * (c + 2 * d * s) * At
    d2Rdx2 = 2 * L
    d2Rdxdy = 2 * d * L - 4 * e * At
    d2Rdy2 = 2 * (d * d - e * e) * L - 8 * d * e * At
    d3Rdx3 = 4 * α / (r * ρ2)
    d3Rdx2dy = 4 * (d * α + e * e * s) / (r * ρ2)
    d3Rdxdy2 = 4 * ((d * d - e * e) * c + (d * d + e * e) * d * s) / (r * ρ2)
    d3Rdy3 = 4 * (d * (d * d - 3 * e * e) * c + (d^4 - e^4) * s) / (r * ρ2)
    ρ4 = ρ2 * ρ2
    d4Rdx4 = -4 * γ / (r2 * ρ4)
    d4Rdx3dy = -4 / r2 * (d / ρ2 + 2 * e * e * s * c / ρ4)
    d4Rdx2dy2 = -4 / r2 * ((d * d + e * e) / ρ2 - 2 * e * e * c * c / ρ4)
    d4Rdxdy3 = -4 / r2 * (
        d * (d * d + e * e) / ρ2 -
        2 * e * e * c * (2 * d * c + (d * d + e * e) * s) / ρ4)
    d4Rdy4 = -4 / r2 * (
        (d^4 - e^4) / ρ2 -
        2 * e * e * c * ((3 * d * d - e * e) * c + 2 * d * (d * d + e * e) * s) / ρ4)
    dSdx = r * e * s * (L - 2) + 2 * r * α * At
    dSdy = r * e * (c + 2 * d * s) * (L - 2) + 2 * r * (d * α - e * e * s) * At
    d2Sdx2 = 2 * At
    d2Sdxdy = e * L + 2 * d * At
    d2Sdy2 = 2 * d * e * L + 2 * (d * d - e * e) * At
    d3Sdx3 = -2 * e * s / (r * ρ2)
    d3Sdx2dy = 2 * e * c / (r * ρ2)
    d3Sdxdy2 = 2 * e * (2 * d * α - (d * d - e * e) * s) / (r * ρ2)
    d3Sdy3 = 2 * e * ((3 * d * d - e * e) * c + 2 * d * (d * d + e * e) * s) / (r * ρ2)
    d4Sdx4 = 4 * e * s * α / (r2 * ρ4)
    d4Sdx3dy = 2 * e / r2 * (1 / ρ2 - 2 * c * α / ρ4)
    d4Sdx2dy2 = -4 * e * c * (d * α + e * e * s) / (r2 * ρ4)
    d4Sdxdy3 = -2 * e / r2 * (
        (d * d + e * e) / ρ2 +
        (2 * (d * d + e * e) * c * α - 4 * e * e * c * c) / ρ4)
    d4Sdy4 = -4 * e / r2 * (
        d * (d * d + e * e) / ρ2 +
        c * (d * (d * d - 3 * e * e) * c + (d^4 - e^4) * s) / ρ4)
    return (; R, S, dRdx, dRdy, d2Rdx2, d2Rdxdy, d2Rdy2,
        d3Rdx3, d3Rdx2dy, d3Rdxdy2, d3Rdy3,
        d4Rdx4, d4Rdx3dy, d4Rdx2dy2, d4Rdxdy3, d4Rdy4,
        dSdx, dSdy, d2Sdx2, d2Sdxdy, d2Sdy2,
        d3Sdx3, d3Sdx2dy, d3Sdxdy2, d3Sdy3,
        d4Sdx4, d4Sdx3dy, d4Sdx2dy2, d4Sdxdy3, d4Sdy4)
end

# calsolfund scaling: w / D22, derivatives without D22 (then /D22 on U,P).
function _aniso_fundamentals(r, θ, p::AnisoThinPlateProps)
    z1 = _RS_at(r, θ, p.d[1], p.e[1])
    z2 = _RS_at(r, θ, p.d[2], p.e[2])
    C1, C2, C3 = p.C1, p.C2, p.C3
    s8 = 8π
    combS(a1, a2, b1, b2) = (C1 * a1 + C2 * a2 + C3 * (b1 - b2)) / s8
    w = combS(z1.R, z2.R, z1.S, z2.S) / p.D22
    dwdx = combS(z1.dRdx, z2.dRdx, z1.dSdx, z2.dSdx)
    dwdy = combS(z1.dRdy, z2.dRdy, z1.dSdy, z2.dSdy)
    d2wdx2 = combS(z1.d2Rdx2, z2.d2Rdx2, z1.d2Sdx2, z2.d2Sdx2)
    d2wdxdy = combS(z1.d2Rdxdy, z2.d2Rdxdy, z1.d2Sdxdy, z2.d2Sdxdy)
    d2wdy2 = combS(z1.d2Rdy2, z2.d2Rdy2, z1.d2Sdy2, z2.d2Sdy2)
    d3wdx3 = combS(z1.d3Rdx3, z2.d3Rdx3, z1.d3Sdx3, z2.d3Sdx3)
    d3wdx2dy = combS(z1.d3Rdx2dy, z2.d3Rdx2dy, z1.d3Sdx2dy, z2.d3Sdx2dy)
    d3wdxdy2 = combS(z1.d3Rdxdy2, z2.d3Rdxdy2, z1.d3Sdxdy2, z2.d3Sdxdy2)
    d3wdy3 = combS(z1.d3Rdy3, z2.d3Rdy3, z1.d3Sdy3, z2.d3Sdy3)
    d4wdx4 = combS(z1.d4Rdx4, z2.d4Rdx4, z1.d4Sdx4, z2.d4Sdx4)
    d4wdx3dy = combS(z1.d4Rdx3dy, z2.d4Rdx3dy, z1.d4Sdx3dy, z2.d4Sdx3dy)
    d4wdx2dy2 = combS(z1.d4Rdx2dy2, z2.d4Rdx2dy2, z1.d4Sdx2dy2, z2.d4Sdx2dy2)
    d4wdxdy3 = combS(z1.d4Rdxdy3, z2.d4Rdxdy3, z1.d4Sdxdy3, z2.d4Sdxdy3)
    d4wdy4 = combS(z1.d4Rdy4, z2.d4Rdy4, z1.d4Sdy4, z2.d4Sdy4)
    return (; w, dwdx, dwdy, d2wdx2, d2wdxdy, d2wdy2,
        d3wdx3, d3wdx2dy, d3wdxdy2, d3wdy3,
        d4wdx4, d4wdx3dy, d4wdx2dy2, d4wdxdy3, d4wdy4)
end

function _aniso_f123(nx, ny, p::AnisoThinPlateProps)
    f1 = p.D11 * nx^2 + 2 * p.D16 * nx * ny + p.D12 * ny^2
    f2 = 2 * (p.D16 * nx^2 + 2 * p.D66 * nx * ny + p.D26 * ny^2)
    f3 = p.D12 * nx^2 + 2 * p.D26 * nx * ny + p.D22 * ny^2
    return f1, f2, f3
end

function _aniso_h1234(nx, ny, p::AnisoThinPlateProps)
    h1 = p.D11 * nx * (1 + ny^2) + 2 * p.D16 * ny^3 - p.D12 * nx * ny^2
    h2 = 4 * p.D16 * nx + p.D12 * ny * (1 + nx^2) + 4 * p.D66 * ny^3 -
         p.D11 * nx^2 * ny - 2 * p.D26 * nx * ny^2
    h3 = 4 * p.D26 * ny + p.D12 * nx * (1 + ny^2) + 4 * p.D66 * nx^3 -
         p.D22 * nx * ny^2 - 2 * p.D16 * nx^2 * ny
    h4 = p.D22 * ny * (1 + nx^2) + 2 * p.D26 * nx^3 - p.D12 * nx^2 * ny
    return h1, h2, h3, h4
end

"""Anisotropic Shi–Bezine kernels. Same 2×2 layout as the isotropic pair."""
function plate_kernels(pg::Point2D, pf::Point2D, n::Point2D, nf::Point2D,
        p::AnisoThinPlateProps)
    rs = pg - pf
    r = norm(rs)
    r < 1e-30 && error("plate_kernels: coincident points")
    θ = atan(rs[2], rs[1])
    nx, ny = n[1], n[2]
    m1, m2 = nf[1], nf[2]
    F = _aniso_fundamentals(r, θ, p)
    f1, f2, f3 = _aniso_f123(nx, ny, p)
    h1, h2, h3, h4 = _aniso_h1234(nx, ny, p)
    D22 = p.D22
    dwdn = (F.dwdx * nx + F.dwdy * ny) / D22
    mn = -(f1 * F.d2wdx2 + f2 * F.d2wdxdy + f3 * F.d2wdy2) / D22
    vn = -(h1 * F.d3wdx3 + h2 * F.d3wdx2dy + h3 * F.d3wdxdy2 + h4 * F.d3wdy3) / D22
    dmndx = -(f1 * F.d3wdx3 + f2 * F.d3wdx2dy + f3 * F.d3wdxdy2)
    dmndy = -(f1 * F.d3wdx2dy + f2 * F.d3wdxdy2 + f3 * F.d3wdy3)
    dvndx = -(h1 * F.d4wdx4 + h2 * F.d4wdx3dy + h3 * F.d4wdx2dy2 + h4 * F.d4wdxdy3)
    dvndy = -(h1 * F.d4wdx3dy + h2 * F.d4wdx2dy2 + h3 * F.d4wdxdy3 + h4 * F.d4wdy4)
    dwdm = -(F.dwdx * m1 + F.dwdy * m2) / D22
    d2wdndm = -(F.d2wdx2 * nx * m1 + F.d2wdxdy * (nx * m2 + ny * m1) +
                F.d2wdy2 * ny * m2) / D22
    dmndm = -(dmndx * m1 + dmndy * m2) / D22
    dvndm = -(dvndx * m1 + dvndy * m2) / D22
    U = @SMatrix [F.w -dwdn; dwdm -d2wdndm]
    P = @SMatrix [vn -mn; dvndm -dmndm]
    return U, P
end

function compute_Rw(pf::Point2D, nf::Point2D, corners::Vector{PlateCorner},
        p::AnisoThinPlateProps)
    nc = length(corners)
    RS = zeros(2, nc)
    WS = zeros(2, nc)
    m1, m2 = nf[1], nf[2]
    D22 = p.D22
    for (i, c) in enumerate(corners)
        pc = c.pos
        na1, na2 = c.n_prev[1], c.n_prev[2]
        nd1, nd2 = c.n_next[1], c.n_next[2]
        sa1, sa2 = -na2, na1
        sd1, sd2 = -nd2, nd1
        rs = pc - pf
        r = norm(rs)
        if r > 1e-14
            θ = atan(rs[2], rs[1])
            F = _aniso_fundamentals(r, θ, p)
            g1a = (p.D12 - p.D11) * na1 * na2 + p.D16 * (na1^2 - na2^2)
            g2a = 2 * (p.D26 - p.D16) * na1 * na2 + 2 * p.D66 * (na1^2 - na2^2)
            g3a = (p.D22 - p.D12) * na1 * na2 + p.D26 * (na1^2 - na2^2)
            g1d = (p.D12 - p.D11) * nd1 * nd2 + p.D16 * (nd1^2 - nd2^2)
            g2d = 2 * (p.D26 - p.D16) * nd1 * nd2 + 2 * p.D66 * (nd1^2 - nd2^2)
            g3d = (p.D22 - p.D12) * nd1 * nd2 + p.D26 * (nd1^2 - nd2^2)
            tna = -(g1a * F.d2wdx2 + g2a * F.d2wdxdy + g3a * F.d2wdy2)
            tnd = -(g1d * F.d2wdx2 + g2d * F.d2wdxdy + g3d * F.d2wdy2)
            dtndxa = -(g1a * F.d3wdx3 + g2a * F.d3wdx2dy + g3a * F.d3wdxdy2)
            dtndya = -(g1a * F.d3wdx2dy + g2a * F.d3wdxdy2 + g3a * F.d3wdy3)
            dtndxd = -(g1d * F.d3wdx3 + g2d * F.d3wdx2dy + g3d * F.d3wdxdy2)
            dtndyd = -(g1d * F.d3wdx2dy + g2d * F.d3wdxdy2 + g3d * F.d3wdy3)
            Rci = (tnd - tna) / D22
            dRci = -((dtndxd * m1 + dtndyd * m2) - (dtndxa * m1 + dtndya * m2)) / D22
            dwdm = -(F.dwdx * m1 + F.dwdy * m2) / D22
            RS[:, i] .= (Rci, dRci)
            WS[:, i] .= (F.w, dwdm)
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

# Radial integrals of R,S for particular-solution RIM (compute_q, a=1).
function _rim_RS_i(r, θ, d, e, Clocal, A, B)
    s, c = sincos(θ)
    α = c + d * s
    β = e * s
    ρ2 = α * α + β * β
    L = log(r * r * ρ2)
    At = atan(β, α)
    c2, s2 = cos(2θ), sin(2θ)
    γlog = -1 - d^2 + e^2 + (-1 + d^2 - e^2) * c2 - 2 * d * s2
    γat = 1 + d^2 - e^2 + (1 - d^2 + e^2) * c2 + 2 * d * s2
    r2, r3, r4 = r * r, r^3, r^4
    AB = A * c + B * s
    intRkrdr = Clocal * (r3 * (-16 * e * At * s * α - (-7 + 2 * L) * γlog)) / 16
    intSkrdr = Clocal * (r3 * (2 * e * (-7 + 2 * L) * s * α + 2 * At * γat)) / 16
    intRABrdr = (r4 * AB * (-40 * e * At * s * α - (-17 + 5 * L) * γlog)) / 50
    intSABrdr = (r4 * AB * (2 * e * (-17 + 5 * L) * s * α + 5 * At * γat)) / 50
    intdRdxABrdr = (r3 * AB * (-4 * e * At * s + (-5 + 2 * L) * α)) / 4
    intdRdyABrdr = (r3 * AB * (-4 * e * At * (c + 2 * d * s) +
                               (-5 + 2 * L) * (d * c + (d^2 - e^2) * s))) / 4
    intdSdxABrdr = (r3 * AB * (e * (-5 + 2 * L) * s + 4 * At * α)) / 8
    intdSdyABrdr = (r3 * AB * (e * (-5 + 2 * L) * (c + 2 * d * s) +
                               4 * At * (d * c + (d^2 - e^2) * s))) / 8
    intdRdxrdr = Clocal * (2 * r2 * (-6 * e * At * s + (-8 + 3 * L) * α)) / 9
    intdRdyrdr = Clocal * (2 * r2 * (-6 * e * At * (c + 2 * d * s) +
                                     (-8 + 3 * L) * (d * c + (d^2 - e^2) * s))) / 9
    intdSdxrdr = Clocal * (r2 * (e * (-8 + 3 * L) * s + 6 * At * α)) / 9
    intdSdyrdr = Clocal * (r2 * (e * (-8 + 3 * L) * (c + 2 * d * s) +
                                 6 * At * (d * c + (d^2 - e^2) * s))) / 9
    return (; intRkrdr, intSkrdr, intRABrdr, intSABrdr,
        intdRdxABrdr, intdRdyABrdr, intdSdxABrdr, intdSdyABrdr,
        intdRdxrdr, intdRdyrdr, intdSdxrdr, intdSdyrdr)
end

function compute_q_el(pf::Point2D, nf::Point2D, mesh, el::Element,
        qsi, w, p::AnisoThinPlateProps)
    A, B, Cload = p.q_a, p.q_b, p.q_c
    q_el = zeros(2)
    m1, m2 = nf[1], nf[2]
    xf, yf = pf[1], pf[2]
    Clocal = A * xf + B * yf + Cload
    C1, C2, C3 = p.C1, p.C2, p.C3
    D22 = p.D22
    for (ig, ξ) in enumerate(qsi)
        pg, J, n = elem_geom(el, ξ)
        rs = pg - pf
        r = norm(rs)
        r < 1e-14 && continue
        θ = atan(rs[2], rs[1])
        nr = n[1] * rs[1] / r + n[2] * rs[2] / r
        z1 = _rim_RS_i(r, θ, p.d[1], p.e[1], Clocal, A, B)
        z2 = _rim_RS_i(r, θ, p.d[2], p.e[2], Clocal, A, B)
        int1 = (C1 * (z1.intRkrdr + z1.intRABrdr) +
                C2 * (z2.intRkrdr + z2.intRABrdr) +
                C3 * ((z1.intSkrdr + z1.intSABrdr) - (z2.intSkrdr + z2.intSABrdr))) *
               nr / (8π * D22)
        intdwdx = (C1 * (z1.intdRdxrdr + z1.intdRdxABrdr) +
                   C2 * (z2.intdRdxrdr + z2.intdRdxABrdr) +
                   C3 * ((z1.intdSdxrdr + z1.intdSdxABrdr) -
                         (z2.intdSdxrdr + z2.intdSdxABrdr))) / (8π)
        intdwdy = (C1 * (z1.intdRdyrdr + z1.intdRdyABrdr) +
                   C2 * (z2.intdRdyrdr + z2.intdRdyABrdr) +
                   C3 * ((z1.intdSdyrdr + z1.intdSdyABrdr) -
                         (z2.intdSdyrdr + z2.intdSdyABrdr))) / (8π)
        int2 = -(intdwdx * m1 + intdwdy * m2) * nr / D22
        q_el .+= (int1, int2) .* (J * w[ig])
    end
    return q_el
end

# Closed-form CPV/HFP of P and G₂₂ on a straight element (integraelemsing).
function integraelemsing(x1::Point2D, x3::Point2D, p::AnisoThinPlateProps, xi0, poly)
    dx = x3[1] - x1[1]
    dy = x3[2] - x1[2]
    L = hypot(dx, dy)
    nx, ny = dy / L, -dx / L
    θ = atan(dy, dx)
    intN, IlogN, intNsr, intNsr2 = _poly_moments(poly, xi0)
    Nlog = (L / 2) .* (IlogN .+ log(L / 2) .* intN)
    nN = length(intN)
    h_el = zeros(2, 2nN)
    g_el = zeros(nN)
    C1, C2, C3 = p.C1, p.C2, p.C3
    D22 = p.D22
    f1, f2, f3 = _aniso_f123(nx, ny, p)
    hh1, hh2, hh3, hh4 = _aniso_h1234(nx, ny, p)
    s, c = sincos(θ)
    for j in 1:nN
        I0 = intN[j] * L / 2
        Ilog = Nlog[j]
        I1 = (2 * intNsr[j] / L) * L / 2
        I2 = (4 * intNsr2[j] / L^2) * L / 2
        function RS_int(d, e)
            α = c + d * s
            ρ2 = α * α + e * e * s * s
            At = atan(e * s, α)
            lgA = log(ρ2)
            d2Rdx2 = 4 * Ilog + 2 * lgA * I0
            d2Rdxdy = 4 * d * Ilog + (2 * d * lgA - 4 * e * At) * I0
            d2Rdy2 = 4 * (d * d - e * e) * Ilog +
                     (2 * (d * d - e * e) * lgA - 8 * d * e * At) * I0
            d2Sdx2 = 2 * At * I0
            d2Sdxdy = 2 * e * Ilog + (e * lgA + 2 * d * At) * I0
            d2Sdy2 = 4 * d * e * Ilog + (2 * d * e * lgA + 2 * (d * d - e * e) * At) * I0
            d3Rdx3 = 4 * I1 * α / ρ2
            d3Rdx2dy = 4 * I1 * (d * α + e * e * s) / ρ2
            d3Rdxdy2 = 4 * I1 * ((d * d - e * e) * c + (d * d + e * e) * d * s) / ρ2
            d3Rdy3 = 4 * I1 * (d * (d * d - 3 * e * e) * c + (d^4 - e^4) * s) / ρ2
            ρ4 = ρ2 * ρ2
            d4Rdx4 = -4 * I2 * (α * α - e * e * s * s) / ρ4
            d4Rdx3dy = -4 * I2 * (d / ρ2 + 2 * e * e * s * c / ρ4)
            d4Rdx2dy2 = -4 * I2 * ((d * d + e * e) / ρ2 - 2 * e * e * c * c / ρ4)
            d4Rdxdy3 = -4 * I2 * (
                d * (d * d + e * e) / ρ2 -
                2 * e * e * c * (2 * d * c + (d * d + e * e) * s) / ρ4)
            d4Rdy4 = -4 * I2 * (
                (d^4 - e^4) / ρ2 -
                2 * e * e * c * ((3 * d * d - e * e) * c + 2 * d * (d * d + e * e) * s) / ρ4)
            d3Sdx3 = -2 * I1 * e * s / ρ2
            d3Sdx2dy = 2 * I1 * e * c / ρ2
            d3Sdxdy2 = 2 * I1 * e * (2 * d * α - (d * d - e * e) * s) / ρ2
            d3Sdy3 = 2 * I1 * e * ((3 * d * d - e * e) * c + 2 * d * (d * d + e * e) * s) / ρ2
            d4Sdx4 = 4 * I2 * e * s * α / ρ4
            d4Sdx3dy = 2 * I2 * e * (1 / ρ2 - 2 * c * α / ρ4)
            d4Sdx2dy2 = -4 * I2 * e * c * (d * α + e * e * s) / ρ4
            d4Sdxdy3 = -2 * I2 * e * (
                (d * d + e * e) / ρ2 +
                (2 * (d * d + e * e) * c * α - 4 * e * e * c * c) / ρ4)
            d4Sdy4 = -4 * I2 * e * (
                d * (d * d + e * e) / ρ2 +
                c * (d * (d * d - 3 * e * e) * c + (d^4 - e^4) * s) / ρ4)
            return (; d2Rdx2, d2Rdxdy, d2Rdy2, d2Sdx2, d2Sdxdy, d2Sdy2,
                d3Rdx3, d3Rdx2dy, d3Rdxdy2, d3Rdy3,
                d4Rdx4, d4Rdx3dy, d4Rdx2dy2, d4Rdxdy3, d4Rdy4,
                d3Sdx3, d3Sdx2dy, d3Sdxdy2, d3Sdy3,
                d4Sdx4, d4Sdx3dy, d4Sdx2dy2, d4Sdxdy3, d4Sdy4)
        end
        z1 = RS_int(p.d[1], p.e[1])
        z2 = RS_int(p.d[2], p.e[2])
        s8D = 8π * D22
        combS(a1, a2, b1, b2) = (C1 * a1 + C2 * a2 + C3 * (b1 - b2)) / s8D
        d2wdx2 = combS(z1.d2Rdx2, z2.d2Rdx2, z1.d2Sdx2, z2.d2Sdx2)
        d2wdxdy = combS(z1.d2Rdxdy, z2.d2Rdxdy, z1.d2Sdxdy, z2.d2Sdxdy)
        d2wdy2 = combS(z1.d2Rdy2, z2.d2Rdy2, z1.d2Sdy2, z2.d2Sdy2)
        d3wdx3 = combS(z1.d3Rdx3, z2.d3Rdx3, z1.d3Sdx3, z2.d3Sdx3)
        d3wdx2dy = combS(z1.d3Rdx2dy, z2.d3Rdx2dy, z1.d3Sdx2dy, z2.d3Sdx2dy)
        d3wdxdy2 = combS(z1.d3Rdxdy2, z2.d3Rdxdy2, z1.d3Sdxdy2, z2.d3Sdxdy2)
        d3wdy3 = combS(z1.d3Rdy3, z2.d3Rdy3, z1.d3Sdy3, z2.d3Sdy3)
        d4wdx4 = combS(z1.d4Rdx4, z2.d4Rdx4, z1.d4Sdx4, z2.d4Sdx4)
        d4wdx3dy = combS(z1.d4Rdx3dy, z2.d4Rdx3dy, z1.d4Sdx3dy, z2.d4Sdx3dy)
        d4wdx2dy2 = combS(z1.d4Rdx2dy2, z2.d4Rdx2dy2, z1.d4Sdx2dy2, z2.d4Sdx2dy2)
        d4wdxdy3 = combS(z1.d4Rdxdy3, z2.d4Rdxdy3, z1.d4Sdxdy3, z2.d4Sdxdy3)
        d4wdy4 = combS(z1.d4Rdy4, z2.d4Rdy4, z1.d4Sdy4, z2.d4Sdy4)
        mn = -(f1 * d2wdx2 + f2 * d2wdxdy + f3 * d2wdy2)
        vn = -(hh1 * d3wdx3 + hh2 * d3wdx2dy + hh3 * d3wdxdy2 + hh4 * d3wdy3)
        dmndx = -(f1 * d3wdx3 + f2 * d3wdx2dy + f3 * d3wdxdy2)
        dmndy = -(f1 * d3wdx2dy + f2 * d3wdxdy2 + f3 * d3wdy3)
        dvndx = -(hh1 * d4wdx4 + hh2 * d4wdx3dy + hh3 * d4wdx2dy2 + hh4 * d4wdxdy3)
        dvndy = -(hh1 * d4wdx3dy + hh2 * d4wdx2dy2 + hh3 * d4wdxdy3 + hh4 * d4wdy4)
        m1, m2 = nx, ny
        d2wdndm = -(d2wdx2 * nx * m1 + d2wdxdy * (nx * m2 + ny * m1) + d2wdy2 * ny * m2)
        dmndm = -(dmndx * m1 + dmndy * m2)
        dvndm = -(dvndx * m1 + dvndy * m2)
        h_el[:, 2j-1:2j] .= [vn -mn; dvndm -dmndm]
        g_el[j] = -d2wdndm
    end
    return h_el, g_el
end

"""
    navier_w_ss_ortho(x, y; a, q, D11, D22, D12, D66, b=a, nterms=80)

Navier series for a simply-supported Kirchhoff plate with ``D_{16}=D_{26}=0``.
"""
function navier_w_ss_ortho(x, y; a, q, D11, D22, D12, D66, b=a, nterms=80)
    H = D12 + 2 * D66
    w = 0.0
    @inbounds for m in 1:2:nterms, n in 1:2:nterms
        α = m * π / a
        β = n * π / b
        den = D11 * α^4 + 2 * H * α^2 * β^2 + D22 * β^4
        qmn = 16q / (π^2 * m * n)
        w += (qmn / den) * sin(α * x) * sin(β * y)
    end
    return w
end
