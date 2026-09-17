# Kirchhoff Dual BEM for cracked plates (Portela–Aliabadi–Rooke / Useche 10.2).
# Outer + face A → CBIE (`plate_kernels`). Face B → traction BIE
# (`plate_hbie_kernels`, source Vn_ξ, Mn_ξ) — the same Dual split as
# Reissner `assemble_fsdt_dual!`. Self/twin: Taylor of the kernel × N
# (four Laurent terms, F₋₄…F₋₁) + Telles remainder + analytic HFP,
# matching MATLAB CalcGtes / `_hbie_sing!`.

# =============================================================================
# Voigt D and moment / Kirchhoff-shear operators
# =============================================================================

function _voigt_D(p::ThinPlateProps)
    D = bending_stiffness(p)
    ν = p.ν
    return D, D, ν * D, 0.0, 0.0, (1 - ν) * D / 2
end
_voigt_D(p::AnisoThinPlateProps) = (p.D11, p.D22, p.D12, p.D16, p.D26, p.D66)

function _f123(nx, ny, D11, D22, D12, D16, D26, D66)
    f1 = D11 * nx^2 + 2 * D16 * nx * ny + D12 * ny^2
    f2 = 2 * (D16 * nx^2 + 2 * D66 * nx * ny + D26 * ny^2)
    f3 = D12 * nx^2 + 2 * D26 * nx * ny + D22 * ny^2
    return f1, f2, f3
end

function _h1234(nx, ny, D11, D22, D12, D16, D26, D66)
    h1 = D11 * nx * (1 + ny^2) + 2 * D16 * ny^3 - D12 * nx * ny^2
    h2 = 4 * D16 * nx + D12 * ny * (1 + nx^2) + 4 * D66 * ny^3 -
         D11 * nx^2 * ny - 2 * D26 * nx * ny^2
    h3 = 4 * D26 * ny + D12 * nx * (1 + ny^2) + 4 * D66 * nx^3 -
         D22 * nx * ny^2 - 2 * D16 * nx^2 * ny
    h4 = D22 * ny * (1 + nx^2) + 2 * D26 * nx^3 - D12 * nx^2 * ny
    return h1, h2, h3, h4
end

# =============================================================================
# Physical w* derivatives through 6th (HBIE)
# =============================================================================

function _iso_d3(xy, c)
    x, y = xy[1], xy[2]
    r2 = x * x + y * y
    r4 = r2 * r2
    t = 2 * c
    return SVector(t * (3x / r2 - 2 * x^3 / r4),
        t * (y / r2 - 2 * x * x * y / r4),
        t * (x / r2 - 2 * x * y * y / r4),
        t * (3y / r2 - 2 * y^3 / r4))
end

function _aniso_d4(xy, p::AnisoThinPlateProps)
    r = hypot(xy[1], xy[2])
    θ = atan(xy[2], xy[1])
    F = _aniso_fundamentals(r, θ, p)
    s = inv(p.D22)
    return SVector(F.d4wdx4, F.d4wdx3dy, F.d4wdx2dy2, F.d4wdxdy3, F.d4wdy4) .* s
end

"""Unique physical derivatives of the force FS through order 6."""
function _wderivs6(rel, p::ThinPlateProps)
    D = bending_stiffness(p)
    c = 1 / (8π * D)
    x, y = rel[1], rel[2]
    r2 = x * x + y * y
    r = sqrt(r2)
    L = log(r)
    xy = [x, y]
    d3 = _iso_d3(xy, c)
    v4 = u -> begin
        J = ForwardDiff.jacobian(v -> _iso_d3(v, c), u)
        SVector(J[1, 1], J[1, 2], J[2, 2], J[3, 2], J[4, 2])
    end
    d4 = v4(xy)
    v5 = u -> begin
        J = ForwardDiff.jacobian(v4, u)
        SVector(J[1, 1], J[1, 2], J[2, 2], J[3, 2], J[4, 2], J[5, 2])
    end
    d5 = v5(xy)
    J6 = ForwardDiff.jacobian(v5, xy)
    d6 = SVector(J6[1, 1], J6[1, 2], J6[2, 2], J6[3, 2], J6[4, 2], J6[5, 2], J6[6, 2])
    return (; w=c * r2 * (L - 0.5), wx=2c * x * L, wy=2c * y * L,
        wxx=2c * (L + x * x / r2), wxy=2c * x * y / r2, wyy=2c * (L + y * y / r2),
        wxxx=d3[1], wxxy=d3[2], wxyy=d3[3], wyyy=d3[4],
        wxxxx=d4[1], wxxxy=d4[2], wxxyy=d4[3], wxyyy=d4[4], wyyyy=d4[5],
        wxxxxx=d5[1], wxxxxy=d5[2], wxxxyy=d5[3], wxxyyy=d5[4],
        wxyyyy=d5[5], wyyyyy=d5[6],
        wxxxxxx=d6[1], wxxxxxy=d6[2], wxxxxyy=d6[3], wxxxyyy=d6[4],
        wxxyyyy=d6[5], wxyyyyy=d6[6], wyyyyyy=d6[7])
end

function _wderivs6(rel, p::AnisoThinPlateProps)
    r = hypot(rel[1], rel[2])
    θ = atan(rel[2], rel[1])
    F = _aniso_fundamentals(r, θ, p)
    s = inv(p.D22)
    xy = [rel[1], rel[2]]
    d4 = SVector(F.d4wdx4, F.d4wdx3dy, F.d4wdx2dy2, F.d4wdxdy3, F.d4wdy4) .* s
    v4 = u -> _aniso_d4(u, p)
    v5 = u -> begin
        J = ForwardDiff.jacobian(v4, u)
        SVector(J[1, 1], J[1, 2], J[2, 2], J[3, 2], J[4, 2], J[5, 2])
    end
    d5 = v5(xy)
    J6 = ForwardDiff.jacobian(v5, xy)
    d6 = SVector(J6[1, 1], J6[1, 2], J6[2, 2], J6[3, 2], J6[4, 2], J6[5, 2], J6[6, 2])
    return (; w=F.w, wx=F.dwdx * s, wy=F.dwdy * s,
        wxx=F.d2wdx2 * s, wxy=F.d2wdxdy * s, wyy=F.d2wdy2 * s,
        wxxx=F.d3wdx3 * s, wxxy=F.d3wdx2dy * s, wxyy=F.d3wdxdy2 * s, wyyy=F.d3wdy3 * s,
        wxxxx=d4[1], wxxxy=d4[2], wxxyy=d4[3], wxyyy=d4[4], wyyyy=d4[5],
        wxxxxx=d5[1], wxxxxy=d5[2], wxxxyy=d5[3], wxxyyy=d5[4],
        wxyyyy=d5[5], wyyyyy=d5[6],
        wxxxxxx=d6[1], wxxxxxy=d6[2], wxxxxyy=d6[3], wxxxyyy=d6[4],
        wxxyyyy=d6[5], wxyyyyy=d6[6], wyyyyyy=d6[7])
end

@inline _Mn(wxx, wxy, wyy, f1, f2, f3) = -(f1 * wxx + f2 * wxy + f3 * wyy)
@inline _Vn(wxxx, wxxy, wxyy, wyyy, h1, h2, h3, h4) =
    -(h1 * wxxx + h2 * wxxy + h3 * wxyy + h4 * wyyy)

# =============================================================================
# HBIE kernels: source (Vn_ξ, Mn_ξ) on CBIE row-1 (w, −∂w/∂n_field)
# =============================================================================

"""
    plate_hbie_kernels(pg, pf, n, nξ, props) -> (U, P)

Traction BIE kernels. `U` multiplies field `(Vn, Mn)`, `P` multiplies
`(w, ∂w/∂n)`. Rows are source `(Vn_ξ, Mn_ξ)`; columns match CBIE
(force, `−∂/∂n` of force). Source 2nd derivs equal field 2nd; source
3rd pick up a minus (`∂_ξ = −∂_x`).
"""
function plate_hbie_kernels(pg::Point2D, pf::Point2D, n::Point2D, nξ::Point2D,
        props::AbstractThinPlateProps)
    rel = pg - pf
    r = hypot(rel[1], rel[2])
    r < 1e-30 && error("plate_hbie_kernels: coincident points")
    d = _wderivs6(rel, props)
    Dij = _voigt_D(props)
    fn1, fn2, fn3 = _f123(n[1], n[2], Dij...)
    hn1, hn2, hn3, hn4 = _h1234(n[1], n[2], Dij...)
    fξ1, fξ2, fξ3 = _f123(nξ[1], nξ[2], Dij...)
    hξ1, hξ2, hξ3, hξ4 = _h1234(nξ[1], nξ[2], Dij...)
    nx, ny = n[1], n[2]

    # Column 1: force FS Φ = w. Mn_ξ uses 2nd_rel; Vn_ξ = −Vn_formula(3rd_rel).
    Mnξ_F = _Mn(d.wxx, d.wxy, d.wyy, fξ1, fξ2, fξ3)
    Vnξ_F = -_Vn(d.wxxx, d.wxxy, d.wxyy, d.wyyy, hξ1, hξ2, hξ3, hξ4)

    # Column 2: Φc = −∂w/∂n (field). Value and derivs of Φc from d.
    # Φc = −(nx wx + ny wy)  →  2nd of Φc = −(nx 3rd + ny 3rd_mixed)
    c_xx = -(nx * d.wxxx + ny * d.wxxy)
    c_xy = -(nx * d.wxxy + ny * d.wxyy)
    c_yy = -(nx * d.wxyy + ny * d.wyyy)
    c_xxx = -(nx * d.wxxxx + ny * d.wxxxy)
    c_xxy = -(nx * d.wxxxy + ny * d.wxxyy)
    c_xyy = -(nx * d.wxxyy + ny * d.wxyyy)
    c_yyy = -(nx * d.wxyyy + ny * d.wyyyy)
    Mnξ_C = _Mn(c_xx, c_xy, c_yy, fξ1, fξ2, fξ3)
    Vnξ_C = -_Vn(c_xxx, c_xxy, c_xyy, c_yyy, hξ1, hξ2, hξ3, hξ4)

    # Field tractions of force (CBIE P row 1): P11=Vn_n(w), P12=−Mn_n(w)
    # 2nd_rel of Vn_n(w) = −(hn · 5th); 3rd_rel = −(hn · 6th)
    Vn_xx = -(hn1 * d.wxxxxx + hn2 * d.wxxxxy + hn3 * d.wxxxyy + hn4 * d.wxxyyy)
    Vn_xy = -(hn1 * d.wxxxxy + hn2 * d.wxxxyy + hn3 * d.wxxyyy + hn4 * d.wxyyyy)
    Vn_yy = -(hn1 * d.wxxxyy + hn2 * d.wxxyyy + hn3 * d.wxyyyy + hn4 * d.wyyyyy)
    Vn_xxx = -(hn1 * d.wxxxxxx + hn2 * d.wxxxxxy + hn3 * d.wxxxxyy + hn4 * d.wxxxyyy)
    Vn_xxy = -(hn1 * d.wxxxxxy + hn2 * d.wxxxxyy + hn3 * d.wxxxyyy + hn4 * d.wxxyyyy)
    Vn_xyy = -(hn1 * d.wxxxxyy + hn2 * d.wxxxyyy + hn3 * d.wxxyyyy + hn4 * d.wxyyyyy)
    Vn_yyy = -(hn1 * d.wxxxyyy + hn2 * d.wxxyyyy + hn3 * d.wxyyyyy + hn4 * d.wyyyyyy)
    # −Mn_n(w) = fn · 2nd; 2nd of that = fn · 4th; 3rd = fn · 5th
    m_xx = fn1 * d.wxxxx + fn2 * d.wxxxy + fn3 * d.wxxyy
    m_xy = fn1 * d.wxxxy + fn2 * d.wxxyy + fn3 * d.wxyyy
    m_yy = fn1 * d.wxxyy + fn2 * d.wxyyy + fn3 * d.wyyyy
    m_xxx = fn1 * d.wxxxxx + fn2 * d.wxxxxy + fn3 * d.wxxxyy
    m_xxy = fn1 * d.wxxxxy + fn2 * d.wxxxyy + fn3 * d.wxxyyy
    m_xyy = fn1 * d.wxxxyy + fn2 * d.wxxyyy + fn3 * d.wxyyyy
    m_yyy = fn1 * d.wxxyyy + fn2 * d.wxyyyy + fn3 * d.wyyyyy

    Mnξ_Vn = _Mn(Vn_xx, Vn_xy, Vn_yy, fξ1, fξ2, fξ3)
    Vnξ_Vn = -_Vn(Vn_xxx, Vn_xxy, Vn_xyy, Vn_yyy, hξ1, hξ2, hξ3, hξ4)
    Mnξ_m = _Mn(m_xx, m_xy, m_yy, fξ1, fξ2, fξ3)
    Vnξ_m = -_Vn(m_xxx, m_xxy, m_xyy, m_yyy, hξ1, hξ2, hξ3, hξ4)

    U = @SMatrix [Vnξ_F Vnξ_C; Mnξ_F Mnξ_C]
    P = @SMatrix [Vnξ_Vn Vnξ_m; Mnξ_Vn Mnξ_m]
    return U, P
end

function _telles_plate(γ, eet)
    eest = eet^2 - 1
    t1 = eet * eest + abs(eest)
    t1 = copysign(abs(t1)^(1 / 3), t1)
    t2 = eet * eest - abs(eest)
    t2 = copysign(abs(t2)^(1 / 3), t2)
    Γ = t1 + t2 + eet
    Q = 1 + 3 * Γ^2
    A = 1 / Q
    B = -3 * Γ / Q
    C = 3 * Γ^2 / Q
    D = -B
    x = ((A * γ + B) * γ + C) * γ + D
    Jt = (3 * A * γ + 2 * B) * γ + C
    return x, Jt
end

@inline function _on_el_plate(el, i)
    @inbounds for k in eachindex(el.index)
        el.index[k] == i && return true
    end
    return false
end
@inline function _on_twin_el_plate(el, i, twin)
    t = twin[i]
    t == 0 && return false
    return _on_el_plate(el, t)
end

function _ξ_on_el_plate(el, i)
    nN = length(el.index)
    qsi, _ = gausslegendre(nN)
    @inbounds for k in 1:nN
        el.index[k] == i && return qsi[k]
    end
    return 0.0
end

# =============================================================================
# HBIE self/twin: Useche / Dirgantara Taylor + Telles (FSDT `_hbie_sing!`)
# P ~ 1/ρ⁴…1/ρ; U ~ 1/ρ²…1/ρ. N up to quadratic (N''). Analytic HFP of
# (ξ−ξ0)^{-k} on [-1,1], remainder Telles-subdivided.
# =============================================================================

function _kronI2(Nrow)
    nN = length(Nrow)
    P = zeros(2, 2nN)
    @inbounds for j in 1:nN
        P[1, 2j - 1] = Nrow[j]
        P[2, 2j] = Nrow[j]
    end
    return P
end

function _kronI2_wt(Nw, Nθ)
    nN = length(Nw)
    P = zeros(2, 2nN)
    @inbounds for j in 1:nN
        P[1, 2j - 1] = Nw[j]
        P[2, 2j] = Nθ[j]
    end
    return P
end

"""CAD-end tip of a crack element, or `nothing`."""
function _el_tip(el, tips)
    isempty(tips) && return nothing
    p0, p1 = _plate_el_ends(el)
    for t in tips
        (norm(p0 - t) < 1e-9 || norm(p1 - t) < 1e-9) && return t
    end
    return nothing
end

"""Vandermonde interpolant `N = V^{-T} b(s)` at nodes `s_nodes`."""
function _vandermonde_N(s_nodes, s, b)
    n = length(s_nodes)
    V = zeros(n, n)
    @inbounds for j in 1:n
        V[j, :] .= b(s_nodes[j])
    end
    return V' \ collect(b(s))
end

"""`w` in `{1, ρ, ρ^{3/2}}`, `θ` in `{1, √ρ, ρ}`. Gauss collocation stays interior.

Quadratic Lagrange in `ξ` cannot represent Hui–Zehnder `θ~√ρ`; the Dual
`1/ρ^4` residual then spikes `Δθ` at the first node. Interpolating in
`s=√ρ` is exact for that leading term without moving collocation onto
the tip (Barsoum+Gauss does).
"""
function _plate_Nwt(el, poly, ξ, tip)
    Nf, _ = shapefun(poly, ξ)
    nN = length(el.index)
    Nw = collect(view(Nf, 1, :))
    Nθ = copy(Nw)
    (tip === nothing || nN != 3) && return Nw, Nθ
    qsi, _ = gausslegendre(nN)
    s_nodes = zeros(nN)
    @inbounds for k in 1:nN
        pg, _, _ = elem_geom(el, qsi[k])
        s_nodes[k] = sqrt(max(hypot(pg[1] - tip[1], pg[2] - tip[2]), 1e-30))
    end
    pg, _, _ = elem_geom(el, ξ)
    s = sqrt(max(hypot(pg[1] - tip[1], pg[2] - tip[2]), 1e-30))
    Nθ = _vandermonde_N(s_nodes, s, σ -> SVector(1.0, σ, σ^2))
    Nw = _vandermonde_N(s_nodes, s, σ -> SVector(1.0, σ^2, σ^3))
    return Nw, Nθ
end

function _plate_phi(el, poly, ξ, tip)
    Nw, Nθ = _plate_Nwt(el, poly, ξ, tip)
    return _kronI2_wt(Nw, Nθ)
end

function _plate_phi_taylor(el, poly, ξ0, tip; h=1e-4)
    φ0 = _plate_phi(el, poly, ξ0, tip)
    φp = _plate_phi(el, poly, ξ0 + h, tip)
    φm = _plate_phi(el, poly, ξ0 - h, tip)
    φ1 = (φp - φm) / (2h)
    φ2 = (φp - 2 * φ0 + φm) / h^2
    φpp = _plate_phi(el, poly, ξ0 + 2h, tip)
    φmm = _plate_phi(el, poly, ξ0 - 2h, tip)
    φ3 = (φpp - 2 * φp + 2 * φm - φmm) / (2h^3)
    return φ0, φ1, φ2, φ3
end

@inline function _accum_PN!(He, Ge, P, U, Nw, Nθ, wJ)
    nN = length(Nw)
    @inbounds for a in 1:nN
        wN, tN = Nw[a] * wJ, Nθ[a] * wJ
        He[:, 2a - 1] .+= P[:, 1] .* wN
        He[:, 2a] .+= P[:, 2] .* tN
        Ge[:, 2a - 1] .+= U[:, 1] .* wN
        Ge[:, 2a] .+= U[:, 2] .* tN
    end
    return nothing
end

"""HFP of `(ξ-ξ0)^{-k}` and `log|ξ-ξ0|` on `[-1,1]`."""
function _intT_hfp(ξ0)
    a = 1 + ξ0
    b = 1 - ξ0
    IntT1 = log(abs(b / a))
    IntT2 = -1 / a - 1 / b
    IntT3 = 0.5 / a^2 - 0.5 / b^2
    IntT4 = -1 / (3 * a^3) - 1 / (3 * b^3)
    IntTlog = log(abs(a * b)) - ξ0 * log(abs(b / a)) - 2
    return IntT1, IntT2, IntT3, IntT4, IntTlog
end

function _d2Nrow(poly, ξ0; h=1e-4)
    _, dNp = shapefun(poly, ξ0 + h)
    _, dNm = shapefun(poly, ξ0 - h)
    return (dNp[1, :] .- dNm[1, :]) ./ (2h)
end

"""Laurent of `P J` / `U J` at `ξ0` for signed `dξ = ξ-ξ0`."""
function _plate_hbie_ctes(el, pf, nξ, ξ0, props)
    s = ξ0 <= 0.5 ? 1.0 : -1.0
    function PJU(ρ)
        ξ = ξ0 + s * ρ
        pg, Jg, n̂ = elem_geom(el, ξ)
        hypot(pg[1] - pf[1], pg[2] - pf[2]) < 1e-30 &&
            return zero(SMatrix{2,2,Float64,4}), zero(SMatrix{2,2,Float64,4})
        U, P = plate_hbie_kernels(pg, pf, n̂, nξ, props)
        return P .* Jg, U .* Jg
    end
    h = 0.02
    CP = laurent_coefficients(ρ -> PJU(ρ)[1], h, Val{-4}())
    CU = laurent_coefficients(ρ -> PJU(ρ)[2], h, Val{-2}())
    # ray ρ → signed dξ: even k unchanged; odd k pick up `s`.
    Cm4, Cm3, Cm2, Cm1, Cm0 = CP
    Cm3 *= s
    Cm1 *= s
    Um2, Um1, Um0 = CU
    Um1 *= s
    return Cm4, Cm3, Cm2, Cm1, Cm0, Um2, Um1, Um0
end

"""Useche/Dirgantara HBIE self/twin (Taylor of kernel × N + Telles remainder)."""
function _hbie_sing_plate!(He, Ge, el, poly, pf, nξ, ξ0, props; nsub::Int=10,
        tip=nothing)
    nN = length(el.index)
    ξ0 = clamp(float(ξ0), nextfloat(-1.0), prevfloat(1.0))
    ϕ0, ϕ1, ϕ2, ϕ3 = _plate_phi_taylor(el, poly, ξ0, tip)
    Cm4, Cm3, Cm2, Cm1, _, Um2, Um1, _ = _plate_hbie_ctes(el, pf, nξ, ξ0, props)
    A4 = Cm4 * ϕ0
    A3 = Cm4 * ϕ1 + Cm3 * ϕ0
    A2 = Cm4 * (ϕ2 ./ 2) + Cm3 * ϕ1 + Cm2 * ϕ0
    A1 = Cm4 * (ϕ3 ./ 6) + Cm3 * (ϕ2 ./ 2) + Cm2 * ϕ1 + Cm1 * ϕ0
    B2 = Um2 * ϕ0
    B1 = Um2 * ϕ1 + Um1 * ϕ0
    qs, ws = gausslegendre(12)
    IH = zeros(2, 2nN)
    IG = zeros(2, 2nN)
    dξs = 2 / nsub
    for k in 1:nsub
        ξa = -1 + (k - 1) * dξs
        ξb = -1 + k * dξs
        eet = clamp((ξ0 - 0.5 * (ξa + ξb)) / (0.5 * (ξb - ξa)), -0.999, 0.999)
        Js = 0.5 * (ξb - ξa)
        for ig in eachindex(qs)
            ξt, Jt = _telles_plate(qs[ig], eet)
            ξ = 0.5 * (ξa + ξb) + 0.5 * (ξb - ξa) * ξt
            dξe = ξ - ξ0
            abs(dξe) < 1e-14 && continue
            pg, Jg, nfield = elem_geom(el, ξ)
            hypot(pg[1] - pf[1], pg[2] - pf[2]) < 1e-30 && continue
            phie = _plate_phi(el, poly, ξ, tip)
            U, P = plate_hbie_kernels(pg, pf, nfield, nξ, props)
            PϕJ = P * phie .* Jg
            UϕJ = U * phie .* Jg
            Psing = A4 ./ dξe^4 .+ A3 ./ dξe^3 .+ A2 ./ dξe^2 .+ A1 ./ dξe
            Using = B2 ./ dξe^2 .+ B1 ./ dξe
            wq = ws[ig] * Jt * Js
            IH .+= (PϕJ .- Psing) .* wq
            IG .+= (UϕJ .- Using) .* wq
        end
    end
    IntT1, IntT2, IntT3, IntT4, _ = _intT_hfp(ξ0)
    He .+= IH .+ A4 .* IntT4 .+ A3 .* IntT3 .+ A2 .* IntT2 .+ A1 .* IntT1
    Ge .+= IG .+ B2 .* IntT2 .+ B1 .* IntT1
    return nothing
end

"""Collinear off-element: pole at `a ∉ [-1,1]`. Ordinary Laurent tails, not HFP."""
function _hbie_collinear!(He, Ge, el, poly, pf, nξ, props; npg::Int=20, tip=nothing)
    nN = length(el.index)
    x1, x3 = el.geo[1], el.geo[end]
    Le = norm(x3 - x1)
    Le < 1e-30 && return false
    t̂ = (x3 - x1) / Le
    pg0, _, n̂ = elem_geom(el, 0.0)
    abs(dot(pf - pg0, n̂)) > 1e-8 * max(Le, 1.0) && return false
    a = 2 * dot(pf - x1, t̂) / Le - 1
    abs(a) <= 0.999 && return false
    qs, ws = gausslegendre(npg)
    # Telles about the near end (clamped parent ξ).
    eet = clamp(a, -0.999, 0.999)
    nsub = 8
    dξs = 2 / nsub
    for k in 1:nsub
        ξa = -1 + (k - 1) * dξs
        ξb = -1 + k * dξs
        eetk = clamp((eet - 0.5 * (ξa + ξb)) / (0.5 * (ξb - ξa)), -0.999, 0.999)
        Js = 0.5 * (ξb - ξa)
        for ig in eachindex(qs)
            ξt, Jt = _telles_plate(qs[ig], eetk)
            ξ = 0.5 * (ξa + ξb) + 0.5 * (ξb - ξa) * ξt
            pg, Jg, nfield = elem_geom(el, ξ)
            R = hypot(pg[1] - pf[1], pg[2] - pf[2])
            R < 1e-14 && continue
            Nw, Nθ = _plate_Nwt(el, poly, ξ, tip)
            U, P = plate_hbie_kernels(pg, pf, nfield, nξ, props)
            wJ = Jg * ws[ig] * Jt * Js
            _accum_PN!(He, Ge, P, U, Nw, Nθ, wJ)
        end
    end
    return true
end

# =============================================================================
# Mesh: lift dual BEMdata (no Kirchhoff corners)
# =============================================================================

"""Apply far-field `M_y = Mo` on `y=±H` (n·M·n = My ny²)."""
function _plate_apply_Mo!(dad::BEMdata{<:AbstractThinPlate}, Mo, Hy)
    eq = dad.eq_type
    @inbounds for i in 1:dad.n
        eq[i] == 1 || continue
        p = dad.Nodes[i]
        ny = dad.Normal[i][2]
        if abs(abs(p[2]) - Hy) < 1e-6 * max(Hy, 1.0)
            dad.BV[2i] = Mo * ny * ny
            dad.BC[2i] = 1
        end
    end
    return dad
end

"""Lift a 2-D dual `BEMdata` to Kirchhoff `PlateMesh` (empty corners). Legacy."""
function plate_mesh_from_dual_dad(dad, props::AbstractThinPlateProps; Mo=0.0, H=nothing)
    n = dad.n
    BC = copy(dad.BC)
    BV = copy(dad.BV)
    eq = copy(dad.eq_type)
    twin = copy(dad.twin)
    Hy = H === nothing ? maximum(abs(p[2]) for p in dad.Nodes) : float(H)
    mesh = PlateMesh(dad.elements, dad.element_type, collect(Float64, dad.elem_weight),
        collect(Point2D, dad.Nodes), collect(Point2D, dad.Normal),
        BC, BV, PlateCorner[], props; eq_type=eq, twin=twin)
    @inbounds for i in 1:n
        eq[i] == 1 || continue
        p = dad.Nodes[i]
        ny = dad.Normal[i][2]
        if abs(abs(p[2]) - Hy) < 1e-6 * max(Hy, 1.0)
            mesh.BV[2i] = Mo * ny * ny
            mesh.BC[2i] = 1
        end
    end
    return mesh
end

"""
    build_rect_plate_crack(; W, H, a, α=0, props, Mo=0, ndiv_b=8, ndiv_h=8, ndiv_crack=16)

Rectangle `[-W,W]×[-H,H]` with a centre crack of half-length `a` in the
Gmsh file ([`quadrado_plate`](@ref) twins `"5;2;5;2"` / `"5;3;5;3"`).
Face A is CBIE, face B HBIE. `Mo` is the far-field `M_y` on `y=±H`.
"""
function build_rect_plate_crack(; W=1.0, H=2.0, a=0.2, α=0.0,
        props=ThinPlateProps(), Mo=0.0, ndiv_b=8, ndiv_h=8, ndiv_crack=16,
        ordem=2, nome="plate_center_crack")
    B = parentmodule(parentmodule(@__MODULE__))
    Cr = B.Crack
    msh = B.quadrado_plate(; Lx=2W, Ly=2H, x0=-W, y0=-H,
        ndiv=(ndiv_b, ndiv_h, ndiv_b, ndiv_h), ordem=ordem, bc="FFFF",
        crack=a, crack_α=α, ndiv_crack=ndiv_crack, nome=nome)
    dad = formatdata(msh, props; tipo=ordem, pontointerno=false)
    Cr.prepare_crack!(dad)
    _plate_apply_Mo!(dad, Mo, H)
    pin_plate_rbm!(dad; W=W, H=H)
    return dad
end

"""Barsoum quarter-point on crack-tip elements so `θ~√ρ` in physical `r`.

Moves the first interior CAD node to `L/4` from the geometric tip and
updates collocation coordinates. Dual `Δθ` can then follow Hui–Zehnder
instead of spiking at the first Gauss node.
"""
function quarter_point_plate_tips!(mesh::PlateMesh)
    tips = _plate_geometric_tips(mesh)
    isempty(tips) && return mesh
    eq = mesh.eq_type
    poly = mesh.element_type
    for el in mesh.elements
        eq[el.index[1]] in (2, 3) || continue
        geo = el.geo
        length(geo) < 3 && continue
        p0, p1 = geo[1], geo[end]
        t0 = findfirst(t -> norm(p0 - t) < 1e-9, tips)
        t1 = findfirst(t -> norm(p1 - t) < 1e-9, tips)
        if t0 !== nothing && t1 === nothing
            geo[2] = p0 + 0.25 * (p1 - p0)
        elseif t1 !== nothing && t0 === nothing
            geo[end - 1] = p1 + 0.25 * (p0 - p1)
        else
            continue
        end
        nN = length(el.index)
        qsi, _ = gausslegendre(nN)
        @inbounds for k in 1:nN
            pg, _, n̂ = elem_geom(el, qsi[k])
            mesh.nodes[el.index[k]] = pg
            nn = hypot(n̂[1], n̂[2])
            nn > 0 && (mesh.Normal[el.index[k]] = n̂ / nn)
        end
    end
    return mesh
end

"""Three kinematic pins (w, ∂w/∂n) on the outer boundary."""
function pin_plate_rbm!(dad::BEMdata{<:AbstractThinPlate}; W=1.0, H=1.0)
    eq = dad.eq_type
    function nearest(pred)
        best, bd = 0, Inf
        @inbounds for i in 1:dad.n
            eq[i] == 1 || continue
            p = dad.Nodes[i]
            pred(p) || continue
            d = abs(p[1]) + abs(p[2])
            if d < bd
                bd = d
                best = i
            end
        end
        return best
    end
    i_left = nearest(p -> abs(p[1] + W) < 1e-6 * max(W, 1) && abs(p[2]) < 0.6H)
    i_bot = nearest(p -> abs(p[2] + H) < 1e-6 * max(H, 1) && abs(p[1]) < 0.6W)
    i_bot2 = nearest(p -> abs(p[2] + H) < 1e-6 * max(H, 1) && abs(p[1]) < 0.25W)
    function set_kin!(inode, dir, val=0.0)
        inode == 0 && return
        dad.BC[2 * (inode - 1) + dir] = 0
        dad.BV[2 * (inode - 1) + dir] = val
        return nothing
    end
    set_kin!(i_bot, 1, 0.0)
    set_kin!(i_left, 2, 0.0)
    if i_bot2 != 0 && i_bot2 != i_bot
        set_kin!(i_bot2, 1, 0.0)
    elseif i_left != 0
        set_kin!(i_left, 1, 0.0)
    end
    return dad
end

function pin_plate_rbm!(mesh::PlateMesh; W=1.0, H=1.0)
    eq = mesh.eq_type
    function nearest(pred)
        best, bd = 0, Inf
        @inbounds for i in 1:length(mesh.nodes)
            eq[i] == 1 || continue
            p = mesh.nodes[i]
            pred(p) || continue
            d = abs(p[1]) + abs(p[2])
            if d < bd
                bd = d
                best = i
            end
        end
        return best
    end
    i_left = nearest(p -> abs(p[1] + W) < 1e-6 * max(W, 1) && abs(p[2]) < 0.6H)
    i_bot = nearest(p -> abs(p[2] + H) < 1e-6 * max(H, 1) && abs(p[1]) < 0.6W)
    i_bot2 = nearest(p -> abs(p[2] + H) < 1e-6 * max(H, 1) && abs(p[1]) < 0.25W)
    function set_kin!(inode, dir, val=0.0)
        inode == 0 && return
        mesh.BC[2 * (inode - 1) + dir] = 0
        mesh.BV[2 * (inode - 1) + dir] = val
        return nothing
    end
    set_kin!(i_bot, 1, 0.0)    # w
    set_kin!(i_left, 2, 0.0)   # ∂w/∂n
    if i_bot2 != 0 && i_bot2 != i_bot
        set_kin!(i_bot2, 1, 0.0)
    elseif i_left != 0
        set_kin!(i_left, 1, 0.0)
    end
    return mesh
end

# =============================================================================
# Assembly
# =============================================================================

function _plate_dual_el!(He, Ge, el, pf, nξ, tipo, props, poly, qs, ws;
        on=false, ξ0=0.0, nsub=8, tip=nothing)
    nN = length(el.index)
    x1, x3 = el.geo[1], el.geo[end]
    if on && tipo == 3
        _hbie_sing_plate!(He, Ge, el, poly, pf, nξ, ξ0, props; nsub=max(nsub, 10),
            tip=tip)
        return nothing
    elseif on
        htmp = zeros(2, 2nN)
        gtmp = zeros(2, 2nN)
        qsf, wf = gausslegendre(max(length(qs), 16))
        _plate_singular_guiggiani!(htmp, gtmp, el, poly, pf, nξ, ξ0, props;
            qsi=qsf, w=wf, tip=tip)
        He .= htmp
        Ge .= gtmp
        return nothing
    end
    if tipo == 3 && _hbie_collinear!(He, Ge, el, poly, pf, nξ, props; tip=tip)
        return nothing
    end
    ndiv = nsub
    dξ = 2 / ndiv
    for k in 1:ndiv
        ξa = -1 + (k - 1) * dξ
        ξb = -1 + k * dξ
        eet = 0.0
        xa = (1 - ξa) / 2 * x1[1] + (1 + ξa) / 2 * x3[1]
        xb = (1 - ξb) / 2 * x1[1] + (1 + ξb) / 2 * x3[1]
        den = xa - xb
        abs(den) > 1e-14 && (eet = (xa + xb - 2 * pf[1]) / den)
        if abs(x3[2] - x1[2]) > abs(x3[1] - x1[1])
            ya = (1 - ξa) / 2 * x1[2] + (1 + ξa) / 2 * x3[2]
            yb = (1 - ξb) / 2 * x1[2] + (1 + ξb) / 2 * x3[2]
            den = ya - yb
            abs(den) > 1e-14 && (eet = (ya + yb - 2 * pf[2]) / den)
        end
        eet = clamp(eet, -0.999, 0.999)
        Jsub = 0.5 * (ξb - ξa)
        for ig in eachindex(qs)
            ξt, Jt = _telles_plate(qs[ig], eet)
            ξ = 0.5 * (ξa + ξb) + 0.5 * (ξb - ξa) * ξt
            abs(ξ) > 1 + 1e-12 && continue
            pg, J, n̂ = elem_geom(el, ξ)
            R = hypot(pg[1] - pf[1], pg[2] - pf[2])
            R < 1e-14 && continue
            Nw, Nθ = _plate_Nwt(el, poly, ξ, tip)
            wJ = J * ws[ig] * Jt * Jsub
            U, P = if tipo == 3
                plate_hbie_kernels(pg, pf, n̂, nξ, props)
            else
                plate_kernels(pg, pf, n̂, nξ, props)
            end
            _accum_PN!(He, Ge, P, U, Nw, Nθ, wJ)
        end
    end
    return nothing
end

"""Collocation source (on-boundary; face B uses the traction BIE)."""
function _plate_dual_src(mesh::PlateMesh, i::Int)
    return mesh.nodes[i]
end

"""
    assemble_plate_dual!(mesh; npg=10, nsub=8)

Outer+face A CBIE; face B traction BIE (Vn, Mn) — Useche 10.2 / Portela.
Self/twin: Taylor+Telles HFP (`_hbie_sing_plate!`). ½I free terms on G
(self+twin), same as [`assemble_fsdt_dual!`](@ref).
"""
function assemble_plate_dual!(mesh::PlateMesh; npg::Int=10, nsub::Int=8)
    n = _n(mesh)
    ndof = 2n
    H = zeros(ndof, ndof)
    G = zeros(ndof, ndof)
    q = zeros(ndof)
    eq = mesh.eq_type
    twin = mesh.twin
    props = mesh.props
    poly = mesh.element_type
    qs, ws = gausslegendre(npg)
    @showprogress "Plate dual H,G" for i in 1:n
        nξ = mesh.Normal[i]
        tipo = eq[i]
        pf = _plate_dual_src(mesh, i)
        rows = 2i-1:2i
        for el in mesh.elements
            nN = length(el.index)
            He = zeros(2, 2nN)
            Ge = zeros(2, 2nN)
            on = _on_el_plate(el, i) || _on_twin_el_plate(el, i, twin)
            ξ0 = 0.0
            if on
                src = _on_el_plate(el, i) ? i : twin[i]
                ξ0 = _ξ_on_el_plate(el, src)
            end
            _plate_dual_el!(He, Ge, el, pf, nξ, tipo, props, poly, qs, ws;
                on=on, ξ0=ξ0, nsub=nsub)
            for a in 1:nN
                ja = el.index[a]
                cols = 2ja-1:2ja
                H[rows, cols] .+= He[:, 2a-1:2a]
                G[rows, cols] .+= Ge[:, 2a-1:2a]
            end
        end
        I2 = Matrix{Float64}(I, 2, 2)
        if tipo == 1
            H[rows, rows] .+= 0.5 .* I2
        elseif tipo == 2
            H[rows, rows] .+= 0.5 .* I2
            tw = twin[i]
            if tw != 0
                H[rows, 2tw-1:2tw] .+= 0.5 .* I2
            end
        elseif tipo == 3
            G[rows, rows] .-= 0.5 .* I2
            tw = twin[i]
            if tw != 0
                G[rows, 2tw-1:2tw] .-= 0.5 .* I2
            end
        end
    end
    mesh.H = H
    mesh.G = G
    mesh.q = q
    return mesh
end

function assemble_plate_dual!(dad::BEMdata{<:AbstractThinPlate}; npg::Int=10,
        nsub::Int=8, threaded::Bool=true, near_factor::Real=1.5, kwargs...)
    n = dad.n
    ndof = 2n
    H = zeros(ndof, ndof)
    G = zeros(ndof, ndof)
    eq = dad.eq_type
    twin = dad.twin
    props = dad.properties
    poly = dad.element_type
    qs, ws = gausslegendre(npg)
    elems = dad.elements
    _collocation_loop!(threaded, n) do i
        nξ = dad.Normal[i]
        tipo = eq[i]
        pf = dad.Nodes[i]
        rows = (2i - 1):(2i)
        for el in elems
            nN = length(el.index)
            He = zeros(2, 2nN)
            Ge = zeros(2, 2nN)
            on = _on_el_plate(el, i) || _on_twin_el_plate(el, i, twin)
            ξ0 = 0.0
            if on
                src = _on_el_plate(el, i) ? i : twin[i]
                ξ0 = _ξ_on_el_plate(el, src)
            end
            _plate_dual_el!(He, Ge, el, pf, nξ, tipo, props, poly, qs, ws;
                on=on, ξ0=ξ0, nsub=nsub)
            for a in 1:nN
                ja = el.index[a]
                cols = (2ja - 1):(2ja)
                H[rows, cols] .+= He[:, 2a-1:2a]
                G[rows, cols] .+= Ge[:, 2a-1:2a]
            end
        end
        I2 = SMatrix{2,2,Float64}(0.5, 0.0, 0.0, 0.5)
        if tipo == 1
            H[rows, rows] .+= I2
        elseif tipo == 2
            H[rows, rows] .+= I2
            tw = twin[i]
            tw != 0 && (H[rows, (2tw - 1):(2tw)] .+= I2)
        elseif tipo == 3
            G[rows, rows] .-= I2
            tw = twin[i]
            tw != 0 && (G[rows, (2tw - 1):(2tw)] .-= I2)
        end
    end
    set_cache!(dad; H, G, plate_q=zeros(ndof))
    return dad
end

"""Row-equilibrate a mixed CBIE/HBIE system (HBIE rows are O(1/L³) vs CBIE O(1))."""
function _equilibrate_rows!(A, b, extra=nothing)
    n = size(A, 1)
    @inbounds for r in 1:n
        nrm = maximum(abs, view(A, r, :))
        extra !== nothing && r <= size(extra, 1) &&
            (nrm = max(nrm, maximum(abs, view(extra, r, :))))
        nrm = max(nrm, 1e-30)
        A[r, :] ./= nrm
        b[r] /= nrm
        extra !== nothing && r <= size(extra, 1) && (extra[r, :] ./= nrm)
    end
    return nothing
end

# =============================================================================
# CTOD SIFs (isotropic Hui–Zehnder thin limit on Δθn)
# =============================================================================

_pnodes(m::PlateMesh) = m.nodes
_pnodes(d::BEMdata) = d.Nodes
_peq(m::PlateMesh) = m.eq_type
_peq(d::BEMdata) = d.eq_type
_ptwin(m::PlateMesh) = m.twin
_ptwin(d::BEMdata) = d.twin
_pu(m::PlateMesh) = m.u
_pu(d::BEMdata) = d.u
_pprops(m::PlateMesh) = m.props
_pprops(d::BEMdata) = d.properties
_pnorm(m::PlateMesh) = m.Normal
_pnorm(d::BEMdata) = d.Normal
_pels(m::PlateMesh) = m.elements
_pels(d::BEMdata) = d.elements

function crack_opening_plate(mesh, inode::Int)
    tw = _ptwin(mesh)[inode]
    tw == 0 && error("node $inode has no twin")
    u = _pu(mesh)
    nrm = _pnorm(mesh)
    uA = SVector(u[2inode - 1], u[2inode])
    uB = SVector(u[2tw - 1], u[2tw])
    return uA, uB, nrm[inode], nrm[tw]
end

function _plate_el_ends(el::Element)
    geo = el.geo
    length(geo) >= 2 && return geo[1], geo[end]
    error("_plate_el_ends: element has no CAD `geo`")
end

function _plate_geometric_tips(mesh)
    eq = _peq(mesh)
    ends = Point2D[]
    counts = Int[]
    for el in _pels(mesh)
        eq[el.index[1]] == 2 || continue
        for p in _plate_el_ends(el)
            idx = findfirst(q -> norm(p - q) < 1e-9, ends)
            if idx === nothing
                push!(ends, p)
                push!(counts, 1)
            else
                counts[idx] += 1
            end
        end
    end
    return [ends[i] for i in eachindex(ends) if counts[i] == 1]
end

function _Cθ_plate(props)
    if props isa ThinPlateProps
        return props.E * props.h^3 / (48 * sqrt(2))
    else
        νe = clamp(props.D12 / max(props.D22, 1e-30), -0.9, 0.9)
        Ee = 12 * props.D22 * (1 - νe^2) / max(props.h, 1e-12)^3
        return Ee * props.h^3 / (48 * sqrt(2))
    end
end

function _dtheta_e2(mesh, inode, e2)
    uA, uB, nA, nB = crack_opening_plate(mesh, inode)
    sA = dot(nA, e2)
    sB = dot(nB, e2)
    θA = abs(sA) > 1e-14 ? uA[2] / sA : uA[2]
    θB = abs(sB) > 1e-14 ? uB[2] / sB : uB[2]
    return θA - θB
end

"""
    sif_ctod_plate(mesh; tip=:right, absK1=true, method=:band)
        -> (K1, K2, rA, rB, Le)

Hui–Zehnder: `K1 = Δ(∇w·e₂) E h³ / (48 √(2ρ))` (Dolbow).

`method=:band` (default) takes the median of `K1(ρ)` on face A for
`ρ ∈ [0.3a, 0.85a]`. `:tip` is two-point √r on the tip element
(Dirgantara).
"""
function sif_ctod_plate(mesh; tip::Symbol=:right, absK1::Bool=true,
        method::Symbol=:band)
    props = _pprops(mesh)
    eq = _peq(mesh)
    tips = _plate_geometric_tips(mesh)
    isempty(tips) && error("sif_ctod_plate: no geometric crack tips")
    if tip === :right
        xm = maximum(p[1] for p in tips)
        cands = [p for p in tips if abs(p[1] - xm) < 1e-9]
        tippos = cands[argmax(p[2] for p in cands)]
    else
        xm = minimum(p[1] for p in tips)
        cands = [p for p in tips if abs(p[1] - xm) < 1e-9]
        tippos = cands[argmin(p[2] for p in cands)]
    end
    eltip = nothing
    for el in _pels(mesh)
        eq[el.index[1]] == 2 || continue
        p0, p1 = _plate_el_ends(el)
        if norm(p0 - tippos) < 1e-9 || norm(p1 - tippos) < 1e-9
            eltip = el
            break
        end
    end
    eltip === nothing && error("sif_ctod_plate: no face-A element at the tip")
    p0, p1 = _plate_el_ends(eltip)
    tdir = norm(p1 - tippos) < norm(p0 - tippos) ? (p0 - p1) : (p1 - p0)
    Le = norm(tdir)
    e1 = tdir / Le
    e2 = Point2D(-e1[2], e1[1])
    idxs = collect(eltip.index)
    nodes = _pnodes(mesh)
    rs = [max(norm(nodes[i] - tippos), 1e-14) for i in idxs]
    perm = sortperm(rs)
    length(perm) >= 3 || error("sif_ctod_plate: tip element needs 3 nodes")
    iB, rB = idxs[perm[2]], rs[perm[2]]
    iA, rA = idxs[perm[3]], rs[perm[3]]
    Cθ = _Cθ_plate(props)
    function Kof(inode, r)
        Δθn = _dtheta_e2(mesh, inode, e2)
        return Cθ * Δθn / sqrt(r)
    end
    KA = Kof(iA, rA)
    KB = Kof(iB, rB)
    Ktip = rA / (rA - rB) * (KB - (rB / rA) * KA)
    if method === :band
        ahlf = 0.5 * (isempty(tips) ? Le : maximum(norm(p - q)
            for p in tips, q in tips))
        Ks = Float64[]
        eq = mesh.eq_type
        for i in 1:length(nodes)
            eq[i] == 2 || continue
            ρ = norm(nodes[i] - tippos)
            (0.3 * ahlf <= ρ <= 0.85 * ahlf) || continue
            push!(Ks, Kof(i, ρ))
        end
        K1b = isempty(Ks) ? Ktip : median(Ks)
        K1 = absK1 ? abs(K1b) : K1b
        return K1, 0.0, rA, rB, Le
    end
    K1 = absK1 ? abs(Ktip) : Ktip
    return K1, 0.0, rA, rB, Le
end
