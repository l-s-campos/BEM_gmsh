# 2D isotropic local BEM (compact Kelvin) with RIM volume extras.
# Companion (a + b r²) I + (c + d r²) r⊗r enforces U*=T*=0 on r=r_i.
# Navier extras use the same local CPD (default PHS(3)+poly) as Laplace M1.

export local_kelvin, local_kelvin_abcd

# =============================================================================
# Compact Kelvin companion
# =============================================================================

"""Coefficients of the regular companion so U* and T* vanish on the circle."""
function local_kelvin_abcd(ri::Real, λ::Real, μ::Real, ν::Real)
    ρ2 = float(ri)^2
    denU = 8π * μ * (1 - ν)
    Γ = -1 / (denU * ρ2)
    b = (1 - 2ν) / (8π * μ * (1 - ν) * ρ2) - Γ / 2
    a = -(3 - 4ν) * log(1 / ri) / denU - b * ρ2
    rhs = 1 / (2π * (1 - ν) * ρ2)
    lum = λ + μ
    coef_d = ρ2 * (-3 * lum + 5λ + 7μ)
    d = (rhs - 2 * lum * b - 3 * lum * Γ) / coef_d
    c = Γ - d * ρ2
    return a, b, c, d
end

function local_kelvin_abcd(props::Elasticity, ri::Real)
    return local_kelvin_abcd(ri, props.lambda, props.mu, effective_nu(props))
end

"""Navier extras: L(companion) = (α0 + α2 r²) e + β (r·e) r."""
function _elast_L_coeffs(λ, μ, b, c, d)
    α0 = (2λ + 6μ) * b + (3λ + 5μ) * c
    α2 = (5λ + 7μ) * d
    β = (10λ + 22μ) * d
    return α0, α2, β
end

function _companion_U(r::SVector{2}, a, b, c, d)
    p = dot(r, r)
    return (a + b * p) * SMatrix{2,2}(1.0, 0.0, 0.0, 1.0) + (c + d * p) * (r * r')
end

function _companion_T(r::SVector{2}, n::SVector{2}, a, b, c, d, λ, μ)
    p = dot(r, r)
    T = MMatrix{2,2,Float64}(undef)
    @inbounds for j in 1:2
        e1 = j == 1 ? 1.0 : 0.0
        e2 = j == 2 ? 1.0 : 0.0
        re = r[1] * e1 + r[2] * e2
        nr = dot(n, r)
        ne = n[1] * e1 + n[2] * e2
        divu = (2b + 3c + 5d * p) * re
        cdp = c + d * p
        εn1 = b * (r[1] * ne + nr * e1) + 2d * re * nr * r[1] +
              cdp / 2 * (r[1] * ne + nr * e1) + cdp * re * n[1]
        εn2 = b * (r[2] * ne + nr * e2) + 2d * re * nr * r[2] +
              cdp / 2 * (r[2] * ne + nr * e2) + cdp * re * n[2]
        T[1, j] = λ * divu * n[1] + 2μ * εn1
        T[2, j] = λ * divu * n[2] + 2μ * εn2
    end
    return SMatrix(T)
end

"""
    local_kelvin(props, r, n, r_i) -> (U, T)

Compact 2D Kelvin: Kelvin + quadratic/quartic companion, zero outside `r_i`.
"""
function local_kelvin(props::Elasticity, r::SVector{2}, n::SVector{2}, ri::Real)
    R = norm(r)
    ri = float(ri)
    R > ri && return zero(SMatrix{2,2,Float64}), zero(SMatrix{2,2,Float64})
    a, b, c, d = local_kelvin_abcd(props, ri)
    λ, μ = props.lambda, props.mu
    Uc = _companion_U(r, a, b, c, d)
    Tc = _companion_T(r, n, a, b, c, d, λ, μ)
    R < 1e-14 && return Uc, Tc
    Uk, Tk = fundamental(props, r, n)
    return _to_smat(Uk) + Uc, _to_smat(Tk) + Tc
end

struct LocalKelvinKernel
    ri::Float64
    a::Float64
    b::Float64
    c::Float64
    d::Float64
    λ::Float64
    μ::Float64
    ν::Float64
end
function LocalKelvinKernel(props::Elasticity, ri::Real)
    a, b, c, d = local_kelvin_abcd(props, ri)
    return LocalKelvinKernel(float(ri), a, b, c, d, props.lambda, props.mu, effective_nu(props))
end
function (K::LocalKelvinKernel)(dad, r, n)
    rv = SVector{2,Float64}(r[1], r[2])
    nv = SVector{2,Float64}(n[1], n[2])
    R = norm(rv)
    R > K.ri && return zero(SMatrix{2,2,Float64}), zero(SMatrix{2,2,Float64})
    Uc = _companion_U(rv, K.a, K.b, K.c, K.d)
    Tc = _companion_T(rv, nv, K.a, K.b, K.c, K.d, K.λ, K.μ)
    R < 1e-14 && return Uc, Tc
    Uk, Tk = fundamental(dad.properties, rv, nv)
    return _to_smat(Uk) + Uc, _to_smat(Tk) + Tc
end

# =============================================================================
# Geometry moments on Ω ∩ B (RIM)
# =============================================================================

function _elast_arc_moments!(S2x, S2y, Sxxx, Sxxy, Sxyy, Syyy, ri, a, b)
    # add integrals over CCW arc θ∈[a,b]
    Δ = b - a
    ri2 = ri * ri
    ri4 = ri2 * ri2
    ri5 = ri4 * ri
    # ∫ r² dΩ arc: already in Laplace S2 = Mxx+Myy via (n·r)r_i r_j/4
    # order-3: ∫ r² r_k dΩ = ∫_arc ri^5 ê_k / 5 dθ
    # ∫ ê dθ = (sin b - sin a, -cos b + cos a)
    ds = sin(b) - sin(a)
    dc = -cos(b) + cos(a)
    S2x[] += ri5 / 5 * ds
    S2y[] += ri5 / 5 * dc
    # ∫ r_i r_j r_m dΩ = ri^5/5 ∫ êi êj êm dθ
    # ∫ cos³ = sin-sin³/3, ∫ sin³ = -cos+cos³/3
    # ∫ cos² sin = -cos³/3, ∫ cos sin² = sin³/3
    sb, sa = sin(b), sin(a)
    cb, ca = cos(b), cos(a)
    Sxxx[] += ri5 / 5 * ((sb - sa) - (sb^3 - sa^3) / 3)
    Syyy[] += ri5 / 5 * ((-cb + ca) + (cb^3 - ca^3) / 3)
    Sxxy[] += ri5 / 5 * (-(cb^3 - ca^3) / 3)
    Sxyy[] += ri5 / 5 * ((sb^3 - sa^3) / 3)
    return nothing
end

function _elast_seg_moments!(Mx, My, Mxx, Mxy, Myy, S2x, S2y, Sxxx, Sxxy, Sxyy, Syyy,
        dad, el, y, ξa, ξb, ηs, ws)
    poly = dad.element_type
    nodes = dad.Nodes[el.index]
    nref = dad.Normal[el.index[1]]
    s = (ξb - ξa) / 2
    m = (ξa + ξb) / 2
    s <= 1e-16 && return nothing
    @inbounds for q in eachindex(ηs)
        ξ = s * ηs[q] + m
        N, dN = shapefun(poly, ξ)
        x = (N * nodes)[1]
        Jv = (dN * nodes)[1]
        J = norm(Jv)
        J < 1e-16 && continue
        n = tan2normal(Jv / J)
        n ⋅ nref < 0 && (n = -n)
        r = x - y
        R2 = dot(r, r)
        R2 < 1e-20 && continue
        nr = n ⋅ r
        wJ = ws[q] * s * J
        mom1 = nr * wJ / 3
        mom2 = nr * wJ / 4
        mom3 = nr * wJ / 5
        Mx[] += mom1 * r[1]
        My[] += mom1 * r[2]
        Mxx[] += mom2 * r[1] * r[1]
        Mxy[] += mom2 * r[1] * r[2]
        Myy[] += mom2 * r[2] * r[2]
        S2x[] += mom3 * R2 * r[1]
        S2y[] += mom3 * R2 * r[2]
        Sxxx[] += mom3 * r[1]^3
        Sxxy[] += mom3 * r[1]^2 * r[2]
        Sxyy[] += mom3 * r[1] * r[2]^2
        Syyy[] += mom3 * r[2]^3
    end
    return nothing
end

function _elast_geom_moments(dad, y, ri, ηs, ws, orient)
    S = Ref(0.0)
    Mx = Ref(0.0); My = Ref(0.0)
    Mxx = Ref(0.0); Mxy = Ref(0.0); Myy = Ref(0.0)
    S2x = Ref(0.0); S2y = Ref(0.0)
    Sxxx = Ref(0.0); Sxxy = Ref(0.0); Sxyy = Ref(0.0); Syyy = Ref(0.0)
    segs, hits = _lbem_collect_clips(dad, y, ri)
    for (el, ξa, ξb) in segs
        _elast_seg_moments!(Mx, My, Mxx, Mxy, Myy, S2x, S2y, Sxxx, Sxxy, Sxyy, Syyy,
            dad, el, y, ξa, ξb, ηs, ws)
        # ID1 / S from Laplace primitive on the same segments
        poly = dad.element_type
        nodes = dad.Nodes[el.index]
        nref = dad.Normal[el.index[1]]
        sm = (ξb - ξa) / 2
        mm = (ξa + ξb) / 2
        for q in eachindex(ηs)
            ξ = sm * ηs[q] + mm
            N, dN = shapefun(poly, ξ)
            x = (N * nodes)[1]
            Jv = (dN * nodes)[1]
            J = norm(Jv)
            J < 1e-16 && continue
            n = tan2normal(Jv / J)
            n ⋅ nref < 0 && (n = -n)
            r = x - y
            R = norm(r)
            R < 1e-14 && continue
            S[] += ws[q] * sm * J * (n ⋅ r) / (R * R) * radial_integral_local_one(R, ri; dim=2)
        end
    end
    if isempty(hits)
        if isempty(segs)
            S[] = π * ri * ri
            Mxx[] = Myy[] = π * ri^4 / 4
        end
    else
        n = length(hits)
        for k in 1:n
            a = hits[k]
            b = k == n ? hits[1] + 2π : hits[k + 1]
            mid = 0.5 * (a + b)
            p = Point2D(y[1] + ri * cos(mid), y[2] + ri * sin(mid))
            _lbem_point_in_omega(dad, p, ηs, ws, orient) || continue
            Δ = b - a
            S[] += radial_integral_local_one(ri, ri; dim=2) * Δ
            # second moments on arc: (n·r) r_i r_j / 4 * dΓ = ri^4 êi êj / 4 dθ
            # ∫ cos² = Δ/2 + (sin2b-sin2a)/4, ∫ sin² = Δ/2 - ..., ∫ cis = -(cos2b-cos2a)/4
            s2b, s2a = sin(2b), sin(2a)
            c2b, c2a = cos(2b), cos(2a)
            ri4 = ri^4
            Mxx[] += ri4 / 4 * (Δ / 2 + (s2b - s2a) / 4)
            Myy[] += ri4 / 4 * (Δ / 2 - (s2b - s2a) / 4)
            Mxy[] += ri4 / 4 * (-(c2b - c2a) / 4)
            _elast_arc_moments!(S2x, S2y, Sxxx, Sxxy, Sxyy, Syyy, ri, a, b)
            # first moments on arc
            ri3 = ri^3 / 3
            Mx[] += ri3 * (sin(b) - sin(a))
            My[] += ri3 * (-cos(b) + cos(a))
        end
    end
    S2 = Mxx[] + Myy[]
    return (; S=S[], Mx=Mx[], My=My[], Mxx=Mxx[], Mxy=Mxy[], Myy=Myy[], S2=S2,
        S2x=S2x[], S2y=S2y[], Sxxx=Sxxx[], Sxxy=Sxxy[], Sxyy=Sxyy[], Syyy=Syyy[])
end

# =============================================================================
# Extra operator V(u) = α0 ∫u + α2 ∫ r² u + β ∫ r (r·u)
# Local CPD on B(y_i, r_i): ∫u from RIM of φ_j; ∫ r² u and ∫ r⊗r u from
# cubature of the same interpolant (p scaled by r_i).
# =============================================================================

function _elast_cpd_weighted_rhs(sys, pts, y, ri, rbf, qx, qw, dens)
    n, npoly, pdeg = sys.n, sys.npoly, sys.pdeg
    IF = zeros(n)
    IP = zeros(npoly)
    invr = 1 / max(ri, 1e-14)
    ids = sys.ids
    @inbounds for q in eachindex(qx)
        x = qx[q]
        wv = qw[q]
        rx = x[1] - y[1]
        ry = x[2] - y[2]
        wt = wv * dens(rx, ry)
        if npoly > 0
            p = _lbem_scaled_monomials(rx * invr, ry * invr, pdeg)
            for α in 1:npoly
                IP[α] += wt * p[α]
            end
        end
        for k in 1:n
            IF[k] += wt * rbf(norm(x - pts[ids[k]]))
        end
    end
    return IF, IP
end

function _elast_extra_block!(V, dad, i, ri, α0, α2, β, mom, pts, rbf,
        segs, hits, ηs, ws, orient)
    sys = _lbem_local_cpd_K(pts, i, ri, rbf)
    (sys === nothing || sys.npoly < 1) && return nothing
    y = pts[i]
    n = sys.n
    IF0 = zeros(n)
    @inbounds for k in 1:n
        IF0[k] = _lbem_rim_phi_omega_i(dad, y, ri, pts[sys.ids[k]], rbf,
            segs, hits, ηs, ws, orient)
    end
    IP0 = _lbem_scaled_IP(sys.pdeg, mom.S, mom.Mx, mom.My, mom.Mxx, mom.Mxy, mom.Myy, ri)
    w0 = _lbem_cpd_quad_weights(sys, IF0, IP0)
    w0 === nothing && return nothing

    qx, qw = _lbem_omega_quad_points(dad, y, ri, segs, hits, ηs, ws, orient)
    isempty(qx) && return nothing
    IF2, IP2 = _elast_cpd_weighted_rhs(sys, pts, y, ri, rbf, qx, qw, (rx, ry) -> rx * rx + ry * ry)
    IFxx, IPxx = _elast_cpd_weighted_rhs(sys, pts, y, ri, rbf, qx, qw, (rx, ry) -> rx * rx)
    IFxy, IPxy = _elast_cpd_weighted_rhs(sys, pts, y, ri, rbf, qx, qw, (rx, ry) -> rx * ry)
    IFyy, IPyy = _elast_cpd_weighted_rhs(sys, pts, y, ri, rbf, qx, qw, (rx, ry) -> ry * ry)
    w2 = _lbem_cpd_quad_weights(sys, IF2, IP2)
    wxx = _lbem_cpd_quad_weights(sys, IFxx, IPxx)
    wxy = _lbem_cpd_quad_weights(sys, IFxy, IPxy)
    wyy = _lbem_cpd_quad_weights(sys, IFyy, IPyy)
    (w2 === nothing || wxx === nothing || wxy === nothing || wyy === nothing) && return nothing

    @inbounds for (k, j) in enumerate(sys.ids)
        jx, jy = 2j - 1, 2j
        c = α0 * w0[k] + α2 * w2[k]
        V[1, jx] += c + β * wxx[k]
        V[1, jy] += β * wxy[k]
        V[2, jx] += β * wxy[k]
        V[2, jy] += c + β * wyy[k]
    end
    return nothing
end

# =============================================================================
# H, G assembly
# =============================================================================

function _lbem_integrate_segment_vec!(h, g, dad, el, nodes, pf, K, ξa, ξb, qsi, w, nref, aξ, dist)
    poly = dad.element_type
    dim = 2
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
        nv = tan2normal(dx / J)
        nv ⋅ nref < 0 && (nv = -nv)
        U, T = K(dad, x - pf, nv)
        wJ = ww[q] * s * J
        for j in 1:nn
            cols = expand(j, dim)
            Nj = N[1, j] * wJ
            for d in 1:dim
                h[d, cols] .+= T[d, :] * Nj
                g[d, cols] .+= U[d, :] * Nj
            end
        end
    end
    return nothing
end

function _assemble_local_HG_elast!(dad, radii; npg::Int=16)
    _init_quadrature!(dad, npg)
    nt, n = dad.nt, dad.n
    dim = 2
    H = zeros(dim * nt, dim * nt)
    G = zeros(dim * nt, dim * n)
    poly = dad.element_type
    qsi, wgt = dad.qsi, dad.w
    props = dad.properties
    for i in 1:nt
        pf = point(dad, i)
        ri = radii[i]
        K = LocalKelvinKernel(props, ri)
        ii = expand(i, dim)
        @inbounds for el in dad.elements
            xj = dad.Nodes[el.index]
            _lbem_elem_may_hit_ball(pf, xj, el, ri) || continue
            segs = clip_element_to_ball(poly, xj, pf, ri)
            isempty(segs) && continue
            nn = length(el)
            hloc = zeros(dim, dim * nn)
            gloc = zeros(dim, dim * nn)
            full = length(segs) == 1 && segs[1][1] <= -1 + 1e-12 && segs[1][2] >= 1 - 1e-12
            if full
                integrate_element(dad, el, xj, pf, hloc, gloc, K)
            else
                aξ, _, dist = closest_point_1d(poly, xj, pf; ξ0=_seed_1d(poly, xj, pf))
                nref = dad.Normal[el.index[1]]
                for (ξa, ξb) in segs
                    _lbem_integrate_segment_vec!(hloc, gloc, dad, el, xj, pf, K, ξa, ξb,
                        qsi, wgt, nref, aξ, dist)
                end
            end
            jj = expand(el.index, dim)
            H[ii, jj] .+= hloc
            G[ii, jj] .+= gloc
        end
    end
    return H, G
end

# =============================================================================
# Mixed BC + solve
# =============================================================================

function _lbem_mixed_system_elast(dad, Au, G, rhs)
    dim = 2
    n = dad.n
    ndof_b = dim * n
    A = Matrix{Float64}(Au)
    b = Vector{Float64}(rhs)
    @inbounds for dof in 1:ndof_b
        if dad.BC[dof] == 0
            b .-= view(Au, :, dof) .* dad.BV[dof]
            A[:, dof] .= .-view(G, :, dof)
        else
            b .+= view(G, :, dof) .* dad.BV[dof]
        end
    end
    return A, b
end

function _lbem_scatter_elast!(dad, x)
    dim = 2
    ndof_b = dim * dad.n
    ndof = dim * dad.nt
    u = zeros(ndof)
    t = zeros(ndof_b)
    @inbounds for dof in 1:ndof_b
        if dad.BC[dof] == 0
            u[dof] = dad.BV[dof]
            t[dof] = x[dof]
        else
            u[dof] = x[dof]
            t[dof] = dad.BV[dof]
        end
    end
    @inbounds for dof in (ndof_b + 1):ndof
        u[dof] = x[dof]
    end
    set_cache!(dad; u=u, traction=t, T=u)
    return u
end

# =============================================================================
# Public API
# =============================================================================

function assemble_local_bem!(dad::BEMdata{<:Elasticity};
        radius=nothing, radius_factor::Real=2.0, rbf=nothing, npg::Int=16)
    dad.dimension == 2 || error("assemble_local_bem!(Elasticity) is 2D only")
    dad.ni > 0 || error("local BEM needs internal nodes (pontointerno=true)")
    radii = _lbem_radii(dad; radius=radius, radius_factor=radius_factor)
    basis = rbf === nothing ? _lbem_default_rbf() : rbf
    props = dad.properties
    λ, μ = props.lambda, props.mu
    _init_quadrature!(dad, npg)
    ηs, ws = dad.qsi, dad.w
    p0 = dad.internalNodes[1]
    w0 = _lbem_winding(dad, p0, ηs, ws)
    orient = w0 < 0 ? -1.0 : 1.0

    H, G = _assemble_local_HG_elast!(dad, radii; npg=npg)
    dim = 2
    nt = dad.nt
    Au = Matrix(H)
    pts = all_points(dad)
    @inbounds for i in 1:nt
        ri = radii[i]
        _, b, c, d = local_kelvin_abcd(props, ri)
        α0, α2, β = _elast_L_coeffs(λ, μ, b, c, d)
        mom = _elast_geom_moments(dad, pts[i], ri, ηs, ws, orient)
        segs, hits = _lbem_collect_clips(dad, pts[i], ri)
        V = zeros(2, 2nt)
        _elast_extra_block!(V, dad, i, ri, α0, α2, β, mom, pts, basis,
            segs, hits, ηs, ws, orient)
        rows = expand(i, dim)
        Au[rows, :] .-= V
    end
    @views for i in 1:nt
        ii = expand(i, dim)
        Au[ii, ii] .= 0.0
        for j in 1:dim
            Au[ii, ii[j]] .= -sum(Au[ii, j:dim:end]; dims=2)
        end
    end
    set_cache!(dad; H=H, G=G, lbem_Au=Au, lbem_radius=radii, lbem_method=:local_elast,
        dibem_rbf=basis)
    return dad
end

function solve_local_bem!(dad::BEMdata{<:Elasticity}, bforce=0;
        radius=nothing, radius_factor::Real=2.0, rbf=nothing, npg::Int=16,
        rebuild::Bool=false)
    if rebuild || !has_cache(dad, :lbem_Au)
        assemble_local_bem!(dad; radius=radius, radius_factor=radius_factor,
            rbf=rbf, npg=npg)
    end
    bforce == 0 || error("elastic local BEM body force not yet implemented")
    ndof = 2 * dad.nt
    rhs = zeros(ndof)
    A, b = _lbem_mixed_system_elast(dad, dad.lbem_Au, dad.G, rhs)
    set_cache!(dad; A=A, b=b)
    x = bem_linsolve(A, b)
    return _lbem_scatter_elast!(dad, x)
end
