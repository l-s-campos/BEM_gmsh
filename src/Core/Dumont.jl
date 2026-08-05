# Dumont singular / quasi-singular integration for 2D curve elements
# -----------------------------------------------------------------------------
# GL of regularised kernels + analytic C_G / C₁ corrections (Dumont EABE 2023).
# Geometry uses BEM_gmsh polynomials + nodal coordinates.
#
# Used by 2D Laplace, isotropic Elasticity (Kelvin), AnisotropicElasticity (Lekhnitskii).
# Helmholtz helper kept for future complex assembly; plates → sinh.
# -----------------------------------------------------------------------------

export SingKind, NO_SING, SING, QUASI_SING
export classify_singularity, singularity_pole
export correction_C1, correction_C2, correction_Clog
export integraelem_dumont!
export supports_dumont

@enum SingKind begin
    NO_SING
    SING
    QUASI_SING
end

"""True if this problem uses Dumont for 2D near/on-element pairs."""
supports_dumont(::BEMdata{<:Laplace}) = true
supports_dumont(::BEMdata{<:Elasticity}) = true              # 2D Kelvin
supports_dumont(::BEMdata{<:AnisotropicElasticity}) = true # 2D Lekhnitskii
# Helmholtz FS are complex; dense assembly is still Float64 — use sinh for now.
supports_dumont(::BEMdata{<:Helmholtz}) = false
supports_dumont(::BEMdata) = false

# =============================================================================
# Complex geometry helpers
# =============================================================================

function _shape_complex(poly::AbstractPolynomial, ξ::Complex)
    x0 = poly.nodes
    n = length(x0)
    # barycentric Lagrange
    # weights
    bw = ones(Float64, n)
    @inbounds for j in 1:n, i in 1:n
        i == j && continue
        bw[j] *= 1 / (x0[j] - x0[i])
    end
    # exact node
    for j in 1:n
        if abs(ξ - x0[j]) < 1e-14
            N = zeros(ComplexF64, n)
            N[j] = 1
            return N
        end
    end
    num = zeros(ComplexF64, n)
    s = zero(ComplexF64)
    @inbounds for i in 1:n
        num[i] = bw[i] / (ξ - x0[i])
        s += num[i]
    end
    return num ./ s
end

function _shape_deriv_complex(poly::AbstractPolynomial, ξ::Complex)
    x0 = poly.nodes
    n = length(x0)
    bw = ones(Float64, n)
    @inbounds for j in 1:n, i in 1:n
        i == j && continue
        bw[j] *= 1 / (x0[j] - x0[i])
    end
    for j in 1:n
        if abs(ξ - x0[j]) < 1e-14
            # derivative at node: standard barycentric
            dN = zeros(ComplexF64, n)
            @inbounds for i in 1:n
                i == j && continue
                dN[i] = bw[i] / (bw[j] * (x0[j] - x0[i]))
                dN[j] -= dN[i]
            end
            return dN
        end
    end
    num = zeros(ComplexF64, n)
    dnum = zeros(ComplexF64, n)
    s = zero(ComplexF64)
    ds = zero(ComplexF64)
    @inbounds for i in 1:n
        δ = ξ - x0[i]
        num[i] = bw[i] / δ
        dnum[i] = -bw[i] / δ^2
        s += num[i]
        ds += dnum[i]
    end
    dN = zeros(ComplexF64, n)
    @inbounds for i in 1:n
        dN[i] = (dnum[i] * s - num[i] * ds) / s^2
    end
    return dN
end

@inline function _cpos(poly, nodes, ξ::Complex)
    N = _shape_complex(poly, ξ)
    z = zero(ComplexF64)
    @inbounds for j in eachindex(N)
        z += N[j] * complex(nodes[j][1], nodes[j][2])
    end
    return z
end

@inline function _ctan(poly, nodes, ξ::Complex)
    dN = _shape_deriv_complex(poly, ξ)
    zp = zero(ComplexF64)
    @inbounds for j in eachindex(dN)
        zp += dN[j] * complex(nodes[j][1], nodes[j][2])
    end
    return zp
end

@inline function _pos_real(poly, nodes, ξ::Float64)
    N, _ = shapefun(poly, ξ)
    return (N * nodes)[1]
end

@inline function _jac_real(poly, nodes, ξ::Float64)
    _, dN = shapefun(poly, ξ)
    return norm((dN * nodes)[1])
end

"""Complex singularity pole ξs with z(ξs) ≈ z_source (Newton)."""
function singularity_pole(poly, nodes::AbstractVector{<:Point}, pf::Point;
                          ξ0=0.0, maxiter=40, tol=1e-14)
    zs = complex(pf[1], pf[2])
    a, _, _ = closest_point_1d(poly, nodes, pf; ξ0=ξ0)
    ξs = ComplexF64(a)
    for _ in 1:maxiter
        z = _cpos(poly, nodes, ξs)
        zp = _ctan(poly, nodes, ξs)
        abs(zp) < 1e-30 && break
        Δ = (z - zs) / zp
        ξs -= Δ
        abs(Δ) < tol && break
    end
    return ξs
end

function classify_singularity(ξs::ComplexF64; threshold=0.75, atol_imag=1e-8)
    a, b = real(ξs), imag(ξs)
    dmin = if a < -1
        abs(ξs + 1)
    elseif a > 1
        abs(ξs - 1)
    else
        abs(b)
    end
    dmin > threshold && return NO_SING
    abs(b) ≤ atol_imag && -1 ≤ a ≤ 1 && return SING
    return QUASI_SING
end

# =============================================================================
# Analytic corrections on [-1,1]
# =============================================================================

"""C₁ = ln((1−s)/(−1−s)) − Σ wᵢ/(ξᵢ−s)"""
function correction_C1(ξs::Number, qsi, w)
    s = ComplexF64(ξs)
    C = log((1 - s) / (-1 - s))
    @inbounds for i in eachindex(qsi)
        C -= w[i] / (qsi[i] - s)
    end
    return C
end

"""C₂ = −2/(1−s²) − Σ wᵢ/(ξᵢ−s)²"""
function correction_C2(ξs::Number, qsi, w)
    s = ComplexF64(ξs)
    C = -2 / (1 - s^2)
    @inbounds for i in eachindex(qsi)
        C -= w[i] / (qsi[i] - s)^2
    end
    return C
end

@inline function _cpos_conj(poly, nodes, ξ::Complex)
    N = _shape_complex(poly, ξ)
    z = zero(ComplexF64)
    @inbounds for j in eachindex(N)
        z += N[j] * conj(complex(nodes[j][1], nodes[j][2]))
    end
    return z
end

"""Map complex first-row action u = a·d + b·conj(d) → real 2×2 on (ux,uy)."""
function _complex_firstrow_to_real(a::ComplexF64, b::ComplexF64)
    ar, ai = reim(a)
    br, bi = reim(b)
    return @SMatrix [
        (ar + br)   (-(ai - bi))
        (ai + bi)    (ar - br)
    ]
end

function _power_log_integral(k::Int, s::ComplexF64)
    I = zero(ComplexF64)
    c = 1.0
    for m in 0:k
        if m > 0
            c *= (k - m + 1) / m
        end
        m1 = m + 1
        term1 = (1 - s)^m1 / m1 * log(1 - s) - (1 - s)^m1 / m1^2
        term0 = (-1 - s)^m1 / m1 * log(-1 - s) - (-1 - s)^m1 / m1^2
        I += c * s^(k - m) * (term1 - term0)
    end
    return I
end

"""∫ ln(ξ−s) N_j |J| dξ for all j (monomial projection of |J|)."""
function _log_NJ_integral(poly, nodes, s::ComplexF64; pJ::Int=8)
    x0 = collect(Float64, poly.nodes)
    n = length(x0)
    Vn = [x0[i]^(j - 1) for i in 1:n, j in 1:n]
    CN = zeros(Float64, n, n)
    rhs = zeros(Float64, n)
    for ℓ in 1:n
        fill!(rhs, 0.0); rhs[ℓ] = 1.0
        CN[:, ℓ] = Vn \ rhs
    end
    qJ, wJ = gausslegendre(pJ + 1)
    Vj = [qJ[i]^(k - 1) for i in eachindex(qJ), k in 1:(pJ + 1)]
    Jsamp = Float64[_jac_real(poly, nodes, qJ[i]) for i in eachindex(qJ)]
    W = Diagonal(sqrt.(wJ))
    cJ = W * Vj \ W * Jsamp
    qmax = (n - 1) + pJ
    Ipow = [_power_log_integral(q, s) for q in 0:qmax]
    Ian = zeros(ComplexF64, n)
    @inbounds for j in 1:n
        acc = zero(ComplexF64)
        for a in 0:n-1, b in 0:pJ
            acc += CN[a + 1, j] * cJ[b + 1] * Ipow[a + b + 1]
        end
        Ian[j] = acc
    end
    return Ian
end

# =============================================================================
# Element integration
# =============================================================================

"""
    integraelem_dumont!(h, g, dad, elem, nodes, pf, qsi, w; threshold=0.75)

Fill views `h`, `g` for one element with Dumont GL + corrections.

For `Laplace`: full Dumont on U=log and T=1/r.
For `Helmholtz`: Laplace leading singularity via Dumont + plain GL of remainder.
"""
function integraelem_dumont!(h::AbstractVector, g::AbstractVector,
                             dad::BEMdata{<:Laplace}, elem, nodes, pf::Point2D,
                             qsi, w; threshold::Float64=0.75)
    fill!(h, 0.0)
    fill!(g, 0.0)
    poly = dad.element_type
    k = dad.properties.k
    nN = length(elem)
    pref_G = -1.0 / (2π * k)
    pref_H = 1.0 / (2π)

    ξ0 = _seed_1d(poly, nodes, pf)
    ξs = singularity_pole(poly, nodes, pf; ξ0=ξ0)
    kind = classify_singularity(ComplexF64(ξs); threshold=threshold)
    s = ComplexF64(ξs)
    zs = complex(pf[1], pf[2])

    Nmat, dNmat = element_shapefun(poly, elem, qsi)

    if kind == NO_SING
        @inbounds for i in eachindex(qsi)
            pg, dx, J, nrm = _geom_i(Nmat, dNmat, nodes, i, nN)
            J < 1e-30 && continue
            r = pg - pf
            norm(r) < 1e-30 && continue
            U, T = fundamental(dad, r, nrm)
            wi = J * w[i]
            for j in 1:nN
                h[j] += Nmat[i, j] * T * wi
                g[j] += Nmat[i, j] * U * wi
            end
        end
        return kind
    end

    # regularised GL + log analytic
    Ian = _log_NJ_integral(poly, nodes, s; pJ=max(nN + 2, 6))
    @inbounds for i in eachindex(qsi)
        ξ = qsi[i]
        pg, dx, J, _ = _geom_i(Nmat, dNmat, nodes, i, nN)
        z = complex(pg[1], pg[2]) - zs
        zp = complex(dx[1], dx[2])
        δ = ξ - s
        (abs(z) < 1e-30 || abs(δ) < 1e-30 || J < 1e-30) && continue
        ln_reg = log(z / δ)
        wi = J * w[i]
        zt = zp / z
        for j in 1:nN
            g[j] += pref_G * real(ln_reg) * Nmat[i, j] * wi
            h[j] += pref_H * imag(zt) * Nmat[i, j] * w[i]
        end
    end
    @inbounds for j in 1:nN
        g[j] += pref_G * real(Ian[j])
    end

    # H C1 for quasi-sing / off-element
    a = clamp(real(s), -1.0, 1.0)
    off = norm(_pos_real(poly, nodes, a) - pf) > 1e-14
    if kind == QUASI_SING || (kind == SING && off)
        C1 = correction_C1(s, qsi, w)
        Ns = _shape_complex(poly, s)
        @inbounds for j in 1:nN
            h[j] += pref_H * imag(Ns[j] * C1)
        end
    end
    return kind
end

"""Helmholtz: Dumont on Laplace singular part + GL of regular remainder."""
function integraelem_dumont!(h::AbstractVector, g::AbstractVector,
                             dad::BEMdata{<:Helmholtz}, elem, nodes, pf::Point2D,
                             qsi, w; threshold::Float64=0.75)
    fill!(h, 0.0)
    fill!(g, 0.0)
    # temporary Laplace problem with k=1 for singular part scaling
    # Helmholtz G ~ -log(R)/(2π) + regular, H ~ (r·n)/(2π R²) + regular
    # Use Laplace dumont with k=1 then add GL of (Helmholtz - Laplace_sing)
    poly = dad.element_type
    nN = length(elem)
    hL = zeros(eltype(h), nN)
    gL = zeros(eltype(g), nN)
    # build a thin wrapper: call Laplace path with k=1 via fake — use internal kernels
    _dumont_laplace_unit!(hL, gL, poly, nodes, pf, qsi, w; threshold=threshold)

    Nmat, dNmat = element_shapefun(poly, elem, qsi)
    @inbounds for i in eachindex(qsi)
        pg, dx, J, nrm = _geom_i(Nmat, dNmat, nodes, i, nN)
        J < 1e-30 && continue
        r = pg - pf
        R = norm(r)
        R < 1e-30 && continue
        UH, TH = fundamental(dad, r, nrm)
        UL = -log(R) / (2π)
        TL = dot(r, nrm) / (2π * R^2)
        wi = J * w[i]
        for j in 1:nN
            g[j] += Nmat[i, j] * (UH - UL) * wi
            h[j] += Nmat[i, j] * (TH - TL) * wi
        end
    end
    @inbounds for j in 1:nN
        g[j] += gL[j]
        h[j] += hL[j]
    end
    return QUASI_SING
end

"""Laplace Dumont with fixed k=1 (helper for Helmholtz split)."""
function _dumont_laplace_unit!(h, g, poly, nodes, pf, qsi, w; threshold=0.75)
    fill!(h, 0.0); fill!(g, 0.0)
    nN = length(nodes)
    pref_G = -1.0 / (2π)
    pref_H = 1.0 / (2π)
    ξ0 = _seed_1d(poly, nodes, pf)
    ξs = singularity_pole(poly, nodes, pf; ξ0=ξ0)
    kind = classify_singularity(ComplexF64(ξs); threshold=threshold)
    s = ComplexF64(ξs)
    zs = complex(pf[1], pf[2])
    Nmat, dNmat = shapefun(poly, qsi)
    if kind == NO_SING
        @inbounds for i in eachindex(qsi)
            pg, dx, J, nrm = _geom_i(Nmat, dNmat, nodes, i, nN)
            J < 1e-30 && continue
            r = pg - pf; R = norm(r); R < 1e-30 && continue
            wi = J * w[i]
            U = -log(R) / (2π); T = dot(r, nrm) / (2π * R^2)
            for j in 1:nN
                h[j] += Nmat[i, j] * T * wi
                g[j] += Nmat[i, j] * U * wi
            end
        end
        return
    end
    Ian = _log_NJ_integral(poly, nodes, s; pJ=max(nN + 2, 6))
    @inbounds for i in eachindex(qsi)
        ξ = qsi[i]
        pg, dx, J, _ = _geom_i(Nmat, dNmat, nodes, i, nN)
        z = complex(pg[1], pg[2]) - zs
        zp = complex(dx[1], dx[2])
        δ = ξ - s
        (abs(z) < 1e-30 || abs(δ) < 1e-30 || J < 1e-30) && continue
        ln_reg = log(z / δ)
        wi = J * w[i]
        zt = zp / z
        for j in 1:nN
            g[j] += pref_G * real(ln_reg) * Nmat[i, j] * wi
            h[j] += pref_H * imag(zt) * Nmat[i, j] * w[i]
        end
    end
    @inbounds for j in 1:nN
        g[j] += pref_G * real(Ian[j])
    end
    a = clamp(real(s), -1.0, 1.0)
    off = norm(_pos_real(poly, nodes, a) - pf) > 1e-14
    if kind == QUASI_SING || (kind == SING && off)
        C1 = correction_C1(s, qsi, w)
        Ns = _shape_complex(poly, s)
        @inbounds for j in 1:nN
            h[j] += pref_H * imag(Ns[j] * C1)
        end
    end
    return
end

function _seed_1d(poly, nodes, pf)
    Δ = nodes[end] - nodes[1]
    ξs = poly.nodes
    ξ0 = (ξs[end] - ξs[1]) * dot(Δ, pf - nodes[1]) / (norm(Δ)^2 + eps()) + ξs[1]
    return clamp(ξ0, -1.0, 1.0)
end

@inline function _geom_i(Nmat, dNmat, nodes, i, nN)
    pg = zero(eltype(nodes))
    dx = zero(eltype(nodes))
    @inbounds for j in 1:nN
        pg += Nmat[i, j] * nodes[j]
        dx += dNmat[i, j] * nodes[j]
    end
    J = norm(dx)
    nrm = J < 1e-30 ? zero(eltype(nodes)) : tan2normal(dx / J)
    return pg, dx, J, nrm
end

# =============================================================================
# 2D Kelvin (isotropic elasticity)
# =============================================================================

"""
    integraelem_dumont!(h, g, dad::BEMdata{<:Elasticity}, elem, nodes, pf, qsi, w)

Fill `h,g` of size `(2, 2·n_nodes)` with Dumont Kelvin integration.
Layout matches assembly: rows = source eqs, cols = (ux,uy) per field node.
"""
function integraelem_dumont!(h::AbstractMatrix, g::AbstractMatrix,
                             dad::BEMdata{<:Elasticity}, elem, nodes, pf::Point2D,
                             qsi, w; threshold::Float64=0.75)
    dad.dimension == 2 || throw(ArgumentError("Kelvin Dumont is 2D only"))
    nN = length(elem)
    size(h, 1) == 2 && size(h, 2) >= 2nN || throw(ArgumentError("h must be 2×(2n)"))
    size(g, 1) == 2 && size(g, 2) >= 2nN || throw(ArgumentError("g must be 2×(2n)"))
    fill!(h, 0.0); fill!(g, 0.0)

    props = dad.properties
    ν = effective_nu(props)
    μ = props.mu
    poly = dad.element_type

    ξ0 = _seed_1d(poly, nodes, pf)
    ξs = singularity_pole(poly, nodes, pf; ξ0=ξ0)
    kind = classify_singularity(ComplexF64(ξs); threshold=threshold)
    s = ComplexF64(ξs)
    a = clamp(real(s), -1.0, 1.0)

    Nmat, dNmat = element_shapefun(poly, elem, qsi)

    # plain GL of full Kelvin kernels
    @inbounds for i in eachindex(qsi)
        pg, dx, J, nrm = _geom_i(Nmat, dNmat, nodes, i, nN)
        J < 1e-30 && continue
        r = pg - pf
        norm(r) < 1e-30 && continue
        U, Tm = fundamental(dad, r, nrm)
        wi = J * w[i]
        for j in 1:nN
            cols = (2(j - 1) + 1):(2j)
            Nj = Nmat[i, j] * wi
            for β in 1:2, α in 1:2
                g[α, cols[β]] += U[α, β] * Nj
                h[α, cols[β]] += Tm[α, β] * Nj
            end
        end
    end
    kind == NO_SING && return kind

    # G log diagonal: U_sing = -β log R · I
    β = (3 - 4ν) / (8π * μ * (1 - ν))
    gcorr = _log_corr_weights(poly, nodes, s, qsi, w, Nmat, dNmat, β)
    @inbounds for j in 1:nN
        cols = (2(j - 1) + 1):(2j)
        c = gcorr[j]
        g[1, cols[1]] += c
        g[2, cols[2]] += c
    end

    # G off-diagonal G1 (quasi-sing, Dumont Eq. 21)
    if kind == QUASI_SING && abs(imag(s)) > 1e-14
        C1 = correction_C1(s, qsi, w)
        Ns = _shape_complex(poly, s)
        zp = _ctan(poly, nodes, s)
        zs = complex(pf[1], pf[2])
        zb_at = _cpos_conj(poly, nodes, s) - conj(zs)
        prefG = 1 / (16π * μ * (1 - ν))
        Ja = _jac_real(poly, nodes, a)
        @inbounds for j in 1:nN
            G1 = zb_at * Ns[j] / zp * C1
            b_c = prefG * Ja * conj(G1)
            R = _complex_firstrow_to_real(0.0 + 0.0im, b_c)
            cols = (2(j - 1) + 1):(2j)
            for β in 1:2, α in 1:2
                g[α, cols[β]] += R[α, β]
            end
        end
    end

    # H: Dumont H1/H2
    off = norm(_pos_real(poly, nodes, a) - pf) > 1e-14
    if kind == QUASI_SING || (kind == SING && off)
        _kelvin_H_correction!(h, poly, nodes, pf, s, qsi, w, ν)
    end
    return kind
end

function _log_corr_weights(poly, nodes, s, qsi, w, Nmat, dNmat, αU)
    nN = size(Nmat, 2)
    Ian = _log_NJ_integral(poly, nodes, s; pJ=max(nN + 2, 6))
    gcorr = zeros(nN)
    @inbounds for j in 1:nN
        Igl = zero(ComplexF64)
        for i in eachindex(qsi)
            δ = qsi[i] - s
            abs(δ) < 1e-15 && continue
            _, _, J, _ = _geom_i(Nmat, dNmat, nodes, i, nN)
            Igl += w[i] * log(δ) * Nmat[i, j] * J
        end
        # U = -αU log R + reg  →  g += -αU * δ(∫log)
        gcorr[j] = -αU * real(Ian[j] - Igl)
    end
    return gcorr
end

function _kelvin_H_correction!(Hm, poly, nodes, pf, s, qsi, w, ν)
    nN = length(nodes)
    pref = im / (8π * (1 - ν))
    sb = conj(ComplexF64(s))
    C1 = correction_C1(s, qsi, w)
    C1b = correction_C1(sb, qsi, w)
    C2 = correction_C2(s, qsi, w)
    Ns = _shape_complex(poly, s)
    Nsb = _shape_complex(poly, sb)
    dNs = _shape_deriv_complex(poly, s)
    zp = _ctan(poly, nodes, s)
    zs = complex(pf[1], pf[2])
    zb_at = _cpos_conj(poly, nodes, s) - conj(zs)

    @inbounds for j in 1:nN
        H1 = (3 - 4ν) * Ns[j] * C1 - Nsb[j] * C1b
        H2 = abs(zp) < 1e-30 ? 0.0 + 0.0im : zb_at / zp * (Ns[j] * C2 + dNs[j] * C1)
        a_c = pref * H1
        b_c = pref * (-conj(H2))
        R = _complex_firstrow_to_real(a_c, b_c)
        cols = (2(j - 1) + 1):(2j)
        for β in 1:2, α in 1:2
            Hm[α, cols[β]] += R[α, β]
        end
    end
end

# =============================================================================
# 2D Lekhnitskii (anisotropic elasticity) — SST (Cordeiro & Leonel 2020)
# =============================================================================

"""
    integraelem_dumont!(h, g, dad::BEMdata{<:AnisotropicElasticity}, …)

Delegates to Subtraction Singularity Technique (`integraelem_sst!`).
"""
function integraelem_dumont!(h::AbstractMatrix, g::AbstractMatrix,
                             dad::BEMdata{<:AnisotropicElasticity}, elem, nodes,
                             pf::Point2D, qsi, w; threshold::Float64=0.75)
    return integraelem_sst!(h, g, dad, elem, nodes, pf, qsi, w; threshold=threshold)
end

"""Complex coordinates (x(ξ), y(ξ)) with complex shape functions (not packed z)."""
function _xy_complex(poly, nodes, ξ::Complex)
    N = _shape_complex(poly, ξ)
    x = zero(ComplexF64)
    y = zero(ComplexF64)
    @inbounds for j in eachindex(N)
        x += N[j] * nodes[j][1]
        y += N[j] * nodes[j][2]
    end
    return x, y
end

function _dxy_complex(poly, nodes, ξ::Complex)
    dN = _shape_deriv_complex(poly, ξ)
    dx = zero(ComplexF64)
    dy = zero(ComplexF64)
    @inbounds for j in eachindex(dN)
        dx += dN[j] * nodes[j][1]
        dy += dN[j] * nodes[j][2]
    end
    return dx, dy
end

"""Complex pole of z_k(ξ) = (x-xs) + μ(y-ys) on the element."""
function _pole_lekhnitskii(poly, nodes, xs, ys, μ; ξ0=0.0, maxiter=40, tol=1e-14)
    ξs = ComplexF64(ξ0)
    for _ in 1:maxiter
        x, y = _xy_complex(poly, nodes, ξs)
        zk = (x - xs) + μ * (y - ys)
        dx, dy = _dxy_complex(poly, nodes, ξs)
        dzk = dx + μ * dy
        abs(dzk) < 1e-30 && break
        Δ = zk / dzk
        ξs -= Δ
        abs(Δ) < tol && break
    end
    return ξs
end
