# Subtraction Singularity Technique (SST) — Cordeiro & Leonel, EABE 119 (2020)
# -----------------------------------------------------------------------------
# Regularise Lekhnitskii U ∼ log z_k and T ∼ 1/z_k by subtracting the leading
# Taylor model about the collocation/closest parameter ξ₀:
#
#   ∫ K φ = ∫ (K − K_s) φ  +  φ(ξ₀) ∫ K_s
#
# Remainder → Gauss–Legendre; singular model → analytic CPV on [-1,1].
# Conventions match BEM_gmsh Fundamental.jl: conj(q)', conj(g)'.
# -----------------------------------------------------------------------------

export integraelem_sst!

"""
    integraelem_sst!(h, g, dad::BEMdata{<:AnisotropicElasticity}, elem, nodes, pf, qsi, w)

SST (Cordeiro & Leonel 2020) for 2D anisotropic elasticity displacement BIE.
`h,g` are 2×(2n) influence blocks.
"""
function integraelem_sst!(h::AbstractMatrix, g::AbstractMatrix,
                          dad::BEMdata{<:AnisotropicElasticity},
                          elem, nodes, pf::Point2D, qsi, w;
                          threshold::Float64=0.75)
    dad.dimension == 2 || throw(ArgumentError("SST Lekhnitskii is 2D only"))
    nN = length(elem)
    size(h, 1) == 2 && size(h, 2) >= 2nN || throw(ArgumentError("h must be 2×(2n)"))
    size(g, 1) == 2 && size(g, 2) >= 2nN || throw(ArgumentError("g must be 2×(2n)"))
    fill!(h, 0.0); fill!(g, 0.0)

    poly = dad.element_type
    p = dad.properties.params
    mi, A, qpar, gmat = p.mi, p.A, p.q, p.g
    xs, ys = pf[1], pf[2]

    seed = _seed_1d(poly, nodes, pf)
    a, pg0, dist = closest_point_1d(poly, nodes, pf; ξ0=seed)
    L = max(elem.Length, norm(nodes[end] - nodes[1]), eps())
    b = dist / L

    # SST (Leonel) is for true on-element singularity: z_k(ξ₀) ≈ 0.
    # Nearly singular (d > 0): kernels stay finite — use adaptive sinh.
    if dist > 1e-12
        if b > threshold
            return _sst_plain_gl!(h, g, dad, poly, nodes, pf, qsi, w, nN)
        end
        return _sst_sinh!(h, g, dad, poly, nodes, pf, a, b, qsi, nN)
    end

    # frozen geometry at ξ₀ = a (paper's straight auxiliary element)
    N0vec, dN0 = shapefun(poly, a)
    dx0 = zero(eltype(nodes))
    @inbounds for j in 1:nN
        dx0 += dN0[j] * nodes[j]
    end
    J0 = norm(dx0)
    J0 < 1e-30 && return _sst_plain_gl!(h, g, dad, poly, nodes, pf, qsi, w, nN)
    # z'_k = x' + μ_k y' = J0 (μ_k n1 - n2)
    zp1 = dx0[1] + mi[1] * dx0[2]
    zp2 = dx0[1] + mi[2] * dx0[2]
    abs(zp1) < 1e-30 && (zp1 += 1e-30)
    abs(zp2) < 1e-30 && (zp2 += 1e-30)

    # analytic singular moments (complex, paper eqs. 30–32)
    I_ln, I_inv = _sst_moments(a)   # ∫ ln(ξ-a) dξ , ∫ dξ/(ξ-a)  (complex CPV)

    Nmat, dNmat = shapefun(poly, qsi)

    # Shape values at ξ₀ (paper freezes N(ξ₀), J(ξ₀) in singular model)
    Nj0 = ntuple(j -> Float64(N0vec[j]), nN)

    # ---- regularised Gauss quadrature (paper eq. 27) ----
    # KU = U N J,  KU_s = U_asymp * N(ξ₀) * J(ξ₀)
    # KT = T N J,  KT_s = T_asymp * N(ξ₀)          [J cancelled in T model]
    @inbounds for i in eachindex(qsi)
        ξ = qsi[i]
        δ = ξ - a
        abs(δ) < 1e-14 && continue
        pg, dx, J, nrm = _geom_i(Nmat, dNmat, nodes, i, nN)
        J < 1e-30 && continue
        r = pg - pf
        norm(r) < 1e-30 && continue

        U_full, T_full = fundamental(dad, r, nrm)
        U_asymp, T_asymp = _sst_model_UT(δ, zp1, zp2, A, qpar, gmat)

        wi = J * w[i]
        for j in 1:nN
            cols = (2(j - 1) + 1):(2j)
            Nj = Nmat[i, j]
            n0j = Nj0[j]
            for β in 1:2, α in 1:2
                # ∫ (U N J − U_asymp N0 J0) dξ
                g[α, cols[β]] += U_full[α, β] * Nj * wi - U_asymp[α, β] * n0j * J0 * w[i]
                # ∫ (T N J − T_asymp N0) dξ
                h[α, cols[β]] += T_full[α, β] * Nj * wi - T_asymp[α, β] * n0j * w[i]
            end
        end
    end

    # ---- analytic ∫ KU_s , ∫ KT_s  (paper eqs. 29–32) ----
    # ∫ U_asymp J0 dξ * N0   and   ∫ T_asymp dξ * N0
    U_an = _sst_U_analytic(I_ln, zp1, zp2, A, qpar)   # ∫ U_asymp dξ
    T_an = _sst_T_analytic(I_inv, A, gmat)             # ∫ T_asymp dξ
    @inbounds for j in 1:nN
        cols = (2(j - 1) + 1):(2j)
        n0j = Nj0[j]
        for β in 1:2, α in 1:2
            g[α, cols[β]] += U_an[α, β] * J0 * n0j
            h[α, cols[β]] += T_an[α, β] * n0j
        end
    end
    return SING
end

# =============================================================================
# Singular models and analytic moments
# =============================================================================

"""
Asymptotic U_s, T_s at offset δ = ξ − ξ₀ (paper eq. 28).
U_s = 2 Re[A diag(ln(δ z'_k)) conj(q)']
T_s = 2 Re[A diag(1/δ) conj(g)']     # so that T_s N0 ≈ T N J singular part
"""
function _sst_model_UT(δ::Float64, zp1, zp2, A, qpar, gmat)
    δc = ComplexF64(δ)
    lnδ = log(δc)   # principal branch
    lns = @SMatrix [lnδ + log(zp1)  0;  0  lnδ + log(zp2)]
    U_s = 2 * real(A * lns * conj(qpar)')
    invs = @SMatrix [1/δc 0; 0 1/δc]
    T_s = 2 * real(A * invs * conj(gmat)')
    return U_s, T_s
end

function _sst_U_analytic(I_ln::ComplexF64, zp1, zp2, A, qpar)
    # ∫ ln(δ z') dξ = I_ln + 2 log(z')
    lns = @SMatrix [I_ln + 2 * log(zp1)  0;  0  I_ln + 2 * log(zp2)]
    return 2 * real(A * lns * conj(qpar)')
end

function _sst_T_analytic(I_inv::ComplexF64, A, gmat)
    invs = @SMatrix [I_inv 0; 0 I_inv]
    return 2 * real(A * invs * conj(gmat)')
end

"""
Complex CPV moments on [-1,1] about a ∈ [-1,1] (paper eqs. 30, 32):

  I_ln  = ∫ ln(ξ − a) dξ = [(1−a)ln(1−a) − (1−a)] − [(−1−a)ln(−1−a) − (−1−a)]
  I_inv = ∫ dξ/(ξ − a)   = ln(1−a) − ln(−1−a)
"""
function _sst_moments(a::Float64)
    a = clamp(a, nextfloat(-1.0), prevfloat(1.0))
    # use complex logs for directed boundary values
    p1 = ComplexF64(1 - a)
    m1 = ComplexF64(-1 - a)
    I_ln = (p1 * log(p1) - p1) - (m1 * log(m1) - m1)
    I_inv = log(p1) - log(m1)
    return I_ln, I_inv
end

function _sst_plain_gl!(h, g, dad, poly, nodes, pf, qsi, w, nN)
    Nmat, dNmat = shapefun(poly, qsi)
    @inbounds for i in eachindex(qsi)
        pg, dx, J, nrm = _geom_i(Nmat, dNmat, nodes, i, nN)
        J < 1e-30 && continue
        r = pg - pf; norm(r) < 1e-30 && continue
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
    return NO_SING
end

function _sst_sinh!(h, g, dad, poly, nodes, pf, a, b, qsi, nN)
    n = nquad_nearfield(b; nmin=max(length(qsi), 24), nmax=160, nbase=length(qsi))
    eta, ww = nearfield_1d(a, b; n=n)
    Nmat, dNmat = shapefun(poly, eta)
    @inbounds for i in eachindex(eta)
        pg, dx, J, nrm = _geom_i(Nmat, dNmat, nodes, i, nN)
        J < 1e-30 && continue
        r = pg - pf; norm(r) < 1e-30 && continue
        U, Tm = fundamental(dad, r, nrm)
        wi = J * ww[i]
        for j in 1:nN
            cols = (2(j - 1) + 1):(2j)
            Nj = Nmat[i, j] * wi
            for β in 1:2, α in 1:2
                g[α, cols[β]] += U[α, β] * Nj
                h[α, cols[β]] += Tm[α, β] * Nj
            end
        end
    end
    return QUASI_SING
end
