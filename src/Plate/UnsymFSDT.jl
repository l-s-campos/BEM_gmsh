# Useche 8.3.1 / Hsu–Hwu (Comp. Struct. 320, 2023; EABE 2024) 5×5
# fundamentals for unsymmetric FSDT (coupled stretching–bending + shear).
# DOF order: (u₁, u₂, β₁, β₂, w). Tractions Hsu–Hwu EABE 156
# T* = (Tx, Ty, Hx, Hy, Qn). Symmetric B=0 stays on `wang_kernels`
# (this Z is singular when B̃=0).
# Assembly + DIBEM: `assemble_fsdt!` / `dibem_fsdt!` on `UnsymFSDTProps`.
# Single traction BIE: `assemble_unsym_fsdt_hbie!` (all boundary `W,S`;
# interiors by Somigliana). Interior resultants: differentiate the CBIE
# then the same T* (`unsym_interior_t`). HBIE `S = L_{nξ}(T*, ∇_ξ T*)`.
# F(ρ) is the Jordan particular integral of F'−JF=ρ^{-2}I (8.24) with the
# even homogeneous constant β=−3/2 that matches Wang U_ww. B=0 stays on
# `wang_kernels`.

"""Unsymmetric FSDT: full ABD + ContsLam shear `AT=[A44 A45; A45 A55]` (K already in AT)."""
mutable struct UnsymFSDTProps <: AbstractFSDTProps
    A::SMatrix{3,3,Float64,9}
    B::SMatrix{3,3,Float64,9}
    D::SMatrix{3,3,Float64,9}
    AT::SMatrix{2,2,Float64,4}
    h::Float64
    ρ::Float64
    q_c::Float64
    nθ::Int
    map::Symbol
    sinh_b::Float64
end

function UnsymFSDTProps(A, B, D, AT; h=0.01, ρ=1.0, q_c=0.0, nθ=12,
        map::Symbol=:telles, sinh_b::Float64=1e-3)
    return UnsymFSDTProps(SMatrix{3,3,Float64}(A), SMatrix{3,3,Float64}(B),
        SMatrix{3,3,Float64}(D), SMatrix{2,2,Float64}(AT),
        Float64(h), Float64(ρ), Float64(q_c), Int(nθ), map, Float64(sinh_b))
end

"""
    laminate_unsym_props(plies; Ks=5/6, G13=nothing, G23=nothing, ρ=1, q_c=0, nθ=12)

ABD (`A`,`B`,`D`) and shear `AT` from plies `(E1,E2,ν12,G12,θ_deg,t)`.
Midplane at ``z=0``. Nonzero `B` is required (unsymmetric stack).
"""
function laminate_unsym_props(plies; Ks=5 / 6, G13=nothing, G23=nothing,
        ρ=1.0, q_c=0.0, nθ=12, map::Symbol=:telles)
    A, B, D, AT, h = _laminate_ABD_AT(plies; Ks=Ks, G13=G13, G23=G23)
    return UnsymFSDTProps(A, B, D, AT; h=h, ρ=ρ, q_c=q_c, nθ=nθ, map=map)
end

bending_stiffness(p::UnsymFSDTProps) = p.D[1, 1]
shear_stiffness(p::UnsymFSDTProps) = p.AT[2, 2]
_ndofn(::UnsymFSDTProps) = 5
n_dof(::UnsymFSDTProps, ::Integer) = 5

# Hsu At = K [A55 A45; A45 A44] — our AT is [A44 A45; A45 A55] with K included.
_hsu_At(AT) = @SMatrix [AT[2, 2] AT[1, 2]; AT[1, 2] AT[1, 1]]

function _ei_real(x::Float64)
    ax = abs(x)
    ax < 1e-30 && return 0.0
    if ax > 40
        # Ei(x) ~ e^x (1/x + 1/x² + …)  /  Ei(-x) = -E₁(x)
        invx = 1 / ax
        s = invx
        term = invx
        for k in 1:20
            term *= k * invx
            s += term
            abs(term) < 1e-16 * abs(s) && break
        end
        return x > 0 ? exp(x) * s : -exp(-ax) * s
    end
    return x > 0 ? Float64(expinti(x)) : -Float64(expint(-x))
end

# exp(x) Ei(-x) and exp(-x) Ei(x), overflow-safe.
function _expei_m(x::Float64)
    # e^x Ei(-x)
    x > 0 && return -Float64(expintx(x))          # -e^x E₁(x)
    x < 0 && return exp(x) * _ei_real(-x)          # e^{-|x|} Ei(|x|)
    return 0.0
end
function _expei_p(x::Float64)
    # e^{-x} Ei(x)
    if x > 40
        invx = 1 / x
        s = invx
        term = invx
        for k in 1:20
            term *= k * invx
            s += term
            abs(term) < 1e-16 * abs(s) && break
        end
        return s
    elseif x > 0
        return exp(-x) * Float64(expinti(x))
    elseif x < 0
        return -Float64(expintx(-x))
    end
    return 0.0
end

"""Jordan blocks of `F(ρ)`. Particular integral of `F'−JF = ρ^{-2} I` (8.24).

Even homogeneous constant `β = −3/2` on each nilpotent chain: this is the unique
Toeplitz constant that matches Wang `U_ww` (same `R² ln R` and `R²` as
`1/(8π D)`). OCR (8.38–8.39) used `β = −1` (`f3 = ρ²(1−2ln)/4`), which leaves
an extra `−ρ²/2` even part and overstates `U_ww` at large `R`.
`f0=−1/ρ`, `f1=−ln|ρ|−3/2`, `f2=−ρ(ln|ρ|+1/2)`, `f3=−½ ρ² ln|ρ|`.
"""
function _unsym_F!(F, ρ; β::Float64=-1.5)
    fill!(F, 0)
    L = log(max(abs(ρ), 1e-30))
    ir = 1 / ρ
    Lm = -L + β
    rL = ρ * (1 - L + β)
    r2L = ρ * ρ * (0.75 - 0.5 * L) + β * ρ * ρ / 2
    @inbounds for off in (0, 2)
        F[1+off, 1+off] = -ir
        F[1+off, 2+off] = Lm
        F[2+off, 2+off] = -ir
    end
    F[5, 5] = -ir; F[5, 6] = Lm; F[5, 7] = rL; F[5, 8] = r2L
    F[6, 6] = -ir; F[6, 7] = Lm; F[6, 8] = rL
    F[7, 7] = -ir; F[7, 8] = Lm
    F[8, 8] = -ir
    return F
end

function _unsym_Fd!(F, ρ, λd)
    # (8.41): e^{λρ} ∫ ρ^{-2} e^{-λρ} dρ = −1/ρ − λ e^{λρ} Ei(−λρ)
    ir = 1 / ρ
    x = λd * ρ
    F[9, 9] = -ir - λd * _expei_m(x)    # λ = +λd
    F[10, 10] = -ir + λd * _expei_p(x)  # λ = −λd
    return F
end

"""`F'(ρ)` of `_unsym_F!`. Maxima `unsym_hbie_maxima.mac`."""
function _unsym_Fp!(Fp, ρ; β::Float64=-1.5)
    fill!(Fp, 0)
    ir = 1 / ρ
    ir2 = ir * ir
    L = log(max(abs(ρ), 1e-30))
    dLm = -ir
    ddiag = ir2
    drL = -L + β
    dr2L = -ρ * (L + 0.5)
    @inbounds for off in (0, 2)
        Fp[1 + off, 1 + off] = ddiag
        Fp[1 + off, 2 + off] = dLm
        Fp[2 + off, 2 + off] = ddiag
    end
    Fp[5, 5] = ddiag
    Fp[5, 6] = dLm
    Fp[5, 7] = drL
    Fp[5, 8] = dr2L
    Fp[6, 6] = ddiag
    Fp[6, 7] = dLm
    Fp[6, 8] = drL
    Fp[7, 7] = ddiag
    Fp[7, 8] = dLm
    Fp[8, 8] = ddiag
    return Fp
end

"""`F'` of the Ei diagonal (8.41)."""
function _unsym_Fdp!(Fp, ρ, λd)
    ir = 1 / ρ
    x = λd * ρ
    Fp[9, 9] = ir * ir - λd^2 * _expei_m(x) - λd * ir
    Fp[10, 10] = ir * ir - λd^2 * _expei_p(x) + λd * ir
    return Fp
end

"""Ω (3×2) and ω (2) for plane-wave direction θ. Useche (8.35)."""
function _unsym_Ωω(θ)
    c, s = cos(θ), sin(θ)
    Ω = @SMatrix [c 0.0; 0.0 s; s c]
    ω = SVector(c, s)
    return Ω, ω
end

function _unsym_Z_L2(θ, A, B, D, At)
    Ω, ω = _unsym_Ωω(θ)
    Ã = Ω' * A * Ω          # 2×2
    B̃ = Ω' * B * Ω
    D̃ = Ω' * D * Ω
    Ãt = dot(ω, At * ω)     # scalar
    Ãt < 1e-30 && return nothing
    L2 = zeros(5, 5)
    L2[1:2, 1:2] .= Ã
    L2[1:2, 3:4] .= B̃
    L2[3:4, 1:2] .= B̃
    L2[3:4, 3:4] .= D̃
    L2[5, 5] = Ãt
    Dred = D̃ - B̃ * (Ã \ B̃)
    Dstar = inv(Dred)                      # 2×2  (8.30)
    I2 = @SMatrix [1.0 0.0; 0.0 1.0]
    Pω = (ω * ω') * At / Ãt
    D0 = Dstar * Ãt * (I2 - Pω)           # (8.29)
    d11, d21 = D0[1, 1], D0[2, 1]
    d = abs(d11) > 1e-14 ? SVector(1.0, d21 / d11) : SVector(0.0, 1.0)
    λd2 = D0[1, 1] + D0[2, 2]
    λd2 <= 0 && return nothing
    λd = sqrt(λd2)
    AiBω = Ã \ (B̃ * ω)
    AiBd = Ã \ (B̃ * d)
    At_d = At * d
    g3p = -(1 / λd) * (dot(ω, At_d) / Ãt)
    g3m = -g3p
    gp = SVector(-AiBd[1], -AiBd[2], d[1], d[2], g3p)
    gm = SVector(-AiBd[1], -AiBd[2], d[1], d[2], g3m)
    # ω6, ω7, ω8  (8.33)
    w6 = SVector(AiBω[1], AiBω[2], -ω[1], -ω[2], 1.0)
    w7 = SVector(0.0, 0.0, -ω[1], -ω[2], 1.0)
    Dsinv = inv(Dstar)
    Atinv = inv(At)
    v8 = -((I2 + Atinv * Dsinv) * ω)
    w8 = SVector(0.0, 0.0, v8[1], v8[2], 0.0)
    Z = zeros(10, 10)
    e1 = SVector(1.0, 0.0, 0.0, 0.0, 0.0)
    e2 = SVector(0.0, 1.0, 0.0, 0.0, 0.0)
    e5 = SVector(0.0, 0.0, 0.0, 0.0, 1.0)
    zcols = (
        (e1, zero(e1)),
        (e1, e1),
        (e2, zero(e2)),
        (e2, e2),
        (e5, zero(e5)),
        (w6, e5),
        (w7, w6),
        (w8, w7),
        (gp, λd * gp),
        (gm, -λd * gm),
    )
    @inbounds for (j, (v, vp)) in enumerate(zcols)
        for i in 1:5
            Z[i, j] = v[i]
            Z[i+5, j] = vp[i]
        end
    end
    return Z, L2, λd
end

"""
    unsym_fsdt_kernels(pg, pf, n, props) -> (U, P)

Hsu–Hwu / Useche 8.3.1 5×5 kernels. `U` multiplies tractions
`(Tx, Ty, Hx, Hy, Qn)` in G (EABE 156); `P` is the same constitutive `T*`
and multiplies kinematics `(u, β, w)` in H.

θ-integral: four π/2 quadrants, clustered at `ρ=0` (`map=` as in Wang).
"""
function unsym_fsdt_kernels(pg::Point2D, pf::Point2D, n::Point2D, p::UnsymFSDTProps)
    RX = pg[1] - pf[1]
    RY = pg[2] - pf[2]
    hypot(RX, RY) < 1e-30 && error("unsym_fsdt_kernels: coincident points")
    At = _hsu_At(p.AT)
    nθ = p.nθ
    eg, wg0 = gausslegendre(nθ)
    Uast = zeros(5, 5)
    Nast = zeros(3, 5)
    Mast = zeros(3, 5)
    Qast = zeros(2, 5)
    F = zeros(10, 10)
    θ0 = atan(-RX, RY)
    n1, n2 = n[1], n[2]
    for iq in 1:4
        eet = (iq == 1 || iq == 3) ? -1.0 : 1.0
        et, wg = _cluster_rule(eg, wg0, eet, p.map; b=p.sinh_b)
        for i in 1:nθ
            ξ = clamp(et[i], -1.0, 1.0)
            θ = θ0 + (ξ + 1) * π / 4
            Ω, ω = _unsym_Ωω(θ)
            ρ = ω[1] * RX + ω[2] * RY
            abs(ρ) < 1e-14 && continue
            got = _unsym_Z_L2(θ, p.A, p.B, p.D, At)
            got === nothing && continue
            Z, L2, λd = got
            cond(Z) > 1e12 && continue
            _unsym_F!(F, ρ)
            _unsym_Fd!(F, ρ, λd)
            M10 = Z * (F / Z)
            L2inv = inv(L2)
            # 10×5 state for the five unit loads: {v; v,ρ} = M10 [0; L2⁻¹]
            Yθ = M10[:, 6:10] * L2inv
            Wθ = Yθ[1:5, :]
            Vp = Yθ[6:10, :]
            wθ = wg[i] * (π / 4)
            Uast .+= Wθ .* wθ
            # N,M from strains (v,ρ); Q = At(β + ω w,ρ)  (8.44)
            Nast .+= (p.A * (Ω * Vp[1:2, :]) + p.B * (Ω * Vp[3:4, :])) .* wθ
            Mast .+= (p.B * (Ω * Vp[1:2, :]) + p.D * (Ω * Vp[3:4, :])) .* wθ
            Qast .+= (At * (Wθ[3:4, :] .+ ω * Vp[5:5, :])) .* wθ
        end
        θ0 += π / 2
    end
    s = 1 / (4 * π^2)
    Uast .*= s
    Nast .*= s
    Mast .*= s
    Qast .*= s
    # Hsu β = −ψ (γ = β+∇w vs Wang γ = ∇w−ψ): U_βw = −U_ψw. Flip the
    # coupling so the 3×3 bending block is Wang's BIE pair (KernelP).
    Uast[3:4, 5] .*= -1
    Uast[5, 3:4] .*= -1
    # Hsu–Hwu EABE 156 T*: Cartesian (Tx, Ty, Hx, Hy, Qn). BIE index
    # order T*_ij = traction_j from load_i. Tiny-B 3×3 matches Wangᵀ.
    return Uast, _unsym_packP(Nast, Mast, Qast, n1, n2)
end

unsym_fsdt_kernels(pg, pf, n, A, B, D, AT; nθ=12, map=:telles, sinh_b=1e-3) =
    unsym_fsdt_kernels(pg, pf, n, UnsymFSDTProps(A, B, D, AT; nθ=nθ, map=map, sinh_b=sinh_b))

"""`r = x - d`. Returns `(U, P)` 5×5 for vectorial `H_G_full_direct`."""
function fundamental(props::UnsymFSDTProps, r::SVector{2}, n::SVector{2})
    return unsym_fsdt_kernels(r, zero(r), n, props)
end
fundamental(dad::BEMdata{<:UnsymFSDTProps}, r::SVector{2}, n::SVector{2}) =
    fundamental(dad.properties, r, n)

"""Hsu–Hwu EABE 156 (2.15b): `T* = (Tx, Ty, Hx, Hy, Qn)` from Voigt `N,M,Q`."""
function _unsym_packP(Nast, Mast, Qast, n1, n2)
    Pn = zeros(5, 5)
    @inbounds for j in 1:5
        Pn[j, 1] = Nast[1, j] * n1 + Nast[3, j] * n2
        Pn[j, 2] = Nast[3, j] * n1 + Nast[2, j] * n2
        Pn[j, 3] = Mast[1, j] * n1 + Mast[3, j] * n2
        Pn[j, 4] = Mast[3, j] * n1 + Mast[2, j] * n2
        Pn[j, 5] = Qast[1, j] * n1 + Qast[2, j] * n2
    end
    return Pn
end

"""Hsu constitutive traction of a 5-vector field `(u, β, w)` and its gradient."""
function _hsu_L(u, ux, uy, nξ, p)
    ε = SVector(ux[1], uy[2], uy[1] + ux[2])
    κ = SVector(ux[3], uy[4], uy[3] + ux[4])
    β = SVector(u[3], u[4])
    ∇w = SVector(ux[5], uy[5])
    N, M, Qv = _unsym_NMQ(ε, κ, β, ∇w, p)
    return _unsym_traction(N, M, Qv, nξ)
end

"""Hsu `T*` of each column of `(U, ∂ξ1 U, ∂ξ2 U)` (Somigliana displacement kernel)."""
function _hsu_L_mat(U, Ux, Uy, nξ, p)
    S = zeros(5, 5)
    @inbounds for j in 1:5
        t = _hsu_L(
            SVector(U[1, j], U[2, j], U[3, j], U[4, j], U[5, j]),
            SVector(Ux[1, j], Ux[2, j], Ux[3, j], Ux[4, j], Ux[5, j]),
            SVector(Uy[1, j], Uy[2, j], Uy[3, j], Uy[4, j], Uy[5, j]),
            nξ, p)
        for i in 1:5
            S[i, j] = t[i]
        end
    end
    return S
end

"""One plane-wave `{v; v,ρ}` and `v,ρρ` (needs `F'` of Hsu `F`)."""
function _unsym_Yθ!(F, Fp, θ, ρ, p, At)
    got = _unsym_Z_L2(θ, p.A, p.B, p.D, At)
    got === nothing && return nothing
    Z, L2, λd = got
    cond(Z) > 1e12 && return nothing
    fill!(F, 0)
    fill!(Fp, 0)
    _unsym_F!(F, ρ)
    _unsym_Fd!(F, ρ, λd)
    _unsym_Fp!(Fp, ρ)
    _unsym_Fdp!(Fp, ρ, λd)
    c = inv(Z)[:, 6:10] * inv(L2)
    Y = Z * (F * c)
    Yp = Z * (Fp * c)
    return Y[1:5, :], Y[6:10, :], Yp[6:10, :]
end

"""Hsu EABE 156 `T*` and `∂T*/∂ρ` at one `θ`, packed as CBIE `P`."""
function _unsym_P_dPρ(θ, RX, RY, n1, n2, p, At, F, Fp)
    Ω, ω = _unsym_Ωω(θ)
    ρ = ω[1] * RX + ω[2] * RY
    abs(ρ) < 1e-14 && return nothing
    got = _unsym_Yθ!(F, Fp, θ, ρ, p, At)
    got === nothing && return nothing
    v, vρ, vρρ = got
    N = p.A * (Ω * vρ[1:2, :]) + p.B * (Ω * vρ[3:4, :])
    M = p.B * (Ω * vρ[1:2, :]) + p.D * (Ω * vρ[3:4, :])
    Qv = At * (v[3:4, :] .+ ω * vρ[5:5, :])
    dN = p.A * (Ω * vρρ[1:2, :]) + p.B * (Ω * vρρ[3:4, :])
    dM = p.B * (Ω * vρρ[1:2, :]) + p.D * (Ω * vρρ[3:4, :])
    dQ = At * (vρ[3:4, :] .+ ω * vρρ[5:5, :])
    Pθ = _unsym_packP(N, M, Qv, n1, n2)
    dP = _unsym_packP(dN, dM, dQ, n1, n2)
    return Pθ, dP, ω
end

"""Hsu traction kernel of the `T*` column: `S = L_{nξ}(T*, ∇_ξ T*)`.

EABE 156 complete solutions: differentiate the displacement BIE, then the
same constitutive `P`. Not `nξ·∇P` of the already-constituted field traction.
`∇_ξ = −ω ∂/∂ρ` (`ρ = ω·(x−ξ)`).
"""
function _unsym_Sθ(θ, RX, RY, n1, n2, nξ, p, At, F, Fp)
    got = _unsym_P_dPρ(θ, RX, RY, n1, n2, p, At, F, Fp)
    got === nothing && return zeros(5, 5)
    Pθ, dP, ω = got
    Px = -ω[1] .* dP
    Py = -ω[2] .* dP
    return _hsu_L_mat(Pθ, Px, Py, nξ, p)
end

"""HFP of `∫ S_θ dθ` on a π-arc whose midpoint is a `ρ=0` pole (`θc`).

Parent `ξ∈[-1,1]`, `θ=θc + ξ π/2`. Interior `1/ρ² ~ 1/ξ²` at `ξ=0`.
Ray Richardson (not the interpolant): `1/sin²(ξ π/2)` is not a stable
`F₋₂` from Gauss nodes of the default Laurent interpolant.
"""
function _unsym_S_pole(θc, RX, RY, n1, n2, nξ, p, nθ)
    At = _hsu_At(p.AT)
    F = zeros(10, 10)
    Fp = zeros(10, 10)
    Jθ = π / 2
    qs, ws = gausslegendre(max(16, 2nθ))
    return guiggiani_integral(0.0, -2; laurent=:richardson, qsi=qs, w=ws) do ξ
        θ = θc + ξ * Jθ
        return _unsym_Sθ(θ, RX, RY, n1, n2, nξ, p, At, F, Fp) .* Jθ
    end
end

"""
    unsym_hbie_kernels(pg, pf, n, nξ, props) -> (W, S)

Hsu–Hwu EABE 156 traction BIE. `W` multiplies field tractions (G), `S`
multiplies kinematics (H, hypersingular).

`W = T*(ξ, x; n_ξ)` (Betti swap of the CBIE Hsu traction kernel).
`S = L_{nξ}(T*, ∇_ξ T*)`: EABE complete solutions (differentiate the
displacement BIE, then the same constitutive `P`). `θ`-integral is
Guiggiani HFP of `1/ρ²` at the two `ω ⊥ r` poles (Richardson rays).
On-element assembly uses Richardson rays too (`S∼1/R²`).
"""
function unsym_hbie_kernels(pg::Point2D, pf::Point2D, n::Point2D, nξ::Point2D,
        p::UnsymFSDTProps)
    RX = pg[1] - pf[1]
    RY = pg[2] - pf[2]
    hypot(RX, RY) < 1e-30 && error("unsym_hbie_kernels: coincident points")
    _, W = unsym_fsdt_kernels(pf, pg, nξ, p)
    n1, n2 = n[1], n[2]
    θ0 = atan(-RX, RY)
    S = _unsym_S_pole(θ0, RX, RY, n1, n2, nξ, p, p.nθ) .+
        _unsym_S_pole(θ0 + π, RX, RY, n1, n2, nξ, p, p.nθ)
    S .*= 1 / (4 * π^2)
    return W, S
end

unsym_hbie_kernels(pg, pf, n, nξ, A, B, D, AT; nθ=12, map=:telles, sinh_b=1e-3) =
    unsym_hbie_kernels(pg, pf, n, nξ,
        UnsymFSDTProps(A, B, D, AT; nθ=nθ, map=map, sinh_b=sinh_b))

"""HFP of `∂U/∂ξ, ∂T*/∂ξ` on a π-arc whose midpoint is a `ρ=0` pole.

`v,ρ ∼ 1/ρ²` at `ω ⊥ r`: same Richardson rays as HBIE `S`.
"""
function _unsym_grad_pole(θc, RX, RY, n1, n2, p, nθ)
    At = _hsu_At(p.AT)
    F = zeros(10, 10)
    Fp = zeros(10, 10)
    Jθ = π / 2
    qs, ws = gausslegendre(max(16, 2nθ))
    return guiggiani_integral(0.0, -2; laurent=:richardson, qsi=qs, w=ws) do ξ
        θ = θc + ξ * Jθ
        Ω, ω = _unsym_Ωω(θ)
        ρ = ω[1] * RX + ω[2] * RY
        abs(ρ) < 1e-14 && return zeros(5, 20)
        got = _unsym_Yθ!(F, Fp, θ, ρ, p, At)
        got === nothing && return zeros(5, 20)
        _, vρ, vρρ = got
        dN = p.A * (Ω * vρρ[1:2, :]) + p.B * (Ω * vρρ[3:4, :])
        dM = p.B * (Ω * vρρ[1:2, :]) + p.D * (Ω * vρρ[3:4, :])
        dQ = At * (vρ[3:4, :] .+ ω * vρρ[5:5, :])
        dP = _unsym_packP(dN, dM, dQ, n1, n2)
        s1, s2 = -ω[1] * Jθ, -ω[2] * Jθ
        return hcat(vρ .* s1, vρ .* s2, dP .* s1, dP .* s2)
    end
end

"""Source derivatives `∂/∂ξ` of Hsu `U, T*` (`ρ = ω·(x−ξ)`).

`U, T*` from the clustered CBIE integral. Gradients are Guiggiani HFP of
`1/ρ²` (same poles as HBIE `S`).
"""
function unsym_fsdt_dkernels(pg::Point2D, pf::Point2D, n::Point2D, p::UnsymFSDTProps)
    RX = pg[1] - pf[1]
    RY = pg[2] - pf[2]
    hypot(RX, RY) < 1e-30 && error("unsym_fsdt_dkernels: coincident points")
    U, P = unsym_fsdt_kernels(pg, pf, n, p)
    n1, n2 = n[1], n[2]
    θ0 = atan(-RX, RY)
    G = _unsym_grad_pole(θ0, RX, RY, n1, n2, p, p.nθ) .+
        _unsym_grad_pole(θ0 + π, RX, RY, n1, n2, p, p.nθ)
    G .*= 1 / (4 * π^2)
    Ux = G[:, 1:5]
    Uy = G[:, 6:10]
    Px = G[:, 11:15]
    Py = G[:, 16:20]
    Ux[3:4, 5] .*= -1
    Ux[5, 3:4] .*= -1
    Uy[3:4, 5] .*= -1
    Uy[5, 3:4] .*= -1
    return U, P, Ux, Uy, Px, Py
end

# =============================================================================
# 5-DOF assembly, DIBEM, Navier
# =============================================================================

@inline function _unsym_on_el(el, i)
    @inbounds for k in eachindex(el.index)
        el.index[k] == i && return true
    end
    return false
end

function _unsym_ξ_on_el(el, i)
    nN = length(el.index)
    qsi, _ = gausslegendre(nN)
    @inbounds for k in 1:nN
        el.index[k] == i && return qsi[k]
    end
    return 0.0
end

"""`(G, H)` kernels: CBIE `(U, P)` or HBIE `(W, S)`."""
function _unsym_GH(pg, pf, n̂, nξ, props, bie::Symbol)
    bie === :hbie && return unsym_hbie_kernels(pg, pf, n̂, nξ, props)
    return unsym_fsdt_kernels(pg, pf, n̂, props)
end

function _add_unsym_el!(He, Ge, Cije, el, poly, pf, props, qs, ws;
        telles=false, eet=0.0, bie::Symbol=:cbie, nξ=Point2D(1.0, 0.0))
    nN = length(el.index)
    for (ig, ξ0) in enumerate(qs)
        ξ, Jt = telles ? _telles(ξ0, eet) : (ξ0, 1.0)
        abs(ξ) > 1 + 1e-12 && continue
        pg, J, n̂ = elem_geom(el, ξ)
        R = norm(pg - pf)
        R < 1e-14 && continue
        U, P = _unsym_GH(pg, pf, n̂, nξ, props, bie)
        Nf, _ = shapefun(poly, ξ)
        wJ = J * ws[ig] * Jt
        @inbounds for a in 1:nN
            Na = Nf[1, a] * wJ
            cols = 5a-4:5a
            He[:, cols] .+= P .* Na
            Ge[:, cols] .+= U .* Na
        end
        Cije .+= P .* wJ
    end
    return nothing
end

"""Near-element Telles + subdivision (CBIE or HBIE)."""
function _unsym_telles_sub!(He, Ge, el, poly, pf, nξ, props, qsi, w;
        nsub::Int=6, map::Symbol=:telles, sinh_b::Float64=1e-3, bie::Symbol=:cbie)
    nN = length(el.index)
    x1, x3 = el.geo[1], el.geo[end]
    ndiv = max(nsub, 4)
    dξ = 2 / ndiv
    for k in 1:ndiv
        ξa = -1 + (k - 1) * dξ
        ξb = -1 + k * dξ
        eet = 0.0
        if abs(x3[1] - x1[1]) > abs(x3[2] - x1[2])
            xa = (1 - ξa) / 2 * x1[1] + (1 + ξa) / 2 * x3[1]
            xb = (1 - ξb) / 2 * x1[1] + (1 + ξb) / 2 * x3[1]
            den = xa - xb
            abs(den) > 1e-14 && (eet = (xa + xb - 2 * pf[1]) / den)
        else
            ya = (1 - ξa) / 2 * x1[2] + (1 + ξa) / 2 * x3[2]
            yb = (1 - ξb) / 2 * x1[2] + (1 + ξb) / 2 * x3[2]
            den = ya - yb
            abs(den) > 1e-14 && (eet = (ya + yb - 2 * pf[2]) / den)
        end
        eet = clamp(eet, -0.999, 0.999)
        ξt, wt = _cluster_rule(qsi, w, eet, map; b=sinh_b)
        Jsub = 0.5 * (ξb - ξa)
        for ig in eachindex(ξt)
            ξ = 0.5 * (ξa + ξb) + 0.5 * (ξb - ξa) * ξt[ig]
            abs(ξ) > 1 + 1e-12 && continue
            pg, J, n̂ = elem_geom(el, ξ)
            R = norm(pg - pf)
            R < 1e-14 && continue
            U, P = _unsym_GH(pg, pf, n̂, nξ, props, bie)
            Nf, _ = shapefun(poly, ξ)
            wJ = J * wt[ig] * Jsub
            @inbounds for a in 1:nN
                Na = Nf[1, a] * wJ
                cols = 5a-4:5a
                He[:, cols] .+= P .* Na
                Ge[:, cols] .+= U .* Na
            end
        end
    end
    return nothing
end

"""On-element Guiggiani. CBIE `(U,P)` orders `(0,-1)`; HBIE `(W,S)` orders `(-1,-2)`."""
function _unsym_guiggiani!(He, Ge, el, poly, pf, nξ, ξ0, props, bie::Symbol;
        ninterp::Int=16)
    nN = length(el.index)
    a = clamp(float(ξ0), nextfloat(-1.0), prevfloat(1.0))
    ncols = 5nN
    z = zeros(5, ncols)
    og, oh = bie === :hbie ? (-1, -2) : (0, -1)
    # CBIE keeps the default interpolant. HBIE `S∼1/R²`: interpolant
    # under-picks `F₋₂`; Richardson rays on the parent element.
    qs, ws = gausslegendre(ninterp)
    method = bie === :hbie ? :richardson : :interp
    Ig, Ih = guiggiani_GH(a; order_G=og, order_H=oh, ninterp=ninterp,
        laurent=method, qsi=qs, w=ws) do ξ
        pg, J, n̂ = elem_geom(el, ξ)
        hypot(pg[1] - pf[1], pg[2] - pf[2]) < 1e-30 && return z, z
        U, P = _unsym_GH(pg, pf, n̂, nξ, props, bie)
        Nf, _ = shapefun(poly, ξ)
        Fg = zeros(5, ncols)
        Fh = zeros(5, ncols)
        @inbounds for j in 1:nN
            NjJ = Nf[1, j] * J
            cols = 5j-4:5j
            Fg[:, cols] .= U .* NjJ
            Fh[:, cols] .= P .* NjJ
        end
        return Fg, Fh
    end
    Ge .+= Ig
    He .+= Ih
    return nothing
end

function _unsym_scale_hbie_rows!(H, G, q, n::Int)
    @inbounds for r in 1:(5n)
        nrm = max(maximum(abs, view(H, r, :)), maximum(abs, view(G, r, :)), 1e-30)
        H[r, :] ./= nrm
        G[r, :] ./= nrm
        q[r] /= nrm
    end
    return nothing
end

"""Build 5×5 `H,G` from Hsu–Hwu kernels.

`bie=:cbie` (default) is the displacement BIE (`U,P`), free term `½I` on
the boundary. `bie=:hbie` is a **single** traction BIE on every boundary
node (`W,S`, `G -= ½I`). Interiors are not collocated; `w` at internal
points is Somigliana (`U,P`) after `solve_fsdt!`. Self-element:
`singular=:guiggiani` (CBIE interpolant `(0,-1)`; HBIE Richardson
`(-1,-2)`) or `:telles`.
"""
function assemble_unsym_fsdt!(mesh::FSDTMesh; npg::Int=8, nsub::Int=6,
        map::Symbol=:telles, sinh_b::Float64=1e-3,
        singular::Symbol=:telles, bie::Symbol=:cbie, ninterp::Int=16,
        scale_hbie::Bool=true)
    (bie === :cbie || bie === :hbie) ||
        throw(ArgumentError("bie must be :cbie or :hbie; got $bie"))
    (singular === :telles || singular === :guiggiani) ||
        throw(ArgumentError("singular must be :telles or :guiggiani; got $singular"))
    n = _n(mesh)
    ni = _ni(mesh)
    ndof = 5n + 5ni
    nb = 5n
    H = zeros(ndof, ndof)
    G = zeros(ndof, nb)
    props = mesh.props
    qsi, w = gausslegendre(npg)
    poly = mesh.element_type
    pts = Point2D[mesh.nodes; mesh.internal]
    I5 = Matrix(1.0 * I(5))
    dummy_n = Point2D(1.0, 0.0)
    nsrc = bie === :hbie ? n : n + ni
    lbl = bie === :hbie ? "unsym FSDT HBIE H,G" : "unsym FSDT H,G"
    @showprogress lbl for i in 1:nsrc
        pf = pts[i]
        rows = 5i-4:5i
        nξ = i <= n ? mesh.Normal[i] : dummy_n
        row_bie = bie === :hbie ? :hbie : :cbie
        for el in mesh.elements
            nN = length(el.index)
            He = zeros(5, 5nN)
            Ge = zeros(5, 5nN)
            Cije = zeros(5, 5)
            x1, x3 = el.geo[1], el.geo[end]
            Le = norm(x3 - x1)
            Rmin = minimum(norm(mesh.nodes[k] - pf) for k in el.index)
            on = i <= n && _unsym_on_el(el, i)
            near = on || Rmin <= Le / 4
            if on && singular === :guiggiani
                ξ0 = _unsym_ξ_on_el(el, i)
                _unsym_guiggiani!(He, Ge, el, poly, pf, nξ, ξ0, props, row_bie;
                    ninterp=ninterp)
            elseif near
                _unsym_telles_sub!(He, Ge, el, poly, pf, nξ, props, qsi, w;
                    nsub=nsub, map=map, sinh_b=sinh_b, bie=row_bie)
            else
                _add_unsym_el!(He, Ge, Cije, el, poly, pf, props, qsi, w;
                    bie=row_bie, nξ=nξ)
            end
            for a in 1:nN
                ja = el.index[a]
                cols = 5ja-4:5ja
                H[rows, cols] .+= He[:, 5a-4:5a]
                G[rows, cols] .+= Ge[:, 5a-4:5a]
            end
        end
        if row_bie === :hbie
            G[rows, rows] .-= 0.5 .* I5
        elseif i <= n
            H[rows, rows] .+= 0.5 .* I5
        else
            H[rows, rows] .+= I5
        end
    end
    if bie === :hbie
        mesh.eq_type = fill(3, n)
        if ni > 0
            @inbounds for k in 1:ni
                rows = 5(n + k)-4:5(n + k)
                H[rows, rows] .= I5
            end
        end
        q = _unsym_q_hbie(mesh; npg=npg)
        scale_hbie && _unsym_scale_hbie_rows!(H, G, q, n)
        mesh.H = H
        mesh.G = G
        mesh.q = q
    else
        mesh.H = H
        mesh.G = G
    end
    return mesh
end

assemble_unsym_fsdt_hbie!(mesh::FSDTMesh; singular::Symbol=:guiggiani, kwargs...) =
    assemble_unsym_fsdt!(mesh; bie=:hbie, singular=singular, kwargs...)

function assemble_unsym_fsdt!(dad::BEMdata{<:AbstractFSDT}; kwargs...)
    assemble!(dad; kwargs...)
    return dad
end

function assemble_unsym_fsdt_hbie!(dad::BEMdata{<:AbstractFSDT}; kwargs...)
    m = FSDTMesh(dad)
    assemble_unsym_fsdt_hbie!(m; kwargs...)
    _sync_fsdt_mesh!(dad, m)
    return dad
end

@inline function _unsym_on_twin_el(el, i, twin)
    t = twin[i]
    t == 0 && return false
    return _unsym_on_el(el, t)
end

function _unsym_scale_dual_hbie_rows!(H, G, q, eq)
    n = length(eq)
    @inbounds for i in 1:n
        eq[i] == 3 || continue
        for k in 1:5
            r = 5i - 5 + k
            nrm = max(maximum(abs, view(H, r, :)), maximum(abs, view(G, r, :)), 1e-30)
            H[r, :] ./= nrm
            G[r, :] ./= nrm
            q[r] /= nrm
        end
    end
    return nothing
end

"""
    assemble_unsym_fsdt_dual!(mesh; npg=8, nsub=6, singular=:guiggiani)

Portela Dual BEM for unsymmetric FSDT. Outer + crack face A: Hsu–Hwu CBIE
`(U, T*)`. Face B: EABE 156 complete-solution traction BIE
`W = T*(ξ,x; n_ξ)`, `S = L_{nξ}(T*, ∇_ξ T*)`. Self/twin Guiggiani
(CBIE interpolant `(0,-1)`, HBIE Richardson `(-1,-2)`). Free terms:
CBIE `H += ½I` (self+twin on the crack), HBIE `G -= ½I` (self+twin).
"""
function assemble_unsym_fsdt_dual!(mesh::FSDTMesh; npg::Int=8, nsub::Int=6,
        singular::Symbol=:guiggiani, ninterp::Int=12, scale_hbie::Bool=true)
    mesh.props isa UnsymFSDTProps ||
        error("assemble_unsym_fsdt_dual!: UnsymFSDTProps")
    (singular === :telles || singular === :guiggiani) ||
        throw(ArgumentError("singular must be :telles or :guiggiani; got $singular"))
    n = _n(mesh)
    ndof = 5n
    H = zeros(ndof, ndof)
    G = zeros(ndof, ndof)
    q = zeros(ndof)
    eq = mesh.eq_type
    twin = mesh.twin
    props = mesh.props
    poly = mesh.element_type
    qsi, w = gausslegendre(npg)
    I5 = Matrix(1.0 * I(5))
    @showprogress "unsym FSDT dual H,G" for i in 1:n
        pf = mesh.nodes[i]
        nξ = mesh.Normal[i]
        rows = 5i-4:5i
        tipo = eq[i]
        row_bie = tipo == 3 ? :hbie : :cbie
        for el in mesh.elements
            nN = length(el.index)
            He = zeros(5, 5nN)
            Ge = zeros(5, 5nN)
            Cije = zeros(5, 5)
            on = _unsym_on_el(el, i) || _unsym_on_twin_el(el, i, twin)
            x1, x3 = el.geo[1], el.geo[end]
            Le = norm(x3 - x1)
            Rmin = minimum(norm(mesh.nodes[k] - pf) for k in el.index)
            near = on || Rmin <= Le / 4
            if on && singular === :guiggiani
                src = _unsym_on_el(el, i) ? i : twin[i]
                ξ0 = _unsym_ξ_on_el(el, src)
                _unsym_guiggiani!(He, Ge, el, poly, pf, nξ, ξ0, props, row_bie;
                    ninterp=ninterp)
            elseif near
                _unsym_telles_sub!(He, Ge, el, poly, pf, nξ, props, qsi, w;
                    nsub=nsub, bie=row_bie)
            else
                _add_unsym_el!(He, Ge, Cije, el, poly, pf, props, qsi, w;
                    bie=row_bie, nξ=nξ)
            end
            for a in 1:nN
                ja = el.index[a]
                cols = 5ja-4:5ja
                H[rows, cols] .+= He[:, 5a-4:5a]
                G[rows, cols] .+= Ge[:, 5a-4:5a]
            end
        end
        if tipo == 3
            G[rows, rows] .-= 0.5 .* I5
            tw = twin[i]
            tw != 0 && (G[rows, 5tw-4:5tw] .-= 0.5 .* I5)
        else
            H[rows, rows] .+= 0.5 .* I5
            if tipo == 2
                tw = twin[i]
                tw != 0 && (H[rows, 5tw-4:5tw] .+= 0.5 .* I5)
            end
        end
    end
    scale_hbie && _unsym_scale_dual_hbie_rows!(H, G, q, eq)
    mesh.H = H
    mesh.G = G
    mesh.q = q
    return mesh
end

function assemble_unsym_fsdt_dual!(dad::BEMdata{<:AbstractFSDT}; kwargs...)
    dad.properties isa UnsymFSDTProps ||
        error("assemble_unsym_fsdt_dual!: UnsymFSDTProps")
    m = FSDTMesh(dad)
    assemble_unsym_fsdt_dual!(m; kwargs...)
    _sync_fsdt_mesh!(dad, m)
    return dad
end

"""`∫_0^R U*(ρ ê) ρ dρ` by Telles–Gauss (no Maxima primitive)."""
function _unsym_Fρ(pg::Point2D, pf::Point2D, props::UnsymFSDTProps; nρ::Int=8)
    RX, RY = pg[1] - pf[1], pg[2] - pf[2]
    R = hypot(RX, RY)
    R < 1e-30 && return zeros(5, 5)
    e = SVector(RX / R, RY / R)
    dummy = Point2D(1.0, 0.0)
    xs, ws = gausslegendre(nρ)
    Acc = zeros(5, 5)
    for (g, w) in zip(xs, ws)
        ξ, Jt = _telles(g, -1.0)
        ρ = (ξ + 1) / 2 * R
        ρ < 1e-16 && continue
        pgρ = pf + ρ * e
        U, _ = unsym_fsdt_kernels(pgρ, pf, dummy, props)
        Acc .+= U .* (ρ * (R / 2) * w * Jt)
    end
    return Acc
end

function _unsym_ID(mesh::FSDTMesh; npg::Int=8)
    n = _n(mesh)
    ni = _ni(mesh)
    pts = Point2D[mesh.nodes; mesh.internal]
    nsrc = n + ni
    ID = zeros(5nsrc, 5)
    qsi, w = gausslegendre(npg)
    props = mesh.props
    @showprogress "unsym ID (RIM)" for i in 1:nsrc
        pf = pts[i]
        Acc = zeros(5, 5)
        for el in mesh.elements
            for (ig, ξ) in enumerate(qsi)
                pg, J, n̂ = elem_geom(el, ξ)
                RX, RY = pg[1] - pf[1], pg[2] - pf[2]
                R = hypot(RX, RY)
                R < 1e-14 && continue
                nr = (n̂[1] * RX + n̂[2] * RY) / R
                F = _unsym_Fρ(pg, pf, props)
                Acc .+= F .* (nr / R * J * w[ig])
            end
        end
        ID[5i-4:5i, :] .= Acc
    end
    return ID
end

"""`∫_0^R W*(ρ ê; n_ξ) ρ dρ` — HBIE domain-load RIM primitive."""
function _unsym_Fρ_W(pg::Point2D, pf::Point2D, nξ::Point2D, props::UnsymFSDTProps;
        nρ::Int=8)
    RX, RY = pg[1] - pf[1], pg[2] - pf[2]
    R = hypot(RX, RY)
    R < 1e-30 && return zeros(5, 5)
    e = SVector(RX / R, RY / R)
    xs, ws = gausslegendre(nρ)
    Acc = zeros(5, 5)
    for (g, w) in zip(xs, ws)
        ξ, Jt = _telles(g, -1.0)
        ρ = (ξ + 1) / 2 * R
        ρ < 1e-16 && continue
        pgρ = pf + ρ * e
        _, W = unsym_fsdt_kernels(pf, pgρ, nξ, props)
        Acc .+= W .* (ρ * (R / 2) * w * Jt)
    end
    return Acc
end

"""RIM of HBIE `W` on boundary collocation (`n_ξ` = nodal normal)."""
function _unsym_ID_W(mesh::FSDTMesh; npg::Int=8)
    n = _n(mesh)
    ID = zeros(5n, 5)
    qsi, w = gausslegendre(npg)
    props = mesh.props
    @showprogress "unsym ID_W (RIM)" for i in 1:n
        pf = mesh.nodes[i]
        nξ = mesh.Normal[i]
        Acc = zeros(5, 5)
        for el in mesh.elements
            for (ig, ξ) in enumerate(qsi)
                pg, J, n̂ = elem_geom(el, ξ)
                RX, RY = pg[1] - pf[1], pg[2] - pf[2]
                R = hypot(RX, RY)
                R < 1e-14 && continue
                nr = (n̂[1] * RX + n̂[2] * RY) / R
                F = _unsym_Fρ_W(pg, pf, nξ, props)
                Acc .+= F .* (nr / R * J * w[ig])
            end
        end
        ID[5i-4:5i, :] .= Acc
    end
    return ID
end

"""HBIE domain load: `q_i = q_c ∫_Ω W_{i5} dΩ` (Gao RIM of `W`, all 5 traction rows).

`W` is packed as CBIE `P` / assembled `G`, so column 5 is the unit `w`-load,
the same slot as CBIE `U_{•5}`. Interiors are not collocated.
"""
function _unsym_q_hbie(mesh::FSDTMesh; npg::Int=8)
    n = _n(mesh)
    ni = _ni(mesh)
    qc = mesh.props.q_c
    q = zeros(5n + 5ni)
    iszero(qc) && return q
    IDW = _unsym_ID_W(mesh; npg=npg)
    @inbounds for i in 1:n
        q[5i-4:5i] .= IDW[5i-4:5i, 5] .* qc
    end
    return q
end

"""`∫_Ω U*(x,ξ) dΩ` at one source (Gao RIM), 5×5."""
function _unsym_ID_point(mesh::FSDTMesh, pf::Point2D; npg::Int=8)
    Acc = zeros(5, 5)
    qsi, w = gausslegendre(npg)
    props = mesh.props
    for el in mesh.elements
        for (ig, ξ) in enumerate(qsi)
            pg, J, n̂ = elem_geom(el, ξ)
            RX, RY = pg[1] - pf[1], pg[2] - pf[2]
            R = hypot(RX, RY)
            R < 1e-14 && continue
            nr = (n̂[1] * RX + n̂[2] * RY) / R
            F = _unsym_Fρ(pg, pf, props)
            Acc .+= F .* (nr / R * J * w[ig])
        end
    end
    return Acc
end

function _unsym_ID_W_point(mesh::FSDTMesh, pf::Point2D, nξ::Point2D; npg::Int=8)
    Acc = zeros(5, 5)
    qsi, w = gausslegendre(npg)
    props = mesh.props
    for el in mesh.elements
        for (ig, ξ) in enumerate(qsi)
            pg, J, n̂ = elem_geom(el, ξ)
            RX, RY = pg[1] - pf[1], pg[2] - pf[2]
            R = hypot(RX, RY)
            R < 1e-14 && continue
            nr = (n̂[1] * RX + n̂[2] * RY) / R
            F = _unsym_Fρ_W(pg, pf, nξ, props)
            Acc .+= F .* (nr / R * J * w[ig])
        end
    end
    return Acc
end

"""CBIE Somigliana `(u, β, w)` at an interior point from solved boundary `u,t`."""
function unsym_interior_u(mesh::FSDTMesh, pf::Point2D; npg::Int=8, nsub::Int=6)
    isempty(mesh.u) && error("solve_fsdt! first")
    props = mesh.props
    poly = mesh.element_type
    qsi, w = gausslegendre(npg)
    dummy = Point2D(1.0, 0.0)
    ub, tb = mesh.u, mesh.t
    acc = zeros(5)
    for el in mesh.elements
        nN = length(el.index)
        He = zeros(5, 5nN)
        Ge = zeros(5, 5nN)
        Cije = zeros(5, 5)
        x1, x3 = el.geo[1], el.geo[end]
        Le = norm(x3 - x1)
        Rmin = minimum(norm(mesh.nodes[j] - pf) for j in el.index)
        if Rmin <= Le / 4
            _unsym_telles_sub!(He, Ge, el, poly, pf, dummy, props, qsi, w;
                nsub=nsub, bie=:cbie)
        else
            _add_unsym_el!(He, Ge, Cije, el, poly, pf, props, qsi, w; bie=:cbie)
        end
        for a in 1:nN
            ja = el.index[a]
            acc .+= Ge[:, 5a-4:5a] * tb[5ja-4:5ja]
            acc .-= He[:, 5a-4:5a] * ub[5ja-4:5ja]
        end
    end
    qc = props.q_c
    qc != 0 && (acc .+= _unsym_ID_point(mesh, pf; npg=npg)[:, 5] .* qc)
    return acc
end

"""Hsu–Hwu EABE 156 complete solution: traction at an interior point.

Differentiate the CBIE Somigliana identity w.r.t. the source, then the same
constitutive `T* = (Tx, Ty, Hx, Hy, Qn)` as the CBIE traction kernel.
No free term (interior). Domain term is `∂ξ` of `q_c ∫ U_{•5} dΩ`.
"""
function unsym_interior_t(mesh::FSDTMesh, pf::Point2D, nξ::Point2D;
        npg::Int=8, nsub::Int=6)
    isempty(mesh.u) && error("solve_fsdt! first")
    props = mesh.props
    poly = mesh.element_type
    qsi, w = gausslegendre(npg)
    ub, tb = mesh.u, mesh.t
    u = unsym_interior_u(mesh, pf; npg=npg, nsub=nsub)
    ux = zeros(5)
    uy = zeros(5)
    for el in mesh.elements
        nN = length(el.index)
        x1, x3 = el.geo[1], el.geo[end]
        Le = norm(x3 - x1)
        Rmin = minimum(norm(mesh.nodes[j] - pf) for j in el.index)
        near = Rmin <= Le / 4
        ndiv = near ? max(nsub, 4) : 1
        dξ = 2 / ndiv
        for k in 1:ndiv
            ξa = -1 + (k - 1) * dξ
            ξb = -1 + k * dξ
            eet = 0.0
            if near
                if abs(x3[1] - x1[1]) > abs(x3[2] - x1[2])
                    xa = (1 - ξa) / 2 * x1[1] + (1 + ξa) / 2 * x3[1]
                    xb = (1 - ξb) / 2 * x1[1] + (1 + ξb) / 2 * x3[1]
                    den = xa - xb
                    abs(den) > 1e-14 && (eet = (xa + xb - 2 * pf[1]) / den)
                else
                    ya = (1 - ξa) / 2 * x1[2] + (1 + ξa) / 2 * x3[2]
                    yb = (1 - ξb) / 2 * x1[2] + (1 + ξb) / 2 * x3[2]
                    den = ya - yb
                    abs(den) > 1e-14 && (eet = (ya + yb - 2 * pf[2]) / den)
                end
                eet = clamp(eet, -0.999, 0.999)
            end
            ξt, wt = near ? _cluster_rule(qsi, w, eet, :telles) : (qsi, w)
            Jsub = 0.5 * (ξb - ξa)
            for ig in eachindex(ξt)
                ξ = ndiv == 1 ? ξt[ig] :
                    0.5 * (ξa + ξb) + 0.5 * (ξb - ξa) * ξt[ig]
                abs(ξ) > 1 + 1e-12 && continue
                pg, J, n̂ = elem_geom(el, ξ)
                R = norm(pg - pf)
                R < 1e-14 && continue
                _, _, dUx, dUy, Px, Py = unsym_fsdt_dkernels(pg, pf, n̂, props)
                Nf, _ = shapefun(poly, ξ)
                wJ = J * wt[ig] * Jsub
                @inbounds for a in 1:nN
                    ja = el.index[a]
                    Na = Nf[1, a] * wJ
                    tloc = tb[5ja-4:5ja]
                    uloc = ub[5ja-4:5ja]
                    ux .+= (dUx * tloc - Px * uloc) .* Na
                    uy .+= (dUy * tloc - Py * uloc) .* Na
                end
            end
        end
    end
    qc = props.q_c
    if qc != 0
        h = 1e-4
        Ix = _unsym_ID_point(mesh, pf + SVector(h, 0.0); npg=npg)[:, 5] .* qc
        Imx = _unsym_ID_point(mesh, pf - SVector(h, 0.0); npg=npg)[:, 5] .* qc
        Iy = _unsym_ID_point(mesh, pf + SVector(0.0, h); npg=npg)[:, 5] .* qc
        Imy = _unsym_ID_point(mesh, pf - SVector(0.0, h); npg=npg)[:, 5] .* qc
        ux .+= (Ix - Imx) / (2h)
        uy .+= (Iy - Imy) / (2h)
    end
    return _hsu_L(SVector{5}(u), SVector{5}(ux), SVector{5}(uy), nξ, props)
end

"""Voigt `N, M` and `Q` from Hsu `(u, β, w)` gradients. `Q = At(β+∇w)`."""
function _unsym_NMQ(ε, κ, β, ∇w, p::UnsymFSDTProps)
    N = p.A * ε + p.B * κ
    M = p.B * ε + p.D * κ
    At = _hsu_At(p.AT)
    Qv = At * (β .+ ∇w)
    return N, M, Qv
end

function _unsym_traction(N, M, Qv, nξ)
    n1, n2 = nξ[1], nξ[2]
    return SVector(
        N[1] * n1 + N[3] * n2,
        N[3] * n1 + N[2] * n2,
        M[1] * n1 + M[3] * n2,
        M[3] * n1 + M[2] * n2,
        Qv[1] * n1 + Qv[2] * n2)
end

"""Constitutive traction at `pf` from central differences of CBIE Somigliana `u`."""
function unsym_interior_t_fd(mesh::FSDTMesh, pf::Point2D, nξ::Point2D;
        h::Float64=1e-3, npg::Int=8, nsub::Int=6)
    hx, hy = SVector(h, 0.0), SVector(0.0, h)
    upx = unsym_interior_u(mesh, pf + hx; npg=npg, nsub=nsub)
    umx = unsym_interior_u(mesh, pf - hx; npg=npg, nsub=nsub)
    upy = unsym_interior_u(mesh, pf + hy; npg=npg, nsub=nsub)
    umy = unsym_interior_u(mesh, pf - hy; npg=npg, nsub=nsub)
    u0 = unsym_interior_u(mesh, pf; npg=npg, nsub=nsub)
    ux = (upx - umx) / (2h)
    uy = (upy - umy) / (2h)
    ε = SVector(ux[1], uy[2], uy[1] + ux[2])
    κ = SVector(ux[3], uy[4], uy[3] + ux[4])
    β = SVector(u0[3], u0[4])
    ∇w = SVector(ux[5], uy[5])
    N, M, Qv = _unsym_NMQ(ε, κ, β, ∇w, mesh.props)
    t = _unsym_traction(N, M, Qv, nξ)
    return t, (N=N, M=M, Q=Qv, u=u0)
end

unsym_interior_u(dad::BEMdata{<:AbstractFSDT}, pf::Point2D; kwargs...) =
    unsym_interior_u(FSDTMesh(dad), pf; kwargs...)
unsym_interior_t(dad::BEMdata{<:AbstractFSDT}, pf::Point2D, nξ::Point2D; kwargs...) =
    unsym_interior_t(FSDTMesh(dad), pf, nξ; kwargs...)
unsym_interior_t_fd(dad::BEMdata{<:AbstractFSDT}, pf::Point2D, nξ::Point2D; kwargs...) =
    unsym_interior_t_fd(FSDTMesh(dad), pf, nξ; kwargs...)

"""Interior displacement by CBIE Somigliana from solved boundary `u,t` + `U` RIM."""
function _unsym_somigliana_interior!(mesh::FSDTMesh; npg::Int=8, nsub::Int=6)
    n = _n(mesh)
    ni = _ni(mesh)
    ni == 0 && return mesh
    length(mesh.u) >= 5(n + ni) || return mesh
    @inbounds for k in 1:ni
        mesh.u[5(n + k)-4:5(n + k)] .= unsym_interior_u(mesh, mesh.internal[k];
            npg=npg, nsub=nsub)
    end
    return mesh
end

"""Laplace-style DIBEM for Hsu 5×5: mass on `(u,v,w)` with `I0`, `β` with `I2`."""
function _dibem_rim_unsym!(mesh::FSDTMesh; npg::Int=8, rbf=PHS())
    isempty(mesh.H) && assemble_unsym_fsdt!(mesh; npg=npg)
    n = _n(mesh)
    ni = _ni(mesh)
    ndof = 5n + 5ni
    pts = Point2D[mesh.nodes; mesh.internal]
    nt = length(pts)
    ID = _unsym_ID(mesh; npg=npg)
    IF = _fsdt_IF(mesh, pts, rbf; npg=npg)
    Frbf = zeros(nt, nt)
    @inbounds for j in 1:nt, i in 1:nt
        i == j && continue
        Frbf[i, j] = rbf(norm(pts[i] - pts[j]))
    end
    _dibem_ridge_F!(Frbf)
    IP = _fsdt_monomial_IP(mesh, rbf; npg=npg)
    c = _dibem_poly_c(Frbf, IF, pts, rbf; IP=IP)
    dummy = Point2D(1.0, 0.0)
    M = zeros(ndof, ndof)
    props = mesh.props
    @inbounds for j in 1:nt
        cj = c[j]
        abs(cj) < 1e-30 && continue
        xj = pts[j]
        for i in 1:nt
            i == j && continue
            U, _ = unsym_fsdt_kernels(xj, pts[i], dummy, props)
            M[5i-4:5i, 5j-4:5j] .= U .* cj
        end
    end
    @inbounds for i in 1:nt
        rowsum = zeros(5, 5)
        for j in 1:nt
            j == i && continue
            rowsum .+= M[5i-4:5i, 5j-4:5j]
        end
        M[5i-4:5i, 5i-4:5i] .= ID[5i-4:5i, :] .- rowsum
    end
    I2 = props.ρ * props.h^3 / 12
    I0 = props.ρ * props.h
    @inbounds for j in 1:nt
        M[:, 5j-4] .*= I0
        M[:, 5j-3] .*= I0
        M[:, 5j-2] .*= I2
        M[:, 5j-1] .*= I2
        M[:, 5j] .*= I0
    end
    mesh.M = M
    q = zeros(ndof)
    qc = props.q_c
    @inbounds for i in 1:nt
        q[5i-4:5i] .= ID[5i-4:5i, 5] .* qc
    end
    mesh.q = q
    _dibem_ibp_wmaps!(mesh, pts, c, dummy)
    return mesh
end

"""5-DOF Navier SS unsymmetric FSDT (`B≠0`). Hsu order `{U,V,βx,βy,W}`."""
function navier_w_ss_unsym(x, y, p::UnsymFSDTProps; a=1.0, q=p.q_c, nterms=19)
    A, B, D, AT = p.A, p.B, p.D, p.AT
    A11, A22, A12, A66 = A[1, 1], A[2, 2], A[1, 2], A[3, 3]
    B11, B22, B12, B66 = B[1, 1], B[2, 2], B[1, 2], B[3, 3]
    D11, D22, D12, D66 = D[1, 1], D[2, 2], D[1, 2], D[3, 3]
    A44, A55 = AT[1, 1], AT[2, 2]
    w = 0.0
    @inbounds for m in 1:2:nterms, n in 1:2:nterms
        α = m * π / a
        β = n * π / a
        K = zeros(5, 5)
        K[1, 1] = A11 * α^2 + A66 * β^2
        K[1, 2] = (A12 + A66) * α * β
        K[1, 3] = B11 * α^2 + B66 * β^2
        K[1, 4] = (B12 + B66) * α * β
        K[2, 2] = A22 * β^2 + A66 * α^2
        K[2, 3] = (B12 + B66) * α * β
        K[2, 4] = B22 * β^2 + B66 * α^2
        K[3, 3] = D11 * α^2 + D66 * β^2 + A55
        K[3, 4] = (D12 + D66) * α * β
        K[3, 5] = A55 * α
        K[4, 4] = D22 * β^2 + D66 * α^2 + A44
        K[4, 5] = A44 * β
        K[5, 5] = A55 * α^2 + A44 * β^2
        K[2, 1] = K[1, 2]; K[3, 1] = K[1, 3]; K[4, 1] = K[1, 4]
        K[3, 2] = K[2, 3]; K[4, 2] = K[2, 4]
        K[4, 3] = K[3, 4]; K[5, 3] = K[3, 5]; K[5, 4] = K[4, 5]
        Δ = K \ [0.0, 0.0, 0.0, 0.0, 16q / (π^2 * m * n)]
        w += Δ[5] * sin(α * x) * sin(β * y)
    end
    return w
end

# =============================================================================
# Von Kármán large deflection (5-DOF Hsu–Hwu)
# =============================================================================

"""DIBEM maps for distributed `fx, fy, m_x, m_y, q` (unscale mass `I0`/`I2`)."""
function _unsym_dibem_maps(mesh::FSDTMesh)
    ndof = _ndof(mesh)
    nt = _n(mesh) + _ni(mesh)
    M = mesh.M
    I0 = mesh.props.ρ * mesh.props.h
    I2 = mesh.props.ρ * mesh.props.h^3 / 12
    Mu = zeros(ndof, nt)
    Mv = zeros(ndof, nt)
    Mbx = zeros(ndof, nt)
    Mby = zeros(ndof, nt)
    Mw = zeros(ndof, nt)
    @inbounds for j in 1:nt
        Mu[:, j] .= M[:, 5j - 4] ./ I0
        Mv[:, j] .= M[:, 5j - 3] ./ I0
        Mbx[:, j] .= M[:, 5j - 2] ./ I2
        Mby[:, j] .= M[:, 5j - 1] ./ I2
        Mw[:, j] .= M[:, 5j] ./ I0
    end
    return (Mu=Mu, Mv=Mv, Mbx=Mbx, Mby=Mby, Mw=Mw, Mx=mesh.Mx, My=mesh.My)
end

function _unsym_pack_u(u, nt)
    T = eltype(u)
    um = Vector{T}(undef, nt)
    vm = Vector{T}(undef, nt)
    bx = Vector{T}(undef, nt)
    by = Vector{T}(undef, nt)
    w = Vector{T}(undef, nt)
    @inbounds for i in 1:nt
        um[i] = u[5i - 4]
        vm[i] = u[5i - 3]
        bx[i] = u[5i - 2]
        by[i] = u[5i - 1]
        w[i] = u[5i]
    end
    return um, vm, bx, by, w
end

"""Von Kármán extras: `N_vk = A ε_NL`, `M_vk = B ε_NL`.

Linear `A ε_L + B κ` already lives in `H`. Membrane body force is
`div N_vk`; couples `div M_vk`. Transverse term is one IBP
`∫ U* ∇·(N∇w) = Γ + Mx vx + My vy`.
"""
function _unsym_vk_loads(mesh::FSDTMesh, u::AbstractVector, Dx, Dy)
    p = mesh.props
    nt = _n(mesh) + _ni(mesh)
    T = eltype(u)
    um, vm, βx, βy, w = _unsym_pack_u(u, nt)
    wx, wy = Dx * w, Dy * w
    ux, uy = Dx * um, Dy * um
    vx, vy = Dx * vm, Dy * vm
    κx, κy = Dx * βx, Dy * βy
    κs = Dy * βx .+ Dx * βy
    εxL, εyL, γL = ux, vy, uy .+ vx
    εxN = T(0.5) .* wx .^ 2
    εyN = T(0.5) .* wy .^ 2
    γN = wx .* wy
    A, B = p.A, p.B
    Nxx_L = A[1, 1] .* εxL .+ A[1, 2] .* εyL .+ A[1, 3] .* γL .+
            B[1, 1] .* κx .+ B[1, 2] .* κy .+ B[1, 3] .* κs
    Nyy_L = A[1, 2] .* εxL .+ A[2, 2] .* εyL .+ A[2, 3] .* γL .+
            B[1, 2] .* κx .+ B[2, 2] .* κy .+ B[2, 3] .* κs
    Nxy_L = A[1, 3] .* εxL .+ A[2, 3] .* εyL .+ A[3, 3] .* γL .+
            B[1, 3] .* κx .+ B[2, 3] .* κy .+ B[3, 3] .* κs
    Nxx_N = A[1, 1] .* εxN .+ A[1, 2] .* εyN .+ A[1, 3] .* γN
    Nyy_N = A[1, 2] .* εxN .+ A[2, 2] .* εyN .+ A[2, 3] .* γN
    Nxy_N = A[1, 3] .* εxN .+ A[2, 3] .* εyN .+ A[3, 3] .* γN
    Mxx_N = B[1, 1] .* εxN .+ B[1, 2] .* εyN .+ B[1, 3] .* γN
    Myy_N = B[1, 2] .* εxN .+ B[2, 2] .* εyN .+ B[2, 3] .* γN
    Mxy_N = B[1, 3] .* εxN .+ B[2, 3] .* εyN .+ B[3, 3] .* γN
    Nxx = Nxx_L .+ Nxx_N
    Nyy = Nyy_L .+ Nyy_N
    Nxy = Nxy_L .+ Nxy_N
    fx = Dx * Nxx_N .+ Dy * Nxy_N
    fy = Dx * Nxy_N .+ Dy * Nyy_N
    mx = Dx * Mxx_N .+ Dy * Mxy_N
    my = Dx * Mxy_N .+ Dy * Myy_N
    vx = Nxx .* wx .+ Nxy .* wy
    vy = Nxy .* wx .+ Nyy .* wy
    return fx, fy, mx, my, vx, vy
end

function _unsym_vk_rhs(mesh::FSDTMesh, u, maps, Dx, Dy)
    fx, fy, mx, my, vx, vy = _unsym_vk_loads(mesh, u, Dx, Dy)
    rhs = maps.Mu * fx .+ maps.Mv * fy .+ maps.Mbx * mx .+ maps.Mby * my
    if !isempty(maps.Mx)
        return rhs .+ ibp_div_Uv(maps.Mx, maps.My, mesh.G, mesh.Normal, vx, vy, 5)
    end
    return rhs
end

function _unsym_disp_from_mixed(is_kin, known, x)
    T = eltype(x)
    u = Vector{T}(undef, length(x))
    @inbounds for dof in eachindex(x)
        u[dof] = is_kin[dof] ? T(known[dof]) : x[dof]
    end
    return u
end

function _unsym_write_mixed!(mesh, is_kin, known, x)
    ndof = length(x)
    nb = _nb(mesh)
    u = zeros(eltype(x), ndof)
    t = zeros(eltype(x), nb)
    @inbounds for dof in 1:ndof
        if is_kin[dof]
            u[dof] = known[dof]
            dof <= nb && (t[dof] = x[dof])
        else
            u[dof] = x[dof]
            dof <= nb && (t[dof] = known[dof])
        end
    end
    mesh.u = Float64.(u)
    mesh.t = Float64.(t)
    return u, t
end

"""
    solve_unsym_fsdt_large!(mesh; nsteps=8, λ_max=1, nonlinear=:picard)

Von Kármán on Hsu–Hwu 5-DOF (`UnsymFSDTProps`). Linear `H,G` is the
unsymmetric FSDT BEM; extras are DIBEM domain loads from `½∇w⊗∇w` and
`∇·(N ∇w)`. `nonlinear=:picard` or `:newton` (ForwardDiff Jacobian).
`λ` scales the linear DIBEM pressure `mesh.q`.
"""
function solve_unsym_fsdt_large!(mesh::FSDTMesh; nsteps::Int=8, λ_max::Float64=1.0,
        nonlinear::Symbol=:picard, e_relax::Float64=0.5, maxiters::Int=12,
        atol::Float64=1e-6, rbf=PHS(3; poly_deg=1))
    mesh.ndofn == 5 || error("solve_unsym_fsdt_large! needs Hsu–Hwu 5-DOF")
    mesh.props isa UnsymFSDTProps ||
        error("solve_unsym_fsdt_large! needs UnsymFSDTProps (nonzero B)")
    isempty(mesh.H) && assemble_fsdt!(mesh)
    isempty(mesh.M) && dibem_fsdt!(mesh)
    iszero(mesh.q) && mesh.props.q_c != 0 && dibem_fsdt!(mesh)
    maps = _unsym_dibem_maps(mesh)
    pts = Point2D[_plate_nodes(mesh); _plate_internal(mesh)]
    ops = rbf_gradient_ops(pts; rbf=rbf)
    Dx, Dy = ops.Fx, ops.Fy
    A, b_full, is_kin, known = apply_bc_fsdt(mesh)
    q0 = copy(mesh.q)
    b_bc = b_full .- q0
    ndof = length(q0)
    n = _n(mesh)
    ni = _ni(mesh)
    iw = ni > 0 ? 5(n + 1) : 5
    x = zeros(ndof)
    x_prev = zeros(ndof)
    λs = Float64[]
    wcs = Float64[]
    @showprogress "unsym FSDT large-deflection" for step in 1:nsteps
        λ = λ_max * step / nsteps
        b0 = λ .* q0 .+ b_bc
        x_lin = A \ b0
        if step == 1
            x .= x_lin
        else
            x .= x_prev
        end
        npic = nonlinear === :picard ? max(maxiters, 6) : min(maxiters, 6)
        for _ in 1:npic
            u = _unsym_disp_from_mixed(is_kin, known, x)
            rhsvk = _unsym_vk_rhs(mesh, u, maps, Dx, Dy)
            all(isfinite, rhsvk) || break
            x_new = try
                A \ (b0 .+ rhsvk)
            catch
                break
            end
            all(isfinite, x_new) || break
            x .= e_relax .* x_new .+ (1 - e_relax) .* x
            all(isfinite, x) || break
        end
        x_pic = copy(x)
        if nonlinear === :newton && all(isfinite, x)
            R0 = z -> (A * z .- (b0 .+ _unsym_vk_rhs(mesh,
                _unsym_disp_from_mixed(is_kin, known, z), maps, Dx, Dy)))
            dx = try
                J = ForwardDiff.jacobian(R0, x)
                J \ R0(x)
            catch
                zeros(ndof)
            end
            if all(isfinite, dx)
                xN = x .- dx
                ok = all(isfinite, xN) && norm(xN) < 20 * (norm(x_pic) + 1)
                x .= ok ? xN : x_pic
            end
        end
        any(!isfinite, x) && (x .= x_pic)
        wtry = _unsym_disp_from_mixed(is_kin, known, x)
        wlin = _unsym_disp_from_mixed(is_kin, known, x_lin)
        if !all(isfinite, x) || abs(wtry[iw]) > 8 * (abs(wlin[iw]) + eps())
            x .= (step == 1 ? x_lin : x_prev)
        end
        u, _ = _unsym_write_mixed!(mesh, is_kin, known, x)
        x_prev .= x
        push!(λs, λ)
        push!(wcs, Float64(u[iw]))
    end
    return (λ=λs, w_center=wcs, u=mesh.u)
end

function solve_unsym_fsdt_large!(dad::BEMdata{<:UnsymFSDTProps}; kwargs...)
    m = FSDTMesh(dad)
    res = solve_unsym_fsdt_large!(m; kwargs...)
    _sync_fsdt_mesh!(dad, m)
    return res
end


