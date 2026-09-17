# FSDT plate BEM: isotropic Vander Weeën (MATLAB `kernelsHGC.m`),
# symmetric-laminate Wang (MATLAB `KernelP.m` / Useche 8.2), and
# unsymmetric Hsu–Hwu 5×5 (Useche 8.3.1).
#
# 3 DOF per collocation: (ψx, ψy, w). Tractions (Mx n, My n, Qn).
# No Kirchhoff corner forces. Domain load/inertia: DIBEM with Maxima RIM
# primitives of U* (`scripts/plates/fsdt_radial_maxima.mac`). DRM (`uqchp`)
# remains as `dibem_fsdt!(; method=:drm)` for the MATLAB twin.
#
# Cracked plates: Portela Dual BEM (`build_rect_fsdt_crack`,
# `assemble_fsdt_dual!`). Reissner (`FSDTProps`) uses Vander Weeën HBIE;
# unsymmetric Hsu–Hwu (`UnsymFSDTProps`) uses EABE 156 `T*` complete
# solutions on face B. Hui–Zehnder XBEM (`solve_fsdt_xbem!`) stays
# isotropic. `assemble_fsdt!` dispatches Dual when the mesh has twins.
# Unsymmetric FSDT also has a single traction BIE on the whole boundary
# (`assemble_unsym_fsdt_hbie!`); interiors are Somigliana after solve.

using SpecialFunctions: expint, expinti, expintx

export FSDTProps, LaminateFSDTProps, FSDTMesh, AbstractFSDTProps, AbstractFSDT
export reissner_lambda, shear_stiffness, build_square_fsdt, build_rect_fsdt, build_circle_fsdt
export assemble_fsdt!, apply_bc_fsdt, solve_fsdt!
export dibem_fsdt!, solve_fsdt_houbolt!, solve_unsym_fsdt_large!
export fsdt_w, fsdt_w_int, fsdt_ψx, fsdt_ψy, navier_w_ss_fsdt
export fsdt_resultants, navier_ss_fsdt_MQ
export wang_kernels, laminate_fsdt_props, fsdt_kernels, fsdt_Fρ
export UnsymFSDTProps, laminate_unsym_props, unsym_fsdt_kernels, unsym_hbie_kernels
export navier_w_ss_unsym, assemble_unsym_fsdt!, assemble_unsym_fsdt_hbie!
export assemble_unsym_fsdt_dual!
export unsym_interior_u, unsym_interior_t, unsym_interior_t_fd
export build_rect_fsdt_crack, assemble_fsdt_dual!, sif_ctod_fsdt
export reissner_hbie_kernels, pin_fsdt_rbm!
export hui_zehnder_M_local, hui_zehnder_global
export assemble_fsdt_xbem!, solve_fsdt_xbem!

# =============================================================================
# Properties & mesh
# =============================================================================

"""Abstract FSDT / Reissner properties (isotropic or laminate)."""
const AbstractFSDTProps = AbstractFSDT

"""Isotropic FSDT / Reissner plate — alias of [`BEM.FSDT`](@ref)."""
const FSDTProps = FSDT

"""Symmetric-laminate FSDT (Wang kernels, MATLAB `KernelP`).

`D` is 3×3 bending Voigt ``(11,22,12)``. `AT` is ContsLam shear
``[A_{44} A_{45}; A_{45} A_{55}]`` (not the ``[A_{55} A_{45}; A_{45} A_{44}]``
order used in some ABD helpers).
"""
mutable struct LaminateFSDTProps <: AbstractFSDT
    D::SMatrix{3,3,Float64,9}
    AT::SMatrix{2,2,Float64,4}
    h::Float64
    ρ::Float64
    q_c::Float64
    nθ::Int
    map::Symbol
    sinh_b::Float64
end

function LaminateFSDTProps(D, AT; h=0.01, ρ=1.0, q_c=0.0, nθ=10,
        map::Symbol=:telles, sinh_b=1e-3)
    return LaminateFSDTProps(SMatrix{3,3,Float64}(D), SMatrix{2,2,Float64}(AT),
        Float64(h), Float64(ρ), Float64(q_c), Int(nθ), map, Float64(sinh_b))
end

"""Isotropic `FSDTProps` as Wang `D`, `AT` (κ=5/6 Reissner shear)."""
function LaminateFSDTProps(p::FSDTProps; nθ=10)
    Dv = bending_stiffness(p)
    ν = p.ν
    As = shear_stiffness(p)
    D = @SMatrix [Dv ν * Dv 0; ν * Dv Dv 0; 0 0 (1 - ν) * Dv / 2]
    AT = @SMatrix [As 0; 0 As]
    return LaminateFSDTProps(D, AT; h=p.h, ρ=p.ρ, q_c=p.q_c, nθ=nθ)
end

import .ThinPlate: bending_stiffness
bending_stiffness(p::FSDTProps) = p.E * p.h^3 / (12 * (1 - p.ν^2))
bending_stiffness(p::LaminateFSDTProps) = p.D[1, 1]
reissner_lambda(p::FSDTProps) = sqrt(10) / p.h
shear_stiffness(p::FSDTProps) = (5 / 6) * p.E * p.h / (2 * (1 + p.ν))
shear_stiffness(p::LaminateFSDTProps) = p.AT[2, 2]   # A55

function _ply_Qbar(E1, E2, ν12, G12, θ)
    ν21 = ν12 * E2 / E1
    den = 1 - ν12 * ν21
    Q11, Q22, Q12, Q66 = E1 / den, E2 / den, ν12 * E2 / den, G12
    m, n = cos(θ), sin(θ)
    m2, n2, m4, n4 = m^2, n^2, m^4, n^4
    Q11b = Q11 * m4 + Q22 * n4 + 2 * (Q12 + 2Q66) * m2 * n2
    Q22b = Q11 * n4 + Q22 * m4 + 2 * (Q12 + 2Q66) * m2 * n2
    Q12b = (Q11 + Q22 - 4Q66) * m2 * n2 + Q12 * (m4 + n4)
    Q66b = (Q11 + Q22 - 2Q12 - 2Q66) * m2 * n2 + Q66 * (m4 + n4)
    Q16b = (Q11 - Q12 - 2Q66) * m^3 * n + (Q12 - Q22 + 2Q66) * m * n^3
    Q26b = (Q11 - Q12 - 2Q66) * m * n^3 + (Q12 - Q22 + 2Q66) * m^3 * n
    return @SMatrix [Q11b Q12b Q16b; Q12b Q22b Q26b; Q16b Q26b Q66b]
end

"""
    laminate_fsdt_props(plies; Ks=5/6, G13=nothing, G23=nothing, ρ=1, q_c=0, nθ=10)

ABD bending `D` and ContsLam shear `AT` from plies
`(E1, E2, ν12, G12, θ_deg, t)`. Midplane at ``z=0``.
"""
function _laminate_ABD_AT(plies; Ks=5 / 6, G13=nothing, G23=nothing)
    h = sum(p[6] for p in plies)
    z = -h / 2
    A = zeros(3, 3)
    B = zeros(3, 3)
    D = zeros(3, 3)
    A44 = A45 = A55 = 0.0
    for p in plies
        E1, E2, ν12, G12, θdeg, t = p
        θ = deg2rad(θdeg)
        Qb = _ply_Qbar(E1, E2, ν12, G12, θ)
        zb, zt = z, z + t
        A .+= Qb .* (zt - zb)
        B .+= Qb .* (zt^2 - zb^2) / 2
        D .+= Qb .* (zt^3 - zb^3) / 3
        g13 = G13 === nothing ? G12 : G13
        g23 = G23 === nothing ? G12 : G23
        m, n = cos(θ), sin(θ)
        Q55 = g13 * m^2 + g23 * n^2
        Q44 = g13 * n^2 + g23 * m^2
        Q45 = (g13 - g23) * m * n
        A44 += Ks * (zt - zb) * Q44
        A45 += Ks * (zt - zb) * Q45
        A55 += Ks * (zt - zb) * Q55
        z = zt
    end
    AT = @SMatrix [A44 A45; A45 A55]
    return SMatrix{3,3,Float64}(A), SMatrix{3,3,Float64}(B), SMatrix{3,3,Float64}(D), AT, h
end

function laminate_fsdt_props(plies; Ks=5 / 6, G13=nothing, G23=nothing,
        ρ=1.0, q_c=0.0, nθ=10)
    _, _, D, AT, h = _laminate_ABD_AT(plies; Ks=Ks, G13=G13, G23=G23)
    return LaminateFSDTProps(D, AT; h=h, ρ=ρ, q_c=q_c, nθ=nθ)
end

"""FSDT mesh. Elements are shared [`Element`](@ref).

BCs length `3n`: pairs `(ψx, Mxn)`, `(ψy, Myn)`, `(w, Vn)` per node.
`0` = kinematic known, `1` = traction known.
Interior collocation has `ndofn` DOF, no traction columns.
3-DOF: `(ψx, ψy, w)`. 5-DOF unsymmetric: `(u, v, βx, βy, w)`.
"""
mutable struct FSDTMesh
    elements::Vector{Element}
    element_type::Any
    elem_weight::Vector{Float64}
    nodes::Vector{Point2D}
    Normal::Vector{Point2D}
    BC::Vector{Int}
    BV::Vector{Float64}
    internal::Vector{Point2D}
    props::AbstractFSDTProps
    ndofn::Int
    H::Matrix{Float64}
    G::Matrix{Float64}
    q::Vector{Float64}
    M::Matrix{Float64}
    Mx::Matrix{Float64}    # IBP dM_x: RIM of -∂U*_w/∂X_x
    My::Matrix{Float64}
    u::Vector{Float64}
    t::Vector{Float64}
    eq_type::Vector{Int}   # 1 outer CBIE, 2 crack CBIE, 3 crack HBIE
    twin::Vector{Int}      # coincident crack-face partner (0 if none)
end

_ndofn(::AbstractFSDTProps) = 3

function FSDTMesh(elems, element_type, elem_weight, nodes, Normal, BC, BV,
        props; internal=Point2D[], eq_type=Int[], twin=Int[])
    n = length(nodes)
    ni = length(internal)
    ndofn = _ndofn(props)
    ndof = ndofn * (n + ni)
    nb = ndofn * n
    eq = isempty(eq_type) ? ones(Int, n) : collect(Int, eq_type)
    tw = isempty(twin) ? zeros(Int, n) : collect(Int, twin)
    nt = n + ni
    return FSDTMesh(elems, element_type, collect(Float64, elem_weight),
        nodes, Normal, BC, BV, internal, props, ndofn,
        zeros(ndof, ndof), zeros(ndof, nb), zeros(ndof),
        zeros(ndof, ndof), zeros(ndof, nt), zeros(ndof, nt),
        zeros(ndof), zeros(nb), eq, tw)
end

_n(m::FSDTMesh) = length(m.nodes)
_ni(m::FSDTMesh) = length(m.internal)
_ndofn(m::FSDTMesh) = m.ndofn
_ndof(m::FSDTMesh) = m.ndofn * (_n(m) + _ni(m))
_nb(m::FSDTMesh) = m.ndofn * _n(m)

fsdt_w(m::FSDTMesh, i::Int) = m.u[m.ndofn * i]
fsdt_w_int(m::FSDTMesh, k::Int=1) = m.u[m.ndofn * (_n(m) + k)]
fsdt_ψx(m::FSDTMesh, i::Int) = m.u[m.ndofn * i - 2]
fsdt_ψy(m::FSDTMesh, i::Int) = m.u[m.ndofn * i - 1]

fsdt_w(dad::BEMdata{<:AbstractFSDT}, i::Int) = dad.u[n_dof(dad) * i]
fsdt_w_int(dad::BEMdata{<:AbstractFSDT}, k::Int=1) =
    dad.u[n_dof(dad) * (dad.n + k)]
fsdt_ψx(dad::BEMdata{<:AbstractFSDT}, i::Int) = dad.u[n_dof(dad) * i - 2]
fsdt_ψy(dad::BEMdata{<:AbstractFSDT}, i::Int) = dad.u[n_dof(dad) * i - 1]

_n(dad::BEMdata) = dad.n
_ni(dad::BEMdata) = dad.ni
_ndofn(dad::BEMdata{<:AbstractFSDT}) = n_dof(dad)
_ndof(dad::BEMdata{<:AbstractFSDT}) = n_dof(dad) * dad.nt
_nb(dad::BEMdata{<:AbstractFSDT}) = n_dof(dad) * dad.n

"""Lift `BEMdata{<:AbstractFSDT}` to `FSDTMesh` (DIBEM / Houbolt / XBEM / shell)."""
function FSDTMesh(dad::BEMdata{<:AbstractFSDT})
    n = dad.n
    internals = dad.ni > 0 ? Point2D[dad.internalNodes;] : Point2D[]
    eq = has_cache(dad, :eq_type) ? collect(Int, dad.eq_type) : Int[]
    tw = has_cache(dad, :twin) ? collect(Int, dad.twin) : Int[]
    m = FSDTMesh(dad.elements, dad.element_type, collect(Float64, dad.elem_weight),
        Point2D[dad.Nodes;], Point2D[dad.Normal;], copy(dad.BC), copy(dad.BV),
        dad.properties; internal=internals, eq_type=eq, twin=tw)
    has_cache(dad, :H) && (m.H = dad.H)
    has_cache(dad, :G) && (m.G = dad.G)
    has_cache(dad, :M) && (m.M = dad.M)
    has_cache(dad, :Mx) && (m.Mx = dad.Mx)
    has_cache(dad, :My) && (m.My = dad.My)
    if has_cache(dad, :fsdt_q)
        m.q = dad.fsdt_q
    elseif has_cache(dad, :q) && dad.q isa AbstractVector
        m.q = dad.q
    end
    has_cache(dad, :u) && (m.u = dad.u)
    try
        m.t = dad.t
    catch
    end
    return m
end

function _sync_fsdt_mesh!(dad::BEMdata, m::FSDTMesh)
    set_cache!(dad; H=m.H, G=m.G)
    !isempty(m.M) && set_cache!(dad; M=m.M)
    !isempty(m.Mx) && set_cache!(dad; Mx=m.Mx, My=m.My)
    if !isempty(m.q)
        set_cache!(dad; fsdt_q=m.q, q=m.q)
    end
    !isempty(m.u) && set_cache!(dad; u=m.u, traction=m.t, T=m.u)
    !isempty(m.eq_type) && set_cache!(dad; eq_type=m.eq_type)
    !isempty(m.twin) && any(!iszero, m.twin) && set_cache!(dad; twin=m.twin)
    dad.BC .= m.BC
    dad.BV .= m.BV
    return dad
end

_plate_nodes(m::FSDTMesh) = m.nodes
_plate_nodes(d::BEMdata) = d.Nodes
_plate_internal(m::FSDTMesh) = m.internal
_plate_internal(d::BEMdata) = d.internalNodes
_plate_props(m::FSDTMesh) = m.props
_plate_props(d::BEMdata) = d.properties
_plate_q(m::FSDTMesh) = m.q
function _plate_q(d::BEMdata)
    has_cache(d, :fsdt_q) && return d.fsdt_q
    return d.q
end
_plate_G(m::FSDTMesh) = m.G
_plate_G(d::BEMdata) = has_cache(d, :G) ? d.G : zeros(0, 0)
_plate_normals(m::FSDTMesh) = m.Normal
_plate_normals(d::BEMdata) = d.Normal
_nt(d::BEMdata) = d.nt

elem_geom(el::Element, ξ) = ThinPlate.elem_geom(el, ξ)

# =============================================================================
# Vander Weeën kernels (MATLAB kernelsHGC.m + Bess.m)
# =============================================================================

"""Modified Bessel K0, K1 via the MATLAB polynomial (z≤2) / asymptotic (z>2)."""
function _bess_K01(z::Float64)
    z = abs(z)
    z < 1e-30 && return (30.0, 1e30)          # unused: caller skips R=0
    if z <= 2
        T = z / 3.75
        TT = T * T
        BESI = (((((0.0045813 * TT + 0.0360768) * TT + 0.2659732) * TT +
                  1.2067492) * TT + 3.0899424) * TT + 3.5156229) * TT + 1.0
        Z = z / 2
        ZZ = Z * Z
        Ko = (((((0.00000740 * ZZ + 0.00010750) * ZZ + 0.00262698) * ZZ +
                0.03488590) * ZZ + 0.23069756) * ZZ + 0.42278420) * ZZ -
             0.57721566 - log(Z) * BESI
        BESI1 = ((((((0.00032411 * TT + 0.00301532) * TT + 0.02658733) * TT +
                    0.15084934) * TT + 0.51498869) * TT + 0.87890594) * TT + 0.5) * z
        K1 = ((((((-0.00004686 * ZZ - 0.00110404) * ZZ - 0.01919402) * ZZ -
                  0.18156897) * ZZ - 0.67278579) * ZZ + 0.15443144) * ZZ + 1.0) / z +
             log(Z) * BESI1
        return Ko, K1
    else
        W = 2 / z
        XSQ = sqrt(z)
        EX0 = ((((((0.00053208 * W - 0.00251540) * W + 0.00587872) * W -
                  0.01062446) * W + 0.02189568) * W - 0.07832358) * W +
               1.25331414) / XSQ
        EX1 = ((((((-0.00068245 * W + 0.00325614) * W - 0.00780353) * W +
                  0.01504268) * W - 0.03655620) * W + 0.23498619) * W +
               1.25331414) / XSQ
        ez = exp(-z)
        return ez * EX0, ez * EX1
    end
end

function _reissner_AB(z::Float64)
    Ko, K1 = _bess_K01(z)
    Az = Ko + 2 / z * (K1 - 1 / z)
    Bz = Ko + 1 / z * (K1 - 1 / z)
    return Az, Bz, K1
end

"""
    fsdt_kernels(pg, pf, n, D, ν, λ) -> (U, P, C)

3×3 Reissner `U*` (G), `P*` (H), and regularized `C` for the free term
(MATLAB `FCij`).
"""
function fsdt_kernels(pg::Point2D, pf::Point2D, n::Point2D, D, ν, λ)
    RX = pg[1] - pf[1]
    RY = pg[2] - pf[2]
    R = hypot(RX, RY)
    R < 1e-30 && error("fsdt_kernels: coincident points")
    DRx = RX / R
    DRy = RY / R
    DRn = DRx * n[1] + DRy * n[2]
    z = λ * R
    Az, Bz, K1z = _reissner_AB(z)
    k2 = 8 * π * D
    om = 1 - ν

    U11 = 1 / (k2 * om) * (8 * Bz - om * (2 * log(z) - 1) - (8 * Az + 2 * om) * DRx^2)
    U12 = -1 / (k2 * om) * (8 * Az + 2 * om) * DRx * DRy
    U13 = 1 / k2 * (2 * log(z) - 1) * R * DRx
    U21 = U12
    U22 = 1 / (k2 * om) * (8 * Bz - om * (2 * log(z) - 1) - (8 * Az + 2 * om) * DRy^2)
    U23 = 1 / k2 * (2 * log(z) - 1) * R * DRy
    U31 = -U13
    U32 = -U23
    U33 = 1 / (k2 * om * λ^2) * (om * z^2 * (log(z) - 1) - 8 * log(z))

    AA = 4 * Az + 2 * z * K1z + om
    BB = 4 * Az + 1 + ν
    CC = 2 * (8 * Az + 2 * z * K1z + om)
    DD = λ^2 / (2 * π)
    GG = -om / (8 * π)
    FF = 2 * (1 + ν) / om
    k3 = -1 / (4 * π * R)
    P11 = k3 * (AA * (DRn + DRx * n[1]) + BB * DRx * n[1] - CC * DRx^2 * DRn)
    P12 = k3 * (AA * DRy * n[1] + BB * DRx * n[2] - CC * DRy * DRx * DRn)
    P21 = k3 * (AA * DRx * n[2] + BB * DRy * n[1] - CC * DRx * DRy * DRn)
    P22 = k3 * (AA * (DRn + DRy * n[2]) + BB * DRy * n[2] - CC * DRy^2 * DRn)
    P13 = DD * (Bz * n[1] - Az * DRx * DRn)
    P23 = DD * (Bz * n[2] - Az * DRy * DRn)
    P31 = GG * ((FF * log(z) - 1) * n[1] + 2 * DRx * DRn)
    P32 = GG * ((FF * log(z) - 1) * n[2] + 2 * DRy * DRn)
    P33 = -1 / (2 * π * R) * DRn

    U = @SMatrix [U11 U12 U13; U21 U22 U23; U31 U32 U33]
    P = @SMatrix [P11 P12 P13; P21 P22 P23; P31 P32 P33]
    C = @SMatrix [
        P11-RX*P13  P12-RY*P13  P13
        P21-RX*P23  P22-RY*P23  P23
        P31-RX*P33  P32-RY*P33  P33
    ]
    return U, P, C
end

fsdt_kernels(p::FSDTProps, pg, pf, n) =
    fsdt_kernels(pg, pf, n, bending_stiffness(p), p.ν, reissner_lambda(p))
fsdt_kernels(p::LaminateFSDTProps, pg, pf, n) =
    wang_kernels(pg, pf, n, p.D, p.AT; nθ=p.nθ, map=p.map, sinh_b=p.sinh_b)

"""`r = x - d`. Returns `(U, P)` 3×3 for vectorial `H_G_full_direct`."""
function fundamental(props::FSDTProps, r::SVector{2}, n::SVector{2})
    U, P, _ = fsdt_kernels(r, zero(r), n, bending_stiffness(props), props.ν,
        reissner_lambda(props))
    return U, P
end
function fundamental(props::LaminateFSDTProps, r::SVector{2}, n::SVector{2})
    U, P, _ = wang_kernels(r, zero(r), n, props.D, props.AT;
        nθ=props.nθ, map=props.map, sinh_b=props.sinh_b)
    return U, P
end
fundamental(dad::BEMdata{<:AbstractFSDT}, r::SVector{2}, n::SVector{2}) =
    fundamental(dad.properties, r, n)

# =============================================================================
# Wang laminate kernels (MATLAB KernelP.m)
# =============================================================================

"""`A0, A1` in Wang φ(ρ). `ρ>0` (θ₀ = atan2(-RX, RY) keeps ρ ≥ 0)."""
function _wang_A01(p, rho)
    z = p * rho
    if z < 40
        Ei = expinti(z)
        E1v = expint(z)
        ez, emz = exp(z), exp(-z)
        return ez * E1v - emz * Ei, ez * E1v + emz * Ei
    end
    sE1 = expintx(z)
    invz = 1 / z
    sEi, term = invz, invz
    for k in 1:25
        term *= k * invz
        sEi += term
        abs(term) < 1e-18 * abs(sEi) && break
    end
    return sE1 - sEi, sE1 + sEi
end

function _wang_tilde(w1, w2, rho, n1, n2,
        D11, D12, D16, D22, D26, D66, A44, A45, A55)
    w12, w22 = w1 * w1, w2 * w2
    w1w2 = w1 * w2
    w13, w23 = w12 * w1, w22 * w2
    w14, w24 = w12 * w12, w22 * w22
    w12w22 = w12 * w22
    w1w23, w13w2 = w1 * w23, w13 * w2

    d11 = D11 * w12 + 2 * D16 * w1w2 + D66 * w22
    d12 = D16 * w12 + (D12 + D66) * w1w2 + D26 * w22
    d13 = A45 * w2 + A55 * w1

    a11 = D66 * A55 * w14 + D22 * A44 * w24 +
          (D66 * A44 + D22 * A55 + 4 * D26 * A45) * w12w22 +
          2 * (D26 * A44 + A45 * D22) * w1w23 +
          2 * (D66 * A45 + A55 * D26) * w13w2
    a12 = -D16 * A55 * w14 - D26 * A44 * w24 -
          (D16 * A44 + D26 * A55 + 2 * A45 * (D12 + D66)) * w12w22 -
          (2 * D26 * A45 + A44 * (D12 + D66)) * w1w23 -
          (2 * D16 * A45 + A55 * (D12 + D66)) * w13w2
    a21 = a12
    a22 = D11 * A55 * w14 + D66 * A44 * w24 +
          (D11 * A44 + D66 * A55 + 4 * D16 * A45) * w12w22 +
          2 * (D16 * A44 + A45 * D66) * w1w23 +
          2 * (D11 * A45 + A55 * D16) * w13w2
    f1 = (D16 * A55 - D66 * A55) * w13 + (D26 * A44 - D22 * A45) * w23 +
         (D16 * A44 + D12 * A45 - 2 * D26 * A55) * w12 * w2 +
         ((D12 + D66) * A44 - D22 * A55 - D26 * A45) * w1 * w22
    f2 = (D16 * A55 - D11 * A45) * w13 + (D26 * A45 - D66 * A44) * w23 +
         (D12 * A45 + D26 * A55 - 2 * D16 * A44) * w1 * w22 +
         ((D12 + D66) * A55 - D11 * A44 - D16 * A45) * w12 * w2
    A = (D11 * D66 - D16^2) * w14 + (D22 * D66 - D26^2) * w24 +
        (D11 * D22 + 2 * D16 * D26 - D12^2 - 2 * D12 * D66) * w12w22 +
        2 * (D11 * D26 - D16 * D12) * w13w2 +
        2 * (D16 * D22 - D12 * D26) * w1w23
    B = (A44 * D11 + A55 * D66 - 2 * A45 * D16) * w12 +
        (A44 * D66 + A55 * D22 - 2 * A45 * D26) * w22 +
        2 * (D16 * A44 + A55 * D26 - A45 * (D12 + D26)) * w1w2
    Csh = A44 * A55 - A45^2

    a = a11 * d11 + a12 * d12
    b = a11 * A55 + Csh * d11 * w12 + a12 * A45 + Csh * d12 * w1w2 + f1 * d13
    (a <= 0 || b <= 0) && return nothing
    p = sqrt(b / a)
    Cf = 1 / (8 * π^2 * p^4 * a)
    A0, A1 = _wang_A01(p, rho)
    ar = abs(rho)
    lr = log(ar)
    p2, p3, p4 = p * p, p^3, p^4
    phi = Cf * (p2 * rho^2 * lr + 2 * lr + 3 + A0)
    d1 = Cf * (2 * p2 * rho * lr + p2 * rho + p * A1)
    d2 = Cf * (2 * p2 * lr + 3 * p2 + p2 * A0)
    d3 = Cf * p3 * A1
    d4 = Cf * p4 * A0
    d5 = Cf * (p^5 * A1 - 2 * p4 / rho)

    U11 = a11 * d4 - Csh * w12 * d2
    U12 = a12 * d4 - Csh * w1w2 * d2
    U22 = a22 * d4 - Csh * w22 * d2
    U13 = f1 * d3 + Csh * w1 * d1
    U23 = f2 * d3 + Csh * w2 * d1
    U33 = A * d4 - B * d2 + Csh * phi
    Utilde = @SMatrix [U11 U12 U13; U12 U22 U23; -U13 -U23 U33]

    k11 = a11 * w1 * d5 - Csh * w1^3 * d3
    k12 = a12 * w2 * d5 - Csh * w1 * w22 * d3
    k16 = (a11 * w2 + a12 * w1) * d5 - 2 * Csh * w2 * w12 * d3
    k21 = a21 * w1 * d5 - Csh * w12 * w2 * d3
    k22 = a22 * w2 * d5 - Csh * w2^3 * d3
    k26 = (a21 * w2 + a22 * w1) * d5 - 2 * Csh * w1 * w22 * d3
    k31 = -f1 * w1 * d4 - Csh * w12 * d2
    k32 = -f2 * w2 * d4 - Csh * w22 * d2
    k36 = -(f1 * w2 + f2 * w1) * d4 - 2 * Csh * w1w2 * d2
    e14 = (f1 * w2 + a12) * d4
    e15 = (f1 * w1 + a11) * d4
    e24 = (f2 * w2 + a22) * d4
    e25 = (f2 * w1 + a21) * d4
    e34 = A * w2 * d5 - (B * w2 + f2) * d3
    e35 = A * w1 * d5 - (B * w1 + f1) * d3
    kij = @SMatrix [k11 k12 k16; k21 k22 k26; k31 k32 k36]
    eij = @SMatrix [e15 e14; e25 e24; e35 e34]
    M = kij * @SMatrix [D11 D12 D16; D12 D22 D26; D16 D26 D66]
    Q = eij * @SMatrix [A55 A45; A45 A44]
    Ptilde = @SMatrix [
        M[1, 1]*n1+M[1, 3]*n2  M[1, 3]*n1+M[1, 2]*n2  Q[1, 1]*n1+Q[1, 2]*n2
        M[2, 1]*n1+M[2, 3]*n2  M[2, 3]*n1+M[2, 2]*n2  Q[2, 1]*n1+Q[2, 2]*n2
        M[3, 1]*n1+M[3, 3]*n2  M[3, 3]*n1+M[3, 2]*n2  Q[3, 1]*n1+Q[3, 2]*n2
    ]
    return Utilde, Ptilde
end

"""Cluster Gauss nodes on [-1,1] at `eet`. `map` ∈ `:gauss,:telles,:sinh,:sinhsinh,:power`."""
function _cluster_rule(qsi, w, eet, map::Symbol; b=1e-3, α=3)
    n = length(qsi)
    if map === :gauss
        return collect(Float64, qsi), collect(Float64, w)
    elseif map === :telles
        x = Vector{Float64}(undef, n)
        ww = Vector{Float64}(undef, n)
        @inbounds for i in 1:n
            ξ, Jt = _telles(qsi[i], eet)
            x[i] = ξ
            ww[i] = w[i] * Jt
        end
        return x, ww
    elseif map === :power
        # algebraic map clustering at the nearer endpoint
        x = Vector{Float64}(undef, n)
        ww = Vector{Float64}(undef, n)
        left = eet <= 0
        @inbounds for i in 1:n
            γ = qsi[i]
            if left
                t = (γ + 1) / 2
                x[i] = 2 * t^α - 1
                ww[i] = w[i] * α * t^(α - 1)
            else
                t = (1 - γ) / 2
                x[i] = 1 - 2 * t^α
                ww[i] = w[i] * α * t^(α - 1)
            end
        end
        return x, ww
    elseif map === :sinh || map === :sinhsinh
        a = clamp(float(eet), -1.0, 1.0)
        bb = max(float(b), 1e-14)
        niter = map === :sinhsinh ? 2 : 1
        return _sinh_cluster(qsi, w, a, bb, niter)
    else
        error("unknown FSDT map '$map' (use :gauss, :telles, :sinh, :sinhsinh, :power)")
    end
end

function _sinh_cluster(qsi, w, a, b, niter::Int)
    T = Float64
    a = T(a)
    b = max(T(b), T(1e-14))
    maps = NTuple{2,T}[]
    aa, bb = a, b
    for _ in 1:niter
        push!(maps, (aa, bb))
        μ = T(0.5) * (asinh((1 + aa) / bb) + asinh((1 - aa) / bb))
        η = T(0.5) * (asinh((1 + aa) / bb) - asinh((1 - aa) / bb))
        μ < T(1e-14) && break
        aa = clamp(η / μ, nextfloat(T(-1)), prevfloat(T(1)))
        bb = max(T(π) / (2 * μ), T(1e-14))
    end
    x = collect(T, qsi)
    J = ones(T, length(qsi))
    for (aa, bb) in Iterators.reverse(maps)
        μ = T(0.5) * (asinh((1 + aa) / bb) + asinh((1 - aa) / bb))
        η = T(0.5) * (asinh((1 + aa) / bb) - asinh((1 - aa) / bb))
        @inbounds for i in eachindex(x)
            s = μ * x[i] - η
            x[i] = aa + bb * sinh(s)
            J[i] *= bb * μ * cosh(s)
        end
    end
    return x, collect(T, w) .* J
end

"""
    wang_kernels(pg, pf, n, D, AT; nθ=10, map=:telles, sinh_b=1e-3) -> (U, P, C)

Wang symmetric-laminate FSDT `U*`, `P*`, regularized `C` (MATLAB `KernelP`).
`D` 3×3 Voigt bending; `AT` ContsLam `[A44 A45; A45 A55]`.
θ₀ = `atan2(-RX, RY)`, two π/2 quadrants, factor 2.
`map` clusters the log singularity at ρ=0: `:telles` (MATLAB), `:gauss`,
`:sinh`, `:sinhsinh`, `:power`.
"""
function wang_kernels(pg::Point2D, pf::Point2D, n::Point2D, D, AT; nθ::Int=10,
        map::Symbol=:telles, sinh_b::Float64=1e-3)
    RX = pg[1] - pf[1]
    RY = pg[2] - pf[2]
    hypot(RX, RY) < 1e-30 && error("wang_kernels: coincident points")
    D11, D12, D16 = D[1, 1], D[1, 2], D[1, 3]
    D22, D26, D66 = D[2, 2], D[2, 3], D[3, 3]
    A44, A45, A55 = AT[1, 1], AT[1, 2], AT[2, 2]
    n1, n2 = n[1], n[2]
    eg, wg0 = gausslegendre(nθ)
    Uast = zeros(3, 3)
    Past = zeros(3, 3)
    θ0 = atan(-RX, RY)
    for iq in 1:2
        eet = iq == 1 ? -1.0 : 1.0
        et, wg = _cluster_rule(eg, wg0, eet, map; b=sinh_b)
        for i in 1:nθ
            ξ = clamp(et[i], -1.0, 1.0)
            θ = θ0 + (ξ + 1) * π / 4
            w1, w2 = cos(θ), sin(θ)
            rho = w1 * RX + w2 * RY
            abs(rho) < 1e-14 && continue
            got = _wang_tilde(w1, w2, rho, n1, n2,
                D11, D12, D16, D22, D26, D66, A44, A45, A55)
            got === nothing && continue
            Ũ, P̃ = got
            wθ = 2 * wg[i] * (π / 4)
            Uast .+= Ũ .* wθ
            Past .+= P̃ .* wθ
        end
        θ0 += π / 2
    end
    U11, U12, U13 = Uast[1, 1], Uast[1, 2], Uast[1, 3]
    U21, U22, U23 = Uast[2, 1], Uast[2, 2], Uast[2, 3]
    U31, U32, U33 = Uast[3, 1], Uast[3, 2], Uast[3, 3]
    P11, P12, P13 = Past[1, 1], Past[1, 2], Past[1, 3]
    P21, P22, P23 = Past[2, 1], Past[2, 2], Past[2, 3]
    P31, P32, P33 = Past[3, 1], Past[3, 2], Past[3, 3]
    U = @SMatrix [U11 U12 U13; U21 U22 U23; U31 U32 U33]
    P = @SMatrix [P11 P12 P13; P21 P22 P23; P31 P32 P33]
    C = @SMatrix [
        P11-RX*P13  P12-RY*P13  P13
        P21-RX*P23  P22-RY*P23  P23
        P31-RX*P33  P32-RY*P33  P33
    ]
    return U, P, C
end

# =============================================================================
# Radial primitives  ∫_0^R U* ρ dρ   (Maxima `fsdt_radial_maxima.mac`)
# =============================================================================

"""`∫_0^R U*_ij(ρ ê) ρ dρ` for isotropic Reissner. `n1,n2` = ê."""
function _reissner_Fρ(R, n1, n2, D, ν, λ)
    R < 1e-14 && return @SMatrix zeros(3, 3)
    Z = λ * R
    Ko, K1z = _bess_K01(Z)
    γe = Base.MathConstants.eulergamma
    IAz = -Z * K1z - 2 * Ko - 2 * log(Z) + 1 + 2 * log(2) - 2 * γe
    IBz = -Z * K1z - Ko - log(Z) + 1 + log(2) - γe
    k2 = 8 * π * D
    om = 1 - ν
    Z2, Z3, Z4 = Z * Z, Z^3, Z^4
    lz = log(Z)
    inv = 1 / (k2 * om * λ^2)
    F11 = inv * (8 * IBz - 8 * n1^2 * IAz - om * Z2 * lz + om * (1 - n1^2) * Z2)
    F22 = inv * (8 * IBz - 8 * n2^2 * IAz - om * Z2 * lz + om * (1 - n2^2) * Z2)
    F12 = -n1 * n2 * inv * (8 * IAz + om * Z2)
    F13 = n1 / (k2 * λ^3) * ((2 / 3) * Z3 * lz - (5 / 9) * Z3)
    F23 = n2 / (k2 * λ^3) * ((2 / 3) * Z3 * lz - (5 / 9) * Z3)
    F33 = 1 / (k2 * om * λ^4) * (om * (Z4 / 4 * lz - 5 * Z4 / 16) - 4 * Z2 * lz + 2 * Z2)
    return @SMatrix [F11 F12 F13; F12 F22 F23; -F13 -F23 F33]
end

"""Wang `G0,G1` groups (Maxima). `s = p ρ > 0`."""
function _wang_G01(s)
    if s < 40
        E1v, Ei = expint(s), expinti(s)
        es, ems = exp(s), exp(-s)
        G0 = (s - 1) * es * E1v + (s + 1) * ems * Ei - 2 * log(s)
        G1 = (s - 1) * es * E1v - (s + 1) * ems * Ei + 2 * s
        return G0, G1
    end
    sE1 = expintx(s)
    invs = 1 / s
    sEi, term = invs, invs
    for k in 1:25
        term *= k * invs
        sEi += term
        abs(term) < 1e-18 * abs(sEi) && break
    end
    G0 = (s - 1) * sE1 + (s + 1) * sEi - 2 * log(s)
    G1 = (s - 1) * sE1 - (s + 1) * sEi + 2 * s
    return G0, G1
end

"""`∫_0^ρ (φ, φ', …, φ'''') σ dσ`, Cf included. Maxima `I_n - I_n(0+)`."""
function _wang_In(p, ρ, Cf)
    s = p * ρ
    G0, G1 = _wang_G01(s)
    γe = Base.MathConstants.eulergamma
    lr = log(ρ)
    ρ2, ρ3, ρ4 = ρ * ρ, ρ^3, ρ^4
    p2 = p * p
    I0 = Cf * (p2 * ρ4 / 4 * lr + ρ2 * lr + ρ2 - p2 * ρ4 / 16 + G0 / p2 - 2 * γe / p2)
    I1 = Cf * (2 * p2 * (ρ3 / 3 * lr - ρ3 / 9) + p2 * ρ3 / 3 + G1 / p)
    I2 = Cf * (p2 * ρ2 * lr + p2 * ρ2 + G0 - 2 * γe)
    I3 = Cf * p * G1
    I4 = Cf * p2 * (G0 - 2 * γe)
    return I0, I1, I2, I3, I4
end

function _wang_U_d(a11, a12, a22, f1, f2, A, B, Csh, w1, w2, d0, d1, d2, d3, d4)
    U11 = a11 * d4 - Csh * w1^2 * d2
    U12 = a12 * d4 - Csh * w1 * w2 * d2
    U22 = a22 * d4 - Csh * w2^2 * d2
    U13 = f1 * d3 + Csh * w1 * d1
    U23 = f2 * d3 + Csh * w2 * d1
    U33 = A * d4 - B * d2 + Csh * d0
    return @SMatrix [U11 U12 U13; U12 U22 U23; -U13 -U23 U33]
end

"""Wang `∫_0^R U* ρ dρ` by θ-quadrature of the Maxima primitive (÷ μ²)."""
function _wang_Fρ(pg::Point2D, pf::Point2D, D, AT; nθ::Int=10,
        map::Symbol=:telles, sinh_b::Float64=1e-3)
    RX = pg[1] - pf[1]
    RY = pg[2] - pf[2]
    R = hypot(RX, RY)
    R < 1e-30 && return @SMatrix zeros(3, 3)
    DRx, DRy = RX / R, RY / R
    D11, D12, D16 = D[1, 1], D[1, 2], D[1, 3]
    D22, D26, D66 = D[2, 2], D[2, 3], D[3, 3]
    A44, A45, A55 = AT[1, 1], AT[1, 2], AT[2, 2]
    eg, wg0 = gausslegendre(nθ)
    Fast = zeros(3, 3)
    θ0 = atan(-RX, RY)
    for iq in 1:2
        eet = iq == 1 ? -1.0 : 1.0
        et, wg = _cluster_rule(eg, wg0, eet, map; b=sinh_b)
        for i in 1:nθ
            ξ = clamp(et[i], -1.0, 1.0)
            θ = θ0 + (ξ + 1) * π / 4
            w1, w2 = cos(θ), sin(θ)
            μ = w1 * DRx + w2 * DRy
            rho = μ * R
            rho < 1e-14 && continue
            w12, w22 = w1 * w1, w2 * w2
            w1w2 = w1 * w2
            d11 = D11 * w12 + 2 * D16 * w1w2 + D66 * w22
            d12 = D16 * w12 + (D12 + D66) * w1w2 + D26 * w22
            d13 = A45 * w2 + A55 * w1
            w13, w23 = w12 * w1, w22 * w2
            w14, w24 = w12 * w12, w22 * w22
            w12w22 = w12 * w22
            w1w23, w13w2 = w1 * w23, w13 * w2
            a11 = D66 * A55 * w14 + D22 * A44 * w24 +
                  (D66 * A44 + D22 * A55 + 4 * D26 * A45) * w12w22 +
                  2 * (D26 * A44 + A45 * D22) * w1w23 +
                  2 * (D66 * A45 + A55 * D26) * w13w2
            a12 = -D16 * A55 * w14 - D26 * A44 * w24 -
                  (D16 * A44 + D26 * A55 + 2 * A45 * (D12 + D66)) * w12w22 -
                  (2 * D26 * A45 + A44 * (D12 + D66)) * w1w23 -
                  (2 * D16 * A45 + A55 * (D12 + D66)) * w13w2
            a22 = D11 * A55 * w14 + D66 * A44 * w24 +
                  (D11 * A44 + D66 * A55 + 4 * D16 * A45) * w12w22 +
                  2 * (D16 * A44 + A45 * D66) * w1w23 +
                  2 * (D11 * A45 + A55 * D16) * w13w2
            f1 = (D16 * A55 - D66 * A55) * w13 + (D26 * A44 - D22 * A45) * w23 +
                 (D16 * A44 + D12 * A45 - 2 * D26 * A55) * w12 * w2 +
                 ((D12 + D66) * A44 - D22 * A55 - D26 * A45) * w1 * w22
            f2 = (D16 * A55 - D11 * A45) * w13 + (D26 * A45 - D66 * A44) * w23 +
                 (D12 * A45 + D26 * A55 - 2 * D16 * A44) * w1 * w22 +
                 ((D12 + D66) * A55 - D11 * A44 - D16 * A45) * w12 * w2
            AA = (D11 * D66 - D16^2) * w14 + (D22 * D66 - D26^2) * w24 +
                 (D11 * D22 + 2 * D16 * D26 - D12^2 - 2 * D12 * D66) * w12w22 +
                 2 * (D11 * D26 - D16 * D12) * w13w2 +
                 2 * (D16 * D22 - D12 * D26) * w1w23
            BB = (A44 * D11 + A55 * D66 - 2 * A45 * D16) * w12 +
                 (A44 * D66 + A55 * D22 - 2 * A45 * D26) * w22 +
                 2 * (D16 * A44 + A55 * D26 - A45 * (D12 + D26)) * w1w2
            Csh = A44 * A55 - A45^2
            a = a11 * d11 + a12 * d12
            b = a11 * A55 + Csh * d11 * w12 + a12 * A45 + Csh * d12 * w1w2 + f1 * d13
            (a <= 0 || b <= 0) && continue
            p = sqrt(b / a)
            Cf = 1 / (8 * π^2 * p^4 * a)
            I0, I1, I2, I3, I4 = _wang_In(p, rho, Cf)
            sc = (R / rho)^2
            F̃ = _wang_U_d(a11, a12, a22, f1, f2, AA, BB, Csh, w1, w2,
                I0, I1, I2, I3, I4) .* sc
            Fast .+= F̃ .* (2 * wg[i] * (π / 4))
        end
        θ0 += π / 2
    end
    return SMatrix{3,3,Float64,9}(Fast)
end

"""`∫_0^R U*(ρ ê) ρ dρ` (3×3). Dispatch on FSDT props."""
fsdt_Fρ(p::FSDTProps, pg, pf, n=Point2D(1.0, 0.0)) = begin
    RX, RY = pg[1] - pf[1], pg[2] - pf[2]
    R = hypot(RX, RY)
    R < 1e-30 && return @SMatrix zeros(3, 3)
    _reissner_Fρ(R, RX / R, RY / R, bending_stiffness(p), p.ν, reissner_lambda(p))
end
fsdt_Fρ(p::LaminateFSDTProps, pg, pf, n=Point2D(1.0, 0.0)) =
    _wang_Fρ(pg, pf, p.D, p.AT; nθ=p.nθ, map=p.map, sinh_b=p.sinh_b)

"""Telles cubic map (MATLAB `telles.m`). `eet` = image of the source in ξ."""
function _telles(γ, eet)
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

# =============================================================================
# Assembly
# =============================================================================

function _add_el!(He, Ge, Cije, el, poly, pf, props, qs, ws; telles=false, eet=0.0)
    nN = length(el.index)
    for (ig, ξ0) in enumerate(qs)
        ξ, Jt = telles ? _telles(ξ0, eet) : (ξ0, 1.0)
        abs(ξ) > 1 + 1e-12 && continue
        pg, J, n̂ = elem_geom(el, ξ)
        R = norm(pg - pf)
        R < 1e-14 && continue
        U, P, C = fsdt_kernels(props, pg, pf, n̂)
        Nf, _ = shapefun(poly, ξ)
        wJ = J * ws[ig] * Jt
        @inbounds for a in 1:nN
            Na = Nf[1, a] * wJ
            cols = 3a-2:3a
            He[:, cols] .+= P .* Na
            Ge[:, cols] .+= U .* Na
        end
        Cije .+= C .* wJ
    end
    return nothing
end

"""
    assemble!(dad::BEMdata{<:AbstractFSDT}; npg=10, …)

FSDT `H, G` via [`H_G_full_direct`](@ref) (Gauss collocation, far lumping,
sinh near-field, Guiggiani). 3-DOF Reissner/Wang or 5-DOF Hsu–Hwu.
Dual-BEM (any `eq_type==3`) uses [`assemble_fsdt_dual!`](@ref) /
[`assemble_unsym_fsdt_dual!`](@ref). `bie=:hbie` is the unsymmetric
single traction BIE. `assemble_fsdt!(dad)` is the same path.
"""
function assemble!(dad::BEMdata{<:AbstractFSDT}; method::Symbol=:dense,
        npg=10, nsub::Int=8, threaded::Bool=true, near_factor::Real=1.5,
        singular::Symbol=:guiggiani, bie::Symbol=:cbie, ninterp::Int=20,
        kwargs...)
    haskey(kwargs, :ninterp) && (ninterp = Int(kwargs[:ninterp]))
    set_cache!(dad; ninterp=ninterp)
    if has_cache(dad, :eq_type) && any(==(3), dad.eq_type)
        return n_dof(dad) == 5 ?
            assemble_unsym_fsdt_dual!(dad; npg=npg, nsub=nsub,
                singular=singular, ninterp=ninterp) :
            assemble_fsdt_dual!(dad; npg=npg, nsub=nsub, threaded=threaded)
    end
    if n_dof(dad) == 5
        # Hsu–Hwu 5×5 kernels: keep Telles/Guiggiani assembly (far lumping
        # of the θ-integral is too coarse on the n_el=2 Navier meshes).
        m = FSDTMesh(dad)
        assemble_unsym_fsdt!(m; npg=npg, nsub=nsub, singular=singular, bie=bie,
            ninterp=ninterp)
        _sync_fsdt_mesh!(dad, m)
        return dad
    end
    method === :dense ||
        throw(ArgumentError("FSDT assemble! method must be :dense; got $method"))
    (singular === :auto || singular === :analytic) && (singular = :guiggiani)
    return H_G_full_direct(dad; npg=npg, threaded=threaded, near_factor=near_factor,
        singular=singular)
end

function assemble_fsdt!(dad::BEMdata{<:AbstractFSDT}; npg=10, kwargs...)
    assemble!(dad; npg=npg, kwargs...)
    return dad
end

function _vectorial_free_term!(H, dad::BEMdata{<:AbstractFSDT})
    dim = n_dof(dad)
    @inbounds for i in 1:dad.n
        ii = expand(i, dim)
        for d in 1:dim
            H[ii[d], ii[d]] += 0.5
        end
    end
    return H
end

function _after_vectorial_assemble!(dad::BEMdata{<:AbstractFSDT}, H, G)
    dim = n_dof(dad)
    @inbounds for k in 1:dad.ni
        i = dad.n + k
        ii = expand(i, dim)
        for d in 1:dim
            H[ii[d], ii[d]] += 1.0
        end
    end
    return nothing
end

function applyBC(dad::BEMdata{<:AbstractFSDT}; kwargs...)
    H = dad.H
    G = dad.G
    ndof = size(H, 1)
    A = copy(H)
    b = zeros(ndof)
    has_cache(dad, :fsdt_q) && (b .+= dad.fsdt_q)
    nb = n_dof(dad) * dad.n
    @inbounds for dof in 1:nb
        val = dad.BV[dof]
        if dad.BC[dof] == 0
            for i in 1:ndof
                b[i] -= A[i, dof] * val
                A[i, dof] = -G[i, dof]
            end
        else
            for i in 1:ndof
                b[i] += G[i, dof] * val
            end
        end
    end
    set_cache!(dad; A, b)
    return nothing
end

function solve(dad::BEMdata{<:AbstractFSDT}; kwargs...)
    qc = hasproperty(dad.properties, :q_c) ? dad.properties.q_c : 0.0
    if !has_cache(dad, :fsdt_q) && qc != 0
        dibem_fsdt!(dad)
    end
    applyBC(dad)
    A, b = dad.A, dad.b
    x = A \ b
    ndof = length(x)
    u = zeros(ndof)
    t = zeros(n_dof(dad) * dad.n)
    nb = length(t)
    @inbounds for dof in 1:nb
        if dad.BC[dof] == 0
            u[dof] = dad.BV[dof]
            t[dof] = x[dof]
        else
            t[dof] = dad.BV[dof]
            u[dof] = x[dof]
        end
    end
    @inbounds for dof in (nb + 1):ndof
        u[dof] = x[dof]
    end
    set_cache!(dad; u=u, traction=t, T=u)
    if n_dof(dad) == 5 && has_cache(dad, :eq_type) && all(==(3), dad.eq_type) &&
            dad.ni > 0
        m = FSDTMesh(dad)
        _unsym_somigliana_interior!(m)
        set_cache!(dad; u=m.u, traction=m.t, T=m.u)
    end
    return dad.u
end

function solve_fsdt!(dad::BEMdata{<:AbstractFSDT}; large::Bool=false, kwargs...)
    large && return solve_unsym_fsdt_large!(dad; kwargs...)
    has_cache(dad, :H) || assemble_fsdt!(dad)
    solve(dad)
    return dad.u, dad.t
end

function dibem_fsdt!(dad::BEMdata{<:AbstractFSDT}; kwargs...)
    m = FSDTMesh(dad)
    dibem_fsdt!(m; kwargs...)
    _sync_fsdt_mesh!(dad, m)
    return dad
end

function solve_fsdt_houbolt!(dad::BEMdata{<:AbstractFSDT}; kwargs...)
    m = FSDTMesh(dad)
    res = solve_fsdt_houbolt!(m; kwargs...)
    _sync_fsdt_mesh!(dad, m)
    return res
end

function fsdt_resultants(dad::BEMdata{<:AbstractFSDT}; kwargs...)
    m = FSDTMesh(dad)
    return fsdt_resultants(m; kwargs...)
end

"""
    assemble_fsdt!(mesh; npg=10, nsub=8, map=:telles, sinh_b=1e-3)

Build `H`, `G`. On/near-element: clustered Gauss + `nsub` subdivisions
(MATLAB `matricesHeGe` uses Telles). `map` is `:telles`, `:gauss`,
`:sinh`, or `:sinhsinh`. Free term from regularized `C` (MATLAB `HeS`).
Unsymmetric (`ndofn=5`): `bie=:cbie` (default) or `:hbie`;
`singular=:telles` (default) or `:guiggiani` on the self-element.
`BEMdata{<:AbstractFSDT}` uses [`H_G_full_direct`](@ref).
"""
function assemble_fsdt!(mesh::FSDTMesh; npg::Int=10, nsub::Int=8,
        map::Symbol=:telles, sinh_b::Float64=1e-3,
        singular::Symbol=:telles, bie::Symbol=:cbie, ninterp::Int=16,
        scale_hbie::Bool=true)
    if mesh.ndofn == 5
        any(!iszero, mesh.twin) &&
            return assemble_unsym_fsdt_dual!(mesh; npg=npg, nsub=nsub,
                singular=singular, ninterp=ninterp, scale_hbie=scale_hbie)
        return assemble_unsym_fsdt!(mesh; npg=npg, nsub=nsub,
            map=map, sinh_b=sinh_b, singular=singular, bie=bie, ninterp=ninterp,
            scale_hbie=scale_hbie)
    end
    any(==(3), mesh.eq_type) && return assemble_fsdt_dual!(mesh; npg=npg, nsub=nsub)
    n = _n(mesh)
    ni = _ni(mesh)
    ndof = 3n + 3ni
    nb = 3n
    H = zeros(ndof, ndof)
    G = zeros(ndof, nb)
    props = mesh.props
    qsi, w = gausslegendre(npg)
    poly = mesh.element_type

    pts = Point2D[mesh.nodes; mesh.internal]
    nsrc = n + ni
    @showprogress "FSDT H,G" for i in 1:nsrc
        pf = pts[i]
        rows = 3i-2:3i
        Csum = zeros(3, 3)
        for el in mesh.elements
            nN = length(el.index)
            He = zeros(3, 3nN)
            Ge = zeros(3, 3nN)
            Cije = zeros(3, 3)
            x1, x3 = el.geo[1], el.geo[end]
            Le = norm(x3 - x1)
            Rmin = minimum(norm(mesh.nodes[k] - pf) for k in el.index)
            near = (i <= n && i in el.index) || Rmin <= Le / 4
            if near
                ndiv = max(nsub, 4)
                dξ = 2 / ndiv
                for k in 1:ndiv
                    ξa = -1 + (k - 1) * dξ
                    ξb = -1 + k * dξ
                    # map Gauss on subelement [ξa,ξb] via Telles in the parent
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
                        U, P, C = fsdt_kernels(props, pg, pf, n̂)
                        Nf, _ = shapefun(poly, ξ)
                        wJ = J * wt[ig] * Jsub
                        @inbounds for a in 1:nN
                            Na = Nf[1, a] * wJ
                            cols = 3a-2:3a
                            He[:, cols] .+= P .* Na
                            Ge[:, cols] .+= U .* Na
                        end
                        Cije .+= C .* wJ
                    end
                end
            else
                _add_el!(He, Ge, Cije, el, poly, pf, props, qsi, w)
            end
            for a in 1:nN
                ja = el.index[a]
                cols = 3ja-2:3ja
                H[rows, cols] .+= He[:, 3a-2:3a]
                G[rows, cols] .+= Ge[:, 3a-2:3a]
            end
            Csum .+= Cije
        end
        if i <= n
            H[rows, rows] .-= Csum
        else
            H[rows, rows] .+= I(3)
        end
    end
    mesh.H = H
    mesh.G = G
    return mesh
end

# =============================================================================
# DRM / DIBEM particular solutions (MATLAB uqchp.m)
# =============================================================================

"""`û, q̂` 3×3 particular solutions at field `x` for a unit body force at `xm`."""
function _uqchp(xm, ym, x, y, nc, D, ν, λ)
    rx = x - xm
    ry = y - ym
    r = hypot(rx, ry)
    if r < 1e-14
        drx = -nc[2]
        dry = nc[1]
        r = 0.0
    else
        drx = rx / r
        dry = ry / r
    end
    a0 = -2 / (9 * D * (1 - ν))
    a1 = (1 + ν) / (45 * D * (1 - ν))
    a2 = λ^2 / (1575 * D)
    b0 = (1 + ν) / (15 * D * (1 - ν))
    b1 = λ^2 / (315 * D)
    c0 = -1 / (225 * D)

    w11 = a0 * r^3 + a1 * r * (4 * rx^2 + ry^2) + a2 * r^3 * (6 * rx^2 + ry^2)
    w12 = b0 * rx * ry * r + b1 * rx * ry * r^3
    w13 = -c0 * rx * r^3 * (5 - λ^2 * r^2 / 7)
    w111 = 3 * a0 * r^2 * drx + a1 * (4 * rx^2 + ry^2) * drx + 8 * a1 * rx * r +
           3 * a2 * r^2 * drx * (6 * rx^2 + ry^2) + 12 * a2 * r^3 * rx
    w112 = 3 * a0 * r^2 * dry + a1 * (4 * rx^2 + ry^2) * dry + 2 * a1 * ry * r +
           3 * a2 * r^2 * dry * (6 * rx^2 + ry^2) + 2 * a2 * r^3 * ry
    w121 = b0 * r * ry + b0 * rx * ry * drx + b1 * ry * r^3 + 3 * b1 * rx * ry * r^2 * drx
    w122 = b0 * r * rx + b0 * rx * ry * dry + b1 * rx * r^3 + 3 * b1 * rx * ry * r^2 * dry
    w131 = -c0 * r^3 * (5 - λ^2 * r^2 / 7) - 15 * c0 * rx * r^2 * drx +
           c0 * 5 / 7 * rx * λ^2 * r^4 * drx
    w132 = -15 * c0 * rx * r^2 * dry + c0 * 5 / 7 * rx * λ^2 * r^4 * dry
    m11 = D * (w111 + ν * w122)
    m22 = D * (ν * w111 + w122)
    m12 = D * (1 - ν) / 2 * (w112 + w121)
    q11 = D * (1 - ν) / 2 * λ^2 * (w11 + w131)
    q12 = D * (1 - ν) / 2 * λ^2 * (w12 + w132)
    w1 = SVector(w11, w12, w13)
    q1 = SVector(m11, m12, q11) * nc[1] + SVector(m12, m22, q12) * nc[2]

    w21 = b0 * r * rx * ry + b1 * r^3 * rx * ry
    w22 = a0 * r^3 + a1 * r * (4 * ry^2 + rx^2) + a2 * r^3 * (6 * ry^2 + rx^2)
    w23 = -c0 * ry * r^3 * (5 - λ^2 * r^2 / 7)
    w211 = b0 * r * ry + b0 * rx * ry * drx + b1 * ry * r^3 + 3 * b1 * rx * ry * r^2 * drx
    w212 = b0 * r * rx + b0 * rx * ry * dry + b1 * rx * r^3 + 3 * b1 * rx * ry * r^2 * dry
    w221 = 3 * a0 * r^2 * drx + a1 * drx * (4 * ry^2 + rx^2) + 2 * a1 * r * rx +
           3 * a2 * r^2 * drx * (6 * ry^2 + rx^2) + 2 * a2 * r^3 * rx
    w222 = 3 * a0 * r^2 * dry + a1 * dry * (4 * ry^2 + rx^2) + 8 * a1 * r * ry +
           3 * a2 * r^2 * dry * (6 * ry^2 + rx^2) + 12 * a2 * r^3 * ry
    w231 = -15 * c0 * ry * r^2 * drx + c0 * 5 / 7 * ry * λ^2 * r^4 * drx
    w232 = -c0 * r^3 * (5 - λ^2 * r^2 / 7) - 15 * c0 * ry * r^2 * dry +
           c0 * 5 / 7 * ry * λ^2 * r^4 * dry
    m11 = D * (w211 + ν * w222)
    m22 = D * (ν * w211 + w222)
    m12 = D * (1 - ν) / 2 * (w212 + w221)
    q21 = D * (1 - ν) / 2 * λ^2 * (w21 + w231)
    q22 = D * (1 - ν) / 2 * λ^2 * (w22 + w232)
    w2 = SVector(w21, w22, w23)
    q2 = SVector(m11, m12, q21) * nc[1] + SVector(m12, m22, q22) * nc[2]

    w31 = -(r^2 / 16 + r^3 / 45) * rx / D
    w32 = -(r^2 / 16 + r^3 / 45) * ry / D
    w33 = -(r^2 / 2 + 2 * r^3 / 9) / ((1 - ν) * λ^2 * D) + (1 / 64 + r / 225) * r^4 / D
    m11 = -(1 / 8 + r / 15) * (rx^2 + ν * ry^2) - (1 + ν) * (r^2 / 16 + r^3 / 45)
    m22 = -(1 / 8 + r / 15) * (ry^2 + ν * rx^2) - (1 + ν) * (r^2 / 16 + r^3 / 45)
    m12 = -(1 - ν) * (1 / 8 + r / 15) * rx * ry
    q31 = -(1 + 2 * r / 3) * rx / 2
    q32 = -(1 + 2 * r / 3) * ry / 2
    w3 = SVector(w31, w32, w33)
    q3 = SVector(m11, m12, q31) * nc[1] + SVector(m12, m22, q32) * nc[2]

    W = hcat(w1, w2, w3)
    Q = hcat(q1, q2, q3)
    return W, Q
end

function _drm_pts(mesh::FSDTMesh)
    return Point2D[mesh.nodes; mesh.internal]
end

"""MATLAB `F` blocks: rotations `r-λ² r³/9`, deflection `1+r`."""
function _drm_F(pts, λ)
    nt = length(pts)
    F = zeros(3nt, 3nt)
    @inbounds for j in 1:nt, i in 1:nt
        r = norm(pts[i] - pts[j])
        fr = r - λ^2 * r^3 / 9
        fw = 1 + r
        I = 3i - 2
        J = 3j - 2
        F[I, J] = fr
        F[I+1, J+1] = fr
        F[I+2, J+2] = fw
    end
    @inbounds for i in 1:3nt
        F[i, i] += 1e-12 * (abs(F[i, i]) + 1)
    end
    return F
end

function _drm_UQ(mesh::FSDTMesh, pts)
    n = _n(mesh)
    nt = length(pts)
    D = bending_stiffness(mesh.props)
    ν = mesh.props.ν
    λ = reissner_lambda(mesh.props)
    Uchp = zeros(3n, 3nt)
    Qchp = zeros(3n, 3nt)
    @inbounds for j in 1:nt
        xm, ym = pts[j][1], pts[j][2]
        for i in 1:n
            nc = mesh.Normal[i]
            W, Q = _uqchp(xm, ym, mesh.nodes[i][1], mesh.nodes[i][2], nc, D, ν, λ)
            Uchp[3i-2:3i, 3j-2:3j] .= W
            Qchp[3i-2:3i, 3j-2:3j] .= Q
        end
    end
    return Uchp, Qchp
end

"""RIM of `U*` (`ID`, 3×3 per source) from Maxima `Fρ`."""
function _fsdt_ID(mesh::FSDTMesh; npg::Int=10)
    n = _n(mesh)
    ni = _ni(mesh)
    pts = Point2D[mesh.nodes; mesh.internal]
    nsrc = n + ni
    ID = zeros(3nsrc, 3)
    qsi, w = gausslegendre(npg)
    props = mesh.props
    @showprogress "FSDT ID (RIM)" for i in 1:nsrc
        pf = pts[i]
        Acc = zeros(3, 3)
        for el in mesh.elements
            for (ig, ξ) in enumerate(qsi)
                pg, J, n̂ = elem_geom(el, ξ)
                RX, RY = pg[1] - pf[1], pg[2] - pf[2]
                R = hypot(RX, RY)
                R < 1e-14 && continue
                nr = (n̂[1] * RX + n̂[2] * RY) / R
                F = fsdt_Fρ(props, pg, pf, n̂)
                Acc .+= F .* (nr / R * J * w[ig])
            end
        end
        ID[3i-2:3i, :] .= Acc
    end
    return ID
end

"""RIM of `φ(|x-x_j|)` at each DIBEM centre (Laplace `int(rbf,x,y)`)."""
function _fsdt_IF(mesh, pts, rbf; npg::Int=10)
    nt = length(pts)
    IF = zeros(nt)
    qsi, w = gausslegendre(npg)
    @inbounds for j in 1:nt
        xj = pts[j]
        s = 0.0
        for el in mesh.elements
            for (ig, ξ) in enumerate(qsi)
                pg, J, n̂ = elem_geom(el, ξ)
                RX, RY = pg[1] - xj[1], pg[2] - xj[2]
                R = hypot(RX, RY)
                R < 1e-14 && continue
                nr = (n̂[1] * RX + n̂[2] * RY) / R
                s += int(rbf, xj, pg) * (nr / R * J * w[ig])
            end
        end
        IF[j] = s
    end
    return IF
end

"""`∫_Ω p_k dΩ` by RIM from the first centroid (Laplace `_dibem_monomial_IP`)."""
function _fsdt_monomial_IP(mesh, rbf; npg::Int=10)
    pdeg = max(poly_deg(rbf), -1)
    pdeg < 0 && return zeros(0)
    npoly = rbf_npoly(2, pdeg)
    npoly == 0 && return zeros(0)
    mon = MonomialBasis(2, pdeg)
    IP = zeros(npoly)
    internals = _plate_internal(mesh)
    nodes = _plate_nodes(mesh)
    x0 = isempty(internals) ? sum(nodes) / length(nodes) : internals[1]
    qsi, w = gausslegendre(npg)
    for el in mesh.elements
        for (ig, ξ) in enumerate(qsi)
            pg, J, n̂ = elem_geom(el, ξ)
            RX, RY = pg[1] - x0[1], pg[2] - x0[2]
            R = hypot(RX, RY)
            R < 1e-14 && continue
            nr = (n̂[1] * RX + n̂[2] * RY) / R
            IP .+= int(mon, x0, pg) .* (nr / R * J * w[ig])
        end
    end
    return IP
end

"""Laplace-style DIBEM: F, D on boundary collocation + domain-cell centroids."""
function _dibem_rim_fsdt!(mesh::FSDTMesh; npg::Int=10, rbf=PHS())
    isempty(mesh.H) && assemble_fsdt!(mesh)
    n = _n(mesh)
    ni = _ni(mesh)
    ndof = 3n + 3ni
    pts = Point2D[mesh.nodes; mesh.internal]
    nt = length(pts)
    ID = _fsdt_ID(mesh; npg=npg)
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
    @inbounds for j in 1:nt
        cj = c[j]
        abs(cj) < 1e-30 && continue
        xj = pts[j]
        for i in 1:nt
            i == j && continue
            U, _, _ = fsdt_kernels(mesh.props, xj, pts[i], dummy)
            M[3i-2:3i, 3j-2:3j] .= U .* cj
        end
    end
    @inbounds for i in 1:nt
        rowsum = zeros(3, 3)
        for j in 1:nt
            j == i && continue
            rowsum .+= M[3i-2:3i, 3j-2:3j]
        end
        M[3i-2:3i, 3i-2:3i] .= ID[3i-2:3i, :] .- rowsum
    end
    I2 = mesh.props.ρ * mesh.props.h^3 / 12
    I0 = mesh.props.ρ * mesh.props.h
    @inbounds for j in 1:nt
        M[:, 3j-2] .*= I2
        M[:, 3j-1] .*= I2
        M[:, 3j] .*= I0
    end
    mesh.M = M
    q = zeros(ndof)
    qc = mesh.props.q_c
    @inbounds for i in 1:nt
        q[3i-2:3i] .= ID[3i-2:3i, 3] .* qc
    end
    mesh.q = q
    _dibem_ibp_wmaps!(mesh, pts, c, dummy)
    return mesh
end

"""Central-difference `∂U/∂p_g` of a kernel `Ufun(pg, pf, n) -> (U, …)`."""
function _fd_dU_pg(Ufun, xj, pf, n̂, hfd::Float64=1e-7)
    hx = Point2D(hfd, 0.0)
    hy = Point2D(0.0, hfd)
    Uxp = Ufun(xj + hx, pf, n̂)
    Uxm = Ufun(xj - hx, pf, n̂)
    Uyp = Ufun(xj + hy, pf, n̂)
    Uym = Ufun(xj - hy, pf, n̂)
    return (Uxp - Uxm) / (2 * hfd), (Uyp - Uym) / (2 * hfd)
end

"""IBP maps `Mx, My`: volume piece `-∫ ∇U*_w · v` with `Mx_ij = -∂U*_w/∂X_x c_j`.

One divergence theorem:
`∫_Ω U* ∇·v = ∫_Γ U*(v·n) - ∫_Ω ∇U* · v ≈ Γ + Mx vx + My vy`.
Diagonal so `Mx 1 + Γ(e_x) = 0` (constant `v` has zero divergence). Same `c`
as `M`. `Γ` is [`ibp_gamma_vn`](@ref) on the unmixed `G`.
"""
function _dibem_ibp_wmaps!(mesh::FSDTMesh, pts, c, dummy)
    ndof = _ndof(mesh)
    nt = length(pts)
    nd = mesh.ndofn
    wcol = nd   # last DOF is w
    Mx = zeros(ndof, nt)
    My = zeros(ndof, nt)
    Ufun = (pg, pf, n̂) -> begin
        U, rest... = fsdt_kernels(mesh.props, pg, pf, n̂)
        return U
    end
    if nd == 5
        Ufun = (pg, pf, n̂) -> begin
            U, _ = unsym_fsdt_kernels(pg, pf, n̂, mesh.props)
            return U
        end
    end
    @inbounds for j in 1:nt
        cj = c[j]
        abs(cj) < 1e-30 && continue
        xj = pts[j]
        for i in 1:nt
            i == j && continue
            dUx, dUy = _fd_dU_pg(Ufun, xj, pts[i], dummy)
            rows = (nd * (i - 1) + 1):(nd * i)
            @inbounds for (k, r) in enumerate(rows)
                Mx[r, j] = -dUx[k, wcol] * cj
                My[r, j] = -dUy[k, wcol] * cj
            end
        end
    end
    G = mesh.G
    nrm = mesh.Normal
    γx = isempty(G) ? zeros(ndof) :
         ibp_gamma_vn(G, nrm, ones(nt), zeros(nt), nd)
    γy = isempty(G) ? zeros(ndof) :
         ibp_gamma_vn(G, nrm, zeros(nt), ones(nt), nd)
    @inbounds for i in 1:nt
        rows = (nd * (i - 1) + 1):(nd * i)
        Mx[rows, i] .= .-view(γx, rows) .- vec(sum(view(Mx, rows, :), dims=2))
        My[rows, i] .= .-view(γy, rows) .- vec(sum(view(My, rows, :), dims=2))
    end
    mesh.Mx = Mx
    mesh.My = My
    return mesh
end

"""`∫_Γ U*(N∇w)·n dΓ ≈ G t` with `Vn_i = n·(N∇w)` in traction slot `vn_slot`.

Kirchhoff: `vn_slot=1` (stride 2). FSDT: `vn_slot=3` (stride 3). Hsu–Hwu:
`vn_slot=5`. Completes the IBP of `∫_Ω U* ∇·(N∇w)` (BEM atual `aplicaT`
`G` term). Uses the **unmixed** `G` so SS unknown `Vn` is not overwritten.
"""
function ibp_gamma_vn(G, normals, vx, vy, vn_slot::Int)
    isempty(G) && return zeros(eltype(vx), 0)
    n = length(normals)
    nb = size(G, 2)
    T = eltype(vx)
    t = zeros(T, nb)
    nd = vn_slot == 1 ? 2 : vn_slot
    @inbounds for i in 1:min(n, length(vx))
        j = nd * (i - 1) + vn_slot
        j > nb && break
        nx, ny = normals[i][1], normals[i][2]
        t[j] = nx * vx[i] + ny * vy[i]
    end
    return G * t
end

"""One IBP of `∫ U* ∇·v`: volume `Mx vx + My vy` plus `Γ = ∫_Γ U*(v·n)`."""
function ibp_div_Uv(Mx, My, G, normals, vx, vy, vn_slot::Int)
    vol = Mx * vx .+ My * vy
    (isempty(G) || isempty(normals)) && return vol
    γ = ibp_gamma_vn(G, normals, vx, vy, vn_slot)
    length(γ) == length(vol) || return vol
    return vol .+ γ
end

"""Polar RIM of Wang ``U^*_{i3} q`` for uniform pressure (MATLAB `int_carga`)."""
function _rim_q_wang!(mesh::FSDTMesh; npg::Int=8, nρ::Int=8)
    n = _n(mesh)
    ni = _ni(mesh)
    ndof = 3n + 3ni
    q = zeros(ndof)
    qc = mesh.props.q_c
    if qc == 0
        mesh.q = q
        return mesh
    end
    qsi, w = gausslegendre(npg)
    ρg, ρw = gausslegendre(nρ)
    pts = Point2D[mesh.nodes; mesh.internal]
    props = mesh.props
    @showprogress "Wang RIM q" for i in 1:(n + ni)
        pf = pts[i]
        qe = zeros(3)
        for el in mesh.elements
            for (ig, ξ) in enumerate(qsi)
                pg, J, n̂ = elem_geom(el, ξ)
                RX = pg[1] - pf[1]
                RY = pg[2] - pf[2]
                R = hypot(RX, RY)
                R < 1e-14 && continue
                nr = (n̂[1] * RX + n̂[2] * RY) / R
                F = zeros(3)
                for (ir, γ) in enumerate(ρg)
                    ρ = (γ + 1) / 2 * R
                    ρ < 1e-14 && continue
                    pgρ = pf + (ρ / R) * (pg - pf)
                    U, _, _ = wang_kernels(pgρ, pf, n̂, props.D, props.AT; nθ=props.nθ)
                    F .+= U[:, 3] .* (ρ * (R / 2) * ρw[ir])
                end
                qe .+= F .* (nr / R * J * w[ig] * qc)
            end
        end
        q[3i-2:3i] .= qe
    end
    mesh.q = q
    return mesh
end

"""
    dibem_fsdt!(mesh; method=:dibem)

- `:dibem` (default) — Laplace DIBEM on **boundary collocation + domain-cell
  centroids** (`format2d` `pontointerno`): PHS Gram `F`, polynomial `c`
  (`F c = IF`), `M_ij = U*(ξ_i,x_j) c_j`, diagonal so `M 1 = ID`.
  `ID` is RIM of the Maxima primitive `∫ U* ρ dρ`. Uniform `q` is
  `ID[:,3] q_c`.
- `:drm` — MATLAB `calcQ_drm` / `calcM_drm` particular solutions (`uqchp`).
  Isotropic only.
"""
function dibem_fsdt!(mesh::FSDTMesh; method::Symbol=:dibem, npg::Int=10,
        rbf=PHS())
    if method === :dibem
        mesh.ndofn == 5 && return _dibem_rim_unsym!(mesh; npg=npg, rbf=rbf)
        return _dibem_rim_fsdt!(mesh; npg=npg, rbf=rbf)
    end
    method === :drm || error("dibem_fsdt! method must be :dibem or :drm")
    mesh.ndofn == 5 && error("DRM uqchp is 3-DOF isotropic; use method=:dibem")
    mesh.props isa LaminateFSDTProps &&
        error("DRM uqchp is isotropic; use method=:dibem for Wang laminates")
    isempty(mesh.H) && assemble_fsdt!(mesh)
    n = _n(mesh)
    ni = _ni(mesh)
    ndof = 3n + 3ni
    pts = _drm_pts(mesh)
    nt = length(pts)
    λ = reissner_lambda(mesh.props)
    F = _drm_F(pts, λ)
    Uchp, Qchp = _drm_UQ(mesh, pts)
    Hc = mesh.H[1:3n, 1:3n]
    Gc = mesh.G[1:3n, 1:3n]
    Op = Hc * Uchp .- Gc * Qchp           # 3n × 3nt
    if ni > 0
        Uint = zeros(3ni, 3nt)
        D = bending_stiffness(mesh.props)
        ν = mesh.props.ν
        nc0 = Point2D(1.0, 0.0)
        @inbounds for j in 1:nt, k in 1:ni
            W, _ = _uqchp(pts[j][1], pts[j][2], mesh.internal[k][1],
                mesh.internal[k][2], nc0, D, ν, λ)
            Uint[3k-2:3k, 3j-2:3j] .= W
        end
        Hd = mesh.H[3n+1:ndof, 1:3n]
        Gd = mesh.G[3n+1:ndof, 1:3n]
        Op = [Op; Uint .+ Hd * Uchp .- Gd * Qchp]
    end
    ρ = mesh.props.ρ
    h = mesh.props.h
    I2 = ρ * h^3 / 12
    I0 = ρ * h
    Λ = zeros(3nt, 3nt)
    @inbounds for j in 1:nt
        Λ[3j-2, 3j-2] = I2
        Λ[3j-1, 3j-1] = I2
        Λ[3j, 3j] = I0
    end
    mesh.M = Op * (F \ Λ)
    qq = zeros(3nt)
    @inbounds for j in 1:nt
        qq[3j] = mesh.props.q_c
    end
    mesh.q = Op * (F \ qq)
    return mesh
end

# =============================================================================
# BC & solve
# =============================================================================

"""Soft SS: w known, moments free. Clamped: ψx,ψy,w known. Free: all traction."""
function apply_bc_fsdt(mesh::FSDTMesh)
    H = copy(mesh.H)
    G = copy(mesh.G)
    ndof = _ndof(mesh)
    nb = _nb(mesh)
    is_kin = falses(ndof)
    known = zeros(ndof)
    for dof in 1:nb
        if mesh.BC[dof] == 0
            is_kin[dof] = true
            known[dof] = mesh.BV[dof]
            colH = H[:, dof]
            colG = G[:, dof]
            H[:, dof] = -colG
            G[:, dof] = -colH
        else
            known[dof] = mesh.BV[dof]
        end
    end
    b = mesh.q .+ G * known[1:nb]
    return H, b, is_kin, known
end

"""Solve the mixed FSDT system; DRM/RIM load if `q_c ≠ 0` and `q` is empty.

`large=true` is Hsu–Hwu 5-DOF von Kármán ([`solve_unsym_fsdt_large!`](@ref)).
3-DOF Wang large deflection stays on [`solve_laminated_shell!`](@ref).
"""
function solve_fsdt!(mesh::FSDTMesh; large::Bool=false, kwargs...)
    large && return solve_unsym_fsdt_large!(mesh; kwargs...)
    isempty(mesh.H) && assemble_fsdt!(mesh)
    if iszero(mesh.q) && mesh.props.q_c != 0
        dibem_fsdt!(mesh)
    end
    A, b, is_kin, known = apply_bc_fsdt(mesh)
    x = A \ b
    ndof = length(x)
    nb = _nb(mesh)
    u = zeros(ndof)
    t = zeros(nb)
    for dof in 1:ndof
        if is_kin[dof]
            u[dof] = known[dof]
            dof <= nb && (t[dof] = x[dof])
        else
            u[dof] = x[dof]
            if dof <= nb
                t[dof] = known[dof]
            end
        end
    end
    mesh.u = u
    mesh.t = t
    if mesh.ndofn == 5 && !isempty(mesh.eq_type) && all(==(3), mesh.eq_type) &&
            _ni(mesh) > 0
        _unsym_somigliana_interior!(mesh)
    end
    return u, t
end

"""
    solve_fsdt_houbolt!(mesh; dt, tmax, qfun=nothing, mass=:raw)

Houbolt on ``M ü + H u = G t + q(t)`` (MATLAB `Dynsolver`):
``(2M/Δt²+H) u^{n+1}`` with traction columns of ``H`` swapped for ``-G``.
`mass=:shift` replaces `M` by ``(M+Mᵀ)/2 + (|λ_min|+ε)I`` so DIBEM's
indefinite spectrum does not invert the implicit operator.
`qfun(t)` scales `mesh.q`. Returns `(t, w_center)`.
"""
function solve_fsdt_houbolt!(mesh::FSDTMesh; dt::Float64=1e-3, tmax::Float64=0.5,
        qfun=nothing, mass::Symbol=:raw)
    isempty(mesh.M) && dibem_fsdt!(mesh)
    H, G = mesh.H, mesh.G
    M = mesh.M
    if mass === :shift
        S = Symmetric((M + M') / 2)
        λmin = minimum(eigvals(S))
        shift = λmin < 0 ? -λmin + 1e-12 * max(1.0, -λmin) : 0.0
        M = Matrix(S) + shift * I
    elseif mass !== :raw
        error("mass must be :raw or :shift")
    end
    n = _n(mesh)
    ni = _ni(mesh)
    ndof = _ndof(mesh)
    nb = _nb(mesh)
    Utemp = zeros(ndof)
    Ftemp = zeros(nb)
    kin = falses(ndof)
    @inbounds for dof in 1:nb
        if mesh.BC[dof] == 0
            kin[dof] = true
            Utemp[dof] = mesh.BV[dof]
        else
            Ftemp[dof] = mesh.BV[dof]
        end
    end
    bfix = G * Ftemp .- H * Utemp
    A = H .+ (2 / dt^2) .* M
    @inbounds for dof in 1:nb
        kin[dof] && (A[:, dof] = -G[:, dof])
    end
    ic = ni > 0 ? mesh.ndofn * (n + 1) : mesh.ndofn
    t = 0.0
    ts = Float64[0.0]
    wcs = Float64[0.0]
    Uhist = zeros(ndof, 3)
    qscale = qfun === nothing ? (tt -> 1.0) : qfun
    nstep = ceil(Int, tmax / dt)
    @showprogress "FSDT Houbolt" for _ in 1:nstep
        t += dt
        rhs = bfix .+ mesh.q .* qscale(t) .+
              (1 / dt^2) .* (M * (5 .* Uhist[:, 3] .- 4 .* Uhist[:, 2] .+ Uhist[:, 1]))
        x = A \ rhs
        u = copy(Utemp)
        @inbounds for dof in 1:ndof
            kin[dof] || (u[dof] = x[dof])
        end
        Uhist[:, 1] .= Uhist[:, 2]
        Uhist[:, 2] .= Uhist[:, 3]
        Uhist[:, 3] .= u
        push!(ts, t)
        push!(wcs, u[ic])
    end
    mesh.u = Uhist[:, 3]
    return (t=ts, w_center=wcs)
end

# =============================================================================
# Mesh
# =============================================================================

function _fsdt_bc_char(c::Char, vn::Float64=0.0)
    c = uppercase(c)
    if c == 'C'
        return (0, 0.0, 0, 0.0, 0, 0.0)   # ψx, ψy, w kinematic
    elseif c == 'S'
        return (1, 0.0, 1, 0.0, 0, 0.0)   # moments free, w=0
    elseif c == 'F'
        return (1, 0.0, 1, 0.0, 1, vn)    # all traction; `vn` is Qn
    else
        error("unknown BC '$c' (use C/S/F)")
    end
end

"""5-DOF SS-1: Navier membrane (u=0 on y=const, v=0 on x=const), β free, w=0."""
function _unsym_bc_char(c::Char, edge::Int, vn::Float64=0.0)
    c = uppercase(c)
    if c == 'C'
        return (0, 0.0, 0, 0.0, 0, 0.0, 0, 0.0, 0, 0.0)
    elseif c == 'S'
        if edge == 1 || edge == 3
            return (0, 0.0, 1, 0.0, 1, 0.0, 1, 0.0, 0, 0.0)  # u, w kin
        else
            return (1, 0.0, 0, 0.0, 1, 0.0, 1, 0.0, 0, 0.0)  # v, w kin
        end
    elseif c == 'F'
        return (1, 0.0, 1, 0.0, 1, 0.0, 1, 0.0, 1, vn)
    else
        error("unknown BC '$c' (use C/S/F)")
    end
end

function _fsdt_edge!(elems, nodes, normals, BC, BV, p0, p1, n_el, bc, qsi, wi, tag;
        ndofn::Int=3)
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
            for k in 1:2:length(bc)
                push!(BC, bc[k]); push!(BV, bc[k + 1])
            end
        end
        L = abs(dot(J, wi))
        push!(elems, Element(;
            index=collect(Int64, idx),
            Jacobian=collect(Float64, J),
            Length=Float64(L),
            Region=Int64(tag),
            geo=X))
    end
    return n_hat
end

"""
    build_rect_fsdt(; Lx, Ly, n_el=8, bc="SSSS", vn=(0,0,0,0), props, n_internal=1, p=2)

Rectangle ``[0,L_x]×[0,L_y]`` via [`quadrado_fsdt`](@ref) then
[`formatdata`](@ref)/`format2d` → `BEMdata{<:AbstractFSDT}` (3-DOF Reissner/Wang
or 5-DOF Hsu–Hwu). `bc` is 4 chars bottom,right,top,left (`C`/`S`/`F`).
`n_el` is one count per edge or a 4-tuple. `vn[e]` is the known ``Q_n`` on a
free edge (MATLAB `BCF` `Vz`). Internals are cell centroids, closest-to-centre first.
"""
function build_rect_fsdt(; Lx=1.0, Ly=1.0, n_el=8, bc="SSSS",
        vn::NTuple{4,Float64}=(0.0, 0.0, 0.0, 0.0),
        props=FSDTProps(), n_internal=1, p::Integer=2)
    length(bc) == 4 || error("bc must have 4 characters")
    p >= 1 || error("degree must be ≥ 1, got $p")
    nels = n_el isa Integer ? (n_el, n_el, n_el, n_el) : Tuple(n_el)
    length(nels) == 4 || error("n_el must be an integer or 4-tuple")
    ndiv = map(n -> n + 1, nels)
    B = parentmodule(@__MODULE__)
    tag = n_el isa Integer ? string(n_el) : join(n_el, "x")
    nd = _ndofn(props)
    msh = B.quadrado_fsdt(; Lx=Lx, Ly=Ly, ndiv=ndiv, ordem=p, bc=bc, vn=vn,
        ndof=nd, nome="fsdt_$(Lx)_$(Ly)_$(tag)_p$(p)_d$(nd)")
    dad = formatdata(msh, props; tipo=p, pontointerno=false)
    set_internal_nodes!(dad, _fsdt_cell_centroids(Lx, Ly, n_internal))
    return dad
end

"""
    build_square_fsdt(; a=1, n_el=8, bc=\"SSSS\", props=FSDTProps(), n_internal=1, p=2)

Square ``[0,a]²`` via [`quadrado_fsdt`](@ref) then [`formatdata`](@ref).
Returns `BEMdata{<:AbstractFSDT}` (Reissner, Wang, or Hsu–Hwu 5-DOF).
`bc` is 4 chars bottom,right,top,left (`C`/`S`/`F`).
Soft SS: ``w=0``, moments free (MATLAB `bcu` 0 0 1).
`props` is `FSDTProps` (Vander Weeën) or `LaminateFSDTProps` (Wang).
"""
function build_square_fsdt(; a=1.0, n_el=8, bc="SSSS",
        props=FSDTProps(), n_internal=1, p::Integer=2)
    return build_rect_fsdt(; Lx=a, Ly=a, n_el=n_el, bc=bc, props=props,
        n_internal=n_internal, p=p)
end

"""Centroids of an `n_div×n_div` Cartesian mesh of `[0,Lx]×[0,Ly]`."""
function _fsdt_cell_centroids(Lx, Ly, n_internal::Integer)
    n_internal <= 0 && return Point2D[]
    n_div = max(1, ceil(Int, sqrt(n_internal)))
    hx, hy = Lx / n_div, Ly / n_div
    pts = Point2D[]
    best, bestd = 1, Inf
    cx, cy = Lx / 2, Ly / 2
    for j in 1:n_div, i in 1:n_div
        p = Point2D((i - 0.5) * hx, (j - 0.5) * hy)
        push!(pts, p)
        d = hypot(p[1] - cx, p[2] - cy)
        if d < bestd
            bestd = d
            best = length(pts)
        end
    end
    if best != 1
        pts[1], pts[best] = pts[best], pts[1]
    end
    return pts
end
_fsdt_cell_centroids(a, n_internal::Integer) = _fsdt_cell_centroids(a, a, n_internal)

"""
    build_circle_fsdt(; R=1, n_el=16, bc='C', props, n_internal=1, p=2)

Disk of radius `R`, `n_el` straight chords, uniform BC (`C`/`S`/`F`).
Internals are a Cartesian grid clipped to the disk, closest-to-centre first.
"""
function build_circle_fsdt(; R=1.0, n_el::Integer=16, bc::Char='C',
        props=FSDTProps(), n_internal=1, p::Integer=2, vn::Float64=0.0)
    n_el >= 3 || error("circle needs n_el ≥ 3")
    poly = Legendre(p)
    qsi, wi = discontinuous_nodes_weights(p)
    elems = Element[]
    nodes = Point2D[]
    normals = Point2D[]
    BC = Int[]
    BV = Float64[]
    ndofn = _ndofn(props)
    bct = ndofn == 5 ? _unsym_bc_char(bc, 1, vn) : _fsdt_bc_char(bc, vn)
    Δ = 2π / n_el
    for e in 1:n_el
        θ0 = (e - 1) * Δ
        θ1 = e * Δ
        p0 = Point2D(R * cos(θ0), R * sin(θ0))
        p1 = Point2D(R * cos(θ1), R * sin(θ1))
        _fsdt_edge!(elems, nodes, normals, BC, BV, p0, p1, 1, bct, qsi, wi, e;
            ndofn=ndofn)
    end
    internal = Point2D[]
    if n_internal >= 1
        push!(internal, Point2D(0.0, 0.0))
        if n_internal > 1
            g = ceil(Int, sqrt(n_internal))
            xs = range(-R, R; length=g + 2)[2:end-1]
            for y in xs, x in xs
                hypot(x, y) < 0.85 * R || continue
                hypot(x, y) < 1e-12 && continue
                push!(internal, Point2D(x, y))
                length(internal) >= n_internal && break
            end
        end
    end
    return FSDTMesh(elems, poly, collect(wi), nodes, normals, BC, BV, props;
        internal=internal)
end

# =============================================================================
# Analytical FSDT Navier (soft SS)
# =============================================================================

"""Centre deflection, isotropic FSDT SS square, uniform ``q`` (κ=5/6)."""
function navier_w_ss_fsdt(x, y; a=1.0, q=1.0, D=1.0, ν=0.3, κGh=1.0, nterms=40)
    D66 = (1 - ν) * D / 2
    w = 0.0
    @inbounds for m in 1:2:nterms, n in 1:2:nterms
        α = m * π / a
        β = n * π / a
        K11 = D * α^2 + D66 * β^2 + κGh
        K12 = (ν * D + D66) * α * β
        K13 = κGh * α
        K22 = D66 * α^2 + D * β^2 + κGh
        K23 = κGh * β
        K33 = κGh * (α^2 + β^2)
        K = [K11 K12 K13; K12 K22 K23; K13 K23 K33]
        Δ = K \ [0.0, 0.0, 16q / (π^2 * m * n)]
        w += Δ[3] * sin(α * x) * sin(β * y)
    end
    return w
end

"""FSDT Navier SS square, cross-ply (`D16=D26=A45=0`)."""
function navier_w_ss_fsdt(x, y, p::LaminateFSDTProps; a=1.0, q=p.q_c, nterms=40)
    D11, D22, D12, D66 = p.D[1, 1], p.D[2, 2], p.D[1, 2], p.D[3, 3]
    A44, A55 = p.AT[1, 1], p.AT[2, 2]
    w = 0.0
    @inbounds for m in 1:2:nterms, n in 1:2:nterms
        α = m * π / a
        β = n * π / a
        K11 = D11 * α^2 + D66 * β^2 + A55
        K12 = (D12 + D66) * α * β
        K13 = A55 * α
        K22 = D66 * α^2 + D22 * β^2 + A44
        K23 = A44 * β
        K33 = A55 * α^2 + A44 * β^2
        K = [K11 K12 K13; K12 K22 K23; K13 K23 K33]
        Δ = K \ [0.0, 0.0, 16q / (π^2 * m * n)]
        w += Δ[3] * sin(α * x) * sin(β * y)
    end
    return w
end

"""`D` (3×3 Voigt) and `AT` (ContsLam) for constitutive recovery."""
function _fsdt_D_AT(p::FSDTProps)
    Dv = bending_stiffness(p)
    ν = p.ν
    As = shear_stiffness(p)
    D = @SMatrix [Dv ν*Dv 0; ν*Dv Dv 0; 0 0 (1 - ν)*Dv/2]
    AT = @SMatrix [As 0; 0 As]
    return D, AT
end
_fsdt_D_AT(p::LaminateFSDTProps) = p.D, p.AT

"""Navier SS FSDT `w, Mx, My, Qx, Qy` (cross-ply)."""
function navier_ss_fsdt_MQ(x, y, p; a=1.0, q=1.0, nterms=40)
    D, AT = _fsdt_D_AT(p)
    D11, D22, D12, D66 = D[1, 1], D[2, 2], D[1, 2], D[3, 3]
    A44, A55 = AT[1, 1], AT[2, 2]
    w = Mx = My = Qx = Qy = 0.0
    @inbounds for m in 1:2:nterms, n in 1:2:nterms
        α = m * π / a
        β = n * π / a
        K11 = D11 * α^2 + D66 * β^2 + A55
        K12 = (D12 + D66) * α * β
        K13 = A55 * α
        K22 = D66 * α^2 + D22 * β^2 + A44
        K23 = A44 * β
        K33 = A55 * α^2 + A44 * β^2
        Δ = [K11 K12 K13; K12 K22 K23; K13 K23 K33] \ [0.0, 0.0, 16q / (π^2 * m * n)]
        X, Y, W = Δ
        s = sin(α * x) * sin(β * y)
        cxs = cos(α * x) * sin(β * y)
        syc = sin(α * x) * cos(β * y)
        w += W * s
        Mx += (-D11 * α * X - D12 * β * Y) * s
        My += (-D12 * α * X - D22 * β * Y) * s
        Qx += A55 * (X + α * W) * cxs
        Qy += A44 * (Y + β * W) * syc
    end
    return (w=w, Mx=Mx, My=My, Qx=Qx, Qy=Qy)
end

"""
    fsdt_resultants(mesh) -> (Mx, My, Mxy, Qx, Qy, w, pts)

Useche 8.5: `M = D κ`, `Q = AT(ψ + ∇w)` from RBF gradients of collocation `ψ, w`.
Symmetric FSDT (`ndofn=3`); `N=0`.
"""
function fsdt_resultants(mesh::FSDTMesh; rbf=PHS(3; poly_deg=1))
    mesh.ndofn == 3 || error("fsdt_resultants is 3-DOF (Wang / Reissner)")
    isempty(mesh.u) && error("solve_fsdt! first")
    D, AT = _fsdt_D_AT(mesh.props)
    pts = Point2D[mesh.nodes; mesh.internal]
    nt = length(pts)
    ψx = [mesh.u[3i - 2] for i in 1:nt]
    ψy = [mesh.u[3i - 1] for i in 1:nt]
    w = [mesh.u[3i] for i in 1:nt]
    ops = rbf_gradient_ops(pts; rbf=rbf)
    ψxx, ψxy = ops.Fx * ψx, ops.Fy * ψx
    ψyx, ψyy = ops.Fx * ψy, ops.Fy * ψy
    wx, wy = ops.Fx * w, ops.Fy * w
    κx, κy, κxy = ψxx, ψyy, ψxy .+ ψyx
    Mx = D[1, 1] .* κx .+ D[1, 2] .* κy .+ D[1, 3] .* κxy
    My = D[1, 2] .* κx .+ D[2, 2] .* κy .+ D[2, 3] .* κxy
    Mxy = D[1, 3] .* κx .+ D[2, 3] .* κy .+ D[3, 3] .* κxy
    γx = ψx .+ wx
    γy = ψy .+ wy
    Qx = AT[2, 2] .* γx .+ AT[1, 2] .* γy
    Qy = AT[1, 1] .* γy .+ AT[1, 2] .* γx
    return (Mx=Mx, My=My, Mxy=Mxy, Qx=Qx, Qy=Qy, w=w, ψx=ψx, ψy=ψy, pts=pts)
end

include("UnsymFSDT.jl")
include("FSDTDual.jl")
include("FSDTXbem.jl")
