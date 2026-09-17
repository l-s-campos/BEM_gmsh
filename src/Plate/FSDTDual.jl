# Reissner Dual BEM for cracked plates (Useche Ch.10 / Dirgantara).
# Same Portela–Aliabadi–Rooke layout as `assemble_dual_elasticity!`:
#   outer + crack face A → displacement BIE (`eq=2`)
#   crack face B         → traction BIE (`eq=3`, hypersingular)
# Twin coincident faces from `mesh_center_crack` + `prepare_crack!`.
# Free terms copy the in-plane dual: CBIE H += 1/2 (self+twin),
# HBIE G -= 1/2 (self+twin); outer CBIE uses regularized C.

# =============================================================================
# HBIE kernels (MATLAB SolFunWijk / SolFunPijk)
# =============================================================================

"""
    reissner_hbie_kernels(pg, pf, n, nξ, D, ν, λ) -> (W, P)

Traction-BIE kernels ``n^ξ_β W_{iβk}`` (G) and ``n^ξ_β P_{iβk}`` (H).
`n` is the field-element normal, `nξ` the source normal.
"""
function reissner_hbie_kernels(pg::Point2D, pf::Point2D, n::Point2D, nξ::Point2D,
        D, ν, λ)
    RX, RY = pg[1] - pf[1], pg[2] - pf[2]
    R = hypot(RX, RY)
    R < 1e-30 && return zeros(3, 3), zeros(3, 3)
    DRx, DRy = RX / R, RY / R
    DRn = DRx * n[1] + DRy * n[2]
    z = λ * R
    Ko, K1 = _bess_K01(z)
    Az = Ko + 2 / z * (K1 - 1 / z)
    Bz = Ko + 1 / z * (K1 - 1 / z)
    DR = (DRx, DRy)
    nv = (n[1], n[2])
    no = (nξ[1], nξ[2])
    om = 1 - ν
    kP1 = D * om / (4 * π * R^2)
    kP2 = D * om * λ^2 / (4 * π * R)
    kP3 = D * om * λ^2 / (4 * π * R^2)
    kW1 = 1 / (4 * π * R)
    kW2 = -om / (8 * π)
    kW3 = λ^2 / (2 * π)
    kW4 = 1 / (2 * π * R)
    P = zeros(3, 3)
    W = zeros(3, 3)
    @inbounds for b in 1:2
        nb = nv[b]
        DRb = DR[b]
        nob = no[b]
        abs(nob) < 1e-16 && continue
        for a in 1:3, g in 1:3
            Pabg = 0.0
            Wabg = 0.0
            if a < 3 && g < 3
                da, dg = a, g
                δga = Float64(g == a)
                δgb = Float64(g == b)
                δab = Float64(a == b)
                na, ng = nv[da], nv[dg]
                DRa, DRg = DR[da], DR[dg]
                Pabg = kP1 * (
                    (4 * Az + 2 * z * K1 + om) * (δga * nb + δgb * na) +
                    (4 * Az + 1 + 3ν) * δab * ng -
                    (16 * Az + 6 * z * K1 + z^2 * Ko + 2 * om) *
                    ((na * DRb + nb * DRa) * DRg + (δga * DRb + δgb * DRa) * DRn) -
                    2 * (8 * Az + 2 * z * K1 + 1 + ν) *
                    (δab * DRg * DRn + ng * DRa * DRb) +
                    4 * (24 * Az + 8 * z * K1 + z^2 * Ko + 2 * om) *
                    DRa * DRb * DRg * DRn)
                Wabg = kW1 * (
                    (4 * Az + 2 * z * K1 + om) * (δgb * DRa + δga * DRb) -
                    2 * (8 * Az + 2 * z * K1 + om) * DRa * DRb * DRg +
                    (4 * Az + 1 + ν) * δab * DRg)
            elseif g == 3 && a < 3
                da = a
                na = nv[da]
                DRa = DR[da]
                δab = Float64(a == b)
                Pabg = kP2 * ((2 * Az + z * K1) * (DRb * na + DRa * nb) -
                              2 * (4 * Az + z * K1) * DRa * DRb * DRn +
                              2 * Az * δab * DRn)
                Wabg = kW2 * ((2 * (1 + ν) / om * log(z) - 1) * δab + 2 * DRa * DRb)
            elseif a == 3 && g < 3
                dg = g
                ng = nv[dg]
                DRg = DR[dg]
                δgb = Float64(g == b)
                Pabg = -kP2 * ((2 * Az + z * K1) * (δgb * DRn + DRg * nb) +
                               2 * Az * ng * DRb -
                               2 * (4 * Az + z * K1) * DRg * DRb * DRn)
                Wabg = kW3 * (Bz * δgb - Az * DRg * DRb)
            else
                Pabg = kP3 * ((z^2 * Bz + 1) * nb - (z^2 * Az + 2) * DRb * DRn)
                Wabg = kW4 * DRb
            end
            P[a, g] += nob * Pabg
            W[a, g] += nob * Wabg
        end
    end
    return W, P
end

function _hbie_contract(ctes, nξ)
    C1 = @SMatrix [ctes[1, 1, 1] ctes[1, 1, 2] ctes[1, 1, 3]
                   ctes[2, 1, 1] ctes[2, 1, 2] ctes[2, 1, 3]
                   ctes[3, 1, 1] ctes[3, 1, 2] ctes[3, 1, 3]]
    C2 = @SMatrix [ctes[1, 2, 1] ctes[1, 2, 2] ctes[1, 2, 3]
                   ctes[2, 2, 1] ctes[2, 2, 2] ctes[2, 2, 3]
                   ctes[3, 2, 1] ctes[3, 2, 2] ctes[3, 2, 3]]
    return nξ[1] * C1 + nξ[2] * C2
end

"""MATLAB `CalcGtes`: 1/ρ² part of `P_ijk` (self-element, DRn=0)."""
function _hbie_Gctes(nξ, n, J, D, ν, λ)
    om = 1 - ν
    k1 = D * om / (4 * π * J)
    k2 = D * om * λ^2 / (4 * π)
    k3 = D * om * λ^2 / (4 * π * J)
    DR = (-n[2], n[1])
    ctes = zeros(3, 2, 3)
    @inbounds for a in 1:3, b in 1:2, g in 1:3
        if a < 3 && g < 3
            ctes[a, b, g] = k1 * (om * ((g == a) * n[b] + (g == b) * n[a]) +
                                  (-1 + 3ν) * (a == b) * n[g] +
                                  2ν * (n[a] * DR[b] + n[b] * DR[a]) * DR[g] +
                                  2 * om * n[g] * DR[a] * DR[b])
        elseif a == 3 && g < 3
            ctes[a, b, g] = k2 * n[g] * DR[b]
        elseif a == 3 && g == 3
            ctes[a, b, g] = k3 * n[b]
        end
    end
    return _hbie_contract(ctes, nξ)
end

"""MATLAB `CalcHtes`: log part of `P_ijk`."""
function _hbie_Hctes(nξ, n, J, D, ν, λ)
    om = 1 - ν
    k1 = D * om * λ^2 * J / (8 * π)
    k3 = -D * om * λ^4 * J / (8 * π)
    ctes = zeros(3, 2, 3)
    @inbounds for a in 1:3, b in 1:2, g in 1:3
        if a < 3 && g < 3
            ctes[a, b, g] = k1 * ((g == a) * n[b] + (g == b) * n[a] - (a == b) * n[g])
        elseif a == 3 && g == 3
            ctes[a, b, g] = k3 * n[b]
        end
    end
    return _hbie_contract(ctes, nξ)
end

"""MATLAB `CalcFtes`: 1/ρ part of `W_ijk`."""
function _hbie_Fctes(nξ, n, ν)
    k1 = 1 / (4 * π)
    k2 = 1 / (2 * π)
    DR = (-n[2], n[1])
    ctes = zeros(3, 2, 3)
    @inbounds for a in 1:3, b in 1:2, g in 1:3
        if a < 3 && g < 3
            ctes[a, b, g] = k1 * ((1 - ν) * ((b == g) * DR[a] + (a == g) * DR[b] -
                                             (a == b) * DR[g]) +
                                  2 * (1 + ν) * DR[a] * DR[b] * DR[g])
        elseif a == 3 && g == 3
            ctes[a, b, g] = k2 * DR[b]
        end
    end
    return _hbie_contract(ctes, nξ)
end

function _kronI3(N)
    nN = length(N)
    P = zeros(3, 3nN)
    @inbounds for j in 1:nN
        P[1, 3j - 2] = N[j]
        P[2, 3j - 1] = N[j]
        P[3, 3j] = N[j]
    end
    return P
end

# =============================================================================
# Mesh: reuse in-plane centre-crack Gmsh, lift to 3-DOF FSDT
# =============================================================================

"""Lift a 2-D dual `BEMdata` (twins already prepared) to `FSDTMesh`.

Reissner: 3 DOF, edge moment on `My`. Unsymmetric Hsu–Hwu: 5 DOF, edge
moment on `Hy` (EABE 156 `T*`).
"""
function fsdt_mesh_from_dual_dad(dad, props::AbstractFSDTProps; Mo=0.0, H=nothing)
    n = dad.n
    ndofn = _ndofn(props)
    BC = ones(Int, ndofn * n)
    BV = zeros(ndofn * n)
    eq = copy(dad.eq_type)
    twin = copy(dad.twin)
    Hy = H === nothing ? maximum(abs(p[2]) for p in dad.Nodes) : float(H)
    my_dof = ndofn == 5 ? 4 : 2
    @inbounds for i in 1:n
        eq[i] == 1 || continue
        p = dad.Nodes[i]
        ny = dad.Normal[i][2]
        if abs(abs(p[2]) - Hy) < 1e-6 * max(Hy, 1.0)
            BV[ndofn * (i - 1) + my_dof] = Mo * ny
        end
    end
    return FSDTMesh(dad.elements, dad.element_type, collect(Float64, dad.elem_weight),
        collect(Point2D, dad.Nodes), collect(Point2D, dad.Normal),
        BC, BV, props; eq_type=eq, twin=twin)
end

"""Apply far-field `M_y = Mo` on `y=±H` (`Myn = Mo ny`)."""
function _fsdt_apply_Mo!(dad::BEMdata{<:AbstractFSDT}, Mo, Hy)
    eq = dad.eq_type
    nd = n_dof(dad)
    my_dof = nd == 5 ? 4 : 2
    @inbounds for i in 1:dad.n
        eq[i] == 1 || continue
        p = dad.Nodes[i]
        ny = dad.Normal[i][2]
        if abs(abs(p[2]) - Hy) < 1e-6 * max(Hy, 1.0)
            dad.BV[nd * (i - 1) + my_dof] = Mo * ny
            dad.BC[nd * (i - 1) + my_dof] = 1
        end
    end
    return dad
end

"""
    build_rect_fsdt_crack(; W, H, a, α=0, props, Mo=0, ndiv_b=8, ndiv_h=8, ndiv_crack=16)

Rectangle `[-W,W]×[-H,H]` with a centre crack of half-length `a` at angle
`α` in the Gmsh file ([`quadrado_fsdt`](@ref) type-5 twins, 3- or 5-DOF).
[`formatdata`](@ref) → `BEMdata{<:AbstractFSDT}`. Face A is CBIE, face B
HBIE. `Mo` is the edge moment on y=±H (`M_y n_y` / Hsu `Hy`).
"""
function build_rect_fsdt_crack(; W=1.0, H=2.0, a=0.2, α=0.0, props=FSDTProps(),
        Mo=0.0, ndiv_b=8, ndiv_h=8, ndiv_crack=16, ordem=2, nome="fsdt_center_crack")
    B = parentmodule(@__MODULE__)
    Cr = B.Crack
    nd = _ndofn(props)
    msh = B.quadrado_fsdt(; Lx=2W, Ly=2H, x0=-W, y0=-H,
        ndiv=(ndiv_b, ndiv_h, ndiv_b, ndiv_h), ordem=ordem, bc="FFFF",
        crack=a, crack_α=α, ndiv_crack=ndiv_crack, ndof=nd, nome=nome)
    dad = formatdata(msh, props; tipo=ordem, pontointerno=false)
    Cr.prepare_crack!(dad)
    _fsdt_apply_Mo!(dad, Mo, H)
    pin_fsdt_rbm!(dad; W=W, H=H)
    return dad
end

"""Kinematic pins on the outer boundary (clone of `pin_plate_rbm!`).

Reissner (3 DOF): `ψx, ψy, w`. Unsymmetric (5 DOF): `u, v, βx, βy, w`.
"""
function pin_fsdt_rbm!(dad::BEMdata{<:AbstractFSDT}; W=1.0, H=1.0)
    eq = dad.eq_type
    nd = n_dof(dad)
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
        dad.BC[nd * (inode - 1) + dir] = 0
        dad.BV[nd * (inode - 1) + dir] = val
        return nothing
    end
    if nd == 5
        set_kin!(i_left, 1, 0.0)   # u
        set_kin!(i_bot, 2, 0.0)    # v
        set_kin!(i_left, 3, 0.0)   # βx
        set_kin!(i_bot == 0 ? i_left : i_bot, 5, 0.0)  # w
        if i_bot2 != 0 && i_bot2 != i_bot
            set_kin!(i_bot2, 4, 0.0)  # βy
        elseif i_left != 0
            set_kin!(i_left, 4, 0.0)
        end
    else
        set_kin!(i_left, 1, 0.0)   # ψx
        set_kin!(i_bot, 3, 0.0)    # w
        if i_bot2 != 0 && i_bot2 != i_bot
            set_kin!(i_bot2, 2, 0.0)  # ψy
        elseif i_left != 0
            set_kin!(i_left, 2, 0.0)
        end
    end
    return dad
end

function pin_fsdt_rbm!(mesh::FSDTMesh; W=1.0, H=1.0)
    eq = mesh.eq_type
    nd = mesh.ndofn
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
        mesh.BC[nd * (inode - 1) + dir] = 0
        mesh.BV[nd * (inode - 1) + dir] = val
        return nothing
    end
    if nd == 5
        set_kin!(i_left, 1, 0.0)   # u
        set_kin!(i_bot, 2, 0.0)    # v
        set_kin!(i_left, 3, 0.0)   # βx
        set_kin!(i_bot == 0 ? i_left : i_bot, 5, 0.0)  # w
        if i_bot2 != 0 && i_bot2 != i_bot
            set_kin!(i_bot2, 4, 0.0)  # βy
        elseif i_left != 0
            set_kin!(i_left, 4, 0.0)
        end
    else
        set_kin!(i_left, 1, 0.0)   # ψx
        set_kin!(i_bot, 3, 0.0)    # w
        if i_bot2 != 0 && i_bot2 != i_bot
            set_kin!(i_bot2, 2, 0.0)  # ψy
        elseif i_left != 0
            set_kin!(i_left, 2, 0.0)
        end
    end
    return mesh
end

# =============================================================================
# Assembly (same loop as assemble_dual_elasticity!)
# =============================================================================

@inline function _on_el(el, i)
    @inbounds for k in eachindex(el.index)
        el.index[k] == i && return true
    end
    return false
end
@inline function _on_twin_el(el, i, twin)
    t = twin[i]
    t == 0 && return false
    return _on_el(el, t)
end

function _ξ_on_el(el, i)
    nN = length(el.index)
    qsi, _ = gausslegendre(nN)
    @inbounds for k in 1:nN
        el.index[k] == i && return qsi[k]
    end
    return 0.0
end

"""CBIE / HBIE element integral. Self and twin use Taylor (Dirgantara); else Telles."""
function _dual_el!(He, Ge, Cije, el, pf, nξ, tipo, props, poly, qs, ws;
        on=false, ξ0=0.0, nsub=8)
    D = bending_stiffness(props)
    ν = props.ν
    λ = reissner_lambda(props)
    nN = length(el.index)
    x1, x3 = el.geo[1], el.geo[end]
    Le = norm(x3 - x1)
    if on && tipo == 3
        _hbie_sing!(He, Ge, el, pf, nξ, ξ0, D, ν, λ, poly; nsub=nsub)
        return nothing
    elseif on && tipo != 3
        _cbie_sing!(He, Ge, Cije, el, pf, ξ0, props, poly, qs, ws; nsub=nsub)
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
            ξt, Jt = _telles(qs[ig], eet)
            ξ = 0.5 * (ξa + ξb) + 0.5 * (ξb - ξa) * ξt
            abs(ξ) > 1 + 1e-12 && continue
            pg, J, n̂ = elem_geom(el, ξ)
            R = norm(pg - pf)
            R < 1e-14 && continue
            Nf, _ = shapefun(poly, ξ)
            wJ = J * ws[ig] * Jt * Jsub
            if tipo == 3
                U, P = reissner_hbie_kernels(pg, pf, n̂, nξ, D, ν, λ)
            else
                U, P, C = fsdt_kernels(pg, pf, n̂, D, ν, λ)
                Cije .+= C .* wJ
            end
            @inbounds for a in 1:nN
                Na = Nf[1, a] * wJ
                cols = 3a-2:3a
                He[:, cols] .+= P .* Na
                Ge[:, cols] .+= U .* Na
            end
        end
    end
    return nothing
end

function _cbie_sing!(He, Ge, Cije, el, pf, ξ0, props, poly, qs, ws; nsub=8)
    D = bending_stiffness(props)
    ν = props.ν
    λ = reissner_lambda(props)
    nN = length(el.index)
    x1, x3 = el.geo[1], el.geo[end]
    ndiv = nsub
    dξ = 2 / ndiv
    for k in 1:ndiv
        ξa = -1 + (k - 1) * dξ
        ξb = -1 + k * dξ
        eet = clamp((ξ0 - 0.5 * (ξa + ξb)) / (0.5 * (ξb - ξa)), -0.999, 0.999)
        Jsub = 0.5 * (ξb - ξa)
        for ig in eachindex(qs)
            ξt, Jt = _telles(qs[ig], eet)
            ξ = 0.5 * (ξa + ξb) + 0.5 * (ξb - ξa) * ξt
            abs(ξ) > 1 + 1e-12 && continue
            pg, J, n̂ = elem_geom(el, ξ)
            R = norm(pg - pf)
            R < 1e-14 && continue
            U, P, C = fsdt_kernels(pg, pf, n̂, D, ν, λ)
            Nf, _ = shapefun(poly, ξ)
            wJ = J * ws[ig] * Jt * Jsub
            Cije .+= C .* wJ
            @inbounds for a in 1:nN
                Na = Nf[1, a] * wJ
                cols = 3a-2:3a
                He[:, cols] .+= P .* Na
                Ge[:, cols] .+= U .* Na
            end
        end
    end
    return nothing
end

function _hbie_sing!(He, Ge, el, pf, nξ, ξ0, D, ν, λ, poly; nsub=10)
    nN = length(el.index)
    x1, x3 = el.geo[1], el.geo[end]
    Le = norm(x3 - x1)
    J = Le / 2
    n̂ = Point2D((x3[2] - x1[2]) / Le, -(x3[1] - x1[1]) / Le)
    N0, dN0 = shapefun(poly, ξ0)
    ϕ = _kronI3(view(N0, 1, :))
    dϕ = _kronI3(view(dN0, 1, :))
    Gn = _hbie_Gctes(nξ, n̂, J, D, ν, λ) * ϕ
    Gdn = _hbie_Gctes(nξ, n̂, J, D, ν, λ) * dϕ
    Hn = _hbie_Hctes(nξ, n̂, J, D, ν, λ) * ϕ
    Fn = _hbie_Fctes(nξ, n̂, ν) * ϕ
    qs, ws = gausslegendre(12)
    IH = zeros(3, 3nN)
    IG = zeros(3, 3nN)
    dξ = 2 / nsub
    for k in 1:nsub
        ξa = -1 + (k - 1) * dξ
        ξb = -1 + k * dξ
        Js = 0.5 * (ξb - ξa)
        for ig in eachindex(qs)
            ξt, Jt = _telles(qs[ig], ξ0)
            ξ = 0.5 * (ξa + ξb) + 0.5 * (ξb - ξa) * ξt
            dξe = ξ - ξ0
            abs(dξe) < 1e-14 && continue
            pg, _, nfield = elem_geom(el, ξ)
            R = abs(dξe) * J
            R < 1e-30 && continue
            DRx = (pg[1] - pf[1]) / R
            DRy = (pg[2] - pf[2]) / R
            DRn = nfield[1] * DRx + nfield[2] * DRy
            Nf, _ = shapefun(poly, ξ)
            phie = _kronI3(view(Nf, 1, :))
            W, P = reissner_hbie_kernels(pg, pf, nfield, nξ, D, ν, λ)
            PϕJ = P * phie .* J
            WϕJ = W * phie .* J
            IH .+= (PϕJ .- (Gn .+ Gdn .* dξe) ./ (dξe * dξe) .- Hn .* log(abs(dξe))) .*
                   (ws[ig] * Jt * Js)
            IG .+= ((WϕJ .* dξe .- Fn) ./ dξe) .* (ws[ig] * Jt * Js)
        end
    end
    IntT1 = log(abs((1 - ξ0) / (1 + ξ0)))
    IntT2 = -1 / (1 + ξ0) - 1 / (1 - ξ0)
    IntT3 = log(abs((1 - ξ0) * (1 + ξ0))) - ξ0 * log(abs((1 - ξ0) / (1 + ξ0))) - 2
    He .+= IH .+ Gn .* IntT2 .+ Gdn .* IntT1 .+ Hn .* IntT3
    Ge .+= IG .+ Fn .* IntT1
    return nothing
end

"""
    assemble_fsdt_dual!(mesh; npg=10, nsub=8)

Mixed CBIE/HBIE on a mesh from [`build_rect_fsdt_crack`](@ref). Layout matches
[`assemble_dual_elasticity!`](@ref): outer+face A displacement BIE, face B
hypersingular traction BIE. Reissner: Vander Weeën, self/twin Taylor.
Unsymmetric Hsu–Hwu: EABE 156 `T*` complete solutions
([`assemble_unsym_fsdt_dual!`](@ref)).
"""
function assemble_fsdt_dual!(mesh::FSDTMesh; npg::Int=10, nsub::Int=8,
        singular::Symbol=:guiggiani, ninterp::Int=12, scale_hbie::Bool=true)
    mesh.props isa UnsymFSDTProps &&
        return assemble_unsym_fsdt_dual!(mesh; npg=npg, nsub=nsub,
            singular=singular, ninterp=ninterp, scale_hbie=scale_hbie)
    mesh.props isa FSDTProps ||
        error("assemble_fsdt_dual!: FSDTProps or UnsymFSDTProps")
    n = _n(mesh)
    ndof = 3n
    H = zeros(ndof, ndof)
    G = zeros(ndof, ndof)
    q = zeros(ndof)
    eq = mesh.eq_type
    twin = mesh.twin
    props = mesh.props
    poly = mesh.element_type
    qs, ws = gausslegendre(npg)
    @showprogress "FSDT dual H,G" for i in 1:n
        pf = mesh.nodes[i]
        nξ = mesh.Normal[i]
        rows = 3i-2:3i
        tipo = eq[i]
        Csum = zeros(3, 3)
        for el in mesh.elements
            nN = length(el.index)
            He = zeros(3, 3nN)
            Ge = zeros(3, 3nN)
            Cije = zeros(3, 3)
            on = _on_el(el, i) || _on_twin_el(el, i, twin)
            ξ0 = 0.0
            if on
                src = _on_el(el, i) ? i : twin[i]
                ξ0 = _ξ_on_el(el, src)
            end
            _dual_el!(He, Ge, Cije, el, pf, nξ, tipo, props, poly, qs, ws;
                on=on, ξ0=ξ0, nsub=nsub)
            for a in 1:nN
                ja = el.index[a]
                cols = 3ja-2:3ja
                H[rows, cols] .+= He[:, 3a-2:3a]
                G[rows, cols] .+= Ge[:, 3a-2:3a]
            end
            Csum .+= Cije
        end
        I3 = Matrix{Float64}(I, 3, 3)
        if tipo == 1
            H[rows, rows] .-= Csum
        elseif tipo == 2
            H[rows, rows] .+= 0.5 .* I3
            tw = twin[i]
            if tw != 0
                H[rows, 3tw-2:3tw] .+= 0.5 .* I3
            end
        elseif tipo == 3
            G[rows, rows] .-= 0.5 .* I3
            tw = twin[i]
            if tw != 0
                G[rows, 3tw-2:3tw] .-= 0.5 .* I3
            end
        end
    end
    mesh.H = H
    mesh.G = G
    mesh.q = q
    return mesh
end

function assemble_fsdt_dual!(dad::BEMdata{<:AbstractFSDT}; npg::Int=10, nsub::Int=8,
        threaded::Bool=true, kwargs...)
    n_dof(dad) == 5 && return assemble_unsym_fsdt_dual!(dad; npg=npg, nsub=nsub,
        kwargs...)
    dad.properties isa FSDTProps ||
        error("assemble_fsdt_dual!: FSDTProps (isotropic Reissner)")
    n = dad.n
    ndof = 3n
    H = zeros(ndof, ndof)
    G = zeros(ndof, ndof)
    eq = dad.eq_type
    twin = dad.twin
    props = dad.properties
    poly = dad.element_type
    qs, ws = gausslegendre(npg)
    elems = dad.elements
    _collocation_loop!(threaded, n) do i
        pf = dad.Nodes[i]
        nξ = dad.Normal[i]
        rows = (3i - 2):(3i)
        tipo = eq[i]
        Csum = zeros(3, 3)
        for el in elems
            nN = length(el.index)
            He = zeros(3, 3nN)
            Ge = zeros(3, 3nN)
            Cije = zeros(3, 3)
            on = _on_el(el, i) || _on_twin_el(el, i, twin)
            ξ0 = 0.0
            if on
                src = _on_el(el, i) ? i : twin[i]
                ξ0 = _ξ_on_el(el, src)
            end
            _dual_el!(He, Ge, Cije, el, pf, nξ, tipo, props, poly, qs, ws;
                on=on, ξ0=ξ0, nsub=nsub)
            for a in 1:nN
                ja = el.index[a]
                cols = (3ja - 2):(3ja)
                H[rows, cols] .+= He[:, 3a-2:3a]
                G[rows, cols] .+= Ge[:, 3a-2:3a]
            end
            Csum .+= Cije
        end
        I3 = SMatrix{3,3,Float64}(I)
        if tipo == 1
            H[rows, rows] .-= Csum
        elseif tipo == 2
            H[rows, rows] .+= 0.5 .* I3
            tw = twin[i]
            tw != 0 && (H[rows, (3tw - 2):(3tw)] .+= 0.5 .* I3)
        elseif tipo == 3
            G[rows, rows] .-= 0.5 .* I3
            tw = twin[i]
            tw != 0 && (G[rows, (3tw - 2):(3tw)] .-= 0.5 .* I3)
        end
    end
    set_cache!(dad; H, G, fsdt_q=zeros(ndof))
    return dad
end

# =============================================================================
# CTOD SIFs (Dirgantara / Useche 10.4.1)
# =============================================================================

_fnodes(m::FSDTMesh) = m.nodes
_fnodes(d::BEMdata) = d.Nodes
_feq(m::FSDTMesh) = m.eq_type
_feq(d::BEMdata) = d.eq_type
_ftwin(m::FSDTMesh) = m.twin
_ftwin(d::BEMdata) = d.twin
_fu(m::FSDTMesh) = m.u
_fu(d::BEMdata) = d.u
_fprops(m::FSDTMesh) = m.props
_fprops(d::BEMdata) = d.properties
_fnorm(m::FSDTMesh) = m.Normal
_fnorm(d::BEMdata) = d.Normal
_fels(m::FSDTMesh) = m.elements
_fels(d::BEMdata) = d.elements

"""Crack opening ``u_A - u_B`` at a twinned node (ψx, ψy, w)."""
function crack_opening_fsdt(mesh, inode::Int)
    tw = _ftwin(mesh)[inode]
    tw == 0 && error("node $inode has no twin")
    u = _fu(mesh)
    uA = SVector(u[3inode - 2], u[3inode - 1], u[3inode])
    uB = SVector(u[3tw - 2], u[3tw - 1], u[3tw])
    return uA - uB
end

"""Parent-curve ends `ξ=±1` (geometric tip), not inset collocation."""
function _fsdt_el_ends(el::Element)
    geo = el.geo
    length(geo) >= 2 && return geo[1], geo[end]
    error("_fsdt_el_ends: element has no CAD `geo`")
end

"""Valence-1 endpoints of face-A crack elements (geometric tips)."""
function _fsdt_geometric_tips(mesh)
    eq = _feq(mesh)
    ends = Point2D[]
    counts = Int[]
    for el in _fels(mesh)
        eq[el.index[1]] == 2 || continue
        for p in _fsdt_el_ends(el)
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

"""
    sif_ctod_fsdt(mesh; tip=:right, absK1=true) -> (K1b, K2b, K3b, rA, rB, Le)

Dirgantara CTOD extrapolation (MATLAB `CalcSIF.m`, Useche 10.15–10.16).
`r` is the distance from the **geometric** tip (element `ξ=±1`), not from
the inset collocation node. On the tip element the nearest node is skipped
(MATLAB C); mid (B, `≈Le/2`) and far (A, `≈5Le/6` for Portela) are used.
`tip=:right` is the geometric end with largest `x` (then `y`).
"""
function sif_ctod_fsdt(mesh; tip::Symbol=:right, absK1::Bool=true)
    props = _fprops(mesh)
    E, ν, h = props.E, props.ν, props.h
    eq = _feq(mesh)
    tips = _fsdt_geometric_tips(mesh)
    isempty(tips) && error("sif_ctod_fsdt: no geometric crack tips")
    if tip === :right
        xm = maximum(p[1] for p in tips)
        cands = [p for p in tips if abs(p[1] - xm) < 1e-9]
        tippos = cands[argmax(p[2] for p in cands)]
    else
        xm = minimum(p[1] for p in tips)
        cands = [p for p in tips if abs(p[1] - xm) < 1e-9]
        tippos = cands[argmin(p[2] for p in cands)]
    end
    # tip element on face A: one CAD end is the geometric tip
    eltip = nothing
    for el in _fels(mesh)
        eq[el.index[1]] == 2 || continue
        p0, p1 = _fsdt_el_ends(el)
        if norm(p0 - tippos) < 1e-9 || norm(p1 - tippos) < 1e-9
            eltip = el
            break
        end
    end
    eltip === nothing && error("sif_ctod_fsdt: no face-A element at the tip")
    p0, p1 = _fsdt_el_ends(eltip)
    tdir = norm(p1 - tippos) < norm(p0 - tippos) ? (p0 - p1) : (p1 - p0)
    Le = norm(tdir)
    e1 = tdir / Le                         # element tangent, into the crack
    e2 = Point2D(-e1[2], e1[1])            # 90° CCW (MATLAB dwaa(2))
    idxs = collect(eltip.index)
    nodes = _fnodes(mesh)
    rs = [max(norm(nodes[i] - tippos), 1e-14) for i in idxs]
    perm = sortperm(rs)                    # nearest = C, then B, then A
    length(perm) >= 3 || error("sif_ctod_fsdt: tip element needs 3 nodes")
    iB, rB = idxs[perm[2]], rs[perm[2]]    # MATLAB Lb ≈ Le/2; GL mid → rB = Le/2
    iA, rA = idxs[perm[3]], rs[perm[3]]    # MATLAB La ≈ 5 Le/6; GL far → rA ≈ 0.887 Le
    function Kof(inode, r)
        Δ = crack_opening_fsdt(mesh, inode)
        nA = _fnorm(mesh)[inode]
        dot(nA, e2) < 0 && (Δ = -Δ)
        Δψt = Δ[1] * e1[1] + Δ[2] * e1[2]
        Δψn = Δ[1] * e2[1] + Δ[2] * e2[2]
        Δw = Δ[3]
        Crot = E * h^3 / 48 * sqrt(π / 2)
        Csh = 5 * E * h / (24 * (1 + ν)) * sqrt(π / 2)
        return SVector(Crot * Δψn, Crot * Δψt, Csh * Δw) ./ sqrt(r)
    end
    KA = Kof(iA, rA)
    KB = Kof(iB, rB)
    Ktip = rA / (rA - rB) * (KB - (rB / rA) * KA)
    K1 = absK1 ? abs(Ktip[1]) : Ktip[1]
    return K1, Ktip[2], Ktip[3], rA, rB, Le
end
