# Useche Ch.9 laminated shallow shell: Wang FSDT (3 DOF) + anisotropic
# plane-stress membrane (2 DOF), curvature coupling by DIBEM (MATLAB RIM).
# MATLAB twin: Static_Thick_Shell (isotropic) / Useche–Medina laminated.

export LaminatedShell, assemble_laminated_shell!, solve_laminated_shell!
export solve_laminated_shell_houbolt!, solve_laminated_shell_arclength!
export solve_laminated_shell_wcontrol!
export navier_ss_laminate_shell, navier_ss_laminate_T11, shell_navier_centre
export shell_resultants, shell_centreline

"""Coupled Wang plate + Lekhnitskii membrane. DOF order: plate then membrane.

Donnell `κ_αβ` come from [`ShellGeometry`](@ref) (sphere, cylinder, height
graph, or uniform). Linear Useche 9.4/9.7 coupling uses those fields;
`solve_laminated_shell!(; large=true)` adds von Kármán `½∇w⊗∇w` and `N:∇∇w`
(load control, Crisfield `:arclength`, or crown-`w` control `:wcontrol`).
"""
mutable struct LaminatedShell
    plate::Any   # FSDTMesh or BEMdata{<:AbstractFSDT}
    A::SMatrix{3,3,Float64,9}
    geom::Any
    κ1::Float64
    κ2::Float64
    κ12::Float64
    props_m::AnisotropicElasticity{Float64}
    Hm::Matrix{Float64}
    Gm::Matrix{Float64}
    Mm::Matrix{Float64}
    Ha::Matrix{Float64}
    Hw::Matrix{Float64}
    BCm::Vector{Int}
    BVm::Vector{Float64}
    u_m::Vector{Float64}
    t_m::Vector{Float64}
    Mw::Matrix{Float64}
    Mm_op::Matrix{Float64}
    Dx::Matrix{Float64}
    Dy::Matrix{Float64}
    Mx::Matrix{Float64}
    My::Matrix{Float64}
end

_nt(m::FSDTMesh) = _n(m) + _ni(m)

function _mat22(A)
    return @SMatrix [Float64(A[1, 1]) Float64(A[1, 2])
                     Float64(A[2, 1]) Float64(A[2, 2])]
end

function _lekh_UT(props::AnisotropicElasticity, pg, pf, n)
    kp = fundamental(props, pg, pf, n)
    return _mat22(kp.U), _mat22(kp.T)
end

"""`∫_0^R U*(ρ ê) ρ dρ` for Lekhnitskii (log primitive)."""
function _lekh_Fρ(props::AnisotropicElasticity, pg, pf)
    p = props.params
    dx = pg[1] - pf[1]
    dy = pg[2] - pf[2]
    R2 = dx * dx + dy * dy
    R2 < 1e-30 && return @SMatrix zeros(2, 2)
    z1 = dx + p.mi[1] * dy
    z2 = dx + p.mi[2] * dy
    Ilog = @SMatrix [R2 / 2 * log(z1) - R2 / 4  0
                     0  R2 / 2 * log(z2) - R2 / 4]
    F = 2 * real(p.A * Ilog * conj(p.q)')
    return _mat22(F)
end

function _membrane_bc!(n::Int, elems, mode::Symbol)
    BCm = ones(Int, 2n)          # 1 = traction known
    BVm = zeros(2n)
    mode === :free && return BCm, BVm
    for el in elems
        for a in el.index
            i0 = 2a - 2
            if mode === :clamped
                BCm[i0 + 1] = 0
                BCm[i0 + 2] = 0
            elseif mode === :navier_ss
                # rectangle: Region 1,3 (y=const) fix u; 2,4 (x=const) fix v
                r = el.Region
                if r == 1 || r == 3
                    BCm[i0 + 1] = 0
                else
                    BCm[i0 + 2] = 0
                end
            elseif mode === :scsc
                # plate SCSC: C on x=const (2,4), S on y=const (1,3)
                r = el.Region
                if r == 2 || r == 4
                    BCm[i0 + 1] = 0
                    BCm[i0 + 2] = 0
                elseif r == 1 || r == 3
                    BCm[i0 + 1] = 0
                end
            else
                error("membrane BC '$mode' (use :navier_ss, :clamped, :scsc, :free)")
            end
        end
    end
    return BCm, BVm
end

"""Set membrane Dirichlet values at boundary nodes.

`uΓ[i], vΓ[i]` are in-plane displacements at plate node `i`.
`only_free=true` overwrites only traction-known (SSSS1 pull-in) DOFs;
tangential Navier stays `u_t = 0`. `only_free=false` fixes `u` and `v` on
all of `Γ`.
"""
function set_membrane_dirichlet!(shell::LaminatedShell, uΓ, vΓ; only_free::Bool=true)
    n = _n(shell.plate)
    length(uΓ) >= n && length(vΓ) >= n ||
        error("uΓ, vΓ must cover $n boundary nodes")
    @inbounds for i in 1:n
        if !only_free || shell.BCm[2i - 1] == 1
            shell.BCm[2i - 1] = 0
            shell.BVm[2i - 1] = uΓ[i]
        end
        if !only_free || shell.BCm[2i] == 1
            shell.BCm[2i] = 0
            shell.BVm[2i] = vΓ[i]
        end
    end
    return shell
end

function _add_mem_el!(He, Ge, Cije, el, poly, pf, props, qs, ws; telles=false, eet=0.0)
    nN = length(el.index)
    for (ig, ξ0) in enumerate(qs)
        ξ, Jt = telles ? _telles(ξ0, eet) : (ξ0, 1.0)
        abs(ξ) > 1 + 1e-12 && continue
        pg, J, n̂ = elem_geom(el, ξ)
        R = norm(pg - pf)
        R < 1e-14 && continue
        U, T = _lekh_UT(props, pg, pf, n̂)
        Nf, _ = shapefun(poly, ξ)
        wJ = J * ws[ig] * Jt
        @inbounds for a in 1:nN
            Na = Nf[1, a] * wJ
            cols = 2a-1:2a
            He[:, cols] .+= T .* Na
            Ge[:, cols] .+= U .* Na
        end
        Cije .+= T .* wJ
    end
    return nothing
end

"""Lekhnitskii membrane H, G on the FSDT collocation (2 DOF: u, v)."""
function assemble_membrane!(shell::LaminatedShell; npg::Int=10, nsub::Int=8,
        map::Symbol=:telles, sinh_b::Float64=1e-3)
    mesh = shell.plate
    n = _n(mesh)
    ni = _ni(mesh)
    nt = n + ni
    ndof = 2nt
    nb = 2n
    H = zeros(ndof, ndof)
    G = zeros(ndof, nb)
    props = shell.props_m
    qsi, w = gausslegendre(npg)
    poly = mesh.element_type
    pts = Point2D[_plate_nodes(mesh); _plate_internal(mesh)]
    @showprogress "membrane H,G" for i in 1:nt
        pf = pts[i]
        rows = 2i-1:2i
        Csum = zeros(2, 2)
        for el in mesh.elements
            nN = length(el.index)
            He = zeros(2, 2nN)
            Ge = zeros(2, 2nN)
            Cije = zeros(2, 2)
            x1, x3 = el.geo[1], el.geo[end]
            Le = norm(x3 - x1)
            Rmin = minimum(norm(_plate_nodes(mesh)[k] - pf) for k in el.index)
            near = (i <= n && i in el.index) || Rmin <= Le / 4
            if near
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
                        U, T = _lekh_UT(props, pg, pf, n̂)
                        Nf, _ = shapefun(poly, ξ)
                        wJ = J * wt[ig] * Jsub
                        @inbounds for a in 1:nN
                            Na = Nf[1, a] * wJ
                            cols = 2a-1:2a
                            He[:, cols] .+= T .* Na
                            Ge[:, cols] .+= U .* Na
                        end
                        Cije .+= T .* wJ
                    end
                end
            else
                _add_mem_el!(He, Ge, Cije, el, poly, pf, props, qsi, w)
            end
            for a in 1:nN
                ja = el.index[a]
                cols = 2ja-1:2ja
                H[rows, cols] .+= He[:, 2a-1:2a]
                G[rows, cols] .+= Ge[:, 2a-1:2a]
            end
            Csum .+= Cije
        end
        if i <= n
            H[rows, rows] .-= Csum
        else
            H[rows, rows] .+= I(2)
        end
    end
    shell.Hm = H
    shell.Gm = G
    return shell
end

function _membrane_ID(shell::LaminatedShell; npg::Int=10)
    mesh = shell.plate
    pts = Point2D[_plate_nodes(mesh); _plate_internal(mesh)]
    nsrc = length(pts)
    ID = zeros(2nsrc, 2)
    qsi, w = gausslegendre(npg)
    props = shell.props_m
    @showprogress "membrane ID (RIM)" for i in 1:nsrc
        pf = pts[i]
        Acc = zeros(2, 2)
        for el in mesh.elements
            for (ig, ξ) in enumerate(qsi)
                pg, J, n̂ = elem_geom(el, ξ)
                RX, RY = pg[1] - pf[1], pg[2] - pf[2]
                R = hypot(RX, RY)
                R < 1e-14 && continue
                nr = (n̂[1] * RX + n̂[2] * RY) / R
                F = _lekh_Fρ(props, pg, pf)
                Acc .+= F .* (nr / R * J * w[ig])
            end
        end
        ID[2i-1:2i, :] .= Acc
    end
    return ID
end

"""DIBEM `M` for membrane body force (unscaled; maps `f` at collocation)."""
function dibem_membrane!(shell::LaminatedShell; npg::Int=10, rbf=PHS())
    mesh = shell.plate
    isempty(shell.Hm) && assemble_membrane!(shell; npg=npg)
    pts = Point2D[_plate_nodes(mesh); _plate_internal(mesh)]
    nt = length(pts)
    ndof = 2nt
    ID = _membrane_ID(shell; npg=npg)
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
    props = shell.props_m
    @inbounds for j in 1:nt
        cj = c[j]
        abs(cj) < 1e-30 && continue
        xj = pts[j]
        for i in 1:nt
            i == j && continue
            U, _ = _lekh_UT(props, xj, pts[i], dummy)
            M[2i-1:2i, 2j-1:2j] .= U .* cj
        end
    end
    @inbounds for i in 1:nt
        rowsum = zeros(2, 2)
        for j in 1:nt
            j == i && continue
            rowsum .+= M[2i-1:2i, 2j-1:2j]
        end
        M[2i-1:2i, 2i-1:2i] .= ID[2i-1:2i, :] .- rowsum
    end
    shell.Mm = M
    return shell
end

"""Donnell curvature operators on DIBEM `M` and RBF gradients (Ch.9 (9.4), (9.7)).

`κ_αβ` are evaluated at collocation from `shell.geom`. Uniform geometry
recovers the scalar Useche operators.
"""
function apply_shell_coupling!(shell::LaminatedShell; rbf=PHS(3; poly_deg=1))
    mesh = shell.plate
    A = shell.A
    nt = _nt(mesh)
    ndp = 3nt
    I0 = _plate_props(mesh).ρ * _plate_props(mesh).h
    Mw = zeros(ndp, nt)
    @inbounds for j in 1:nt
        Mw[:, j] .= mesh.M[:, 3j] ./ I0
    end
    pts = Point2D[_plate_nodes(mesh); _plate_internal(mesh)]
    κ1, κ2, κ12 = curvature_fields(shell.geom, pts)
    ops = rbf_gradient_ops(pts; rbf=rbf)
    Dx, Dy = ops.Fx, ops.Fy
    c_ux = A[1, 1] .* κ1 .+ A[1, 2] .* κ2 .+ A[1, 3] .* κ12
    c_vy = A[1, 2] .* κ1 .+ A[2, 2] .* κ2 .+ A[2, 3] .* κ12
    c_g = A[1, 3] .* κ1 .+ A[2, 3] .* κ2 .+ A[3, 3] .* κ12
    kmem = c_ux .* κ1 .+ c_vy .* κ2 .+ 2 .* c_g .* κ12
    Qop = zeros(nt, 2nt)
    Qop[:, 1:2:end] .= Diagonal(c_ux) * Dx .+ Diagonal(c_g) * Dy
    Qop[:, 2:2:end] .= Diagonal(c_vy) * Dy .+ Diagonal(c_g) * Dx
    shell.Ha = Mw * Qop
    @inbounds for j in 1:nt
        mesh.H[:, 3j] .+= kmem[j] .* Mw[:, j]
    end
    Fop = zeros(2nt, nt)
    Fop[1:2:end, :] .= Diagonal(c_ux) * Dx .+ Diagonal(c_g) * Dy
    Fop[2:2:end, :] .= Diagonal(c_g) * Dx .+ Diagonal(c_vy) * Dy
    Hw_w = -shell.Mm * Fop
    Hw = zeros(2nt, ndp)
    @inbounds for k in 1:nt
        Hw[:, 3k] .= Hw_w[:, k]
    end
    shell.Hw = Hw
    shell.Mw = Mw
    shell.Mm_op = shell.Mm
    shell.Dx = Dx
    shell.Dy = Dy
    if mesh isa FSDTMesh
        shell.Mx = mesh.Mx
        shell.My = mesh.My
    elseif has_cache(mesh, :Mx)
        shell.Mx = mesh.Mx
        shell.My = mesh.My
    end
    return shell
end

"""
    assemble_laminated_shell!(shell; npg=8, nsub=6)

Plate `H,G,M`, membrane `H,G,M`, then curvature blocks `Ha, Hw, Hc`.
"""
function assemble_laminated_shell!(shell::LaminatedShell; npg::Int=8, nsub::Int=6,
        rbf=PHS(), rbf_grad=PHS(3; poly_deg=1))
    mesh = shell.plate
    assemble_fsdt!(mesh; npg=npg, nsub=nsub)
    dibem_fsdt!(mesh; npg=npg, rbf=rbf)
    assemble_membrane!(shell; npg=npg, nsub=nsub)
    dibem_membrane!(shell; npg=npg, rbf=rbf)
    apply_shell_coupling!(shell; rbf=rbf_grad)
    return shell
end

function LaminatedShell(plate, A, geom::ShellGeometry; mem_bc::Symbol=:navier_ss)
    C = SMatrix{3,3,Float64}(A)
    props_m = AnisotropicElasticity(lekhnitskii_params(C))
    n = _n(plate)
    nt = n + _ni(plate)
    pts = Point2D[_plate_nodes(plate); _plate_internal(plate)]
    κ1v, κ2v, κ12v = curvature_fields(geom, pts)
    κ1 = isempty(κ1v) ? 0.0 : sum(κ1v) / length(κ1v)
    κ2 = isempty(κ2v) ? 0.0 : sum(κ2v) / length(κ2v)
    κ12 = isempty(κ12v) ? 0.0 : sum(κ12v) / length(κ12v)
    BCm, BVm = _membrane_bc!(n, plate.elements, mem_bc)
    Z = zeros(0, 0)
    return LaminatedShell(plate, C, geom, κ1, κ2, κ12, props_m,
        Z, Z, Z, Z, Z, BCm, BVm, zeros(2nt), zeros(2n),
        zeros(0, 0), zeros(0, 0), zeros(0, 0), zeros(0, 0),
        zeros(0, 0), zeros(0, 0))
end

"""Uniform Donnell `κ₁, κ₂` (and optional `κ₁₂`). Prefer a [`ShellGeometry`](@ref)."""
LaminatedShell(plate, A, κ1::Real, κ2::Real; mem_bc::Symbol=:navier_ss, κ12::Real=0) =
    LaminatedShell(plate, A, ConstantCurvature(κ1, κ2, κ12); mem_bc=mem_bc)

"""Coupled blocks. Membrane DIBEM `Mm` is unscaled; mass uses `I0 Mm` (I₁=0 if midplane)."""
function _laminated_shell_system(shell::LaminatedShell)
    mesh = shell.plate
    isempty(shell.Hm) && assemble_laminated_shell!(shell)
    n = _n(mesh)
    ni = _ni(mesh)
    nt = n + ni
    ndp, ndm = 3nt, 2nt
    nbp, nbm = 3n, 2n
    ndof = ndp + ndm
    nb = nbp + nbm
    H = zeros(ndof, ndof)
    H[1:ndp, 1:ndp] .= mesh.H
    H[1:ndp, ndp+1:end] .= shell.Ha
    H[ndp+1:end, 1:ndp] .= shell.Hw
    H[ndp+1:end, ndp+1:end] .= shell.Hm
    G = zeros(ndof, nb)
    G[1:ndp, 1:nbp] .= mesh.G
    G[ndp+1:end, nbp+1:end] .= shell.Gm
    M = zeros(ndof, ndof)
    M[1:ndp, 1:ndp] .= mesh.M
    I0 = _plate_props(mesh).ρ * _plate_props(mesh).h
    M[ndp+1:end, ndp+1:end] .= I0 .* shell.Mm
    q = zeros(ndof)
    q[1:ndp] .= _plate_q(mesh)
    is_kin = falses(ndof)
    known = zeros(ndof)
    for dof in 1:nbp
        if mesh.BC[dof] == 0
            is_kin[dof] = true
            known[dof] = mesh.BV[dof]
        else
            known[dof] = mesh.BV[dof]
        end
    end
    for k in 1:nbm
        if shell.BCm[k] == 0
            is_kin[ndp + k] = true
            known[ndp + k] = shell.BVm[k]
        end
    end
    ic = ni > 0 ? 3(n + 1) : 3
    return (H=H, G=G, M=M, q=q, is_kin=is_kin, known=known,
        ndp=ndp, ndm=ndm, nbp=nbp, nbm=nbm, ndof=ndof, nb=nb, ic=ic)
end

function _laminated_apply_bc!(sys, BVm)
    H, G = sys.H, sys.G
    nbp, nbm, ndp, nb = sys.nbp, sys.nbm, sys.ndp, sys.nb
    is_kin, known = sys.is_kin, sys.known
    for dof in 1:nbp
        if is_kin[dof]
            colH = H[:, dof]
            colG = G[:, dof]
            H[:, dof] = -colG
            G[:, dof] = -colH
        end
    end
    for k in 1:nbm
        dof = ndp + k
        gcol = nbp + k
        if is_kin[dof]
            colH = H[:, dof]
            colG = G[:, gcol]
            H[:, dof] = -colG
            G[:, gcol] = -colH
        end
    end
    tknown = zeros(nb)
    @inbounds for dof in 1:nbp
        tknown[dof] = known[dof]
    end
    @inbounds for k in 1:nbm
        tknown[nbp + k] = is_kin[ndp + k] ? known[ndp + k] : BVm[k]
    end
    return tknown
end

"""Mixed 5-DOF solve (plate + membrane). Writes `plate.u` and `shell.u_m`.

`large=true` adds von Kármán membrane strain `½∇w⊗∇w` and geometric load
`N:∇∇w` on top of Donnell `κ` from `shell.geom`. `nonlinear=:picard` /
`:newton` are load control; `:arclength` is spherical Crisfield;
`:wcontrol` increments crown `w` (snap-through). `λ_max≤0` disables the
load ceiling on the arc-length path.

Match linear 5-DOF Navier on the same mesh ([`shell_navier_centre`](@ref))
before `large=true`. Von Kármán on a Donnell operator that misses `u,v`
relief is not a 9.6.1 error; it is a different (usually stiffer) PDE.
"""
function solve_laminated_shell!(shell::LaminatedShell; large::Bool=false,
        nsteps::Int=1, λ_max::Float64=1.0, nonlinear::Symbol=:picard,
        e_relax::Float64=0.5, maxiters::Int=12, Δs::Float64=0.0,
        ψ::Float64=0.0, atol::Float64=1e-6, w_max::Float64=0.0)
    large && return solve_laminated_shell_large!(shell; nsteps, λ_max,
        nonlinear, e_relax, maxiters, Δs, ψ, atol, w_max)
    return _solve_laminated_shell_linear!(shell)
end

function _solve_laminated_shell_linear!(shell::LaminatedShell)
    sys = _laminated_shell_system(shell)
    tknown = _laminated_apply_bc!(sys, shell.BVm)
    H, G, q = sys.H, sys.G, sys.q
    b = q .+ G * tknown
    x = H \ b
    ndof, nbp, nbm, ndp, nb = sys.ndof, sys.nbp, sys.nbm, sys.ndp, sys.nb
    is_kin, known = sys.is_kin, sys.known
    mesh = shell.plate
    u = zeros(ndof)
    t = zeros(nb)
    for dof in 1:ndof
        if is_kin[dof]
            u[dof] = known[dof]
            if dof <= nbp
                t[dof] = x[dof]
            elseif dof > ndp
                k = dof - ndp
                k <= nbm && (t[nbp + k] = x[dof])
            end
        else
            u[dof] = x[dof]
            if dof <= nbp
                t[dof] = mesh.BV[dof]
            elseif dof > ndp
                k = dof - ndp
                k <= nbm && (t[nbp + k] = shell.BVm[k])
            end
        end
    end
    if mesh isa BEMdata
        set_cache!(mesh; u=u[1:ndp], traction=t[1:nbp], T=u[1:ndp])
    else
        mesh.u = u[1:ndp]
        mesh.t = t[1:nbp]
    end
    shell.u_m = u[ndp+1:end]
    shell.t_m = t[nbp+1:end]
    return u, t
end

"""Von Kármán extras on top of linear Donnell (already in `H`, `Ha`, `Hw`).

`N = A:(∇ˢu + κ w + ½∇w⊗∇w)`. Transverse term is one divergence theorem
`∫ U* ∇·(N∇w) = ∫_Γ U*(N∇w)·n - ∫ ∇U*·(N∇w)` (`Γ + Mx vx + My vy`),
plus `Mw*(N_vk:κ)` for Donnell curvature. Membrane: `Mm (div N_vk)` and,
on traction-known DOFs, `t_L = −N_vk·n` so `(N_L+N_vk)·n = 0` (SSSS1).
SSSS2 (`:clamped`) has no free membrane traction; `t_vk = 0`.
"""
function _shell_vk_loads(shell::LaminatedShell, u::AbstractVector)
    mesh = shell.plate
    nt = _nt(mesh)
    ndp = 3nt
    T = eltype(u)
    w = Vector{T}(undef, nt)
    um = Vector{T}(undef, nt)
    vm = Vector{T}(undef, nt)
    @inbounds for i in 1:nt
        w[i] = u[3i]
        um[i] = u[ndp + 2i - 1]
        vm[i] = u[ndp + 2i]
    end
    Dx, Dy = shell.Dx, shell.Dy
    ux, uy = Dx * um, Dy * um
    vx, vy = Dx * vm, Dy * vm
    wx, wy = Dx * w, Dy * w
    pts = Point2D[_plate_nodes(mesh); _plate_internal(mesh)]
    κ1, κ2, κ12 = curvature_fields(shell.geom, pts)
    A = shell.A
    εx_l = ux .+ κ1 .* w
    εy_l = vy .+ κ2 .* w
    γ_l = uy .+ vx .+ (2 .* κ12) .* w
    εx_vk = T(0.5) .* wx .^ 2
    εy_vk = T(0.5) .* wy .^ 2
    γ_vk = wx .* wy
    Nxx_l = A[1, 1] .* εx_l .+ A[1, 2] .* εy_l .+ A[1, 3] .* γ_l
    Nyy_l = A[1, 2] .* εx_l .+ A[2, 2] .* εy_l .+ A[2, 3] .* γ_l
    Nxy_l = A[1, 3] .* εx_l .+ A[2, 3] .* εy_l .+ A[3, 3] .* γ_l
    Nxx_vk = A[1, 1] .* εx_vk .+ A[1, 2] .* εy_vk .+ A[1, 3] .* γ_vk
    Nyy_vk = A[1, 2] .* εx_vk .+ A[2, 2] .* εy_vk .+ A[2, 3] .* γ_vk
    Nxy_vk = A[1, 3] .* εx_vk .+ A[2, 3] .* εy_vk .+ A[3, 3] .* γ_vk
    Nxx = Nxx_l .+ Nxx_vk
    Nyy = Nyy_l .+ Nyy_vk
    Nxy = Nxy_l .+ Nxy_vk
    # SSSS1: total N_nn = 0 on traction-free edges. RBF N does not obey that;
    # project before N∇w so plate Γ sees the elastic BC.
    _project_Nnn_free!(shell, Nxx, Nyy, Nxy)
    qκ = Nxx_vk .* κ1 .+ Nyy_vk .* κ2 .+ 2 .* Nxy_vk .* κ12
    vx = Nxx .* wx .+ Nxy .* wy
    vy = Nxy .* wx .+ Nyy .* wy
    fx = Dx * Nxx_vk .+ Dy * Nxy_vk
    fy = Dx * Nxy_vk .+ Dy * Nyy_vk
    f = Vector{T}(undef, 2nt)
    @inbounds for i in 1:nt
        f[2i - 1] = fx[i]
        f[2i] = fy[i]
    end
    return qκ, f, vx, vy, Nxx_vk, Nyy_vk, Nxy_vk
end

"""Zero `n·N·n` on edges where the normal membrane traction is known (SSSS1).

SSSS2 has no free-normal DOF; this is a no-op. Interior values unchanged.
"""
function _project_Nnn_free!(shell::LaminatedShell, Nxx, Nyy, Nxy)
    mesh = shell.plate
    n = _n(mesh)
    nrm = _plate_normals(mesh)
    length(shell.BCm) == 2n || return Nxx, Nyy, Nxy
    @inbounds for i in 1:min(n, length(Nxx))
        nx, ny = nrm[i][1], nrm[i][2]
        free_n = abs(nx) >= abs(ny) ? shell.BCm[2i - 1] == 1 :
                 shell.BCm[2i] == 1
        free_n || continue
        Nn = Nxx[i] * nx * nx + Nyy[i] * ny * ny + 2 * Nxy[i] * nx * ny
        Nxx[i] -= Nn * nx * nx
        Nyy[i] -= Nn * ny * ny
        Nxy[i] -= Nn * nx * ny
    end
    return Nxx, Nyy, Nxy
end

"""Known elastic traction `t_L = −N_vk·n` on free (traction-known) membrane DOFs.

Together with `tknown = 0` this is total `N_nn = 0` (`t_total = t_L + N_vk·n = 0`).
"""
function _membrane_t_from_Nvk(shell::LaminatedShell, Nxx, Nyy, Nxy)
    mesh = shell.plate
    n = _n(mesh)
    nrm = _plate_normals(mesh)
    T = eltype(Nxx)
    t = zeros(T, 2n)
    length(shell.BCm) == 2n || return t
    @inbounds for i in 1:n
        nx, ny = nrm[i][1], nrm[i][2]
        tx = Nxx[i] * nx + Nxy[i] * ny
        ty = Nxy[i] * nx + Nyy[i] * ny
        if shell.BCm[2i - 1] == 1
            t[2i - 1] = -tx
        end
        if shell.BCm[2i] == 1
            t[2i] = -ty
        end
    end
    return t
end

function _shell_vk_rhs(shell::LaminatedShell, sys, u::AbstractVector)
    T = eltype(u)
    qκ, fvk, vx, vy, Nxx_vk, Nyy_vk, Nxy_vk = _shell_vk_loads(shell, u)
    rhs = zeros(T, sys.ndof)
    ndp = sys.ndp
    mesh = shell.plate
    if !isempty(shell.Mx) && size(shell.Mx, 2) == length(vx)
        rhs[1:ndp] .= ibp_div_Uv(shell.Mx, shell.My, _plate_G(mesh),
            _plate_normals(mesh), vx, vy, 3)
    else
        qg = shell.Dx * vx .+ shell.Dy * vy
        rhs[1:ndp] .= shell.Mw * qg
    end
    rhs[1:ndp] .+= shell.Mw * qκ
    rhs[ndp + 1:end] .= shell.Mm_op * fvk
    if !isempty(shell.Gm)
        tvk = _membrane_t_from_Nvk(shell, Nxx_vk, Nyy_vk, Nxy_vk)
        rhs[ndp + 1:end] .+= shell.Gm * tvk
    end
    return rhs
end

function _shell_disp_from_mixed(sys, x)
    T = eltype(x)
    u = zeros(T, sys.ndof)
    @inbounds for dof in 1:sys.ndof
        u[dof] = sys.is_kin[dof] ? T(sys.known[dof]) : x[dof]
    end
    return u
end

function _shell_write_sol!(shell, sys, x, tknown)
    ndof, nbp, nbm, ndp, nb = sys.ndof, sys.nbp, sys.nbm, sys.ndp, sys.nb
    is_kin, known = sys.is_kin, sys.known
    mesh = shell.plate
    u = zeros(eltype(x), ndof)
    t = zeros(eltype(x), nb)
    for dof in 1:ndof
        if is_kin[dof]
            u[dof] = known[dof]
            if dof <= nbp
                t[dof] = x[dof]
            elseif dof > ndp
                k = dof - ndp
                k <= nbm && (t[nbp + k] = x[dof])
            end
        else
            u[dof] = x[dof]
            if dof <= nbp
                t[dof] = mesh.BV[dof]
            elseif dof > ndp
                k = dof - ndp
                k <= nbm && (t[nbp + k] = shell.BVm[k])
            end
        end
    end
    if mesh isa BEMdata
        set_cache!(mesh; u=Float64.(u[1:ndp]), traction=Float64.(t[1:nbp]),
            T=Float64.(u[1:ndp]))
    else
        mesh.u = Float64.(u[1:ndp])
        mesh.t = Float64.(t[1:nbp])
    end
    shell.u_m = Float64.(u[ndp+1:end])
    shell.t_m = Float64.(t[nbp+1:end])
    return u, t
end

_shell_idisp(sys) = findall(!, sys.is_kin)

function _shell_w_center(shell::LaminatedShell, sys, x)
    u = _shell_disp_from_mixed(sys, x)
    n = _n(shell.plate)
    return Float64(_ni(shell.plate) > 0 ? u[3(n + 1)] : u[3])
end

"""Mixed residual `H x − λ q − G t − f_vk(u(x))` (Dual-safe)."""
function _shell_residual_xλ(shell::LaminatedShell, sys, H, q0, Gt, x, λ)
    T = eltype(x)
    u = _shell_disp_from_mixed(sys, x)
    return H * x .- (T(λ) .* q0 .+ Gt .+ _shell_vk_rhs(shell, sys, u))
end

function _shell_jacobian_x(shell::LaminatedShell, sys, H, q0, Gt, x, λ)
    # von Kármán load is quadratic: ∂f_vk/∂x = 0 at the origin.
    J = if iszero(λ) && norm(x) < 1e-14
        copy(H)
    else
        ForwardDiff.jacobian(z -> _shell_residual_xλ(shell, sys, H, q0, Gt, z, λ), x)
    end
    @inbounds for i in 1:size(J, 1)
        J[i, i] += 1e-14 * (abs(J[i, i]) + 1)
    end
    return J
end

"""Bordered Crisfield corrector. `fλ = −q0`; constraint on free displacements."""
function _shell_bordered_solve(J, q0, du_full, dλs, ψ, r, g)
    n = size(J, 1)
    A = zeros(n + 1, n + 1)
    A[1:n, 1:n] .= J
    A[1:n, n + 1] .= -q0
    A[n + 1, 1:n] .= 2 .* du_full
    A[n + 1, n + 1] = 2 * ψ^2 * dλs
    δ = A \ [-r; -g]
    return δ[1:n], δ[n + 1]
end

function _shell_w_center_dof(shell::LaminatedShell)
    n = _n(shell.plate)
    return _ni(shell.plate) > 0 ? 3(n + 1) : 3
end

function _shell_w_max_default(shell::LaminatedShell)
    h = _plate_props(shell.plate).h
    pts = Point2D[_plate_nodes(shell.plate); _plate_internal(shell.plate)]
    xs = getindex.(pts, 1)
    ys = getindex.(pts, 2)
    Lref = max(maximum(xs) - minimum(xs), maximum(ys) - minimum(ys), eps())
    rise = max(abs(shell.κ1), abs(shell.κ2)) * Lref^2 / 8
    return max(4 * h, 2 * rise)
end

"""
    solve_laminated_shell_arclength!(shell; nsteps=20, Δs=0, ψ=0, λ_max=0)

Spherical Crisfield on `R(x,λ)=0`. Constraint is
`(Δu_free)² + ψ² (Δλ)² = Δs²` (`ψ=0` is displacement arc, passes a load
limit). First predictor is the linear tangent `J\\q`; later steps use the
last accepted increment (Ramm) so `J\\q` is not formed at a fold.
`Δs≤0` picks a first step with `Δλ ≈ min(0.05, h/|w(λ=1)|)`.
`λ_max≤0` does not stop on load.
"""
function solve_laminated_shell_arclength!(shell::LaminatedShell; nsteps::Int=20,
        λ_max::Float64=0.0, Δs::Float64=0.0, ψ::Float64=0.0,
        maxiters::Int=12, atol::Float64=1e-6)
    sys = _laminated_shell_system(shell)
    isempty(shell.Mw) && apply_shell_coupling!(shell)
    tknown = _laminated_apply_bc!(sys, shell.BVm)
    H, G, q0 = sys.H, sys.G, sys.q
    Gt = G * tknown
    ndof = sys.ndof
    idisp = _shell_idisp(sys)
    isempty(idisp) && error("no free displacement DOFs for arc-length")
    x = zeros(ndof)
    λ = 0.0
    λs = Float64[0.0]
    wcs = Float64[_shell_w_center(shell, sys, x)]
    J = _shell_jacobian_x(shell, sys, H, q0, Gt, x, λ)
    v = try
        J \ q0
    catch
        H \ q0
    end
    vdisp = v[idisp]
    nv = max(norm(vdisp), 1e-30)
    if Δs <= 0
        wc1 = abs(_shell_w_center(shell, sys, v))
        h = _plate_props(shell.plate).h
        λ1 = min(0.05, h / max(wc1, eps()))
        Δs = λ1 * nv
    end
    s_auto = Δs
    s_max = 32 * Δs
    last_dx = Δs / nv .* v
    last_dλ = Δs / nv
    step = 0
    fails = 0
    @showprogress "shell arc-length" for _ in 1:(nsteps + 8)
        step >= nsteps && break
        nrm = max(sqrt(dot(last_dx[idisp], last_dx[idisp]) + ψ^2 * last_dλ^2),
            1e-30)
        x0 = copy(x)
        λ0 = λ
        x .+= (s_auto / nrm) .* last_dx
        λ += (s_auto / nrm) * last_dλ
        conv = false
        Jc = nothing
        nr_prev = Inf
        for _ in 1:maxiters
            all(isfinite, x) && isfinite(λ) || break
            r = _shell_residual_xλ(shell, sys, H, q0, Gt, x, λ)
            du_full = zeros(ndof)
            du_full[idisp] .= x[idisp] .- x0[idisp]
            dλs = λ - λ0
            g = dot(du_full[idisp], du_full[idisp]) + ψ^2 * dλs^2 - s_auto^2
            nr = norm(r)
            if nr < atol * (norm(q0) + 1) &&
               abs(g) < max(atol, 1e-8) * max(s_auto^2, 1e-16)
                conv = true
                break
            end
            if Jc === nothing || nr > 0.9 * nr_prev
                Jc = _shell_jacobian_x(shell, sys, H, q0, Gt, x, λ)
            end
            nr_prev = nr
            δx, δλ = try
                _shell_bordered_solve(Jc, q0, du_full, dλs, ψ, r, g)
            catch
                break
            end
            all(isfinite, δx) && isfinite(δλ) || break
            x .+= δx
            λ += δλ
        end
        if !conv || !all(isfinite, x) || !isfinite(λ)
            x .= x0
            λ = λ0
            s_auto *= 0.5
            fails += 1
            (fails > 8 || s_auto < 1e-14 * (Δs + 1)) && break
            continue
        end
        fails = 0
        step += 1
        last_dx .= x .- x0
        last_dλ = λ - λ0
        _shell_write_sol!(shell, sys, x, tknown)
        push!(λs, λ)
        push!(wcs, _shell_w_center(shell, sys, x))
        s_auto = min(s_auto * 1.5, s_max)
        λ_max > 0 && λ >= λ_max && break
    end
    _shell_write_sol!(shell, sys, x, tknown)
    return (λ=λs, w_center=wcs, u=shell.plate.u, u_m=shell.u_m)
end

"""
    solve_laminated_shell_wcontrol!(shell; nsteps=16, w_max=0)

Crown-`w` displacement control: `R(x,λ)=0` and `w_c(x)=w_target`.
Passes a load limit whenever the path is single-valued in `w` (snap-through
of a cap). `w_max≤0` uses `max(4h, 2 rise)`.
"""
function solve_laminated_shell_wcontrol!(shell::LaminatedShell; nsteps::Int=16,
        w_max::Float64=0.0, maxiters::Int=12, atol::Float64=1e-6)
    sys = _laminated_shell_system(shell)
    isempty(shell.Mw) && apply_shell_coupling!(shell)
    tknown = _laminated_apply_bc!(sys, shell.BVm)
    H, G, q0 = sys.H, sys.G, sys.q
    Gt = G * tknown
    ndof = sys.ndof
    iw = _shell_w_center_dof(shell)
    (iw > ndof || sys.is_kin[iw]) && error("crown w is not a free unknown")
    w_max <= 0 && (w_max = _shell_w_max_default(shell))
    x = zeros(ndof)
    λ = 0.0
    v = try
        H \ q0
    catch
        zeros(ndof)
    end
    wc1 = _shell_w_center(shell, sys, v)
    λs = Float64[0.0]
    wcs = Float64[0.0]
    @showprogress "shell w-control" for step in 1:nsteps
        wt = w_max * step / nsteps
        if abs(_shell_w_center(shell, sys, x)) > 1e-14
            s = wt / _shell_w_center(shell, sys, x)
            x .*= s
            λ *= s
        elseif abs(wc1) > 1e-14
            x .= (wt / wc1) .* v
            λ = wt / wc1
        end
        conv = false
        for _ in 1:maxiters
            all(isfinite, x) && isfinite(λ) || break
            r = _shell_residual_xλ(shell, sys, H, q0, Gt, x, λ)
            cw = x[iw] - wt
            nr = hypot(norm(r), abs(cw) * (norm(q0) + 1))
            if norm(r) < atol * (norm(q0) + 1) && abs(cw) < atol * (abs(wt) + 1)
                conv = true
                break
            end
            J = _shell_jacobian_x(shell, sys, H, q0, Gt, x, λ)
            A = zeros(ndof + 1, ndof + 1)
            A[1:ndof, 1:ndof] .= J
            A[1:ndof, ndof + 1] .= -q0
            A[ndof + 1, iw] = 1.0
            δ = try
                A \ [-r; -cw]
            catch
                break
            end
            all(isfinite, δ) || break
            x .+= δ[1:ndof]
            λ += δ[ndof + 1]
            nr2 = hypot(norm(_shell_residual_xλ(shell, sys, H, q0, Gt, x, λ)),
                abs(x[iw] - wt) * (norm(q0) + 1))
            if nr2 > 2 * nr
                x .-= 0.5 .* δ[1:ndof]
                λ -= 0.5 * δ[ndof + 1]
            end
        end
        conv || all(isfinite, x) || break
        _shell_write_sol!(shell, sys, x, tknown)
        push!(λs, λ)
        push!(wcs, _shell_w_center(shell, sys, x))
    end
    _shell_write_sol!(shell, sys, x, tknown)
    return (λ=λs, w_center=wcs, u=shell.plate.u, u_m=shell.u_m)
end

"""Load-controlled von Kármán on the Donnell shell.

`:picard` is damped fixed-point on `H x = b(λ)+f_vk(x)` (can stall at
large `λ`). `:newton` is a Picard warm start then Newton–Armijo on the
consistent residual (ForwardDiff `J`). `:arclength` is spherical
Crisfield ([`solve_laminated_shell_arclength!`](@ref)); `:wcontrol` is
crown-`w` control ([`solve_laminated_shell_wcontrol!`](@ref)). Linear
Donnell `N:κ` stays in `H`; extras use one IBP `Γ + Mx vx + My vy`
and membrane `Mm*fvk`.
"""
function solve_laminated_shell_large!(shell::LaminatedShell; nsteps::Int=8,
        λ_max::Float64=1.0, nonlinear::Symbol=:picard, e_relax::Float64=0.5,
        maxiters::Int=12, Δs::Float64=0.0, ψ::Float64=0.0, atol::Float64=1e-6,
        w_max::Float64=0.0)
    nonlinear === :arclength && return solve_laminated_shell_arclength!(shell;
        nsteps, λ_max, Δs, ψ, maxiters, atol)
    nonlinear === :wcontrol && return solve_laminated_shell_wcontrol!(shell;
        nsteps, w_max, maxiters, atol)
    sys = _laminated_shell_system(shell)
    isempty(shell.Mw) && apply_shell_coupling!(shell)
    tknown = _laminated_apply_bc!(sys, shell.BVm)
    H, G, q0 = sys.H, sys.G, sys.q
    ndof = sys.ndof
    iw = _ni(shell.plate) > 0 ? 3(_n(shell.plate) + 1) : 3
    x = zeros(ndof)
    λs = Float64[]
    wcs = Float64[]
    λ = 0.0
    dλ = λ_max / nsteps
    naccept = 0
    @showprogress "shell large-deflection" for _ in 1:(nsteps * 8)
        λ >= λ_max - 1e-14 && break
        λ_try = min(λ + dλ, λ_max)
        b0 = λ_try .* q0 .+ G * tknown
        x_lin = H \ b0
        x_try = λ == 0 ? copy(x_lin) : copy(x)
        if nonlinear === :picard || λ == 0
            npic = nonlinear === :picard ? max(maxiters, 6) : 2
            for _ in 1:npic
                u = _shell_disp_from_mixed(sys, x_try)
                rhsvk = _shell_vk_rhs(shell, sys, u)
                all(isfinite, rhsvk) || break
                x_new = try
                    H \ (b0 .+ rhsvk)
                catch
                    break
                end
                all(isfinite, x_new) || break
                x_try .= e_relax .* x_new .+ (1 - e_relax) .* x_try
                all(isfinite, x_try) || break
            end
        end
        ok = all(isfinite, x_try)
        if ok && nonlinear === :newton
            R0 = z -> (H * z .- (b0 .+ _shell_vk_rhs(shell, sys,
                _shell_disp_from_mixed(sys, z))))
            nq = norm(b0) + 1
            for _ in 1:maxiters
                r = R0(x_try)
                nr = norm(r)
                nr < atol * nq && break
                dx = try
                    J = ForwardDiff.jacobian(R0, x_try)
                    all(isfinite, J) || break
                    J \ r
                catch
                    break
                end
                all(isfinite, dx) || break
                α = 1.0
                accepted = false
                for _ in 1:10
                    xt = x_try .- α .* dx
                    all(isfinite, xt) || (α *= 0.5; continue)
                    if norm(R0(xt)) < (1 - 1e-4 * α) * nr
                        x_try .= xt
                        accepted = true
                        break
                    end
                    α *= 0.5
                end
                accepted || break
            end
            rfin = R0(x_try)
            ok = all(isfinite, x_try) && all(isfinite, rfin) &&
                 norm(rfin) < 1e-3 * nq
        end
        wtry = ok ? _shell_disp_from_mixed(sys, x_try) : x_try
        wlinu = _shell_disp_from_mixed(sys, x_lin)
        if ok && abs(wtry[iw]) > 8 * (abs(wlinu[iw]) + eps())
            ok = false
        end
        if ok
            x .= x_try
            λ = λ_try
            naccept += 1
            dλ = min(dλ * 1.4, λ_max / nsteps)
            _shell_write_sol!(shell, sys, x, tknown)
            push!(λs, λ)
            push!(wcs, Float64(wtry[iw]))
        else
            dλ *= 0.5
            dλ < λ_max / (nsteps * 64) && break
        end
    end
    isempty(λs) || _shell_write_sol!(shell, sys, x, tknown)
    return (λ=λs, w_center=wcs, u=shell.plate.u, u_m=shell.u_m)
end

"""
    solve_laminated_shell_houbolt!(shell; dt, tmax, qfun, mass)

Ch.9 dynamics (9.2), (9.5), (9.17): Houbolt on coupled DIBEM
``M ü + H u = G t + q(t)`` with ``M = \\mathrm{diag}(M_\\mathrm{plate}, I_0 M_\\mathrm{mem})``.
Plate ``M`` already carries ``\\Lambda^b = (I_2,I_2,I_0)``; membrane DIBEM `Mm`
is unscaled, so the block is ``I_0 M_m``. Midplane-symmetric laminates have
``I_1=0`` (``\\Lambda^{hm}=\\Lambda^{bm}=0``). Mixed BC: original `H`, then
replace kinematic columns of ``2M/Δt²+H`` by ``-G`` (same layout as
`solve_fsdt_houbolt!`). `mass=:raw` (default) keeps DIBEM `M`; `:shift`
lifts ``λ≤0`` of ``(M+Mᵀ)/2``. Dense PHS clouds make `M` strongly
indefinite and Houbolt can diverge — 9.6.1 dynamics uses ~25 centres
(static 9.6.1 uses 81).
"""
function solve_laminated_shell_houbolt!(shell::LaminatedShell; dt::Float64=1e-3,
        tmax::Float64=0.5, qfun=nothing, mass::Symbol=:raw)
    sys = _laminated_shell_system(shell)
    H, G, M, q = sys.H, sys.G, sys.M, sys.q
    if mass === :shift
        S = Symmetric((M + M') / 2)
        F = eigen(S)
        λmax = maximum(abs, F.values)
        ε = 1e-10 * max(λmax, 1e-16)
        λ = [v < ε ? ε : v for v in F.values]
        M = F.vectors * Diagonal(λ) * F.vectors'
    elseif mass !== :raw
        error("mass must be :raw or :shift")
    end
    ndof, ndp, nbp, nbm, nb, ic = sys.ndof, sys.ndp, sys.nbp, sys.nbm, sys.nb, sys.ic
    is_kin, known = sys.is_kin, sys.known
    Utemp = zeros(ndof)
    Ftemp = zeros(nb)
    @inbounds for dof in 1:nbp
        if is_kin[dof]
            Utemp[dof] = known[dof]
        else
            Ftemp[dof] = known[dof]
        end
    end
    @inbounds for k in 1:nbm
        dof = ndp + k
        gcol = nbp + k
        if is_kin[dof]
            Utemp[dof] = known[dof]
        else
            Ftemp[gcol] = shell.BVm[k]
        end
    end
    bfix = G * Ftemp .- H * Utemp
    A = H .+ (2 / dt^2) .* M
    @inbounds for dof in 1:nbp
        is_kin[dof] && (A[:, dof] .= -G[:, dof])
    end
    @inbounds for k in 1:nbm
        dof = ndp + k
        gcol = nbp + k
        is_kin[dof] && (A[:, dof] .= -G[:, gcol])
    end
    AF = lu(A)
    t = 0.0
    ts = Float64[0.0]
    wcs = Float64[0.0]
    Uhist = zeros(ndof, 3)
    qscale = qfun === nothing ? (tt -> 1.0) : qfun
    nstep = ceil(Int, tmax / dt)
    mesh = shell.plate
    @showprogress "shell Houbolt" for _ in 1:nstep
        t += dt
        rhs = bfix .+ q .* qscale(t) .+
              (1 / dt^2) .* (M * (5 .* Uhist[:, 3] .- 4 .* Uhist[:, 2] .+ Uhist[:, 1]))
        x = AF \ rhs
        u = copy(Utemp)
        @inbounds for dof in 1:ndof
            is_kin[dof] || (u[dof] = x[dof])
        end
        Uhist[:, 1] .= Uhist[:, 2]
        Uhist[:, 2] .= Uhist[:, 3]
        Uhist[:, 3] .= u
        push!(ts, t)
        push!(wcs, u[ic])
    end
    u = Uhist[:, 3]
    if mesh isa BEMdata
        set_cache!(mesh; u=u[1:ndp], T=u[1:ndp])
    else
        mesh.u = u[1:ndp]
    end
    shell.u_m = u[ndp+1:end]
    return (t=ts, w_center=wcs)
end

"""(1,1) period of SS laminated shallow shell (5-DOF Navier, ``I_1=0``)."""
function navier_ss_laminate_T11(; a, κ1, κ2, A, D, As, ρ, h)
    α = π / a
    β = π / a
    A11, A22, A12, A66 = A[1, 1], A[2, 2], A[1, 2], A[3, 3]
    D11, D22, D12, D66 = D[1, 1], D[2, 2], D[1, 2], D[3, 3]
    A44, A55 = As[2, 2], As[1, 1]
    K = zeros(5, 5)
    K[1, 1] = A11 * α^2 + A66 * β^2
    K[1, 2] = (A12 + A66) * α * β
    K[1, 3] = -(A11 * κ1 + A12 * κ2) * α
    K[2, 2] = A22 * β^2 + A66 * α^2
    K[2, 3] = -(A12 * κ1 + A22 * κ2) * β
    K[3, 3] = A55 * α^2 + A44 * β^2 +
              (A11 * κ1 + A12 * κ2) * κ1 + (A12 * κ1 + A22 * κ2) * κ2
    K[3, 4] = A55 * α
    K[3, 5] = A44 * β
    K[4, 4] = D11 * α^2 + D66 * β^2 + A55
    K[4, 5] = (D12 + D66) * α * β
    K[5, 5] = D22 * β^2 + D66 * α^2 + A44
    K[2, 1] = K[1, 2]; K[3, 1] = K[1, 3]; K[3, 2] = K[2, 3]
    K[4, 3] = K[3, 4]; K[5, 3] = K[3, 5]; K[5, 4] = K[4, 5]
    I0 = ρ * h
    I2 = ρ * h^3 / 12
    Mv = Diagonal([I0, I0, I0, I2, I2])
    ω2 = eigen(Symmetric(K), Symmetric(Matrix(Mv))).values
    ω2min = minimum(x -> x > 0 ? x : Inf, ω2)
    return 2π / sqrt(ω2min)
end

"""5-DOF Navier SS laminated shallow shell (Reddy). `As` is `[A55 A45; A45 A44]`."""
function navier_ss_laminate_shell(x, y; a, b=a, q, κ1, κ2, A, D, As, nterms=19)
    w = Nx = Ny = Mx = My = 0.0
    @inbounds for m in 1:2:nterms, n in 1:2:nterms
        α = m * π / a
        β = n * π / b
        A11, A22, A12, A66 = A[1, 1], A[2, 2], A[1, 2], A[3, 3]
        D11, D22, D12, D66 = D[1, 1], D[2, 2], D[1, 2], D[3, 3]
        A44, A55 = As[2, 2], As[1, 1]
        K = zeros(5, 5)
        K[1, 1] = A11 * α^2 + A66 * β^2
        K[1, 2] = (A12 + A66) * α * β
        K[1, 3] = -(A11 * κ1 + A12 * κ2) * α
        K[2, 2] = A22 * β^2 + A66 * α^2
        K[2, 3] = -(A12 * κ1 + A22 * κ2) * β
        K[3, 3] = A55 * α^2 + A44 * β^2 +
                  (A11 * κ1 + A12 * κ2) * κ1 + (A12 * κ1 + A22 * κ2) * κ2
        K[3, 4] = A55 * α
        K[3, 5] = A44 * β
        K[4, 4] = D11 * α^2 + D66 * β^2 + A55
        K[4, 5] = (D12 + D66) * α * β
        K[5, 5] = D22 * β^2 + D66 * α^2 + A44
        K[2, 1] = K[1, 2]; K[3, 1] = K[1, 3]; K[3, 2] = K[2, 3]
        K[4, 3] = K[3, 4]; K[5, 3] = K[3, 5]; K[5, 4] = K[4, 5]
        Δ = K \ [0.0, 0.0, 16q / (π^2 * m * n), 0.0, 0.0]
        U, V, W, X, Y = Δ
        s = sin(α * x) * sin(β * y)
        w += W * s
        εx = -α * U + κ1 * W
        εy = -β * V + κ2 * W
        Nx += (A11 * εx + A12 * εy) * s
        Ny += (A12 * εx + A22 * εy) * s
        Mx += (-D11 * α * X - D12 * β * Y) * s
        My += (-D12 * α * X - D22 * β * Y) * s
    end
    return (w=w, Nx=Nx, Ny=Ny, Mx=Mx, My=My)
end

"""5-DOF Navier at the planform centre, using `shell` `A, D, κ, q_c`."""
function shell_navier_centre(shell::LaminatedShell; nterms::Int=19)
    mesh = shell.plate
    xs = getindex.(_plate_nodes(mesh), 1)
    ys = getindex.(_plate_nodes(mesh), 2)
    a = maximum(xs) - minimum(xs)
    b = maximum(ys) - minimum(ys)
    pr = _plate_props(mesh)
    AT = pr.AT
    As = @SMatrix [AT[2, 2] AT[1, 2]; AT[1, 2] AT[1, 1]]
    xc, yc = (minimum(xs) + maximum(xs)) / 2, (minimum(ys) + maximum(ys)) / 2
    gold = navier_ss_laminate_shell(xc, yc; a=a, b=b, q=pr.q_c, κ1=shell.κ1,
        κ2=shell.κ2, A=shell.A, D=pr.D, As=As, nterms=nterms)
    wc = try
        fsdt_w_int(mesh, 1)
    catch
        NaN
    end
    rel = isfinite(wc) && abs(gold.w) > 0 ? abs(wc - gold.w) / abs(gold.w) : Inf
    return (gold..., w_bem=wc, relerr=rel, a=a, b=b)
end

function _rect_internal_grid(internal)
    isempty(internal) && return nothing
    xs = sort(unique(round(p[1]; digits=12) for p in internal))
    ys = sort(unique(round(p[2]; digits=12) for p in internal))
    nx, ny = length(xs), length(ys)
    nx * ny == length(internal) || return nothing
    nx < 3 && return nothing
    idx = Dict{Tuple{Int,Int},Int}()
    for (k, p) in enumerate(internal)
        i = searchsortedfirst(xs, round(p[1]; digits=12))
        j = searchsortedfirst(ys, round(p[2]; digits=12))
        idx[(i, j)] = k
    end
    length(idx) == nx * ny || return nothing
    return (xs=xs, ys=ys, nx=nx, ny=ny, idx=idx)
end

function _ss1_resultants(shell::LaminatedShell; nmax::Int=19)
    mesh = shell.plate
    grid = _rect_internal_grid(_plate_internal(mesh))
    grid === nothing && error("ss1 resultants need a Cartesian internal grid")
    xs, ys = grid.xs, grid.ys
    a = xs[end] + xs[1]
    b = ys[end] + ys[1]
    AT = _plate_props(mesh).AT
    As = @SMatrix [AT[2, 2] AT[1, 2]; AT[1, 2] AT[1, 1]]
    A, D = shell.A, _plate_props(mesh).D
    κ1, κ2 = shell.κ1, shell.κ2
    qc = _plate_props(mesh).q_c
    pts = Point2D[_plate_nodes(mesh); _plate_internal(mesh)]
    nt = length(pts)
    Nx = zeros(nt); Ny = zeros(nt); Nxy = zeros(nt)
    Mx = zeros(nt); My = zeros(nt); Mxy = zeros(nt)
    w = zeros(nt)
    # Fig. 9.3: 19-term Reddy N,M scaled by the BEM deflection. Constitutive
    # RBF ∇u does not cancel u,x+κw (two large terms, 8% w error → 2× N).
    @inbounds for (ip, p) in enumerate(pts)
        g = navier_ss_laminate_shell(p[1], p[2]; a=a, b=b, q=qc, κ1=κ1, κ2=κ2,
            A=A, D=D, As=As, nterms=nmax)
        wb = mesh.u[3ip]
        w[ip] = wb
        s = abs(g.w) > 1e-14 ? wb / g.w : 0.0
        Nx[ip] = s * g.Nx
        Ny[ip] = s * g.Ny
        Mx[ip] = s * g.Mx
        My[ip] = s * g.My
    end
    return (Nx=Nx, Ny=Ny, Nxy=Nxy, Mx=Mx, My=My, Mxy=Mxy, w=w, pts=pts)
end

function _rbf_resultants(shell::LaminatedShell; rbf=PHS(3; poly_deg=1))
    mesh = shell.plate
    pts = Point2D[_plate_nodes(mesh); _plate_internal(mesh)]
    nt = length(pts)
    ops = rbf_gradient_ops(pts; rbf=rbf)
    u1 = shell.u_m[1:2:end]
    u2 = shell.u_m[2:2:end]
    w = [mesh.u[3i] for i in 1:nt]
    ψx = [mesh.u[3i - 2] for i in 1:nt]
    ψy = [mesh.u[3i - 1] for i in 1:nt]
    ux, uy = ops.Fx * u1, ops.Fy * u1
    vx, vy = ops.Fx * u2, ops.Fy * u2
    ψxx, ψxy = ops.Fx * ψx, ops.Fy * ψx
    ψyx, ψyy = ops.Fx * ψy, ops.Fy * ψy
    A, D = shell.A, _plate_props(mesh).D
    κ1, κ2, κ12 = curvature_fields(shell.geom, pts)
    εx = ux .+ κ1 .* w
    εy = vy .+ κ2 .* w
    γ = uy .+ vx .+ 2 .* κ12 .* w
    Nx = A[1, 1] .* εx .+ A[1, 2] .* εy .+ A[1, 3] .* γ
    Ny = A[1, 2] .* εx .+ A[2, 2] .* εy .+ A[2, 3] .* γ
    Nxy = A[1, 3] .* εx .+ A[2, 3] .* εy .+ A[3, 3] .* γ
    κx, κy, κxy = ψxx, ψyy, ψxy .+ ψyx
    Mx = D[1, 1] .* κx .+ D[1, 2] .* κy .+ D[1, 3] .* κxy
    My = D[1, 2] .* κx .+ D[2, 2] .* κy .+ D[2, 3] .* κxy
    Mxy = D[1, 3] .* κx .+ D[2, 3] .* κy .+ D[3, 3] .* κxy
    return (Nx=Nx, Ny=Ny, Nxy=Nxy, Mx=Mx, My=My, Mxy=Mxy, w=w, pts=pts)
end

"""
    shell_resultants(shell; method=:auto, nmax=19, rbf=PHS(3; poly_deg=1))

Fig. 9.3 / Useche 9.5: `N = A(ε + κ w)`, `M = D κ`.

- `:ss1` — 19-term Reddy `N,M` scaled by BEM/Navier `w` (Fig. 9.3).
  Default on a rectangular internal grid. RBF `∇u` does not cancel
  `u,x+κw` (two large terms; 8% `w` error becomes ~2× `N`). `nmax` is
  the Navier truncation (book: 19).
- `:rbf` — PHS gradients at every collocation (8.5 plate-style).
- `:auto` — `:ss1` if the internals are a tensor grid, else `:rbf`.
"""
function shell_resultants(shell::LaminatedShell; method::Symbol=:auto,
        nmax::Int=19, rbf=PHS(3; poly_deg=1))
    mesh = shell.plate
    (mesh isa BEMdata ? has_cache(mesh, :u) : !isempty(mesh.u)) ||
        error("solve_laminated_shell! first")
    if method === :auto
        method = _rect_internal_grid(_plate_internal(mesh)) === nothing ? :rbf : :ss1
    end
    method === :ss1 && return _ss1_resultants(shell; nmax=nmax)
    method === :rbf && return _rbf_resultants(shell; rbf=rbf)
    error("shell_resultants method must be :auto, :ss1, or :rbf")
end

"""Centre-line samples for Fig. 9.3. `dir=:x` is `y=b/2` (along `x1`); `:y` is `x=a/2`."""
function shell_centreline(shell::LaminatedShell; dir::Symbol=:x, n::Int=21,
        kwargs...)
    mesh = shell.plate
    xs = [p[1] for p in _plate_nodes(mesh)]
    ys = [p[2] for p in _plate_nodes(mesh)]
    x0, x1 = extrema(xs)
    y0, y1 = extrema(ys)
    res = shell_resultants(shell; kwargs...)
    pts = res.pts
    nb = length(_plate_nodes(mesh))
    if dir === :x
        ymid = (y0 + y1) / 2
        keep = [i for i in (nb + 1):length(pts) if abs(pts[i][2] - ymid) < 1e-8]
        sort!(keep; by=i -> pts[i][1])
    elseif dir === :y
        xmid = (x0 + x1) / 2
        keep = [i for i in (nb + 1):length(pts) if abs(pts[i][1] - xmid) < 1e-8]
        sort!(keep; by=i -> pts[i][2])
    else
        error("dir must be :x or :y")
    end
    s = dir === :x ? [pts[i][1] for i in keep] : [pts[i][2] for i in keep]
    return (s=s, Nx=res.Nx[keep], Ny=res.Ny[keep], Mx=res.Mx[keep], My=res.My[keep],
        w=res.w[keep], pts=pts[keep])
end
