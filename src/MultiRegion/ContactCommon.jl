# Shared multibody frictional-contact infrastructure (Contato unknown layout)
#
# Unknowns: mixed local BIE DOFs per region, then per pair (t_n¹, t_t¹, t_n², t_t²).
# Gaps: g_n = h - dun (positive open), g_t = dut - ht - ut_lock. Dual: λ = -t.
# Contact frame: common normal n_AB = (n_A E_A − n_B E_B) / ‖·‖ (slave outward).


"""Young's modulus for the common-normal weights; `1` if the region has no `E`."""
_contact_young(dad) = hasproperty(dad.properties, :E) ? float(dad.properties.E) : 1.0

"""
    contact_common_normal(nA, EA, nB, EB) -> n_AB

Stiffness-weighted common contact normal (outward on body A / the slave):

```
n_AB = (n_A E_A − n_B E_B) / ‖n_A E_A − n_B E_B‖
```

`n_A`, `n_B` are each body's outward geometric normals. Equal moduli and
opposed normals recover `n_A`. A much stiffer body B yields `n_AB ≈ −n_B`
(gap along the rigid surface).
"""
function contact_common_normal(nA, EA::Real, nB, EB::Real)
    v = SVector(float(nA[1]), float(nA[2])) * float(EA) -
        SVector(float(nB[1]), float(nB[2])) * float(EB)
    nrm = hypot(v[1], v[2])
    nrm < eps(Float64) && throw(ArgumentError(
        "contact_common_normal: n_A E_A − n_B E_B is zero (check outward signs)"))
    return Point2D(v[1] / nrm, v[2] / nrm)
end

"""
    apply_contact_common_normals!(prob)

Set each contact node's `dad.Normal` to the pair common normal `n_AB` on the
slave and `−n_AB` on the master, and recompute `gap0 = max(0, (x_B−x_A)·n_AB)`.

Call **after** `H_G_full_direct` so the BIE kernels keep the geometric
collocation normal; `transform_HG_local` then puts contact unknowns in the
common frame.
"""
function apply_contact_common_normals!(prob::MultiRegionProblem)
    isempty(prob.contacts) && return prob
    for cp in prob.contacts
        da, db = prob.regions[cp.reg_a], prob.regions[cp.reg_b]
        nAB = contact_common_normal(da.Normal[cp.node_a], _contact_young(da),
            db.Normal[cp.node_b], _contact_young(db))
        da.Normal[cp.node_a] = nAB
        db.Normal[cp.node_b] = Point2D(-nAB[1], -nAB[2])
        pa, pb = da.Nodes[cp.node_a], db.Nodes[cp.node_b]
        cp.gap0 = max(0.0, (pb[1] - pa[1]) * nAB[1] + (pb[2] - pa[2]) * nAB[2])
    end
    return prob
end

"""Per-region data after local transform and exterior BC column exchange."""
struct _RegContactPrep
    dad::BEMdata
    A::Matrix{Float64}          # mixed BIE operator (ndof×ndof)
    b::Vector{Float64}          # known RHS from exterior BC
    Gc_cols::Dict{Int,UnitRange{Int}}  # node → columns of G_local for contact t
    G_local::Matrix{Float64}
    BC_ext::Vector{Int}         # exterior BC in local (contact marked Neumann)
    BV_ext::Vector{Float64}
    is_contact_node::Vector{Bool}
    ndof::Int
    off::Int                    # global offset of this region's mixed DOFs
end

function _default_contact_r(prep)
    E = mean(float(p.dad.properties.E) for p in prep)
    xmin = ymin = Inf
    xmax = ymax = -Inf
    for p in prep
        for pt in p.dad.Nodes
            xmin = min(xmin, pt[1]); xmax = max(xmax, pt[1])
            ymin = min(ymin, pt[2]); ymax = max(ymax, pt[2])
        end
    end
    L = max(xmax - xmin, ymax - ymin, 1e-12)
    r = 10.0 * E / L
    return r, r
end

"""
Local kinematics at contact pair `k` (Contato / verify convention).

Returns `(dun, dut, gn, gt, tn1, tt1, tn2, tt2, R, iu1, iu2, ot)` with
``g_n = h - dun`` (positive = open), ``g_t = dut``.
"""
function _contact_pair_kinematics(prep, cp, h_k, x, k, nx; ht_k::Real=0.0)
    pr1 = prep[cp.reg_a]
    pr2 = prep[cp.reg_b]
    na, nb = cp.node_a, cp.node_b
    ot = nx + 4(k - 1)
    tn1 = x[ot + 1]
    tt1 = x[ot + 2]
    tn2 = x[ot + 3]
    tt2 = x[ot + 4]
    iu1 = pr1.off + (2na - 1)
    iu2 = pr2.off + (2nb - 1)
    un1 = x[iu1]
    ut1 = x[iu1 + 1]
    un2 = x[iu2]
    ut2 = x[iu2 + 1]
    R1 = Matrix(node_rotation2d(pr1.dad.Normal[na]))
    R2 = Matrix(node_rotation2d(pr2.dad.Normal[nb]))
    R = R2 * R1'
    u2_in_1 = R * SVector(un2, ut2)
    dun = un1 - u2_in_1[1]
    dut = ut1 - u2_in_1[2]
    gn = h_k - dun
    # Incremental tangential gap for stick/AC:
    #   gt = (dut - ht) - ut_lock
    # Stick enforces gt = 0 ⇒ dut = ut_lock + ht (Mindlin residual via ut_lock).
    ut_lock = float(cp.ut_lock)
    gt = dut - float(ht_k) - ut_lock
    return (; dun, dut, gn, gt, tn1, tt1, tn2, tt2, R, iu1, iu2, ot,
        un1, ut1, un2, ut2, ht=float(ht_k), ut_lock)
end

"""Normalize optional per-pair tangential rigid shift to `length(h)`."""
function _contact_ht_vec(h, ht)
    n = length(h)
    ht === nothing && return zeros(n)
    ht isa Real && return fill(float(ht), n)
    length(ht) == n || throw(DimensionMismatch("ht length $(length(ht)) ≠ $(n)"))
    return collect(Float64, ht)
end

function _add_gn_row!(J, r, iu1, iu2, R, α)
    # gn = h - (un1 - R[1,:]·u2) ⇒ ∂gn/∂un1=-1, ∂gn/∂u2=R[1,:]
    J[r, iu1]     += α * (-1.0)
    J[r, iu2]     += α * R[1, 1]
    J[r, iu2 + 1] += α * R[1, 2]
    return J
end

function _add_gt_row!(J, r, iu1, iu2, R, α)
    # gt = ut1 - R[2,:]·u2
    J[r, iu1 + 1] += α * 1.0
    J[r, iu2]     += α * (-R[2, 1])
    J[r, iu2 + 1] += α * (-R[2, 2])
    return J
end
function _prepare_regions_contact_local(regs, BC0, BV0)
    # mark contact nodes
    contact_nodes = [falses(d.n) for d in regs]
    # will fill after we know pairs — first pass: any BC type 4
    for (r, dad) in enumerate(regs)
        for i in 1:dad.n
            if BC0[r][2i-1] == BC_CONTACT || BC0[r][2i] == BC_CONTACT
                contact_nodes[r][i] = true
            end
        end
    end

    preps = _RegContactPrep[]
    off = 0
    for (r, dad) in enumerate(regs)
        H, G = dad.H, dad.G
        Hloc, Gloc = transform_HG_local(H, G, dad)
        ndof = 2 * dad.n
        A = Matrix(Hloc[1:ndof, 1:ndof])
        Bmat = Matrix(Gloc[1:ndof, 1:ndof])
        b = zeros(ndof)

        BC = copy(BC0[r])
        BV = copy(BV0[r])
        # exterior → local; force contact nodes to Neumann (t free)
        _exterior_bc_to_local!(BC, BV, dad)
        for i in 1:dad.n
            if contact_nodes[r][i]
                BC[2i-1] = BC_NEUMANN; BV[2i-1] = 0.0
                BC[2i]   = BC_NEUMANN; BV[2i]   = 0.0
            end
        end

        # Column exchange for exterior Dirichlet only (non-contact)
        for dof in 1:ndof
            if BC[dof] == BC_DIRICHLET
                # swap A/B columns: unknown becomes t
                colA = A[:, dof]
                A[:, dof] .= .-Bmat[:, dof]
                Bmat[:, dof] .= .-colA
            end
        end
        # RHS from known values (Dirichlet u or Neumann t on exterior)
        for dof in 1:ndof
            if contact_nodes[r][cld(dof, 2)]
                # contact: t not in known RHS (extra unknown); u free
                continue
            end
            # Known exterior: Neumann t (Bmat=G) or Dirichlet u after swap
            # (Bmat=-H). Both: b += Bmat * BV.
            b .+= Bmat[:, dof] .* BV[dof]
        end

        Gc = Dict{Int,UnitRange{Int}}()
        for i in 1:dad.n
            contact_nodes[r][i] || continue
            Gc[i] = 2i-1:2i
        end

        push!(preps, _RegContactPrep(dad, A, b, Gc, Gloc[1:ndof, 1:ndof],
            BC, BV, contact_nodes[r], ndof, off))
        off += ndof
    end
    return preps
end
function _assemble_contact_system(prep::Vector{_RegContactPrep}, pairs, h, x; ht=nothing)
    ht = _contact_ht_vec(h, ht)
    nx = sum(p.ndof for p in prep)
    np = length(pairs)
    nt = 4 * np
    N = nx + nt
    A = zeros(N, N)
    b = zeros(N)

    # --- BIE blocks (diagonal) + -G_c * t_c columns ---
    for (r, pr) in enumerate(prep)
        o = pr.off
        nd = pr.ndof
        A[o+1:o+nd, o+1:o+nd] .= pr.A
        b[o+1:o+nd] .= pr.b
    end

    # map pair index → traction unknown offset
    # t unknowns layout per pair k (1-based): [tn1, tt1, tn2, tt2]
    for (k, cp) in enumerate(pairs)
        pr1 = prep[cp.reg_a]
        pr2 = prep[cp.reg_b]
        na, nb = cp.node_a, cp.node_b
        ot = nx + 4(k - 1)          # 0-based start of t block
        # G columns for contact tractions contribute -G to BIE of each region
        if haskey(pr1.Gc_cols, na)
            cols = pr1.Gc_cols[na]
            # t1 = (tn1, tt1) multiplies G_local columns of node na
            A[pr1.off+1:pr1.off+pr1.ndof, ot+1:ot+2] .-= pr1.G_local[:, cols]
        end
        if haskey(pr2.Gc_cols, nb)
            cols = pr2.Gc_cols[nb]
            A[pr2.off+1:pr2.off+pr2.ndof, ot+3:ot+4] .-= pr2.G_local[:, cols]
        end
    end

    # --- Contact constraint rows (MATLAB aplica_contato_com_atrito_multicorpos) ---
    for (k, cp) in enumerate(pairs)
        pr1 = prep[cp.reg_a]
        pr2 = prep[cp.reg_b]
        na, nb = cp.node_a, cp.node_b
        # local→global rotations
        R1 = Matrix(node_rotation2d(pr1.dad.Normal[na]))
        R2 = Matrix(node_rotation2d(pr2.dad.Normal[nb]))
        R = R2 * R1'   # maps local-1 related quantities (MATLAB)
        μ = cp.μ
        st = cp.state
        ot = nx + 4(k - 1)
        # mixed unknown indices for u at contact nodes (Neumann → unknown is u)
        # After BC swap: contact nodes are Neumann so unknown DOF is u (column still H)
        iu1 = pr1.off + (2na - 1)
        iu2 = pr2.off + (2nb - 1)
        # rows for this pair's 4 equations
        rows = ot+1:ot+4

        # Contato MATLAB (aplica_contato_com_atrito_multicorpos / _sem_atrito):
        # unknowns ordered (un1,ut1,un2,ut2) and (tn1,tt1,tn2,tt2); R = R2*R1'.
        if st == 1  # open: all tractions zero
            A[rows[1], ot+1] = 1.0
            A[rows[2], ot+2] = 1.0
            A[rows[3], ot+3] = 1.0
            A[rows[4], ot+4] = 1.0
            b[rows] .= 0.0
        elseif abs(μ) < 1e-14 || (abs(st) != 1 && abs(μ) < 1e-14)
            # Contato frictionless closed (aplica_contato_sem_atrito_multicorpos):
            #   un1 - R1·u2 = h,  tn1 + R1·t2 = 0,  tt1 = 0,  tt2 = 0
            A[rows[1], iu1] = 1.0
            A[rows[1], iu2] = -R[1, 1]
            A[rows[1], iu2 + 1] = -R[1, 2]
            b[rows[1]] = h[k]
            A[rows[2], ot+1] = 1.0
            A[rows[2], ot+3] = R[1, 1]
            A[rows[2], ot+4] = R[1, 2]
            b[rows[2]] = 0.0
            A[rows[3], ot+2] = 1.0   # tt1 = 0
            b[rows[3]] = 0.0
            A[rows[4], ot+4] = 1.0   # tt2 = 0
            b[rows[4]] = 0.0
        elseif abs(st) == 2  # slip (Contato tipocontato = ±2)
            sμ = μ * (st >= 0 ? 1.0 : -1.0)  # mi*sign(tipocontato)
            # un1 - R row1 u2 = h
            A[rows[1], iu1] = 1.0
            A[rows[1], iu2] = -R[1, 1]
            A[rows[1], iu2 + 1] = -R[1, 2]
            b[rows[1]] = h[k]
            # tn1 + R row1 t2 = 0
            A[rows[2], ot+1] = 1.0
            A[rows[2], ot+3] = R[1, 1]
            A[rows[2], ot+4] = R[1, 2]
            b[rows[2]] = 0.0
            # tt1 - sμ*tn1 = 0   (Contato: mi*sign, 1, 0, 0 on t)
            A[rows[3], ot+1] = sμ
            A[rows[3], ot+2] = 1.0
            b[rows[3]] = 0.0
            # tt1 + R row2 t2 = 0
            A[rows[4], ot+2] = 1.0
            A[rows[4], ot+3] = R[2, 1]
            A[rows[4], ot+4] = R[2, 2]
            b[rows[4]] = 0.0
        else  # stick / adesão (Contato tipocontato = 3)
            # un1 - R row1 u2 = h
            A[rows[1], iu1] = 1.0
            A[rows[1], iu2] = -R[1, 1]
            A[rows[1], iu2 + 1] = -R[1, 2]
            b[rows[1]] = h[k]
            # ut1 - R row2 u2 = ut_lock + ht
            A[rows[2], iu1 + 1] = 1.0
            A[rows[2], iu2] = -R[2, 1]
            A[rows[2], iu2 + 1] = -R[2, 2]
            b[rows[2]] = cp.ut_lock + ht[k]
            # tn1 + R row1 t2 = 0
            A[rows[3], ot+1] = 1.0
            A[rows[3], ot+3] = R[1, 1]
            A[rows[3], ot+4] = R[1, 2]
            b[rows[3]] = 0.0
            # tt1 + R row2 t2 = 0
            A[rows[4], ot+2] = 1.0
            A[rows[4], ot+3] = R[2, 1]
            A[rows[4], ot+4] = R[2, 2]
            b[rows[4]] = 0.0
        end
    end
    return A, b
end
function _verify_contact_states!(pairs, prep, h, x; ht=nothing, epsc=1e-7)
    ht = _contact_ht_vec(h, ht)
    nx = sum(p.ndof for p in prep)
    for (k, cp) in enumerate(pairs)
        kin = _contact_pair_kinematics(prep, cp, h[k], x, k, nx; ht_k=ht[k])
        tn1, tt1 = kin.tn1, kin.tt1
        μ = cp.μ
        # Contato: sign(0)=0 — do not map 0→1 (that traps over-Coulomb stick).
        sinal_tt = sign(tt1)
        sinal_dut = sign(kin.gt)
        if abs(μ) < 1e-14
            sinal_tt = 1.0
        end

        # Contato verfica_contato_com_atrito_multicorpos:
        #   tn≈0 → open/penetration; tn>0 tension→open;
        #   |tt|>μ|tn| → slip unless sinal_dut*sinal_tt>0 (heal to stick);
        #   else stick.  μ≈0 closed → state 3 + frictionless assemble (tt=0).
        if abs(tn1) <= epsc
            cp.state = kin.dun + epsc > h[k] ? 3 : 1
        elseif tn1 > epsc
            cp.state = 1
        elseif abs(μ) < 1e-14
            cp.state = 3
        elseif abs(tt1) + epsc > μ * abs(tn1)
            if sinal_dut * sinal_tt > 0
                cp.state = 3
            else
                s = sinal_tt == 0 ? (sinal_dut == 0 ? 1.0 : sinal_dut) : sinal_tt
                cp.state = Int(s * 2)
            end
        else
            cp.state = 3
        end
        cp.tn = tn1
        cp.tt = tt1
    end
    return pairs
end

"""
Update incremental stick locks after a load step.

Closed pairs (stick or slip) store ``ut_lock = dut - ht`` so the *next* step's
stick constraint freezes the current relative tangential gap. That is what
retains Mindlin residual shear when bulk ``u_x`` unloads. Open pairs reset
``ut_lock = 0``.
"""
function _update_contact_ut_locks!(pairs, prep, h, x; ht=nothing)
    ht = _contact_ht_vec(h, ht)
    nx = sum(p.ndof for p in prep)
    for (k, cp) in enumerate(pairs)
        kin = _contact_pair_kinematics(prep, cp, h[k], x, k, nx; ht_k=ht[k])
        if abs(cp.state) == 1
            cp.ut_lock = 0.0
        else
            # freeze whatever relative gap the step actually produced
            cp.ut_lock = kin.dut - ht[k]
        end
    end
    return pairs
end

function _scatter_contact_solution!(prob, prep, pairs, x)
    nx = sum(p.ndof for p in prep)
    # per region: recover local u,t then map to global
    for pr in prep
        dad = pr.dad
        nd = pr.ndof
        xr = x[pr.off+1:pr.off+nd]
        u_loc = zeros(nd)
        t_loc = zeros(nd)
        BC = pr.BC_ext
        BV = pr.BV_ext
        for dof in 1:nd
            inode = cld(dof, 2)
            if pr.is_contact_node[inode]
                # unknown is u; t from contact block filled below
                u_loc[dof] = xr[dof]
            elseif BC[dof] == BC_DIRICHLET
                u_loc[dof] = BV[dof]
                t_loc[dof] = xr[dof]   # after swap unknown was t
            else
                t_loc[dof] = BV[dof]
                u_loc[dof] = xr[dof]
            end
        end
        # write contact tractions from global t block
        for (k, cp) in enumerate(pairs)
            ot = nx + 4(k - 1)
            if cp.reg_a == findfirst(p -> p.dad === dad, prep)
                na = cp.node_a
                t_loc[2na-1] = x[ot+1]
                t_loc[2na]   = x[ot+2]
            end
            if cp.reg_b == findfirst(p -> p.dad === dad, prep)
                nb = cp.node_b
                t_loc[2nb-1] = x[ot+3]
                t_loc[2nb]   = x[ot+4]
            end
        end
        u_glb = local_to_global_field(dad, u_loc)
        t_glb = local_to_global_field(dad, t_loc)
        set_cache!(dad; u=u_glb, traction=t_glb, T=u_glb,
                   u_local=u_loc, traction_local=t_loc)
    end
    return prob
end

"""Convert exterior BC/BV arrays from global (x,y) to nodal (n,t)."""
function _exterior_bc_to_local!(BC::Vector{Int}, BV::Vector{Float64}, dad::BEMdata)
    @inbounds for j in 1:dad.n
        d1, d2 = 2j - 1, 2j
        bcx, bcy = BC[d1], BC[d2]
        if bcx == BC_CONTACT || bcy == BC_CONTACT
            BC[d1] = BC_CONTACT
            BC[d2] = BC_CONTACT
            continue
        end
        n̂, t̂ = local_basis2d(dad.Normal[j])
        R = @SMatrix [n̂[1] t̂[1]; n̂[2] t̂[2]]
        vx, vy = BV[d1], BV[d2]
        if bcx == bcy
            vl = R' * SVector(vx, vy)
            BV[d1] = vl[1]
            BV[d2] = vl[2]
            continue
        end
        # Mixed global BCs → local
        dir_x = bcx == BC_DIRICHLET
        dir_y = bcy == BC_DIRICHLET
        # default: both Neumann 0, then overwrite
        BC[d1] = BC_NEUMANN; BV[d1] = 0.0
        BC[d2] = BC_NEUMANN; BV[d2] = 0.0
        ugx = dir_x ? vx : NaN
        ugy = dir_y ? vy : NaN
        tgx = dir_x ? NaN : vx
        tgy = dir_y ? NaN : vy
        # Dirichlet → local axis most aligned with the constrained global axis.
        # On axis-aligned faces: un = n_k * u_k (other u free), etc.
        tg = SVector(isnan(tgx) ? 0.0 : tgx, isnan(tgy) ? 0.0 : tgy)
        if dir_x && !dir_y
            if abs(n̂[1]) >= abs(t̂[1])
                BC[d1] = BC_DIRICHLET
                BV[d1] = n̂[1] * ugx          # un ≈ n₁ uₓ
                BC[d2] = BC_NEUMANN
                BV[d2] = dot(t̂, tg)          # t_t
            else
                BC[d2] = BC_DIRICHLET
                BV[d2] = t̂[1] * ugx
                BC[d1] = BC_NEUMANN
                BV[d1] = dot(n̂, tg)
            end
        elseif dir_y && !dir_x
            if abs(n̂[2]) >= abs(t̂[2])
                BC[d1] = BC_DIRICHLET
                BV[d1] = n̂[2] * ugy
                BC[d2] = BC_NEUMANN
                BV[d2] = dot(t̂, tg)
            else
                BC[d2] = BC_DIRICHLET
                BV[d2] = t̂[2] * ugy
                BC[d1] = BC_NEUMANN
                BV[d1] = dot(n̂, tg)
            end
        end
    end
    return nothing
end

"""
    set_farfield_displacement!(dad; ux=0, uy=0, face=:top|:bottom)

Dirichlet ``(u_x,u_y)`` on the horizontal far face (`:top` = max ``y``).
For fretting, drive bulk slip by ``u_x`` on the top block — do **not** use a
contact-gap tangential shift if residual Mindlin shear must be retained.
"""
function set_farfield_displacement!(dad; ux::Real=0.0, uy::Real=0.0,
        face::Symbol=:top, tol::Real=0.0)
    ys = [pt[2] for pt in dad.Nodes]
    yref = face === :top ? maximum(ys) : minimum(ys)
    thr = tol > 0 ? float(tol) : 1e-9 * max(abs(yref), maximum(abs, ys), 1.0)
    ux = float(ux); uy = float(uy)
    nset = 0
    for i in 1:dad.n
        if abs(dad.Nodes[i][2] - yref) <= thr
            dad.BC[2i - 1] = 0; dad.BV[2i - 1] = ux
            dad.BC[2i]     = 0; dad.BV[2i]     = uy
            nset += 1
        end
    end
    nset == 0 && @warn "set_farfield_displacement!: no nodes on face" face=face yref=yref
    return nset
end

function _contact_friction_setup(prob::MultiRegionProblem{<:Elasticity};
        method::Symbol=:ntn, npg=12, common_normal::Bool=false,
        near_factor::Real=1.5, singular::Symbol=:guiggiani)
    isempty(prob.contacts) && pair_contacts!(prob; method=method)
    regs = prob.regions
    length(regs) >= 2 || error("need ≥2 regions")
    for dad in regs
        dad.dimension == 2 || error("elasticity contact is 2D only")
        has_cache(dad, :H) || H_G_full_direct(dad; npg=npg, near_factor=near_factor,
            singular=singular)
    end
    # BIE kernels stay on the geometric n; contact frame uses n_AB.
    # Contato MATLAB keeps the geometric n (common_normal=false).
    common_normal && apply_contact_common_normals!(prob)
    BC0 = [copy(d.BC) for d in regs]
    BV0 = [copy(d.BV) for d in regs]
    prep = _prepare_regions_contact_local(regs, BC0, BV0)
    pairs = prob.contacts
    isempty(pairs) && (@warn "no contact pairs"; return nothing)
    N = sum(p.ndof for p in prep) + 4 * length(pairs)
    return (; prep, pairs, N)
end

