# Multi-region BEM with interface coupling (type-3 BC)
# Inspired by SubregioesPotencialConstante (MATLAB)

export InterfacePair, ContactPair, MultiRegionProblem
export pair_interfaces!, pair_contacts!, assemble_multiregion, solve_multiregion!
export solve_contact_friction!
export CohesiveContactState, solve_cohesive_contact!

"""
BC type codes (Gmsh physical name `"type;value"`):
- `0` Dirichlet (value = prescribed T / u)
- `1` Neumann   (value = prescribed q / t)
- `3` Interface (perfect bond between subregions; value unused)
- `4` Contact candidate (value = friction coefficient μ)
"""
const BC_DIRICHLET = 0
const BC_NEUMANN = 1
const BC_INTERFACE = 3
const BC_CONTACT = 4

"""Paired collocation nodes on a perfect interface (type 3)."""
struct InterfacePair
    reg_a::Int
    node_a::Int
    reg_b::Int
    node_b::Int
end

"""Paired collocation nodes that may enter frictional contact (type 4)."""
mutable struct ContactPair
    reg_a::Int
    node_a::Int
    reg_b::Int
    node_b::Int
    μ::Float64
    gap0::Float64          # initial gap (≥0 open)
    state::Int             # 1=open, 2=slip, 3=stick
end

"""
    MultiRegionProblem

Several [`BEMdata`](@ref) subregions coupled by interfaces and/or contact pairs.
"""
mutable struct MultiRegionProblem{P<:Problem}
    regions::Vector{BEMdata{P}}
    interfaces::Vector{InterfacePair}
    contacts::Vector{ContactPair}
    name::String
end

MultiRegionProblem(regions::Vector{<:BEMdata{P}}; name="multi") where {P} =
    MultiRegionProblem{P}(regions, InterfacePair[], ContactPair[], name)

# =============================================================================
# Geometric pairing
# =============================================================================

"""
    pair_interfaces!(prob; tol=1e-9)

Pair collocation nodes tagged `BC == 3` between different regions by nearest
neighbour (within `tol`). Builds `prob.interfaces`.
"""
function pair_interfaces!(prob::MultiRegionProblem; tol=1e-4)
    regs = prob.regions
    # collect interface nodes per region
    nodes = Vector{Vector{Int}}(undef, length(regs))
    for (r, dad) in enumerate(regs)
        if dad.properties isa Scalar
            nodes[r] = findall(==(BC_INTERFACE), dad.BC)
        else
            # vectorial: interface if any dof is type 3; use node index
            dim = dad.dimension
            nd = Int[]
            for i in 1:dad.n
                if any(dad.BC[dim*(i-1)+k] == BC_INTERFACE for k in 1:dim)
                    push!(nd, i)
                end
            end
            nodes[r] = nd
        end
    end
    pairs = InterfacePair[]
    used = [falses(length(nodes[r])) for r in eachindex(nodes)]
    for ra in 1:length(regs)-1, rb in ra+1:length(regs)
        for (ia, na) in enumerate(nodes[ra])
            used[ra][ia] && continue
            pa = regs[ra].Nodes[na]
            best_d, best_ib = tol, 0
            for (ib, nb) in enumerate(nodes[rb])
                used[rb][ib] && continue
                d = norm(pa - regs[rb].Nodes[nb])
                if d < best_d
                    best_d, best_ib = d, ib
                end
            end
            if best_ib > 0
                push!(pairs, InterfacePair(ra, na, rb, nodes[rb][best_ib]))
                used[ra][ia] = true
                used[rb][best_ib] = true
            end
        end
    end
    prob.interfaces = pairs
    return pairs
end

"""
    pair_contacts!(prob; tol=1e-6)

Pair collocation nodes tagged `BC == 4` between regions. Friction coefficient
`μ` is taken from `BV` of the first node of each pair. Initial gap from
geometry (normal separation).
"""
function pair_contacts!(prob::MultiRegionProblem; tol=1e-4)
    regs = prob.regions
    nodes = Vector{Vector{Int}}(undef, length(regs))
    mus = Vector{Vector{Float64}}(undef, length(regs))
    for (r, dad) in enumerate(regs)
        if dad.properties isa Scalar
            idx = findall(==(BC_CONTACT), dad.BC)
            nodes[r] = idx
            mus[r] = dad.BV[idx]
        else
            dim = dad.dimension
            nd = Int[]; μs = Float64[]
            for i in 1:dad.n
                # contact flagged on normal-ish dof (first component carries μ)
                if dad.BC[dim*(i-1)+1] == BC_CONTACT || dad.BC[dim*(i-1)+dim] == BC_CONTACT
                    push!(nd, i)
                    # μ stored in BV of the contact-typed dof
                    μ = 0.0
                    for k in 1:dim
                        if dad.BC[dim*(i-1)+k] == BC_CONTACT
                            μ = dad.BV[dim*(i-1)+k]
                            break
                        end
                    end
                    push!(μs, μ)
                end
            end
            nodes[r] = nd
            mus[r] = μs
        end
    end
    pairs = ContactPair[]
    used = [falses(length(nodes[r])) for r in eachindex(nodes)]
    for ra in 1:length(regs)-1, rb in ra+1:length(regs)
        for (ia, na) in enumerate(nodes[ra])
            used[ra][ia] && continue
            pa = regs[ra].Nodes[na]
            na_n = regs[ra].Normal[na]
            best_d, best_ib = Inf, 0
            for (ib, nb) in enumerate(nodes[rb])
                used[rb][ib] && continue
                d = norm(pa - regs[rb].Nodes[nb])
                if d < best_d
                    best_d, best_ib = d, ib
                end
            end
            if best_ib > 0 && best_d < tol * 10 || best_d < 1.0  # generous pairing
                nb = nodes[rb][best_ib]
                pb = regs[rb].Nodes[nb]
                # gap along average normal (positive = open)
                n̂ = na_n / (norm(na_n) + eps())
                gap0 = max(0.0, dot(pb - pa, n̂))
                μ = max(mus[ra][ia], mus[rb][best_ib])
                push!(pairs, ContactPair(ra, na, rb, nb, μ, gap0, 1))
                used[ra][ia] = true
                used[rb][best_ib] = true
            end
        end
    end
    prob.contacts = pairs
    return pairs
end

# =============================================================================
# Multi-region assembly + solve (Laplace / scalar)
# =============================================================================

"""
    assemble_multiregion(prob::MultiRegionProblem{<:Laplace})

Assemble each subregion (`H_G_full_direct`) then build a global coupled system
enforcing interface continuity ``T_a = T_b`` and flux balance ``q_a + q_b = 0``.
"""
function assemble_multiregion(prob::MultiRegionProblem{<:Laplace}; npg=16)
    for dad in prob.regions
        H_G_full_direct(dad, npg)
    end
    return nothing
end

"""
    solve_multiregion!(prob::MultiRegionProblem{<:Laplace})

Solve multi-region Laplace with Dirichlet/Neumann exterior BCs and type-3
interface coupling (displacement continuity + traction equilibrium).
"""
function solve_multiregion!(prob::MultiRegionProblem{<:Laplace})
    regs = prob.regions
    isempty(prob.interfaces) && pair_interfaces!(prob)

    # per-region sizes (boundary only for coupling; keep full nt for internals)
    ns = [dad.n for dad in regs]
    nts = [dad.nt for dad in regs]
    offsets = cumsum([0; nts[1:end-1]])
    N = sum(nts)

    # Build block-diagonal A x = b treating interface nodes as Neumann unknowns
    # then add coupling.  Strategy:
    # 1. For each region, form local system with standard BC on non-interface
    #    nodes; interface nodes kept as unknown T with unknown q (both free).
    # 2. Global unknown vector: for each region the standard mixed unknown
    #    (q on Dir, T on Neu/Interface/internal).
    # Simpler monolithic approach used here:
    # unknown = all T on every node (boundary+internal) of every region, plus
    # q on every boundary node.  Then equations:
    #   H T - G q = 0  (per region, nt eqs)
    #   BC Dirichlet: T_i = T̄
    #   BC Neumann:   q_i = q̄
    #   Interface:    T_a - T_b = 0,  q_a + q_b = 0

    # Count DOFs
    # We'll use condensed system: apply known BC, free DOFs only.
    # Free per region: unknown T (Neu + interface + internal) and unknown q (Dir + interface)

    # --- Build global dense system with coupling rows ---
    # Unknown layout per region r:
    #   [q_1..q_n | T_1..T_nt ] but with knowns eliminated → use full and Lagrange

    # Full unknown: x = [x_1; x_2; ...] where x_r is the standard mixed unknown
    # of size nt_r after applyBC-style column swap for Dir/Neu only; interface
    # treated as Neumann (unknown T).

    # Step 1: tag interface nodes as Neumann for local applyBC
    for (r, dad) in enumerate(regs)
        for ip in prob.interfaces
            if ip.reg_a == r
                dad.BC[ip.node_a] = BC_NEUMANN
                dad.BV[ip.node_a] = 0.0   # temporary
            elseif ip.reg_b == r
                dad.BC[ip.node_b] = BC_NEUMANN
                dad.BV[ip.node_b] = 0.0
            end
        end
        # restore type-3 was only for pairing; contact type-4 left as Neumann open
        for cp in prob.contacts
            if cp.reg_a == r
                dad.BC[cp.node_a] = BC_NEUMANN
                dad.BV[cp.node_a] = 0.0
            elseif cp.reg_b == r
                dad.BC[cp.node_b] = BC_NEUMANN
                dad.BV[cp.node_b] = 0.0
            end
        end
        applyBC(dad)
    end

    # Local systems A_r x_r = b_r of size nt_r
    # Interface coupling: we need q at interface nodes as well.
    # Extract from split: after solve, q is recovered.
    # For coupling during solve, expand unknowns to include interface fluxes.

    # Monolithic construction:
    # unknowns per region: T[1:nt] and q[1:n]
    # eqs per region: H T - G q = 0  (nt)
    # + Dirichlet T_i = T̄, Neumann q_i = q̄ for exterior
    # + interface T_a=T_b, q_a+q_b=0

    nT = sum(nts)
    nq = sum(ns)
    n_if = length(prob.interfaces)
    # unknowns: all T (nT) + all q (nq)  — then replace knowns with identity rows
    nu = nT + nq
    A = zeros(nu + 2n_if, nu)   # extra rows for interface (may overdet; use square via replacement)
    # Better: start with nt_total eqs from BIE + BC rows replacing, + interface

    # Rebuild square system of size nT + n_if
    # Standard multi-region: each region contributes nt eqs (BIE with BC applied
    # treating interface as unknown T and free q). Coupling adds 2 eqs per pair
    # but also 2 unknowns (T shared counts once... ).

    # Practical approach matching MATLAB compatibilidade_equilibrio:
    # Block-diagonal A from each region (interface = unknown T, like Neumann).
    # Then ADD rows: T_a - T_b = 0 and columns for interface q with q_a + q_b = 0
    # by introducing interface traction unknowns.

    # --- Block diagonal of local A (size sum nt) ---
    Nloc = sum(nts)
    n_if = length(prob.interfaces)
    # Extra unknowns: interface flux q_if[k] (= q_a = -q_b)
    N = Nloc + n_if
    Ag = zeros(N, N)
    bg = zeros(N)

    off_T = offsets_from(nts)   # start index of T-block (= local x) per region

    for (r, dad) in enumerate(regs)
        o = off_T[r]
        nr = nts[r]
        # Local A x = b with interface as Neumann 0; we'll correct interface columns
        Ag[o+1:o+nr, o+1:o+nr] .= dad.A
        bg[o+1:o+nr] .= dad.b
    end

    # For each interface pair, the local systems currently assume q=0 on those
    # nodes (Neumann 0). True equation: contribution G*q_if on side a and
    # G*(-q_if) on side b.
    # Local unknown x mixes q (Dir columns) and T (Neu columns).
    # Interface nodes are Neumann → unknown is T at position node index in x.
    # We need to add G columns * q_if.

    for (k, ip) in enumerate(prob.interfaces)
        col_q = Nloc + k   # global column for q_if
        # region a: BIE has ... - G[:,na] * q_a, and q_a = q_if
        # After applyBC with Neu, A has H columns for interface T.
        # The residual is A x - b - G[:,na]*q_if = 0 on side a
        #                 A x - b - G[:,nb]*(-q_if) = 0 on side b
        dad_a = regs[ip.reg_a]
        dad_b = regs[ip.reg_b]
        oa, ob = off_T[ip.reg_a], off_T[ip.reg_b]
        Ga = dad_a.G
        Gb = dad_b.G
        na, nb = ip.node_a, ip.node_b
        # Add -G[:,na] to column col_q for rows of region a
        if na <= size(Ga, 2)
            Ag[oa+1:oa+nts[ip.reg_a], col_q] .-= Ga[:, na]
        end
        # Add +G[:,nb] for region b (since q_b = -q_if)
        if nb <= size(Gb, 2)
            Ag[ob+1:ob+nts[ip.reg_b], col_q] .+= Gb[:, nb]
        end
    end

    # Continuity rows: T_a - T_b = 0
    # T is the unknown at Neumann/interface position = node index in local x
    for (k, ip) in enumerate(prob.interfaces)
        row = Nloc + k
        oa, ob = off_T[ip.reg_a], off_T[ip.reg_b]
        # local unknown index for T at interface node = node index (Neu)
        Ag[row, oa + ip.node_a] = 1.0
        Ag[row, ob + ip.node_b] = -1.0
        bg[row] = 0.0
    end

    x = Ag \ bg

    # Split solution back
    for (r, dad) in enumerate(regs)
        o = off_T[r]
        xr = x[o+1:o+nts[r]]
        Tfull = zeros(dad.nt)
        qfull = zeros(dad.n)
        Tfull .= xr
        # recover q from split for Dir/Neu
        split_sol!(dad, Tfull, qfull)
        # interface q from q_if
        for (k, ip) in enumerate(prob.interfaces)
            qif = x[Nloc + k]
            if ip.reg_a == r
                qfull[ip.node_a] = qif
                # T already in Tfull
            elseif ip.reg_b == r
                qfull[ip.node_b] = -qif
            end
        end
        set_cache!(dad; T=Tfull, q=qfull)
    end
    return x
end

function offsets_from(sizes)
    off = zeros(Int, length(sizes))
    s = 0
    for i in eachindex(sizes)
        off[i] = s
        s += sizes[i]
    end
    return off
end

# =============================================================================
# Frictional contact between subregions (type 4) — Laplace frictionless first,
# elasticity with Coulomb stick/slip
# =============================================================================

"""
    solve_contact_friction!(prob; δ=0.0, tol=1e-8, maxiter=50)

Frictional contact iteration for type-4 pairs (from ContatoMultiCorpos2 logic).

States per pair: `1=open`, `2=slip`, `3=stick`.

# Laplace (scalar)
Contact is frictionless unilateral: gap ≥ 0, q ≤ 0 (compression), complementarity.
`μ` ignored for scalar.

# Elasticity
Coulomb: stick ``u_t^a = u_t^b``, ``|t_t| ≤ μ |t_n|``;
slip ``t_t = ±μ t_n``, gap closed in normal direction.
`δ` = additional rigid normal approach.
"""
function solve_contact_friction!(prob::MultiRegionProblem{<:Laplace};
    δ=0.0, tol=1e-8, maxiter=40)
    isempty(prob.contacts) && pair_contacts!(prob)
    # Frictionless unilateral for scalar potential/heat: treat like Signorini
    # on the normal flux.  We iterate active set.
    for dad in prob.regions
        has_cache(dad, :H) || H_G_full_direct(dad, 16)
    end

    for _it in 1:maxiter
        # set BC from contact state
        for cp in prob.contacts
            da, db = prob.regions[cp.reg_a], prob.regions[cp.reg_b]
            if cp.state == 1  # open: Neumann 0 both
                da.BC[cp.node_a] = BC_NEUMANN; da.BV[cp.node_a] = 0.0
                db.BC[cp.node_b] = BC_NEUMANN; db.BV[cp.node_b] = 0.0
            else  # closed: interface-like continuity of T, balance of q
                da.BC[cp.node_a] = BC_INTERFACE
                db.BC[cp.node_b] = BC_INTERFACE
            end
        end
        # rebuild interfaces from closed contacts + type-3
        pair_interfaces!(prob)
        # also add closed contacts as interfaces
        for cp in prob.contacts
            if cp.state != 1
                push!(prob.interfaces, InterfacePair(cp.reg_a, cp.node_a, cp.reg_b, cp.node_b))
            end
        end
        solve_multiregion!(prob)

        changed = false
        for cp in prob.contacts
            da, db = prob.regions[cp.reg_a], prob.regions[cp.reg_b]
            Ta, Tb = da.T[cp.node_a], db.T[cp.node_b]
            qa, qb = da.q[cp.node_a], db.q[cp.node_b]
            # gap estimate: geometric + (Tb - Ta) as relative "penetration" proxy
            # For potential this is not a mechanical gap; use flux sign
            gap = cp.gap0 - δ + (Tb - Ta)  # heuristic
            compression = -(qa)  # positive if flux into a from contact
            if cp.state == 1  # open
                if gap < -tol
                    cp.state = 3; changed = true
                end
            else  # closed
                if compression < -tol  # tension → open
                    cp.state = 1; changed = true
                end
            end
        end
        !changed && break
    end
    return prob
end

function solve_contact_friction!(prob::MultiRegionProblem{<:Elasticity};
    δ=0.0, tol=1e-8, maxiter=40)
    isempty(prob.contacts) && pair_contacts!(prob)
    regs = prob.regions
    for dad in regs
        has_cache(dad, :H) || H_G_full_direct(dad, 16)
    end

    # Local normal/tangent basis from region-a normal
    for _it in 1:maxiter
        # Apply exterior BC + contact conditions into a global elasticity system
        # Simplified: assemble each region with applyBC treating contact nodes
        # according to state, then couple.
        _apply_contact_states_elasticity!(prob, δ)
        for dad in regs
            applyBC(dad)
        end
        # block-diagonal solve then enforce stick/slip traction balance approximately
        for dad in regs
            x = dad.A \ dad.b
            dim = dad.dimension
            u = zeros(dim * dad.n)
            t = zeros(dim * dad.n)
            split_sol!(dad, x, u, t)
            set_cache!(dad; u=u, traction=t, T=u)
        end
        # update states
        changed = _update_contact_states_elasticity!(prob, δ, tol)
        !changed && break
    end
    return prob
end

function _apply_contact_states_elasticity!(prob::MultiRegionProblem{<:Elasticity}, δ)
    for cp in prob.contacts
        da, db = prob.regions[cp.reg_a], prob.regions[cp.reg_b]
        dim = da.dimension
        na, nb = cp.node_a, cp.node_b
        n̂ = da.Normal[na]
        n̂ = n̂ / (norm(n̂) + eps())
        t̂ = Point2D(-n̂[2], n̂[1])   # 2D tangent
        μ = cp.μ
        if cp.state == 1  # open: zero traction both sides
            for k in 1:dim
                da.BC[dim*(na-1)+k] = BC_NEUMANN
                da.BV[dim*(na-1)+k] = 0.0
                db.BC[dim*(nb-1)+k] = BC_NEUMANN
                db.BV[dim*(nb-1)+k] = 0.0
            end
        elseif cp.state == 3  # stick: Dirichlet relative zero (approx fix master)
            # fix body a free; body b: u_b = u_a - gap*n  (master-slave)
            # Without simultaneous solve, pin slave normal/tangential to 0 relative
            # Use Neumann balance iteratively: set slave displacement from master
            ua = has_cache(da, :u) ? da.u : zeros(dim * da.n)
            for k in 1:dim
                db.BC[dim*(nb-1)+k] = BC_DIRICHLET
                # target: ub · e_k ≈ ua · e_k - δ n_k (only normal gap)
            end
            ua_n = dim == 2 ? (ua[2na-1]*n̂[1] + ua[2na]*n̂[2]) : 0.0
            ua_t = dim == 2 ? (ua[2na-1]*t̂[1] + ua[2na]*t̂[2]) : 0.0
            # slave displacement in global coords
            ub_vec = (ua_n - (cp.gap0 - δ)) * n̂ + ua_t * t̂
            if dim == 2
                db.BV[2nb-1] = ub_vec[1]
                db.BV[2nb] = ub_vec[2]
            end
            # master: keep previous BC (usually Neumann from contact pressure unknown)
            # leave master as Neumann 0 initially — pressure emerges from slave reaction
        else  # slip state == 2
            # normal gap closed (Dirichlet normal), tangential Coulomb on traction
            ua = has_cache(da, :u) ? da.u : zeros(dim * da.n)
            ta = has_cache(da, :traction) ? da.traction : zeros(dim * da.n)
            tn = dim == 2 ? (ta[2na-1]*n̂[1] + ta[2na]*n̂[2]) : 0.0
            # direction of slip from previous tangential relative velocity ~ -sign(ut_rel)
            ua_t = dim == 2 ? (ua[2na-1]*t̂[1] + ua[2na]*t̂[2]) : 0.0
            ub = has_cache(db, :u) ? db.u : zeros(dim * db.n)
            ub_t = dim == 2 ? (ub[2nb-1]*t̂[1] + ub[2nb]*t̂[2]) : 0.0
            s = sign(ua_t - ub_t); s = s == 0 ? 1.0 : s
            # slave: un prescribed, tt = -s μ tn_master (applied as Neumann)
            un_target = (dim == 2 ? (ua[2na-1]*n̂[1] + ua[2na]*n̂[2]) : 0.0) - (cp.gap0 - δ)
            # apply as Dirichlet in normal by projecting — simplified: set both components Neumann with Coulomb
            tt = -s * μ * abs(tn)
            t_vec = tn * n̂ + tt * t̂
            if dim == 2
                for (dad, node) in ((da, na), (db, nb))
                    dad.BC[2node-1] = BC_NEUMANN
                    dad.BC[2node] = BC_NEUMANN
                end
                da.BV[2na-1] = t_vec[1]
                da.BV[2na] = t_vec[2]
                db.BV[2nb-1] = -t_vec[1]
                db.BV[2nb] = -t_vec[2]
            end
        end
    end
    return nothing
end

function _update_contact_states_elasticity!(prob, δ, tol)
    changed = false
    for cp in prob.contacts
        da, db = prob.regions[cp.reg_a], prob.regions[cp.reg_b]
        dim = da.dimension
        na, nb = cp.node_a, cp.node_b
        n̂ = da.Normal[na]; n̂ = n̂ / (norm(n̂) + eps())
        t̂ = Point2D(-n̂[2], n̂[1])
        ua, ub = da.u, db.u
        ta, tb = da.traction, db.traction
        if dim != 2
            continue
        end
        gap = cp.gap0 - δ + (ub[2nb-1]-ua[2na-1])*n̂[1] + (ub[2nb]-ua[2na])*n̂[2]
        tn = ta[2na-1]*n̂[1] + ta[2na]*n̂[2]
        tt = ta[2na-1]*t̂[1] + ta[2na]*t̂[2]
        μ = cp.μ
        old = cp.state
        if cp.state == 1
            gap < -tol && (cp.state = 3)
        elseif cp.state == 3
            tn > tol && (cp.state = 1)                 # tension → open
            abs(tt) > μ * abs(tn) + tol && (cp.state = 2)  # slip
        else  # slip
            tn > tol && (cp.state = 1)
            # stick if tangential motion small and within cone
            ut_rel = (ua[2na-1]-ub[2nb-1])*t̂[1] + (ua[2na]-ub[2nb])*t̂[2]
            abs(ut_rel) < tol && abs(tt) < μ * abs(tn) - tol && (cp.state = 3)
        end
        cp.state != old && (changed = true)
    end
    return changed
end

# =============================================================================
# Cohesive-frictional multi-region contact (Alfano–Sacco / Cordeiro states)
# =============================================================================

"""Per-pair cohesive history attached to a multi-region [`ContactPair`](@ref)."""
mutable struct CohesiveContactState
    hist::Any                 # CohesiveHistory (from Crack module)
    tn::Float64
    tt::Float64
    kn::Float64
    kt::Float64
end
CohesiveContactState() = CohesiveContactState(nothing, 0.0, 0.0, 0.0, 0.0)

"""
    solve_cohesive_contact!(prob, law; δ=0.0, nsteps=10, tol=1e-6, kn_pen=1e12)

Multi-region elasticity contact using a cohesive law (e.g. `AlfanoSaccoLaw`,
`BilinearCZM`) on type-4 pairs.

At each load step the rigid approach `δ` is applied proportionally, openings
are evaluated from the current displacements, and the law supplies tractions
that are imposed as Neumann BCs (action–reaction). Active-set iteration
updates open/contact from the law state.

Requires `using BEM.Crack` so that `evaluate_surface!` / `CohesiveHistory` are
available. Returns a `Vector{CohesiveContactState}` aligned with `prob.contacts`.
"""
function solve_cohesive_contact!(
        prob::MultiRegionProblem{<:Elasticity},
        law;
        δ::Float64 = 0.0,
        nsteps::Int = 10,
        tol::Float64 = 1e-6,
        kn_pen::Float64 = 1e12,
        maxiter::Int = 30,
    )
    isempty(prob.contacts) && pair_contacts!(prob)
    regs = prob.regions
    for dad in regs
        has_cache(dad, :H) || H_G_full_direct(dad, 16)
    end

    # lazy history init (avoid hard dep at struct def time)
    states = [CohesiveContactState() for _ in prob.contacts]
    for st in states
        st.hist = Crack.CohesiveHistory()
    end

    for step in 1:nsteps
        δs = δ * step / nsteps
        for it in 1:maxiter
            # 1) evaluate openings from previous u (or geometry if first)
            for (k, cp) in enumerate(prob.contacts)
                da, db = regs[cp.reg_a], regs[cp.reg_b]
                na, nb = cp.node_a, cp.node_b
                n̂ = da.Normal[na]; n̂ = n̂ / (norm(n̂) + eps())
                t̂ = Point2D(-n̂[2], n̂[1])
                if has_cache(da, :u) && has_cache(db, :u)
                    ua = da.u; ub = db.u
                    # relative displacement of b w.r.t a; opening along n̂ of a
                    # gap = gap0 - δ + (ub - ua)·n̂  (positive = open)
                    gap = cp.gap0 - δs +
                        (ub[2nb-1] - ua[2na-1]) * n̂[1] +
                        (ub[2nb] - ua[2na]) * n̂[2]
                    slip = (ub[2nb-1] - ua[2na-1]) * t̂[1] +
                           (ub[2nb] - ua[2na]) * t̂[2]
                    # cohesive δn = -gap (compression negative gap → positive contact)
                    # Our law uses δn>0 tension/opening. Map: δn = gap (open positive)
                    δn = gap
                    δt = slip
                else
                    δn = cp.gap0 - δs
                    δt = 0.0
                end
                tn, tt, kn, kt, stt = Crack.evaluate_surface!(
                    law, δn, δt, states[k].hist; kn_pen = kn_pen,
                )
                states[k].tn = tn; states[k].tt = tt
                states[k].kn = kn; states[k].kt = kt
                # map law state → contact pair state codes
                if stt == Crack.STATE_CONTACT || δn <= 0
                    cp.state = abs(tt) > 0 && abs(tt) >= law_mu(law) * abs(tn) - 1e-12 ? 2 : 3
                    δn > 1e-12 && (cp.state = 1)  # actually open
                elseif stt == Crack.STATE_FAILED && δn > 0
                    cp.state = 1
                else
                    # cohesive tension: still transmit traction (not open Neumann 0)
                    cp.state = 3
                end
            end

            # 2) apply Neumann from cohesive tractions (action-reaction)
            for (k, cp) in enumerate(prob.contacts)
                da, db = regs[cp.reg_a], regs[cp.reg_b]
                na, nb = cp.node_a, cp.node_b
                n̂ = da.Normal[na]; n̂ = n̂ / (norm(n̂) + eps())
                t̂ = Point2D(-n̂[2], n̂[1])
                tn, tt = states[k].tn, states[k].tt
                # traction on body a (local n of a): tn along n̂, tt along t̂
                # Note: law tn>0 is tension on cohesive surface → pulls faces
                ta_vec = tn * n̂ + tt * t̂
                dim = 2
                if cp.state == 1 && states[k].hist.state != Crack.STATE_SOFTENING &&
                   states[k].hist.state != Crack.STATE_UNLOAD
                    # truly open free surface
                    da.BC[2na-1] = BC_NEUMANN; da.BV[2na-1] = 0.0
                    da.BC[2na] = BC_NEUMANN; da.BV[2na] = 0.0
                    db.BC[2nb-1] = BC_NEUMANN; db.BV[2nb-1] = 0.0
                    db.BC[2nb] = BC_NEUMANN; db.BV[2nb] = 0.0
                else
                    da.BC[2na-1] = BC_NEUMANN; da.BV[2na-1] = ta_vec[1]
                    da.BC[2na] = BC_NEUMANN; da.BV[2na] = ta_vec[2]
                    db.BC[2nb-1] = BC_NEUMANN; db.BV[2nb-1] = -ta_vec[1]
                    db.BC[2nb] = BC_NEUMANN; db.BV[2nb] = -ta_vec[2]
                end
            end

            # 3) solve each region
            for dad in regs
                applyBC(dad)
                x = dad.A \ dad.b
                dim = dad.dimension
                u = zeros(dim * dad.n)
                t = zeros(dim * dad.n)
                split_sol!(dad, x, u, t)
                set_cache!(dad; u = u, traction = t, T = u)
            end

            # 4) check residual on openings vs tractions (simple gap change)
            max_pen = 0.0
            for (k, cp) in enumerate(prob.contacts)
                da, db = regs[cp.reg_a], regs[cp.reg_b]
                na, nb = cp.node_a, cp.node_b
                n̂ = da.Normal[na]; n̂ = n̂ / (norm(n̂) + eps())
                ua, ub = da.u, db.u
                gap = cp.gap0 - δs +
                    (ub[2nb-1] - ua[2na-1]) * n̂[1] +
                    (ub[2nb] - ua[2na]) * n̂[2]
                # penetration while claiming open
                if cp.state == 1 && gap < -tol
                    max_pen = max(max_pen, -gap)
                end
            end
            max_pen < tol && break
        end
    end
    return states
end

law_mu(law) = hasproperty(law, :μ) ? law.μ : 0.0
