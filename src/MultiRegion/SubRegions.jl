# Multi-region BEM with interface coupling (type-3 BC)
# Inspired by SubregioesPotencialConstante (MATLAB)

export InterfacePair, ContactPair, MultiRegionProblem
export pair_interfaces!, pair_contacts!, assemble_multiregion, solve_multiregion!
export solve_contact_friction!, solve_contact_friction_stepped!
export alart_curnier
export CohesiveContactState, solve_cohesive_contact!
export solve_multibody_elasticity_contact!
export contact_nodes_elasticity, project_point_to_segment2d

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

"""
Paired collocation contact link (type 4).

- `reg_a` / `node_a` — **slave** region and node (where gap is collated)
- `reg_b` / `node_b` — **master** region and primary node (NTN target, or
  closest master node for NTS)
- `method` — `:ntn` (node-to-node) or `:nts` (node-to-segment)
- `master_nodes`, `ξ`, `N1`, `N2` — linear master segment support for NTS
- `tn`, `tt` — latest normal/tangential traction on the slave (`t · n`, `t · τ`)
"""
mutable struct ContactPair
    reg_a::Int
    node_a::Int
    reg_b::Int
    node_b::Int
    μ::Float64
    gap0::Float64          # initial gap (≥0 open)
    state::Int             # 1=open, 2=slip, 3=stick
    method::Symbol         # :ntn | :nts
    master_nodes::Vector{Int}
    ξ::Float64             # segment coord in [-1, 1] (NTS)
    N1::Float64
    N2::Float64
    tn::Float64
    tt::Float64
end

function ContactPair(
        reg_a::Integer,
        node_a::Integer,
        reg_b::Integer,
        node_b::Integer,
        μ::Real,
        gap0::Real,
        state::Integer = 1;
        method::Symbol = :ntn,
        master_nodes = nothing,
        ξ::Real = 0.0,
        N1::Real = 1.0,
        N2::Real = 0.0,
        tn::Real = 0.0,
        tt::Real = 0.0,
    )
    mnodes = master_nodes === nothing ? Int[Int(node_b)] : collect(Int, master_nodes)
    return ContactPair(
        Int(reg_a), Int(node_a), Int(reg_b), Int(node_b),
        float(μ), float(gap0), Int(state),
        method, mnodes, float(ξ), float(N1), float(N2), float(tn), float(tt),
    )
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
    pair_contacts!(prob; method=:ntn, tol=1e-4, slave_reg=0, master_reg=0)

Pair collocation nodes tagged `BC == 4` between regions.

# Keywords
- `method` — `:ntn` nearest node-to-node (default) or `:nts` node-to-segment
  (slave node projected onto the master contact polyline)
- `slave_reg`, `master_reg` — optional 1-based region indices; when both 0
  (default) every region pair is considered and the first region in the pair
  acts as slave
- `tol` — NTN matching length scale (also used as a soft filter)

Friction coefficient `μ` is taken from `BV` of the contact-typed dof. Initial
gap is the geometric normal separation (positive = open).
"""
function pair_contacts!(prob::MultiRegionProblem; method::Symbol=:ntn, tol=1e-4,
        slave_reg::Int=0, master_reg::Int=0)
    method === :ntn || method === :nts ||
        throw(ArgumentError("method must be :ntn or :nts (got $method)"))
    regs = prob.regions
    nodes = Vector{Vector{Int}}(undef, length(regs))
    mus = Vector{Vector{Float64}}(undef, length(regs))
    for (r, dad) in enumerate(regs)
        if dad.properties isa Scalar
            idx = findall(==(BC_CONTACT), dad.BC)
            nodes[r] = idx
            mus[r] = dad.BV[idx]
        else
            nd, μs = contact_nodes_elasticity(dad)
            nodes[r] = nd
            mus[r] = μs
        end
    end

    pairs = ContactPair[]
    if method === :nts && length(regs) >= 2
        # dedicated slave → master projection path
        ra = slave_reg == 0 ? 1 : slave_reg
        rb = master_reg == 0 ? (ra == 1 ? min(2, length(regs)) : 1) : master_reg
        ra == rb && throw(ArgumentError("slave_reg and master_reg must differ"))
        append!(pairs, _pair_contacts_nts(regs, nodes, mus, ra, rb; tol=tol))
    else
        used = [falses(length(nodes[r])) for r in eachindex(nodes)]
        region_pairs = if slave_reg != 0 && master_reg != 0
            [(slave_reg, master_reg)]
        else
            [(ra, rb) for ra in 1:length(regs)-1 for rb in ra+1:length(regs)]
        end
        for (ra, rb) in region_pairs
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
                if best_ib > 0 && (best_d < tol * 10 || best_d < 1.0)
                    nb = nodes[rb][best_ib]
                    pb = regs[rb].Nodes[nb]
                    n̂ = na_n / (norm(na_n) + eps())
                    gap0 = max(0.0, dot(pb - pa, n̂))
                    μ = max(mus[ra][ia], mus[rb][best_ib])
                    push!(pairs, ContactPair(ra, na, rb, nb, μ, gap0, 1;
                        method=:ntn, master_nodes=Int[nb], N1=1.0, N2=0.0))
                    used[ra][ia] = true
                    used[rb][best_ib] = true
                end
            end
        end
    end
    prob.contacts = pairs
    return pairs
end

"""Elasticity nodes tagged BC type 4 (any dof) and their μ values."""
function contact_nodes_elasticity(dad::BEMdata{<:Elasticity})
    dim = dad.dimension
    nd = Int[]
    μs = Float64[]
    for i in 1:dad.n
        if dad.BC[dim*(i-1)+1] == BC_CONTACT || dad.BC[dim*(i-1)+dim] == BC_CONTACT
            push!(nd, i)
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
    return nd, μs
end

"""
    project_point_to_segment2d(p, a, b) -> (; ξ, N1, N2, p̄, d2)

Project point `p` onto segment `a—b`. `ξ ∈ [-1,1]`, `N1=(1-ξ)/2`, `N2=(1+ξ)/2`,
clamped to the segment.
"""
function project_point_to_segment2d(p, a, b)
    ab = b - a
    L2 = dot(ab, ab)
    t = L2 > 0 ? clamp(dot(p - a, ab) / L2, 0.0, 1.0) : 0.0
    ξ = 2t - 1
    N1 = 1 - t
    N2 = t
    p̄ = a + t * ab
    d2 = dot(p - p̄, p - p̄)
    return (; ξ, N1, N2, p̄, d2, t)
end

function _sort_contact_polyline(dad, nodes::Vector{Int})
    isempty(nodes) && return Int[]
    xs = [dad.Nodes[i][1] for i in nodes]
    ys = [dad.Nodes[i][2] for i in nodes]
    perm = (maximum(xs) - minimum(xs)) >= (maximum(ys) - minimum(ys)) ?
        sortperm(xs) : sortperm(ys)
    return nodes[perm]
end

function _pair_contacts_nts(regs, nodes, mus, ra::Int, rb::Int; tol=1e-4)
    dad_s, dad_m = regs[ra], regs[rb]
    slave = nodes[ra]
    master = _sort_contact_polyline(dad_m, nodes[rb])
    isempty(slave) && return ContactPair[]
    length(master) < 1 && return ContactPair[]

    pairs = ContactPair[]
    # master μ: max over master contact nodes
    μ_m = isempty(mus[rb]) ? 0.0 : maximum(mus[rb])
    for (ia, na) in enumerate(slave)
        ps = dad_s.Nodes[na]
        n̂ = dad_s.Normal[na]
        n̂ = n̂ / (norm(n̂) + eps())
        if length(master) == 1
            nb = master[1]
            pb = dad_m.Nodes[nb]
            gap0 = max(0.0, dot(pb - ps, n̂))
            μ = max(mus[ra][ia], μ_m)
            push!(pairs, ContactPair(ra, na, rb, nb, μ, gap0, 1;
                method=:nts, master_nodes=Int[nb], ξ=0.0, N1=1.0, N2=0.0))
            continue
        end
        best = (d2=Inf, ξ=0.0, N1=1.0, N2=0.0, p̄=ps, m1=master[1], m2=master[1])
        for k in 1:(length(master) - 1)
            m1, m2 = master[k], master[k + 1]
            pr = project_point_to_segment2d(ps, dad_m.Nodes[m1], dad_m.Nodes[m2])
            if pr.d2 < best.d2
                best = (d2=pr.d2, ξ=pr.ξ, N1=pr.N1, N2=pr.N2, p̄=pr.p̄, m1=m1, m2=m2)
            end
        end
        # primary master node = larger weight
        nb = best.N1 >= best.N2 ? best.m1 : best.m2
        gap0 = max(0.0, dot(best.p̄ - ps, n̂))
        μ = max(mus[ra][ia], μ_m)
        push!(pairs, ContactPair(ra, na, rb, nb, μ, gap0, 1;
            method=:nts, master_nodes=Int[best.m1, best.m2],
            ξ=best.ξ, N1=best.N1, N2=best.N2))
    end
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

"""
Elasticity frictional contact — Contato (MATLAB) multi-body **active-set** scheme.

Builds a **coupled** global system in the nodal (n,t) frame (Leonardo / Contato):

1. Each region: ``Ĥ, Ĝ`` via local rotation; exterior BCs applied; contact faces
   keep free ``u`` with contact tractions as extra unknowns.
2. Contact pairs contribute 4 algebraic rows (open / slip / stick) as in
   `aplica_contato_com_atrito_multicorpos.m`.
3. **Explicit active-set loop** (Contato with frozen set = one linear solve):
   ```text
   state ← verify(x)
   A, b  ← assemble(state)
   x     ← A \\ b
   until ‖Δx‖ small and/or states stable
   ```
   Equivalent to one Newton step on ``R = Ax - b`` with ``J = A``.

States: `1=open`, `±2=slip`, `3=stick` (MATLAB codes).

Solvers:
- `:activeset` (default) — Contato verify → assemble → ``x=A\\b``
- `:ssn` — semi-smooth Newton on Alart–Curnier residual (same unknowns)

For robustness under large approach/load prefer
[`solve_contact_friction_stepped!`](@ref) (outer load loop + warm start).

# Keywords
- `δ` — rigid approach (``h = g₀ - δ``)
- `solver` — `:activeset` | `:ssn`
- `rn`, `rt` — AC augmentation (default: auto from ``E/L``); only `:ssn`
- `x0` — optional warm-start unknown vector
- `reset_states` — if `true` (default), all pairs start open
- `return_x` — also return the unknown vector for warm starts
"""
function solve_contact_friction!(prob::MultiRegionProblem{<:Elasticity};
        δ=0.0, tol=1e-8, maxiter=40, npg=12, verbose=false,
        method::Symbol=:ntn,
        solver::Symbol=:activeset,
        rn::Union{Nothing,Real}=nothing,
        rt::Union{Nothing,Real}=nothing,
        x0::Union{Nothing,AbstractVector}=nothing,
        reset_states::Bool=true,
        return_x::Bool=false)
    ctx = _contact_friction_setup(prob; method=method, npg=npg)
    ctx === nothing && return return_x ? (prob, Float64[]) : prob
    prep, pairs = ctx.prep, ctx.pairs
    h = [cp.gap0 - δ for cp in pairs]
    N = ctx.N
    x_init = if x0 === nothing
        zeros(N)
    else
        length(x0) == N || throw(DimensionMismatch("x0 length $(length(x0)) ≠ $N"))
        collect(Float64, x0)
    end
    if reset_states
        for cp in pairs
            cp.state = 1
        end
    end
    x, ok = _contact_inner_solve!(prep, pairs, h, x_init;
        solver=solver, tol=tol, maxiter=maxiter, verbose=verbose, rn=rn, rt=rt)
    ok || @warn "solve_contact_friction! did not fully converge" δ=δ solver=solver
    _verify_contact_states!(pairs, prep, h, x; epsc=1e-7)
    _scatter_contact_solution!(prob, prep, pairs, x)
    if !isempty(prob.regions)
        set_cache!(prob.regions[1]; contact_x=copy(x))
    end
    return return_x ? (prob, x) : prob
end

"""
    solve_contact_friction_stepped!(prob; δ_end, nsteps=10, ...)

**Load-stepped** frictional contact: outer loop on rigid approach ``δ``, inner
solver (Contato active-set or semi-smooth Newton).

```text
for s = 1:nsteps
    δ_s = δ_end * s/nsteps
    x ← inner_solve(δ_s; warm-start x)   # :activeset or :ssn
end
```

# Keywords
- `δ_end` / `δ_start` / `nsteps` / `δ_path` — approach schedule
- `solver` — `:activeset` (default) | `:ssn`
- `rn`, `rt` — Alart–Curnier scales for `:ssn` (default auto)
- `adaptive` — bisect a failed step once and retry
- `tol`, `maxiter`, `method`, `verbose`, `npg`

History: `contact_δ_hist`, `contact_tn_hist`, `contact_x` on `prob.regions[1]`.
"""
function solve_contact_friction_stepped!(prob::MultiRegionProblem{<:Elasticity};
        δ_end::Union{Nothing,Real}=nothing,
        δ_start::Real=0.0,
        nsteps::Int=10,
        δ_path::Union{Nothing,AbstractVector}=nothing,
        adaptive::Bool=true,
        tol=1e-8,
        maxiter=40,
        npg=12,
        verbose=false,
        method::Symbol=:ntn,
        solver::Symbol=:activeset,
        rn::Union{Nothing,Real}=nothing,
        rt::Union{Nothing,Real}=nothing)
    ctx = _contact_friction_setup(prob; method=method, npg=npg)
    ctx === nothing && return prob
    prep, pairs = ctx.prep, ctx.pairs
    N = ctx.N

    if δ_path !== nothing
        path = collect(Float64, δ_path)
    else
        δ_end === nothing && throw(ArgumentError("pass δ_end or δ_path"))
        nsteps >= 1 || throw(ArgumentError("nsteps ≥ 1"))
        path = collect(range(float(δ_start), float(δ_end); length=nsteps + 1))[2:end]
    end

    x = zeros(N)
    if has_cache(prob.regions[1], :contact_x)
        xw = prob.regions[1].contact_x
        length(xw) == N && (x = collect(Float64, xw))
    end
    for cp in pairs
        cp.state = 1
    end

    δ_hist = Float64[]
    tn_hist = Float64[]
    s = 1
    δ_ref = δ_end === nothing ? (isempty(path) ? 1.0 : path[end]) : float(δ_end)
    while s <= length(path)
        δ = path[s]
        h = [cp.gap0 - δ for cp in pairs]
        x_try, ok = _contact_inner_solve!(prep, pairs, h, x;
            solver=solver, tol=tol, maxiter=maxiter, verbose=verbose, rn=rn, rt=rt)
        if !ok && adaptive
            δ_prev = s == 1 ? float(δ_start) : path[s - 1]
            δ_mid = 0.5 * (δ_prev + δ)
            if abs(δ_mid - δ_prev) > 1e-14 * max(abs(δ_ref), 1.0)
                verbose && @info "contact step failed; bisecting" δ=δ δ_mid=δ_mid solver=solver
                insert!(path, s, δ_mid)
                continue
            end
        end
        if !ok
            @warn "solve_contact_friction_stepped! step failed" s=s δ=δ solver=solver
        end
        x = x_try
        _verify_contact_states!(pairs, prep, h, x; epsc=1e-7)
        push!(δ_hist, δ)
        tn_mean = mean(abs(cp.tn) for cp in pairs)
        push!(tn_hist, tn_mean)
        verbose && @info "contact step" s=s δ=δ solver=solver n_closed=count(cp -> abs(cp.state) != 1, pairs) tn_mean=tn_mean
        s += 1
    end

    _scatter_contact_solution!(prob, prep, pairs, x)
    set_cache!(prob.regions[1]; contact_x=copy(x),
        contact_δ_hist=δ_hist, contact_tn_hist=tn_hist)
    return prob
end

# ---------------------------------------------------------------------------
# Shared setup + explicit active-set driver (Contato)
# ---------------------------------------------------------------------------

function _contact_friction_setup(prob::MultiRegionProblem{<:Elasticity};
        method::Symbol=:ntn, npg=12)
    isempty(prob.contacts) && pair_contacts!(prob; method=method)
    regs = prob.regions
    length(regs) >= 2 || error("need ≥2 regions")
    for dad in regs
        dad.dimension == 2 || error("elasticity contact is 2D only")
        has_cache(dad, :H) || H_G_full_direct(dad, npg)
    end
    BC0 = [copy(d.BC) for d in regs]
    BV0 = [copy(d.BV) for d in regs]
    prep = _prepare_regions_contact_local(regs, BC0, BV0)
    pairs = prob.contacts
    isempty(pairs) && (@warn "no contact pairs"; return nothing)
    N = sum(p.ndof for p in prep) + 4 * length(pairs)
    return (; prep, pairs, N)
end

"""
Contato active-set iteration (explicit linear solves).

```text
for it = 1:maxiter
    state ← verify(x)                 # open / stick / ±slip
    A, b  ← assemble(state, h)
    x_new ← A \\ b                    # exact for frozen set
    stop if ‖x_new - x‖ < tol
    x ← x_new
end
```

Returns `(x, converged)`.
"""
function _contact_activeset!(prep, pairs, h, x_init;
        tol=1e-8, maxiter=40, verbose=false, epsc=1e-7)
    x = collect(Float64, x_init)
    ok = false
    for it in 1:maxiter
        _verify_contact_states!(pairs, prep, h, x; epsc=epsc)
        A, b = _assemble_contact_system(prep, pairs, h, x)
        x_new = A \ b
        dist = norm(x_new - x)
        verbose && @info "contact active-set" it dist n_closed=count(cp -> abs(cp.state) != 1, pairs)
        x .= x_new
        if dist < tol
            ok = true
            break
        end
    end
    return x, ok
end

"""Dispatch inner contact solve: `:activeset` or `:ssn`."""
function _contact_inner_solve!(prep, pairs, h, x_init;
        solver::Symbol=:activeset, tol=1e-8, maxiter=40, verbose=false,
        rn=nothing, rt=nothing)
    if solver === :activeset
        return _contact_activeset!(prep, pairs, h, x_init; tol=tol, maxiter=maxiter,
            verbose=verbose)
    elseif solver === :ssn
        return _contact_ssn!(prep, pairs, h, x_init; tol=tol, maxiter=maxiter,
            verbose=verbose, rn=rn, rt=rt)
    else
        throw(ArgumentError("unknown contact solver $(repr(solver)); use :activeset or :ssn"))
    end
end

# -----------------------------------------------------------------------------
# Semi-smooth Newton (Alart–Curnier) on Contato unknowns
# -----------------------------------------------------------------------------

"""Default AC scales ``r ∼ 10 E / L`` from region properties / bbox."""
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
function _contact_pair_kinematics(prep, cp, h_k, x, k, nx)
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
    gt = dut
    return (; dun, dut, gn, gt, tn1, tt1, tn2, tt2, R, iu1, iu2, ot,
        un1, ut1, un2, ut2)
end

"""
Alart–Curnier NCF for 2D Coulomb (compression multiplier ``λ_n = -t_n``).

Returns `(Cn, Ct, regime, s_slip, τn, τt, λn⁺)` with
`regime ∈ (:open, :stick, :slip)` and `s_slip = sign(τt)` on the slip piece.
"""
function alart_curnier(gn::Real, gt::Real, tn::Real, tt::Real, μ::Real,
        rn::Real, rt::Real)
    λn = -float(tn)
    λt = -float(tt)
    τn = λn - rn * gn
    τt = λt - rt * gt
    λn⁺ = max(0.0, τn)
    if τn <= 0.0
        # open: proj radius 0
        Cn = λn
        Ct = λt
        return Cn, Ct, :open, 0.0, τn, τt, λn⁺
    end
    bound = μ * λn⁺
    if abs(τt) <= bound + 1e-15
        # stick
        Cn = rn * gn          # = λn - τn
        Ct = rt * gt          # = λt - τt
        return Cn, Ct, :stick, 0.0, τn, τt, λn⁺
    end
    # slip
    s = τt == 0.0 ? 1.0 : sign(τt)
    λt_hat = s * bound
    Cn = rn * gn
    Ct = λt - λt_hat         # = λt - s μ (λn - rn gn)
    return Cn, Ct, :slip, s, τn, τt, λn⁺
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

"""
Assemble SSN residual ``R`` and generalized Jacobian ``J`` on Contato layout.

Unknowns: mixed BIE DOFs then per pair ``(t_n¹,t_t¹,t_n²,t_t²)``.
Contact rows: ``(C_n, C_t, E_n, E_t)`` with Alart–Curnier + traction equilibrium.
"""
function _assemble_contact_R_J(prep, pairs, h, x; rn::Real=1.0, rt::Real=1.0)
    nx = sum(p.ndof for p in prep)
    np = length(pairs)
    N = nx + 4 * np
    R = zeros(N)
    J = zeros(N, N)

    # --- BIE blocks ---
    for pr in prep
        o = pr.off
        nd = pr.ndof
        xr = @view x[o+1:o+nd]
        J[o+1:o+nd, o+1:o+nd] .= pr.A
        mul!(@view(R[o+1:o+nd]), pr.A, xr)
        R[o+1:o+nd] .-= pr.b
    end
    for (k, cp) in enumerate(pairs)
        pr1 = prep[cp.reg_a]
        pr2 = prep[cp.reg_b]
        na, nb = cp.node_a, cp.node_b
        ot = nx + 4(k - 1)
        t1 = SVector(x[ot+1], x[ot+2])
        t2 = SVector(x[ot+3], x[ot+4])
        if haskey(pr1.Gc_cols, na)
            cols = pr1.Gc_cols[na]
            G1 = pr1.G_local[:, cols]
            R[pr1.off+1:pr1.off+pr1.ndof] .-= G1 * t1
            J[pr1.off+1:pr1.off+pr1.ndof, ot+1:ot+2] .-= G1
        end
        if haskey(pr2.Gc_cols, nb)
            cols = pr2.Gc_cols[nb]
            G2 = pr2.G_local[:, cols]
            R[pr2.off+1:pr2.off+pr2.ndof] .-= G2 * t2
            J[pr2.off+1:pr2.off+pr2.ndof, ot+3:ot+4] .-= G2
        end
    end

    # --- contact NCF + equilibrium ---
    for (k, cp) in enumerate(pairs)
        kin = _contact_pair_kinematics(prep, cp, h[k], x, k, nx)
        Rmat = kin.R
        ot, iu1, iu2 = kin.ot, kin.iu1, kin.iu2
        tn1, tt1, tn2, tt2 = kin.tn1, kin.tt1, kin.tn2, kin.tt2
        Cn, Ct, regime, s, _, _, _ = alart_curnier(kin.gn, kin.gt, tn1, tt1, cp.μ, rn, rt)

        r1, r2, r3, r4 = ot + 1, ot + 2, ot + 3, ot + 4
        R[r1] = Cn
        R[r2] = Ct
        R[r3] = tn1 + Rmat[1, 1] * tn2 + Rmat[1, 2] * tt2
        R[r4] = tt1 + Rmat[2, 1] * tn2 + Rmat[2, 2] * tt2

        # equilibrium Jacobian (always)
        J[r3, ot+1] = 1.0
        J[r3, ot+3] = Rmat[1, 1]
        J[r3, ot+4] = Rmat[1, 2]
        J[r4, ot+2] = 1.0
        J[r4, ot+3] = Rmat[2, 1]
        J[r4, ot+4] = Rmat[2, 2]

        μ = cp.μ
        if regime === :open
            # Cn = λn = -tn1, Ct = λt = -tt1
            J[r1, ot+1] = -1.0
            J[r2, ot+2] = -1.0
        elseif regime === :stick
            # Cn = rn gn, Ct = rt gt
            _add_gn_row!(J, r1, iu1, iu2, Rmat, rn)
            _add_gt_row!(J, r2, iu1, iu2, Rmat, rt)
        else
            # slip: Cn = rn gn
            # Ct = -tt + s μ tn + s μ rn gn   (λ form chained through λ=-t)
            _add_gn_row!(J, r1, iu1, iu2, Rmat, rn)
            J[r2, ot+2] = -1.0
            J[r2, ot+1] = s * μ
            _add_gn_row!(J, r2, iu1, iu2, Rmat, s * μ * rn)
        end

        # store regime as Contato state for diagnostics (overwritten by verify later)
        if regime === :open
            cp.state = 1
        elseif regime === :stick
            cp.state = 3
        else
            # slip sign from traction (Contato: ±2)
            cp.state = Int((tt1 == 0.0 ? s : sign(tt1)) * 2)
            cp.state == 0 && (cp.state = Int(s * 2))
        end
        cp.tn = tn1
        cp.tt = tt1
    end
    return R, J
end

"""Residual-only evaluation (for line search)."""
function _assemble_contact_R(prep, pairs, h, x; rn::Real=1.0, rt::Real=1.0)
    R, _ = _assemble_contact_R_J(prep, pairs, h, x; rn=rn, rt=rt)
    return R
end

"""
Semi-smooth Newton on Alart–Curnier contact residual.

```text
for it
    R, J ← assemble_R_J(x)
    solve J Δx = -R
    line-search α on ‖R‖
    x ← x + α Δx
```
"""
function _contact_ssn!(prep, pairs, h, x_init;
        tol=1e-8, maxiter=40, verbose=false,
        rn::Union{Nothing,Real}=nothing,
        rt::Union{Nothing,Real}=nothing,
        ls_max::Int=8)
    x = collect(Float64, x_init)
    rn0, rt0 = _default_contact_r(prep)
    rn_ = rn === nothing ? rn0 : float(rn)
    rt_ = rt === nothing ? rt0 : float(rt)
    ok = false
    nR = Inf
    for it in 1:maxiter
        R, J = _assemble_contact_R_J(prep, pairs, h, x; rn=rn_, rt=rt_)
        nR = norm(R)
        verbose && @info "contact SSN" it nR rn=rn_ n_closed=count(cp -> abs(cp.state) != 1, pairs)
        if nR < tol
            ok = true
            break
        end
        dx = J \ (-R)
        # Armijo-like backtracking on ‖R‖
        α = 1.0
        nR_new = nR
        x_trial = similar(x)
        accepted = false
        for _ls in 1:ls_max
            x_trial .= x .+ α .* dx
            R_try = _assemble_contact_R(prep, pairs, h, x_trial; rn=rn_, rt=rt_)
            nR_new = norm(R_try)
            if nR_new < (1.0 - 1e-4 * α) * nR || nR_new < tol
                accepted = true
                break
            end
            α *= 0.5
        end
        if !accepted
            # take smallest trial anyway (damped progress)
            x_trial .= x .+ α .* dx
            nR_new = norm(_assemble_contact_R(prep, pairs, h, x_trial; rn=rn_, rt=rt_))
        end
        x .= x_trial
        if nR_new < tol
            ok = true
            break
        end
        # also stop on tiny step
        if α * norm(dx) < tol * max(1.0, norm(x))
            ok = nR_new < max(tol, 1e-6 * max(1.0, nR))
            break
        end
    end
    # final residual / state tags
    R, _ = _assemble_contact_R_J(prep, pairs, h, x; rn=rn_, rt=rt_)
    nR = norm(R)
    ok = ok || nR < tol
    verbose && @info "contact SSN done" ok nR
    return x, ok
end

# -----------------------------------------------------------------------------
# Region preparation (local frame + exterior BC)
# -----------------------------------------------------------------------------

"""Per-region data after local transform and exterior BC column exchange."""
struct _RegContactPrep
    dad::BEMdata
    A::Matrix{Float64}          # mixed BIE operator (ndof×ndof)
    b::Vector{Float64}          # known RHS from exterior BC
    Gc_cols::Dict{Int,UnitRange{Int}}  # node → columns of G_local for contact t
    G_local::Matrix{Float64}
    H_local::Matrix{Float64}
    BC_ext::Vector{Int}         # exterior BC in local (contact marked Neumann)
    BV_ext::Vector{Float64}
    is_contact_node::Vector{Bool}
    ndof::Int
    off::Int                    # global offset of this region's mixed DOFs
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
            if BC[dof] == BC_DIRICHLET
                # after swap, Bmat column holds -H; known is u
                b .-= Bmat[:, dof] .* BV[dof]
            else
                b .+= Bmat[:, dof] .* BV[dof]
            end
        end

        Gc = Dict{Int,UnitRange{Int}}()
        for i in 1:dad.n
            contact_nodes[r][i] || continue
            Gc[i] = 2i-1:2i
        end

        push!(preps, _RegContactPrep(dad, A, b, Gc, Gloc[1:ndof, 1:ndof], Hloc[1:ndof, 1:ndof],
            BC, BV, contact_nodes[r], ndof, off))
        off += ndof
    end
    return preps
end

# -----------------------------------------------------------------------------
# Coupled system assembly (BIE + contact constraints)
# -----------------------------------------------------------------------------

function _assemble_contact_system(prep::Vector{_RegContactPrep}, pairs, h, x)
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

        if st == 1  # open: tn1=tt1=tn2=tt2=0
            A[rows[1], ot+1] = 1.0
            A[rows[2], ot+2] = 1.0
            A[rows[3], ot+3] = 1.0
            A[rows[4], ot+4] = 1.0
            b[rows] .= 0.0
        elseif abs(st) == 2  # slip
            sμ = μ * sign(st == 0 ? 1 : st)  # st = ±2
            # un1 - R[1,:]·u2 = h
            A[rows[1], iu1] = 1.0
            A[rows[1], iu1+1] = 0.0
            A[rows[1], iu2] = -R[1, 1]
            A[rows[1], iu2+1] = -R[1, 2]
            b[rows[1]] = h[k]
            # tn1 + R[1,:]·t2 = 0  (equilibrium normal)
            A[rows[2], ot+1] = 1.0
            A[rows[2], ot+3] = R[1, 1]
            A[rows[2], ot+4] = R[1, 2]
            b[rows[2]] = 0.0
            # tt1 - sμ*tn1 = 0
            A[rows[3], ot+1] = sμ
            A[rows[3], ot+2] = 1.0
            b[rows[3]] = 0.0
            # tt2 related: tt1 + R[2,:]·t2 = 0  (MATLAB: 0 1 R21 R22 on t)
            A[rows[4], ot+2] = 1.0
            A[rows[4], ot+3] = R[2, 1]
            A[rows[4], ot+4] = R[2, 2]
            b[rows[4]] = 0.0
        else  # stick (3)
            # un1 - R row1 u2 = h
            A[rows[1], iu1] = 1.0
            A[rows[1], iu2] = -R[1, 1]
            A[rows[1], iu2+1] = -R[1, 2]
            b[rows[1]] = h[k]
            # ut1 - R row2 u2 = 0
            A[rows[2], iu1+1] = 1.0
            A[rows[2], iu2] = -R[2, 1]
            A[rows[2], iu2+1] = -R[2, 2]
            b[rows[2]] = 0.0
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

# -----------------------------------------------------------------------------
# State update (MATLAB verfica_contato_com_atrito_multicorpos)
# -----------------------------------------------------------------------------

function _verify_contact_states!(pairs, prep, h, x; epsc=1e-7)
    nx = sum(p.ndof for p in prep)
    for (k, cp) in enumerate(pairs)
        pr1 = prep[cp.reg_a]
        pr2 = prep[cp.reg_b]
        na, nb = cp.node_a, cp.node_b
        ot = nx + 4(k - 1)
        tn1 = x[ot+1]
        tt1 = x[ot+2]
        # local displacements at contact (Neumann → unknown is u)
        un1 = x[pr1.off + 2na - 1]
        ut1 = x[pr1.off + 2na]
        un2 = x[pr2.off + 2nb - 1]
        ut2 = x[pr2.off + 2nb]
        R1 = Matrix(node_rotation2d(pr1.dad.Normal[na]))
        R2 = Matrix(node_rotation2d(pr2.dad.Normal[nb]))
        R = R2 * R1'
        u2_in_1 = R * SVector(un2, ut2)
        dun = un1 - u2_in_1[1]
        dut = ut1 - u2_in_1[2]
        μ = cp.μ
        sinal_tt = tt1 == 0 ? 1.0 : sign(tt1)
        sinal_dut = dut == 0 ? 1.0 : sign(dut)
        μ == 0 && (sinal_tt = 1.0)

        if abs(tn1) <= epsc  # free / open traction
            if dun + epsc > h[k]   # penetration → stick
                cp.state = 3
            else
                cp.state = 1
            end
        else
            if tn1 > epsc          # tension → open
                cp.state = 1
            elseif abs(tt1) + epsc > μ * abs(tn1)  # slip
                if sinal_dut * sinal_tt > 0
                    cp.state = 3     # reverse slip sense → stick
                else
                    cp.state = Int(sinal_tt * 2)
                end
            else
                cp.state = 3
            end
        end
        cp.tn = tn1
        cp.tt = tt1
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

# ==============================================================================
# Multibody elasticity contact — penalty NTN / NTS
# ==============================================================================

"""Interpolate master displacement at a contact pair (NTN or NTS)."""
function _master_u(cp::ContactPair, db::BEMdata{<:Elasticity})
    u = db.u
    if cp.method === :nts && length(cp.master_nodes) >= 2
        m1, m2 = cp.master_nodes[1], cp.master_nodes[2]
        ux = cp.N1 * u[2m1-1] + cp.N2 * u[2m2-1]
        uy = cp.N1 * u[2m1] + cp.N2 * u[2m2]
        return ux, uy
    else
        nb = cp.node_b
        return u[2nb-1], u[2nb]
    end
end

function _master_x(cp::ContactPair, db::BEMdata)
    if cp.method === :nts && length(cp.master_nodes) >= 2
        m1, m2 = cp.master_nodes[1], cp.master_nodes[2]
        return cp.N1 * db.Nodes[m1] + cp.N2 * db.Nodes[m2]
    else
        return db.Nodes[cp.node_b]
    end
end

"""
    solve_multibody_elasticity_contact!(prob; method=:ntn, frame=:local, kn, kt, δ, ...)

Penalty frictional contact between elastic subregions with type-4 faces.

Supports:
- `:ntn` — node-to-node pairing (`pair_contacts!`)
- `:nts` — node-to-segment (slave node → master contact polyline)
- `frame=:local` (default) — Leonardo §4.7 nodal (n,t) BEM: contact tractions
  applied as local Neumann `(t_n, t_t)`, bodies solved with [`solve_local`](@ref)
- `frame=:global` — legacy path with global `(t_x, t_y)` Neumann + `applyBC`

Algorithm (fixed-point / active set):
1. Snapshot exterior BCs; convert to local if `frame=:local`
2. From current global `u`, evaluate gap / tangential slip at each slave node
3. Penalty + Coulomb → slave `(tn, tt)`; action–reaction on master
4. Solve each body (`solve_local` or global `applyBC`)
5. Under-relax tractions and iterate

`δ` is an additional rigid normal approach subtracted from the geometric gap
(positive `δ` closes the joint). Returns `prob` with `contacts` updated
(`state`, `tn`, `tt`).
"""
function solve_multibody_elasticity_contact!(
        prob::MultiRegionProblem{<:Elasticity};
        method::Symbol = :ntn,
        frame::Symbol = :local,
        kn::Real = 1e5,
        kt::Real = 1e5,
        δ::Real = 0.0,
        tol::Real = 1e-6,
        maxiter::Int = 50,
        ω::Real = 0.6,
        slave_reg::Int = 0,
        master_reg::Int = 0,
        npg::Int = 12,
        pair::Bool = true,
    )
    frame === :local || frame === :global ||
        throw(ArgumentError("frame must be :local or :global (got $frame)"))
    regs = prob.regions
    if pair || isempty(prob.contacts) || any(cp -> cp.method !== method, prob.contacts)
        pair_contacts!(prob; method=method, slave_reg=slave_reg, master_reg=master_reg)
    end
    isempty(prob.contacts) && @warn "solve_multibody_elasticity_contact!: no contact pairs"

    # snapshot exterior BCs as provided (typically global from Gmsh)
    BC0 = [copy(dad.BC) for dad in regs]
    BV0 = [copy(dad.BV) for dad in regs]

    # exterior BCs used each iteration (local or global)
    BC_ext = [copy(bc) for bc in BC0]
    BV_ext = [copy(bv) for bv in BV0]
    if frame === :local
        for (r, dad) in enumerate(regs)
            _exterior_bc_to_local!(BC_ext[r], BV_ext[r], dad)
        end
    end

    for dad in regs
        has_cache(dad, :H) || H_G_full_direct(dad, npg)
        dim = dad.dimension
        if !has_cache(dad, :u)
            set_cache!(dad; u=zeros(dim * dad.n), traction=zeros(dim * dad.n))
        end
    end

    kn = float(kn); kt = float(kt); δ = float(δ); ω = float(ω)
    for it in 1:maxiter
        # restore exterior BCs; contact faces → Neumann accumulators
        for (r, dad) in enumerate(regs)
            dad.BC .= BC_ext[r]
            dad.BV .= BV_ext[r]
            for i in 1:dad.n
                if BC0[r][2i-1] == BC_CONTACT || BC0[r][2i] == BC_CONTACT ||
                   BC_ext[r][2i-1] == BC_CONTACT || BC_ext[r][2i] == BC_CONTACT
                    dad.BC[2i-1] = BC_NEUMANN; dad.BV[2i-1] = 0.0
                    dad.BC[2i]   = BC_NEUMANN; dad.BV[2i]   = 0.0
                end
            end
        end

        max_pen = 0.0
        max_dt = 0.0
        for cp in prob.contacts
            da, db = regs[cp.reg_a], regs[cp.reg_b]
            na = cp.node_a
            n̂s, t̂s = local_basis2d(da.Normal[na])

            ua = da.u
            usx, usy = ua[2na-1], ua[2na]
            umx, umy = _master_u(cp, db)

            # gap > 0 open: g = g0 - δ + (u_m - u_s)·n̂_s
            gap = cp.gap0 - δ + (umx - usx) * n̂s[1] + (umy - usy) * n̂s[2]
            slip = (usx - umx) * t̂s[1] + (usy - umy) * t̂s[2]

            tn_new = 0.0
            tt_new = 0.0
            if gap < 0
                tn_new = kn * gap            # compression ⇒ tn < 0
                tt_trial = -kt * slip
                τmax = cp.μ * abs(tn_new)
                if abs(tt_trial) <= τmax + 1e-14
                    tt_new = tt_trial
                else
                    tt_new = sign(tt_trial) == 0 ? τmax : sign(tt_trial) * τmax
                end
                max_pen = max(max_pen, -gap)
            end

            ωc = gap < 0 ? ω : max(ω, 0.85)
            tn = (1 - ωc) * cp.tn + ωc * tn_new
            tt = (1 - ωc) * cp.tt + ωc * tt_new
            max_dt = max(max_dt, abs(tn - cp.tn) + abs(tt - cp.tt))
            cp.tn = tn
            cp.tt = tt
            if tn >= -1e-14 * max(kn, 1.0)
                cp.state = 1
                cp.tt = 0.0
                cp.tn = 0.0
                tt = 0.0
                tn = 0.0
            else
                τmax = cp.μ * abs(tn)
                cp.state = abs(tt) >= τmax - 1e-12 * max(kn, 1.0) ? 2 : 3
            end

            # global traction on slave (outward n_s)
            tx = tn * n̂s[1] + tt * t̂s[1]
            ty = tn * n̂s[2] + tt * t̂s[2]

            if frame === :local
                # slave: local Neumann (t_n, t_t) — natural contact variables
                da.BC[2na-1] = BC_NEUMANN; da.BV[2na-1] += tn
                da.BC[2na]   = BC_NEUMANN; da.BV[2na]   += tt
                # master: action–reaction in global, then to each master local frame
                _accumulate_master_traction_local!(db, cp, -tx, -ty)
            else
                da.BC[2na-1] = BC_NEUMANN; da.BV[2na-1] += tx
                da.BC[2na]   = BC_NEUMANN; da.BV[2na]   += ty
                if cp.method === :nts && length(cp.master_nodes) >= 2
                    m1, m2 = cp.master_nodes[1], cp.master_nodes[2]
                    for (m, w) in ((m1, cp.N1), (m2, cp.N2))
                        db.BC[2m-1] = BC_NEUMANN; db.BV[2m-1] += -w * tx
                        db.BC[2m]   = BC_NEUMANN; db.BV[2m]   += -w * ty
                    end
                else
                    nb = cp.node_b
                    db.BC[2nb-1] = BC_NEUMANN; db.BV[2nb-1] += -tx
                    db.BC[2nb]   = BC_NEUMANN; db.BV[2nb]   += -ty
                end
            end
        end

        # solve each region
        for dad in regs
            if frame === :local
                solve_local(dad)
            else
                applyBC(dad)
                x = dad.A \ dad.b
                dim = dad.dimension
                u = zeros(dim * dad.n)
                t = zeros(dim * dad.n)
                split_sol!(dad, x, u, t)
                set_cache!(dad; u=u, traction=t, T=u)
            end
        end

        if it > 2 && max_dt < tol * max(kn, kt, 1.0) && max_pen < 10 * tol
            break
        end
    end
    return prob
end

"""Accumulate global traction `(tx,ty)` onto master node(s) as local Neumann."""
function _accumulate_master_traction_local!(db::BEMdata, cp::ContactPair, tx, ty)
    tg = SVector(float(tx), float(ty))
    if cp.method === :nts && length(cp.master_nodes) >= 2
        m1, m2 = cp.master_nodes[1], cp.master_nodes[2]
        for (m, w) in ((m1, cp.N1), (m2, cp.N2))
            Rm = node_rotation2d(db.Normal[m])
            tl = Rm' * (w * tg)
            db.BC[2m-1] = BC_NEUMANN; db.BV[2m-1] += tl[1]
            db.BC[2m]   = BC_NEUMANN; db.BV[2m]   += tl[2]
        end
    else
        nb = cp.node_b
        Rm = node_rotation2d(db.Normal[nb])
        tl = Rm' * tg
        db.BC[2nb-1] = BC_NEUMANN; db.BV[2nb-1] += tl[1]
        db.BC[2nb]   = BC_NEUMANN; db.BV[2nb]   += tl[2]
    end
    return nothing
end

"""
Convert exterior BC/BV arrays from global (x,y) to nodal (n,t).

- Same type on both DOFs: rotate BV by ``R^T`` (Leonardo §4.7)
- Contact (type 4): left tagged for the contact loop
- Mixed Dir/Neu: map common roller / symmetry cases on axis-aligned faces;
  otherwise fall back to rotating the Neumann part and assigning Dirichlet to
  the local axis best aligned with the constrained global axis
"""
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
