# Multi-region BEM with interface coupling (type-3 BC)
# Inspired by SubregioesPotencialConstante (MATLAB)

export InterfacePair, ContactPair, MultiRegionProblem
export pair_interfaces!, pair_contacts!, assemble_multiregion, solve_multiregion!
export multiregion_ndof
export solve_contact_friction!, solve_contact_friction_stepped!
export solve_contact_friction_fretting!, set_farfield_displacement!
export alart_curnier
export project_contact_traction, project_contact_multipliers
export CohesiveContactState, solve_cohesive_contact!
export solve_multibody_elasticity_contact!
export contact_nodes_elasticity, project_point_to_segment2d
export contact_common_normal, apply_contact_common_normals!

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
    """Locked relative tangential gap for incremental stick (Mindlin residual)."""
    ut_lock::Float64
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
        ut_lock::Real = 0.0,
    )
    mnodes = master_nodes === nothing ? Int[Int(node_b)] : collect(Int, master_nodes)
    return ContactPair(
        Int(reg_a), Int(node_a), Int(reg_b), Int(node_b),
        float(μ), float(gap0), Int(state),
        method, mnodes, float(ξ), float(N1), float(N2), float(tn), float(tt),
        float(ut_lock),
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
gap (`gap=:euclidean`, Contato `calc_gap_subreg`) is ``‖x_B-x_A‖``; `gap=:normal`
uses ``(x_B-x_A)·n_{AB}``.
"""
function pair_contacts!(prob::MultiRegionProblem; method::Symbol=:ntn, tol=1e-4,
        slave_reg::Int=0, master_reg::Int=0, gap::Symbol=:euclidean)
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
                    n̂ = contact_common_normal(na_n, _contact_young(regs[ra]),
                        regs[rb].Normal[nb], _contact_young(regs[rb]))
                    gap0 = if gap === :euclidean
                        d = norm(pb - pa)
                        d < 1e-8 ? 0.0 : d
                    else
                        max(0.0, dot(pb - pa, n̂))
                    end
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
    gap === :euclidean && _contato_snap_gaps!(pairs)
    return pairs
end

"""Contato `calc_gap_subreg`: dist < 1e-8 → 0. If the closest pair is a
light geometric miss (Gmsh), shift so min h = 0 (cylinder sits on the flat)."""
function _contato_snap_gaps!(pairs)
    isempty(pairs) && return pairs
    for cp in pairs
        cp.gap0 < 1e-8 && (cp.gap0 = 0.0)
    end
    hmin = minimum(cp.gap0 for cp in pairs)
    if 0 < hmin < 1e-3
        for cp in pairs
            cp.gap0 = max(0.0, cp.gap0 - hmin)
        end
    end
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
        nA = dad_s.Normal[na]
        if length(master) == 1
            nb = master[1]
            pb = dad_m.Nodes[nb]
            n̂ = contact_common_normal(nA, _contact_young(dad_s),
                dad_m.Normal[nb], _contact_young(dad_m))
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
        n̂ = contact_common_normal(nA, _contact_young(dad_s),
            dad_m.Normal[nb], _contact_young(dad_m))
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

Assemble each subregion independently (`H_G_full_direct`). Interface coupling
is applied in [`solve_multiregion!`](@ref).
"""
function assemble_multiregion(prob::MultiRegionProblem{<:Laplace}; npg=16)
    for dad in prob.regions
        H_G_full_direct(dad, npg)
    end
    return nothing
end

"""
    solve_multiregion!(prob::MultiRegionProblem{<:Laplace}; strategy=:dense)

Perfect-interface Laplace (`T_a = T_b`, `q_a + q_b = 0` with
`q = -k ∂T/∂n`). Each zone is mixed-BC collocation with interface nodes
treated as Neumann (unknown `T`); the unknown interface flux `q_if` is
shared (`q_b = -q_if`).

# Strategies

- `:dense` — fill a single dense matrix of size `∑n_t + n_if` (original
  MATLAB-style assembly).
- `:noncondensing` — Kane / Saigal (IJNME 1990) **noncondensing** multi-zone:
  same equations, stored as block-diagonal zone matrices plus interface
  border. Block Gaussian elimination factors each full `A_r` and a Schur
  complement of size `n_if` (continuity). All zone DOFs stay in the
  unreduced system.
- `:condense` — Kane **zone condensation**: eliminate exterior DOFs of each
  zone (`A_ee`) first. The global system is the stacked condensed interface
  maps on shared `(T_if, q_if)`, size `2 n_if`. Zone interiors are recovered
  by back-substitution.

Aliases: `:blocked` → `:noncondensing`, `:condensation` → `:condense`.
Set `apply_bc=false` to reuse an already mixed `A,b` (timing / repeated solves).
"""
function solve_multiregion!(prob::MultiRegionProblem{<:Laplace};
        strategy::Symbol=:dense, apply_bc::Bool=true)
    strategy = _mr_strategy(strategy)
    isempty(prob.interfaces) && pair_interfaces!(prob)
    if apply_bc
        _tag_coupling_as_neumann!(prob)
        for dad in prob.regions
            applyBC(dad)
        end
    end
    if strategy === :dense
        return _solve_mr_dense!(prob)
    elseif strategy === :noncondensing
        return _solve_mr_noncondensing!(prob)
    elseif strategy === :condense
        return _solve_mr_condense!(prob)
    else
        throw(ArgumentError("unknown multi-region strategy $strategy; use :dense, :noncondensing, or :condense"))
    end
end

_mr_strategy(s::Symbol) =
    s === :blocked ? :noncondensing :
    s === :kane ? :noncondensing :
    s === :condensation ? :condense : s

"""Global unknown count for `strategy` (`:dense`, `:noncondensing`, `:condense`)."""
function multiregion_ndof(prob::MultiRegionProblem; strategy::Symbol=:dense)
    strategy = _mr_strategy(strategy)
    nts = [dad.nt for dad in prob.regions]
    n_if = length(prob.interfaces)
    Nloc = sum(nts)
    if strategy === :dense || strategy === :noncondensing
        return Nloc + n_if
    elseif strategy === :condense
        return 2 * n_if
    else
        throw(ArgumentError("unknown multi-region strategy $strategy"))
    end
end

function _tag_node_neumann!(dad, node::Int)
    if dad.properties isa Scalar
        dad.BC[node] = BC_NEUMANN
        dad.BV[node] = 0.0
    else
        dim = dad.dimension
        for k in 1:dim
            dad.BC[dim * (node - 1) + k] = BC_NEUMANN
            dad.BV[dim * (node - 1) + k] = 0.0
        end
    end
    return nothing
end

function _tag_coupling_as_neumann!(prob::MultiRegionProblem)
    regs = prob.regions
    for (r, dad) in enumerate(regs)
        for ip in prob.interfaces
            if ip.reg_a == r
                _tag_node_neumann!(dad, ip.node_a)
            elseif ip.reg_b == r
                _tag_node_neumann!(dad, ip.node_b)
            end
        end
        for cp in prob.contacts
            if cp.reg_a == r
                _tag_node_neumann!(dad, cp.node_a)
            elseif cp.reg_b == r
                _tag_node_neumann!(dad, cp.node_b)
            end
        end
    end
    return nothing
end

"""`B_r` such that zone `r` reads `A_r x_r + B_r q_if = b_r` (`q_b = -q_if`)."""
function _mr_flux_columns(prob::MultiRegionProblem{<:Laplace})
    regs = prob.regions
    n_if = length(prob.interfaces)
    Bs = [zeros(dad.nt, n_if) for dad in regs]
    for (k, ip) in enumerate(prob.interfaces)
        dad_a = regs[ip.reg_a]
        dad_b = regs[ip.reg_b]
        na, nb = ip.node_a, ip.node_b
        Ga, Gb = dad_a.G, dad_b.G
        if na <= size(Ga, 2)
            @views Bs[ip.reg_a][:, k] .-= Ga[:, na]
        end
        if nb <= size(Gb, 2)
            @views Bs[ip.reg_b][:, k] .+= Gb[:, nb]
        end
    end
    return Bs
end

function _scatter_mr_sol!(prob::MultiRegionProblem{<:Laplace},
        xs::Vector{<:AbstractVector}, qif::AbstractVector)
    regs = prob.regions
    for (r, dad) in enumerate(regs)
        Tfull = zeros(dad.nt)
        qfull = zeros(dad.n)
        Tfull .= xs[r]
        split_sol!(dad, Tfull, qfull)
        for (k, ip) in enumerate(prob.interfaces)
            if ip.reg_a == r
                qfull[ip.node_a] = qif[k]
            elseif ip.reg_b == r
                qfull[ip.node_b] = -qif[k]
            end
        end
        set_cache!(dad; T=Tfull, q=qfull)
    end
    return nothing
end

function _pack_mr_x(xs, qif)
    nloc = sum(length, xs)
    x = zeros(nloc + length(qif))
    o = 0
    for xr in xs
        n = length(xr)
        x[o+1:o+n] .= xr
        o += n
    end
    x[o+1:end] .= qif
    return x
end

# ---------------------------------------------------------------------------
# :dense — fill ∑n_t + n_if  (compatibilidade / equilíbrio)
# ---------------------------------------------------------------------------
function _solve_mr_dense!(prob::MultiRegionProblem{<:Laplace})
    regs = prob.regions
    nts = [dad.nt for dad in regs]
    n_if = length(prob.interfaces)
    Nloc = sum(nts)
    N = Nloc + n_if
    Ag = zeros(N, N)
    bg = zeros(N)
    off = offsets_from(nts)
    Bs = _mr_flux_columns(prob)
    for (r, dad) in enumerate(regs)
        o = off[r]
        nr = nts[r]
        Ag[o+1:o+nr, o+1:o+nr] .= dad.A
        bg[o+1:o+nr] .= dad.b
        if n_if > 0
            Ag[o+1:o+nr, Nloc+1:N] .= Bs[r]
        end
    end
    for (k, ip) in enumerate(prob.interfaces)
        row = Nloc + k
        Ag[row, off[ip.reg_a] + ip.node_a] = 1.0
        Ag[row, off[ip.reg_b] + ip.node_b] = -1.0
    end
    x = bem_linsolve(Ag, bg)
    xs = [x[off[r]+1:off[r]+nts[r]] for r in eachindex(regs)]
    qif = n_if == 0 ? Float64[] : x[Nloc+1:N]
    _scatter_mr_sol!(prob, xs, qif)
    return x
end

# ---------------------------------------------------------------------------
# :noncondensing — Kane blocked system, all zone DOFs kept
#   A_r x_r + B_r q_if = b_r
#   T_a - T_b = 0
# Factor each full A_r; Schur on q_if is n_if × n_if.
# ---------------------------------------------------------------------------
function _solve_mr_noncondensing!(prob::MultiRegionProblem{<:Laplace})
    regs = prob.regions
    n_if = length(prob.interfaces)
    nR = length(regs)
    Bs = _mr_flux_columns(prob)
    Ys = Vector{Matrix{Float64}}(undef, nR)
    zs = Vector{Vector{Float64}}(undef, nR)
    for r in 1:nR
        A = Matrix{Float64}(regs[r].A)
        b = collect(Float64, regs[r].b)
        if n_if == 0
            zs[r] = bem_linsolve(A, b)
            Ys[r] = zeros(length(b), 0)
        else
            F = lu(A)
            zs[r] = F \ b
            Ys[r] = F \ Bs[r]
        end
    end
    qif = zeros(n_if)
    if n_if > 0
        S = zeros(n_if, n_if)
        rhs = zeros(n_if)
        for (k, ip) in enumerate(prob.interfaces)
            S[k, :] .= view(Ys[ip.reg_a], ip.node_a, :) .- view(Ys[ip.reg_b], ip.node_b, :)
            rhs[k] = zs[ip.reg_a][ip.node_a] - zs[ip.reg_b][ip.node_b]
        end
        qif .= bem_linsolve(S, rhs)
    end
    xs = [zs[r] .- Ys[r] * qif for r in 1:nR]
    _scatter_mr_sol!(prob, xs, qif)
    return _pack_mr_x(xs, qif)
end

# ---------------------------------------------------------------------------
# :condense — Kane zone condensation onto shared (T_if, q_if)
# Exterior DOFs of zone r: A_ee x_e + A_ei T_if + B_e q_if = b_e
# Interface rows become the condensed map after eliminating x_e.
# ---------------------------------------------------------------------------
function _solve_mr_condense!(prob::MultiRegionProblem{<:Laplace})
    regs = prob.regions
    n_if = length(prob.interfaces)
    nR = length(regs)
    if n_if == 0
        xs = [bem_linsolve(Matrix{Float64}(regs[r].A), collect(Float64, regs[r].b)) for r in 1:nR]
        _scatter_mr_sol!(prob, xs, Float64[])
        return _pack_mr_x(xs, Float64[])
    end
    Bs = _mr_flux_columns(prob)

    # stacked condensed rows: one block of n_if_local per zone
    n_rows = 0
    zone_if = Vector{Vector{Int}}(undef, nR)
    zone_pair = Vector{Vector{Int}}(undef, nR)
    for r in 1:nR
        if_local = Int[]
        pair_of = Int[]
        for (k, ip) in enumerate(prob.interfaces)
            if ip.reg_a == r
                push!(if_local, ip.node_a)
                push!(pair_of, k)
            elseif ip.reg_b == r
                push!(if_local, ip.node_b)
                push!(pair_of, k)
            end
        end
        zone_if[r] = if_local
        zone_pair[r] = pair_of
        n_rows += length(if_local)
    end
    K = zeros(n_rows, 2 * n_if)
    f = zeros(n_rows)
    # expansion data
    e_idx = Vector{Vector{Int}}(undef, nR)
    AeeF = Vector{Any}(undef, nR)
    Aei = Vector{Matrix{Float64}}(undef, nR)
    Be = Vector{Matrix{Float64}}(undef, nR)
    be = Vector{Vector{Float64}}(undef, nR)

    row0 = 0
    for r in 1:nR
        dad = regs[r]
        A = Matrix{Float64}(dad.A)
        b = collect(Float64, dad.b)
        B = Bs[r]
        nt = dad.nt
        if_local = zone_if[r]
        pair_of = zone_pair[r]
        nl = length(if_local)
        e = setdiff(1:nt, if_local)
        e_idx[r] = e
        ne = length(e)
        if nl == 0
            # zone not on any interface: solve fully, no condensed rows
            AeeF[r] = lu(A)
            Aei[r] = zeros(nt, 0)
            Be[r] = zeros(nt, n_if)
            be[r] = b
            e_idx[r] = collect(1:nt)
            continue
        end
        Aii = A[if_local, if_local]
        Bi = B[if_local, :]
        bi = b[if_local]
        if ne == 0
            AeeF[r] = nothing
            Aei[r] = zeros(0, nl)
            Be[r] = zeros(0, n_if)
            be[r] = Float64[]
            See = Aii
            Ce = Bi
            fe = bi
        else
            F = lu(A[e, e])
            AeeF[r] = F
            Aei[r] = A[e, if_local]
            Be[r] = B[e, :]
            be[r] = b[e]
            Aie = A[if_local, e]
            RHS = hcat(Aei[r], Be[r], be[r])
            Y = F \ RHS
            See = Aii - Aie * view(Y, :, 1:nl)
            Ce = Bi - Aie * view(Y, :, nl+1:nl+n_if)
            fe = bi - Aie * view(Y, :, nl+n_if+1)
        end
        rows = row0+1:row0+nl
        @inbounds for j in 1:nl
            K[rows, pair_of[j]] .= view(See, :, j)
        end
        K[rows, n_if+1:2n_if] .= Ce
        f[rows] .= fe
        row0 += nl
    end
    y = bem_linsolve(K, f)
    Tif = y[1:n_if]
    qif = y[n_if+1:end]

    xs = Vector{Vector{Float64}}(undef, nR)
    for r in 1:nR
        nt = regs[r].nt
        xr = zeros(nt)
        if_local = zone_if[r]
        pair_of = zone_pair[r]
        if isempty(if_local)
            xr .= AeeF[r] \ be[r]
        else
            xr[if_local] .= Tif[pair_of]
            if AeeF[r] !== nothing
                rhs_e = be[r] .- Aei[r] * Tif[pair_of] .- Be[r] * qif
                xr[e_idx[r]] .= AeeF[r] \ rhs_e
            end
        end
        xs[r] = xr
    end
    _scatter_mr_sol!(prob, xs, qif)
    return _pack_mr_x(xs, qif)
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
# Multi-region assembly + solve (2-D elasticity, perfect interface)
# =============================================================================

const ElasticMR = Union{Elasticity, AnisotropicElasticity}

"""Assemble each elastic subregion (`assemble!` or [`H_G_hyper`](@ref))."""
function assemble_multiregion(prob::MultiRegionProblem{<:ElasticMR};
        npg::Int=16, bie::Symbol=:cbie, threaded::Bool=false)
    for dad in prob.regions
        if bie === :hbie
            H_G_hyper(dad; npg=npg, threaded=threaded)
        else
            H_G_full_direct(dad; npg=npg, threaded=threaded)
        end
    end
    return nothing
end

"""
    solve_multiregion!(prob::MultiRegionProblem{<:Union{Elasticity,AnisotropicElasticity}})

Perfect interface: ``u_a = u_b``, ``t_a + t_b = 0``. Dense system of size
``∑ 2 n_r + 2 n_{if}``.
"""
function solve_multiregion!(prob::MultiRegionProblem{<:ElasticMR}; apply_bc::Bool=true)
    isempty(prob.interfaces) && pair_interfaces!(prob)
    if apply_bc
        _tag_coupling_as_neumann!(prob)
        for dad in prob.regions
            applyBC(dad)
        end
    end
    return _solve_mr_dense_elast!(prob)
end

function _mr_traction_columns(prob::MultiRegionProblem{<:ElasticMR})
    regs = prob.regions
    n_if = length(prob.interfaces)
    dim = 2
    Bs = [zeros(2 * dad.n, dim * n_if) for dad in regs]
    for (k, ip) in enumerate(prob.interfaces)
        Ga, Gb = regs[ip.reg_a].G, regs[ip.reg_b].G
        for d in 1:dim
            col = dim * (k - 1) + d
            dofa = dim * (ip.node_a - 1) + d
            dofb = dim * (ip.node_b - 1) + d
            if dofa <= size(Ga, 2)
                @views Bs[ip.reg_a][:, col] .-= Ga[:, dofa]
            end
            if dofb <= size(Gb, 2)
                @views Bs[ip.reg_b][:, col] .+= Gb[:, dofb]
            end
        end
    end
    return Bs
end

function _scatter_mr_sol_elast!(prob::MultiRegionProblem{<:ElasticMR},
        xs::Vector{<:AbstractVector}, tif::AbstractVector)
    regs = prob.regions
    dim = 2
    for (r, dad) in enumerate(regs)
        nd = dim * dad.n
        u = zeros(nd)
        t = zeros(nd)
        split_sol!(dad, xs[r], u, t)
        for (k, ip) in enumerate(prob.interfaces)
            for d in 1:dim
                col = dim * (k - 1) + d
                if ip.reg_a == r
                    t[dim * (ip.node_a - 1) + d] = tif[col]
                elseif ip.reg_b == r
                    t[dim * (ip.node_b - 1) + d] = -tif[col]
                end
            end
        end
        set_cache!(dad; u=u, traction=t, T=u)
    end
    return nothing
end

function _solve_mr_dense_elast!(prob::MultiRegionProblem{<:ElasticMR})
    regs = prob.regions
    dim = 2
    nts = [dim * dad.n for dad in regs]
    n_if = length(prob.interfaces)
    n_tif = dim * n_if
    Nloc = sum(nts)
    N = Nloc + n_tif
    Ag = zeros(N, N)
    bg = zeros(N)
    off = offsets_from(nts)
    Bs = _mr_traction_columns(prob)
    for (r, dad) in enumerate(regs)
        o = off[r]
        nr = nts[r]
        Ag[o+1:o+nr, o+1:o+nr] .= dad.A
        bg[o+1:o+nr] .= dad.b
        if n_tif > 0
            Ag[o+1:o+nr, Nloc+1:N] .= Bs[r]
        end
    end
    for (k, ip) in enumerate(prob.interfaces)
        oa, ob = off[ip.reg_a], off[ip.reg_b]
        for d in 1:dim
            row = Nloc + dim * (k - 1) + d
            Ag[row, oa + dim * (ip.node_a - 1) + d] = 1.0
            Ag[row, ob + dim * (ip.node_b - 1) + d] = -1.0
        end
    end
    x = bem_linsolve(Ag, bg)
    xs = [x[off[r]+1:off[r]+nts[r]] for r in eachindex(regs)]
    tif = n_tif == 0 ? Float64[] : x[Nloc+1:N]
    _scatter_mr_sol_elast!(prob, xs, tif)
    return x
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

# =============================================================================
# Frictional contact solvers (included)
# =============================================================================
include("ContactCommon.jl")
include("ContactActiveSet.jl")
include("ContactSSN.jl")
include("ContactProjectedNewton.jl")
include("ContactFriction.jl")
