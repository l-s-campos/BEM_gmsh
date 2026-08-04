# Double-layer interpolation BEM (DLI-BEM) for 2D Laplace
# Zhang, Lin, Dong, Ju — Appl. Math. Modelling 51 (2017) 250–269
#
# First layer: continuous polynomial interpolation on source + virtual nodes
# Collocation: source nodes only (interior Gauss points of each element)
# Second layer: MLS (IIMLS-style) relates virtual-node values to source values
# → square system on source DOFs, C⁰ continuity of potential, discontinuous flux at corners

export DLIMData, dlim_from_bemdata, assemble_dlim!, solve_dlim!
export dlim_rel_error, solve_dlim_laplace, build_condensation!
export compare_dlim_second_layer

"""
    DLIMData

Double-layer interpolation mesh for 2D potential problems.

# Fields
- `source_pos`: collocation / source node coordinates (size `n_s`)
- `all_pos`: source + virtual node coordinates (size `n_all`)
- `is_source`: mask into `all_pos`
- `elements`: each element lists indices into `all_pos` ordered
  `[virtual_left, source_1, …, source_{n_s}, virtual_right]`
- `src_of_elem`: source-node global indices (1…n_s) per element
- `Φu`, `Φq`: condensation matrices (`n_all × n_s`) for potential and flux
- `H`, `G`: collocation matrices (`n_s × n_all`) before condensation
- `Hs`, `Gs`: condensed (`n_s × n_s`)
"""
mutable struct DLIMData
    source_pos::Vector{Point2D}
    all_pos::Vector{Point2D}
    normals_src::Vector{Point2D}
    is_source::Vector{Bool}
    elements::Vector{Vector{Int}}   # indices into all_pos
    src_global::Vector{Int}         # all_pos index of each source (length n_s)
    virt_global::Vector{Int}        # all_pos index of each virtual
    src_of_elem::Vector{Vector{Int}} # source numbers 1:n_s per element
    entity::Vector{Int}             # geometric entity / Region per element
    BC::Vector{Int}                 # BC type at source nodes
    BV::Vector{Float64}
    k::Float64
    Φu::Matrix{Float64}
    Φq::Matrix{Float64}
    H::Matrix{Float64}
    G::Matrix{Float64}
    Hs::Matrix{Float64}
    Gs::Matrix{Float64}
    T::Vector{Float64}              # potential at source nodes
    q::Vector{Float64}              # flux at source nodes
    second_layer::Symbol            # :mls (Shepard) or :rbf
end

# =============================================================================
# Build from discontinuous format2d BEMdata
# =============================================================================

"""
    dlim_from_bemdata(dad::BEMdata{<:Laplace}) -> DLIMData

Construct a DLIM mesh: Gauss collocation nodes of `dad` become **source** nodes;
element endpoints become shared **virtual** nodes.
"""
function dlim_from_bemdata(dad::BEMdata{<:Laplace})
    dad.dimension == 2 || error("DLIM implemented for 2D only")
    k = dad.properties.k
    n_el = length(dad.elements)
    n_s_per = length(dad.elements[1].index)

    # --- recover geometric endpoints of each element (ξ = ±1) ---
    # source collocation at Gauss ξ; geometry via Lagrange through sources is not
    # exact for ±1 — use chord endpoints from first/last source with normal offset 0
    # Better: extrapolate linear map from Gauss range to [-1,1]
    qsi, _ = discontinuous_nodes_weights(n_s_per - 1)

    endpoints = Vector{NTuple{2,Point2D}}(undef, n_el)  # (left, right) per elem
    for (e, el) in enumerate(dad.elements)
        Xs = [Point2D(dad.Nodes[i]) for i in el.index]
        # fit linear geometry: x(ξ) ≈ c0 + c1*ξ using least squares on Gauss pts
        A = ones(n_s_per, 2)
        A[:, 2] .= qsi
        # solve for each coordinate
        cx = A \ [p[1] for p in Xs]
        cy = A \ [p[2] for p in Xs]
        left = Point2D(cx[1] - cx[2], cy[1] - cy[2])   # ξ=-1
        right = Point2D(cx[1] + cx[2], cy[1] + cy[2]) # ξ=+1
        endpoints[e] = (left, right)
    end

    # --- unique virtual nodes by position ---
    virt_pos = Point2D[]
    virt_key = Dict{NTuple{2,Int},Int}()  # quantized key → index
    function virt_id(p::Point2D)
        key = (round(Int, p[1] * 1e9), round(Int, p[2] * 1e9))
        if haskey(virt_key, key)
            return virt_key[key]
        end
        push!(virt_pos, p)
        id = length(virt_pos)
        virt_key[key] = id
        return id
    end

    el_vL = zeros(Int, n_el)
    el_vR = zeros(Int, n_el)
    for e in 1:n_el
        el_vL[e] = virt_id(endpoints[e][1])
        el_vR[e] = virt_id(endpoints[e][2])
    end
    n_v = length(virt_pos)

    # all_pos = [sources…; virtuals…]
    source_pos = Point2D[Point2D(dad.Nodes[i]) for i in 1:dad.n]
    n_s = length(source_pos)
    all_pos = vcat(source_pos, virt_pos)
    is_source = vcat(trues(n_s), falses(n_v))
    src_global = collect(1:n_s)
    virt_global = collect((n_s + 1):(n_s + n_v))

    # element connectivity into all_pos: [vL, s1..sns, vR]
    elements = Vector{Vector{Int}}(undef, n_el)
    src_of_elem = Vector{Vector{Int}}(undef, n_el)
    entity = zeros(Int, n_el)
    for (e, el) in enumerate(dad.elements)
        sids = collect(el.index)                     # already 1…n_s global source ids
        elements[e] = vcat(n_s + el_vL[e], sids, n_s + el_vR[e])
        src_of_elem[e] = sids
        entity[e] = el.Region
    end

    normals_src = Point2D[Point2D(dad.Normal[i]) for i in 1:dad.n]
    BC = copy(dad.BC[1:n_s])
    BV = copy(dad.BV[1:n_s])

    Φu = Matrix{Float64}(I, n_s + n_v, n_s)   # filled later
    Φq = copy(Φu)
    return DLIMData(source_pos, all_pos, normals_src, is_source, elements, src_global,
        virt_global, src_of_elem, entity, BC, BV, float(k), Φu, Φq,
        zeros(0, 0), zeros(0, 0), zeros(0, 0), zeros(0, 0), zeros(n_s), zeros(n_s), :mls)
end

# =============================================================================
# Continuous Lagrange on element nodes (first-layer)
# =============================================================================

function _lagrange_N(ξ, ξnodes)
    m = length(ξnodes)
    N = ones(typeof(ξ), m)
    @inbounds for i in 1:m, j in 1:m
        if j != i
            N[i] *= (ξ - ξnodes[j]) / (ξnodes[i] - ξnodes[j])
        end
    end
    return N
end

function _lagrange_dN(ξ, ξnodes)
    m = length(ξnodes)
    dN = zeros(typeof(ξ), m)
    @inbounds for i in 1:m
        s = zero(ξ)
        for k in 1:m
            k == i && continue
            term = 1 / (ξnodes[i] - ξnodes[k])
            for j in 1:m
                if j != i && j != k
                    term *= (ξ - ξnodes[j]) / (ξnodes[i] - ξnodes[j])
                end
            end
            s += term
        end
        dN[i] = s
    end
    return dN
end

"""Natural coordinates of element nodes: virtuals at ±1, sources at Gauss ξ."""
function _elem_xi(n_src::Int)
    qsi, _ = discontinuous_nodes_weights(n_src - 1)
    return vcat(-1.0, collect(qsi), 1.0)
end

# =============================================================================
# Second-layer condensation (virtual ← source): MLS/Shepard or RBF
# =============================================================================

"""
    build_condensation!(d; method=:mls, radius_factor=2.5, rbf=PHS(3; poly_deg=1))

Build Φ (`n_all × n_s`) so `u_all ≈ Φ * u_source` (and likewise for flux).

# Methods
- `:mls` — compact Shepard / inverse-distance MLS (default, paper-style local)
- `:rbf` — global/local RBF (PHS) interpolant on neighbouring source nodes

At corners (multiple entities), flux uses one-sided support (discontinuous q);
potential stays continuous across entities.
"""
function build_condensation!(d::DLIMData; method=:mls, radius_factor=2.5,
    rbf=PHS(3; poly_deg=1))

    method = Symbol(lowercase(string(method)))
    method in (:mls, :shepard, :rbf) || error("unknown second-layer method $method (use :mls or :rbf)")
    method == :shepard && (method = :mls)
    d.second_layer = method

    n_s = length(d.source_pos)
    n_all = length(d.all_pos)
    Φu = zeros(n_all, n_s)
    Φq = zeros(n_all, n_s)
    @inbounds for i in 1:n_s
        Φu[i, i] = 1.0
        Φq[i, i] = 1.0
    end

    n_v = length(d.virt_global)
    ent_sources = Dict{Int,Vector{Int}}()
    for (e, sids) in enumerate(d.src_of_elem)
        ent = d.entity[e]
        lst = get!(ent_sources, ent, Int[])
        append!(lst, sids)
    end
    for (_, lst) in ent_sources
        unique!(lst)
    end

    virt_ents = [Int[] for _ in 1:n_v]
    for (e, el) in enumerate(d.elements)
        vL = el[1] - n_s
        vR = el[end] - n_s
        push!(virt_ents[vL], d.entity[e])
        push!(virt_ents[vR], d.entity[e])
    end
    foreach(unique!, virt_ents)

    hs = [norm(d.all_pos[el[end]] - d.all_pos[el[1]]) for el in d.elements]
    hmed = median(hs)
    R = radius_factor * hmed

    for v in 1:n_v
        iv = n_s + v
        pv = d.all_pos[iv]
        ents = virt_ents[v]
        src_u = Int[]
        for ent in ents
            append!(src_u, get(ent_sources, ent, Int[]))
        end
        unique!(src_u)
        src_q = isempty(ents) ? src_u : unique(get(ent_sources, ents[1], Int[]))

        if method == :mls
            Φu[iv, :] .= _shepard_row(pv, d.source_pos, src_u, R)
            Φq[iv, :] .= _shepard_row(pv, d.source_pos, src_q, R)
        else
            Φu[iv, :] .= _rbf_row(pv, d.source_pos, src_u, rbf; R=R)
            Φq[iv, :] .= _rbf_row(pv, d.source_pos, src_q, rbf; R=R)
        end
    end
    d.Φu = Φu
    d.Φq = Φq
    return d
end

function _shepard_row(pv, source_pos, src_ids, R)
    n_s = length(source_pos)
    row = zeros(n_s)
    isempty(src_ids) && return row
    wsum = 0.0
    ww = Float64[]
    ids = Int[]
    for j in src_ids
        r = norm(source_pos[j] - pv)
        if r < 1e-14
            row[j] = 1.0
            return row
        end
        if r < R
            w = (1 - r / R)^2 * (1 / r)
            push!(ww, w)
            push!(ids, j)
            wsum += w
        end
    end
    if wsum == 0
        dists = [(norm(source_pos[j] - pv), j) for j in src_ids]
        sort!(dists; by=first)
        take = dists[1:min(2, end)]
        wsum = sum(1 / max(d, 1e-14) for (d, _) in take)
        for (d, j) in take
            row[j] = (1 / max(d, 1e-14)) / wsum
        end
        return row
    end
    for (w, j) in zip(ww, ids)
        row[j] = w / wsum
    end
    return row
end

"""RBF row via shared [`rbf_cardinal`](@ref) (1D edge cloud in 2D ambient coords)."""
function _rbf_row(pv, source_pos, src_ids, rbf; R=Inf)
    n_s = length(source_pos)
    row = zeros(n_s)
    isempty(src_ids) && return row

    ids = Int[]
    for j in src_ids
        r = norm(source_pos[j] - pv)
        if r < 1e-14
            row[j] = 1.0
            return row
        end
        if r < R || length(src_ids) <= 8
            push!(ids, j)
        end
    end
    if length(ids) < 2
        return _shepard_row(pv, source_pos, src_ids, isfinite(R) ? R : 1.0)
    end
    try
        return rbf_cardinal(pv, source_pos, ids; basis=rbf, ridge=1e-12)
    catch
        return _shepard_row(pv, source_pos, src_ids, isfinite(R) ? R : 1.0)
    end
end

# =============================================================================
# Assembly — collocate at sources, integrate with continuous first-layer N
# =============================================================================

"""
    assemble_dlim!(d; npg=12)

Assemble `H`, `G` (source × all) then condense with Φ → `Hs`, `Gs`.
"""
function assemble_dlim!(d::DLIMData; npg=12, method=:mls, kwargs...)
    build_condensation!(d; method=method, kwargs...)
    n_s = length(d.source_pos)
    n_all = length(d.all_pos)
    H = zeros(n_s, n_all)
    G = zeros(n_s, n_all)
    qsi_g, w_g = gausslegendre(npg)
    k = d.k

    @showprogress "DLIM assemble H,G" for i in 1:n_s
        pf = d.source_pos[i]
        for (e, el) in enumerate(d.elements)
            X = [d.all_pos[j] for j in el]
            n_loc = length(el)
            ξnodes = _elem_xi(n_loc - 2)   # n_src = n_loc-2
            # singular if source belongs to this element
            on_el = i in d.src_of_elem[e]
            for (ig, ξ) in enumerate(qsi_g)
                N = _lagrange_N(ξ, ξnodes)
                dN = _lagrange_dN(ξ, ξnodes)
                x = sum(N[a] * X[a] for a in 1:n_loc)
                dx = sum(dN[a] * X[a] for a in 1:n_loc)
                J = norm(dx)
                J < 1e-16 && continue
                n̂ = Point2D(dx[2] / J, -dx[1] / J)
                rvec = x - pf
                R = norm(rvec)
                if R < 1e-14
                    continue
                end
                # Laplace kernels (same convention as package: G=Tast, H=Qast)
                # fundamental returns (G, H) = (-log(R)/(2πk), r·n/(R² 2π))
                Gker = -log(R) / (2π * k)
                Hker = dot(rvec, n̂) / (R^2 * 2π)
                wJ = J * w_g[ig]
                for a in 1:n_loc
                    ja = el[a]
                    H[i, ja] += Hker * N[a] * wJ
                    G[i, ja] += Gker * N[a] * wJ
                end
            end
            if on_el
                # free term 1/2 on the diagonal source contribution handled globally
            end
        end
    end

    # free term: c = 1/2 on smooth boundary for source collocation
    @inbounds for i in 1:n_s
        # move diagonal of H to free-term form: H_ii = 0.5 - sum_{j≠i mapped}
        # After condensation we apply free term on Hs
        s = sum(H[i, :])
        # Standard: H_ii += 0.5, but continuous interpolation means free term
        # is applied on the source identity after condensation:
        # (0.5 I + H Φu) u_s = G Φq q_s
        # Store raw integral H without free term; add 0.5 I after condensation
        nothing
    end

    d.H = H
    d.G = G
    # condense
    d.Hs = H * d.Φu
    d.Gs = G * d.Φq
    @inbounds for i in 1:n_s
        d.Hs[i, i] += 0.5
    end
    return d
end

# =============================================================================
# BC + solve
# =============================================================================

"""
    solve_dlim!(d) -> T_source

Apply mixed BC at **source** nodes and solve condensed system.
"""
function solve_dlim!(d::DLIMData)
    n = length(d.source_pos)
    size(d.Hs, 1) == n || error("call assemble_dlim! first")
    A = copy(d.Hs)
    B = copy(d.Gs)
    b = zeros(n)
    BC, BV = d.BC, d.BV
    @inbounds for j in 1:n
        if BC[j] == 0
            # Dirichlet: unknown is q_j — swap columns
            # A u = G q  → after swap:  -G col for unknown q, RHS -= H col * ū
            colH = A[:, j]
            colG = B[:, j]
            A[:, j] = -colG
            B[:, j] = -colH   # not used further
            b .-= colH .* BV[j]
        else
            # Neumann known q
            b .+= B[:, j] .* BV[j]
        end
    end
    x = A \ b
    T = zeros(n)
    q = zeros(n)
    @inbounds for j in 1:n
        if BC[j] == 0
            T[j] = BV[j]
            q[j] = x[j]
        else
            q[j] = BV[j]
            T[j] = x[j]
        end
    end
    d.T = T
    d.q = q
    return T
end

"""Relative L2 error of source-node potential vs analytical function `Tana(x,y)`."""
function dlim_rel_error(d::DLIMData, Tana::Function)
    num = 0.0
    den = 0.0
    @inbounds for (i, p) in enumerate(d.source_pos)
        te = Tana(p[1], p[2])
        num += (d.T[i] - te)^2
        den += te^2
    end
    return sqrt(num / max(den, eps()))
end

"""Convenience: build, assemble, solve DLIM Laplace from a formatted `dad`."""
function solve_dlim_laplace(dad::BEMdata{<:Laplace}; npg=12, method=:mls, kwargs...)
    d = dlim_from_bemdata(dad)
    assemble_dlim!(d; npg=npg, method=method, kwargs...)
    solve_dlim!(d)
    return d
end

"""
    compare_dlim_second_layer(dad; Tana, npg=12) -> NamedTuple

Run DLIM twice (MLS/Shepard vs RBF second layer) and compare source-node errors
against analytical `Tana(x,y)`.
"""
function compare_dlim_second_layer(dad::BEMdata{<:Laplace}, Tana::Function;
    npg=12, rbf=PHS(3; poly_deg=1))

    t_mls = @elapsed begin
        d_mls = solve_dlim_laplace(dad; npg=npg, method=:mls)
    end
    t_rbf = @elapsed begin
        d_rbf = solve_dlim_laplace(dad; npg=npg, method=:rbf, rbf=rbf)
    end
    err_mls = dlim_rel_error(d_mls, Tana)
    err_rbf = dlim_rel_error(d_rbf, Tana)
    # max |T_mls - T_rbf| on sources
    diff_max = maximum(abs, d_mls.T .- d_rbf.T)
    return (
        err_mls=err_mls,
        err_rbf=err_rbf,
        diff_max=diff_max,
        time_mls=t_mls,
        time_rbf=t_rbf,
        n_s=length(d_mls.source_pos),
        n_v=length(d_mls.virt_global),
        T_mls=d_mls.T,
        T_rbf=d_rbf.T,
    )
end
