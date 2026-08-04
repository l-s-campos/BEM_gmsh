# Dual interpolation BFM for 2D elasticity
# Zhang, Lin, Dong — Eur. J. Mech. A/Solids 73 (2019) 500–511
#
# First layer: continuous Lagrange on source (Gauss) + virtual (endpoints)
# Collocation: Kelvin BIE at source nodes only
# Second layer: MLS/Shepard or 2D RBF condensation of virtual DOFs
#   u_v = Φu u_s ,  t_v = Φt t_s   (per component, Cartesian support)

export DiBFMElastData, dibfm_elast_from_bemdata
export assemble_dibfm_elast!, solve_dibfm_elast!
export dibfm_elast_rel_error, solve_dibfm_elasticity
export compare_dibfm_elasticity

"""
DiBFM mesh + operators for 2D isotropic elasticity.
"""
mutable struct DiBFMElastData
    source_pos::Vector{Point2D}
    all_pos::Vector{Point2D}
    normals_src::Vector{Point2D}
    elements::Vector{Vector{Int}}      # into all_pos: [vL, s…, vR]
    src_of_elem::Vector{Vector{Int}}
    entity::Vector{Int}
    src_global::Vector{Int}
    virt_global::Vector{Int}
    BC::Vector{Int}                    # length 2 n_s
    BV::Vector{Float64}
    props::Elasticity
    Φu::Matrix{Float64}                # n_all × n_s
    Φt::Matrix{Float64}
    H::Matrix{Float64}                 # 2n_s × 2 n_all
    G::Matrix{Float64}
    Hs::Matrix{Float64}                # 2n_s × 2n_s
    Gs::Matrix{Float64}
    u::Vector{Float64}                 # 2n_s
    t::Vector{Float64}
    second_layer::Symbol               # :mls | :rbf
end

# =============================================================================
# Build from format2d BEMdata
# =============================================================================

function dibfm_elast_from_bemdata(dad::BEMdata{<:Elasticity})
    dad.dimension == 2 || error("DiBFM elasticity is 2D only")
    n_el = length(dad.elements)
    n_s_per = length(dad.elements[1].index)
    qsi, _ = discontinuous_nodes_weights(n_s_per - 1)

    # endpoints via linear fit through Gauss collocation (same as DLIM)
    endpoints = Vector{NTuple{2,Point2D}}(undef, n_el)
    for (e, el) in enumerate(dad.elements)
        Xs = [Point2D(dad.Nodes[i]) for i in el.index]
        A = ones(n_s_per, 2)
        A[:, 2] .= qsi
        cx = A \ [p[1] for p in Xs]
        cy = A \ [p[2] for p in Xs]
        endpoints[e] = (Point2D(cx[1] - cx[2], cy[1] - cy[2]),
                        Point2D(cx[1] + cx[2], cy[1] + cy[2]))
    end

    virt_pos = Point2D[]
    virt_key = Dict{NTuple{2,Int},Int}()
    function virt_id(p)
        key = (round(Int, p[1] * 1e9), round(Int, p[2] * 1e9))
        haskey(virt_key, key) && return virt_key[key]
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

    source_pos = Point2D[Point2D(dad.Nodes[i]) for i in 1:dad.n]
    n_s = length(source_pos)
    n_v = length(virt_pos)
    all_pos = vcat(source_pos, virt_pos)

    elements = Vector{Vector{Int}}(undef, n_el)
    src_of_elem = Vector{Vector{Int}}(undef, n_el)
    entity = zeros(Int, n_el)
    for (e, el) in enumerate(dad.elements)
        sids = collect(el.index)
        elements[e] = vcat(n_s + el_vL[e], sids, n_s + el_vR[e])
        src_of_elem[e] = sids
        entity[e] = el.Region
    end

    normals_src = Point2D[Point2D(dad.Normal[i]) for i in 1:dad.n]
    BC = copy(dad.BC[1:2n_s])
    BV = copy(dad.BV[1:2n_s])
    Φu = Matrix{Float64}(I, n_s + n_v, n_s)
    Φt = copy(Φu)

    return DiBFMElastData(source_pos, all_pos, normals_src, elements, src_of_elem,
        entity, collect(1:n_s), collect((n_s+1):(n_s+n_v)), BC, BV, dad.properties,
        Φu, Φt, zeros(0, 0), zeros(0, 0), zeros(0, 0), zeros(0, 0),
        zeros(2n_s), zeros(2n_s), :mls)
end

# =============================================================================
# Second-layer condensation (scalar Φ applied per component)
# =============================================================================

function build_elast_condensation!(d::DiBFMElastData; method=:mls, radius_factor=2.5,
    rbf=PHS(3; poly_deg=0))
    method = Symbol(lowercase(string(method)))
    method in (:mls, :shepard, :rbf) || error("use :mls or :rbf")
    method == :shepard && (method = :mls)
    d.second_layer = method

    n_s = length(d.source_pos)
    n_v = length(d.virt_global)
    n_all = n_s + n_v
    Φu = zeros(n_all, n_s)
    Φt = zeros(n_all, n_s)
    @inbounds for i in 1:n_s
        Φu[i, i] = 1.0
        Φt[i, i] = 1.0
    end

    ent_sources = Dict{Int,Vector{Int}}()
    for (e, sids) in enumerate(d.src_of_elem)
        append!(get!(ent_sources, d.entity[e], Int[]), sids)
    end
    foreach(unique!, values(ent_sources))

    virt_ents = [Int[] for _ in 1:n_v]
    for (e, el) in enumerate(d.elements)
        push!(virt_ents[el[1]-n_s], d.entity[e])
        push!(virt_ents[el[end]-n_s], d.entity[e])
    end
    foreach(unique!, virt_ents)

    hs = [norm(d.all_pos[el[end]] - d.all_pos[el[1]]) for el in d.elements]
    hmed = median(hs)
    R = radius_factor * hmed

    for v in 1:n_v
        iv = n_s + v
        pv = d.all_pos[iv]
        ents = virt_ents[v]
        # displacement continuous across corner
        src_u = Int[]
        for ent in ents
            append!(src_u, get(ent_sources, ent, Int[]))
        end
        unique!(src_u)
        # traction discontinuous at corner → one-sided
        src_t = isempty(ents) ? src_u : unique(get(ent_sources, ents[1], Int[]))

        if method == :mls
            Φu[iv, :] .= _shepard_row(pv, d.source_pos, src_u, R)
            Φt[iv, :] .= _shepard_row(pv, d.source_pos, src_t, R)
        else
            ids_u = _elast_nbr(d, pv, src_u, R)
            ids_t = _elast_nbr(d, pv, src_t, R)
            Φu[iv, :] .= rbf_cardinal(pv, d.source_pos, ids_u; basis=rbf, ridge=1e-12)
            Φt[iv, :] .= rbf_cardinal(pv, d.source_pos, ids_t; basis=rbf, ridge=1e-12)
        end
    end
    d.Φu = Φu
    d.Φt = Φt
    return d
end

function _elast_nbr(d, pv, pool, R; nmin=4)
    ids = Int[j for j in pool if norm(d.source_pos[j] - pv) < R]
    if length(ids) < nmin && !isempty(pool)
        dists = sort([(norm(d.source_pos[j] - pv), j) for j in pool]; by=first)
        for (_, j) in dists
            j in ids || push!(ids, j)
            length(ids) >= min(nmin + 2, length(pool)) && break
        end
    end
    return ids
end

"""Block-diagonal expansion: scalar Φ (n_all×n_s) → P (2n_all×2n_s) node-major."""
function _block_phi(Φ::AbstractMatrix)
    n_all, n_s = size(Φ)
    P = zeros(2n_all, 2n_s)
    @inbounds for j in 1:n_s, i in 1:n_all
        a = Φ[i, j]
        P[2i-1, 2j-1] = a
        P[2i, 2j] = a
    end
    return P
end

# =============================================================================
# Assembly
# =============================================================================

function assemble_dibfm_elast!(d::DiBFMElastData; npg=12, method=:mls, kwargs...)
    build_elast_condensation!(d; method=method, kwargs...)
    n_s = length(d.source_pos)
    n_all = length(d.all_pos)
    H = zeros(2n_s, 2n_all)
    G = zeros(2n_s, 2n_all)
    qsi_g, w_g = gausslegendre(npg)
    props = d.props

    @showprogress "DiBFM elasticity assemble" for i in 1:n_s
        pf = d.source_pos[i]
        rows = 2i-1:2i
        for el in d.elements
            X = [d.all_pos[j] for j in el]
            n_loc = length(el)
            ξnodes = _elem_xi(n_loc - 2)
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
                R < 1e-14 && continue
                kp = fundamental(props, rvec, n̂)
                U = _to_smat(kp.U)
                T = _to_smat(kp.T)
                wJ = J * w_g[ig]
                for a in 1:n_loc
                    ja = el[a]
                    cols = 2ja-1:2ja
                    H[rows, cols] .+= T .* (N[a] * wJ)
                    G[rows, cols] .+= U .* (N[a] * wJ)
                end
            end
        end
    end
    d.H = H
    d.G = G

    # condense: H * P_u * u_s  with free term 0.5 I on source diagonal blocks
    Pu = _block_phi(d.Φu)
    Pt = _block_phi(d.Φt)
    Hs = H * Pu
    Gs = G * Pt
    @inbounds for i in 1:n_s
        Hs[2i-1:2i, 2i-1:2i] .+= 0.5 * I(2)
    end
    d.Hs = Hs
    d.Gs = Gs
    return d
end

# =============================================================================
# Solve
# =============================================================================

function solve_dibfm_elast!(d::DiBFMElastData)
    n = length(d.source_pos)
    ndof = 2n
    A = copy(d.Hs)
    B = copy(d.Gs)
    b = zeros(ndof)
    BC, BV = d.BC, d.BV
    @inbounds for j in 1:ndof
        if BC[j] == 0
            colH = A[:, j]
            colG = B[:, j]
            A[:, j] = -colG
            b .-= colH .* BV[j]
        else
            b .+= B[:, j] .* BV[j]
        end
    end
    x = A \ b
    u = zeros(ndof)
    t = zeros(ndof)
    @inbounds for j in 1:ndof
        if BC[j] == 0
            u[j] = BV[j]
            t[j] = x[j]
        else
            t[j] = BV[j]
            u[j] = x[j]
        end
    end
    d.u = u
    d.t = t
    return u
end

function dibfm_elast_rel_error(d::DiBFMElastData, uana::Function)
    # uana(x,y) -> SVector(ux,uy)
    num = 0.0
    den = 0.0
    @inbounds for (i, p) in enumerate(d.source_pos)
        ua = uana(p[1], p[2])
        un = SVector(d.u[2i-1], d.u[2i])
        num += sum(abs2, un - ua)
        den += sum(abs2, ua)
    end
    return sqrt(num / max(den, eps()))
end

function solve_dibfm_elasticity(dad::BEMdata{<:Elasticity}; npg=12, method=:mls, kwargs...)
    d = dibfm_elast_from_bemdata(dad)
    assemble_dibfm_elast!(d; npg=npg, method=method, kwargs...)
    solve_dibfm_elast!(d)
    return d
end

"""Compare standard elasticity BEM vs DiBFM-MLS vs DiBFM-RBF."""
function compare_dibfm_elasticity(dad::BEMdata{<:Elasticity}, uana::Function; npg=12)
    t0 = @elapsed begin
        H_G_full_direct(dad; npg=npg, threaded=false)
        solve(dad)
    end
    err0 = rel_error(dad)

    t_m = @elapsed (d_m = solve_dibfm_elasticity(dad; npg=npg, method=:mls))
    err_m = dibfm_elast_rel_error(d_m, uana)

    t_r = @elapsed (d_r = solve_dibfm_elasticity(dad; npg=npg, method=:rbf))
    err_r = dibfm_elast_rel_error(d_r, uana)

    return (
        err_std=err0, err_dibfm_mls=err_m, err_dibfm_rbf=err_r,
        time_std=t0, time_mls=t_m, time_rbf=t_r,
        n_s=length(d_m.source_pos), n_v=length(d_m.virt_global),
    )
end
