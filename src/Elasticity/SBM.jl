# Singular Boundary Method — 2D isotropic elasticity (Kelvin)
# Coincident source/collocation on Γ. Origin intensity factors:
#   u_ij = (1/L_m) ∫_{Γ_m} U_ij(x^m, s) dΓ
#   t_ii = (1/L_m) [ I_ij − κ Σ_{n≠m} L_n T_ij(s^n, x^m) ]
#   I_ij = ∫_{Γ_m} [ T_ij(x^m, s) + T_ij(s, x^m) ] dΓ
# Guiggiani on the BEM parent element of collocation m (log U, CPV T). κ = +1 interior.

export ElasticSBMData, sbm_from_bemdata, assemble_sbm!, solve_sbm!
export solve_sbm_elasticity, origin_intensity_factors!
export sbm_eval_u, sbm_eval_t, sbm_eval_internal!
export sbm_rel_error, sbm_rel_error_internal

"""SBM on a 2D elasticity `BEMdata` only (OIFs on the BEM parent element)."""
mutable struct ElasticSBMData
    dad::BEMdata{<:Elasticity}
    nodes::AbstractVector       # `dad.Nodes` (format2d GL collocation)
    normals::AbstractVector
    lengths::Vector{Float64}
    internal::AbstractVector
    BC::AbstractVector
    BV::AbstractVector
    props::Elasticity
    G::Matrix{Float64}          # 2n×2n single-layer (U, incl. U_ii)
    H::Matrix{Float64}          # 2n×2n traction (T, incl. T_ii)
    α::Vector{Float64}
    u::Vector{Float64}
    t::Vector{Float64}
    ui::Vector{Float64}
    U_ii::Vector{SMatrix{2,2,Float64,4}}
    T_ii::Vector{SMatrix{2,2,Float64,4}}
    name::String
    col_el::Vector{Int}
    col_loc::Vector{Int}
    kappa::Float64              # +1 interior, −1 exterior
end

Base.length(d::ElasticSBMData) = length(d.nodes)
function Base.show(io::IO, d::ElasticSBMData)
    print(io, "ElasticSBMData(\"$(d.name)\"; n=$(length(d.nodes)), ni=$(length(d.internal)))")
end

@inline _edof(i::Integer, d::Integer) = 2 * (i - 1) + d

function sbm_from_bemdata(dad::BEMdata{<:Elasticity}; internal=nothing)
    dad.dimension == 2 || error("elastic SBM is 2D only")
    ints = if internal === nothing
        dad.ni > 0 ? dad.internalNodes : Point2D[]
    elseif internal === false
        Point2D[]
    else
        Point2D[Point2D(p) for p in internal]
    end
    g = sbm_bind_geometry(dad)
    return ElasticSBMData(
        dad,
        dad.Nodes, dad.Normal, g.lengths, ints, dad.BC, dad.BV, dad.properties,
        zeros(0, 0), zeros(0, 0),
        Float64[], Float64[], Float64[], Float64[],
        SMatrix{2,2,Float64,4}[], SMatrix{2,2,Float64,4}[],
        string(dad.name),
        g.col_el, g.col_loc, 1.0,
    )
end

@inline function _kelvin_UT(props::Elasticity, x::Point2D, s::Point2D, n::Point2D)
    kp = fundamental(props, x - s, n)
    return _to_smat(kp.U), _to_smat(kp.T)
end

function _fill_H!(H, d::ElasticSBMData, Tii)
    n = length(d.nodes)
    props, nodes, normals = d.props, d.nodes, d.normals
    @inbounds for i in 1:n
        xi, ni = nodes[i], normals[i]
        ii = 2i - 1
        for j in 1:n
            jj = 2j - 1
            if i == j
                H[ii:ii+1, jj:jj+1] .= Tii[i]
            else
                _, T = _kelvin_UT(props, xi, nodes[j], ni)
                H[ii:ii+1, jj:jj+1] .= T
            end
        end
    end
    return H
end

function _fill_G!(G, d::ElasticSBMData, Uii)
    n = length(d.nodes)
    props, nodes, normals = d.props, d.nodes, d.normals
    @inbounds for i in 1:n
        xi, ni = nodes[i], normals[i]
        ii = 2i - 1
        for j in 1:n
            jj = 2j - 1
            if i == j
                G[ii:ii+1, jj:jj+1] .= Uii[i]
            else
                U, _ = _kelvin_UT(props, xi, nodes[j], ni)
                G[ii:ii+1, jj:jj+1] .= U
            end
        end
    end
    return G
end

"""3 rigid-mode columns (2 translations + rotation) for a 2n traction system."""
function _rigid_C(nodes)
    n = length(nodes)
    C = zeros(2n, 3)
    @inbounds for i in 1:n
        C[2i - 1, 1] = 1.0
        C[2i, 2] = 1.0
        C[2i - 1, 3] = -nodes[i][2]
        C[2i, 3] = nodes[i][1]
    end
    return C
end

"""
    origin_intensity_factors!(d::ElasticSBMData; kappa=d.kappa) -> (U_ii, T_ii)

Integral OIFs (Guiggiani on Γ_m):

```
u_ij = (1/L_m) ∫_{Γ_m} U_ij(x^m, s) dΓ
T_ii = (1/L_m) [ I_ij − κ Σ_{n≠m} L_n T_ij(s^n, x^m) ]
I_ij = ∫_{Γ_m} [ T_ij(x^m, s) + T_ij(s, x^m) ] dΓ
```
"""
function origin_intensity_factors!(d::ElasticSBMData; kappa=nothing)
    κ = float(something(kappa, d.kappa))
    d.kappa = κ
    n = length(d.nodes)
    L = d.lengths
    qsi, w = gausslegendre(20)
    Z = zero(SMatrix{2,2,Float64,4})
    Uii = Vector{SMatrix{2,2,Float64,4}}(undef, n)
    Tii = Vector{SMatrix{2,2,Float64,4}}(undef, n)
    @inbounds for m in 1:n
        Lm = L[m]
        IU, II = _sbm_elastic_gamma_integrals(d, m; qsi=qsi, w=w)
        Uii[m] = IU / Lm
        sT = Z
        xm = d.nodes[m]
        for nn in 1:n
            nn == m && continue
            _, T = _kelvin_UT(d.props, d.nodes[nn], xm, d.normals[nn])
            sT += L[nn] * T
        end
        Tii[m] = (II - κ * sT) / Lm
    end
    d.U_ii = Uii
    d.T_ii = Tii
    return Uii, Tii
end

"""Guiggiani ∫_{Γ_m} U dΓ and ∫_{Γ_m} [T(x^m,s)+T(s,x^m)] dΓ on the BEM element."""
function _sbm_elastic_gamma_integrals(d::ElasticSBMData, m::Integer; qsi, w)
    geo, poly, ξm0 = sbm_element_geom(d, m)
    ξa, ξb = sbm_xi_interval(d, m)
    sξ = (ξb - ξa) / 2
    if sξ < 1e-14
        ξa, ξb, sξ = -1.0, 1.0, 1.0
    end
    cξ = (ξb + ξa) / 2
    ξm = clamp((ξm0 - cξ) / sξ, nextfloat(-1.0), prevfloat(1.0))
    xm = d.nodes[m]
    nm = d.normals[m]
    Z = zero(SMatrix{2,2,Float64,4})
    Ig, Ih = guiggiani_GH(ξm; order_G=0, order_H=-1, qsi=qsi, w=w) do η
        ξ = sξ * η + cξ
        pg, J, nrm = sbm_geom_at(geo, poly, ξ)
        J < 1e-30 && return Z, Z
        hypot((xm - pg)[1], (xm - pg)[2]) < 1e-30 && return Z, Z
        wJ = J * sξ
        Uxs, Txs = _kelvin_UT(d.props, xm, pg, nm)
        _, Tsx = _kelvin_UT(d.props, pg, xm, nrm)
        return Uxs * wJ, (Txs + Tsx) * wJ
    end
    return Ig, Ih
end

function assemble_sbm!(d::ElasticSBMData; kappa=nothing)
    n = length(d.nodes)
    Uii, Tii = origin_intensity_factors!(d; kappa=kappa)
    G = zeros(2n, 2n)
    H = zeros(2n, 2n)
    _fill_G!(G, d, Uii)
    _fill_H!(H, d, Tii)
    d.G = G
    d.H = H
    return d
end

function assemble_sbm!(dad::BEMdata{<:Elasticity}; kwargs...)
    d = sbm_from_bemdata(dad)
    assemble_sbm!(d; kwargs...)
    dad.cache.extras[:sbm] = d
    return d
end

function origin_intensity_factors!(dad::BEMdata{<:Elasticity}; kwargs...)
    return origin_intensity_factors!(sbm_from_bemdata(dad); kwargs...)
end

function solve_sbm!(d::ElasticSBMData)
    n = length(d.nodes)
    nd = 2n
    size(d.G, 1) == nd || error("call assemble_sbm! first")
    BC, BV = d.BC, d.BV
    G, H = d.G, d.H
    A = zeros(nd, nd)
    b = zeros(nd)
    n_dir = 0
    @inbounds for k in 1:nd
        if BC[k] == 0
            n_dir += 1
            A[k, :] .= view(G, k, :)
            b[k] = BV[k]
        else
            A[k, :] .= view(H, k, :)
            b[k] = BV[k]
        end
    end
    if n_dir < 3
        C = _rigid_C(d.nodes)
        A = [A C; C' zeros(3, 3)]
        b = [b; zeros(3)]
        x = A \ b
        α = x[1:nd]
    else
        α = A \ b
    end
    u = G * α
    t = H * α
    @inbounds for k in 1:nd
        if BC[k] == 0
            u[k] = BV[k]
        else
            t[k] = BV[k]
        end
    end
    d.α = α
    d.u = u
    d.t = t
    return u
end

function solve_sbm_elasticity(dad::BEMdata{<:Elasticity};
        internal=nothing, kwargs...)
    d = sbm_from_bemdata(dad; internal=internal)
    assemble_sbm!(d; kwargs...)
    solve_sbm!(d)
    isempty(d.internal) || sbm_eval_internal!(d)
    dad.cache.extras[:sbm] = d
    return d
end

function sbm_eval_u(d::ElasticSBMData, x::Point2D)
    isempty(d.α) && error("call solve_sbm! first")
    s = SVector(0.0, 0.0)
    n0 = d.normals[1]
    @inbounds for j in eachindex(d.nodes)
        U, _ = _kelvin_UT(d.props, x, d.nodes[j], n0)
        s += U * SVector(d.α[2j - 1], d.α[2j])
    end
    return s
end
sbm_eval_u(d::ElasticSBMData, x::AbstractVector{<:Real}) =
    sbm_eval_u(d, Point2D(x[1], x[2]))

function sbm_eval_t(d::ElasticSBMData, x::Point2D, nrm::Point2D)
    isempty(d.α) && error("call solve_sbm! first")
    s = SVector(0.0, 0.0)
    @inbounds for j in eachindex(d.nodes)
        _, T = _kelvin_UT(d.props, x, d.nodes[j], nrm)
        s += T * SVector(d.α[2j - 1], d.α[2j])
    end
    return s
end

function sbm_eval_internal!(d::ElasticSBMData; points=nothing)
    isempty(d.α) && error("call solve_sbm! first")
    pts = points === nothing ? d.internal : Point2D[Point2D(p) for p in points]
    ui = zeros(2 * length(pts))
    @inbounds for i in eachindex(pts)
        uv = sbm_eval_u(d, pts[i])
        ui[2i - 1] = uv[1]
        ui[2i] = uv[2]
    end
    d.internal = pts
    d.ui = ui
    return d
end

function sbm_rel_error(d::ElasticSBMData, ana::AnalyticalSolution)
    num = 0.0
    den = 0.0
    @inbounds for (i, p) in enumerate(d.nodes)
        ue = ana.u(p)
        num += (d.u[2i - 1] - ue[1])^2 + (d.u[2i] - ue[2])^2
        den += ue[1]^2 + ue[2]^2
    end
    return sqrt(num / max(den, eps()))
end

function sbm_rel_error_internal(d::ElasticSBMData, ana::AnalyticalSolution)
    isempty(d.ui) && error("call sbm_eval_internal! first")
    num = 0.0
    den = 0.0
    @inbounds for (i, p) in enumerate(d.internal)
        ue = ana.u(p)
        num += (d.ui[2i - 1] - ue[1])^2 + (d.ui[2i] - ue[2])^2
        den += ue[1]^2 + ue[2]^2
    end
    return sqrt(num / max(den, eps()))
end
