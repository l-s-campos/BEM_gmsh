# Singular Boundary Method (SBM)
# Coincident source/collocation on Γ. Origin intensity factors:
#   u_ii = (1/L_m) ∫_{Γ_m} U(x^m, s) dΓ
#   q_ii = (1/L_m) [ I − κ Σ_{n≠m} L_n T(s^n, x^m) ]
#   I    = ∫_{Γ_m} [ T(x^m, s) + T(s, x^m) ] dΓ
# Guiggiani on the BEM parent element of collocation m (log U, CPV T). κ = +1 interior.
#
# Package flux q = −k ∂u/∂n: U* = −log(R)/(2πk), T = (r·n_x)/(2π R²).

export SBMData, sbm_from_bemdata, assemble_sbm!, solve_sbm!
export solve_sbm_laplace, sbm_rel_error, sbm_rel_error_internal
export compare_sbm_bem, origin_intensity_factors!
export sbm_eval_u, sbm_eval_grad, sbm_eval_q, sbm_eval_internal!

"""
    SBMData

SBM on a formatted Laplace `BEMdata` only (no meshless node/normal constructor).
OIFs use the parent BEM element of each `format2d` collocation.
"""
mutable struct SBMData
    dad::BEMdata{<:Laplace}
    nodes::AbstractVector       # `dad.Nodes` (format2d GL collocation)
    normals::AbstractVector
    internal::AbstractVector    # field eval only
    BC::AbstractVector
    BV::AbstractVector
    k::Float64
    H::Matrix{Float64}          # Neumann kernel matrix (incl. q_ii)
    G::Matrix{Float64}          # Dirichlet kernel matrix (incl. u_ii)
    α::Vector{Float64}
    T::Vector{Float64}
    q::Vector{Float64}
    Ti::Vector{Float64}
    qi::Vector{Float64}
    u_ii::Vector{Float64}       # Dirichlet OIFs
    q_ii::Vector{Float64}       # Neumann OIFs
    name::String
    lengths::Vector{Float64}
    col_el::Vector{Int}         # parent BEM element of each collocation
    col_loc::Vector{Int}        # local index on that element
    kappa::Float64              # +1 interior, −1 exterior
end

Base.length(d::SBMData) = length(d.nodes)

function Base.show(io::IO, d::SBMData)
    print(io, "SBMData(\"$(d.name)\"; n=$(length(d.nodes)), ni=$(length(d.internal)), k=$(d.k))")
end

# =============================================================================
# Build
# =============================================================================

"""
    sbm_from_bemdata(dad; internal=nothing) -> SBMData

Wraps `dad` without copying collocation: `d.nodes === dad.Nodes` (GL points
from `format2d`). OIFs use the same `dad.elements`.
`internal` defaults to mesh internals (eval only).
"""
function sbm_from_bemdata(dad::BEMdata{<:Laplace}; internal=nothing)
    dad.dimension == 2 || error("SBM supports 2D Laplace only")
    ints = if internal === nothing
        dad.ni > 0 ? dad.internalNodes : Point2D[]
    elseif internal === false
        Point2D[]
    else
        Point2D[Point2D(p) for p in internal]
    end
    g = sbm_bind_geometry(dad)
    return SBMData(
        dad,
        dad.Nodes, dad.Normal, ints, dad.BC, dad.BV, float(dad.properties.k),
        zeros(0, 0), zeros(0, 0),
        Float64[], Float64[], Float64[], Float64[], Float64[],
        Float64[], Float64[],
        string(dad.name),
        g.lengths, g.col_el, g.col_loc, 1.0,
    )
end

# =============================================================================
# Kernels  (package conventions)
# =============================================================================

@inline function _sbm_U(r::Point2D, k::Float64)
    R = hypot(r[1], r[2])
    R < 1e-30 && return 0.0
    return -log(R) / (2π * k)
end

"""Q*_pkg = −k ∂U*/∂n_x = (r·n_x)/(2π R²)."""
@inline function _sbm_Q_field(r::Point2D, n_field::Point2D)
    R2 = r[1]^2 + r[2]^2
    R2 < 1e-30 && return 0.0
    return (r[1] * n_field[1] + r[2] * n_field[2]) / (R2 * 2π)
end

@inline function _sbm_gradU(r::Point2D, k::Float64)
    R2 = r[1]^2 + r[2]^2
    R2 < 1e-30 && return Point2D(0.0, 0.0)
    s = -1.0 / (2π * k * R2)
    return Point2D(s * r[1], s * r[2])
end

# =============================================================================
# Origin intensity factors (Chen–Gu 2012)
# =============================================================================

"""
    origin_intensity_factors!(d; kappa=d.kappa) -> (u_ii, q_ii)

Integral OIFs (Guiggiani on the BEM parent element Γ_m of collocation `m`):

```
u_ii = (1/L_m) ∫_{Γ_m} U(x^m, s) dΓ
q_ii = (1/L_m) [ I − κ Σ_{n≠m} L_n T(s^n, x^m) ]
I    = ∫_{Γ_m} [ T(x^m, s) + T(s, x^m) ] dΓ
```

`κ = +1` interior, `−1` exterior. `L_n` is the parent-element length over
the number of collocation nodes on that element (`Length/2` for linear).
`T(x,s)` is the package flux kernel at `x` (normal at `x`).
"""
function origin_intensity_factors!(d::SBMData; kappa=nothing)
    κ = float(something(kappa, d.kappa))
    d.kappa = κ
    n = length(d.nodes)
    k = d.k
    nodes, normals, L = d.nodes, d.normals, d.lengths
    qsi, w = gausslegendre(20)

    uii = zeros(n)
    qii = zeros(n)
    @inbounds for m in 1:n
        Lm = L[m]
        xm, nm = nodes[m], normals[m]
        IU, II = _sbm_laplace_gamma_integrals(d, m; qsi=qsi, w=w)
        uii[m] = IU / Lm
        sT = 0.0
        for nn in 1:n
            nn == m && continue
            # T(s^n, x^m): flux at source n due to point force at collocation m
            sT += L[nn] * _sbm_Q_field(nodes[nn] - xm, normals[nn])
        end
        qii[m] = (II - κ * sT) / Lm
    end
    d.u_ii = uii
    d.q_ii = qii
    return uii, qii
end

"""Guiggiani ∫_{Γ_m} U dΓ and ∫_{Γ_m} [T(x^m,s)+T(s,x^m)] dΓ on the BEM element."""
function _sbm_laplace_gamma_integrals(d::SBMData, m::Integer; qsi, w)
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
    dad = d.dad
    Ig, Ih = guiggiani_GH(ξm; order_G=0, order_H=-1, qsi=qsi, w=w) do η
        ξ = sξ * η + cξ
        pg, J, nrm = sbm_geom_at(geo, poly, ξ)
        J < 1e-30 && return 0.0, 0.0
        rxs = xm - pg                       # x^m − s
        hypot(rxs[1], rxs[2]) < 1e-30 && return 0.0, 0.0
        wJ = J * sξ
        U, Txs = fundamental(dad, rxs, nm)
        _, Tsx = fundamental(dad, pg - xm, nrm)
        return U * wJ, (Txs + Tsx) * wJ
    end
    return Ig, Ih
end

# =============================================================================
# Assemble / solve
# =============================================================================

"""
    assemble_sbm!(d) -> d

Build N×N `G`, `H` with integral/Guiggiani OIFs on the diagonal.
"""
function assemble_sbm!(d::SBMData; kappa=nothing)
    n = length(d.nodes)
    k = d.k
    nodes = d.nodes
    normals = d.normals

    uii, qii = origin_intensity_factors!(d; kappa=kappa)

    G = zeros(n, n)
    H = zeros(n, n)
    @inbounds for i in 1:n
        xi = nodes[i]
        ni = normals[i]
        for j in 1:n
            if i == j
                G[i, i] = uii[i]
                H[i, i] = qii[i]
                continue
            end
            r = xi - nodes[j]
            G[i, j] = _sbm_U(r, k)
            H[i, j] = _sbm_Q_field(r, ni)
        end
    end
    d.G = G
    d.H = H
    return d
end

function assemble_sbm!(dad::BEMdata{<:Laplace}; kwargs...)
    d = sbm_from_bemdata(dad)
    assemble_sbm!(d; kwargs...)
    dad.cache.extras[:sbm] = d
    return d
end

function origin_intensity_factors!(dad::BEMdata{<:Laplace}; kwargs...)
    return origin_intensity_factors!(sbm_from_bemdata(dad); kwargs...)
end

"""
    solve_sbm!(d) -> T

Mixed BC collocation (paper §2.3): row i is G or H according to BC.
Pure Neumann: pin Σα = 0 (replace one row).
"""
function solve_sbm!(d::SBMData)
    n = length(d.nodes)
    size(d.G, 1) == n || error("call assemble_sbm! first")
    BC, BV = d.BC, d.BV
    G, H = d.G, d.H

    A = zeros(n, n)
    b = zeros(n)
    n_dir = 0
    @inbounds for i in 1:n
        if BC[i] == 0
            n_dir += 1
            for j in 1:n
                A[i, j] = G[i, j]
            end
            b[i] = BV[i]
        else
            for j in 1:n
                A[i, j] = H[i, j]
            end
            b[i] = BV[i]
        end
    end

    if n_dir == 0
        # pure Neumann: replace last equation by Σ α = 0
        A[n, :] .= 1.0
        b[n] = 0.0
    end

    α = A \ b
    T = G * α
    q = H * α
    @inbounds for i in 1:n
        if BC[i] == 0
            T[i] = BV[i]
        else
            q[i] = BV[i]
        end
    end
    d.α = α
    d.T = T
    d.q = q
    return T
end

function solve_sbm!(dad::BEMdata{<:Laplace}; kwargs...)
    d = haskey(dad.cache.extras, :sbm) ? dad.cache.extras[:sbm] : assemble_sbm!(dad; kwargs...)
    size(d.G, 1) == dad.n || assemble_sbm!(d; kwargs...)
    solve_sbm!(d)
    isempty(d.internal) || sbm_eval_internal!(d)
    dad.cache.extras[:sbm] = d
    return d
end

"""
    solve_sbm_laplace(dad; internal=nothing, kwargs...) -> SBMData

Singular boundary method (Chen–Gu) on a formatted Laplace `BEMdata`:
build origin intensity factors, mix Dirichlet/Neumann rows, solve for
source intensities `α`. Result is cached as `dad.cache.extras[:sbm]`.
"""
function solve_sbm_laplace(dad::BEMdata{<:Laplace};
                           internal=nothing, kwargs...)
    d = sbm_from_bemdata(dad; internal=internal)
    assemble_sbm!(d; kwargs...)
    solve_sbm!(d)
    isempty(d.internal) || sbm_eval_internal!(d)
    dad.cache.extras[:sbm] = d
    return d
end

# =============================================================================
# Field evaluation
# =============================================================================

function sbm_eval_u(d::SBMData, x::Point2D)
    isempty(d.α) && error("call solve_sbm! first")
    k = d.k
    s = 0.0
    @inbounds for j in eachindex(d.nodes)
        s += d.α[j] * _sbm_U(x - d.nodes[j], k)
    end
    return s
end
sbm_eval_u(d::SBMData, x::AbstractVector{<:Real}) = sbm_eval_u(d, Point2D(x[1], x[2]))

function sbm_eval_grad(d::SBMData, x::Point2D)
    isempty(d.α) && error("call solve_sbm! first")
    k = d.k
    gx = 0.0
    gy = 0.0
    @inbounds for j in eachindex(d.nodes)
        g = _sbm_gradU(x - d.nodes[j], k)
        gx += d.α[j] * g[1]
        gy += d.α[j] * g[2]
    end
    return Point2D(gx, gy)
end
sbm_eval_grad(d::SBMData, x::AbstractVector{<:Real}) = sbm_eval_grad(d, Point2D(x[1], x[2]))

function sbm_eval_q(d::SBMData, x::Point2D, nrm::Point2D)
    isempty(d.α) && error("call solve_sbm! first")
    s = 0.0
    @inbounds for j in eachindex(d.nodes)
        s += d.α[j] * _sbm_Q_field(x - d.nodes[j], nrm)
    end
    return s
end
sbm_eval_q(d::SBMData, x::AbstractVector{<:Real}, nrm) =
    sbm_eval_q(d, Point2D(x[1], x[2]), Point2D(nrm[1], nrm[2]))

function sbm_eval_internal!(d::SBMData; points=nothing, normals=nothing)
    isempty(d.α) && error("call solve_sbm! first")
    pts = points === nothing ? d.internal : Point2D[Point2D(p) for p in points]
    ni = length(pts)
    Ti = Vector{Float64}(undef, ni)
    @inbounds for i in 1:ni
        Ti[i] = sbm_eval_u(d, pts[i])
    end
    d.internal = pts
    d.Ti = Ti
    if normals !== nothing
        length(normals) == ni || throw(ArgumentError("normals length mismatch"))
        qi = Vector{Float64}(undef, ni)
        @inbounds for i in 1:ni
            qi[i] = sbm_eval_q(d, pts[i], Point2D(normals[i][1], normals[i][2]))
        end
        d.qi = qi
    else
        d.qi = Float64[]
    end
    return d
end

function sbm_rel_error(d::SBMData, Tana)
    num = 0.0
    den = 0.0
    @inbounds for (i, p) in enumerate(d.nodes)
        te = try
            float(Tana(p[1], p[2]))
        catch
            float(Tana(p))
        end
        num += (d.T[i] - te)^2
        den += te^2
    end
    return sqrt(num / max(den, eps()))
end
sbm_rel_error(d::SBMData, ana::AnalyticalSolution) = sbm_rel_error(d, p -> ana.u(p))

function sbm_rel_error_internal(d::SBMData, Tana)
    isempty(d.Ti) && error("call sbm_eval_internal! first")
    num = 0.0
    den = 0.0
    @inbounds for (i, p) in enumerate(d.internal)
        te = try
            float(Tana(p[1], p[2]))
        catch
            float(Tana(p))
        end
        num += (d.Ti[i] - te)^2
        den += te^2
    end
    return sqrt(num / max(den, eps()))
end
sbm_rel_error_internal(d::SBMData, ana::AnalyticalSolution) =
    sbm_rel_error_internal(d, p -> ana.u(p))

function compare_sbm_bem(dad::BEMdata{<:Laplace}, ana=nothing;
                         npg::Integer=16, threaded::Bool=false, internal=nothing)
    t_bem_asm = @elapsed H_G_full_direct(dad; npg=npg, threaded=threaded)
    t_bem_sol = @elapsed solve(dad)
    T_bem = copy(dad.T[1:dad.n])
    q_bem = has_cache(dad, :q) ? copy(dad.q[1:dad.n]) : Float64[]
    if ana !== nothing
        ana_obj = ana isa AnalyticalSolution ? ana :
                  AnalyticalSolution("cmp", p -> float(ana(p[1], p[2])))
        dad.cache.analytical = ana_obj
    end
    err_bem = ana !== nothing ? rel_error(dad) : NaN

    d = sbm_from_bemdata(dad; internal=internal)
    t_sbm_asm = @elapsed assemble_sbm!(d)
    t_sbm_sol = @elapsed begin
        solve_sbm!(d)
        isempty(d.internal) || sbm_eval_internal!(d)
    end
    err_sbm = ana === nothing ? NaN : sbm_rel_error(d, ana)
    err_sbm_i = (ana !== nothing && !isempty(d.Ti)) ? sbm_rel_error_internal(d, ana) : NaN
    diff_T = norm(d.T .- T_bem) / max(norm(T_bem), eps())
    return (;
        n=dad.n, ni=length(d.internal),
        err_bem, err_sbm, err_sbm_i, diff_T,
        t_bem_asm, t_bem_sol, t_sbm_asm, t_sbm_sol,
        t_bem=t_bem_asm + t_bem_sol, t_sbm=t_sbm_asm + t_sbm_sol,
        T_bem, T_sbm=copy(d.T), q_bem, q_sbm=copy(d.q),
        Ti_sbm=copy(d.Ti),
        u_ii=copy(d.u_ii), q_ii=copy(d.q_ii), sbm=d, dad,
    )
end
