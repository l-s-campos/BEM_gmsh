# Topological derivative of the thermal potential energy for Laplace
# (homogeneous Neumann hole): DT = k |∇T|²  (Novotny et al. 2003, Pacheco 2020).

export topological_derivative, thermal_conductance, design_objective
export boundary_grad_T, interior_grad_T

"""
    boundary_grad_T(dad) -> Vector{<:SVector}

`∇T` at every boundary collocation node from `q = -k ∂T/∂n` and the
tangential derivative of `T` (2-D: `Dmat / J`; 3-D: surface metric).
"""
function boundary_grad_T(dad::BEMdata{<:Laplace})
    return dad.dimension == 2 ? _boundary_grad_T_2d(dad) : _boundary_grad_T_3d(dad)
end

function _boundary_grad_T_2d(dad::BEMdata{<:Laplace})
    k = dad.properties.k
    n = dad.n
    g = Vector{SVector{2,Float64}}(undef, n)
    T = dad.T
    q = dad.q
    poly = dad.element_type
    D = poly.Dmat
    filled = falses(n)
    for el in dad.elements
        idx = el.index
        nn = length(idx)
        Tel = T[idx]
        dTdξ = D * Tel
        for a in 1:nn
            i = idx[a]
            filled[i] && continue
            J = el.Jacobian[a]
            J < 1e-16 && continue
            n̂ = dad.Normal[i]
            t̂ = SVector(-n̂[2], n̂[1])          # increasing ξ
            dTdn = -q[i] / k
            dTdt = dTdξ[a] / J
            g[i] = dTdn * n̂ + dTdt * t̂
            filled[i] = true
        end
    end
    @inbounds for i in 1:n
        filled[i] && continue
        n̂ = dad.Normal[i]
        g[i] = (-q[i] / k) * n̂
    end
    return g
end

function _boundary_grad_T_3d(dad::BEMdata{<:Laplace})
    k = dad.properties.k
    n = dad.n
    g = Vector{SVector{3,Float64}}(undef, n)
    T = dad.T
    q = dad.q
    poly = dad.element_type
    ξs = poly.nodes
    nξ = length(ξs)
    filled = falses(n)
    for el in dad.elements
        idx = el.index
        X = dad.Nodes[idx]
        nn = length(idx)
        Tel = T[idx]
        for a in 1:nn
            i = idx[a]
            filled[i] && continue
            iξ = (a - 1) % nξ + 1
            iη = (a - 1) ÷ nξ + 1
            ξ = ξs[iξ]
            η = ξs[min(iη, nξ)]
            L, Lξ, Lη = shapefun2D(poly, poly, ξ, η)
            xξ = zero(X[1])
            xη = zero(X[1])
            Tξ = 0.0
            Tη = 0.0
            nN = size(L, 2)
            for j in 1:nN
                xξ += Lξ[1, j] * X[j]
                xη += Lη[1, j] * X[j]
                Tξ += Lξ[1, j] * Tel[j]
                Tη += Lη[1, j] * Tel[j]
            end
            nvec = dad.Normal[i]
            nlen = norm(nvec)
            nlen < 1e-16 && continue
            n̂ = nvec / nlen
            G = hcat(SVector(xξ[1], xξ[2], xξ[3]), SVector(xη[1], xη[2], xη[3]))
            GtG = G' * G
            ∇Γ = if det(GtG) < 1e-20
                zero(SVector{3,Float64})
            else
                αβ = GtG \ SVector(Tξ, Tη)
                αβ[1] * xξ + αβ[2] * xη
            end
            dTdn = -q[i] / k
            g[i] = dTdn * n̂ + ∇Γ
            filled[i] = true
        end
    end
    @inbounds for i in 1:n
        filled[i] && continue
        n̂ = dad.Normal[i]
        nlen = norm(n̂)
        n̂ = nlen > 1e-16 ? n̂ / nlen : n̂
        g[i] = (-q[i] / k) * n̂
    end
    return g
end

"""
    interior_grad_T(dad, pts=dad.internalNodes) -> Vector{<:SVector}

`∇T` at interior (or arbitrary off-boundary) points from the differentiated
representation formula, using [`fundamental_grad`](@ref).
"""
function interior_grad_T(dad::BEMdata{<:Laplace}, pts::AbstractVector=dad.internalNodes)
    npts = length(pts)
    PT = eltype(dad.Nodes)
    g = fill(zero(PT), npts)
    npts == 0 && return g
    T = dad.T
    q = dad.q
    elems = dad.elements
    has_cache(dad, :qsi) || _init_quadrature!(dad, 16)
    @inbounds for ip in 1:npts
        pf = pts[ip]
        acc = zero(PT)
        for el in elems
            xj = dad.Nodes[el.index]
            N, r, nrm, wwJ = _quad_geom(dad, el, xj, pf)
            nn = size(N, 2)
            for iq in eachindex(wwJ)
                dU, dT = fundamental_grad(dad, r[iq], nrm[iq])
                wi = wwJ[iq]
                for a in 1:nn
                    Nj = N[iq, a] * wi
                    ja = el.index[a]
                    acc += (dU * q[ja] - dT * T[ja]) * Nj
                end
            end
        end
        # fundamental_grad is ∇_d of (U,H); the interior BIE for ∇T at d
        # is the negative of that contraction (2-D and 3-D).
        g[ip] = -acc
    end
    return g
end

"""Polarization factor for an insulating (Neumann) hole: 2-D circle `1`, 3-D sphere `3/2`."""
_laplace_dt_factor(dim::Integer) = dim == 3 ? 1.5 : 1.0

"""
    topological_derivative(dad) -> (DT_boundary, DT_internal)

Homogeneous Neumann hole, Laplace: `DT = c k |∇T|²` with `c = 1` (2-D) or
`3/2` (3-D spherical cavity; Novotny).
"""
function topological_derivative(dad::BEMdata{<:Laplace})
    k = dad.properties.k
    c = _laplace_dt_factor(dad.dimension)
    gb = boundary_grad_T(dad)
    gi = interior_grad_T(dad)
    DTb = [c * k * sum(abs2, g) for g in gb]
    DTi = [c * k * sum(abs2, g) for g in gi]
    return DTb, DTi
end

"""
    thermal_conductance(dad) -> Float64

`J = ∫_Ω k |∇T|² dΩ = -∫_Γ T q dΓ` (package flux `q = -k ∂T/∂n`, no sources).
"""
function thermal_conductance(dad::BEMdata{<:Laplace})
    J = 0.0
    w = dad.elem_weight
    T = dad.T
    q = dad.q
    for el in dad.elements
        for a in eachindex(el.index)
            i = el.index[a]
            J -= T[i] * q[i] * el.Jacobian[a] * w[a]
        end
    end
    return J
end

"""Objective: thermal conductance (Laplace) or compliance (elasticity)."""
design_objective(dad::BEMdata{<:Laplace}) = thermal_conductance(dad)
