# Plane-stress topological derivative (Novotny / Carretero–Neches / Coelho 2021).
#
# DT = 2/(1+ν) σ:ε + (3ν-1)/(2(1-ν²)) (tr σ)(tr ε)
# with ε_zz = -ν(σxx+σyy)/E and tensor shear ε_xy = (1+ν) τ_xy / E.

export boundary_stress_strain, interior_stress_strain
export elastic_compliance, plane_stress_DT, isotropic_3d_DT

"""Voigt (σxx, σyy, τxy) and (εxx, εyy, εzz, εxy) at every boundary collocation node."""
function boundary_stress_strain(dad::BEMdata{<:Elasticity})
    E = dad.properties.E
    ν = dad.properties.nu
    n = dad.n
    u = dad.u
    t = dad.traction
    σ = zeros(n, 3)
    ε = zeros(n, 4)
    poly = dad.element_type
    D = poly.Dmat
    filled = falses(n)
    dim = 2
    for el in dad.elements
        idx = el.index
        nn = length(idx)
        ux = [u[dim * (idx[a] - 1) + 1] for a in 1:nn]
        uy = [u[dim * (idx[a] - 1) + 2] for a in 1:nn]
        dux = D * ux
        duy = D * uy
        for a in 1:nn
            i = idx[a]
            filled[i] && continue
            J = el.Jacobian[a]
            J < 1e-16 && continue
            n̂ = dad.Normal[i]
            # tangent = increasing ξ, consistent with tan2normal = (ty, -tx)
            t̂ = SVector(-n̂[2], n̂[1])
            tx = t[dim * (i - 1) + 1]
            ty = t[dim * (i - 1) + 2]
            # local: e1 = n, e2 = t  (Coelho lij)
            s11 = n̂[1] * tx + n̂[2] * ty          # σ_nn
            s12 = t̂[1] * tx + t̂[2] * ty          # σ_nt
            ut_nodes = [t̂[1] * ux[b] + t̂[2] * uy[b] for b in 1:nn]
            dutdξ = (D * ut_nodes)[a]
            e22 = dutdξ / J                         # ε_tt
            s22 = E * e22 + ν * s11                 # plane-stress σ_tt
            # rotate local (n,t) → global
            # lji maps local components to global: col1=n, col2=t
            σxx = n̂[1]^2 * s11 + t̂[1]^2 * s22 + 2 * n̂[1] * t̂[1] * s12
            σyy = n̂[2]^2 * s11 + t̂[2]^2 * s22 + 2 * n̂[2] * t̂[2] * s12
            τxy = n̂[1] * n̂[2] * s11 + t̂[1] * t̂[2] * s22 + (n̂[1] * t̂[2] + n̂[2] * t̂[1]) * s12
            σ[i, :] .= (σxx, σyy, τxy)
            εxx = (σxx - ν * σyy) / E
            εyy = (σyy - ν * σxx) / E
            εzz = -ν * (σxx + σyy) / E
            εxy = (1 + ν) * τxy / E
            ε[i, :] .= (εxx, εyy, εzz, εxy)
            filled[i] = true
        end
    end
    return σ, ε
end

"""Interior Voigt stress/strain. 2-D: `n×3` / `n×4` (tensor εxy); 3-D: both `n×6` (engineering γ)."""
function interior_stress_strain(dad::BEMdata{<:Elasticity}, pts::AbstractVector=dad.internalNodes)
    return dad.dimension == 2 ? _interior_ss_2d(dad, pts) : _interior_ss_3d(dad, pts)
end

function _interior_ss_2d(dad::BEMdata{<:Elasticity}, pts)
    npts = length(pts)
    σ = zeros(npts, 3)
    ε = zeros(npts, 4)
    npts == 0 && return σ, ε
    E = dad.properties.E
    ν = dad.properties.nu
    u = dad.u
    t = dad.traction
    dim = 2
    has_cache(dad, :qsi) || _init_quadrature!(dad, 16)
    @inbounds for ip in 1:npts
        pf = pts[ip]
        sxx = syy = txy = 0.0
        for el in dad.elements
            xj = dad.Nodes[el.index]
            N, r, nrm, wwJ = _quad_geom(dad, el, xj, pf)
            nn = size(N, 2)
            for iq in eachindex(wwJ)
                sk = fundamental_stress(dad, r[iq], nrm[iq])
                DD, SS = sk.D, sk.S
                wi = wwJ[iq]
                for a in 1:nn
                    ja = el.index[a]
                    ux = u[dim * (ja - 1) + 1]
                    uy = u[dim * (ja - 1) + 2]
                    tx = t[dim * (ja - 1) + 1]
                    ty = t[dim * (ja - 1) + 2]
                    Na = N[iq, a] * wi
                    # σ_ij = D_kij t_k - S_kij u_k
                    sxx += (DD[1, 1, 1] * tx + DD[2, 1, 1] * ty - SS[1, 1, 1] * ux - SS[2, 1, 1] * uy) * Na
                    syy += (DD[1, 2, 2] * tx + DD[2, 2, 2] * ty - SS[1, 2, 2] * ux - SS[2, 2, 2] * uy) * Na
                    txy += (DD[1, 1, 2] * tx + DD[2, 1, 2] * ty - SS[1, 1, 2] * ux - SS[2, 1, 2] * uy) * Na
                end
            end
        end
        σ[ip, :] .= (sxx, syy, txy)
        εxx = (sxx - ν * syy) / E
        εyy = (syy - ν * sxx) / E
        εzz = -ν * (sxx + syy) / E
        εxy = (1 + ν) * txy / E
        ε[ip, :] .= (εxx, εyy, εzz, εxy)
    end
    return σ, ε
end

function _interior_ss_3d(dad::BEMdata{<:Elasticity}, pts)
    npts = length(pts)
    σ = zeros(npts, 6)
    ε = zeros(npts, 6)
    npts == 0 && return σ, ε
    E = dad.properties.E
    ν = dad.properties.nu
    u = dad.u
    t = dad.traction
    has_cache(dad, :qsi) || _init_quadrature!(dad, 16)
    @inbounds for ip in 1:npts
        pf = pts[ip]
        s11 = s22 = s33 = s23 = s13 = s12 = 0.0
        for el in dad.elements
            xj = dad.Nodes[el.index]
            N, r, nrm, wwJ = _quad_geom(dad, el, xj, pf)
            nn = size(N, 2)
            for iq in eachindex(wwJ)
                sk = fundamental_stress(dad, r[iq], nrm[iq])
                DD, SS = sk.D, sk.S
                wi = wwJ[iq]
                for a in 1:nn
                    ja = el.index[a]
                    ux = u[3ja - 2]; uy = u[3ja - 1]; uz = u[3ja]
                    tx = t[3ja - 2]; ty = t[3ja - 1]; tz = t[3ja]
                    Na = N[iq, a] * wi
                    s11 += (DD[1, 1, 1] * tx + DD[2, 1, 1] * ty + DD[3, 1, 1] * tz -
                            SS[1, 1, 1] * ux - SS[2, 1, 1] * uy - SS[3, 1, 1] * uz) * Na
                    s22 += (DD[1, 2, 2] * tx + DD[2, 2, 2] * ty + DD[3, 2, 2] * tz -
                            SS[1, 2, 2] * ux - SS[2, 2, 2] * uy - SS[3, 2, 2] * uz) * Na
                    s33 += (DD[1, 3, 3] * tx + DD[2, 3, 3] * ty + DD[3, 3, 3] * tz -
                            SS[1, 3, 3] * ux - SS[2, 3, 3] * uy - SS[3, 3, 3] * uz) * Na
                    s23 += (DD[1, 2, 3] * tx + DD[2, 2, 3] * ty + DD[3, 2, 3] * tz -
                            SS[1, 2, 3] * ux - SS[2, 2, 3] * uy - SS[3, 2, 3] * uz) * Na
                    s13 += (DD[1, 1, 3] * tx + DD[2, 1, 3] * ty + DD[3, 1, 3] * tz -
                            SS[1, 1, 3] * ux - SS[2, 1, 3] * uy - SS[3, 1, 3] * uz) * Na
                    s12 += (DD[1, 1, 2] * tx + DD[2, 1, 2] * ty + DD[3, 1, 2] * tz -
                            SS[1, 1, 2] * ux - SS[2, 1, 2] * uy - SS[3, 1, 2] * uz) * Na
                end
            end
        end
        σ[ip, :] .= (s11, s22, s33, s23, s13, s12)
        ε[ip, 1] = (s11 - ν * (s22 + s33)) / E
        ε[ip, 2] = (s22 - ν * (s11 + s33)) / E
        ε[ip, 3] = (s33 - ν * (s11 + s22)) / E
        ε[ip, 4] = 2 * (1 + ν) * s23 / E
        ε[ip, 5] = 2 * (1 + ν) * s13 / E
        ε[ip, 6] = 2 * (1 + ν) * s12 / E
    end
    return σ, ε
end

"""Plane-stress DT (Coelho / Novotny) from Voigt σ, ε rows."""
function plane_stress_DT(σ::AbstractMatrix, ε::AbstractMatrix, ν::Real)
    n = size(σ, 1)
    DT = zeros(n)
    cons = (3ν - 1) / (2 * (1 - ν^2))
    c1 = 2 / (1 + ν)
    @inbounds for i in 1:n
        dt = c1 * (σ[i, 1] * ε[i, 1] + 2 * σ[i, 3] * ε[i, 4] + σ[i, 2] * ε[i, 2])
        dt2 = cons * (σ[i, 1] + σ[i, 2]) * (ε[i, 1] + ε[i, 2] + ε[i, 3])
        DT[i] = dt + dt2
    end
    return DT
end

"""3-D spherical-cavity DT (Novotny). `σ, ε` are Voigt with engineering shear `γ`."""
function isotropic_3d_DT(σ::AbstractMatrix, ε::AbstractMatrix, ν::Real)
    n = size(σ, 1)
    DT = zeros(n)
    c = 3 * (1 - ν) / (2 * (7 - 5ν))
    c2 = (1 - 5ν) / (1 - 2ν)
    @inbounds for i in 1:n
        # tensor contraction σ:ε = σ11ε11+σ22ε22+σ33ε33 + σ23γ23+σ13γ13+σ12γ12
        sdot = σ[i, 1] * ε[i, 1] + σ[i, 2] * ε[i, 2] + σ[i, 3] * ε[i, 3] +
               σ[i, 4] * ε[i, 4] + σ[i, 5] * ε[i, 5] + σ[i, 6] * ε[i, 6]
        trσ = σ[i, 1] + σ[i, 2] + σ[i, 3]
        trε = ε[i, 1] + ε[i, 2] + ε[i, 3]
        DT[i] = c * (10 * sdot - c2 * trσ * trε)
    end
    return DT
end

function topological_derivative(dad::BEMdata{<:Elasticity})
    ν = dad.properties.nu
    if dad.dimension == 2
        σb, εb = boundary_stress_strain(dad)
        σi, εi = interior_stress_strain(dad)
        return plane_stress_DT(σb, εb, ν), plane_stress_DT(σi, εi, ν)
    end
    σb, εb = recover_strain_stress!(dad)
    σi, εi = interior_stress_strain(dad)
    return isotropic_3d_DT(σb, εb, ν), isotropic_3d_DT(σi, εi, ν)
end

"""Compliance `J = ∫ t·u dΓ` on the Neumann part of the boundary."""
function elastic_compliance(dad::BEMdata{<:Elasticity})
    J = 0.0
    w = dad.elem_weight
    u = dad.u
    t = dad.traction
    dim = dad.dimension
    for el in dad.elements
        for a in eachindex(el.index)
            i = el.index[a]
            wJ = el.Jacobian[a] * w[a]
            s = 0.0
            for d in 1:dim
                s += t[dim * (i - 1) + d] * u[dim * (i - 1) + d]
            end
            J += s * wJ
        end
    end
    return J
end

design_objective(dad::BEMdata{<:Elasticity}) = elastic_compliance(dad)
