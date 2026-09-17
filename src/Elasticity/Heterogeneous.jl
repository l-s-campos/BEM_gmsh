# Smoothly inhomogeneous isotropic elasticity via DIBEM.
# E(x) varies, ν fixed. Kelvin H, G at the solid E0 stored on `dad.properties`.
#
# Scale k = E/E0. Equilibrium ∇·(k σ_ref) = 0  ⇒  L0(u) + b = 0 with
# b = (∇ ln k) · σ_ref. BIE at E0:
#   H u − G t_ref = M b ,   t_phys = k t_ref .
# `b` is lagged (Picard): stress from the current `u` via RBF, under-relaxed.
# Uniform E ⇒ b=0 recovers the homogeneous Kelvin system (G scaled by 1/k).

"""
    solve_heterogeneous!(dad::BEMdata{<:Elasticity}, E; rbf=PHS(1), npicard=8) -> u

Solve `∇·(C(E) : ε(u)) = 0` with nodal (or callable) Young modulus `E` and
the Poisson ratio on `dad.properties`. Assembles DIBEM `M` once when `∇E ≠ 0`.
Stores `dad.u`, `dad.traction` (physical), `dad.ufull` (boundary + interior).
"""
function solve_heterogeneous!(dad::BEMdata{<:Elasticity}, E;
        rbf=PHS(1; poly_deg=-1), npicard::Integer=4, relax::Float64=0.55,
        kmin::Float64=0.05)
    has_cache(dad, :H) || error("call assemble!(dad) first")
    Ev = eval_material_field(dad, E)
    E0 = float(dad.properties.E)
    E0 > 0 || error("dad.properties.E must be positive")
    k = Ev ./ E0
    @inbounds for i in eachindex(k)
        k[i] = clamp(k[i], kmin, Inf)
    end
    dim = dad.dimension
    n = dad.n
    nt = dad.nt
    ndof = dim * nt
    H = Matrix{Float64}(dad.H)
    G = Matrix{Float64}(dad.G)
    size(H, 1) == ndof || error("H is $(size(H)); expected $(ndof)×$(ndof)")

    pts = all_points(dad)
    gE = _grad_field(pts, log.(k), rbf)
    graded = sum(maximum(abs, gE[α]) for α in 1:dim) > 1e-12

    A0, rhs0 = _het_elast_mixed(dad, H, G, k)
    x = bem_linsolve(A0, rhs0)
    ufull = zeros(ndof)
    _het_elast_store!(dad, ufull, k, Ev, x)

    graded || return dad.u

    has_cache(dad, :M) || dibem_elasticity!(dad; rbf=rbf)
    M = dad.M
    ρm = float(dad.properties.rho)
    Mop = (abs(ρm) > 1e-14 && abs(ρm - 1) > 1e-8) ? (M ./ ρm) : M

    Jprev = _elast_compliance(dad)
    x_prev = copy(x)
    @inbounds for _ in 1:npicard
        σ = _elastic_stress_ref(dad, rbf)
        bforce = _het_elast_bodyforce(σ, gE, dim)
        rhs = rhs0 .+ Mop * bforce
        x_try = bem_linsolve(A0, rhs)
        x = (1 - relax) .* x_prev .+ relax .* x_try
        _het_elast_store!(dad, ufull, k, Ev, x)
        J = _elast_compliance(dad)
        if !isfinite(J) || J <= 0
            _het_elast_store!(dad, ufull, k, Ev, x_prev)
            break
        end
        if abs(J - Jprev) < 1e-4 * max(abs(Jprev), 1e-16)
            break
        end
        x_prev = x
        Jprev = J
    end
    return dad.u
end

"""Mixed BC system `A x = rhs` for `H u − G t_ref = 0` with `t_ref = t_phys / k`."""
function _het_elast_mixed(dad::BEMdata{<:Elasticity}, H, G, k)
    dim = dad.dimension
    ndof = size(H, 1)
    ndof_b = dim * dad.n
    A = copy(H)
    rhs = zeros(ndof)
    BC = dad.BC
    BV = dad.BV
    @inbounds for dof in 1:ndof_b
        node = (dof - 1) ÷ dim + 1
        kn = k[node]
        if BC[dof] == 0
            A[:, dof] .= .-view(G, :, dof)
            rhs .-= view(H, :, dof) .* BV[dof]
        else
            rhs .+= view(G, :, dof) .* (BV[dof] / kn)
        end
    end
    return A, rhs
end

function _het_elast_store!(dad::BEMdata{<:Elasticity}, ufull, k, Ev, x)
    dim = dad.dimension
    n = dad.n
    nt = dad.nt
    ndof_b = dim * n
    u = zeros(ndof_b)
    traction = zeros(ndof_b)
    BC = dad.BC
    BV = dad.BV
    @inbounds for dof in 1:ndof_b
        node = (dof - 1) ÷ dim + 1
        if BC[dof] == 0
            u[dof] = BV[dof]
            traction[dof] = k[node] * x[dof]
        else
            u[dof] = x[dof]
            traction[dof] = BV[dof]
        end
        ufull[dof] = u[dof]
    end
    uint = nt > n ? x[ndof_b+1:end] : Float64[]
    if nt > n
        ufull[ndof_b+1:end] .= uint
    end
    set_cache!(dad; u=u, traction=traction, T=u, uint=uint, ufull=ufull,
        het_E=Ev, het_k=k)
    return dad
end

function _elast_compliance(dad::BEMdata{<:Elasticity})
    J = 0.0
    w = dad.elem_weight
    u = dad.u
    t = dad.traction
    dim = dad.dimension
    @inbounds for el in dad.elements
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

function _het_elast_ufull(dad::BEMdata{<:Elasticity})
    dim = dad.dimension
    nt = dad.nt
    if has_cache(dad, :ufull) && length(dad.ufull) == dim * nt
        return dad.ufull
    end
    v = zeros(dim * nt)
    u = dad.u
    ncopy = min(length(u), dim * dad.n)
    v[1:ncopy] .= view(u, 1:ncopy)
    if has_cache(dad, :uint) && dad.nt > dad.n
        v[dim * dad.n + 1:end] .= dad.uint
    end
    return v
end

"""Voigt `σ_ref` (E0 Lamé) from RBF derivatives of `dad.ufull`. 2-D: `n×3`, 3-D: `n×6`."""
function _elastic_stress_ref(dad::BEMdata{<:Elasticity}, rbf)
    dim = dad.dimension
    nt = dad.nt
    ufull = _het_elast_ufull(dad)
    pts = all_points(dad)
    Dops, _ = rbf_diff_ops(pts, rbf)
    λ = dad.properties.lambda
    μ = dad.properties.mu
    if dim == 2
        ux, uy = ufull[1:2:end], ufull[2:2:end]
        Dx, Dy = Dops[1], Dops[2]
        εxx = Dx * ux
        εyy = Dy * uy
        γxy = Dx * uy + Dy * ux
        σ = zeros(nt, 3)
        @inbounds for i in 1:nt
            σ[i, 1] = (λ + 2μ) * εxx[i] + λ * εyy[i]
            σ[i, 2] = λ * εxx[i] + (λ + 2μ) * εyy[i]
            σ[i, 3] = μ * γxy[i]
        end
        return σ
    end
    ux, uy, uz = ufull[1:3:end], ufull[2:3:end], ufull[3:3:end]
    Dx, Dy, Dz = Dops[1], Dops[2], Dops[3]
    εxx = Dx * ux
    εyy = Dy * uy
    εzz = Dz * uz
    γyz = Dy * uz + Dz * uy
    γxz = Dx * uz + Dz * ux
    γxy = Dx * uy + Dy * ux
    σ = zeros(nt, 6)
    @inbounds for i in 1:nt
        tr = εxx[i] + εyy[i] + εzz[i]
        σ[i, 1] = λ * tr + 2μ * εxx[i]
        σ[i, 2] = λ * tr + 2μ * εyy[i]
        σ[i, 3] = λ * tr + 2μ * εzz[i]
        σ[i, 4] = μ * γyz[i]
        σ[i, 5] = μ * γxz[i]
        σ[i, 6] = μ * γxy[i]
    end
    return σ
end

"""Nodal strain-energy density `σ_ref : ε` at the solid `E0`."""
function _elastic_sed_ref(dad::BEMdata{<:Elasticity}, rbf)
    σ = _elastic_stress_ref(dad, rbf)
    dim = dad.dimension
    nt = dad.nt
    ufull = _het_elast_ufull(dad)
    pts = all_points(dad)
    Dops, _ = rbf_diff_ops(pts, rbf)
    sed = zeros(nt)
    if dim == 2
        ux, uy = ufull[1:2:end], ufull[2:2:end]
        Dx, Dy = Dops[1], Dops[2]
        εxx = Dx * ux
        εyy = Dy * uy
        γxy = Dx * uy + Dy * ux
        @inbounds for i in 1:nt
            sed[i] = σ[i, 1] * εxx[i] + σ[i, 2] * εyy[i] + σ[i, 3] * γxy[i]
        end
        return sed
    end
    ux, uy, uz = ufull[1:3:end], ufull[2:3:end], ufull[3:3:end]
    Dx, Dy, Dz = Dops[1], Dops[2], Dops[3]
    εxx = Dx * ux
    εyy = Dy * uy
    εzz = Dz * uz
    γyz = Dy * uz + Dz * uy
    γxz = Dx * uz + Dz * ux
    γxy = Dx * uy + Dy * ux
    @inbounds for i in 1:nt
        sed[i] = σ[i, 1] * εxx[i] + σ[i, 2] * εyy[i] + σ[i, 3] * εzz[i] +
                 σ[i, 4] * γyz[i] + σ[i, 5] * γxz[i] + σ[i, 6] * γxy[i]
    end
    return sed
end

"""Body force `b = (∇ ln k) · σ_ref` (physical components, length `dim*nt`)."""
function _het_elast_bodyforce(σ::AbstractMatrix, gE, dim::Integer)
    nt = size(σ, 1)
    b = zeros(dim * nt)
    if dim == 2
        @inbounds for i in 1:nt
            gx, gy = gE[1][i], gE[2][i]
            b[2i - 1] = gx * σ[i, 1] + gy * σ[i, 3]
            b[2i]     = gx * σ[i, 3] + gy * σ[i, 2]
        end
        return b
    end
    @inbounds for i in 1:nt
        gx, gy, gz = gE[1][i], gE[2][i], gE[3][i]
        b[3i - 2] = gx * σ[i, 1] + gy * σ[i, 6] + gz * σ[i, 5]
        b[3i - 1] = gx * σ[i, 6] + gy * σ[i, 2] + gz * σ[i, 4]
        b[3i]     = gx * σ[i, 5] + gy * σ[i, 4] + gz * σ[i, 3]
    end
    return b
end
