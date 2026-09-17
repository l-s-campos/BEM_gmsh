# Thermoelasticity BEM (port of termoelasticidade.jl)
#
# Thermal strain is converted to an equivalent boundary traction and (for
# non-uniform θ) a domain body-force term treated with DIBEM / RBF gradients:
#
#   t^{th} = k̂ θ n ,   b = -k̂ ∇θ ,   k̂ = E α /(1-2ν)
#   H u = G t + G t^{th} - M (k̂ ∇θ) + M b_mech
#   σ = D t - S u + … - k̂ θ I
#
export thermal_modulus, eval_temperature
export rbf_gradient_ops, thermal_boundary_load
export solve_thermoelastic!, stress_thermoelastic
export analytical_constrained_thermal_stress
# dibem_elasticity! lives in Elasticity/Domain.jl

# =============================================================================
# Temperature field
# =============================================================================

"""
    eval_temperature(dad, θ) -> Vector{Float64}

Evaluate temperature at all collocation points (boundary + internal).
`θ` may be a `Number`, a vector of length `dad.nt`, or a function `(x,y)->θ`.
"""
function eval_temperature(dad::BEMdata, θ)
    nt = dad.nt
    if θ isa Number
        return fill(float(θ), nt)
    elseif θ isa AbstractVector
        length(θ) == nt || error("θ vector length $(length(θ)) ≠ dad.nt=$nt")
        return float.(θ)
    elseif θ isa Function
        out = Vector{Float64}(undef, nt)
        @inbounds for i in 1:dad.n
            p = dad.Nodes[i]
            out[i] = float(θ(p[1], p[2]))
        end
        @inbounds for i in 1:length(dad.internalNodes)
            p = dad.internalNodes[i]
            out[dad.n+i] = float(θ(p[1], p[2]))
        end
        return out
    else
        error("θ must be Number, Vector, or Function; got $(typeof(θ))")
    end
end

const _all_points = all_points  # Structures.jl

# =============================================================================
# RBF gradient operators (montaFs-style, PHS + optional poly)
# =============================================================================

"""
    rbf_gradient_ops(pts; rbf=PHS(3; poly_deg=1)) -> (; F, Fx, Fy)

Build differentiation matrices so `Fx * f ≈ ∂f/∂x`, `Fy * f ≈ ∂f/∂y` at `pts`.
"""
function rbf_gradient_ops(pts::AbstractVector{<:Point}; rbf=PHS(3; poly_deg=1))
    n = length(pts)
    dim = length(pts[1])
    npoly = rbf_npoly(dim, rbf.poly_deg)
    mon = npoly > 0 ? MonomialBasis(dim, rbf.poly_deg) : nothing

    F = zeros(n, n)
    dFx = zeros(n, n)
    dFy = zeros(n, n)
    @inbounds for j in 1:n, i in 1:n
        r = norm(pts[i] - pts[j])
        F[i, j] = rbf(r)
        if r > 0
            dFx[i, j] = ∂(rbf, 1, pts[i], pts[j])
            dFy[i, j] = ∂(rbf, 2, pts[i], pts[j])
        end
    end

    if npoly == 0 || rbf.poly_deg < 0
        Finv = inv(F)
        return (F=F, Fx=dFx * Finv, Fy=dFy * Finv)
    end

    P = zeros(n, npoly)
    dPx = zeros(n, npoly)
    dPy = zeros(n, npoly)
    @inbounds for i in 1:n
        P[i, :] = mon(pts[i])
        dPx[i, :] = ∂(mon, 1, pts[i])
        dPy[i, :] = ∂(mon, 2, pts[i])
    end
    # Hermite / augmented RBF: N = F_aug^{-1} style (see montaFs)
    # Solve F λ + P μ = f,  P' λ = 0
    # Gradient: dF λ + dP μ
    A = [F P; P' zeros(npoly, npoly)]
    # For each collocation value e_k, coefficients = A \ [e_k; 0]
    # Fx_ik = (dFx λ + dPx μ)_i for rhs e_k
    Fx = zeros(n, n)
    Fy = zeros(n, n)
    rhs = zeros(n + npoly)
    @inbounds for k in 1:n
        fill!(rhs, 0)
        rhs[k] = 1
        c = A \ rhs
        λ = @view c[1:n]
        μ = @view c[n+1:end]
        Fx[:, k] = dFx * λ + dPx * μ
        Fy[:, k] = dFy * λ + dPy * μ
    end
    return (F=F, Fx=Fx, Fy=Fy)
end

"""Stack ∇θ into a 2n vector: `[∂θ/∂x_1, ∂θ/∂y_1, …]`."""
function stack_grad(Fx, Fy, θ::AbstractVector)
    n = length(θ)
    dθx = Fx * θ
    dθy = Fy * θ
    g = zeros(2n)
    @inbounds for i in 1:n
        g[2i-1] = dθx[i]
        g[2i] = dθy[i]
    end
    return g
end

# =============================================================================
# Thermal boundary load  q1 = G (k̂ θ n)
# =============================================================================

"""
    thermal_boundary_load(dad, θ; k̂=thermal_modulus(dad.properties)) -> q1

Equivalent Neumann load from temperature on the boundary (length `2·dad.nt`).
Only boundary traction columns of `G` are used; internal block is zero.
"""
function thermal_boundary_load(dad::BEMdata{<:Union{Elasticity,AnisotropicElasticity3D}}, θ;
    k̂=thermal_modulus(dad.properties))
    dim = dad.dimension
    n = dad.n
    nt = dad.nt
    G = dad.G
    θv = eval_temperature(dad, θ)
    tth = zeros(dim * n)
    @inbounds for i in 1:n
        ni = dad.Normal[i]
        tth[dim*(i-1)+1] = k̂ * θv[i] * ni[1]
        tth[dim*(i-1)+2] = k̂ * θv[i] * ni[2]
    end
    # G is (dim·nt) × (dim·n)
    return G * tth
end

# =============================================================================
# Solve  (DIBEM M from Elasticity/Domain.jl via dibem_elasticity!)
# =============================================================================

"""
    solve_thermoelastic!(dad; θ=0, bodyforce=nothing, npg_dibem=10) -> u

Solve the thermoelastic BEM problem.

# Arguments
- `θ`: temperature (`Number`, `Vector`, or `(x,y)->θ`). Uses `dad.properties.α`.
- `bodyforce`: optional mechanical body force `(x,y)->SVector(bx,by)` or nodal `2nt` vector.

Assumes `H_G_full_direct` has already been called. Builds DIBEM `M` when
`θ` is non-uniform or `bodyforce` varies in space. A **constant** body
force uses the RIM remainder ``ID`` only (`M (1⊗b) = ID b`).
"""
function solve_thermoelastic!(dad::BEMdata{<:Union{Elasticity,AnisotropicElasticity3D}};
    θ=0.0, bodyforce=nothing, npg_dibem=10, rbf=PHS(3; poly_deg=1))

    has_cache(dad, :H) || error("call H_G_full_direct(dad) first")
    props = dad.properties
    k̂ = thermal_modulus(props)
    θv = eval_temperature(dad, θ)
    dim = dad.dimension
    nt = dad.nt
    n = dad.n

    applyBC(dad)
    b = copy(dad.b)

    # boundary thermal load
    if abs(k̂) > 0 && any(!iszero, θv)
        b .+= thermal_boundary_load(dad, θv; k̂=k̂)
    end

    θ_nonuniform = maximum(θv) - minimum(θv) > 1e-14 * (1 + maximum(abs, θv))
    bn = bodyforce === nothing ? nothing : _eval_bodyforce(dad, bodyforce)
    const_b = bn !== nothing && _is_constant_nodal(bn, dim)
    can_id = const_b && (has_cache(dad, :dibem_ID) || dad isa BEMdata{<:Elasticity})
    need_domain = bodyforce !== nothing || (abs(k̂) > 0 && θ_nonuniform)
    need_M = (abs(k̂) > 0 && θ_nonuniform) || (bodyforce !== nothing && !can_id)

    if need_domain
        if need_M
            has_cache(dad, :M) || dibem_elasticity!(dad; npg=npg_dibem, rbf=rbf)
        elseif can_id && !has_cache(dad, :dibem_ID)
            _, ID, _ = _dibem_elast_IF_ID(dad, rbf; npg=npg_dibem)
            set_cache!(dad; dibem_ID=ID, dibem_rbf=rbf)
        end
        q_dom = zeros(dim * nt)
        if abs(k̂) > 0 && θ_nonuniform
            pts = all_points(dad)
            ops = rbf_gradient_ops(pts; rbf=rbf)
            gθ = stack_grad(ops.Fx, ops.Fy, θv)
            q_dom .-= dad.M * (k̂ .* gθ)     # -q2  (termoelasticidade.jl)
        end
        if bn !== nothing
            if can_id && has_cache(dad, :dibem_ID)
                q_dom .+= dad.dibem_ID * bn[1:dim]
            else
                q_dom .+= dad.M * bn
            end
        end
        b .+= q_dom
    end

    x = dad.A \ b
    u = zeros(eltype(x), dim * n)
    traction = zeros(eltype(x), dim * n)
    # unknowns may include internal u (rows dim*n+1:end of x are uint)
    split_sol!(dad, x[1:dim*n], u, traction)
    uint = nt > n ? x[dim*n+1:end] : eltype(x)[]
    set_cache!(dad; u=u, traction=traction, T=u, uint=uint, θ=θv, b_thermo=b)
    return u
end

function _is_constant_nodal(bn::AbstractVector, dim::Integer)
    n = length(bn)
    dim < 1 && return false
    n % dim == 0 || return false
    nt = n ÷ dim
    nt < 2 && return true
    @inbounds for i in 2:nt
        for d in 1:dim
            bn[dim * (i - 1) + d] == bn[d] || return false
        end
    end
    return true
end

function _eval_bodyforce(dad, bodyforce)
    dim = dad.dimension
    nt = dad.nt
    bn = zeros(dim * nt)
    if bodyforce isa AbstractVector
        length(bodyforce) == dim * nt || error("bodyforce length mismatch")
        return float.(bodyforce)
    end
    pts = all_points(dad)
    @inbounds for i in 1:nt
        p = pts[i]
        f = _call_bodyforce(bodyforce, p)
        for d in 1:dim
            bn[dim*(i-1)+d] = f[d]
        end
    end
    return bn
end

function _call_bodyforce(f, p::SVector{2})
    return f(p[1], p[2])
end
function _call_bodyforce(f, p::SVector{3})
    if applicable(f, p)
        return f(p)
    elseif applicable(f, p[1], p[2], p[3])
        return f(p[1], p[2], p[3])
    else
        return f(p)
    end
end

# =============================================================================
# Stress recovery
# =============================================================================

"""
    stress_thermoelastic(dad, u=dad.u, t=dad.traction, θ=dad.θ; npg=10) -> σ

Nodal stresses ``[σ_{11}, σ_{22}, σ_{12}]`` at boundary + internal points.
Includes the thermal isotropic part ``-k̂ θ`` on the normal components.
"""
function stress_thermoelastic(dad::BEMdata{<:Elasticity},
    u=dad.u, t=dad.traction, θ=nothing; npg=10)

    dim = 2
    n = dad.n
    nt = dad.nt
    θv = θ === nothing ? (has_cache(dad, :θ) ? dad.θ : zeros(nt)) : eval_temperature(dad, θ)
    k̂ = thermal_modulus(dad.properties)
    pts = all_points(dad)
    qsi, w = gausslegendre(npg)

    S = zeros(3nt, dim * n)
    Dmat = zeros(3nt, dim * n)

    @showprogress "stress S,D" for i in 1:nt
        pf = pts[i]
        rows = 3i-2:3i
        for elem in dad.elements
            X = dad.Nodes[elem.index]
            for (ig, ξ) in enumerate(qsi)
                nn = length(elem.index)
                if nn == 2
                    N = SVector(0.5(1 - ξ), 0.5(1 + ξ))
                    dN = SVector(-0.5, 0.5)
                else
                    N = SVector(0.5ξ*(ξ - 1), 1 - ξ^2, 0.5ξ*(ξ + 1))
                    dN = SVector(ξ - 0.5, -2ξ, ξ + 0.5)
                end
                pg = sum(N[k] * X[k] for k in 1:nn)
                dx = sum(dN[k] * X[k] for k in 1:nn)
                J = norm(dx)
                J < 1e-16 && continue
                n̂ = Point2D(dx[2] / J, -dx[1] / J)
                rvec = pg - pf
                R = norm(rvec)
                R < 1e-12 && continue
                sk = fundamental_stress(dad, rvec, n̂)
                DD = sk.D
                SS = sk.S
                # σ_ij = D_kij t_k - S_kij u_k ; Voigt (11,22,12)
                wJ = J * w[ig]
                for a in 1:nn
                    ja = elem.index[a]
                    cols = dim*(ja-1)+1:dim*ja
                    Na = N[a] * wJ
                    Dblk = @SMatrix [
                        DD[1,1,1] DD[2,1,1]
                        DD[1,2,2] DD[2,2,2]
                        DD[1,1,2] DD[2,1,2]
                    ]
                    Sblk = @SMatrix [
                        SS[1,1,1] SS[2,1,1]
                        SS[1,2,2] SS[2,2,2]
                        SS[1,1,2] SS[2,1,2]
                    ]
                    Dmat[rows, cols] .+= Dblk .* Na
                    S[rows, cols] .+= Sblk .* Na
                end
            end
        end
    end

    # thermal boundary contribution to stress: dq1 = D * (k̂ θ n)
    tth = zeros(dim * n)
    @inbounds for i in 1:n
        ni = dad.Normal[i]
        tth[2i-1] = k̂ * θv[i] * ni[1]
        tth[2i]   = k̂ * θv[i] * ni[2]
    end
    σvec = Dmat * t .- S * u .+ Dmat * tth

    # free term factor: 2 on boundary, 1 interior (as in reference)
    σ = zeros(nt, 3)
    @inbounds for i in 1:nt
        fac = i <= n ? 2.0 : 1.0
        σ[i, 1] = fac * σvec[3i-2] - k̂ * θv[i]
        σ[i, 2] = fac * σvec[3i-1] - k̂ * θv[i]
        σ[i, 3] = fac * σvec[3i]
    end
    set_cache!(dad; stress=σ)
    return σ
end

"""
Analytical stress for a fully constrained body under uniform Δθ (u ≡ 0):
``σ_{11} = σ_{22} = -k̂ Δθ``, ``σ_{12} = 0``.
"""
analytical_constrained_thermal_stress(props::Elasticity, Δθ) =
    -thermal_modulus(props) * Δθ
