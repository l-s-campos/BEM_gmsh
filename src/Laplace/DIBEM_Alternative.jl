# =============================================================================
# DIBEM alternative — variable-velocity advection–diffusion
# (Pinheiro thesis Ch.8 — alternative direct-interpolation formulation)
#
# Classic domain mass/inertia: [`DIBEM`](@ref) in Domain.jl.
# This file: alternative *advection* operator (implicit ∇·v treatment).
#
# Governing (α = diffusivity):
#   α ∇²u = v · ∇u = b
#
# Regularize ∫ b u* dΩ, approximate [b−b(ξ)] u* by RBFs (direct interpolation),
# recover gradients of u as in DRM:  ∇u = (∇F) F⁻¹ u,  b = M′ u.
# Final discrete system (thesis 8.22):
#   H u − G q = M_ID u
#   (H − M_ID) u = G q
# =============================================================================

export dibem_alt_variable_velocity!
export build_dibem_S_matrix, build_dibem_Mprime
export solve_dibem_alt!
export dibem_c8e1_analytic, dibem_c8e1_velocity, dibem_c8e1_flux
export setup_dibem_c8e1, test_dibem_c8e1

"""
    build_dibem_S_matrix(dad; rbf=PHS(3; poly_deg=-1), k=dad.properties.k) -> S

Regularized DIBEM operator `S` such that

```
∫_Ω β(X) u*(ξ,X) dΩ  ≈  (S β)(ξ)
```

(thesis §8.1.1–8.1.2; same regularization idea as classic [`DIBEM`](@ref)).
"""
function build_dibem_S_matrix(dad::BEMdata{<:Laplace};
        rbf=PHS(3; poly_deg=-1), k=nothing)
    kk = k === nothing ? float(dad.properties.k) : float(k)
    nt = dad.nt
    pts = _dibem_points(dad)

    F = zeros(nt, nt)
    D = zeros(nt, nt)
    @inbounds for j in 1:nt, i in 1:nt
        r2 = sqeuclidean(pts[i], pts[j])
        F[i, j] = rbf(r2)
        if r2 > 0
            D[i, j] = -log(r2) / (4π * kk)   # u* (2D Laplace)
        end
    end
    ε = 1e-12 * (tr(F) / nt + 1)
    @inbounds for i in 1:nt
        F[i, i] += ε
    end

    # N[j] = ∫_Γ η^j dΓ,  η = n · ∇ψ,  ∇²ψ = F^j
    N = zeros(nt)
    ID = zeros(nt)
    @inbounds for j in 1:nt
        xj = pts[j]
        for elem in dad.elements
            for q in eachindex(elem.index)
                ind = elem.index[q]
                xq = ind <= dad.n ? dad.Nodes[ind] : dad.internalNodes[ind - dad.n]
                rvec = xq - xj
                R = norm(rvec)
                R < 1e-14 && continue
                wJ = dad.elem_weight[q] * elem.Jacobian[q]
                n_dot = dot(dad.Normal[ind], rvec) / R^2
                N[j] += int(rbf, xq, xj) * wJ * n_dot
            end
        end
    end
    @inbounds for i in 1:nt
        xi = pts[i]
        for elem in dad.elements
            for q in eachindex(elem.index)
                ind = elem.index[q]
                xq = ind <= dad.n ? dad.Nodes[ind] : dad.internalNodes[ind - dad.n]
                rvec = xq - xi
                R = norm(rvec)
                R < 1e-14 && continue
                wJ = dad.elem_weight[q] * elem.Jacobian[q]
                n_dot = dot(dad.Normal[ind], rvec) / R^2
                ID[i] += -(2 * R^2 * log(R) - R^2) / (8 * π * kk) * wJ * n_dot
            end
        end
    end

    c = F \ N
    S = D .* c'
    @inbounds for i in 1:nt
        srow = sum(view(S, i, :))
        S[i, i] = 0.0
        S[i, i] = -srow + ID[i]
    end
    return S
end

"""
    build_dibem_Mprime(dad, velocity; rbf=...) -> M′

DRM-style gradient recovery (thesis 8.18–8.21):

```
u = F β,  β = F⁻¹ u
u_,ℓ = F_,ℓ β = F_,ℓ F⁻¹ u
b = v₁ u_,1 + v₂ u_,2 = M′ u
```
"""
function build_dibem_Mprime(dad::BEMdata{<:Laplace}, velocity;
        rbf=PHS(3; poly_deg=-1))
    nt = dad.nt
    dim = dad.dimension
    pts = _dibem_points(dad)

    F = zeros(nt, nt)
    @inbounds for j in 1:nt, i in 1:nt
        F[i, j] = rbf(sqeuclidean(pts[i], pts[j]))
    end
    ε = 1e-12 * (tr(F) / nt + 1)
    @inbounds for i in 1:nt
        F[i, i] += ε
    end
    Finv = inv(F)

    Fx = zeros(nt, nt)
    Fy = zeros(nt, nt)
    Fz = dim == 3 ? zeros(nt, nt) : nothing
    @inbounds for j in 1:nt, i in 1:nt
        i == j && continue
        Fx[i, j] = ∂(rbf, 1, pts[i], pts[j])
        Fy[i, j] = ∂(rbf, 2, pts[i], pts[j])
        if dim == 3
            Fz[i, j] = ∂(rbf, 3, pts[i], pts[j])
        end
    end

    vx = zeros(nt)
    vy = zeros(nt)
    vz = dim == 3 ? zeros(nt) : nothing
    @inbounds for i in 1:nt
        v = velocity(pts[i])
        vx[i] = float(v[1])
        vy[i] = float(v[2])
        if dim == 3
            vz[i] = float(v[3])
        end
    end

    Gx = Fx * Finv
    Gy = Fy * Finv
    M′ = Diagonal(vx) * Gx + Diagonal(vy) * Gy
    if dim == 3
        M′ = M′ + Diagonal(vz) * (Fz * Finv)
    end
    return M′
end

"""
    dibem_alt_variable_velocity!(dad, velocity; rbf, α=1.0, modify_H=true) -> M_ID

Assemble the **alternative DIBEM** transport matrix `M_ID = S M′` (thesis 8.22)
for variable velocity `v(X)` and optionally

```
H ← H − M_ID / α
```

so `H u = G q` solves `α ∇²u = v·∇u`.

“Alternative” = advection treatment (implicit `∇·v`), not a different method name.
"""
function dibem_alt_variable_velocity!(dad::BEMdata{<:Laplace}, velocity;
        rbf=PHS(3; poly_deg=-1), α::Real=1.0, modify_H::Bool=true)
    has_cache(dad, :H) || error("call H_G_full_direct(dad) first")
    has_cache(dad, :G) || error("call H_G_full_direct(dad) first")
    α = float(α)
    α > 0 || throw(ArgumentError("α must be > 0"))

    S = build_dibem_S_matrix(dad; rbf=rbf, k=dad.properties.k)
    M′ = build_dibem_Mprime(dad, velocity; rbf=rbf)
    M_ID = S * M′
    M_trans = M_ID ./ α
    set_cache!(dad; M_ID=M_trans, M_prime=M′, S_dibem=S)

    if modify_H
        H0 = has_cache(dad, :H0_dibem_alt) ? dad.H0_dibem_alt : copy(dad.H)
        set_cache!(dad; H0_dibem_alt=H0)
        dad.H .= H0 .- M_trans
        if has_cache(dad, :A)
            dad.cache.A = nothing
        end
    end
    return M_trans
end

"""
    solve_dibem_alt!(dad, velocity; kwargs...) -> T

Build alternative DIBEM advection operators and call [`solve`](@ref).
"""
function solve_dibem_alt!(dad::BEMdata{<:Laplace}, velocity; kwargs...)
    dibem_alt_variable_velocity!(dad, velocity; kwargs...)
    return solve(dad)
end

# ---------------------------------------------------------------------------
# Example 8.2.1 helpers (Pinheiro thesis C8E1)
# ---------------------------------------------------------------------------

"""Analytic field for C8E1: `u = exp(m x y)`."""
dibem_c8e1_analytic(m::Real) = (p) -> exp(float(m) * p[1] * p[2])

"""Velocity for C8E1: `v = (m y, m x)` (divergence-free)."""
dibem_c8e1_velocity(m::Real) = (p) -> SVector(float(m) * p[2], float(m) * p[1])

"""Normal flux for C8E1 with package convention `q = -k ∂u/∂n` (k=1)."""
function dibem_c8e1_flux(m::Real)
    return (p, nrm) -> begin
        u = exp(float(m) * p[1] * p[2])
        dudx = float(m) * p[2] * u
        dudy = float(m) * p[1] * u
        return -(dudx * nrm[1] + dudy * nrm[2])
    end
end

"""C8E1 setup from an existing Gmsh mesh (all-Dirichlet `u=e^{mxy}` + internal poles)."""
function setup_dibem_c8e1(msh; m=1.0, n_int=5, ordem=1)
    dad = format2d(msh, Laplace(1.0); tipo=ordem, pontointerno=false)
    return _dibem_c8e1_bc_and_poles!(dad, m, n_int)
end

function setup_dibem_c8e1(; m=1.0, ndiv=10, n_int=5, ordem=1, nome="c8e1",
        mesh_fn=nothing)
    mesh_fn === nothing && error("setup_dibem_c8e1(; mesh_fn=quadrado) or pass a mesh")
    msh = mesh_fn(; ndiv=ndiv, ordem=ordem, show=false, nome=nome)
    return setup_dibem_c8e1(msh; m=m, n_int=n_int, ordem=ordem)
end

function _dibem_c8e1_bc_and_poles!(dad, m, n_int)
    uana = dibem_c8e1_analytic(m)
    for i in 1:dad.n
        dad.BC[i] = 0
        dad.BV[i] = uana(dad.Nodes[i])
    end
    if n_int > 0
        xs = range(0.5 / (n_int + 1), 1 - 0.5 / (n_int + 1); length=n_int)
        empty!(dad.internalNodes)
        for y in xs, x in xs
            push!(dad.internalNodes, SVector(float(x), float(y)))
        end
        dad.ni = length(dad.internalNodes)
        dad.nt = dad.n + dad.ni
    end
    return dad
end

"""
    test_dibem_c8e1(dad; m=1.0, npg=12) -> NamedTuple

Pinheiro §8.2.1: mean relative flux error on **right** and **bottom** edges.
"""
function test_dibem_c8e1(dad::BEMdata{<:Laplace}; m=1.0, npg=12,
        rbf=PHS(3; poly_deg=-1), verbose=true)
    H_G_full_direct(dad, npg)
    solve_dibem_alt!(dad, dibem_c8e1_velocity(m); rbf=rbf, α=1.0)

    uana = dibem_c8e1_analytic(m)
    qana = dibem_c8e1_flux(m)
    T = dad.T
    q = dad.q
    err_s = 0.0
    n_s = 0
    qmax = 0.0
    @inbounds for i in 1:dad.n
        p = dad.Nodes[i]
        nrm = dad.Normal[i]
        on_bottom = p[2] < 1e-9
        on_right = p[1] > 1 - 1e-9
        (on_bottom || on_right) || continue
        qa = qana(p, nrm)
        qmax = max(qmax, abs(qa))
        err_s += abs(q[i] - qa)
        n_s += 1
    end
    err_u = sum(abs(T[i] - uana(dad.Nodes[i])) for i in 1:dad.n) / dad.n
    flux_err_pct = (qmax > 0 && n_s > 0) ? 100 * (err_s / n_s) / qmax : NaN
    verbose && @info "C8E1 DIBEM-alt" m n=dad.n nPI=dad.ni flux_err_pct err_u n_flux=n_s
    return (; flux_err_pct, err_u, dad, n_flux=n_s, qmax)
end

function _dibem_points(dad::BEMdata)
    nt = dad.nt
    pts = Vector{typeof(dad.Nodes[1])}(undef, nt)
    @inbounds for i in 1:dad.n
        pts[i] = dad.Nodes[i]
    end
    @inbounds for i in 1:dad.ni
        pts[dad.n + i] = dad.internalNodes[i]
    end
    return pts
end
