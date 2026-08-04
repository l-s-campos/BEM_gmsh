# =============================================================================
# Diffuse–advective BEM via DIBEM (variable velocity)
# Pinheiro thesis Ch.8 — direct-interpolation treatment of advection–diffusion
#
# Governing (α = diffusivity, v = velocity):
#   α ∇²u = v · ∇u = b          (steady diffuse–advective / advection–diffusion)
#
# Domain integral of density β uses the same operator M from Domain.jl:
#   ∫ β u* dΩ ≈ M β     ←  DIBEM(dad) / dibem_matrix(dad)
#
# Gradients of u via DRM-style RBF:  ∇u ≈ (∇F) F⁻¹ u,  b = M′ u.
# Discrete system (thesis 8.22):
#   H u − G q = M_DA u ,   M_DA = M * M′ / α
#   (H − M_DA) u = G q
# =============================================================================

export dibem_diffuse_advective!, assemble_diffuse_advective!
export build_da_S_matrix, build_da_Mprime
export solve_diffuse_advective!
export exp_mxy_solution, exp_mxy_velocity, exp_mxy_flux
export setup_da_square_exp_mxy, test_da_square_exp_mxy

"""
    build_da_S_matrix(dad; rbf=PHS(), rebuild=false) -> M

Return the regularized DIBEM domain operator **from [`Domain.jl`](@ref DIBEM)**:

```
∫_Ω β u* dΩ ≈ M β
```

This is exactly `dad.cache.M` produced by [`DIBEM`](@ref) / [`dibem_matrix`](@ref).
No separate assembly path — diffuse–advective reuses the same matrix.
"""
function build_da_S_matrix(dad::BEMdata{<:Laplace}; rbf=PHS(), rebuild::Bool=false)
    return Matrix{Float64}(dibem_matrix(dad; rbf=rbf, rebuild=rebuild))
end

"""
    build_da_Mprime(dad, velocity; rbf=PHS()) -> M′

DRM-style gradient recovery (thesis 8.18–8.21):

```
u = F β,  β = F⁻¹ u
u_,ℓ = F_,ℓ β = F_,ℓ F⁻¹ u
b = v₁ u_,1 + v₂ u_,2 = M′ u
```

If `DIBEM` already stored `dibem_F` with the same basis, that factorization is reused.
"""
function build_da_Mprime(dad::BEMdata{<:Laplace}, velocity; rbf=PHS())
    nt = dad.nt
    dim = dad.dimension
    pts = _da_points(dad)

    # Prefer F from DIBEM when available and size matches
    F = if has_cache(dad, :dibem_F) && size(dad.dibem_F) == (nt, nt)
        Matrix{Float64}(dad.dibem_F)
    else
        Fnew = zeros(nt, nt)
        @inbounds for j in 1:nt, i in 1:nt
            Fnew[i, j] = rbf(sqeuclidean(pts[i], pts[j]))
        end
        Fnew
    end
    ε = 1e-12 * (tr(F) / max(nt, 1) + 1)
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
    dibem_diffuse_advective!(dad, velocity; rbf=PHS(), α=1.0, modify_H=true) -> M_DA

Assemble the diffuse–advective transport matrix

```
M_DA = M * M′ / α
```

where **`M` comes from [`DIBEM`](@ref)** (`Domain.jl`) and `M′` maps `u → v·∇u`.

Optionally

```
H ← H − M_DA
```

so `H u = G q` solves `α ∇²u = v·∇u`.
"""
function dibem_diffuse_advective!(dad::BEMdata{<:Laplace}, velocity;
        rbf=PHS(), α::Real=1.0, modify_H::Bool=true, rebuild_M::Bool=false)
    has_cache(dad, :H) || error("call H_G_full_direct(dad) first")
    has_cache(dad, :G) || error("call H_G_full_direct(dad) first")
    α = float(α)
    α > 0 || throw(ArgumentError("α must be > 0"))

    # M from Domain.jl (DIBEM)
    M = build_da_S_matrix(dad; rbf=rbf, rebuild=rebuild_M)
    M′ = build_da_Mprime(dad, velocity; rbf=rbf)
    M_DA = (M * M′) ./ α
    set_cache!(dad; M_DA=M_DA, M_prime=M′, S_da=M)

    if modify_H
        H0 = has_cache(dad, :H0_da) ? dad.H0_da : copy(dad.H)
        set_cache!(dad; H0_da=H0)
        dad.H .= H0 .- M_DA
        if has_cache(dad, :A)
            dad.cache.A = nothing
        end
    end
    return M_DA
end

"""
    solve_diffuse_advective!(dad, velocity; kwargs...) -> T

Build diffuse–advective operators (`DIBEM` + `M′`) and call [`solve`](@ref).
"""
function solve_diffuse_advective!(dad::BEMdata{<:Laplace}, velocity; kwargs...)
    dibem_diffuse_advective!(dad, velocity; kwargs...)
    return solve(dad)
end

const assemble_diffuse_advective! = dibem_diffuse_advective!

# ---------------------------------------------------------------------------
# Manufactured example: unit square, u=exp(m x y), v=(m y, m x)
# (Pinheiro thesis Ch.8 §8.2.1)
# ---------------------------------------------------------------------------

"""Manufactured potential `u = exp(m x y)` (diffuse–advective test)."""
exp_mxy_solution(m::Real) = (p) -> exp(float(m) * p[1] * p[2])

"""Matching velocity `v = (m y, m x)` so that `∇²u = v·∇u` when `u = exp(m x y)`."""
exp_mxy_velocity(m::Real) = (p) -> SVector(float(m) * p[2], float(m) * p[1])

"""Boundary flux `q = -∂u/∂n` for `u = exp(m x y)` (package sign convention)."""
function exp_mxy_flux(m::Real)
    return (p, nrm) -> begin
        u = exp(float(m) * p[1] * p[2])
        dudx = float(m) * p[2] * u
        dudy = float(m) * p[1] * u
        return -(dudx * nrm[1] + dudy * nrm[2])
    end
end

"""
    setup_da_square_exp_mxy(msh; m=1.0, n_int=5)

Unit-square diffuse–advective problem with manufactured field `u=exp(m x y)`,
Dirichlet data from `u`, and a regular grid of internal DIBEM poles.
"""
function setup_da_square_exp_mxy(msh; m=1.0, n_int=5, ordem=1)
    dad = format2d(msh, Laplace(1.0); tipo=ordem, pontointerno=false)
    return _da_square_exp_mxy_bc_and_poles!(dad, m, n_int)
end

function setup_da_square_exp_mxy(; m=1.0, ndiv=10, n_int=5, ordem=1, nome="da_exp_mxy",
        mesh_fn=nothing)
    mesh_fn === nothing && error("setup_da_square_exp_mxy(; mesh_fn=quadrado) or pass a mesh")
    msh = mesh_fn(; ndiv=ndiv, ordem=ordem, show=false, nome=nome)
    return setup_da_square_exp_mxy(msh; m=m, n_int=n_int, ordem=ordem)
end

function _da_square_exp_mxy_bc_and_poles!(dad, m, n_int)
    uana = exp_mxy_solution(m)
    for i in 1:dad.n
        dad.BC[i] = 0
        dad.BV[i] = uana(dad.Nodes[i])
    end
    if n_int > 0
        xs = range(0.5 / (n_int + 1), 1 - 0.5 / (n_int + 1); length=n_int)
        internals = [SVector(float(x), float(y)) for y in xs for x in xs]
        set_internal_nodes!(dad, internals)
    end
    return dad
end

"""
    test_da_square_exp_mxy(dad; m=1.0, npg=12) -> NamedTuple

Solve the unit-square manufactured problem `u=exp(m x y)`, `v=(m y, m x)` and
report mean relative flux error on the **right** and **bottom** edges
(same metric as Pinheiro Ch.8 §8.2.1).
"""
function test_da_square_exp_mxy(dad::BEMdata{<:Laplace}; m=1.0, npg=12,
        rbf=PHS(), verbose=true)
    H_G_full_direct(dad, npg)
    solve_diffuse_advective!(dad, exp_mxy_velocity(m); rbf=rbf, α=1.0)

    uana = exp_mxy_solution(m)
    qana = exp_mxy_flux(m)
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
    verbose && @info "da square exp(mxy)" m n=dad.n nPI=dad.ni flux_err_pct err_u n_flux=n_s
    return (; flux_err_pct, err_u, dad, n_flux=n_s, qmax)
end

const _da_points = all_points
