# DIBEM & diffuse–advective

## Classic DIBEM (domain / mass operator)

```julia
DIBEM(dad; rbf=PHS())
# alias: dibem!(dad)
```

Builds `dad.cache.M` for inertia and body-force style domain terms (wave/heat).

Call after `H_G_full_direct` (and internal poles if needed):

```julia
H_G_full_direct(dad)
DIBEM(dad)
solve_Houbolt(dad, Δt, tf)
```

## Diffuse–advective (variable velocity)

Steady advection–diffusion

```math
α ∇² u = v · ∇ u
```

discretized with **DIBEM** on the advective density \(b = v·∇u\)
(Pinheiro thesis Ch.8 formulation).

```julia
solve_diffuse_advective!(dad, velocity; α=1.0, rbf=PHS(3; poly_deg=-1))
# or step-by-step:
dibem_diffuse_advective!(dad, velocity; α=1.0)  # H ← H − M_DA/α
solve(dad)

S  = build_da_S_matrix(dad)
Mp = build_da_Mprime(dad, velocity)
```

### Manufactured square test \(u = e^{mxy}\)

```julia
exp_mxy_solution(m)   # u = exp(m*x*y)
exp_mxy_velocity(m)   # v = (m*y, m*x)
exp_mxy_flux(m)       # q = -∂u/∂n

dad = setup_da_square_exp_mxy(msh; m=1.0, n_int=7)
res = test_da_square_exp_mxy(dad; m=1.0)
@show res.flux_err_pct
```

Demo: `scripts/diffuse_advective_exp_mxy.jl`.
