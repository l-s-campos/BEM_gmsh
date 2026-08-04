# DIBEM & diffuse–advective

## Classic DIBEM (domain / mass operator)

```julia
DIBEM(dad; method=:dense)      # default — Domain.jl dense M
DIBEM(dad; method=:hmatrix)    # H-matrix F + M (ACA)
DIBEM(dad; method=:fmm)        # FMM Laplace factor + matrix-free M
# aliases: dibem!, DIBEM_dense, DIBEM_Hmat, DIBEM_FMM
```

Builds `dad.cache.M` ≈ domain integral operator
`∫ β u* dΩ ≈ M β` (inertia / body force / diffuse–advective).

```julia
H_G_full_direct(dad)
DIBEM(dad; method=:fmm)   # or :hmatrix for large nt
solve_Houbolt(dad, Δt, tf)
```

| method | `F` (RBF) | `D` / `M` (FS) | `M` type |
|--------|-----------|----------------|----------|
| `:dense` | dense | dense | `Matrix` |
| `:hmatrix` | H-matrix + GMRES | H-matrix | `HMatrix` |
| `:fmm` | dense or H-matrix | FMM matvec | `DibemFMMOperator` |

## Diffuse–advective (variable velocity)

Steady advection–diffusion

```math
α ∇² u = v · ∇ u
```

discretized with **DIBEM** on the advective density \(b = v·∇u\)
(Pinheiro thesis Ch.8 formulation).

```julia
solve_diffuse_advective!(dad, velocity; α=1.0)
# Internally:
#   M  = DIBEM(dad)              # Domain.jl  — ∫ β u* dΩ ≈ M β
#   M′ = build_da_Mprime(...)    # b = v·∇u ≈ M′ u
#   M_DA = M * M′ / α
#   H ← H − M_DA

# explicit steps:
DIBEM(dad)
dibem_diffuse_advective!(dad, velocity; α=1.0)  # reuses dad.M
solve(dad)

M  = build_da_S_matrix(dad)   # == dad.M from Domain.jl
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
