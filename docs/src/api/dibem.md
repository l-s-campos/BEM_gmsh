# DIBEM & diffuse–advective

## Classic DIBEM (domain / mass operator)

```julia
DIBEM(dad; method=:dense)      # default — dense M
DIBEM(dad; method=:hmatrix)    # classical H-matrix (ACA)
DIBEM(dad; method=:hodlr)      # HODLR
DIBEM(dad; method=:hss)        # HSS
DIBEM(dad; method=:hbs)        # HBS (≡ HSS)
DIBEM(dad; method=:h2)         # H² (proxy bases)
DIBEM(dad; method=:fmm)        # FMM Laplace + matrix-free M
# also: DIBEM_Hmat, DIBEM_HODLR, DIBEM_HSS, DIBEM_H2, DIBEM_FMM
```

Builds `dad.cache.M` ≈ `∫ β u* dΩ ≈ M β`.

```julia
H_G_full_direct(dad)
DIBEM(dad; method=:hodlr)   # or :hss, :h2, :fmm, :hmatrix
solve_Houbolt(dad, Δt, tf)
```

| method | `F` (RBF) | Laplace / `M` | `M` type |
|--------|-----------|---------------|----------|
| `:dense` | dense | dense | `Matrix` |
| `:hmatrix` | H + GMRES | H | `HMatrix` |
| `:hodlr` | HODLR + GMRES | HODLR off-diag + diag | `DibemStructuredOperator` |
| `:hss` / `:hbs` | HSS + GMRES | HSS off-diag + diag | `DibemStructuredOperator` |
| `:h2` | dense `F` solve | H² on plain `u*`, then `M x = D(c∘x)+diag∘x` | `DibemFMMOperator{H2Matrix}` |

**H² note:** Discrete weights satisfy `c_j = c(y_j)`. Compressing the product
kernel `K(x,y)=c(y)u*(x,y)` with proxies fails; the correct H² path is the
**factored** form (same as FMM): H² only on Newtonian `u*`, scale columns via `c`.
| `:fmm` | dense/H | FMM matvec | `DibemFMMOperator` |

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
