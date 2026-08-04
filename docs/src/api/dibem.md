# DIBEM & diffuse–advective

## Classic DIBEM (domain / mass operator)

```julia
DIBEM(dad; method=:dense)      # dense M (Domain.jl)
DIBEM(dad; method=:hmatrix)    # H-matrix D = u*
DIBEM(dad; method=:hodlr)      # HODLR D
DIBEM(dad; method=:hss)        # HSS D
DIBEM(dad; method=:hbs)        # HBS (≡ HSS)
DIBEM(dad; method=:h2)         # H² D
DIBEM(dad; method=:fmm)        # FMM D
```

### Unified factored form (all compressed methods)

```text
F c = IF                  # RBF  (DIBEM-specific weights)
D_ij = u*(x_i, x_j)       # same single-layer kernel family as G
M x = D (c ∘ x) + diag ∘ x ,   diag = ID − D c
```

So **G and D share `u*`**. DIBEM only adds the weight vector `c` (and Galerkin `ID`).

| | BEM `G` | DIBEM factor `D` |
|--|---------|------------------|
| kernel | `u*(x_i, x_j)` | same |
| columns | boundary nodes | all collocation poles |
| column scale | quadrature `w_j` | DIBEM `c_j` |
| size | `nt × n` | `nt × nt` |

| method | compresses `D ∼ u*` | `M` type |
|--------|---------------------|----------|
| `:dense` | dense product form | `Matrix` |
| `:hmatrix` … `:fmm` | H / HODLR / HSS / H² / FMM | `DibemFactoredOperator` |

```julia
H_G_full_direct(dad)          # G ~ u* w_j  (boundary)
DIBEM(dad; method=:hodlr)     # D ~ u*, then M from c
solve_Houbolt(dad, Δt, tf)
```

## Diffuse–advective (variable velocity)

Steady advection–diffusion ``α ∇² u = v · ∇ u``, with

```julia
M = DIBEM(dad)                 # any method above
M′ = build_da_Mprime(dad, v) # b = v·∇u
M_DA = M * M′ / α
H ← H − M_DA
```

```julia
solve_diffuse_advective!(dad, velocity; α=1.0)
```

### Manufactured square test ``u = e^{mxy}``

```julia
dad = setup_da_square_exp_mxy(msh; m=1.0, n_int=7)
test_da_square_exp_mxy(dad; m=1.0)
```
