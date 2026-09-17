# Multi-region

`using BEM.MultiRegion`. Type-3 perfect interface; type-4 frictional contact
(active-set / SSN / projected Newton / GNM-ls 2010). Contact kinematics use
the stiffness-weighted common normal ``n_{AB}=(n_A E_A-n_B E_B)/‖·‖``.

Laplace type-3 coupling is `T_a = T_b` and `q_a + q_b = 0` (`q = -k ∂T/∂n`).
Each zone is assembled independently; [`solve_multiregion!`](@ref) chooses
the linear-algebra organisation:

| `strategy` | Global size | What is factored |
|------------|-------------|------------------|
| `:dense` | `∑ n_t + n_if` | one dense matrix (zeros between zones filled) |
| `:noncondensing` | `∑ n_t + n_if` | each full zone `A_r` + Schur of size `n_if` (Kane blocked / noncondensing) |
| `:condense` | `2 n_if` | each exterior block `A_ee` + stacked interface maps (Kane zone condensation) |

```julia
assemble_multiregion(prob)
solve_multiregion!(prob; strategy=:condense)
```

Compare the three on the two-slab analytic problem:
`scripts/laplace/two_regions_strategies.jl`.

```@docs
BEM.MultiRegion
BEM.MultiRegion.MultiRegionProblem
BEM.MultiRegion.pair_interfaces!
BEM.MultiRegion.assemble_multiregion
BEM.MultiRegion.solve_multiregion!
BEM.MultiRegion.multiregion_ndof
```
