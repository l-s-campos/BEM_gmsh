# Analytical solutions

```@docs
AnalyticalSolution
attach_analytical!
analytical
rel_error
ana_laplace_linear
ana_laplace_quadratic
ana_heat_1d
ana_heat_insulated_sides
ana_heat_dirichlet_square
ana_elasticity_patch
```

Typical verification pattern:

```julia
attach_analytical!(dad, ana_laplace_linear(; direction=SA[1.0, 0.0]))
solve(dad)
err = rel_error(dad)
```
