# Analytical solutions

Attach a field with [`attach_analytical!`](@ref) then [`rel_error`](@ref)
(primal) / [`rel_error_flux`](@ref) (flux or traction; use this when all
BCs are Dirichlet).

```@docs
AnalyticalSolution
attach_analytical!
apply_analytical_bc!
rel_error
rel_error_flux
ana_laplace_linear
ana_elasticity_patch
```
