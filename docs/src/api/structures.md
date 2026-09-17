# Data structures

Core types live in `src/Core/Structures.jl`.

- `BEMdata` holds mesh, BCs, physics, and a mutable [`BEMCache`](@ref)
  for matrices (`H`, `G`, `A`, `M`, …) and solutions (`T`, `q`, `u`, …).
- Cache fields are accessible as `dad.H`, `dad.T`, … via `getproperty`.
- Prefer [`set_cache!`](@ref) over rebuilding tuples.

```@docs
BEM
BEMdata
Laplace
OrthotropicLaplace
AnisotropicLaplace
Helmholtz
Elasticity
BEMCache
has_cache
set_cache!
point
set_internal_nodes!
```
