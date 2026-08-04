# Data structures

```@docs
Point2D
Point3D
BEMdata
BEMCache
set_cache!
has_cache
Element
Laplace
Helmholtz
Elasticity
AnisotropicElasticity
LekhnitskiiParams
```

Core types live in `src/Structures.jl`.

- `BEMdata` holds mesh, BCs, physics, and a mutable [`BEMCache`](@ref)
  for matrices (`H`, `G`, `A`, `M`, …) and solutions (`T`, `q`, `u`, …).
- Cache fields are accessible as `dad.H`, `dad.T`, … via `getproperty`.
- Prefer `set_cache!(dad; H=H, T=T)` over rebuilding tuples.
