# Analytical solutions

_See source docstrings in `src/` (HTML `@docs` disabled in lightweight build)._

Typical verification pattern:

```julia
attach_analytical!(dad, ana_laplace_linear(; direction=SA[1.0, 0.0]))
solve(dad)
err = rel_error(dad)
```
