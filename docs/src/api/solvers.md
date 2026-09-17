# Solvers

Always call [`dibem!`](@ref) / [`DIBEM`](@ref) before the transient drivers.

```@docs
solve
solve_Houbolt
solve_transient
solve_transient_o2
solve_mmm!
```

Elasticity local frame: `solve(dad; frame=:local)` or [`solve_local`](@ref).
