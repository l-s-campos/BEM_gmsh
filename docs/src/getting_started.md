# Getting started

```julia
using Pkg; Pkg.activate("."); Pkg.instantiate()
using BEM
```

Day-1 meshes (`quadrado`, `quadrado_elasticity`) come from `BEM.Examples` and
are reexported. You do **not** need DrWatson or `include(datadir(...))` for the
first example.

## Workflow

1. **Physics**
   ```julia
   props = Laplace(1.0)                    # conductivity k
   # props = Elasticity(E, ν, ρ)
   ```

2. **Mesh → `BEMdata`**
   ```julia
   dad = format2d(quadrado(ndiv=20, show=false), props; pontointerno=true)
   ```

3. **Optional analytical field**
   ```julia
   ana = ana_laplace_linear(; direction=SA[1.0, 0.0])  # T=x; q=-k ∂T/∂n
   attach_analytical!(dad, ana)
   ```

4. **Assemble**
   ```julia
   assemble!(dad, 20)                         # dense
   # assemble!(dad; method=:hmatrix, atol=1e-6)
   ```

5. **Domain term (transient / body load)**
   ```julia
   dibem!(dad)                 # mass-like M in dad.cache.M
   ```

6. **Solve**
   ```julia
   solve(dad)                           # steady
   # solve_Houbolt(dad, Δt, tf)
   # solve_transient_o2(dad, Δt, tf)
   ```

7. **Check & plot**
   ```julia
   rel_error(dad)
   plot_geo(dad)
   ```

## Boundary-condition encoding

Physical group **names** in Gmsh carry the BC:

| Problem | Name pattern | Meaning |
|---------|--------------|---------|
| Laplace | `"0;T"` | Dirichlet, value `T` |
| Laplace | `"1;q"` | Neumann, value `q = -k ∂T/∂n` |
| Elasticity 2D | `"tx;ux;ty;uy"` | per-component type/value |

`quadrado`: left `"0;0"`, right `"1;-1"`, top/bottom `"1;0"` → exact field ``T=x``.

## Dense vs H-matrix

| | `assemble!(dad)` | `assemble!(dad; method=:hmatrix)` | `assemble!(dad; method=:gpu)` |
|--|------------------|-----------------------------------|------------------------------|
| Cost | ``O(N^2)`` | typically ``O(N\log N)`` | ``O(N^2)`` far kernel on GPU |
| Best for | ``N \lesssim 5\cdot 10^3`` | large meshes | 2-D Laplace / Kelvin, NVIDIA GPU |
| Solver | LU / LinearSolve | GMRES on mixed BC operator | same host LU as dense |

## Tests

```bash
julia --project=. test/runtests.jl
```

One file per family under `test/` (smoke + one analytic). Longer suites live in
`scripts/debug/legacy_tests/`.
