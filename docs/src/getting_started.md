# Getting started

## Project activation

```julia
using DrWatson
@quickactivate :BEM   # loads the BEM module and project env
```

All data paths should go through DrWatson helpers:

```julia
datadir("Laplace", "quadrado.msh")
srcdir("Solver.jl")
```

Geometry generators in `data/Laplace/Laplace_dad.jl` already call `datadir`.

## Workflow

1. **Choose physics**
   ```julia
   props = Laplace(1.0)                    # k = conductivity
   # props = Elasticity(E, ν, ρ)
   ```

2. **Build or load a mesh**
   ```julia
   include(datadir("Laplace", "Laplace_dad.jl"))
   msh = quadrado(ndiv=20, show=false)
   dad = format2d(msh, props; pontointerno=true)
   ```

3. **(Optional) attach analytical solution**
   ```julia
   ana = ana_laplace_linear(; direction=SA[1.0, 0.0])  # T=x; q=-k ∂T/∂n
   attach_analytical!(dad, ana)
   # or impose pure Dirichlet from the field:
   # apply_analytical_bc!(dad, ana)
   ```

4. **Assemble**
   ```julia
   H_G_full_direct(dad, 20)   # dense
   # H_G_Hmat(dad; atol=1e-6) # hierarchical, large n
   ```

5. **Domain term (transient / body load)**
   ```julia
   DIBEM(dad)                 # builds mass-like matrix M
   ```

6. **Solve**
   ```julia
   solve(dad)                           # steady
   # solve_Houbolt(dad, Δt, tf)
   # solve_transient(dad, Δt, tf)       # 1st-order (heat)
   # solve_transient_o2(dad, Δt, tf)    # 2nd-order (wave-like)
   ```

7. **Check & plot**
   ```julia
   rel_error(dad)
   plot_geo(dad)
   # export_results_to_gmsh(dad, msh, :T; viewer=false)
   ```

## Boundary condition encoding

Physical group **names** in Gmsh carry the BC:

| Problem | Name pattern | Meaning |
|---------|--------------|---------|
| Laplace | `"0;T"` | Dirichlet, value `T` |
| Laplace | `"1;q"` | Neumann, value `q = -k ∂T/∂n` |
| Elasticity 2D | `"tx;ux;ty;uy"` | per-component type/value |

Example (`quadrado`): left `"0;0"`, right `"1;-1"`, top/bottom `"1;0"` → exact field ``T=x`` (because ``q=-k∂T/∂n``).

## Choosing dense vs H-matrix

| | Dense `H_G_full_direct` | Hierarchical `H_G_Hmat` |
|--|-------------------------|-------------------------|
| Cost | ``O(N^2)`` memory/time | ``O(N\\log N)`` typical |
| Integration | singular + near-field quad | collocation + diagonal fix |
| Best for | ``N \\lesssim 5\\cdot 10^3`` | large meshes |
| Solver | LU / `LinearSolve` | GMRES on [`MixedBCOperator`](@ref) |
