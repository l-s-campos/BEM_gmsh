# BEM_gmsh

Julia package module: **`BEM`**.

> 🌐 **English** · [Português (BR)](pt-br/index.md)

**BEM_gmsh** is a Julia toolkit for the **Boundary Element Method**, driven by
[Gmsh](https://gmsh.info/) meshes:

- 2D/3D **Laplace** and **linear elasticity**
- Dense, **H-matrix**, and **FMM** assembly
- Domain integrals via **DIBEM** / DRM / local BEM / SBM
- Steady and **transient** solvers (Houbolt, OrdinaryDiffEq, MMM)
- Dual BEM cracks, contact, plates, topology optimization
- Built-in **analytical** fields for verification (`rel_error`)

## Install

```julia
using Pkg
Pkg.activate("path/to/BEM_gmsh")
Pkg.instantiate()   # needs the Gmsh system library
```

Julia ≥ 1.10. Package name: **`BEM`**. Repository name: **`BEM_gmsh`**.

## 5-minute path

```julia
using BEM

dad = format2d(quadrado(ndiv=20, show=false), Laplace(1.0))
attach_analytical!(dad, ana_laplace_linear(; direction=SA[1.0, 0.0]))  # T = x
assemble!(dad, 20)            # or assemble!(dad; method=:hmatrix)
solve(dad)

@show rel_error(dad)
# plot_geo(dad)
```

**Pipeline:** mesh → `format2d` → `assemble!` → optional `dibem!` → `solve` → `dad.T` / `dad.q`.

## Feature map

| Want… | Start here |
|-------|------------|
| Steady Laplace | [`assemble!`](@ref), [`solve`](@ref) |
| Domain source / Poisson | [`DIBEM`](@ref), `solve_poisson_rbf_bem!`, `solve_local_bem!` |
| Heat / wave in time | `solve_Houbolt`, `solve_transient_o2`, `solve_mmm!` |
| Elasticity | `Elasticity`, `solve(dad; frame=:local)` |
| Cracks / cohesive | `using BEM.Crack` |
| Contact | `using BEM.Contact` |
| Topology | `using BEM.Topology` |
| H-matrices / FMM | `using BEM.HMatrices`, `using BEM.FMM` |

Advanced names are **submodules** — they are not dumped into `using BEM`.

## Next pages

- [Getting started](getting_started.md)
- [Recipes](recipes.md) — copy-paste, one per family
- [Theory notes](theory.md)
- [API](api/structures.md)
- [Architecture](architecture.md)
