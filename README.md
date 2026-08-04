# BEM_gmsh

**Boundary Element Method in Julia**, driven by [Gmsh](https://gmsh.info/) meshes.

Laplace & elasticity · dense / H-matrix / FMM · DIBEM & MECID domain terms ·  
transient (Houbolt, DiffEq) · modal MMM · dual BEM + cohesive contact · plates.

> Package module name: **`BEM`**. Repository name: **`BEM_gmsh`**.

## Install

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()   # needs Gmsh.jl system library
```

Julia ≥ 1.10.

## 5-minute path

```julia
using BEM
include(joinpath(@__DIR__, "data", "Laplace", "Laplace_dad.jl"))

dad = format2d(quadrado(ndiv=20, show=false), Laplace(1.0))
attach_analytical!(dad, ana_laplace_linear(; direction=SA[1.0, 0.0]))  # T = x
H_G_full_direct(dad, 20)
solve(dad)

@show rel_error(dad)
# plot_geo(dad)   # needs Makie backend
```

**Pipeline:** mesh → `format2d` → `H_G_full_direct` → `solve` → `dad.T` / `dad.q`.

## Feature map

| Want… | Start here |
|-------|------------|
| Steady Laplace | `solve`, `H_G_full_direct` / `H_G_Hmat` |
| Domain source / Poisson | `DIBEM`, `solve_poisson_rbf_bem!` |
| Variable velocity (MECID alt) | `solve_mecid_alt!`, `scripts/mecid_c8e1_demo.jl` |
| Heat / wave in time | `solve_Houbolt`, `solve_transient`, `solve_transient_o2` |
| Modal transient (MMM) | `solve_mmm!`, `scripts/mmm_membrane_demo.jl` |
| Elasticity | `Elasticity`, same assembly path |
| Cracks / cohesive | `BEM.Crack`, `scripts/cohesive_gmsh_modeI.jl` |
| IGA (Bézier) | `format2d(...; discretization=:iga)` |
| Contact half-space | `BEM.ContactHalfSpace` |

## Layout

```text
src/
  BEM.jl           # module entry (includes below)
  Core/            # mesh I/O, elements, RBF, integration
  Laplace/         # assembly, BC, solvers, DIBEM, MECID, MMM
  Elasticity/ Helmholtz/ Crack/ Contact/ Plate/ MultiRegion/
  Hmat/ FMM/       # hierarchical & multipole accelerators
data/              # Gmsh .geo/.msh builders + analytics (include as needed)
scripts/           # runnable demos
test/              # test_*.jl  (+ runtests.jl)
docs/              # Documenter (EN + pt-BR) + architecture.md
```

Design notes and cleanup plan: [`docs/architecture.md`](docs/architecture.md).

## Tests & demos

```bash
julia --project=. test/runtests.jl
julia --project=. test/test_mecid_alt.jl
julia --project=. test/test_mmm.jl
julia --project=. scripts/mecid_c8e1_demo.jl
julia --project=. scripts/mmm_membrane_demo.jl
julia --project=. scripts/wave_propagation.jl
```

## Documentation

```bash
julia --project=. docs/make.jl
# docs/build/index.html
# docs/build/pt-br/   (Português)
```

## Citation / theory hooks

Implementations track standard BEM texts and local theses, including:

- Direct interpolation (MECID / DIBEM) — Loeffler, Mansur, Pinheiro (Ch.8 alternative)
- Modal modified method (MMM) — Prodonoff & Zepka; Santos thesis Ch.4
- Dual BEM + cohesive laws — package `Crack` module

## License

Research code — add an explicit license before public redistribution if required by your institution.
