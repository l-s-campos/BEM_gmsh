# BEM_gmsh

[![Docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://l-s-campos.github.io/BEM_gmsh/)
[![GitHub](https://img.shields.io/badge/GitHub-BEM__gmsh-black?logo=github)](https://github.com/l-s-campos/BEM_gmsh)

**Boundary Element Method in Julia**, driven by [Gmsh](https://gmsh.info/) meshes.

Laplace & elasticity · dense / H-matrix / FMM · DIBEM domain terms ·  
transient (Houbolt, DiffEq) · modal MMM · dual BEM + cohesive contact · plates.

> Package module name: **`BEM`**. Repository name: **`BEM_gmsh`**.

## Install

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()   # needs Gmsh.jl system library
```

Julia ≥ 1.10.

Dev-only tools (`Revise`, `Infiltrator`, `BenchmarkTools`) are **not** in the
default environment — `] add` them in your personal env if you want them.
Documenter lives in `docs/`. Meshless comparisons (Macchiato / RBF-FD) use
`scripts/meshless/` rather than the main project.

## 5-minute path

```julia
using BEM

dad = format2d(quadrado(ndiv=20, show=false), Laplace(1.0))
attach_analytical!(dad, ana_laplace_linear(; direction=SA[1.0, 0.0]))  # T = x
assemble!(dad, 20)            # dense; or assemble!(dad; method=:hmatrix)
solve(dad)

@show rel_error(dad)
# plot_geo(dad)   # Plots.jl
```

**Pipeline:** mesh → `format2d` → `assemble!` → `solve` → `dad.T` / `dad.q`.

## Feature map

| Want… | Start here |
|-------|------------|
| Steady Laplace | `solve`, `assemble!` / `assemble!(dad; method=:hmatrix)` |
| Domain source / Poisson | `DIBEM`, `solve_poisson_rbf_bem!`, `solve_local_bem!` |
| Diffuse–advective (variable v) | `solve_diffuse_advective!`, `scripts/laplace/diffuse_advective_exp_mxy.jl` |
| Heat / wave in time | `solve_Houbolt`, `solve_transient`, `solve_transient_o2` |
| Modal transient (MMM) | `solve_mmm!`, `scripts/transient/mmm_membrane_demo.jl` |
| Elasticity | `Elasticity`, `solve_local_bem!` (compact Kelvin) |
| Cracks / cohesive | `BEM.Crack`, `scripts/crack/cohesive_gmsh_modeI.jl` |
| SBM Laplace (Chen–Gu) | `solve_sbm_laplace`, `scripts/sbm_drm/sbm_vs_bem.jl` |
| SBM–DRM heat | `solve_sbm_drm`, `scripts/sbm_drm/sbm_drm_vs_dibem.jl` |
| Contact half-space | `BEM.Contact` (`using BEM.Contact`) — Pohrt–Li, layered, Uzawa wear, rolling, wheel–rail |
| Topology opt. (heat / elasticity, 2-D + 3-D density) | `BEM.Topology`, `scripts/topology/topology_compare.jl`, `scripts/topology/dibem_simp_3d.jl` |

## Layout

```text
src/
  BEM.jl           # teaching spine (types, assemble!, solve)
  Core/            # mesh I/O, elements, RBF, integration
  Laplace/ Elasticity/ Helmholtz/
  Crack/ Contact/ Plate/ MultiRegion/ Topology/   # BEM.<Name>
  Hmat/ FMM/       # BEM.HMatrices, BEM.FMM
data/              # Gmsh .geo builders + analytics
scripts/           # demos by family — see scripts/README.md
test/
docs/
```

Advanced APIs: `using BEM.Crack`, `BEM.Contact`, `BEM.Plate`, `BEM.Topology`,
`BEM.MultiRegion`, `BEM.HMatrices`.

Design notes and cleanup plan: [`docs/architecture.md`](docs/architecture.md).

## Tests & demos

```bash
julia --project=. test/runtests.jl
julia --project=. scripts/laplace/diffuse_advective_exp_mxy.jl
julia --project=. scripts/transient/mmm_membrane_demo.jl
```

## Documentation

**Online:** [https://l-s-campos.github.io/BEM_gmsh/](https://l-s-campos.github.io/BEM_gmsh/)

Build locally:

```bash
julia --project=docs docs/make.jl
# open docs/build/index.html
```

## Citation / theory hooks

Implementations track standard BEM texts and local theses, including:

- Direct interpolation (DIBEM) — Loeffler, Mansur; Pinheiro Ch.8 diffuse–advective
- Modal modified method (MMM) — Prodonoff & Zepka; Santos thesis Ch.4
- Dual BEM + cohesive laws — package `Crack` module

## License

Research code — add an explicit license before public redistribution if required by your institution.
