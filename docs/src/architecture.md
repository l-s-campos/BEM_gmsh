# Architecture

Current layout after the didactic reorg (phases 1–5).

## Pipeline

```text
mesh (Gmsh)  →  format2d / format3d  →  BEMdata
                      ↓
                assemble!   (dense | :hmatrix | :gpu)
                      ↓
         optional: dibem! / DRM / local BEM / SBM
                      ↓
                solve / solve_Houbolt / solve_mmm!
                      ↓
                dad.T, dad.q   (+ plot_geo, rel_error)
```

Everything else is an optional branch behind a **submodule**.

## Module tree

```text
module BEM                    # teaching spine
  Core/                       # types, mesh, integration, dense assembly (includes)
  Laplace/ Elasticity/ Helmholtz/
  module HMatrices            # BEM.HMatrices
  module FMM                  # BEM.FMM
  module Crack
  module Contact              # Pohrt–Li, layered, Uzawa, rolling, wheel–rail, Cattaneo, mortar
  module Plate                # Kirchhoff / FSDT (Reissner, Wang) / large / shell
  module Topology
  module MultiRegion          # interfaces + frictional contact
  module Examples             # quadrado, …
end
```

There is **no** `module Laplace`: that name is the problem type.

`using BEM` exports the spine (`Laplace`, `format2d`, `assemble!`, `solve`,
`dibem!`, `quadrado`, …). Load specialists explicitly:

```julia
using BEM.Crack
using BEM.Contact
using BEM.HMatrices
```

## Tests and scripts

- CI: `test/runtests.jl` — one file per family.
- Longer suites: `scripts/debug/legacy_tests/`.
- Demos: `scripts/<family>/` — see `scripts/README.md`.

## Cache keys

`BEMdata.cache` holds assembled operators and solutions. Typed fields:
`H`, `G`, `A`, `b`, `T`, `q`, `u`, `traction`, `M`. Unknown names go in
`cache.extras` (`:cells`, `:dibem_c`, `:bc_idx`, `:sbm`, `:twin`, …).
Prefer [`set_cache!`](@ref) / [`has_cache`](@ref). See `BEMCache`.
