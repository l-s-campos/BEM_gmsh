# Mesh I/O

_See source docstrings in `src/` (HTML `@docs` disabled in lightweight build)._

For multi-step Gmsh pipelines use
`format2d(...; finalize=false, reopen=false)` inside [`with_gmsh`](@ref)
so the API session stays open.

## Isogeometric / Bézier (`discretization=:iga`)

```julia
dad = format2d("model.geo", Laplace(1.0);
    discretization = :iga,      # or :bezier / :nurbs
    iga_mode = :cad,            # Gmsh CAD curves → B-spline → Bézier extraction
    # iga_mode = :bezier_mesh,  # high-order mesh → Lagrange-to-Bézier
    iga_degree = 2,
    iga_nel = 8,
    tipo = 2,
)
@assert dad.element_type isa Bernstein
@assert all(e -> e.extraction !== nothing, dad.elements)
```

Elements keep the same interface as Lagrange (`index`, `Jacobian`, `Length`,
`Region`) plus `extraction`, `nurbs_weights`, and optional `controls`.
Shape evaluation goes through `element_shapefun` (Bernstein × extraction,
rationalized when weights are set). See `src/Core/Bezier.jl`.

Geometry scripts (use after `@quickactivate :BEM`):

```julia
include(datadir("Laplace", "Laplace_dad.jl"))
quadrado(...)
placa_com_furo(...)
quadrado_elasticity(...)
```

All writers call `datadir(...)` and create parent directories as needed.
