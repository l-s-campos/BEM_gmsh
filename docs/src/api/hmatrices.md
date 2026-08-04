# H-matrices

The vendored module `BEM.HMatrices` provides hierarchical matrix types and
algorithms (originally based on HMatrices.jl-style designs).

Important exports reexported by `BEM`:

- `HMatrix`, `assemble_hmatrix`, `PartialACA`, `StrongAdmissibilityStd`
- `ClusterTree`, `PrincipalComponentSplitter`, `GeometricSplitter`
- `compression_ratio`, `nodes`, `pivot`, …

```julia
H_G_Hmat(dad; atol=1e-6, nmax=32, eta=3.0)
solve(dad)   # GMRES through MixedBCOperator

plot_hmatrix(dad.H)          # Makie block-structure figure
plot_hmatrix(dad.H; show_rank=true)
```

`plot_hmatrix` uses **Makie/GLMakie** (same stack as `plot_geo`). RecipesBase/Plots are not used.
