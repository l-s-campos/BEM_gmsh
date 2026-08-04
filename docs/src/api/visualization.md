# Visualization

```@docs
plot_geo
export_results_to_gmsh
```

`plot_geo` draws:

- element edges and collocation / internal nodes
- **Dirichlet** dofs as inward triangles (color = prescribed value)
- **Neumann** dofs as arrows (length scaled to `arrow_scale`)
