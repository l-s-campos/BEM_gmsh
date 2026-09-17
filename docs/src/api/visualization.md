# Visualization

Geometry, BC glyphs, and H-matrix block plots use **Plots.jl** (GR).
Field contours go through Gmsh / VTK.

```@docs
plot_geo
export_results_to_gmsh
export_vtk
```

`plot_hmatrix` lives in `BEM.HMatrices` (`using BEM.HMatrices: plot_hmatrix`).
