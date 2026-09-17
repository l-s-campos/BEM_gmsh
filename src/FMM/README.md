# FMM (vendored into BEM)

Fast multipole **kernels and matvecs** only.

## Shared with HMatrices

| Concern | Owner |
|---------|--------|
| `ClusterTree`, splitters, `HyperRectangle`, admissibility | **HMatrices** |
| Multipole expansions, plans, physics kernels | **FMM** |
| `FMMKernelMatrix`, `KelvinFMMMatrix`, `KelvinFMMMatrix3D` | **FMM** (`operators/`) |

```julia
tree = ClusterTree(pts, hmatrix_splitter(; nmax=40); cube=true)  # FMM3D/H² octree
A = fmm_laplace2d_matrix(P; tree=tree)          # plan adopts tree
plan = build_laplace2d_plan(P; tree=tree)
rfmm2d(1e-8, P; charges=x, pg=1, plan=plan)    # reuse plan
```

Laplace apply is threaded (`Threads.@threads` over target boxes, including M2L).
Run Julia with `-t auto`. Default tree is the H² cubic octree / quadtree
(`DyadicSplitter(tight=false); cube=true`). 2D expansions match Flatiron FMM2D
(`l2d*`). 3D uses FMM3D-style dual-tree lists and spherical-harmonic translations
(`l3dterms` order, FMM3D `Y_n^m` packing). Same-size octree M2L is a
Sommerfeld plane-wave shift; mixed-size pairs stay equivalent-sphere. 2D leaf size follows
Flatiron `lndiv2d`. 3D leaf size follows Flatiron `lndiv` (`200` at `1e-8`);
pass a smaller `nmax=` for a deeper octree or `p=` to cut the expansion order.

## Layout

```
FMM/
  mod_FMM.jl
  core/                 expansions, dual-tree engine, Laplace-2D plan
  kernels/              Laplace3D, Helmholtz, Stokes, Yukawa, Cauchy, KIFMM, Body
  operators/            FMMKernelMatrix, Kelvin
```

HMatrices layout (sibling module):

```
Hmat/
  tree/       ClusterTree, splitters, boxes, admissibility
  compress/   ACA, AnchorNet
  formats/    H, BLR, NNCA, KernelMatrix
  arith/      mul, LU, precond
```

**Do not** hang FMM expansions or interaction lists on `ClusterTree` — use `Laplace2DFMMPlan` / `build_laplace3d_plan`.
