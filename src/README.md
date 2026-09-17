# Source layout

`using BEM` is the teaching spine (types, `format2d`, `assemble!`, `solve`,
`dibem!`). Advanced folders are **submodules** (`BEM.Crack`, `BEM.Contact`,
`BEM.Plate`, `BEM.Topology`, `BEM.MultiRegion`, `BEM.HMatrices`, `BEM.FMM`).
There is no `module Laplace`: that name is the problem type.

```
src/
  BEM.jl                 # module entry / includes
  Core/                  # shared infrastructure
    Interpolation.jl     # barycentric Lagrange / Legendre / equispaced
    Structures.jl        # BEMdata, Problem types, BEMCache
    LinearSolveUtils.jl  # bem_linsolve
    Kernels.jl           # KernelPair, StressKernels
    Integration.jl       # Newton closest-point + sinh + Guiggiani
    Input.jl             # format2d / format3d + Gmsh session
    SBM_geom.jl          # SBM parent-element map / L_m
    Radial_Basis_Functions.jl
    RBF_Extensions.jl    # included from Radial_Basis_Functions.jl
    Visualization.jl     # plot_geo, export_vtk
    GeometricProperties.jl
    Assembly_factored.jl # ColWeightedOp, mixed-BC blocks, BlockHLU
    DIBEM_common.jl      # RIM, CPD, factored M, cell mass
    SurfaceDIBEM.jl      # 3-D face integrals: PHS3+poly on parent edges + radial ID
    Assembly_full.jl     # dense H,G / assemble!
    Boundary_conditions.jl
    Solver.jl            # steady solve
    Analytical.jl        # AnalyticalSolution + catalog
  Laplace/
    Fundamental.jl
    Orthotropic.jl       # anisotropic/orthotropic conductivity (2D+3D)
    Assembly_H.jl
    Assembly_GPU.jl      # 2-D Laplace far-field KernelAbstractions kernel
    DIBEM_GPU.jl         # 2-D Laplace dense DIBEM on GPU (F, D, far RIM)
    Assembly_galerkin.jl
    Domain.jl            # DIBEM dense
    Domain_fast.jl
    Heterogeneous.jl
    AnisotropicDIBEM.jl
    Solver.jl            # Houbolt / heat / wave
    LocalBEM.jl
    SBM.jl
    SBM_DRM.jl
    ParticularSolution.jl
    Lubrication.jl       # Reynolds / Guiggiani films, Laplace FS + DIBEM
    ElrodAdams.jl        # mass-conserving p–θ cavitation (structured FVM)
  Elasticity/
    Assembly_GPU.jl      # 2-D Kelvin H,G + DIBEM F,D (KernelAbstractions)
    Fundamental.jl
    Anisotropic3D.jl
    Domain.jl
    Domain_fast.jl
    LocalBEM.jl
    Thermoelasticity.jl
    Axisymmetric.jl
    StrainStress.jl
    PlasticKernels.jl
    Plasticity.jl        # constant-cell von Mises
    Transient.jl
    SBM.jl
  Helmholtz/
    Fundamental.jl
  Topology/  Crack/  Plate/  MultiRegion/
  Contact/               # Pohrt–Li, layered, Uzawa, rolling, wheel–rail, Cattaneo, mortar
  Hmat/                  # hierarchical matrices (vendored)
  FMM/
```
