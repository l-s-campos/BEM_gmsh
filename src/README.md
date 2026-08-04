# Source layout

```
src/
  BEM.jl                 # module entry / includes
  Core/                  # shared infrastructure
    Structures.jl
    Kernels.jl           # KernelPair, helpers
    Interpolation.jl
    Integration.jl       # Newton/NonlinearSolve projection, sinh-transform
    GmshSession.jl       # refcounted gmsh.initialize/finalize
    Bezier.jl            # Bernstein basis, Bézier extraction, IGA shapes
    Input.jl             # Gmsh I/O (format2d/3d; discretization=:iga)
    Visualization.jl
    Radial_Basis_Functions.jl
    Parallel.jl
    GeometricProperties.jl  # 2D/3D area-volume-centroid (propgeo)
  Laplace/
    Fundamental.jl
    Orthotropic.jl       # orthotropic conductivity
    Assembly_full.jl
    Assembly_H.jl
    Boundary_conditions.jl
    Solver.jl
    Domain.jl            # DIBEM
    Analytical.jl
  Elasticity/
    Fundamental.jl
    Thermoelasticity.jl
    Axisymmetric.jl      # axisym FS (elliptic integrals)
  Helmholtz/
    Fundamental.jl
  MultiRegion/
    SubRegions.jl        # BC types 3 & 4
  Contact/
    ContactHalfSpace.jl  # Pohrt–Li 3D surface
    ContactHalfPlane2D.jl
    HalfSpaceBEM.jl      # dense/FFT/Hmat/FMM + wear
  Crack/
    Crack.jl             # dual BEM + MTS/SED/Paris propagation (unified)
    DualCore.jl          # dual assembly / Gmsh BC type 5 (included by Crack.jl)
  Plate/
    ThinPlate.jl         # isotropic Kirchhoff plate BEM (Shi–Bezine)
    LargePlate.jl        # von Kármán large deflection + NonlinearSolve
    Buckling.jl          # plate / thermal buckling (eigenvalue)
    Shell.jl             # shallow shell (plate + membrane + curvature)
  Core/GeometricProperties.jl  # 2D/3D area-volume-centroid (propgeo)
  Laplace/Orthotropic.jl       # orthotropic conductivity Laplace
  Elasticity/Axisymmetric.jl   # axisym FS (elliptic integrals)
  Hmat/                  # hierarchical matrices (vendored)
```
