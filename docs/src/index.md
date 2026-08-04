# BEM.jl

> 🌐 **English** · [Português (BR)](pt-br/index.md)

**BEM.jl** is a Julia package for the **Boundary Element Method**, with:

- 2D/3D **Laplace** (potential / heat conduction) and **linear elasticity**
- Mesh generation and I/O through **Gmsh**
- Dense and **hierarchical (H-matrix)** assembly
- Domain integrals via **DIBEM** (radial basis functions)
- Steady and **transient** solvers (Houbolt, Method of Lines, 2nd-order ODE)
- Built-in **analytical solutions** for verification
- Visualization of geometry and boundary conditions (`plot_geo`)

The project is organised with [DrWatson.jl](https://juliadynamics.github.io/DrWatson.jl/stable/) for reproducible paths (`datadir`, `srcdir`, …).

## Install / activate

```julia
using Pkg
Pkg.activate("path/to/BEM_gmsh")
Pkg.instantiate()
```

```julia
using DrWatson
@quickactivate :BEM
```

## Minimal example

```julia
using DrWatson
@quickactivate :BEM
include(datadir("Laplace", "Laplace_dad.jl"))

props = Laplace(1.0)
msh = quadrado(ndiv=20, show=false)          # writes via datadir(...)
dad = format2d(msh, props)

attach_analytical!(dad, ana_laplace_linear(; direction=SA[1.0, 0.0]))

H_G_full_direct(dad, 20)                     # or H_G_Hmat(dad) for large meshes
solve(dad)

println("relative error = ", rel_error(dad))
plot_geo(dad)
```

## Package layout

```
src/
  BEM.jl                 # module entry
  Structures.jl          # BEMdata, Laplace, Helmholtz, Elasticity, Anisotropic…
  Fundamental_Solutions.jl  # Kelvin, Lekhnitskii, Helmholtz, hypersingular (Tensorial)
  Input.jl               # format2d / format3d (Gmsh)
  Assembly_full.jl       # dense H, G
  Assembly_H.jl          # H-matrix H, G  (calc_HeG_Hd style)
  Boundary_conditions.jl
  Solver.jl              # steady + transient
  Domain.jl              # DIBEM
  Analytical.jl          # reference solutions
  Visualization.jl       # plot_geo, Gmsh export
  Hmat/                  # hierarchical matrix library
data/Laplace/            # meshes + generators (datadir)
scripts/intro.jl         # demo
test/runtests.jl
docs/
```

## Next pages

- [Getting started](getting_started.md)
- [Theory notes](theory.md)
- [Examples](examples.md)
- API reference under **API**
