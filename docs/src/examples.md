# Examples

Longer scripts live under `scripts/` (see [`scripts/README.md`](https://github.com/l-s-campos/BEM_gmsh/blob/main/scripts/README.md)).
The snippets below use the current public names.

Lubrication pad (Guiggiani 2020, Laplace FS + DIBEM):
`scripts/laplace/guiggiani_lubrication.jl`.

## 1. Steady Laplace

```julia
using BEM
dad = format2d(quadrado(ndiv=30, show=false), Laplace(1.0))
attach_analytical!(dad, ana_laplace_linear(; direction=SA[1.0, 0.0]))
assemble!(dad, 20)
solve(dad)
@show rel_error(dad)
plot_geo(dad)
```

Exact field for default BCs: ``T=x``, ``q=-k∂T/∂n``.

## 2. H-matrix

```julia
dad = format2d(quadrado(ndiv=40, show=false), Laplace(1.0); pontointerno=false)
attach_analytical!(dad, ana_laplace_linear(; direction=SA[1.0, 0.0]))
assemble!(dad; method=:hmatrix, atol=1e-6, nmax=32)
solve(dad)
@show rel_error(dad)
```

`dad.H` is a `ColWeightedOp` wrapping an H-matrix.

## 3. Transient

```julia
dad = format2d(quadrado(ndiv=16, show=false), Laplace(1.0); pontointerno=true)
assemble!(dad, 16)
dibem!(dad)
sol = solve_transient_o2(dad, 0.01, 0.5)
```

## 4. Elasticity patch

```julia
dad = format2d(quadrado_elasticity(ndiv=15, show=false), Elasticity(1.0, 0.3, 1.0))
apply_analytical_bc!(dad, ana_elasticity_patch(; E=1.0, ν=0.3, εxx=0.01))
assemble!(dad, 16)
solve(dad)
@show rel_error(dad)
```

## 5. Fundamentals

```julia
using BEM
r, n = Point2D(0.3, 0.4), Point2D(1, 0)
kp = fundamental(Laplace(1.0), r, n)
el = Elasticity(1.0, 0.3, 1.0)
fundamental(el, r, n)
fundamental_stress(el, r, n)
```

Walkthrough: `scripts/intro/fundamentals_demo.jl`.

## Script index

| Family | Folder |
|--------|--------|
| Intro | `scripts/intro/` |
| Laplace / Helmholtz | `scripts/laplace/` |
| Elasticity | `scripts/elasticity/` |
| DIBEM | `scripts/dibem/` |
| Transient | `scripts/transient/` |
| SBM / DRM | `scripts/sbm_drm/` |
| Crack | `scripts/crack/` |
| Contact | `scripts/contact/`, `scripts/julia_lerma/` |
| Plates | `scripts/plates/` |
| Topology | `scripts/topology/` |
| Papers / plots | `scripts/papers/` |
