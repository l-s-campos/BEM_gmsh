# Recipes

Copy-paste workflows. Activate the project first:

```julia
using Pkg; Pkg.activate("."); Pkg.instantiate()
using BEM
include(joinpath(@__DIR__, "data", "Laplace", "Laplace_dad.jl"))  # or datadir(...)
```

---

## 1. Steady Laplace on the unit square

```julia
dad = format2d(quadrado(ndiv=20, show=false), Laplace(1.0))
attach_analytical!(dad, ana_laplace_linear(; direction=SA[1.0, 0.0]))  # T = x
H_G_full_direct(dad, 20)
solve(dad)
@show rel_error(dad)
```

---

## 2. DIBEM mass + wave (Houbolt)

```julia
dad = format2d(quadrado(ndiv=16, show=false), Laplace(1.0); pontointerno=true)
H_G_full_direct(dad, 12)
DIBEM(dad)                 # dad.cache.M
solve_Houbolt(dad, 0.01, 1.0)
```

---

## 3. Diffuse–advective (variable velocity)

Manufactured field \(u = e^{mxy}\), \(v = (my, mx)\):

```julia
msh = quadrado(ndiv=16, show=false)
dad = setup_da_square_exp_mxy(msh; m=1.0, n_int=7)
res = test_da_square_exp_mxy(dad; m=1.0)
@show res.flux_err_pct

# or manually:
# H_G_full_direct(dad)
# solve_diffuse_advective!(dad, exp_mxy_velocity(1.0); α=1.0)
```

Demo script: `scripts/diffuse_advective_exp_mxy.jl`.

---

## 4. Modal transient (MMM)

```julia
# after H_G_full_direct + DIBEM and free-DOF setup:
U, t, basis = solve_mmm!(dad, 0.01, 2.0; nmodes=12)
@show basis.ω[1:min(4, end)]
```

Demo: `scripts/mmm_membrane_demo.jl`.

---

## 5. Cohesive crack (Gmsh center crack)

```julia
using BEM.Crack
# see scripts/cohesive_gmsh_modeI.jl
```

---

## 6. Large problems (H-matrix)

```julia
H_G_Hmat(dad)   # instead of H_G_full_direct
solve(dad)
```
