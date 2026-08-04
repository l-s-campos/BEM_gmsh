# Examples

## 1. Steady Laplace on the unit square

```julia
using DrWatson
@quickactivate :BEM
include(datadir("Laplace", "Laplace_dad.jl"))

props = Laplace(1.0)
msh = quadrado(ndiv=30, show=false)
dad = format2d(msh, props)

attach_analytical!(dad, ana_laplace_linear(; direction=SA[1.0, 0.0]))
H_G_full_direct(dad, 20)
solve(dad)

@show rel_error(dad)
fig = plot_geo(dad)
```

Exact solution for the default BCs: ``T(x,y)=x`` with flux ``q=-k∂T/∂n``.

## 2. Large mesh with H-matrices

```julia
msh = quadrado(ndiv=80, show=false, nome="quad_fine")
dad = format2d(msh, Laplace(1.0); pontointerno=false)
attach_analytical!(dad, ana_laplace_linear(; direction=SA[1.0, 0.0]))

H_G_Hmat(dad; atol=1e-6, nmax=32)
@show compression_ratio(dad.H)
solve(dad)
@show rel_error(dad)
```

## 3. Transient heat (1st order)

```julia
dad = format2d(msh, Laplace(1.0))
H_G_full_direct(dad, 16)
DIBEM(dad)
sol = solve_transient(dad, 0.01, 1.0)
# dad.T[:, k] at time dad.t[k]
```

## 4. Second-order time integrator

```julia
sol = solve_transient_o2(dad, 0.01, 0.5)
# uses SecondOrderODEProblem; positions in dad.T
```

## 5. Elasticity Dirichlet patch

```julia
include(datadir("Laplace", "Laplace_dad.jl"))
msh = quadrado_elasticity(ndiv=15, show=false)
dad = format2d(msh, Elasticity(1.0, 0.3, 1.0); pontointerno=false)

ana = ana_elasticity_patch(; E=1.0, ν=0.3, εxx=0.01)
apply_analytical_bc!(dad, ana)   # all Dirichlet from u = ε·x

H_G_full_direct(dad, 16)
solve(dad)
@show rel_error(dad)
plot_geo(dad)
```

## 6. Fundamental solutions catalogue

```julia
using DrWatson
@quickactivate :BEM

r, n = Point2D(0.3, 0.4), Point2D(1, 0)

# Laplace
kp = fundamental(Laplace(1.0), r, n)
@show kp.U, kp.T

# Helmholtz
kp = fundamental(Helmholtz(; ω=2π, c=1.0), r, n)

# Kelvin elasticity (Tensorial Mat{2,2})
el = Elasticity(1.0, 0.3, 1.0)
kp = fundamental(el, r, n)
@show kp.U
sk = fundamental_stress(el, r, n)   # D, S tensors

# Lekhnitskii anisotropy
p = lekhnitskii_params(25.0, 1.0, 0.5, 0.25; θ_deg=30)
kp = fundamental(AnisotropicElasticity(p), r, zero(r), n)
```

Full walkthrough: `scripts/fundamentals_demo.jl`.

## 7. Compare numerical vs analytical along a line

```julia
attach_analytical!(dad, ana_laplace_linear(; direction=SA[1.0, 0.0]))
Tana = analytical(dad)
# boundary nodes only:
using GLMakie
scatter(dad.T[1:dad.n], label="BEM")
scatter!(Tana[1:dad.n], label="exact")
axislegend()
current_figure()
```
