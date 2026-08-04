# Exemplos

> 🌐 [English](../examples.md) · **Português (BR)**

Scripts prontos também em `data/examples/` e `scripts/`.

## 1. Laplace estacionário no quadrado unitário

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

Solução exata com as CDCs padrão: ``T(x,y)=x`` e fluxo ``q=-k∂T/∂n``.

## 2. Malha grande com H-matrizes

```julia
msh = quadrado(ndiv=80, show=false, nome="quad_fine")
dad = format2d(msh, Laplace(1.0); pontointerno=false)
attach_analytical!(dad, ana_laplace_linear(; direction=SA[1.0, 0.0]))

H_G_Hmat(dad; atol=1e-6, nmax=32)
@show compression_ratio(dad.H)
solve(dad)
@show rel_error(dad)
```

## 3. Calor transiente (1ª ordem)

```julia
dad = format2d(msh, Laplace(1.0))
H_G_full_direct(dad, 16)
DIBEM(dad)
sol = solve_transient(dad, 0.01, 1.0)
# dad.T[:, k] no instante dad.t[k]
```

## 4. Integrador de 2ª ordem (onda)

```julia
# Após H_G_full_direct + DIBEM:
sol = solve_transient_o2(dad, 0.01, 0.5)   # DifferentialEquations, sistema completo
# ou Houbolt clássico (H + 2M/Δt²):
# solve_Houbolt(dad, 0.05, 2.0)
```

## 5. Elasticidade — patch de Dirichlet

```julia
include(datadir("Laplace", "Laplace_dad.jl"))
msh = quadrado_elasticity(ndiv=15, show=false)
dad = format2d(msh, Elasticity(1.0, 0.3, 1.0); pontointerno=false)

ana = ana_elasticity_patch(; E=1.0, ν=0.3, εxx=0.01)
apply_analytical_bc!(dad, ana)   # todo Dirichlet de u = ε·x

H_G_full_direct(dad, 16)
solve(dad)
@show rel_error(dad)
plot_geo(dad)
```

## 6. Catálogo de soluções fundamentais

```julia
using DrWatson
@quickactivate :BEM

r, n = Point2D(0.3, 0.4), Point2D(1, 0)

kp = fundamental(Laplace(1.0), r, n)
kp = fundamental(Helmholtz(; ω=2π, c=1.0), r, n)

el = Elasticity(1.0, 0.3, 1.0)
kp = fundamental(el, r, n)
sk = fundamental_stress(el, r, n)

p = lekhnitskii_params(25.0, 1.0, 0.5, 0.25; θ_deg=30)
kp = fundamental(AnisotropicElasticity(p), r, zero(r), n)
```

Demo completa: `scripts/fundamentals_demo.jl`.

## 7. Propagação de ondas (barra / membrana)

```julia
include(datadir("Laplace", "Laplace_dad.jl"))
include(datadir("Laplace", "potencial_problems.jl"))
include(datadir("Laplace", "wave_propagation.jl"))

dad, meta = wave_problem(:bar_sudden; ndiv=12, n_int=6)
H_G_full_direct(dad; npg=10, threaded=false)
DIBEM(dad; rbf=PHS(3; poly_deg=0))
solve_Houbolt(dad, 0.05, 2.0)

# Outros casos: :annulus, :membrane_v0, :bar_periodic,
#               :membrane_forced, :ricker
# Lista: wave_problem_names()
```

Script: `scripts/wave_propagation.jl` (`WAVE_CASE`, `WAVE_SOLVER=houbolt|diffeq|both`).

## 8. Contato / desgaste (semi-espaço)

```bash
julia --project=. scripts/compare_contact_acceleration.jl
julia --project=. scripts/hertz_line_2d.jl
```

## 9. Benchmarks Laplace clássicos

```bash
julia --project=. scripts/potencial_direto.jl
julia --project=. data/examples/potencial_direto.jl
```
