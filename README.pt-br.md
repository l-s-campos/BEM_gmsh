# BEM.jl

Método dos Elementos de Contorno em Julia — Laplace e elasticidade, malhas Gmsh,
montagem densa e H-matriz, integrais de domínio DIBEM, solvers transientes.

Projeto reproduzível com [DrWatson](https://juliadynamics.github.io/DrWatson.jl/stable/).

## Início rápido

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()

using BEM

props = Laplace(1.0)
msh = quadrado(ndiv=20, show=false)
dad = format2d(msh, props)

attach_analytical!(dad, ana_laplace_linear(; direction=SA[1.0, 0.0]))  # T=x, q=-k∂T/∂n
assemble!(dad, 20)    # ou assemble!(dad; method=:hmatrix) para problemas grandes
solve(dad)

@show rel_error(dad)
plot_geo(dad)
```

## Funcionalidades

| Área | Status |
|------|--------|
| Laplace 2D estacionário | ✅ densa + H-matriz |
| Calor transiente (Houbolt / DiffEq) | ✅ |
| Tempo 2ª ordem (`solve_transient_o2` / onda) | ✅ sistema completo |
| Integrais de domínio DIBEM | ✅ |
| Elasticidade 2D montar + resolver | ✅ |
| Soluções fundamentais | ✅ Tensorial.jl |
| Soluções analíticas + `rel_error` | ✅ |
| Contato half-space / desgaste | ✅ |
| Propagação de ondas escalar | ✅ `wave_propagation.jl` |
| `plot_geo` / `export_vtk` (Plots.jl) | ✅ |
| I/O Gmsh via `datadir` | ✅ |

## Testes

```bash
julia --project=. test/runtests.jl
julia --project=. test/runtests.jl
```

## Documentação

Documenter.jl com **inglês + pt-BR** (árvore paralela `docs/src/pt-br/`).

```bash
julia --project=. docs/make.jl
# abrir docs/build/index.html — menu "Português (BR)"
```

- EN: [docs/src/index.md](docs/src/index.md)
- pt-BR: [docs/src/pt-br/index.md](docs/src/pt-br/index.md)

> Documenter não tem i18n nativo (ao contrário do Jekyll + Polyglot). A solução
> é registrar duas árvores de páginas no mesmo `makedocs` — ver `docs/make.jl`.

## Organização

Ver [documentação pt-BR](docs/src/pt-br/index.md).
