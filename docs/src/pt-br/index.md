# BEM.jl

> 🌐 [English](../index.md) · **Português (BR)**

**BEM.jl** é um pacote Julia para o **Método dos Elementos de Contorno**, com:

- **Laplace** 2D/3D (potencial / condução de calor) e **elasticidade linear**
- Geração e I/O de malhas via **Gmsh**
- Montagem densa e por **matrizes hierárquicas (H-matrizes)**
- Integrais de domínio via **DIBEM** (funções de base radial)
- Soluções **estacionárias** e **transientes** (Houbolt, Method of Lines, EDO de 2ª ordem)
- **Soluções analíticas** embutidas para verificação
- Visualização de geometria e condições de contorno (`plot_geo`)

O projeto usa [DrWatson.jl](https://juliadynamics.github.io/DrWatson.jl/stable/)
para caminhos reproduzíveis (`datadir`, `srcdir`, …).

## Instalação / ativação

```julia
using Pkg
Pkg.activate("caminho/para/BEM_gmsh")
Pkg.instantiate()
```

```julia
using DrWatson
@quickactivate :BEM
```

## Exemplo mínimo

```julia
using DrWatson
@quickactivate :BEM
include(datadir("Laplace", "Laplace_dad.jl"))

props = Laplace(1.0)
msh = quadrado(ndiv=20, show=false)          # grava via datadir(...)
dad = format2d(msh, props)

attach_analytical!(dad, ana_laplace_linear(; direction=SA[1.0, 0.0]))

H_G_full_direct(dad, 20)                     # ou H_G_Hmat(dad) para malhas grandes
solve(dad)

println("erro relativo = ", rel_error(dad))
plot_geo(dad)
```

## Organização do pacote

```
src/
  BEM.jl                 # entrada do módulo
  Structures.jl          # BEMdata, Laplace, Helmholtz, Elasticity, …
  Fundamental_Solutions.jl
  Input.jl               # format2d / format3d (Gmsh)
  Assembly_full.jl       # H, G densas
  Assembly_H.jl          # H, G hierárquicas
  Boundary_conditions.jl
  Solver.jl              # estacionário + transiente
  Domain.jl              # DIBEM
  Analytical.jl          # soluções de referência
  Visualization.jl
  Hmat/                  # biblioteca de H-matrizes
data/Laplace/            # malhas + geradores (datadir)
scripts/                 # demos e benchmarks
test/
docs/                    # esta documentação (EN + pt-BR)
```

## Documentação em dois idiomas

A documentação é gerada com **Documenter.jl**. O pacote **não** tem i18n nativo
(como o Jekyll Polyglot); a abordagem usada aqui é uma **árvore paralela de
páginas** sob `docs/src/pt-br/`, listada em `docs/make.jl`.

| Idioma | Pasta | Menu |
|--------|--------|------|
| English (padrão) | `docs/src/*.md` | raiz do sidebar |
| Português (BR) | `docs/src/pt-br/*.md` | seção **Português (BR)** |

A API automática (`@docs`) permanece em inglês (nomes de símbolos Julia).
As páginas conceituais e tutoriais estão traduzidas.

## Próximas páginas

- [Começando](getting_started.md)
- [Notas de teoria](theory.md)
- [Exemplos](examples.md)
- [Desempenho](performance.md)
- API (inglês): [Data structures](../api/structures.md), …
