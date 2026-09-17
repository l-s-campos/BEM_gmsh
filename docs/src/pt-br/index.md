# BEM_gmsh

Módulo Julia: **`BEM`**.

> 🌐 [English](../index.md) · **Português (BR)**

**BEM_gmsh** é um pacote Julia para o **Método dos Elementos de Contorno**,
com malhas [Gmsh](https://gmsh.info/):

- **Laplace** 2D/3D e **elasticidade linear**
- Montagem densa, **H-matriz** e **FMM**
- Integrais de domínio **DIBEM** / DRM / BEM local / SBM
- Solvers estacionários e **transientes** (Houbolt, OrdinaryDiffEq, MMM)
- Trincas (BEM dual), contato, placas, otimização topológica
- Campos **analíticos** para verificação (`rel_error`)

## Instalação

```julia
using Pkg
Pkg.activate("caminho/para/BEM_gmsh")
Pkg.instantiate()   # biblioteca de sistema do Gmsh
```

Julia ≥ 1.10. Nome do pacote: **`BEM`**. Repositório: **`BEM_gmsh`**.

## Caminho de 5 minutos

```julia
using BEM

dad = format2d(quadrado(ndiv=20, show=false), Laplace(1.0))
attach_analytical!(dad, ana_laplace_linear(; direction=SA[1.0, 0.0]))  # T = x
assemble!(dad, 20)            # ou assemble!(dad; method=:hmatrix)
solve(dad)

@show rel_error(dad)
```

**Pipeline:** malha → `format2d` → `assemble!` → `dibem!` opcional → `solve` → `dad.T` / `dad.q`.

Nomes avançados ficam em **submódulos** (`using BEM.Crack`, `BEM.Contact`, …).

## Próximas páginas

- [Começando](getting_started.md)
- [Receitas](recipes.md)
- [Teoria](theory.md)
- [API](api.md)
- [Arquitetura](architecture.md)
