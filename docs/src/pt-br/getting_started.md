# Começando

```julia
using Pkg; Pkg.activate("."); Pkg.instantiate()
using BEM
```

As malhas do primeiro dia (`quadrado`, `quadrado_elasticity`) vêm de
`BEM.Examples` e são reexportadas. **Não** é preciso DrWatson nem
`include(datadir(...))` no primeiro exemplo.

## Fluxo

1. Física: `Laplace(1.0)` ou `Elasticity(E, ν, ρ)`
2. Malha: `dad = format2d(quadrado(ndiv=20, show=false), props)`
3. Analítica: `attach_analytical!(dad, ana_laplace_linear(; direction=SA[1.0, 0.0]))`
4. Montagem: `assemble!(dad, 20)` ou `assemble!(dad; method=:hmatrix)`
5. Domínio (transiente): `dibem!(dad)`
6. Solver: `solve(dad)` / `solve_Houbolt` / `solve_transient_o2`
7. Checagem: `rel_error(dad)`, `plot_geo(dad)`

## Condições de contorno

Nomes de grupos físicos no Gmsh: Laplace `"0;T"` (Dirichlet) e `"1;q"`
(Neumann, ``q=-k∂T/∂n``). O gerador `quadrado` reproduz ``T=x``.

## Testes

```bash
julia --project=. test/runtests.jl
```
