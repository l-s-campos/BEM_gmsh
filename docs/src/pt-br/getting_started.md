# Começando

> 🌐 [English](../getting_started.md) · **Português (BR)**

## Ativação do projeto

```julia
using DrWatson
@quickactivate :BEM   # carrega o módulo BEM e o ambiente do projeto
```

Caminhos de dados devem usar os helpers do DrWatson:

```julia
datadir("Laplace", "quadrado.msh")
srcdir("Solver.jl")
```

Os geradores em `data/Laplace/Laplace_dad.jl` já chamam `datadir`.

## Fluxo de trabalho

1. **Escolher a física**
   ```julia
   props = Laplace(1.0)                    # k = condutividade
   # props = Elasticity(E, ν, ρ)
   ```

2. **Gerar ou carregar a malha**
   ```julia
   include(datadir("Laplace", "Laplace_dad.jl"))
   msh = quadrado(ndiv=20, show=false)
   dad = format2d(msh, props; pontointerno=true)
   ```

3. **(Opcional) associar solução analítica**
   ```julia
   ana = ana_laplace_linear(; direction=SA[1.0, 0.0])  # T=x; q=-k ∂T/∂n
   attach_analytical!(dad, ana)
   # ou impor Dirichlet puro a partir do campo:
   # apply_analytical_bc!(dad, ana)
   ```

4. **Montar**
   ```julia
   H_G_full_direct(dad, 20)   # densa
   # H_G_Hmat(dad; atol=1e-6) # hierárquica, N grande
   ```

5. **Termo de domínio (transiente / carga de volume)**
   ```julia
   DIBEM(dad)                 # monta a matriz tipo massa M
   ```

6. **Resolver**
   ```julia
   solve(dad)                           # estacionário
   # solve_Houbolt(dad, Δt, tf)         # onda, esquema Houbolt (sistema completo)
   # solve_transient(dad, Δt, tf)       # 1ª ordem (calor)
   # solve_transient_o2(dad, Δt, tf)    # 2ª ordem (onda, DiffEq)
   ```

7. **Verificar e plotar**
   ```julia
   rel_error(dad)
   plot_geo(dad)
   # export_results_to_gmsh(dad, msh, :T; viewer=false)
   ```

## Codificação das condições de contorno

Os **nomes** dos grupos físicos no Gmsh carregam a CDC:

| Problema | Padrão do nome | Significado |
|----------|----------------|-------------|
| Laplace | `"0;T"` | Dirichlet, valor `T` |
| Laplace | `"1;q"` | Neumann, valor `q = -k ∂T/∂n` |
| Elasticidade 2D | `"tx;ux;ty;uy"` | tipo/valor por componente |

Exemplo (`quadrado`): esquerda `"0;0"`, direita `"1;-1"`, topo/base `"1;0"`
→ campo exato ``T=x`` (porque ``q=-k∂T/∂n``).

## Densa vs H-matriz

| | Densa `H_G_full_direct` | Hierárquica `H_G_Hmat` |
|--|-------------------------|-------------------------|
| Custo | ``O(N^2)`` memória/tempo | tipicamente ``O(N\log N)`` |
| Integração | singular + quase-singular | colocação + correção diagonal |
| Ideal para | ``N \lesssim 5\cdot 10^3`` | malhas grandes |
| Solver | LU / `LinearSolve` | GMRES em [`MixedBCOperator`](@ref) |
