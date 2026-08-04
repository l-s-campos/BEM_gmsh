# Referência de API

> 🌐 As páginas de API com `@docs` ficam em **inglês** (nomes dos símbolos Julia).
> Use os links abaixo.

| Tópico | Página |
|--------|--------|
| Estruturas de dados | [Data structures](../api/structures.md) |
| Soluções fundamentais | [Fundamentals](../api/fundamentals.md) |
| Entrada de malha | [Mesh I/O](../api/input.md) |
| Montagem | [Assembly](../api/assembly.md) |
| Solvers | [Solvers](../api/solvers.md) |
| Soluções analíticas | [Analytical](../api/analytical.md) |
| Visualização | [Visualization](../api/visualization.md) |
| H-matrizes | [H-matrices](../api/hmatrices.md) |
| Contato half-space | [Contact](../api/contact.md) |
| Trincas | [Crack](../api/crack.md) |

Convenções importantes (também em [Notas de teoria](theory.md)):

- Fluxo Laplace: ``q = -k\,\partial T/\partial n`` (normal exterior)
- Colocação descontínua Gauss em `format2d` (`tipo` = nº de nós − 1)
- Onda: `solve_Houbolt` e `solve_transient_o2` no sistema completo ``M ü + A u = b``
