# Arquitetura

```text
malha (Gmsh) → format2d → BEMdata → assemble! → [dibem!] → solve → dad.T
```

`using BEM` é a espinha de ensino. Especialistas:

`BEM.Crack`, `BEM.Contact`, `BEM.Plate`, `BEM.Topology`, `BEM.MultiRegion`,
`BEM.HMatrices`, `BEM.FMM`.

Não existe `module Laplace`: esse nome é o tipo do problema.

CI: `test/runtests.jl` (um arquivo por família). Demos: `scripts/`.

Detalhe: [Architecture](../architecture.md).
