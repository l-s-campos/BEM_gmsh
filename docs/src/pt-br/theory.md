# Notas de teoria

Convenção de fluxo em todo o pacote:

```math
q = -k\,\frac{\partial T}{\partial n}
```

(``n`` exterior). Em 2D, ``G = -\log r/(2\pi k)``. A identidade de campo
constante fixa a diagonal de ``\mathbf{H}`` (termo livre).

DIBEM converte integrais de volume (massa / inércia) em integrais de contorno
via RBF; o resultado é ``\mathbf{M}`` em `dad.cache.M`.

Elastoplasticidade 2-D: tensão inicial em células constantes (`solve_elastoplastic!`,
von Mises). Texto completo: [Theory notes](../theory.md).

Calor (1ª ordem): ``H T - G q = M \dot T``.
Onda (2ª ordem): ``M\ddot u + A u = b`` (`solve_transient_o2`).

Trincas: BEM dual em `BEM.Crack`. Contato half-space (Pohrt–Li, desgaste Uzawa,
roda–trilho): `BEM.Contact`. Contato multibodies em `BEMdata`: `BEM.MultiRegion`.
Placas: `BEM.Plate`. Topologia: `BEM.Topology` (Pacheco, SIMP, level-set, e
forma de Portela 2012: CBIE/HBIE no contorno de projeto, sem nucleação;
3-D: densidade DIBEM-SIMP / DT-ρ em malha fixa).

Texto completo (equações e referências): [Theory notes](../theory.md).
