# Notas de teoria

> 🌐 [English](../theory.md) · **Português (BR)**

## Equação integral de contorno (Laplace)

Para ``\nabla\cdot(k\nabla T)=0`` em um domínio ``\Omega`` de contorno ``\Gamma``,

```math
c(\mathbf{x})\,T(\mathbf{x})
+ \int_\Gamma T(\mathbf{y})\,\frac{\partial G}{\partial n_y}(\mathbf{x},\mathbf{y})\,d\Gamma_y
=
\int_\Gamma q(\mathbf{y})\,G(\mathbf{x},\mathbf{y})\,d\Gamma_y.
```

**Convenção de fluxo em todo o BEM.jl:**

```math
q = -k\,\frac{\partial T}{\partial n}
```

(com ``n`` a normal unitária **exterior**). Isso coincide com o fluxo de calor
``\mathbf{q}_{\mathrm{heat}} = -k\nabla T``.

Solução fundamental 2D consistente com essa convenção:

```math
G = -\frac{\log r}{2\pi k},\qquad
\frac{\partial G}{\partial n}
= \frac{\mathbf{r}\cdot\mathbf{n}}{2\pi r^2}.
```

A colocação produz

```math
\mathbf{H}\,\mathbf{T} = \mathbf{G}\,\mathbf{q}.
```

CDCs mistas entram por troca de colunas (colunas de Dirichlet de ``\mathbf{H}``
trocadas por ``-\mathbf{G}``), gerando ``\mathbf{A}\mathbf{x}=\mathbf{b}``.

### Benchmark do quadrado padrão

`quadrado` impõe à esquerda ``T=0``, à direita ``q=-1``, topo/base ``q=0``.
Com ``q=-k\partial T/\partial n`` e ``k=1`` o campo exato é ``T=x``.

## Termo livre / diagonal

A diagonal de ``\mathbf{H}`` vem da identidade de campo constante
``\mathbf{H}\mathbf{1}=\mathbf{0}`` (soma de linha), que fornece o termo livre
correto no contorno (``\approx -1/2``) e em pontos interiores (``\approx -1``)
para os sinais de núcleo usados aqui.

A diagonal singular de ``\mathbf{G}`` pode ser recuperada de uma identidade de
campo linear (ver `corrige_diagonais!` no caminho de H-matriz).

## Matriz de massa DIBEM

Integrais de domínio (capacidade, inércia) são aproximadas expandindo a fonte
com RBFs poliarmônicas e convertendo volume → contorno (Dupla Reciprocidade /
DIBEM). O resultado é a matriz ``\mathbf{M}`` em `dad.cache.M`.

## Formas transientes

**Calor (1ª ordem)** — `solve_transient` / `solve_Houbolt_heat`:

```math
\mathbf{H}\,\mathbf{T} - \mathbf{G}\,\mathbf{q}
= \mathbf{M}\,\dot{\mathbf{T}}.
```

**Tipo onda (2ª ordem)** — `solve_Houbolt` / `solve_transient_o2`:

```math
\mathbf{M}\,\ddot{\mathbf{u}} + \mathbf{A}\,\mathbf{u} = \mathbf{b},
```

no **sistema completo** (mesmo ``A,b,M`` do Houbolt clássico), integrado com
`SecondOrderODEProblem` e resíduo `f(ddu, du, u, p, t)`.

## Elasticidade

O núcleo de Kelvin (e Lekhnitskii anisotrópico) entra da mesma forma
``\mathbf{H}\mathbf{u}=\mathbf{G}\mathbf{t}``. Contato em semi-espaço usa
operadores de Boussinesq–Cerruti / Flamant (FFT, H-matriz ou denso) — ver
páginas de contato na API (inglês).

## Propagação de ondas (escalar)

Problemas clássicos de barra e membrana (carga súbita, velocidade inicial,
Ricker, etc.) estão em `data/Laplace/wave_propagation.jl`:

```julia
include(datadir("Laplace", "Laplace_dad.jl"))
include(datadir("Laplace", "potencial_problems.jl"))
include(datadir("Laplace", "wave_propagation.jl"))

dad, meta = wave_problem(:bar_sudden; ndiv=16, n_int=8)
H_G_full_direct(dad; npg=12)
DIBEM(dad; rbf=PHS(3; poly_deg=0))
solve_Houbolt(dad, meta.Δt, 2.0)
# ou: solve_transient_o2(dad, meta.Δt, 2.0)
```
