# Desempenho, threads, GPU e AD

> 🌐 [English](../performance.md) · **Português (BR)**

## Ponto singular (Newton)

`closest_point_1d` / `closest_point_2d` substituem a projeção linear antiga do
ponto fonte no elemento. Resolvem
``(x(ξ)-p)·x'(ξ)=0`` (curva) ou o sistema 2×2 análogo em superfícies com
Newton e projeção no elemento de referência.

Usados automaticamente em `transform` → quadratura com transformação seno
hiperbólico.

## Threads

```julia
H_G_full_direct(dad; npg=20, threaded=true)  # padrão
```

Linhas de colocação são independentes; use

```bash
JULIA_NUM_THREADS=8 julia --project=. ...
```

Verifique com `nthreads_bem()`.

## Perfil / alocações

```bash
julia --project=. scripts/profile_assembly.jl
```

Usa `BenchmarkTools`. Ganhos típicos já no código:

| Técnica | Efeito |
|---------|--------|
| `@threads` por linha | speedup quase linear na montagem |
| Arrays `Xel` pré-computados | menos alocações de índice |
| Colocação de campo distante | evita quadratura se ``r>2L`` |
| `@inbounds` nos loops quentes | menos checagem de limites |
| H-matriz / FFT semi-espaço | sub-quadrático para N grande |

Ideias futuras: `Bumper.jl` / buffers de quadratura por thread;
Struct-of-arrays; `@turbo` (LoopVectorization) no campo distante.

## GPU

A avaliação de núcleo no campo distante é data-paralela e mapeia para
`KernelAbstractions` / CUDA. Newton + quad adaptativa no campo próximo é
ramificada → permanece na CPU.

Esqueleto: `farfield_gpu!`, ativo com `BEM_USE_GPU=1` e CUDA funcional.
Montagem GPU completa é trabalho futuro.

## Diferenciação automática

Sistemas reduzidos de calor são álgebra linear pura:

```julia
prob, sys = build_heat_ode(dad; tspan=(0,1))
# fora de lugar, amigável a Dual:
```

Para onda, `build_wave_ode` / `wave_full_rhs!` operam no sistema completo
``M ü + A u = b`` (mesmo operadores do Houbolt). A qualidade de ``M`` (DIBEM)
domina a estabilidade — veja testes em `test/test_wave_propagation.jl`.

## Contato half-space

Em grades uniformes, FFT (embedding circulante) costuma vencer H-matriz e FMM
didático em tempo e memória. Comparativo:

```bash
julia --project=. scripts/compare_contact_acceleration.jl
```
