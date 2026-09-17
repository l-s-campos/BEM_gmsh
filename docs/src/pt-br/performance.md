# Desempenho

Montagem densa: `assemble!(dad; threaded=true)` (linhas de colocação
independentes). `JULIA_NUM_THREADS=8`.

```bash
julia --project=. scripts/profile/profile_assembly.jl
```

GPU (Laplace 2-D): `using CUDA` then `assemble!(dad; method=:gpu, T=Float32)`.
DIBEM denso: `DIBEM(dad; method=:gpu)` (default `Float64`). Os kernels são
KernelAbstractions (Julia). Comparação:
`scripts/laplace/gpu_vs_cpu_assembly.jl`,
`scripts/dibem/gpu_vs_cpu_dibem.jl`.

H-matriz / FMM para ``N`` grande. Perfil completo: [Performance](../performance.md).
