# Performance, threading, GPU, AD

## Singular point (Newton)

`closest_point_1d` / `closest_point_2d` replace the old linear projection of the
source point onto the element. They solve
``(x(ξ)-p)·x'(ξ)=0`` (curve) or the analogous 2×2 system on surfaces with
Newton iteration and clamp to the reference element.

Used automatically inside `transform` → sinh-transform quadrature.

## Threading

```julia
assemble!(dad; npg=20, threaded=true)  # default
```

Collocation **rows** are independent; set

```bash
JULIA_NUM_THREADS=8 julia --project=. ...
```

Check with `Threads.nthreads()`.

## GPU (2-D Laplace)

Far-field `H` and `G` run as a KernelAbstractions kernel (Julia, not CUDA-C).
Near and singular pairs stay on the CPU (`integrate_element`). The linear
solve is host LinearSolve.jl.

```julia
using CUDA   # loads the device; `using BEM` does not
using BEM
dad = format2d(quadrado(ndiv=40, show=false), Laplace(1.0))
attach_analytical!(dad, ana_laplace_linear(; direction=SA[1.0, 0.0]))
assemble!(dad; method=:gpu, T=Float32, npg=12)  # or H_G_gpu(dad; T=Float64)
solve(dad)
gpu_float_support()
```

`device=:cpu` runs the same kernel on the KA CPU backend (tests, no NVIDIA).

This machine: **GTX 1660 SUPER**, Turing SM 7.5, 6 GiB, no Tensor cores.

| Format | Hardware | Notes |
|--------|----------|--------|
| `Float32` | native | default GPU assembly |
| `Float64` | IEEE, ~1/32 of FP32 | accuracy path |
| `Float16` | native CUDA cores | too weak for `log` / `1/R` |
| BF16 / TF32 / FP8 | no | Ampere+ / Ada+ |

2-D isotropic elasticity (Kelvin) uses the same far-field split:

```julia
assemble!(dad; method=:gpu, T=Float32)          # H, G
DIBEM(dad; method=:gpu, T=Float64)              # F, D on GPU; IF/ID on host
```

DIBEM (pairwise `F`, `D` and far RIM) uses the same kernels:

```julia
DIBEM(dad; method=:gpu, T=Float64, rbf=PHS(3; poly_deg=1))  # default T=Float64 (F-solve)
# or DIBEM_gpu(dad; device=:cpu)  # KA CPU backend
```

H-matrix / NNCA apply (packed leaves; ACA still on the host).
`H * CuArray(x)` keeps permute and GEMV on the device:

```julia
using CUDA
H_G_Hmat(dad; device=:cuda)                 # or format=:H2
DIBEM(dad; method=:hmatrix, device=:cuda)
DIBEM(dad; method=:h2, device=:cuda)
yd = H * CuArray(x)                         # no per-apply host copy
```

Compare CPU vs GPU assembly and solve:

```bash
julia --project=. scripts/laplace/gpu_vs_cpu_assembly.jl
julia --project=. scripts/dibem/gpu_vs_cpu_dibem.jl
julia --project=. scripts/elasticity/gpu_vs_cpu_assembly.jl
julia --project=. scripts/hmat_fmm/gpu_vs_cpu_hmat.jl
```

## Profiling / allocations

```bash
julia --project=. scripts/profile/profile_assembly.jl
```

Kirchhoff plates are `BEMdata{<:ThinPlate}` and use the same
`assemble!` / `H_G_full_direct` loop as 2-D elasticity (far lumping, sinh
near-field, Guiggiani). Override with `near_factor=Inf` or `threaded=false`.

Uses `BenchmarkTools`. Typical gains already in tree:

| Technique | Effect |
|-----------|--------|
| Row-wise `@threads` | near-linear speedup on assembly |
| Precomputed `Xel` node arrays | fewer index allocations |
| Far-field point collocation | skips quadrature when ``r>1.5L`` |
| `@inbounds` on hot loops | less bounds checking |
| H-matrix / FFT half-space | sub-quadratic large-N |

Further ideas: `Bumper.jl` / preallocated quadrature buffers per thread;
Struct-of-arrays for nodes; `@turbo` (LoopVectorization) on far-field.

## Automatic differentiation

Transient reduced systems are pure linear algebra:

```julia
prob, sys = build_heat_ode(dad; tspan=(0,1))
# out-of-place, Dual-friendly:
du = heat_rhs(u, prob.p, t)

using ForwardDiff
g = u -> sum(abs2, heat_rhs(u, prob.p, 0.0))
∇g = ForwardDiff.gradient(g, u)
```

Second-order: `build_wave_ode`, `wave_rhs!`.

Avoid mutating `dad.cache` inside differentiated code paths; pass `(B,f)` as
the ODE parameter.

## 3D BEM

```julia
include(datadir("Laplace", "cube_mesh.jl"))
dad = format3d(mesh_cube(ndiv=4), Laplace(1.0); pontointerno=false)
H_G_full_direct(dad; npg=8, threaded=true)
solve(dad)
```

Helmholtz far lumping is [`auto_near_factor`](@ref): `near_factor=1.5` when
``κ L_{\max}\le 0.6`` (≳10 points per wavelength), otherwise `Inf`. Override
with `assemble!(dad; near_factor=…)`.

Nearly-singular faces: `dad.nearfield` is `:tanp3c` by default (polar +
Granados eq. 41). `:polar` is polar + sinh. `:dibem` interpolates the
full integrand on the parent square from vertices + edge Gauss (PHS3 +
linear) when `d/L < 0.05` and restores the spike with an analytic
radial `ID`; farther faces use `:auto`.

Half-space 3D contact grid:

```julia
dad3 = HalfSpace3D(-1, 1, 32, -1, 1, 32; E=1.0)
K = build_operator(dad3, :fft)   # or :dense, :hmatrix, :fmm
p, g = contact_pressure_force(dad3, K, W)
```
