# Performance, threading, GPU, AD

## Singular point (Newton)

`closest_point_1d` / `closest_point_2d` replace the old linear projection of the
source point onto the element. They solve
``(x(ξ)-p)·x'(ξ)=0`` (curve) or the analogous 2×2 system on surfaces with
Newton iteration and clamp to the reference element.

Used automatically inside `transform` → sinh-transform quadrature.

## Threading

```julia
H_G_full_direct(dad; npg=20, threaded=true)  # default
```

Collocation **rows** are independent; set

```bash
JULIA_NUM_THREADS=8 julia --project=. ...
```

Check with `nthreads_bem()`.

## Profiling / allocations

```bash
julia --project=. scripts/profile_assembly.jl
```

Uses `BenchmarkTools`. Typical gains already in tree:

| Technique | Effect |
|-----------|--------|
| Row-wise `@threads` | near-linear speedup on assembly |
| Precomputed `Xel` node arrays | fewer index allocations |
| Far-field point collocation | skips quadrature when ``r>2L`` |
| `@inbounds` on hot loops | less bounds checking |
| H-matrix / FFT half-space | sub-quadratic large-N |

Further ideas: `Bumper.jl` / preallocated quadrature buffers per thread;
Struct-of-arrays for nodes; `@turbo` (LoopVectorization) on far-field.

## GPU

Far-field kernel evaluation is data-parallel and maps to
`KernelAbstractions` / CUDA. Near-field Newton + adaptive quad is
branchy → keep on CPU.

Scaffold: `farfield_gpu!`, enabled when `BEM_USE_GPU=1` and CUDA is functional.
Full GPU assembly is future work.

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

Half-space 3D contact grid:

```julia
dad3 = HalfSpace3D(-1, 1, 32, -1, 1, 32; E=1.0)
K = build_operator(dad3, :fft)   # or :dense, :hmatrix
p, g = contact_pressure_force(dad3, K, W)
```
