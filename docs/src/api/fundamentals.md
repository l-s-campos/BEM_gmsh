# Fundamental solutions

All kernels live in `src/Fundamental_Solutions.jl`, ported from the legacy
`calsolfund` / `caldsolfund` / `calsolfund_hiper` routines and rewritten with
**StaticArrays** + **Tensorial.jl**.

## Quick reference

| Physics | Type | Main API | Notes |
|---------|------|----------|-------|
| Laplace | [`Laplace`](@ref) | `fundamental`, `fundamental_hyper` | ``q=-k∂T/∂n`` |
| Helmholtz | [`Helmholtz`](@ref) | `fundamental`, `fundamental_hyper` | Hankel ``H_n^{(1)}`` |
| Isotropic elasticity | [`Elasticity`](@ref) | `fundamental`, `fundamental_stress`, `fundamental_grad` | Kelvin; `Mat{dim,dim}` |
| Anisotropic elasticity | [`AnisotropicElasticity`](@ref) | `fundamental`, `fundamental_stress`, [`lekhnitskii_params`](@ref) | Lekhnitskii |

Return types:

```julia
KernelPair(U, T)       # U → G (Neumann/traction), T → H (Dirichlet/displacement)
StressKernels(D, S)    # 3rd-order tensors for interior stress
```

Iterable: `G, H = fundamental(...)` still works via `KernelPair` iteration
when you use the `props`-based API; the `BEMdata` methods keep the legacy
`(U, T)` / `(G, H)` tuple return for assembly.

## Laplace

```julia
lap = Laplace(1.0)
r, n = Point2D(0.3, 0.4), Point2D(1, 0)
kp = fundamental(lap, r, n)          # KernelPair
G, H = kp.U, kp.T

# hypersingular (HTBIE)
kh = fundamental_hyper(lap, r, n, Point2D(0, 1))
```

2D:
```math
G=-\frac{\log R}{2\pi k},\qquad
H=\frac{\mathbf r\cdot\mathbf n}{2\pi R^2}.
```

3D:
```math
G=\frac{1}{4\pi k R},\qquad
H=\frac{\mathbf r\cdot\mathbf n}{4\pi R^3}.
```

## Helmholtz

```julia
helm = Helmholtz(; ω=2π, c=340.0)
κ = wavenumber(helm)                 # ω/c
kp = fundamental(helm, r, n)         # complex KernelPair
```

```math
G=\frac{i}{4}H_0^{(1)}(κR),\qquad
H=-κ\frac{i}{4}H_1^{(1)}(κR)\,\frac{\mathbf r\cdot\mathbf n}{R}.
```

## Isotropic elasticity (Kelvin)

Uses **Tensorial** `Mat{2,2}` / `Mat{3,3}` for the displacement and traction
kernels, and `Tensor{Tuple{2,2,2}}` for stress kernels.

```julia
el = Elasticity(210e9, 0.3, 7800.0; plane_strain=true)
kp = fundamental(el, r, n)
U, T = kp.U, kp.T                    # Mat{2,2}

# interior stress recovery kernels
sk = fundamental_stress(el, r, n)    # D, S  (3rd-order)
# σ_ij = D_kij t_k - S_kij u_k

# Cartesian derivatives of U, T
Ux, Tx, Uy, Ty = fundamental_grad(el, r, n)
```

Plane stress is obtained by the standard mapping
``ν̃ = ν/(1+ν)`` (`plane_strain=false`).

Helpers: [`shear_modulus`](@ref), [`lame_λ`](@ref), [`effective_nu`](@ref).

## Anisotropic elasticity (Lekhnitskii)

```julia
# orthotropic engineering constants → complex parameters
pars = lekhnitskii_params(E1, E2, G12, ν12; θ_deg=30)
mat  = AnisotropicElasticity(pars)

# field y, source x, normal at y
kp = fundamental(mat, y, x, n)
sk = fundamental_stress(mat, y, x, n)
```

Built from the compliance characteristic equation (legacy `Compute_Material`).

## BEMdata integration

Assembly still calls:

```julia
Gij, Hij = fundamental(dad, r, n)   # numbers or SMatrix
```

which dispatches on `dad.properties`.

## Demo script

```bash
julia --project=. scripts/fundamentals_demo.jl
```
