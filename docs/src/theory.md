# Theory notes

## Boundary integral equation (Laplace)

For ``\nabla\cdot(k\nabla T)=0`` in a domain ``\Omega`` with boundary ``\Gamma``,

```math
c(\mathbf{x})\,T(\mathbf{x})
+ \int_\Gamma T(\mathbf{y})\,\frac{\partial G}{\partial n_y}(\mathbf{x},\mathbf{y})\,d\Gamma_y
=
\int_\Gamma q(\mathbf{y})\,G(\mathbf{x},\mathbf{y})\,d\Gamma_y.
```

**Flux convention used throughout BEM.jl:**

```math
q = -k\,\frac{\partial T}{\partial n}
```

(with ``n`` the **outward** unit normal). This matches the heat-flux definition
``\mathbf{q}_{\mathrm{heat}} = -k\nabla T``.

2D fundamental solution consistent with that convention:

```math
G = -\frac{\log r}{2\pi k},\qquad
\frac{\partial G}{\partial n}
= \frac{\mathbf{r}\cdot\mathbf{n}}{2\pi r^2}.
```

Collocation yields

```math
\mathbf{H}\,\mathbf{T} = \mathbf{G}\,\mathbf{q}.
```

Mixed BCs are imposed by column swapping (Dirichlet columns of ``\mathbf{H}``
replaced by ``-\mathbf{G}``), producing ``\mathbf{A}\mathbf{x}=\mathbf{b}``.

### Default square benchmark

`quadrado` sets left ``T=0``, right ``q=-1``, top/bottom ``q=0``.
With ``q=-k\partial T/\partial n`` and ``k=1`` the exact field is ``T=x``.

## Free-term / diagonal

The diagonal of ``\mathbf{H}`` is set from the constant-field identity
``\mathbf{H}\mathbf{1}=\mathbf{0}`` (row-sum), which automatically gives the
correct free term on the boundary (``\approx -1/2``) and at interior
collocation points (``\approx -1``) for the kernel signs used here.

The singular diagonal of ``\mathbf{G}`` can be recovered from a linear-field
identity (see `corrige_diagonais!` on the H-matrix path).

## DIBEM mass matrix

Domain integrals (capacity, inertia) are approximated by expanding the source
with polyharmonic spline RBFs and converting volume integrals to boundary
integrals (Dual Reciprocity / DIBEM). The result is a matrix ``\mathbf{M}``
stored in `dad.cache.M`.

## Transient forms

**Heat (1st order)** — `solve_transient` / `solve_Houbolt`:

```math
\mathbf{H}\,\mathbf{T} - \mathbf{G}\,\mathbf{q}
= \mathbf{M}\,\dot{\mathbf{T}}.
```

**Wave-like (2nd order)** — `solve_transient_o2`:

```math
\mathbf{M}\,\ddot{\mathbf{u}} + \mathbf{A}\,\mathbf{u} = \mathbf{b},
```

integrated with `SecondOrderODEProblem` and residual signature
`f(ddu, du, u, p, t)`.

## Elasticity

Kelvin fundamental solutions give tensor kernels ``\mathbf{U},\mathbf{T}``
(implemented with **Tensorial.jl** `Mat{dim,dim}`). Stress recovery uses the
third-order kernels ``D_{kij}, S_{kij}`` (`fundamental_stress`).
Anisotropic 2D materials use Lekhnitskii complex potentials
(`lekhnitskii_params`, `AnisotropicElasticity`).

Assembly builds block matrices of size `(dim·N)²`. BCs are per DOF
(displacement / traction).

## Helmholtz

Time-harmonic acoustics with wavenumber ``κ=ω/c`` and Hankel kernels
``H_0^{(1)}, H_1^{(1)}`` (`Helmholtz`, `fundamental` / `fundamental_hyper`).

## H-matrix compression

Far blocks of kernel matrices are compressed with partial ACA under a
standard admissibility condition ``\mathrm{diam}\le \eta\,\mathrm{dist}``.
See the vendored `Hmat/` module (`HMatrix`, `PartialACA`, `ClusterTree`, …).

The mixed-BC solve uses a matrix-free [`MixedBCOperator`](@ref) with GMRES.
