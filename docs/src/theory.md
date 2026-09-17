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

## Elastoplasticity (constant cells)

Plastic strain is an initial stress ``σ^p = D^e ε^p``. The displacement BIE
gains a domain integral of the Kelvin strain kernel ``E_{ijk}``. On a
**constant cell** that integral is already an edge integral of ``U``, and the
centroid is off those edges, so interior stress is Hooke applied to
``∂/∂p`` of the same edge integral (regular ``1/r``), minus ``σ^p`` in the
owning cell (``σ = σ^e - σ^p``). There is no strongly singular
``E_{ijkl}`` volume integral; Gao–Davies is only a check
([`cell_integral_Estress_gao`](@ref)). Operators ``H,G,Q`` are formed once;
each load step iterates ``σ^p`` (damped Picard, or inverse Broyden) with a
von Mises radial return at centroids. Default stress coupling is the free
term `:jump`; the full cell ``Sσ`` is consistent for uniform ``σ^p`` but
the iteration diverges. `domain=:dibem` uses the **format2d internal
points** (cell centroids) as RBF centres: near-field cell-edge ``Q``,
far ``c_k E``. That is a far-field quadrature of the same constant-cell
operator (cylinder ``u(b)`` still ~4%), not a higher-order interpolant
of a plastic jump. See [`solve_elastoplastic!`](@ref).

## DIBEM mass matrix

Domain integrals (capacity, inertia) are approximated by expanding the source
with polyharmonic spline RBFs and converting volume integrals to boundary
integrals (Dual Reciprocity / DIBEM). The result is a matrix ``\mathbf{M}``
stored in `dad.cache.M`.

## Local BEM (compact kernel)

The test function is the Laplace fundamental solution plus a quadratic
companion so that ``u_i^*`` and ``∂_r u_i^*`` vanish at a finite radius
``r_i``. Green's identity on ``Ω ∩ B(y_i,r_i)`` then has **no** integrals on
the artificial circle. The remaining volume terms

```math
\int_{Ω^i} u\,\mathrm{d}x,\qquad
\int_{Ω^i} f\,u_i^*\,\mathrm{d}x
```

are converted on ``∂(Ω ∩ B)``. ``∫_{Ω^i} u`` and ``∫_{Ω^i} f u_i^*`` use
local CPD (default ``φ=r^3`` plus polynomials) on the nodes in the ball
(`source=:global` is DIBEM lumping of the source). Green gives
``A_u u - G q_n = -M f``. A full interior disk is just ``2π Ψ(r_i)``.
See [`assemble_local_bem!`](@ref) and `docs/src/api/local_bem.md`.

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

Time-harmonic acoustics with wavenumber ``κ=ω/c``. Flux is the acoustic
normal derivative ``q=∂u/∂n`` (not the Laplace ``q=-k∂T/∂n``).
`r = x_{\mathrm{field}}-x_{\mathrm{source}}`, ``R=|r|``.

2-D (Hankel of the first kind):

```math
G=\frac{i}{4}H_0^{(1)}(κR),\qquad
H=\frac{\partial G}{\partial n_x}
=-\frac{iκ}{4}H_1^{(1)}(κR)\,\frac{r\cdot n}{R}.
```

3-D:

```math
G=\frac{e^{iκR}}{4πR},\qquad
H=\frac{\partial G}{\partial n_x}
=e^{iκR}(iκR-1)\frac{r\cdot n}{4π R^3}.
```

As ``κ\to 0`` with Laplace conductivity ``k=1``, ``G`` matches the Laplace
kernel and ``H`` matches ``-`` the Laplace double layer (the two flux
conventions). Hypersingular kernels `fundamental_hyper` are ``∂G/∂n_ξ``
and ``∂H/∂n_ξ``; collocation HBIE is [`H_G_hyper`](@ref) in 2-D and 3-D.

## H-matrix compression

Far blocks of kernel matrices are compressed with partial ACA under a
standard admissibility condition ``\mathrm{diam}\le \eta\,\mathrm{dist}``.
See the vendored `Hmat/` module (`HMatrix`, `PartialACA`, `ClusterTree`, …).

The mixed-BC solve uses packed BC blocks (`BlockMixedOperator`) with GMRES or one-level block LU.

## Dual BEM (cracks)

Coincident crack faces: displacement BIE on one face (`eq=2`) and traction BIE
on the twin (`eq=3`). Gmsh physical name type `CRACK_BC = 5`.
SIFs from COD: `sif_cod_dual`. See `using BEM.Crack`.
Reissner plates use the same layout (`build_rect_fsdt_crack`,
`assemble_fsdt_dual!`, `sif_ctod_fsdt`) for Useche 10.5.1. Unsymmetric
Hsu–Hwu Dual uses EABE 156 `T*` complete solutions on face B
(`assemble_unsym_fsdt_dual!`).

## Contact

Half-space / half-plane influence kernels (Boussinesq–Cerruti, FFT convolution)
live in `BEM.Contact`, including two-body Kalker combination, Bagault layered
anisotropy, Uzawa/orthotropic wear, and planar wheel–rail. Multibody frictional
contact on `BEMdata` (type-4 BC) lives in `BEM.MultiRegion` (active-set, SSN,
projected Newton).

## Plates

Isotropic Kirchhoff BEM (Shi–Bezine fundamentals) in `BEM.Plate`: unknowns
``(w, ∂w/∂n)``, tractions ``(V_n, M_n)``. FSDT uses 3 DOF ``(ψ_x,ψ_y,w)``
with Vander Weeën kernels (isotropic Reissner), Wang kernels (symmetric
laminate, MATLAB `KernelP`), or Hsu–Hwu 5×5 kernels (Useche 8.3.1,
unsymmetric ABD; CBIE `unsym_fsdt_kernels` with Hsu–Hwu EABE 156
`T* = (Tx,Ty,Hx,Hy,Qn)`, traction BIE `unsym_hbie_kernels`). Discontinuous elements use the shared
[`Element`](@ref) (CAD `geo`, Gauss–Legendre collocation, `Legendre`
field / `Equispaced` geometry — same as `format2d`). Large deflection couples
the plate to a plane-stress membrane BEM.

## Topology

Explicit boundary loops → collocation `BEMdata` (no Gmsh after the initial
mesh). Heat DT ``k|∇T|^2`` (2-D) / ``\tfrac{3}{2}k|∇T|^2`` (3-D spherical
cavity); plane-stress DT and 3-D isotropic spherical-cavity DT (Novotny).
Pacheco node motion and Amstutz / HJ level-set stay 2-D:
`using BEM.Topology`. 3-D is DIBEM-SIMP / DT-ρ on a fixed outer surface
(`heat_cube_3d`, `cantilever_cube_3d`); the iso-cut is marching
tetrahedra, then closed interior components are inserted as cavities and
collocation is rebuilt (`bemdata_from_iso`). Portela (2012) is
**shape** only: mixed CBIE / hypersingular collocation on the traction-free
(or insulated) design boundary, not crack twins; the free-boundary optimality
condition is constant hoop energy density ``W``, with area enforced by
``\int v_n\,d\Gamma``. That is distinct from DT nucleation (Pacheco) and from
nodal density (DIBEM-SIMP).
