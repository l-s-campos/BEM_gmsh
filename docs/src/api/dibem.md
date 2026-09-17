# DIBEM

Domain integrals via RBF interpolation, converted to boundary integrals.
Result: mass-like `M` in `dad.cache.M`.

```@docs
DIBEM
dibem!
DIBEM_gpu
DibemFactoredOperator
solve_poisson_dibem!
```

2-D dense DIBEM on the GPU: `DIBEM(dad; method=:gpu)` (Laplace or isotropic
elasticity; KernelAbstractions; `using CUDA`). Laplace near RIM stays on the
CPU; elasticity IF/ID is host Gauss.

3-D: same `DIBEM` / `solve_poisson_dibem!` on a `format3d` cube (RIM uses
``(n·r)/R³`` and ``∫_0^R φ ρ² dρ``). Compression: `DIBEM(dad; method=:hmatrix)`
(Laplace also `:fmm` / `:h2`; Kelvin `:hmatrix` is a tensor H-matrix
on the point tree, `:h2` is block NNCA, `:fmm` is 2-D or 3-D). Elastic body
forces: `solve_thermoelastic!(dad; bodyforce=...)`.

Off-element **surface** integrals on a 3-D face can use the same product
interpolant on the parent square (`nearfield=:dibem`): PHS3 + linear at
vertices and edge Gauss, spike recovered by an analytic polar `ID` of
`1/R` and `(r·n)/R³`, when `d/L < 0.05`. Far faces use `:auto`.
On-element stays Guiggiani. Default 3-D map is still `:tanp3c`.

Compressed backends: `DIBEM(dad; method=:hmatrix|:h2|:fmm)`
(factored `M`). `device=:cuda` packs the H-matrix / NNCA apply on the GPU
(assembly still CPU; `using CUDA` first).

Kirchhoff plates: `dibem_plate!` (BEM atual `Monta_M_RIMd`) builds `M` on
w-columns of the 2-DOF + corner layout. FSDT uses `dibem_fsdt!`.

Diffuse–advective (variable velocity): `solve_diffuse_advective!`,
`setup_da_square_exp_mxy`.

## Heterogeneous conductivity (DIBEM + DST)

Barcelos, Loeffler & Lara (EABE 2021): `∇·(K ∇u)=0` with Poisson `u*`.
The `∇K` domain integral is interpolated by DIBEM (regularized kernel
`[u(X)-u(ξ)] ∇K·∇u*`) and taken to `Γ`. Homogeneous `K` recovers
`assemble!` / `solve`. Geometry `dM` is cached on `dad`; changing `K`
only updates `∇K` and `A = dM diag(∇K)` (row-sum regularized).

```julia
dad = format2d(quadrado(ndiv=12, show=false), Laplace(1.0); pontointerno=true)
assemble!(dad)
solve_heterogeneous!(dad, p -> 1 + p[2])   # K(x,y)=1+y
```

`heterogeneous_K_operator`, `heterogeneous_system`, `HeterogeneousSector`.
Check: `scripts/dibem/heterogeneous_laplace.jl`.

## Anisotropic conductivity (constant ``K``)

- Strategy 2: [`AnisotropicLaplace`](@ref) Green's function.
- Strategy 1: `solve_anisotropic_dibem!` — Poisson FS + DIBEM residual
  ``f^*=(1/k)∇·(ΔK∇u)`` (RBF Hessian of ``u``).
- Strategy 3: `solve_anisotropic_ibp!` — one IBP, then DIBEM on
  ``∫∇Φ·(ΔK∇u)``. Radial weights are `int(rbf,x,xj)` (same ``c`` as
  [`DIBEM`](@ref)); RIM of ``∇Φ`` is analytic. No Hessian of ``u``.

Script: `scripts/laplace/ortho_plate_fenics_compare.jl`.

## EHL semi-system (SLIPPY equivalent)

Staggered coupling of heterogeneous DIBEM Reynolds with a Pohrt–Li FFT
half-space on the Reynolds interior grid (`solve_semi_system!`).
Ball-on-flat comparison: `scripts/laplace/slippy_semi_system.jl`.

## Lubrication (Reynolds / Guiggiani 2020)

`𝒫 = p h^{3/2}` turns Reynolds into `∇²𝒫 + f 𝒫 = g`. Laplace `u*` plus
DIBEM mass: `(H + M Diag(f)) 𝒫 − G q = M g`. Special films `h1`–`h5`
make `f` constant; the linear wedge is allowed (`f` variable).

```julia
film = film_h2(; a=2.0, hi=2.0)          # hi/ho = 2, Guiggiani Fig. 5
msh = mesh_guiggiani_pad(show=false)
dad = format2d(msh, Laplace(1.0); pontointerno=false)
internal_grid!(dad, 13, 9; d_min=0.02, layout=:cell)
assemble!(dad; npg=12)
solve_reynolds_dibem!(dad, film)
p_nd = reynolds_pressure.(dad.T, Ref(film))
```

`solve_reynolds_particular!` is the paper's particular-integral Laplace
path (`h2` only). Infinite bearing: `infinite_bearing_pressure`.

Schultz et al. 2025 Interpretation I (JFO as a characteristic fixed
point): `P(θ)` is that heterogeneous DIBEM solve; `C(p)` traces
`θ = h(x_r)/h(x)` along streamlines. `solve_reynolds_cfp!`.
Script: `scripts/laplace/schultz_cfp_dibem.jl`.

Unfolded journal (periodic in `x`): pair `x=0` with `x=L`

```julia
mark_periodic_x!(dad; x0=0.0, x1=L)           # p_L=p_R, qn_L+qn_R=0
solve_reynolds_cfp!(dad, film; periodic_x=true)
```

Full-film Reynolds without the `𝒫` map is the same heterogeneous DIBEM
as `∇·(K ∇u)=f` with `K=h³` and `f=6μU ∂h/∂x`:

```julia
solve_reynolds_het!(dad, film_linear(; a=2.0, hi=2.0))
```

Script: `scripts/laplace/guiggiani_lubrication.jl`.

## Mass-conserving cavitation (Elrod–Adams)

JFO complementarity `(p-p_cav)(1-θ)=0` is a convection–diffusion
problem, not a linear BEM. Structured FVM / Ausas sweep:

```julia
c = profito_single_slider()
p, θ = solve_elrod_1d(c.h, c.x[2]-c.x[1]; U=c.U, rheo=c.rheo,
    pleft=c.pleft, pright=c.pright, opt=c.opt)
```

Rheology: Dowson–Higginson `ρ(p)`, Barus `μ(p)`, optional Eyring.
Cases: `profito_single_slider`, `profito_double_slider`,
`profito_journal`, `profito_squeeze`, `profito_pocket`.
Script: `scripts/laplace/profito_cavitation.jl` (Profito 2015 §4.1).
