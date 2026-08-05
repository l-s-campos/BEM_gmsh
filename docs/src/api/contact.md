# Half-space contact BEM (Pohrt & Li 2014)

Implementation of

> R. Pohrt & Q. Li, *Complete Boundary Element Formulation for Normal and
> Tangential Contact Problems*, Physical Mesomechanics **17** (2014) 334–340.
> DOI: [10.1134/S1029959914040109](https://doi.org/10.1134/S1029959914040109)

Module: `BEM.ContactHalfSpace` (reexported by `BEM`).

## Physics

Elastic half-space, surface only, **uniform rectangular grid**.
Surface stresses ``(τ_x, τ_y, p)`` map to surface displacements ``(u_x, u_y, u_z)``
through nine influence operators obtained by integrating Boussinesq/Cerruti
kernels over flat rectangular patches (Love-type closed forms).

Sign convention: positive ``p`` and ``u_z`` point **into** the solid.

```math
u_a^{ij} = \sum_{i'j'} K_{ab}^{ij,i'j'}\, b_b^{i'j'}
```

## Influence coefficients

```julia
hs = ElasticHalfSpace(G, ν; hx=1.0, hy=1.0)
K = influence_coeff(Kzz, di, dj, hs)   # relative offset (i-i', j-j')
```

Components: `Kxx Kxy Kxz Kyx Kyy Kyz Kzx Kzy Kzz` (`InfluenceComponent` enum).

## FFT convolution

```julia
prep = precompute_kernels(nx, ny, hs; components=(Kzz, Kxx))
u = fc_forward(p, Kzz, prep)           # u = FC(p)
```

Complexity ``O(N^2 \log N)`` per evaluation on an ``N\times N`` grid.

## Inverse (CG)

```julia
σ = fc_inverse(u_target, mask, Kzz, prep; tol=1e-8)
```

Polonsky–Keer / Pohrt–Li conjugate gradients restricted to `mask`.

## Frictionless normal contact

```julia
sol = solve_normal_contact(gap0, δ, hs)
# sol.p, sol.u, sol.contact, sol.force
```

## Coulomb partial slip

```julia
ps = solve_partial_slip(sol.p, sol.contact, d, μ; hs)
# ps.τ, ps.stick, ps.slip, ps.force_t
```

Starts from full stick, transfers points that exceed ``|τ|=μ p`` into slip,
iterates until the stick/slip partition is stable (paper §5). Decoupled
normal/tangential response (exact at ``ν=1/2``).

## 2D line contact (Flamant / Hertz cylinder)

```julia
hp = ElasticHalfPlane2D(G, ν; h=hx)
sol = solve_line_contact(gap0, δ, hp)
hz  = hertz_line(F, R, hp)          # a, p0 analytical
```

Demo: `scripts/hertz_line_2d.jl`.

## Multi-region interfaces (BC type 3)

Gmsh physical name `"3;0"` marks an interface. After loading each subregion:

```julia
prob = MultiRegionProblem([dadL, dadR])
pair_interfaces!(prob)              # nearest-neighbour pairing
assemble_multiregion(prob)
solve_multiregion!(prob)            # T continuous, q_a + q_b = 0
```

## Frictional contact (BC type 4)

Gmsh physical name `"4;μ"` (scalar) or `"4;μ;4;μ"` (elasticity) — value slot
holds the friction coefficient.

### Pairing: NTN vs NTS

```julia
pair_contacts!(prob; method=:ntn)   # node-to-node (nearest neighbour)
pair_contacts!(prob; method=:nts, slave_reg=1, master_reg=2)  # node-to-segment
# pair.state: 1=open, 2=slip, 3=stick
# NTS fills pair.master_nodes, ξ, N1, N2 for linear master segments
```

### Multibody elasticity (penalty NTN / NTS)

```julia
include(datadir("elastico", "two_blocks_contact.jl"))
props = Elasticity(100.0, 0.3, 1.0; plane_strain=true)
prob = load_two_blocks_contact(props; gap=0.02,
                               ndiv_bot=8, ndiv_top=5)  # non-matching → NTS
# close the joint by rigid approach δ > gap (penalty NTN or NTS)
# frame=:local (default) — Leonardo §4.7 (n,t) BEM; contact as (t_n, t_t)
solve_multibody_elasticity_contact!(prob; method=:nts, frame=:local,
                                    kn=5e3, kt=2e3, δ=0.035,
                                    slave_reg=1, master_reg=2)
# pair.tn, pair.tt — slave normal/tangential tractions
# pair.state: 1=open, 2=slip, 3=stick
# dad.u_local, dad.traction_local available on each region
```

### Multibody elasticity — Contato active-set / SSN / ALM

Coupled multi-region system in local ``(n,t)`` (`aplica_contato_com_atrito_multicorpos`
layout). Inner solvers share the same unknowns
``(u_{\mathrm{mix}}, t_n^1,t_t^1,t_n^2,t_t^2)``:

| `solver` | Method | Notes |
|----------|--------|-------|
| `:activeset` (default) | Contato verify → ``A,b`` → ``x=A\\b`` | frozen open/stick/±slip |
| `:ssn` | Alart–Curnier NCF + Newton | superlinear; same KKT as ALM |
| `:alm` | Uzawa augmented Lagrangian | multiplier projection + BIE with fixed ``t`` |

ALM update (open-positive gap ``g_n``, ``λ=-t``):

```math
λ_n ← Π_{ℝ_+}(λ_n - r_n g_n),\qquad
λ_t ← Π_{|·|≤μ λ_n}(λ_t - r_t g_t),
```

then ``t^1=-λ``, ``t^2=-R\\t^1``, and each region solves
``A x_{\mathrm{mix}} = b + G_c t``.

**Prefer load stepping** for large approach:

```julia
# outer loop on δ, warm-start x (recommended)
solve_contact_friction_stepped!(prob; δ_end=0.035, nsteps=10, tol=1e-8)
# semi-smooth Newton (Alart–Curnier); rn,rt default ∼ 10E/L
solve_contact_friction_stepped!(prob; δ_end=0.035, nsteps=10,
                                 solver=:ssn, tol=1e-8)
# Uzawa ALM (optional under-relaxation / r growth)
solve_contact_friction_stepped!(prob; δ_end=0.035, nsteps=10,
                                 solver=:alm, alm_omega=0.7, r_grow=1.2)
# pair.state: 1=open, ±2=slip, 3=stick; pair.tn, pair.tt from solution
# history: dad.contact_δ_hist, dad.contact_tn_hist

# single shot (less robust for large δ)
solve_contact_friction!(prob; δ=0.035)
solve_contact_friction!(prob; δ=0.035, solver=:ssn)
solve_contact_friction!(prob; δ=0.035, solver=:alm)
```

Compare active-set, SSN, ALM, and penalty at the same ``δ_end`` / mesh.

## Half-space operators with acceleration (`HalfSpaceBEM`)

```julia
dad = HalfSpace2D(-1, 1, N; E=Estar)   # E = E/(1-ν²)
K = build_operator(dad, :fft)          # :dense | :fft | :hmatrix | :fmm
p, g = contact_pressure_force(dad, K, W)
w1, w2, ph = wear_2d(dad, K, W; k_ar1=1e-13, δ=1e-5, nsteps=100)  # desgaste_2D
```

## Cattaneo–Mindlin (Loyola 2022 §9.3.1)

Two elastically similar cylinders under constant normal load ``P`` and cyclic
tangential load ``Q`` (partial slip). Parameters from Table 9.19:

```julia
par = loyola_cattaneo_params(; Q_over_fP=0.5)
# par.a ≈ 1.186 mm, par.p0 ≈ 698 MPa, load_steps A…E
```

| Model | API |
|-------|-----|
| Analytical Hertz + Cattaneo / Mindlin–Deresiewicz | `cattaneo_pressure`, `cattaneo_shear`, `mindlin_shear_history` |
| Half-space NTS (Flamant BEM) | `solve_cattaneo_halfplane` |
| Cohesive penalty + Coulomb | `solve_cattaneo_cohesive_halfplane` |
| Mortar STS (non-matching grids) | `solve_cattaneo_mortar_halfplane` |

```julia
G = par.E_eq * (1 - par.ν) / 2
x = range(-3par.a, 3par.a; length=201) |> collect
hp = ElasticHalfPlane2D(G, par.ν; h=x[2]-x[1])
sol = solve_cattaneo_halfplane(x, par.R_eq, par.P, par.Qmax, par.f, hp)
```

Mortar builds dual-ish ``D`` and projection ``M`` so master tractions satisfy
``M t_m = D t_s`` (weak action–reaction), addressing the coarse-mesh contact
half-width issue noted in the thesis (node-to-node → segment-to-segment).

## Demos

```bash
julia --project=. scripts/contact_pohrt_li.jl              # 3D surface Hertz + partial slip
julia --project=. scripts/hertz_line_2d.jl                 # 2D Flamant vs Hertz cylinder
julia --project=. scripts/cattaneo_mindlin_compare.jl     # Loyola Cattaneo–Mindlin 4 models
julia --project=. scripts/multibody_contact_ntn_nts.jl    # two-block elasticity NTN vs NTS
julia --project=. scripts/two_regions_interface.jl         # multi-region type-3
julia --project=. scripts/compare_contact_acceleration.jl # paper figures (PDF)
```

Paper draft: `OneDrive/artigos/escritos/2026/BEM-wear/main.tex`.
Thesis: `OneDrive/banca/221/2022-Fernando_Loyola.pdf`.
