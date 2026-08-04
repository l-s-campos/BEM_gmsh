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

Gmsh physical name `"4;μ"` — value slot holds the friction coefficient.

```julia
pair_contacts!(prob)                # pairs type-4 nodes, stores μ, gap0
solve_contact_friction!(prob; δ=0.01)
# pair.state: 1=open, 2=slip, 3=stick
```

## Half-space operators with acceleration (`HalfSpaceBEM`)

```julia
dad = HalfSpace2D(-1, 1, N; E=Estar)   # E = E/(1-ν²)
K = build_operator(dad, :fft)          # :dense | :fft | :hmatrix | :fmm
p, g = contact_pressure_force(dad, K, W)
w1, w2, ph = wear_2d(dad, K, W; k_ar1=1e-13, δ=1e-5, nsteps=100)  # desgaste_2D
```

## Demos

```bash
julia --project=. scripts/contact_pohrt_li.jl              # 3D surface Hertz + partial slip
julia --project=. scripts/hertz_line_2d.jl                 # 2D Flamant vs Hertz cylinder
julia --project=. scripts/two_regions_interface.jl         # multi-region type-3
julia --project=. scripts/compare_contact_acceleration.jl # paper figures (PDF)
```

Paper draft: `OneDrive/artigos/escritos/2026/BEM-wear/main.tex`.
