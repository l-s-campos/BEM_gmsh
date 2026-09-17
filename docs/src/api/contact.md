# Contact

`using BEM.Contact`. Half-space (Pohrt–Li), layered/anisotropic Fourier
compliance, Uzawa wear, rolling, planar wheel–rail, 2-D half-plane,
Cattaneo–Mindlin, mortar.

Multibody frictional contact on `BEMdata` is [`BEM.MultiRegion`](@ref).

```@docs
BEM.Contact
BEM.Contact.ElasticHalfSpace
BEM.Contact.combined_halfspace
BEM.Contact.solve_normal_contact
BEM.Contact.build_pohrt_operator
BEM.Contact.loyola_cattaneo_params
BEM.Contact.cattaneo_pressure
BEM.Contact.cattaneo_shear
```

## Pohrt–Li half-space

> R. Pohrt & Q. Li, *Complete Boundary Element Formulation for Normal and
> Tangential Contact Problems*, Physical Mesomechanics **17** (2014) 334–340.
> DOI: [10.1134/S1029959914040109](https://doi.org/10.1134/S1029959914040109)

Elastic half-space, surface only, **uniform rectangular grid**. Surface
stresses ``(τ_x, τ_y, p)`` map to surface displacements ``(u_x, u_y, u_z)``
through nine influence operators obtained by integrating Boussinesq/Cerruti
kernels over flat rectangular patches (Love-type closed forms).

Sign convention: positive ``p`` and ``u_z`` point **into** the solid.

```math
u_a^{ij} = \sum_{i'j'} K_{ab}^{ij,i'j'}\, b_b^{i'j'}
```

```julia
using BEM.Contact
hs = ElasticHalfSpace(G, ν; hx=1.0, hy=1.0)
K = influence_coeff(Kzz, di, dj, hs)   # relative offset (i-i', j-j')
# two-body Kalker combination (Juliá Lerma / Paper 1 eq. 6)
hs2 = combined_halfspace(G_A, ν_A, G_B, ν_B; hx=1.0, hy=1.0)
prep = precompute_kernels(nx, ny, hs; components=(Kzz, Kxx))
u = fc_forward(p, Kzz, prep)
ux, uy, uz = fc_displacements(px, py, pn, prep)
sol = solve_normal_contact(gap0, δ, hs)
ps = solve_partial_slip(sol.p, sol.contact, d, μ, hs)
# same solvers with H-matrix / H² / FMM (Kzz) matvecs
prepH = precompute_kernels(nx, ny, hs; components=(Kzz,), method=:hmatrix)
prep2 = precompute_kernels(nx, ny, hs; components=(Kzz,), method=:h2)
prepF = precompute_kernels(nx, ny, hs; components=(Kzz,), method=:fmm)
```

Uniform grids stay on `:fft` (exact circulant embedding). `:hmatrix`
compresses the flattened Love/Cerruti matrix with the same `assemble_hmatrix`
path as Laplace DIBEM. `:fmm` (`Kzz`) uses that Laplace 3-D FMM
(`fmm_laplace3d_matrix`, octree plan) as the far `1/r` field, scaled by
`4A/E*`, then replaces a near stencil with Love `influence_coeff` — the
same split as `HalfSpaceBEM` `lfmm3d` + exact near panels. H / H² already
assemble Love, so they match FFT without that correction.

Identical materials give coupling `K = 0`. `Kxz`/`Kyz` use principal-value
`atan(y/x)` (not `atan2`) so the kernels stay odd.

## 2D line contact (Flamant / Hertz cylinder)

```julia
hp = ElasticHalfPlane2D(G, ν; h=hx)
sol = solve_line_contact(gap0, δ, hp)
hz  = hertz_line(F, R, hp)
```

Demo: `scripts/contact/hertz_line_2d.jl`.

## Layered / anisotropic half-space (Bagault et al. 2013)

Fourier-domain surface compliance of a coated anisotropic solid, same FFT
contact solver as Pohrt. Stroh eigenvalues in each layer, propagator through
the coating, radiation condition in the substrate. Isotropic materials are
sent to Stroh as cubic with Coulomb modulus ±1 % (Bagault §4.1).

```julia
hs = isotropic_halfspace(E, ν; hx, hy)
hs = isotropic_coated(Ec, νc, Zc, Es, νs; hx, hy)
C  = orthotropic_C(E1, E2, E3, ν12, ν13, ν23, G12, G13, G23)
hs = homogeneous(rotate_C_about_x(C, θm); hx, hy)
prep = precompute_kernels(nx, ny, hs; components=(Kzz,))
sol  = solve_normal_contact(gap0, δ, hs; prep=prep)
```

Reference: C. Bagault, D. Nélias, M.C. Baietto, T.C. Ovaert, *Int. J. Solids
Struct.* **50** (2013) 743–754. Paper figures:
`scripts/julia_lerma/bagault_paper.jl`. Gates: `test/test_layered_aniso.jl`.

## Orthotropic contact + wear (Juliá Lerma 2025 Ch. 2)

Uzawa / Alart–Curnier on the same FFT operator: elliptic Coulomb `(μ1, μ2, β)`,
orthotropic Archard wear `(i1, i2)`, optional load control on `δ`.

```julia
hs = combined_halfspace(G, ν, G, ν; hx, hy)
grid = make_grid(x, y, hs, sphere_gap(x, y, R))
prep = precompute_kernels(nx, ny, hs)
st = init_state(grid)
law = OrthotropicLaw(μ1, μ2, i1, i2, β)   # β in radians
solve_contact_step!(st, grid, prep, law, δ, gx_o, gy_o)
P, Qx, Qy = contact_resultants(st, hs)
σ = subsurface_stress(0.0, 0.0, z, st.ptx, st.pty, st.pn, x, y, hs, ν)
```

Rolling: `solve_rolling_step!(st, grid, prep, law, RollingKinematics(V, ξx, ξy, φ), δ)`.
Wear is the circumferential groove ``∫ |p_n| ‖s‖_i dx / V`` (Paper 3 eqs. 16, 33).

Load control: `set_approach_for_load!` (Hertz ``P ∝ δ^{3/2}``) for fresh
spherical contact; `match_load!` (Sneddon ``2 E^* a``) after wear / punches;
`rolling_match_load!` for rolling.

Notes: `_research/julia_lerma_2025/`. Demos: `scripts/julia_lerma/`.
Gates: `test/test_julia_lerma_contact.jl`, `test/test_julia_lerma_convergence.jl`.

## Wheel–rail (Vollebregt CONTACT module 1, D=2)

Planar wheel–rail in front of the Pohrt–Uzawa rolling solver. SIMPACK
`.prr/.prw` and slice catalogues `.slcw` live in `data/contact/vollebregt/`.

```julia
rail = read_rail_profile(joinpath(vollebregt_data_dir(), "MBench_UIC60_v3.prr"))
wheel = read_wheel_profile(joinpath(vollebregt_data_dir(), "MBench_S1002_v3.prw"))
res = solve_wheel_rail(TrackGeom(), WheelsetGeom(z=0.1981, vs=2000.0), rail, wheel;
                       side=:left)
```

Demo: `scripts/julia_lerma/wheel_rail_mbench.jl`. Gates: `test/test_wheel_rail.jl`.

## Legacy half-space operators (`HalfSpaceBEM`)

Log-kernel stack (`E = E/(1-ν²)`). Coupled Pohrt–Li 9-kernel contact is
`ContactHalfSpace` + `OrthotropicUzawa`. The 2-D kernel here is ``-4/(π E)``
(twice Flamant if `E` is combined ``E*``). `wear_2d`'s `δ` is sliding distance.

```julia
dad = HalfSpace2D(-1, 1, N; E=Estar)
K = build_operator(dad, :fft)          # :dense | :fft | :hmatrix | :h2 | :fmm
p, g = contact_pressure_force(dad, K, W)
```

## Cattaneo–Mindlin (Loyola 2022 §9.3.1)

```julia
par = loyola_cattaneo_params(; Q_over_fP=0.5)
G = par.E_eq * (1 - par.ν) / 2
x = range(-3par.a, 3par.a; length=201) |> collect
hp = ElasticHalfPlane2D(G, par.ν; h=x[2]-x[1])
hist = solve_cattaneo_history_halfplane(x, par.R_eq, par.f, hp, par.load_steps)
```

Demos: `scripts/contact/cattaneo_mindlin_compare.jl`,
`scripts/contact/contact_pohrt_li.jl`, `scripts/contact/hertz_line_2d.jl`.
