# Analytical Gmsh examples

All examples:

1. Build geometry with **Gmsh**
2. Load with **`format2d`** (discontinuous Gauss collocation; `tipo` = number of nodes − 1)
3. Compare against a **known analytical** field / quantity

Run from the project root:

```bash
julia --project=. data/examples/laplace_linear_Tx.jl
julia --project=. data/examples/potencial_direto.jl
julia --project=. data/examples/elasticity_patch.jl
julia --project=. data/examples/plate_ss_navier.jl
julia --project=. data/examples/crack_feddersen.jl
julia --project=. data/examples/geo_unit_square.jl
```

Full potential suite (timings + all Laquini / Moulton / annulus):

```bash
julia --project=. scripts/potencial_direto.jl
```

Scalar **wave propagation** (DIBEM + full-system Houbolt / DifferentialEquations):

```bash
julia --project=. scripts/wave_propagation.jl
WAVE_CASE=membrane_v0 WAVE_SOLVER=both WAVE_NDIV=16 julia --project=. scripts/wave_propagation.jl
```

Data: `data/Laplace/wave_propagation.jl` (`:bar_sudden`, `:annulus`, `:membrane_v0`,
`:bar_periodic`, `:membrane_forced`, `:ricker`).

Elasticity data split by constitutive model — see `data/elastico/README.md`:

```julia
# isotropic
include(datadir("elastico", "iso", "pressurized_tube.jl"))
include(datadir("elastico", "iso", "cattaneo_mindlin.jl"))

# anisotropic / orthotropic
include(datadir("elastico", "aniso", "composite_materials.jl"))
include(datadir("elastico", "aniso", "plate_hole_orthotropic.jl"))
```

| Script | Physics | Analytical |
|--------|---------|------------|
| `laplace_linear_Tx.jl` | Laplace | ``T=x``, ``q=-k∂T/∂n`` |
| `potencial_direto.jl` | Laplace | potencial1d, Laquini 1–3, quarter annulus, Moulton |
| `elasticity_patch.jl` | Elasticity | constant strain patch |
| `plate_ss_navier.jl` | Kirchhoff plate | Navier SS square ``w_max`` |
| `crack_feddersen.jl` | Dual BEM crack | Feddersen ``K_I`` |
| `geo_unit_square.jl` | Geometry | unit square area/centroid |

Mesh generators live next to physics folders (`Laplace/Laplace_dad.jl`, `Laplace/potencial_problems.jl`, `elastico/center_crack.jl`).
