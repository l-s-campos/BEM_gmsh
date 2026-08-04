# Elasticity data

Split by constitutive model:

```
elastico/
  iso/     # isotropic Elasticity(...)
  aniso/   # orthotropic / Lekhnitskii AnisotropicElasticity(...)
  *.msh    # legacy meshes at this level (optional)
```

## Isotropic — `iso/`

| File | Problem |
|------|---------|
| `pressurized_tube.jl` | thick tube, Lamé |
| `plate_with_hole.jl` | Kirsch plate |
| `cantilever_beam.jl` | Timoshenko beam |
| `pressurized_cavity.jl` | cavity (isotropic) |
| `cattaneo_mindlin.jl` | frictional Hertz contact params |
| `bulk_fretting.jl` | fretting + bulk load |
| `fatigue_life.jl` | Al/Ti life materials |
| `center_crack.jl` | center-cracked plate (dual BEM) |

```julia
include(datadir("elastico", "iso", "pressurized_tube.jl"))
include(datadir("elastico", "iso", "center_crack.jl"))
```

## Anisotropic — `aniso/`

| File | Problem |
|------|---------|
| `hollow_cylinder_pressure.jl` | quarter hollow cylinder (E1, E2, G12) |
| `rotating_orthotropic_disk.jl` | rotating disk + body force |
| `plate_hole_orthotropic.jl` | hole SCF (graphite/epoxy, FCT) |
| `plate_hole_tension_x.jl` | hole + remote σx (several composites) |
| `plate_multi_holes.jl` | multi-hole plate |
| `cavity_infinite_aniso.jl` | pressurized cavity (aniso) |
| `composite_materials.jl` | lamina library for `lekhnitskii_params` |

```julia
include(datadir("elastico", "aniso", "composite_materials.jl"))
include(datadir("elastico", "aniso", "plate_hole_orthotropic.jl"))
# p = lekhnitskii_params(graphite_epoxy.E1, graphite_epoxy.E2, graphite_epoxy.G12, graphite_epoxy.ν12)
```

Meshes written by helpers go to `datadir("elastico", "iso"|"aniso", name * ".msh")`.
