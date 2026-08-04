# Crack analysis (dual BEM + propagation)

Single module `BEM.Crack`, based on dual-BEM codes of Marcel Sato /
Éder Albuquerque (UNICAMP) (`orientação/.../trinca/propagation`).

## Dual BEM

Gmsh mesh + `format2d` with **BC type 5** on crack faces:

| Physical name | Role |
|---------------|------|
| `"5;2;5;2"` | crack face A — displacement BIE |
| `"5;3;5;3"` | crack face B — traction BIE |
| outer BCs | standard elasticity `"tx;vx;ty;vy"` |

| Region | Equation | `eq_type` |
|--------|----------|-----------|
| Outer boundary | displacement BIE + rigid-body free term | 1 |
| Crack face A | displacement BIE (no RBM) | 2 |
| Crack face B | traction (hypersingular) BIE | 3 |

```julia
using DrWatson
@quickactivate :BEM
using .Crack
include(datadir("elastico", "iso", "center_crack.jl"))

msh = mesh_center_crack(; W=5, H=10, a=1, σ=1)
dad = format2d(msh, Elasticity(3000, 0.2, 1.0); tipo=2, pontointerno=false)
mesh = dual_mesh_from_bemdata(dad)
assemble_dual!(mesh); solve_dual!(mesh)
KI, KII = sif_cod_dual(mesh, mesh.tip_nodes[1])
```

Or one-shot:

```julia
mesh = build_center_crack_mesh(; W=5, H=10, a=1, σ=1, n_crack=12)
assemble_dual!(mesh); solve_dual!(mesh)
```

## Propagation

| Piece | API |
|-------|-----|
| Max. circumferential stress | `max_tens_circ` |
| Strain energy density | `strain_energy_density_angle` |
| Paris fatigue | `paris_cycles`, `tanaka_deltaK` |
| Geometry update | `extend_crack_tip!`, `propagate!` |
| Benchmark | `analytical_KI_center_crack` |

```julia
sifs = [sif_cod_dual(mesh, t) for t in mesh.tip_nodes]
θs, ΔN = propagate!(prob, sifs; da=0.25, criterion=:MTS)
```

## Demo

```bash
julia --project=. scripts/crack_central.jl
```
