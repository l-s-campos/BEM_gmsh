# Receitas

Os identificadores de código permanecem em inglês. Ative o projeto:

```julia
using Pkg; Pkg.activate("."); Pkg.instantiate()
using BEM
```

## Laplace estacionário

```julia
dad = format2d(quadrado(ndiv=20, show=false), Laplace(1.0))
attach_analytical!(dad, ana_laplace_linear(; direction=SA[1.0, 0.0]))
assemble!(dad, 20)
solve(dad)
@show rel_error(dad)
```

## Elasticidade (patch)

```julia
dad = format2d(quadrado_elasticity(ndiv=12, show=false), Elasticity(1.0, 0.3, 1.0))
apply_analytical_bc!(dad, ana_elasticity_patch(; E=1.0, ν=0.3, εxx=0.01))
assemble!(dad, 12)
solve(dad)
```

## DIBEM + Houbolt

```julia
dad = format2d(quadrado(ndiv=16, show=false), Laplace(1.0); pontointerno=true)
assemble!(dad, 12)
dibem!(dad)
solve_Houbolt(dad, 0.01, 1.0)
```

## Trinca dual

```julia
using BEM.Crack
dad = build_center_crack_mesh(; W=5.0, H=10.0, a=1.0, σ=1.0, E=3000.0, ν=0.2)
assemble_dual!(dad; npg=8, threaded=false)
solve_dual!(dad; threaded=false)
```

Placa Reissner isotrópica (Useche 10.5.1): `build_rect_fsdt_crack` +
`assemble_fsdt_dual!` + `sif_ctod_fsdt` (`FSDTProps` apenas).

Demais famílias: versão em inglês em [Recipes](../recipes.md) (mesmos trechos).
Scripts: `scripts/README.md`.
