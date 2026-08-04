# BEM.FMM

Vendored pure-Julia FMM (from `D:/fmm/FMM2D`), loaded as submodule `BEM.FMM`.

## Entry point

On Windows, the module file is **`mod_FMM.jl`** (not `FMM.jl`) because the
filesystem is case-insensitive and would collide with `fmm_core.jl`.

```julia
# from BEM.jl
include("FMM/mod_FMM.jl")
@reexport using .FMM
```

## Sync from upstream

```bash
# from repo root
rm -rf src/FMM
mkdir -p src/FMM
cp -r /path/to/fmm/FMM2D/src/* src/FMM/
mv src/FMM/fmm.jl src/FMM/fmm_core.jl
# restore mod_FMM.jl entry (see git)
```

## Contact half-space

`HalfSpaceBEM.build_fmm` uses `FMM.rfmm2d` (2D) / `FMM.lfmm3d` (3D) with
exact near-field panel corrections.
