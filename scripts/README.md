# Scripts

Runnable demos, paper figures, and probes. Nothing here is deleted; it is
grouped by topic. Activate the package from the **repo root**:

```bash
julia --project=. scripts/<folder>/<script>.jl
```

| Folder | What lives here |
|--------|-----------------|
| `intro/` | First demos (`intro.jl`, fundamentals, potencial direto) |
| `laplace/` | Poisson, Galerkin, local BEM (2D + 3D cube), diffuse–advective, Helmholtz/BM, 3D cube + Poisson DIBEM, two regions (dense / Kane noncondensing / condensation), GPU vs CPU assembly (`gpu_vs_cpu_assembly.jl`), Guiggiani lubrication (`guiggiani_lubrication.jl`), Profito cavitation (`profito_cavitation.jl`) |
| `elasticity/` | Bar / beam elastodynamics, 3D cube strain-patch and body-force DIBEM, constant-cell plasticity (`thick_cylinder_plastic.jl`, Escudero 2018 `escudero2018_plastic.jl`), GPU vs CPU (`gpu_vs_cpu_assembly.jl`) |
| `dibem/` | DIBEM residuals, cells, RBF compares, GPU vs CPU (`gpu_vs_cpu_dibem.jl`) |
| `transient/` | Wave, Houbolt/MMM, dt sweeps, Ricker wavelet (`ricker_wavelet_dibem.jl`, anisotropic `ricker_anisotropic_dibem.jl`) |
| `sbm_drm/` | SBM, SBM–DRM, Kansa vs DRM (includes paper figure drivers that share the study file) |
| `meshless/` | Macchiato / RBF-FD compare — **own** `Project.toml` |
| `crack/` | Dual BEM, cohesive mode I, slit sweeps, Erdogan–Sih / Ke MTS, CSTBD |
| `contact/` | Hertz, Cattaneo, mortar, active-set, projected Newton, GNM-ls 2010 (`compare_rt2010.jl`) |
| `julia_lerma/` | Pohrt–Li / Bagault / Juliá Lerma Ch. 3 (pin-on-disc, fretting, rolling, wheel–rail) |
| `plates/` | Kirchhoff / FSDT / laminated shells (Useche Ch.7–9), Reissner Dual BEM (10.5.1), Hsu–Hwu CBIE vs HBIE (8.3.1), Houbolt DIBEM, large deflection |
| `topology/` | Pacheco / DT stand-in vs quantile / level-set / DIBEM-SIMP (2-D + 3-D density) / Portela dual-BEM shape |
| `hmat_fmm/` | H-matrix / NNCA / FMM demos; `cost_order.jl` (dense / H / FMM assembly+matvec scaling, 2D+3D); `matvec_compare.jl` (dense / H / H2 / FMM matvec to N~1e5); `compare_hss_ulv.jl` (HSS+ULV vs nested H² LU vs `rskelf`); `gpu_vs_cpu_hmat.jl` |
| `papers/` | Thesis/paper figure scripts (`plot_*`, `thesis_*`, `*_paper.jl`) |
| `debug/` | `_chk_*`, `_dbg_*`, tmp probes — not CI |
| `profile/` | Timing / allocation (`profile_hss.jl`: HSS+ULV vs nested H² LU vs `rskelf`) |

Meshless compare:

```bash
julia --project=scripts/meshless -e 'using Pkg; Pkg.instantiate()'
julia --project=scripts/meshless scripts/meshless/macchiato_heat_compare.jl
```
