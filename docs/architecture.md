# Architecture review & improvement plan

Snapshot review of **BEM.jl** / `BEM_gmsh` for clarity, dead code, and next cleanups.

## What the package is

A Julia BEM toolkit: Gmsh meshes → collocation assembly → dense / H-matrix / FMM
solvers for Laplace, elasticity, cracks (DBEM + cohesive), plates, contact, RBF/DIBEM
domain terms, and modal / DIBEM variants.

## Mental model (target for newcomers)

```text
mesh (Gmsh)  →  format2d → BEMdata
                    ↓
              H_G_full_direct / H_G_Hmat
                    ↓
         optional: DIBEM / RBF particular
                    ↓
              applyBC → solve / solve_Houbolt / solve_mmm!
                    ↓
              dad.T, dad.q   (+ plot_geo, rel_error)
```

Keep this path **one screen** in the README. Everything else is an optional branch.

## Module map

| Path | Role | Clarity |
|------|------|---------|
| `Core/` | Mesh I/O, elements, integration, RBF, cache | Mixed — `Input.jl` is large |
| `Laplace/` | Assembly, BC, solvers, DIBEM, MMM, DRM | Growing — split OK but names overlap |
| `Hmat/`, `FMM/` | Vendored accelerators | Fine; treat as deps |
| `Crack/` | Dual BEM + cohesive | Dense but coherent |
| `Contact/`, `Plate/`, `MultiRegion/` | Specializations | OK as submodules |
| `data/` | Mesh builders + analytics | Should be pure *examples*, not API |
| `scripts/` | Demos | Good; some profile noise |
| `test/` | Many files, partial wiring to `runtests.jl` | Incomplete CI surface |

## Usability issues (high impact)

1. **Heavy default environment**  
   `Project.toml` pulls CUDA, GLMakie, WGLMakie, CairoMakie, Infiltrator, Revise,
   BenchmarkTools, Documenter into the *main* env.  
   **Fix:** move viz / CUDA / profiling to extensions or `docs/` / `benchmark/` envs;
   keep core deps: LinearAlgebra, StaticArrays, Gmsh, Krylov, DiffEq, SpecialFunctions.

2. **Too many public names at top level**  
   `@reexport` of DrWatson, Infiltrator, TimerOutputs, GLMakie, entire HMatrices/FMM
   floods `using BEM`.  
   **Fix:** export a small public API; use submodules (`BEM.Crack`, `BEM.HMatrices`)
   without reexporting everything.

3. **Naming inconsistency**  
   `DIBEM` vs `dibem_alt` vs `solve_mmm!` vs `solve_Houbolt`.  
   **Fix:** glossary in docs + stable aliases (`dibem!` = `DIBEM`, …).

4. **Mesh builders live in `data/`**  
   Users must `include(datadir(...))` — easy to miss.  
   **Fix:** `BEM.Examples` module or `ext` that loads mesh recipes.

5. **`BEMdata.cache` is a bag of symbols**  
   Powerful but hard to discover (`H`, `M`, `M_ID`, `modal_basis`, …).  
   **Fix:** document cache keys; optional typed `LaplaceCache` / `WaveCache`.

6. **Tests not all in `runtests.jl`**  
   New suites (`test_mmm`, `test_dibem_alt`, `test_poisson_drm`, cohesive, IGA)
   are easy to forget in CI.  
   **Fix:** `@testset` includes or a `test/Project.toml` + matrix in CI.

7. **Portuguese/English split is good** — keep; add a one-page “recipe book”
   (Laplace steady, wave Houbolt, MMM, DIBEM-alt C8E1, cohesive mode I).

## Unused / suspect code

| Item | Status | Action |
|------|--------|--------|
| `scripts/profile_bem_last.txt` | Artifact | gitignore / delete |
| `_research/` | Already gitignored | keep local |
| `src/Core/RBF_Extensions.jl` | **Used** via `Radial_Basis_Functions.jl` | keep |
| `src/Core/Parallel.jl` | Thin helpers | keep or fold into Assembly |
| `src/Core/SST_Leonel.jl` | Niche integration path | document or gate behind flag |
| `Infiltrator` / `Revise` in deps | Dev-only | remove from runtime deps |
| Orphan `*_snip.jl` / `Input_iga_tail` / `_patch_rbf.py` | **Removed** or absent in tree | good |
| ONELAB module (earlier work) | Not in current `src/` tree | re-add or drop from README claims |
| Duplicate demos (`cohesive_modeI_demo` vs `cohesive_gmsh_modeI`) | Overlap | merge docs links |
| Vendored full `Hmat/` + `FMM/` | Large but used | optional future: Julia packages |

Heuristic: if a file is only referenced by itself and one old script, mark
`# status: experimental` at the top.

## Suggested refactors (priority order)

### P0 — understandability (1–2 days)
- [ ] Slim README: 15-line quickstart + feature table + link to recipes
- [ ] `docs/src/recipes.md`: 5 copy-paste workflows
- [ ] Public API page listing *only* supported entry points
- [ ] Wire `test_mmm.jl`, `test_dibem_alt.jl`, cohesive, IGA into CI
- [ ] `.gitignore`: `*.txt` profiles, `*.msh` regenerable noise if desired

### P1 — package hygiene (3–5 days)
- [ ] Split `Project.toml`: `[weakdeps]` / extensions for Makie & CUDA
- [ ] Stop `@reexport using Infiltrator, TimerOutputs, GLMakie`
- [ ] `BEM.Examples` for `quadrado`, wave meshes, C8E1
- [ ] Consistent bang naming: mutating assembly `H_G_full_direct!`

### P2 — structure (1–2 weeks)
- [ ] Split `Input.jl` (Gmsh parse / IGA / BC tags)
- [ ] Split `CohesiveDBEM.jl` (pairs / residual / drivers)
- [ ] Typed cache or `solve` options struct instead of ad-hoc kwargs
- [ ] Single `src/BEM.jl` submodule tree with fewer top-level includes

### P3 — science quality
- [ ] Benchmark notebook vs Pinheiro/Áquila thesis tables
- [ ] Classical DIBEM variable-velocity (Ch.7) next to alternative (Ch.8)
- [ ] Flux sign convention documented once (`q = -k ∂u/∂n`)

## “Easy to understand” checklist

A new user should, in **&lt; 30 minutes**:

1. `Pkg.instantiate()` without GPU/Makie pain  
2. Run square Laplace `T=x` and plot  
3. Find “how do I add a domain source?” → DIBEM / particular  
4. Find “variable velocity advection?” → DIBEM alt  
5. Find “crack with cohesion?” → `scripts/cohesive_gmsh_modeI.jl`  

If any step needs reading 3 source files, the API failed.

## Non-goals (for now)

- Rewriting Hmat/FMM  
- Full 3D elasticity production polish  
- Replacing Gmsh
