# Plan: improve `HMatrices` (BEM)

**Location:** `src/Hmat/`  
**Reference:** H2Opus (`D:\h2opus`) capability gap analysis  
**Date:** 2026-08-05  
**Status:** **in progress** — Slice 1 + Slice 2 MVP landed (2026-08-05)  

### Progress log

| Date | Item | Notes |
|------|------|--------|
| 2026-08-05 | **A3/A4 tests** | `test/test_hmat_algebra.jl` — matvec, multi-RHS, wired in `runtests.jl` |
| 2026-08-05 | **B2 `hadd!`** | Compatible-tree structured add + TSVD recompress (`hlru.jl`) |
| 2026-08-05 | **B3 `hlru!`** | `H ← H + XY'` on classic `HMatrix` leaves |
| 2026-08-05 | **C1 `h2_orthog!` / C5 `h2_compress!`** | Nested QR + far project; far-block recompress MVP (`h2_basis.jl`) |
| 2026-08-05 | **C3 HARA MVP** | `AbstractMatvecSampler`, `FunctionSampler`, `KernelMatvecSampler`, `hara` → classic H (`hara.jl`) |
| 2026-08-05 | **Docs** | `docs/src/api/hmatrices.md` |
| 2026-08-05 | **Kernel `mul!`** | Multi-RHS for `KernelMatrix` (HARA speed) |
| 2026-08-05 | **A6 multi-RHS H/H²** | Blocked leaf GEMM + H² `_h2_matvec_multi` level sweeps |
| 2026-08-05 | **A4/B factors** | `test/test_hmat_factor.jl` — H LU residual vs dense |
| 2026-08-05 | **C4 HARA product** | `scripts/hara_product_demo.jl` + test `A(Bv)` |
| 2026-08-05 | **B4/B5 precond** | `cholesky(H; ridge=)`, `gmres_h`, `hara_product`, `solve_Hmat(; Pl=)` |
| 2026-08-05 | **HARA via H apply** | `hara_product(A,B,trees)` — no dense `A*B` (large-n path) |
| 2026-08-05 | **C3 nested-H² HARA** | `hara_h2` / `hara(; format=:H2)` — matvec nested bases + skeleton couplings |
| 2026-08-05 | **FMM → H²** | `assemble_h2_fmm` / `DIBEM(...; method=:h2, hss_method=:fmm)` |
| 2026-08-05 | **H2Lib `lrdecomp_h2matrix`** | `h2_to_hmatrix`, `lrdecomp_h2matrix`, `lu(::H2Matrix)` → `H2LU`, `choldecomp_h2matrix` (`h2_factor.jl`) — MVP: H²→H then H-LU |

**Still open:** nested H² LR with local low-rank updates (full Börm–Reimer / H2Lib `addmul_h2matrix`), A2 unified compress API, B1 hmul policy docs, Phase D BEM end-to-end Laplace+Pl example, Phase E–F (GPU/MPI), HARA workspace reuse.

**Verify:**
```bash
julia --project=. -e 'using Test; include("test/test_hmat_algebra.jl"); include("test/test_hmat_factor.jl"); include("test/test_hmat_precond.jl")'
julia --project=. scripts/hara_product_demo.jl
julia --project=. scripts/hmat_gmres_precond_demo.jl
```
Demo (n=100): unprecond GMRES ~14 iters → LU/Chol precond **1 iter**.

---

## Goal

Make hierarchical matrices a **reliable BEM backend** (assemble → matvec → precond/solve), then close the **H² algebra** gap vs H2Opus — without a GPU rewrite first.

**Guiding rule:** BEM-first. Prefer work that speeds Laplace/elasticity `H`/`G`, DIBEM, half-space contact, and GMRES. Defer multi-GPU/MPI until CPU algebra is solid.

---

## 0. Current baseline

| Strengths | Weak spots |
|-----------|------------|
| Formats: H, BLR, HODLR, HSS/HBS, H² | H² = build + matvec; little post-build algebra |
| ACA / TSVD / recompress in `hmul!` | No HARA (matvec-only build) |
| H LU/Chol, BLR LU | No nested orthog/compress; no HLRU |
| Wired into `Assembly_H`, DIBEM, half-space | Tests uneven (H² ACA yes; `hmul`/LU thin) |
| AnchorNet, scalarize | Multi-RHS matvec not BLAS-3 optimized |

**Do not:** clone H2Opus C++/CUDA.  
**Do:** steal its *API ideas* (sampler, horthog, hcompress, hlru) in Julia on existing types.

### H2Opus vs BEM (summary)

| Capability | H2Opus | BEM `Hmat` | Priority gap |
|------------|--------|------------|--------------|
| H / ACA assembly | secondary | strong | — |
| H² nested build | core | yes (proxy + ACA far) | partial |
| HSS / HODLR / BLR | TLR focus | yes | BEM ahead |
| Matvec | batched GPU | CPU threads | perf later |
| **HARA** (matvec sampler → H²) | yes | **no** | **P1** |
| **Horthog / Hcompress** | yes | **no** | **P1** |
| **HLRU** | yes | **no** | **P2** |
| H × H (`hmul`) | via HARA | yes (classic H) | OK |
| H LU / Chol | TLR SPD | yes H/BLR | polish |
| GPU / MPI | yes | no | P3 / later |

---

## 1. Principles

1. **One public algebra surface** on `AbstractStructuredMatrix` where possible: `mul!`, `ldiv!`, `add!`, `compress!`.
2. **Tolerances everywhere**: `rtol` / `atol` / `rank` on arithmetic, not only assembly.
3. **Every feature ships with a test** (small kernel + error vs dense).
4. **Git commits per phase** (easy revert).
5. **Measure**: compression ratio, matvec time, GMRES iters on a fixed BEM fixture.
6. **LinearSolve** only on main BEM solvers — not a bulk replace of internal `A\b` in H-algebra kernels.

---

## 2. Phased roadmap

### Phase A — Foundation & quality (1–2 weeks)

**Why first:** algebra on a shaky base wastes time.

| Task | Detail | Files | Done |
|------|--------|--------|------|
| A1. Inventory & docs | One `docs/src/api/hmatrices.md` | `docs/src/api/hmatrices.md` | partial |
| A2. Unified compress API | extend `compress!` to H leaves / H² | `compressor.jl` | open |
| A3. Matvec correctness suite | Dense vs H, multi-RHS | `test/test_hmat_algebra.jl` | **yes** |
| A4. Algebra smoke tests | hlru / hadd / hara / h2 | same | **yes** (hmul/LU still thin) |
| A5. Diagnostics | `compression_ratio`, `maxrank` | existing | partial |
| A6. Multi-RHS `mul!` | Blocked leaf GEMM H + H² multi | `multiplication.jl`, `h2matrix.jl` | **yes** |

**Exit criteria:** CI tests green; known relative matvec error e.g. `< 10× rtol` on fixtures. *(matvec tests green)*

---

### Phase B — Classic H algebra polish (2–3 weeks)

**Why:** this is what BEM already uses (`assemble_hmatrix`, LU precond).

| Task | Detail | Done |
|------|--------|------|
| B1. Stable `hmul!` recompression | Default compressor policy docs | open |
| B2. `hadd!(C, A, B, α, β)` | Structured add + TSVD | **yes** |
| B3. Low-rank update on H | `hlru!(H, X, Y; rtol)` | **yes** |
| B4. Factor robustness | LU + Chol + ridge tests | **yes** |
| B5. BEM precond path | `gmres_h` + `solve_Hmat(; Pl=)` | **yes** |
| B6. Buffer reuse | alloc audit | open |

**Exit criteria:** Laplace H-mat GMRES with H-LU or H-Chol precond beats unpreconditioned baseline on medium mesh; `hlru!` error test passes. *(`hlru!` test passes; precond path open)*

**BEM payoff:** faster/robust compressed precond; cheap operator tweaks.

---

### Phase C — H² as a real format (3–4 weeks) ★ main upgrade

**Why:** closes the important H2Opus gap without GPU.

#### C1. Nested basis tools — **done (MVP)**

```julia
h2_orthog!(H2)                      # QR nested U; project Bfar
h2_compress!(H2; rtol, atol, rank)  # recompress far blocks
```

Implemented in `src/Hmat/h2_basis.jl`. Compress MVP focuses on far blocks (safe nesting).

#### C2. Levelized H² matvec — **done (multi-RHS)**

`_h2_matvec_multi` shares upsweep / far / downsweep / near for `n×s` RHS.

#### C3. Sampler interface + HARA — **done (classic H + nested H²)**

```julia
AbstractMatvecSampler
FunctionSampler / KernelMatvecSampler
hara(S, rowtree, coltree; ...) -> HMatrix     # classic
hara_h2(S, tree; nsample, alpha, ...) -> H2Matrix
hara(S, tree; format=:H2) -> H2Matrix
```

Files: `hara.jl`, `hara_h2.jl`. Nested path: one multi-RHS `Y=A*Ω`, row-ID bases bottom-up, skeleton far blocks by identity sampling, then `h2_orthog!`/`h2_compress!`.

#### C4. Product without `hmul` — **done (demo)**

```julia
sampler = FunctionSampler((Y,X) -> mul!(Y, A, B*X), n; f_adj! = ...)
H = hara(sampler, tree, tree; rtol=…)
```

Script: `scripts/hara_product_demo.jl`.

#### C5. Tests — **done for MVP**

- HARA vs dense matvec — yes  
- h2_orthog preserves matvec — yes  
- Product `A*B` via HARA — yes  
- H LU + multi-RHS — yes  

**Exit criteria:** HARA matvec error controlled by `rtol` *(classic H: yes)*; product demo green.

**BEM payoff:** black-box recompression path ready; wire into FMM/product next.

---

### Phase D — BEM integration (parallel with end of C / after)

| Task | Detail |
|------|--------|
| D1. Single entry | `assemble_structured(K, tree; format=:H\|:H2\|:HSS\|:BLR, …)` — BEM always goes through it |
| D2. Laplace/Elasticity | Document recommended format per problem size; default `rtol` |
| D3. Factored DIBEM | Ensure `dibem_D` H²/HSS path + GMRES stays stable after C |
| D4. Half-space contact | Optional HARA rebuild if operator is FFT/FMM apply-only |
| D5. Precond in main solve | Hierarchical factors only on main solvers (`bem_linsolve` / `Solver.jl`) |

**Exit criteria:** one end-to-end Laplace and one elasticity compressed solve in tests/scripts with reported speedup vs dense.

---

### Phase E — Performance (after correctness)

| Task | Notes |
|------|--------|
| E1. Profile matvec / `hmul` / HARA | Allocs, threads |
| E2. Better multi-RHS | Block ACA samples; H² batch upsweep |
| E3. Nearfield BLAS | Contiguous leaf storage where possible |
| E4. Optional TLR Chol | Only if SPD precond needs it (Laplace-like) |
| E5. GPU | **Out of scope** until E1–E3 done; then CUDA matvec only, not full H2Opus port |

---

### Phase F — Optional / later

- Distributed `DHMatrix` matvec completion  
- PETSc-like external bindings — skip  
- Full strong-admissibility HARA with H2Opus weight packets — only if MVP is insufficient  
- Automatic format picker (`n`, kernel smoothness → H vs H² vs HSS)

---

## 3. Suggested API targets (end state)

```julia
# Assembly (existing + polish)
H  = assemble_hmatrix(K, clt, clt; adm, comp)
H2 = assemble_h2(K, clt; rtol, far_method=:aca)

# Sampler / HARA (new)
S  = KernelMatvecSampler(K)          # or FMMSampler(fmm)
H2 = hara(S, clt; adm, rtol=1e-4, batch=16)
h2_orthog!(H2)
h2_compress!(H2; rtol=1e-4)

# Algebra
mul!(y, H, x)
mul!(Y, H, X)                        # multi-RHS
hmul!(C, A, B, 1, 0, comp)
hadd!(C, A, B, 1, 1, comp)
hlru!(H, X, Y; rtol=1e-4)

# Factors / solve
F = lu(H, comp)          # or cholesky
ldiv!(F, b)
```

---

## 4. Priority order (if time is short)

```text
1. Phase A tests + multi-RHS matvec     # safety net
2. Phase B hlru! + H precond path       # immediate BEM value
3. Phase C h2_orthog! / h2_compress!    # unlock H² life-cycle
4. Phase C HARA MVP                     # strategic vs H2Opus
5. Phase D wire into one BEM solver
6. Phase E profile
```

Skip until needed: GPU, MPI, full TLR, distributed.

---

## 5. Test matrix (minimum)

| Test | Checks |
|------|--------|
| `test/test_hmat_matvec.jl` | formats × kernel × rtol |
| `test/test_hmat_algebra.jl` | hadd, hmul, hlru vs dense |
| `test/test_hmat_factor.jl` | LU/Chol solve residual |
| `test/test_hara.jl` | sampler build vs dense / vs `assemble_h2` |
| `test/test_h2_basis.jl` | orthog/compress rank ↓, matvec error held |
| Existing | `test_h2_aca_far`, HSS/DIBEM — keep green |

**Fixture:** random points in a box + `1/|x−y|` or Laplace fundamental; `n ∈ {200, 800}`.

---

## 6. Success metrics

| Metric | Target (indicative) |
|--------|---------------------|
| Matvec error | O(rtol) vs dense |
| HARA error | ≤ few × `rtol` |
| Compression | clearly `< 1` storage ratio on smooth kernels |
| GMRES | fewer iters or less time with H-factor precond |
| Regressions | no break of DIBEM H²/HSS or half-space H path |

---

## 7. Risks & mitigations

| Risk | Mitigation |
|------|------------|
| HARA hard to stabilize | Start box/weak adm; abs + rel residual; cap rank |
| Rank explosion in `hmul` | Aggressive recompress; tests on rank growth |
| H² layout mismatch with orthog | Sketch data flow before coding; small `n` first |
| Scope creep (GPU) | Explicit Phase E/F freeze |
| Breaking BEM assembly | Feature flags; commit per phase; keep old `assemble_h2` |

---

## 8. First implementation slices

### Slice 1 (small, high value) — **done 2026-08-05**

1. `test/test_hmat_algebra.jl` for H matvec / multi-RHS  
2. `hlru!(H::HMatrix, X, Y; rtol, atol, rank)` + `hadd!`  
3. `docs/src/api/hmatrices.md`  

### Slice 2 — **done 2026-08-05 (MVP)**

4. `h2_orthog!` / `h2_compress!` in `h2_basis.jl`  
5. `AbstractMatvecSampler` + `hara` MVP (classic H) in `hara.jl`  

### Slice 3 — **done 2026-08-05**

6. `test_hmat_factor.jl` — LU residual + multi-RHS + HARA product  
7. HARA product demo `scripts/hara_product_demo.jl`  
8. Multi-RHS H / H² matvec (blocked)  

### Slice 4 — **done 2026-08-05**

9. `gmres_h` + `solve_Hmat(; Pl=)` + `scripts/hmat_gmres_precond_demo.jl`  
10. `cholesky(H; ridge=)` + `add_diag_ridge!` + tests  
11. `hara_product(A,B,trees)` — HARA from hierarchical applies  

### Slice 5 — nested H² HARA **done 2026-08-05**

12. `hara_h2` + tests `test/test_hara_h2.jl`  

### Slice 6 (next)

13. Profile / reduce allocs in HARA sampling buffers  
14. Laplace H end-to-end GMRES+`Pl=lu` script on Gmsh mesh  
15. Phase E light profiling of multi-RHS H gemv  
16. Optional: far-only / level-wise sampling to improve `hara_h2` accuracy  

---

## 9. Out of scope (explicit)

- Porting H2Opus CUDA/KBLAS  
- Full PETSc integration  
- Replacing FMM with HARA everywhere  
- Perfect bit-compatibility with H2Opus  

---

## 10. Background notes

### HARA (one line)

Randomized SVD, done hierarchically on the block cluster tree, using only matvecs, producing a nested-basis H² matrix with adaptive ranks.

### H-matrix algebra (one line)

Recursive block linear algebra with low-rank leaves and mandatory recompression; matvec is easy; multiplication and factorization are the heavy structured analogues of GEMM and LU; H² adds nested-basis bookkeeping (orthog/compress).

### Key source files today

| File | Role |
|------|------|
| `src/Hmat/HMatrices.jl` | module entry / exports |
| `src/Hmat/hmatrix.jl` | classic H |
| `src/Hmat/h2matrix.jl` | H² assemble + matvec |
| `src/Hmat/multiplication.jl` | `hmul!`, `mul!` |
| `src/Hmat/compressor.jl` | ACA, TSVD, recompress |
| `src/Hmat/lu.jl`, `cholesky.jl` | factors |
| `src/Hmat/blr.jl`, `hodlr.jl`, `hss.jl` | other formats |
| `src/Laplace/Assembly_H.jl` | BEM H/G assembly |
| `src/Core/DIBEM_common.jl` | format dispatch incl. H² |

### Related commits / context

- Contact / multibody work is separate; Hmat improvements should not break half-space `HMatrix` backends.  
- Prefer incremental commits: `feat(hmat): …` with revert-friendly messages.

---

## Bottom line

Improve `HMatrices` in three layers:

1. **Harden** what exists (tests, multi-RHS, H precond, `hlru!`)  
2. **Complete H² lifecycle** (orthog + compress)  
3. **Add HARA** so operators defined only by matvecs become H²  

That matches BEM needs and the important H2Opus gaps, without boiling the ocean.
