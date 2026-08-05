# Hierarchical matrices (`HMatrices`)

Module: `BEM.HMatrices` (reexported by `BEM`).

Rank-structured formats for BEM operators (`H`, `G`, DIBEM masses, half-space
kernels). See also the research plan
[`_research/hmatrices_improvement_plan.md`](../../_research/hmatrices_improvement_plan.md).

## Formats

| Type | Assemble | Typical use |
|------|----------|-------------|
| `HMatrix` | `assemble_hmatrix` | General BEM `H`/`G`, ACA far blocks |
| `H2Matrix` | `assemble_h2` | Nested bases, large smooth kernels |
| `HSSMatrix` / `HBSMatrix` | `assemble_hss` / `assemble_hbs` | DIBEM / 1D-like clusters |
| `HODLRMatrix` | `assemble_hodlr` | Weak admissibility |
| `BLRMatrix` | `assemble_blr` | Flat tiles + LU |

## Assembly

```julia
pts = # Vector{SVector}
K = KernelMatrix((x,y) -> ..., pts, pts)
tree = ClusterTree(pts, PrincipalComponentSplitter(; nmax=32))
H  = assemble_hmatrix(K, tree, tree; adm=StrongAdmissibilityStd(; eta=2),
                      comp=PartialACA(; rtol=1e-6))
H2 = assemble_h2(K, tree; rtol=1e-6, far_method=:aca, alpha=0.5)

# recursive H2Lib-style block tree (sons / uniform / dense)
root = h2_repackage(H2)   # -> H2Node with pack.U nested bases
y2 = root * x             # same matvec as flat H2 (verification)
```

## Algebra

```julia
y = H * x
Y = H * X   # multi-RHS (blocked leaf GEMM / H² level sweeps)

# structured product (classic H)
hmul!(C, A, B, 1, 0, PartialACA(; rtol=1e-6))

# hierarchical low-rank update: H ← H + X*Y'
hlru!(H, X, Y; rtol=1e-6)

# compatible-tree add: C ← A + B
hadd!(C, A, B, 1, 1; rtol=1e-6)

# factors (classic H)
F = lu(H; rtol=1e-6)
x = F \ b
Fc = cholesky(H; ridge=1e-10, rtol=1e-6)

# H² factorization (H2Lib lrdecomp_h2matrix)
# nested LR on recursive H2Node (true H2Lib recursion on sons):
Fn = lu(H2; method=:nested)            # -> H2NodeLU
x  = Fn \ b
out = lrdecomp_h2matrix(H2; method=:nested)
# practical path H²→H then H-LU:
F2 = lu(H2; method=:block, rtol=1e-4)  # -> H2LU
Hh = h2_to_hmatrix(H2; method=:block)

# GMRES (+ optional hierarchical left precond)
x, stats = gmres_h(H, b; Pl=F, rtol=1e-8)
```

`solve_Hmat(dad; Pl=F)` accepts the same optional `Pl` for Laplace H-systems.

## HARA (sampler build)

Build hierarchical matrices from **matvecs only** (black-box operator):

```julia
S = KernelMatvecSampler(K)

# classic H (blockwise low-rank leaves)
Hh = hara(S, tree, tree; rtol=1e-3, batch=8)

# nested H² (no proxies / no kernel entries)
H2h = hara_h2(S, tree; rtol=1e-4, nsample=64, alpha=0.5)
# or: hara(S, tree; format=:H2, rtol=1e-4)

# product C ≈ A*B from hierarchical applies only (no dense A*B)
Hc = hara_product(A, B, tree, tree; rtol=1e-3, batch=8)
# nested H² of a product sampler:
H2c = hara_h2(FunctionSampler((Y,X)->mul!(Y,A,B*X), n; f_adj! = ...), tree)
```

| API | Output | Needs |
|-----|--------|--------|
| `hara(S, rowtree, coltree)` | `HMatrix` | matvecs |
| `hara_h2(S, tree)` | `H2Matrix` | matvecs |
| `assemble_h2(K, tree)` | `H2Matrix` | entries / proxies |
| `assemble_h2_fmm(A, tree)` / `assemble_h2_fmm(points; kernel=…)` | `H2Matrix` | **FMM matvecs** (HARA) |

```julia
# FMM → nested H² (large-n path)
A  = FMM.fmm_laplace2d_matrix(Pmat; eps=1e-6)
H2 = assemble_h2_fmm(A, tree; rtol=1e-4, nsample=64)
# or one-shot:
H2 = assemble_h2_fmm(Pmat; kernel=:laplace2d, scale=-1/(2π))
# DIBEM:
DIBEM(dad; method=:h2, hss_method=:fmm)
```

Demos: `scripts/hara_product_demo.jl`, `scripts/hmat_gmres_precond_demo.jl`.

## H² basis maintenance

```julia
h2_orthog!(H2)                 # nested QR + project couplings
h2_compress!(H2; rtol=1e-6)    # recompress far blocks
```

## Diagnostics

```julia
compression_ratio(H)   # dense_bytes / hierarchical_bytes (>1 is good)
maxrank(H2)
```
