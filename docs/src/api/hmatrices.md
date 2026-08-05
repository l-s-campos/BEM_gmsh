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
H  = assemble_hmatrix(K, tree, tree; adm=StrongAdmissibilityStd(2),
                      comp=PartialACA(; rtol=1e-6))
H2 = assemble_h2(K, tree; rtol=1e-6, far_method=:aca, alpha=0.5)
```

## Algebra

```julia
y = H * x
Y = H * X   # multi-RHS

# structured product (classic H)
hmul!(C, A, B, 1, 0, PartialACA(; rtol=1e-6))

# hierarchical low-rank update: H ← H + X*Y'
hlru!(H, X, Y; rtol=1e-6)

# compatible-tree add: C ← A + B
hadd!(C, A, B, 1, 1; rtol=1e-6)

# factors
F = lu(H, PartialACA(; rtol=1e-6))
ldiv!(F, b)
```

## HARA (sampler build)

Build a classic `HMatrix` from **matvecs only** (black-box operator):

```julia
S = KernelMatvecSampler(K)   # or FunctionSampler(f!, n; f_adj!)
Hh = hara(S, tree, tree; rtol=1e-3, batch=8)
```

Useful for products `v ↦ A(B*v)` without forming `C = A*B`.

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
