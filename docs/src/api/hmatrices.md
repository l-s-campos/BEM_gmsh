# H-matrices

Vendored hierarchical matrices. Load with `using BEM.HMatrices`.
`assemble!(dad; method=:hmatrix)` stores a `ColWeightedOp` wrapping
an `HMatrix` in `dad.H`.

Tensor kernels (`SMatrix{p,p}`, e.g. Kelvin): assemble
`assemble_hmatrix(K, tree, tree)` on the **point** tree (entries stay
`SMatrix`). `assemble_h2(K, tree)` is block NNCA — point skeletons, apply
size `(p n)×(p n)`. `assemble_h2(K, rowtree, coltree)` is rectangular
(`n_row × n_col`); share a root `container` for equal-size boxes.
`lu(A::NNCAMatrix)` is nested H² LR (H2Lib `lrdecomp_h2matrix`) via
`h2node(A)` — couplings stay nested. `method=:hmatrix` expands to H-LU.
Matvecs take `Vector{SVector{p}}` or a flat length-`p n` vector.

GPU apply (assembly stays on the host):

```julia
using CUDA
H = assemble_hmatrix(K, tree, tree; device=:cuda)   # or gpu(H; device=:cuda)
A = assemble_h2(K, tree; device=:cuda)
y = H * x                 # host x: permute on CPU, copy, apply, copy back
yd = H * CuArray(x)       # whole matvec on the GPU (permute + kernels)
mul!(yd, H, xd)           # keep xd/yd as CuArray across GMRES iterations
# each leaf GEMV is one thread per output row (coloring still serializes
# H-matrix leaves that share a row cluster)
```

`device=:cpu` runs the same kernels on the KernelAbstractions CPU backend.

[`ilut`](@ref) is Saad ILUT, including on the H² near field via [`near_sparse`](@ref).

[`assemble_hss`](@ref) builds a weak nested HSS matrix on a **binary**
[`PrincipalComponentSplitter`](@ref) tree by FLAM-style nested ID
(`method=:id`) or partial ACA (`method=:aca`) against neighbors + proxy
(not SVD of `A(t, t^c)`). `assemble_hss(K, rowtree, coltree)` is rectangular
(matvec only). [`ulv`](@ref) is the Chandrasekaran–Gu–Pals ULV inverse and
requires square HSS from a single tree.

Entry-based skeletonization: [`rskelf`](@ref) is weak recursive skeletonization
(proxy far field). It returns [`RSKELFFactor`](@ref) with `mul!` / `ldiv!`
(GMRES `Pl`). Use [`circle_proxy`](@ref) in 2D and [`sphere_proxy`](@ref) in 3D.

```@docs
BEM.HMatrices
BEM.HMatrices.HMatrix
BEM.HMatrices.NNCAMatrix
BEM.HMatrices.ClusterTree
BEM.HMatrices.HyperRectangle
BEM.HMatrices.RkMatrix
BEM.HMatrices.PartialACA
BEM.HMatrices.TSVD
BEM.HMatrices.assemble_hmatrix
BEM.HMatrices.assemble_h2
BEM.HMatrices.ilut
BEM.HMatrices.ILUTFactor
BEM.HMatrices.near_sparse
BEM.HMatrices.rskelf
BEM.HMatrices.assemble_hss
BEM.HMatrices.HSSMatrix
BEM.HMatrices.ulv
BEM.HMatrices.ULVFactor
BEM.HMatrices.RSKELFFactor
BEM.HMatrices.circle_proxy
BEM.HMatrices.sphere_proxy
BEM.HMatrices.interpolative_decomp
BEM.HMatrices.gpu
BEM.HMatrices.GPUHMatrix
BEM.HMatrices.GPUNNCAMatrix
BEM.HMatrices.compression_ratio
```
