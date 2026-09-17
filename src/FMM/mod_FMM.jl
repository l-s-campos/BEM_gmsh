"""
    module FMM

Pure Julia fast multipole methods (2D & 3D). Vendored into BEM as `BEM.FMM`.

**Trees and hierarchical formats** come from the sibling [`BEM.HMatrices`](@ref)
module (loaded first by `BEM.jl`). FMM owns multipole expansions and kernel matvecs
only (`FMMKernelMatrix`, `KelvinFMMMatrix`, `KelvinFMMMatrix3D`, `lfmm2d`, …).

Layout:
```
FMM/
  core/        expansions, dual-tree engine, Laplace-2D plan
  kernels/     physics kernels (Laplace3D, Helmholtz, Stokes, …)
  operators/   AbstractMatrix wrappers (FMMKernelMatrix, Kelvin, …)
```
"""
module FMM

using LinearAlgebra
using StaticArrays
using Statistics
using Printf
using SpecialFunctions
using FastGaussQuadrature

using ..HMatrices:
    ClusterTree,
    GeometricSplitter,
    DyadicSplitter,
    CardinalitySplitter,
    HyperRectangle,
    leaves

using ..HMatrices:
    isleaf,
    isroot,
    children,
    parentnode,
    container,
    root_elements,
    index_range,
    elements,
    loc2glob,
    glob2loc,
    diameter,
    distance,
    center,
    high_corner,
    low_corner,
    radius,
    nodes,
    filter_tree,
    depth,
    bounding_box,
    compression_ratio,
    maxrank,
    node_id,
    nnodes,
    assign_node_ids!,
    FMMStrongAdmissibility,
    AbstractKernelMatrix,
    hmatrix_splitter,
    neighbor_il_lists,
    boxes_touch,
    nodes_by_depth

export FMMVals
export lfmm2d, rfmm2d, l2ddir, r2ddir
export Laplace2DFMMPlan, build_laplace2d_plan, apply_laplace2d!
export build_laplace3d_plan, Laplace3DFMMPlan, apply_laplace3d!
export hfmm2d, h2ddir
export cfmm2d, c2ddir
export stfmm2d, st2ddir
export lfmm3d, l3ddir
export hfmm3d, h3ddir
export stfmm3d, st3ddir
export yfmm2d, y2ddir, yfmm3d, y3ddir
export Body, evaluate!
export Laplace2D, Laplace3D, Helmholtz2D, Helmholtz3D, Yukawa2D, Yukawa3D
export kifmm3d, KILaplace3D, KIYukawa3D, KIHelmholtz3D
export FMMKernelMatrix, fmm_laplace3d_matrix, fmm_laplace2d_matrix, fmm_yukawa3d_matrix
export fmm_laplace2d_double_layer_matrix
export KelvinFMMMatrix, fmm_kelvin2d_matrix
export KelvinFMMMatrix3D, fmm_kelvin3d_matrix
export ClusterTree, GeometricSplitter, DyadicSplitter, CardinalitySplitter, HyperRectangle

# --- core ---
include("core/expansions.jl")
include("core/kernel_constants.jl")
include("core/fmm_core.jl")
include("core/dualtree.jl")

# --- physics kernels ---
include("kernels/cauchy2d/cauchy2d.jl")
include("kernels/helmholtz2d/helmholtz2d.jl")
include("kernels/stokes2d/stokes2d.jl")
include("kernels/laplace3d/laplace3d.jl")
include("kernels/helmholtz3d/helmholtz3d.jl")
include("kernels/stokes3d/stokes3d.jl")
include("kernels/yukawa2d/yukawa2d.jl")
include("kernels/yukawa3d/yukawa3d.jl")
include("kernels/body.jl")
include("kernels/kifmm/surface.jl")
include("kernels/kifmm/kernels.jl")
include("kernels/kifmm/kifmm.jl")

# --- operators ---
include("operators/fmm_kernel.jl")

end # module
