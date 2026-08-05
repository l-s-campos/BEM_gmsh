"""
    module FMM

Pure Julia fast multipole methods (2D & 3D). Vendored into BEM as `BEM.FMM`.

**Trees and HSS formats** come from the sibling [`HMatrices`](@ref) module
(loaded first by `BEM.jl`). FMM owns multipole expansions and kernel matvecs
only (`FMMKernelMatrix`, `KelvinFMMMatrix`, `lfmm2d`, …).

HSS / H² from FMM matvecs:
```julia
H  = FMM.assemble_hss_fmm(A, tree)          # → HMatrices.HSSMatrix
H2 = FMM.assemble_h2_fmm(A, tree)           # → HMatrices.H2Matrix (HARA)
H2 = FMM.assemble_h2_fmm(points; kernel=:laplace2d)
```
"""
module FMM

using LinearAlgebra
using StaticArrays
using Statistics
using Printf
using SpecialFunctions

# Shared geometry + HSS (parent BEM must include HMatrices before FMM)
using ..HMatrices:
    ClusterTree,
    GeometricSplitter,
    CardinalitySplitter,
    PrincipalComponentSplitter,
    HyperRectangle,
    leaves,
    HSSMatrix,
    HSSBasisID,
    H2Matrix,
    assemble_hss,
    hara_h2,
    KernelMatvecSampler,
    expand_tree,
    ExpandedClusterTree

# Non-exported tree helpers used throughout multipole code
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
    maxrank

export FMMVals
export lfmm2d, rfmm2d, l2ddir, r2ddir
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
export assemble_hss_fmm
export assemble_h2_fmm, assemble_h2_fmm_kernel
export FMMKernelMatrix, fmm_laplace3d_matrix, fmm_laplace2d_matrix, fmm_yukawa3d_matrix
export fmm_laplace2d_double_layer_matrix
export KelvinFMMMatrix, fmm_kelvin2d_matrix
# Re-export tree types for FMM callers (same objects as HMatrices)
export ClusterTree, GeometricSplitter, CardinalitySplitter, HyperRectangle

include("expansions.jl")
include("kernels.jl")
include("fmm_core.jl")          # StrongAdmissibility + dual-tree Laplace 2D
include("common/dualtree.jl")
include("cauchy2d/cauchy2d.jl")
include("helmholtz2d/helmholtz2d.jl")
include("stokes2d/stokes2d.jl")
include("laplace3d/laplace3d.jl")
include("helmholtz3d/helmholtz3d.jl")
include("stokes3d/stokes3d.jl")
include("yukawa2d/yukawa2d.jl")
include("yukawa3d/yukawa3d.jl")
include("common/body.jl")
include("kifmm/surface.jl")
include("kifmm/kernels.jl")
include("kifmm/kifmm.jl")
include("hss/fmm_kernel.jl")
include("hss/assemble_hss_fmm.jl")
include("hss/assemble_h2_fmm.jl")

end # module
