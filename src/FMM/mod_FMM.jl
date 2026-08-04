"""
    module FMM

Pure Julia fast multipole methods (2D & 3D). Vendored into BEM as `BEM.FMM`.

Tree types are not re-exported (use `FMM.ClusterTree`) to avoid clashing with HMatrices.
"""
module FMM

using LinearAlgebra
using StaticArrays
using Statistics: median, mean
using Printf
using SpecialFunctions

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
export FMMKernelMatrix, fmm_laplace3d_matrix, fmm_laplace2d_matrix, fmm_yukawa3d_matrix

include("tree/hyperrectangle.jl")
include("tree/utils.jl")
include("tree/clustertree.jl")
include("tree/splitter.jl")
include("expansions.jl")
include("kernels.jl")
include("fmm_core.jl")
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
include("hss/hss_basis.jl")
include("hss/hss_matrix.jl")
include("hss/fmm_kernel.jl")
include("hss/assemble_hss_fmm.jl")

end # module
