"""
    BEM.HMatrices

Vendored hierarchical matrices: cluster trees, ACA / BLR / NNCA H²,
and `HMatrix` matvecs used by `assemble!(dad; method=:hmatrix)`.

Load with `using BEM.HMatrices` (not reexported by `using BEM`).
Trees here are also the spatial index for [`BEM.FMM`](@ref).
"""
module HMatrices

# When included as a submodule, pkgdir may be `nothing` — fall back safely.
const PROJECT_ROOT = let p = try
        pkgdir(@__MODULE__)
    catch
        nothing
    end
    p === nothing ? dirname(dirname(@__DIR__)) : p
end

using StaticArrays
using LinearAlgebra
using Random
using Statistics
using Printf
using Distributed
using Base.Threads
using SparseArrays
using Krylov
using KernelAbstractions
using Plots
export plot_hmatrix

const AdjOrMat = Union{Matrix, Adjoint{<:Any, <:Matrix}}

"""
    abstract type AbstractStructuredMatrix{T} <: AbstractMatrix{T}

Abstract supertype for rank-structured matrices in this package
(`HMatrix`, `BLRMatrix`, `NNCAMatrix`, ...).
"""
abstract type AbstractStructuredMatrix{T} <: AbstractMatrix{T} end

"""
    getblock!(block,K,irange,jrange)

Fill `block` with `K[i,j]` for `i ∈ irange`, `j ∈ jrange`, where `block` is of
size `length(irange) × length(jrange)`.

A default implementation exists which relies on `getindex(K,i,j)`, but this
method can be overloaded for better performance if e.g. a vectorized way of
computing a block is available.
"""
function getblock!(out, K, irange_, jrange_)
    irange = irange_ isa Colon ? axes(K, 1) : irange_
    jrange = jrange_ isa Colon ? axes(K, 2) : jrange_
    for (jloc, j) in enumerate(jrange)
        for (iloc, i) in enumerate(irange)
            out[iloc, jloc] = K[i, j]
        end
    end
    return out
end

function getblock!(out, Kadj::Adjoint, irange_, j::Int)
    getblock!(transpose(out), parent(Kadj), j:j, irange_)
    return out .= conj.(out)
end

"""
    use_threads()::Bool

Default choice of whether threads will be used throughout the package.
"""
use_threads() = true

"""
    use_global_index()::Bool

Default choice of whether operations will use the global indexing system
throughout the package.
"""
use_global_index() = true

# tree/ — geometry only (no FMM payload)
include("tree/utils.jl")
include("tree/hyperrectangle.jl")
include("tree/clustertree.jl")
include("tree/admissibility.jl")
include("tree/splitter.jl")
include("tree/interaction_list.jl")
# formats/ — structured matrix types + kernels
include("formats/kernelmatrix.jl")
include("formats/scalarize.jl")
include("formats/rkmatrix.jl")
include("compress/compressor.jl")
include("compress/id.jl")
include("formats/hmatrix.jl")
include("formats/blr.jl")
include("arith/nnca_aca.jl")
include("formats/nnca.jl")
include("formats/h2_cheb.jl")
include("formats/hss.jl")
include("arith/hss_add.jl")
include("formats/h2_node.jl")
include("compress/anchornet.jl")
include("formats/structured.jl")
include("formats/dhmatrix.jl")
# arith/ — matvec, factorizations, updates
include("arith/multiplication.jl")
include("arith/hlru.jl")
include("arith/matvec_sampler.jl")
include("arith/triangular.jl")
include("arith/lu.jl")
include("arith/h2_rkupdate.jl")
include("arith/h2_lr.jl")
include("arith/h2lu.jl")
include("arith/srs.jl")
include("arith/rskelf.jl")
include("arith/hss_ulv.jl")
include("arith/hodlr_lu.jl")
include("arith/cholesky.jl")
include("arith/precond.jl")
include("arith/ilut.jl")
isfile(joinpath(@__DIR__, "arith", "gpu.jl")) && include("arith/gpu.jl")

export ClusterTree,
    CardinalitySplitter,
    DyadicSplitter,
    hmatrix_splitter,
    GeometricSplitter,
    GeometricMinimalSplitter,
    PrincipalComponentSplitter,
    HyperRectangle,
    # abstract types
    AbstractKernelMatrix,
    AbstractStructuredMatrix,
    # types
    HMatrix,
    BLRMatrix,
    NNCAMatrix,
    GPUHMatrix,
    GPUNNCAMatrix,
    AnchorNetCompressor,
    DataDrivenLR,
    KernelMatrix,
    StrongAdmissibilityStd,
    FMMStrongAdmissibility,
    WeakAdmissibilityStd,
    PartialACA,
    TSVD,
    ACAWithRecompression,
    ACAWithRecompressionBuffer,
    # functions
    allocate_buffer,
    compression_ratio,
    maxrank,
    assemble_hmatrix,
    assemble_blr,
    assemble_h2,
    assemble_nnca,
    hmatrix,
    h2node,
    H2Node,
    H2NodeLU,
    lrdecomp_h2node,
    h2_rkupdate!,
    h2_rkupdate_nested!,
    prepare_h2_weights,
    H2ClusterOperator,
    h2_clone,
    h2_addmul!,
    gpu,
    gpu_wrap,
    hlru!,
    hadd!,
    AbstractMatvecSampler,
    FunctionSampler,
    KernelMatvecSampler,
    add_diag_ridge!,
    gmres_h,
    h2_lu_prec,
    ilut,
    ILUTFactor,
    near_sparse,
    srs_factor,
    srs_factor_matvec,
    SRSFactor,
    interpolative_decomp,
    rskelf,
    RSKELFFactor,
    assemble_hss,
    hss_add,
    assemble_hss_BDC,
    assemble_hss_schur,
    assemble_hss_2x2,
    HSSMatrix,
    HSSBDC,
    HSSSchur,
    HSSBlock2x2,
    ulv,
    ULVFactor,
    hodlr_lu_2x2,
    hodlr_ulv_2x2,
    HODLR2x2LU,
    HODLR2x2ULV,
    circle_proxy,
    sphere_proxy,
    assemble_structured,
    anchor_net,
    anchor_net_sample,
    farthest_point_sample,
    dd_onesided,
    dd_twosided,
    scalarize,
    descalarize,
    expand_range,
    # tree / block introspection (not exporting nodes -- clashes with BEM.nodes)
    leaves,
    node_id,
    nnodes,
    assign_node_ids!,
    neighbor_il_lists,
    boxes_touch,
    pivot,
    rowperm,
    colperm,
    rowrange,
    colrange,
    hasdata,
    isadmissible

end
