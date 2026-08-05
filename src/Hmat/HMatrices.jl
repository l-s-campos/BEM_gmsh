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
using Statistics
using Printf
using Distributed
using Base.Threads
using SparseArrays
using GLMakie
export plot_hmatrix

const AdjOrMat = Union{Matrix, Adjoint{<:Any, <:Matrix}}

"""
    abstract type AbstractStructuredMatrix{T} <: AbstractMatrix{T}

Abstract supertype for rank-structured matrices in this package
(`HMatrix`, `BLRMatrix`, `HODLRMatrix`, `HSSMatrix`/`HBSMatrix`, ...).
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

include("utils.jl")
include("hyperrectangle.jl")
include("clustertree.jl")
include("splitter.jl")
include("kernelmatrix.jl")
include("scalarize.jl")
include("rkmatrix.jl")
include("compressor.jl")
include("hmatrix.jl")
include("blr.jl")
include("hodlr.jl")
include("hss.jl")
include("h2matrix.jl")
include("anchornet.jl")
include("structured.jl")
include("dhmatrix.jl")
include("multiplication.jl")
include("hlru.jl")
include("hara.jl")
include("h2_basis.jl")
include("triangular.jl")
include("lu.jl")
include("cholesky.jl")

export ClusterTree,
    CardinalitySplitter,
    DyadicSplitter,
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
    HODLRMatrix,
    HSSMatrix,
    HSSBasisID,
    HSSScatteringNode,
    HBSMatrix,
    HBSBasisID,
    HBSScatteringNode,
    H2Matrix,
    H2BoxAdmissibility,
    AnchorNetCompressor,
    DataDrivenLR,
    ScalarizedMatrix,
    ExpandedClusterTree,
    KernelMatrix,
    StrongAdmissibilityStd,
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
    assemble_hodlr,
    assemble_hss,
    assemble_hbs,
    assemble_h2,
    h2_proxy_entry,
    h2_proxy_block,
    h2_orthog!,
    h2_compress!,
    hlru!,
    hadd!,
    AbstractMatvecSampler,
    FunctionSampler,
    KernelMatvecSampler,
    hara,
    assemble_structured,
    anchor_net,
    anchor_net_sample,
    farthest_point_sample,
    dd_onesided,
    dd_twosided,
    scalarize,
    descalarize,
    scalarize_kernel,
    expand_tree,
    expand_range,
    assemble_hmatrix_scalarized,
    apply_scalarized,
    # tree / block introspection (not exporting nodes -- clashes with BEM.nodes)
    leaves,
    pivot,
    rowperm,
    colperm,
    rowrange,
    colrange,
    hasdata,
    isadmissible

end
