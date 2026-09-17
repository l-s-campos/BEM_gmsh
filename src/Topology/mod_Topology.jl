# Pacheco / level-set / topological-derivative optimization.
# Not reexported by `using BEM` — use `BEM.Topology` or `using BEM.Topology`.
"""
    BEM.Topology

Explicit-boundary topology optimization: loops → BEM, Pacheco node motion,
Amstutz / Hamilton–Jacobi level-set, DIBEM-SIMP (heat and elasticity, 2-D
and 3-D density), Portela (2012) dual-BEM shape design, heat and
elasticity topological derivatives (2-D plane stress / 3-D spherical cavity).
"""
module Topology

using LinearAlgebra
using StaticArrays
using Statistics
using SparseArrays
using Printf
using Gmsh
using NearestNeighbors
using DrWatson: datadir

# Files in this folder were written as BEM includes. Bind parent names here
# so we do not maintain a hand list of `import ..foo`.
let P = parentmodule(@__MODULE__)
    for n in names(P; all=true)
        s = String(n)
        (startswith(s, "#") || n === :eval || n === :include) && continue
        isdefined(P, n) || continue
        try
            Core.eval(@__MODULE__, Expr(:import, Expr(:., :., :., n)))
        catch
        end
    end
end

include("Topology.jl")
include("MarchingSquares.jl")
include("MarchingCubes.jl")
include("TopologicalDerivative.jl")
include("ElasticityDT.jl")
include("Pacheco.jl")
include("DibemSIMP.jl")
include("LevelSet.jl")
include("PortelaShape.jl")
include("Problems.jl")
include("Topology3D.jl")

# JuMP/HiGHS/NLopt are loaded in `__init__` so BEM can precompile without
# SparseMatrixColoringsJuMPExt (Julia 1.13 pkgimage flags).
function __init__()
    try
        include(joinpath(@__DIR__, "JumpShape.jl"))
    catch e
        @debug "Topology JumpShape (JuMP/HiGHS) not loaded" exception = e
    end
    return
end

end
