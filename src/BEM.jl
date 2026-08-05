module BEM
using Reexport
@reexport using DrWatson
@reexport using FastGaussQuadrature
@reexport using Infiltrator
@reexport using LinearAlgebra
@reexport using StaticArrays
# Statistics (mean, median, …) available in BEM and all bare-included files;
# nested submodules also `using Statistics` explicitly.
@reexport using Statistics
@reexport using TimerOutputs
@reexport using GLMakie
@reexport using Gmsh
using Tensorial
using Krylov
using Distances
using NearestNeighbors
using SparseArrays
using ProgressMeter
using LinearSolve
using DifferentialEquations
using SpecialFunctions
using NonlinearSolve
using ADTypes
using Printf
try
    using ForwardDiff
catch
end

# ---------------------------------------------------------------------------
# Hierarchical matrices (vendored)
# ---------------------------------------------------------------------------
include("Hmat/HMatrices.jl")
@reexport using .HMatrices

# ---------------------------------------------------------------------------
# Fast multipole methods (vendored)
# ---------------------------------------------------------------------------
# Note: entry file is mod_FMM.jl (not FMM.jl) — Windows is case-insensitive
# and would clash with fmm_core.jl / historical FMM.jl names.
include("FMM/mod_FMM.jl")
@reexport using .FMM

# ---------------------------------------------------------------------------
# Core — geometry, mesh I/O, RBF (no physics fundamentals yet)
# ---------------------------------------------------------------------------
include("Core/Interpolation.jl")
include("Core/Structures.jl")
include("Core/LinearSolveUtils.jl")  # bem_linsolve → LinearSolve.jl
include("Core/Bezier.jl")
include("Core/Kernels.jl")
include("Core/Integration.jl")
include("Core/GmshSession.jl")
include("Core/Input.jl")
include("Core/Dumont.jl")
include("Core/SST_Leonel.jl")
include("Core/Radial_Basis_Functions.jl")
include("Core/Visualization.jl")
include("Core/Parallel.jl")
include("Core/GeometricProperties.jl")
include("Core/DIBEM_common.jl")   # factored M, F-solve (needs RBF + Hmat)
include("Core/Assembly_factored.jl")  # ColWeightedOp, node_weights, MixedBC

# ---------------------------------------------------------------------------
# Problem types — fundamentals
# ---------------------------------------------------------------------------
include("Laplace/Fundamental.jl")
include("Laplace/Orthotropic.jl")
include("Helmholtz/Fundamental.jl")
include("Elasticity/Fundamental.jl")
include("Elasticity/Axisymmetric.jl")
include("Elasticity/DiBFM.jl")

# ---------------------------------------------------------------------------
# Core — assembly / BC / steady solve / analytics
# (after fundamentals: Assembly_full dispatches on fundamental)
# ---------------------------------------------------------------------------
include("Core/Assembly_full.jl")
include("Core/Boundary_conditions.jl")
include("Core/Solver.jl")
include("Core/Analytical.jl")

# ---------------------------------------------------------------------------
# Laplace (H-matrix assembly, DIBEM, transient, specialty methods)
# ---------------------------------------------------------------------------
include("Laplace/Assembly_H.jl")
include("Laplace/Domain.jl")
include("Laplace/Domain_fast.jl")
include("Laplace/Solver.jl")          # Houbolt / heat / wave
include("Laplace/DLIM.jl")
include("Laplace/DiBFM_HMLS.jl")
include("Laplace/ParticularSolution.jl")
include("Laplace/ModalModified.jl")
include("Laplace/DiffuseAdvective.jl")

# ---------------------------------------------------------------------------
# Elasticity DIBEM + thermo
# ---------------------------------------------------------------------------
include("Elasticity/Domain.jl")
include("Elasticity/Domain_fast.jl")
include("Elasticity/Thermoelasticity.jl")
include("Elasticity/LocalFrame.jl")   # Leonardo §4.7 local (n,t) BEM

# ---------------------------------------------------------------------------
# Multi-region / interfaces / frictional contact BEM
# ---------------------------------------------------------------------------
include("MultiRegion/SubRegions.jl")

# ---------------------------------------------------------------------------
# Contact half-space
# ---------------------------------------------------------------------------
include("Contact/ContactHalfSpace.jl")
@reexport using .ContactHalfSpace

include("Contact/ContactHalfPlane2D.jl")
@reexport using .ContactHalfPlane2D

include("Contact/HalfSpaceBEM.jl")
@reexport using .HalfSpaceBEM

include("Contact/CattaneoMindlin.jl")
@reexport using .CattaneoMindlin

include("Contact/MortarContact2D.jl")
@reexport using .MortarContact2D

# ---------------------------------------------------------------------------
# Crack — dual BEM + propagation
# ---------------------------------------------------------------------------
include("Crack/Crack.jl")
@reexport using .Crack

# ---------------------------------------------------------------------------
# Kirchhoff thin plates
# ---------------------------------------------------------------------------
include("Plate/ThinPlate.jl")
@reexport using .ThinPlate
include("Plate/LargePlate.jl")
include("Plate/Buckling.jl")
include("Plate/Shell.jl")

const AVOID_INF = 1.0e-16

end # module
