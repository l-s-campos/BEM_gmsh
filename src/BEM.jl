module BEM
using Reexport
@reexport using DrWatson
@reexport using FastGaussQuadrature
@reexport using Infiltrator
@reexport using LinearAlgebra
@reexport using StaticArrays
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
# Fast multipole methods (vendored from D:/fmm/FMM2D)
# ---------------------------------------------------------------------------
# Note: entry file is mod_FMM.jl (not FMM.jl) — Windows is case-insensitive
# and would clash with fmm_core.jl / historical FMM.jl names.
include("FMM/mod_FMM.jl")
@reexport using .FMM

# ---------------------------------------------------------------------------
# Core (shared infrastructure)
# ---------------------------------------------------------------------------
include("Core/Interpolation.jl")
include("Core/Structures.jl")
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

# ---------------------------------------------------------------------------
# Problem types — fundamentals
# ---------------------------------------------------------------------------
include("Laplace/Fundamental.jl")
include("Laplace/Orthotropic.jl")
include("Helmholtz/Fundamental.jl")
include("Elasticity/Fundamental.jl")
include("Elasticity/Thermoelasticity.jl")
include("Elasticity/Axisymmetric.jl")
include("Elasticity/DiBFM.jl")

# ---------------------------------------------------------------------------
# Laplace (assembly, BC, solvers, DIBEM, analytics)
# ---------------------------------------------------------------------------
include("Laplace/Boundary_conditions.jl")
include("Laplace/Assembly_full.jl")
include("Laplace/Assembly_H.jl")
include("Laplace/Solver.jl")
include("Laplace/Domain.jl")
include("Laplace/Domain_fast.jl")   # DIBEM_Hmat, DIBEM_FMM, DIBEM(; method=...)
include("Laplace/Analytical.jl")
include("Laplace/DLIM.jl")
include("Laplace/DiBFM_HMLS.jl")
include("Laplace/ParticularSolution.jl")
include("Laplace/ModalModified.jl")
include("Laplace/DiffuseAdvective.jl")

# ---------------------------------------------------------------------------
# Multi-region / interfaces / frictional contact BEM
# ---------------------------------------------------------------------------
include("MultiRegion/SubRegions.jl")

# ---------------------------------------------------------------------------
# Contact half-space (Pohrt–Li, Flamant, wear, FMM/Hmat/FFT)
# ---------------------------------------------------------------------------
include("Contact/ContactHalfSpace.jl")
@reexport using .ContactHalfSpace

include("Contact/ContactHalfPlane2D.jl")
@reexport using .ContactHalfPlane2D

include("Contact/HalfSpaceBEM.jl")
@reexport using .HalfSpaceBEM

# ---------------------------------------------------------------------------
# Crack — dual BEM + propagation (COD, MTS/SED, Paris)
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
