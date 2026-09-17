"""
    BEM

2-D/3-D boundary element teaching spine:

```
mesh → format2d → assemble! → [dibem!] → solve → dad.T / dad.q
```

`using BEM` is collocation BEM for Laplace, Helmholtz, and elasticity.
Specialist physics is **not** reexported — load it from a submodule.
"""
module BEM
using Reexport
using DrWatson
@reexport using FastGaussQuadrature
@reexport using LinearAlgebra
@reexport using StaticArrays
@reexport using Statistics
using Plots
@reexport using Gmsh
using Tensorial
using Krylov
try
    using LinearMaps
catch
end
try
    using ArnoldiMethod
catch
end

using NearestNeighbors
using SparseArrays
using ProgressMeter
using LinearSolve
try
    using OrdinaryDiffEq: ODEProblem, ODEFunction, Rodas5P
    import OrdinaryDiffEq
catch
    try
        using DifferentialEquations
    catch
    end
end
using SpecialFunctions
using NonlinearSolve
using ADTypes
using Printf
try
    using Richardson
catch
end
try
    using ForwardDiff
catch
end
try
    using Distances
catch
end
try
    using Infiltrator
catch
end
try
    using TimerOutputs
catch
end

include("Hmat/HMatrices.jl")
using .HMatrices

include("FMM/mod_FMM.jl")
using .FMM

include("Core/Interpolation.jl")
include("Core/Structures.jl")
include("Core/LinearSolveUtils.jl")
include("Core/Kernels.jl")
include("Core/Integration.jl")
include("Core/Input.jl")
include("Core/InternalPoints.jl")
include("Core/SBM_geom.jl")
include("Core/Radial_Basis_Functions.jl")
include("Core/Visualization.jl")
include("Core/GeometricProperties.jl")
include("Core/Assembly_factored.jl")
include("Core/DIBEM_common.jl")

include("Laplace/Fundamental.jl")
include("Laplace/Orthotropic.jl")
include("Helmholtz/Fundamental.jl")
include("Elasticity/Fundamental.jl")
include("Elasticity/Anisotropic3D.jl")
include("Elasticity/Axisymmetric.jl")
include("Elasticity/DiBFM.jl")

include("Core/SurfaceDIBEM.jl")
include("Core/Assembly_full.jl")
include("Core/Boundary_conditions.jl")
include("Core/Solver.jl")
include("Core/Analytical.jl")

include("Laplace/Assembly_H.jl")
include("Laplace/Assembly_GPU.jl")
include("Laplace/Assembly_galerkin.jl")
include("Laplace/Domain.jl")
include("Laplace/DIBEM_GPU.jl")
include("Laplace/Domain_fast.jl")
include("Laplace/Heterogeneous.jl")
include("Laplace/AnisotropicDIBEM.jl")
include("Laplace/Solver.jl")
include("Laplace/DLIM.jl")
include("Laplace/SBM.jl")
include("Laplace/SBM_DRM.jl")
include("Laplace/KansaBEM_Heat.jl")
include("Laplace/KansaSBM_Heat.jl")
include("Laplace/DiBFM_HMLS.jl")
include("Laplace/ParticularSolution.jl")
include("Laplace/Lubrication.jl")
include("Laplace/ElrodAdams.jl")
include("Laplace/LocalBEM.jl")
include("Laplace/BurtonMiller.jl")
include("Laplace/ModalModified.jl")
include("Laplace/DiffuseAdvective.jl")

include("Elasticity/Domain.jl")
include("Elasticity/Domain_fast.jl")
include("Elasticity/Assembly_GPU.jl")
include("Elasticity/Heterogeneous.jl")
include("Elasticity/LocalBEM.jl")
include("Elasticity/Thermoelasticity.jl")
include("Elasticity/LocalFrame.jl")
include("Elasticity/StrainStress.jl")
include("Elasticity/PlasticKernels.jl")
include("Elasticity/Plasticity.jl")
include("Elasticity/Transient.jl")
include("Elasticity/SBM.jl")

include("Topology/mod_Topology.jl")
include("MultiRegion/MultiRegion.jl")
include("Contact/Contact.jl")
include("Laplace/SemiSystem.jl")
include("Crack/Crack.jl")
include("Plate/Plate.jl")

include("Examples.jl")
@reexport using .Examples

const AVOID_INF = 1.0e-16

end # module
