# Kirchhoff plates, large deflection, buckling, shallow shells.
# Not reexported by `using BEM` — use `BEM.Plate` or `using BEM.Plate`.
"""
    BEM.Plate

Thin-plate BEM (Shi–Bezine isotropic and Lekhnitskii anisotropic),
FSDT/Reissner (Vander Weeën), symmetric-laminate Wang kernels,
unsymmetric Hsu–Hwu 5×5 (Useche 8.3.1), Reissner Dual BEM cracks (Ch.10),
von Kármán large deflection, buckling, and laminated shallow shells
(Useche Ch.9: Wang plate + Lekhnitskii membrane, DIBEM curvature/inertia).
Day-1 names (`ThinPlateProps`, `AnisoThinPlateProps`, `solve_plate!`, …)
are reexported from `ThinPlate`.
"""
module Plate

using LinearAlgebra
using Statistics
using StaticArrays
using FastGaussQuadrature
using ProgressMeter
using NonlinearSolve
using ADTypes: AutoForwardDiff
using ForwardDiff
using Printf

let P = parentmodule(@__MODULE__)
    for n in names(P; all=true)
        s = String(n)
        (startswith(s, "#") || n === :eval || n === :include || n === :ThinPlate) && continue
        isdefined(P, n) || continue
        try
            Core.eval(@__MODULE__, Expr(:import, Expr(:., :., :., n)))
        catch
        end
    end
end

include("ThinPlate.jl")
using .ThinPlate
using Reexport
@reexport using .ThinPlate

include("FSDT.jl")
include("ThinPlateDibem.jl")
include("ShellGeometry.jl")
include("LaminatedShell.jl")
include("LargePlate.jl")
include("Buckling.jl")
include("Shell.jl")

end
