# Half-space / half-plane / mortar / Cattaneo–Mindlin / Juliá Lerma contact.
# Not reexported by `using BEM` — use `BEM.Contact` or `using BEM.Contact`.
"""
    BEM.Contact

Elastic contact on half-spaces and half-planes:

- Pohrt–Li 9-kernel operators (FFT / H-matrix / H² / FMM), two-body Kalker combination
- Layered/anisotropic Fourier compliance (Bagault et al. 2013)
- Uzawa / Alart–Curnier with elliptic Coulomb and orthotropic Archard wear
- Rolling (Kalker slip) and planar wheel–rail (Vollebregt CONTACT D=2)
- Legacy 2-D/3-D log-kernel operators (FFT / H-matrix / FMM)
- Cattaneo–Mindlin and mortar (segment-to-segment)

Multibody frictional contact on `BEMdata` lives in [`BEM.MultiRegion`](@ref).
"""
module Contact

using ..HMatrices
using ..FMM

include("ContactHalfSpace.jl")
include("LayeredAniso.jl")
include("OrthotropicUzawa.jl")
include("SubsurfaceStress.jl")
include("RollingContact.jl")
include("WheelRail.jl")
include("JuliaLermaCases.jl")
include("ContactHalfPlane2D.jl")
include("HalfSpaceBEM.jl")
include("CattaneoMindlin.jl")
include("MortarContact2D.jl")

using Reexport
@reexport using .ContactHalfSpace
@reexport using .LayeredAniso
@reexport using .OrthotropicUzawa
@reexport using .SubsurfaceStress
@reexport using .RollingContact
@reexport using .WheelRail
@reexport using .JuliaLermaCases
@reexport using .ContactHalfPlane2D
@reexport using .HalfSpaceBEM
@reexport using .CattaneoMindlin
@reexport using .MortarContact2D

end
