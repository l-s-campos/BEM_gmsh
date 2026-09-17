"""
    BEM.Examples

Day-1 Gmsh mesh builders reexported by `using BEM`: `quadrado`,
`quadrado_elasticity`, `placa_com_furo`. Extra `.geo` / `.jl` generators
stay under `data/` and are `include`d from `scripts/`.
"""
module Examples

using Gmsh
using DrWatson: datadir

export quadrado, quadrado_elasticity, quadrado_plate, quadrado_fsdt, placa_com_furo, placa_furo_orto, placa_furo_orto_3d, cubo_furo_orto

include(joinpath(@__DIR__, "..", "data", "Laplace", "Laplace_dad.jl"))

end
