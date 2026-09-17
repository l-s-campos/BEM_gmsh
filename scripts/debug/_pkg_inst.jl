using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))
Pkg.instantiate()
Pkg.status("OrdinaryDiffEq")
using OrdinaryDiffEq
println("OrdinaryDiffEq loaded")
