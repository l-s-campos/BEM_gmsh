using DrWatson
@quickactivate :BEM
include(joinpath(dirname(dirname(@__FILE__)), "test", "dibem.jl"))
println("dibem tests ok")
