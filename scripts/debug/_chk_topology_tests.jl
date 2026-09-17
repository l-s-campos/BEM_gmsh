using Test
using LinearAlgebra
using StaticArrays
using DrWatson
using Statistics
@quickactivate :BEM

include(joinpath(@__DIR__, "legacy_tests", "test_topology.jl"))
println("all topology tests done")
