using Test
using DrWatson
@quickactivate :BEM

println("Starting BEM.jl tests")
ti = time()

@testset "BEM.jl" begin
    include("core.jl")
    include("laplace.jl")
    include("elasticity.jl")
    include("plasticity.jl")
    include("dibem.jl")
    include("transient.jl")
    include("domain_methods.jl")
    include("helmholtz.jl")
    include("crack.jl")
    include("cohesive.jl")
    include("contact.jl")
    include("test_julia_lerma_contact.jl")
    include("test_julia_lerma_convergence.jl")
    include("test_layered_aniso.jl")
    include("test_wheel_rail.jl")
    include("multiregion.jl")
    include("plates.jl")
    include("topology.jl")
    include("hmat.jl")
    include("fmm.jl")
end

println("\nTests finished in ", round((time() - ti) / 60; digits=3), " minutes")
