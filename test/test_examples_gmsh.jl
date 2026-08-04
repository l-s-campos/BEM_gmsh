# Smoke-run analytical Gmsh examples (discontinuous format2d)
using Test
using DrWatson
@quickactivate :BEM

@testset "discontinuous format2d nodes" begin
    ξ1, w1 = discontinuous_nodes_weights(1)
    ξ2, w2 = discontinuous_nodes_weights(2)
    ξ3, w3 = discontinuous_nodes_weights(3)
    @test length(ξ1) == 2 && length(ξ2) == 3 && length(ξ3) == 4
    # Gauss nodes = interior (discontinuous)
    @test all(abs.(ξ1) .< 1) && all(abs.(ξ2) .< 1)
    @test sum(w1) ≈ 2 rtol=1e-14
    @test sum(w2) ≈ 2 rtol=1e-14
    # match FastGaussQuadrature
    g2, wg2 = gausslegendre(3)
    @test ξ2 ≈ g2 && w2 ≈ wg2
end

@testset "gmsh examples" begin
    for f in (
        "laplace_linear_Tx.jl",
        "elasticity_patch.jl",
        "geo_unit_square.jl",
        "plate_ss_navier.jl",
        "crack_feddersen.jl",
    )
        path = datadir("examples", f)
        @info "running" f
        include(path)
        @test true
    end
end
