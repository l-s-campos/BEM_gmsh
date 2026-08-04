# Standalone thermoelasticity tests (fast — no full suite)
using Test
using DrWatson
@quickactivate :BEM

@testset "thermoelasticity" begin
    E, ν, α, Δθ = 1000.0, 0.3, 1e-5, 50.0
    props = Elasticity(E, ν, 1.0; plane_strain=true, α=α)
    k̂ = thermal_modulus(props)
    @test k̂ ≈ E * α / (1 - 2ν)
    @test analytical_constrained_thermal_stress(props, Δθ) ≈ -k̂ * Δθ

    include(datadir("Laplace", "Laplace_dad.jl"))  # quadrado_elasticity
    msh = quadrado_elasticity(ndiv=6, show=false, nome="test_thermo")
    dad = format2d(msh, props; pontointerno=false)
    fill!(dad.BC, 0)
    fill!(dad.BV, 0.0)
    H_G_full_direct(dad; npg=10, threaded=false)
    u = solve_thermoelastic!(dad; θ=Δθ)
    @test maximum(abs, u) < 1e-8

    t = dad.traction
    err = 0.0
    for i in 1:dad.n
        t_ana = -k̂ * Δθ * dad.Normal[i]
        err = max(err, abs(t[2i-1] - t_ana[1]), abs(t[2i] - t_ana[2]))
    end
    @test err / (abs(k̂ * Δθ) + eps()) < 0.15
    @info "thermoelasticity constrained" k̂ err_rel = err / (abs(k̂ * Δθ) + eps())

    # non-uniform θ path (builds DIBEM M)
    θfun = (x, y) -> Δθ * (1 + 0.1 * x)
    u2 = solve_thermoelastic!(dad; θ=θfun, npg_dibem=8)
    @test length(u2) == 2 * dad.n
    @test has_cache(dad, :M)
end
