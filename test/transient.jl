# Transients: second-order API + MMM free-vibration scale.
using Test
using LinearAlgebra
using StaticArrays
using BEM

@testset "solve_transient_o2 smoke" begin
    dad = format2d(quadrado(ndiv=6, show=false, nome="t_o2"), Laplace(1.0);
        pontointerno=true)
    assemble!(dad, 10)
    DIBEM(dad)
    sol = solve_transient_o2(dad, 0.05, 0.15; abstol=1e-4, reltol=1e-4)
    @test sol !== nothing
    @test all(isfinite, dad.T)
end

@testset "Newmark body force finite" begin
    dad = format2d(quadrado(ndiv=6, show=false, nome="t_nmk_force"), Laplace(1.0);
        pontointerno=true)
    dad.BC .= 1
    dad.BV .= 0.0
    assemble!(dad, 8)
    DIBEM(dad; rbf=PHS(3; poly_deg=1))
    g = [exp(-(p[1] - 0.5)^2 - (p[2] - 0.5)^2) for p in all_points(dad)]
    solve_Newmark(dad, 0.05, 0.15; force=t -> sin(π * t) .* g)
    @test all(isfinite, dad.T)
    @test size(dad.T, 2) == length(0:0.05:0.15)
    @test maximum(abs, dad.T) > 0
end

@testset "MMM time-dependent body force" begin
    dad = format2d(quadrado(ndiv=6, show=false, nome="t_mmm_ft"), Laplace(1.0);
        tipo=1, pontointerno=false)
    dad.BC .= 1
    dad.BV .= 0.0
    xs = range(0.2, 0.8; length=3)
    set_internal_nodes!(dad, [SVector(x, y) for y in xs for x in xs])
    assemble!(dad, 8)
    DIBEM(dad; rbf=PHS(3; poly_deg=1))
    g = [exp(-(p[1] - 0.5)^2 - (p[2] - 0.5)^2) for p in all_points(dad)]
    Mg = dad.M * g
    U, t, basis = solve_mmm!(dad, 0.05, 0.2; f=tt -> sin(π * tt) .* Mg)
    @test all(isfinite, U)
    @test length(t) == length(0:0.05:0.2)
    @test length(basis.ω) >= 1
end

@testset "MMM membrane ω₁ scale" begin
    dad = format2d(quadrado(ndiv=6, show=false, nome="t_mmm"), Laplace(1.0);
        tipo=1, pontointerno=false)
    dad.BC .= 0
    dad.BV .= 0.0
    xs = range(0.2, 0.8; length=3)
    set_internal_nodes!(dad, [SVector(x, y) for y in xs for x in xs])
    assemble!(dad, 10)
    DIBEM(dad)
    sys = build_modal_system(dad)
    basis = modal_analysis_mmm(sys; nmodes=min(3, size(sys.M, 1)))
    ω_ana = π * sqrt(2)
    @test basis.ω[1] > 0
    @test abs(basis.ω[1] - ω_ana) / ω_ana < 0.5
end
