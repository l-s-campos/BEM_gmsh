using Test
using LinearAlgebra
using StaticArrays
using DrWatson
@quickactivate :BEM
using BEM.Topology

@testset "topology loops → BEM" begin
    d = pacheco_inverted_v(; ne=8, nint=8, degree=1)
    @test design_area(d) ≈ 1.0 atol=1e-8
    dad = bemdata_from_loops(d)
    @test dad.n > 0
    @test dad.ni > 0
    A = geometric_props(dad).area
    @test A ≈ 1.0 rtol=0.05
    H_G_full_direct(dad; npg=8, threaded=false)
    solve(dad)
    @test all(isfinite, dad.T)
    J = thermal_conductance(dad)
    @test J > 0
end

@testset "DT on T = x" begin
    include(datadir("Laplace", "Laplace_dad.jl"))
    msh = quadrado(ndiv=8, show=false, nome="topo_Tx")
    dad = format2d(msh, Laplace(1.0); pontointerno=true)
    H_G_full_direct(dad; npg=10, threaded=false)
    solve(dad)
    DTb, DTi = topological_derivative(dad)
    @test median(DTb) ≈ 1.0 atol=0.25
    @test median(DTi) ≈ 1.0 atol=0.35
end

@testset "Dirichlet vertices stay frozen" begin
    d = pacheco_inverted_v(; ne=6, nint=6, degree=1)
    dad = bemdata_from_loops(d)
    H_G_full_direct(dad; npg=8, threaded=false)
    solve(dad)
    DTb, DTi = topological_derivative(dad)
    before = deepcopy(d.loops[1][1].verts)
    @test d.loops[1][1].bc == [0]
    opt = PachecoOptions(; vmax=0.2, passos=2, nucleate_first=false, nucleate_every=typemax(Int), verbose=false)
    move_boundary!(d, dad, DTb, opt)
    @test all(before .≈ d.loops[1][1].verts)
end

@testset "marching squares closed curve" begin
    xs = range(-1, 1; length=25)
    ys = range(-1, 1; length=25)
    R = 0.55
    Z = [R - hypot(x, y) for x in xs, y in ys]
    lines = marching_squares(xs, ys, Z, 0.0)
    closed = filter(ln -> length(ln) ≥ 8 && norm(ln[1] - ln[end]) < 0.2, lines)
    @test !isempty(closed)
    pts = closed[1][1] ≈ closed[1][end] ? closed[1][1:end-1] : closed[1]
    a = abs(polygon_area(pts))
    @test a ≈ π * R^2 rtol=0.2
end

@testset "HJ volume shift shrinks a circle" begin
    xs = collect(range(-1, 1; length=33))
    ys = collect(range(-1, 1; length=33))
    φ = [0.45 - hypot(x, y) for x in xs, y in ys]
    g = LevelSetGrid(xs, ys, φ)
    A0 = BEM._area_from_phi(g)
    BEM._shift_phi_to_area!(g, 0.55 * A0)
    A1 = BEM._area_from_phi(g)
    @test A1 < 0.75 * A0
    @test A1 > 0.35 * A0
end

@testset "HJ reinit of a circle" begin
    d = pacheco_inverted_v(; ne=6, nint=6, degree=1)
    g = LevelSetGrid(d; ngrid=32)
    # overwrite φ with a circle signed distance
    for j in eachindex(g.ys), i in eachindex(g.xs)
        g.φ[i, j] = 0.3 - hypot(g.xs[i] - 0.5, g.ys[j] - 0.5)
    end
    φ0 = copy(g.φ)
    signed_distance!(g; niter=12)
    # interface stays near the circle; |∇φ| ~ 1 away from it
    mag = Float64[]
    nx, ny = size(g.φ)
    for j in 3:(ny - 2), i in 3:(nx - 2)
        abs(φ0[i, j]) < 0.05 || continue
        gx = (g.φ[i + 1, j] - g.φ[i - 1, j]) / (2g.dx)
        gy = (g.φ[i, j + 1] - g.φ[i, j - 1]) / (2g.dy)
        push!(mag, hypot(gx, gy))
    end
    @test !isempty(mag)
    @test abs(median(mag) - 1) < 0.4
end

@testset "Pacheco smoke (problem 1, few iters)" begin
    d, opt = pacheco_problem(1; ne=8, nint=8, degree=1)
    opt.maxiter = 2
    opt.passos = 2
    opt.verbose = false
    opt.nucleate_every = typemax(Int)
    A0 = design_area(d)
    d2, dad, hist = solve_topology!(d, opt; history=true)
    @test hist.area[end] ≤ A0 + 1e-10
    @test length(hist.area) ≥ 2
    @test all(isfinite, dad.T)
end

# ---------------------------------------------------------------------------
# Elasticity (Coelho 2021)
# ---------------------------------------------------------------------------

@testset "plane-stress DT formula" begin
    ν = 0.3
    σ = reshape([1.0, 0.0, 0.0], 1, 3)
    ε = reshape([1.0, -ν, -ν, 0.0], 1, 4)
    DT = plane_stress_DT(σ, ε, ν)
    c1 = 2 / (1 + ν)
    cons = (3ν - 1) / (2 * (1 - ν^2))
    expected = c1 * 1.0 + cons * (1 - 2ν)
    @test DT[1] ≈ expected atol=1e-12
end

@testset "elasticity loops → BEM + clamp freeze" begin
    d, opt = coelho_problem(3; ne=4, nint=6, degree=1)
    @test design_area(d) ≈ 1.5 atol=1e-8
    @test d.properties isa Elasticity
    @test plane_stress(d.properties)
    dad = bemdata_from_loops(d)
    @test dad.n > 0
    @test dad.ni > 0
    H_G_full_direct(dad; npg=8, threaded=false)
    solve(dad)
    @test all(isfinite, dad.u)
    J = elastic_compliance(dad)
    @test J > 0
    DTb, DTi = topological_derivative(dad)
    @test all(isfinite, DTb)
    @test !isempty(DTi) && all(isfinite, DTi)
    clamp_i = findfirst(s -> s.bc == [0, 0], d.loops[1])
    @test clamp_i !== nothing
    before = deepcopy(d.loops[1][clamp_i].verts)
    opt.vmax = 0.2
    opt.passos = 2
    opt.nucleate_first = false
    opt.nucleate_every = typemax(Int)
    opt.verbose = false
    move_boundary!(d, dad, DTb, opt)
    @test all(before .≈ d.loops[1][clamp_i].verts)
end

@testset "Coelho cases 4–5 geometry" begin
    d4, o4 = coelho_problem(4; ne=3, nint=4, degree=1)
    d5, o5 = coelho_problem(5; ne=3, nint=4, degree=1)
    @test design_area(d4) ≈ 2.0 atol=1e-8
    @test design_area(d5) ≈ 2.0 atol=1e-8
    @test o4.ΔA ≈ 0.60
    @test o5.ΔA ≈ 0.65
    @test any(s -> s.bc == [1, 1] && s.value[2] < 0, d4.loops[1])
    @test any(s -> s.bc == [1, 1] && s.value[2] < 0, d5.loops[1])
    @test any(s -> s.bc == [0, 0], d4.loops[1])
    @test any(s -> s.bc == [0, 0], d5.loops[1])
end

@testset "Coelho cantilever smoke" begin
    d, opt = coelho_problem(3; ne=4, nint=6, degree=1)
    opt.maxiter = 2
    opt.passos = 2
    opt.verbose = false
    opt.nucleate_every = typemax(Int)
    A0 = design_area(d)
    d2, dad, hist = solve_topology!(d, opt; history=true)
    @test hist.area[end] ≤ A0 + 1e-10
    @test length(hist.area) ≥ 2
    @test all(isfinite, dad.u)
    @test hist.J[1] > 0
end
