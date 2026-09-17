# Topology: loops → BEM area + DT of T=x.
using Test
using LinearAlgebra
using Statistics: median
using StaticArrays
using BEM
using BEM.Topology

@testset "loops → BEM area" begin
    d = pacheco_inverted_v(; ne=8, nint=8, degree=1)
    @test design_area(d) ≈ 1.0 atol=1e-8
    dad = bemdata_from_loops(d)
    @test geometric_props(dad).area ≈ 1.0 rtol=0.05
    assemble!(dad; npg=8, threaded=false)
    solve(dad)
    @test thermal_conductance(dad) > 0
end

@testset "DT of T=x is ~1" begin
    dad = format2d(quadrado(ndiv=8, show=false, nome="t_topo_x"), Laplace(1.0);
        pontointerno=true)
    assemble!(dad; npg=10, threaded=false)
    solve(dad)
    DTb, DTi = topological_derivative(dad)
    @test median(DTb) ≈ 1.0 atol=0.25
    @test median(DTi) ≈ 1.0 atol=0.35
end

@testset "DIBEM-SIMP K=1 matches homogeneous" begin
    d = pacheco_inverted_v(; ne=6, nint=6, degree=1)
    dad = bemdata_from_loops(d)
    assemble!(dad; npg=8, threaded=false)
    solve(dad)
    C0 = thermal_conductance(dad)
    dad2 = bemdata_from_loops(d)
    assemble!(dad2; npg=8, threaded=false)
    ρ = ones(dad2.nt)
    opt = DibemSimpOptions(p=3.0, K0=1.0, Kmin=1e-3, rbf=PHS(1; poly_deg=-1))
    K = simp_conductivity(ρ, opt)
    solve_heterogeneous!(dad2, K; rbf=opt.rbf)
    @test all(isfinite, dad2.T)
    @test abs(thermal_conductance(dad2) - C0) / max(abs(C0), 1e-12) < 0.25
end

@testset "DIBEM-SIMP one OC step is finite" begin
    d = pacheco_inverted_v(; ne=6, nint=6, degree=1)
    dad = bemdata_from_loops(d)
    assemble!(dad; npg=8, threaded=false)
    opt = DibemSimpOptions(volfrac=0.5, rmin=0.15, n_simp=1, cut=false, verbose=false)
    w = dibem_volume_weights(dad, opt.rbf)
    ρ = fill(opt.volfrac, dad.nt)
    st = dibem_simp_step!(dad, ρ, w, opt)
    @test isfinite(st.C)
    @test 0.2 < st.V < 0.8
    @test all(x -> 0 < x ≤ 1, ρ)
end

@testset "elastic heterogeneous E=E0 matches homogeneous" begin
    d = coelho_cantilever(; ne=5, nint=5, degree=1)
    dad = bemdata_from_loops(d)
    assemble!(dad; npg=8, threaded=false)
    solve(dad)
    J0 = elastic_compliance(dad)
    dad2 = bemdata_from_loops(d)
    assemble!(dad2; npg=8, threaded=false)
    solve_heterogeneous!(dad2, fill(dad2.properties.E, dad2.nt))
    J1 = elastic_compliance(dad2)
    @test all(isfinite, dad2.u)
    @test abs(J1 - J0) / max(abs(J0), 1e-12) < 0.05
end

@testset "DIBEM-SIMP elasticity one OC step is finite" begin
    d = coelho_cantilever(; ne=5, nint=5, degree=1)
    dad = bemdata_from_loops(d)
    assemble!(dad; npg=8, threaded=false)
    opt = DibemSimpOptions(volfrac=0.4, rmin=0.2, n_simp=1, cut=false, verbose=false)
    w = dibem_volume_weights(dad, opt.rbf)
    ρ = fill(opt.volfrac, dad.nt)
    st = dibem_simp_step!(dad, ρ, w, opt)
    @test isfinite(st.C)
    @test st.C > 0
    @test 0.15 < st.V < 0.7
    @test all(x -> 0 < x ≤ 1, ρ)
end

@testset "DT-ρ one shot is finite and hits volume" begin
    d = pacheco_inverted_v(; ne=6, nint=6, degree=1)
    dad = bemdata_from_loops(d)
    assemble!(dad; npg=8, threaded=false)
    solve(dad)
    w = dibem_volume_weights(dad, PHS(1; poly_deg=-1))
    frozen = BEM.Topology._freeze_design_nodes(dad)
    DT = nodal_DT(dad)
    ρ = density_from_dt(DT, w, 0.5, frozen)
    @test all(x -> 0 < x ≤ 1, ρ)
    @test 0.25 < dot(ρ, w) / sum(w) < 0.75
    @test maximum(DT) > minimum(DT)
end

@testset "volume-matched cut reduces area" begin
    d = pacheco_inverted_v(; ne=8, nint=10, degree=1)
    dad = bemdata_from_loops(d)
    ρ = ones(dad.nt)
    c = Point2D(0.5, 0.42)
    for i in 1:dad.nt
        p = point(dad, i)
        if norm(p - c) < 0.22 && i > dad.n
            ρ[i] = 0.05
        end
    end
    A0 = design_area(d)
    opt = DibemSimpOptions(volfrac=0.5, cut=true, pacheco=false, ngrid=41, verbose=false)
    nadd, _, lev = BEM.Topology._cut_with_levels!(d, dad, ρ, opt)
    @test 0.12 ≤ lev ≤ 0.80
    @test nadd ≥ 1 || n_holes(d) > 0
    @test design_area(d) ≤ A0 + 1e-12
end

@testset "loop self-intersection detects a bow-tie" begin
    sq = [Point2D(0.0, 0.0), Point2D(1.0, 0.0), Point2D(1.0, 1.0), Point2D(0.0, 1.0)]
    @test !BEM.Topology._loop_self_intersects(sq)
    bow = [Point2D(0.0, 0.0), Point2D(1.0, 1.0), Point2D(1.0, 0.0), Point2D(0.0, 1.0)]
    @test BEM.Topology._loop_self_intersects(bow)
end

@testset "SIMP+cut+Pacheco approaches volfrac" begin
    d = pacheco_inverted_v(; ne=8, nint=8, degree=1)
    opt = DibemSimpOptions(; volfrac=0.5, n_simp=6, rmin=0.12, ngrid=41,
        cut=true, pacheco=true, verbose=false, npg=8,
        pacheco_opt=PachecoOptions(maxiter=10, verbose=false,
            nucleate_first=false, nucleate_every=typemax(Int), area_rtol=0.08))
    d, dad, ρ, hist = solve_dibem_simp!(d, opt)
    A = design_area(d)
    @test 0.40 < A < 0.65
    @test isfinite(design_objective(dad))
    @test !isempty(hist)
end

@testset "cut_low_density! punches a hole" begin
    d = pacheco_inverted_v(; ne=8, nint=12, degree=1)
    dad = bemdata_from_loops(d)
    ρ = ones(dad.nt)
    c = Point2D(0.5, 0.45)
    for i in 1:dad.nt
        p = point(dad, i)
        if norm(p - c) < 0.18 && i > dad.n
            ρ[i] = 0.05
        end
    end
    n0 = n_holes(d)
    nadd = cut_low_density!(d, dad, ρ; ρ_cut=0.35)
    @test nadd ≥ 1 || n_holes(d) > n0
    @test design_area(d) < 1.0 - 1e-4 || nadd == 0
    # manufactured blob may miss the iso-curve on a coarse interior grid
    @test nadd ≥ 0
end

@testset "JuMP linear step is finite and not folded" begin
    d = pacheco_inverted_v(; ne=6, nint=4, degree=1)
    A0 = design_area(d)
    opt = PachecoOptions(ΔA=0.5, maxiter=3, npg=8, verbose=false, vmax=0.04,
        motion=:jump, jump_mode=:linear, nucleate_first=false,
        nucleate_every=typemax(Int), area_rtol=0.15, nsearch=2)
    d, dad, hist = solve_topology!(d, opt)
    @test isfinite(design_objective(dad))
    @test !BEM.Topology._design_folded(d)
    @test design_area(d) ≤ A0 + 1e-8
end

@testset "DT stand-in motion is finite and not folded" begin
    d = pacheco_inverted_v(; ne=8, nint=6, degree=1)
    A0 = design_area(d)
    opt = PachecoOptions(ΔA=0.5, maxiter=4, npg=8, verbose=false, vmax=0.05,
        motion=:standin, nucleate_first=false, nucleate_every=typemax(Int),
        area_rtol=0.15, standin_α=0.3, nsearch=3)
    d, dad, hist = solve_topology!(d, opt)
    @test isfinite(design_objective(dad))
    @test !BEM.Topology._design_folded(d)
    @test design_area(d) ≤ A0 + 1e-8
    @test !isempty(hist.area)
end

@testset "Portela element grads: constant W, δΨ1 = ℓ/2" begin
    ξ, w = gausslegendre(3)
    ℓ = 2.4
    J = fill(ℓ / 2, 3)
    W = ones(3)
    g0, g1 = portela_element_grads(W, ξ, J, w)
    @test g1[1] ≈ ℓ / 2 atol=1e-12
    @test g1[2] ≈ ℓ / 2 atol=1e-12
    @test g0[1] ≈ -ℓ / 2 atol=1e-12
    @test g0[2] ≈ -ℓ / 2 atol=1e-12
    @test isfinite(g0[1]) && isfinite(g0[2])
end

@testset "prepare_design_dual! tags free Neumann as HBIE" begin
    d = portela_heat_hole(; a=0.3, ne=4, nint=4, degree=1)
    dad = bemdata_from_loops(d)
    prepare_design_dual!(dad, d)
    @test has_cache(dad, :eq_type)
    @test count(==(3), dad.eq_type) > 0
    @test count(==(1), dad.eq_type) > 0
    H_G_full_direct(dad; npg=8, threaded=false)
    solve(dad)
    @test all(isfinite, dad.T)
    @test thermal_conductance(dad) > 0
    W = strain_energy_density(dad)
    @test length(W) == dad.n
    @test all(x -> isfinite(x) && x ≥ 0, W)
end

@testset "Portela heat hole: finite, no fold, area near A0" begin
    d = portela_heat_hole(; a=0.3, ne=4, nint=4, degree=1)
    A0 = design_area(d)
    opt = PortelaOptions(maxiter=3, npg=8, state=:cbie, param=:radial,
        origin=SVector(0.5, 0.5), vmax=0.03, α=0.25, verbose=false, nsearch=3)
    d, dad, hist = solve_portela!(d, opt)
    A = design_area(d)
    @test isfinite(thermal_conductance(dad))
    @test !BEM.Topology._design_folded(d)
    @test 0.80 * A0 < A < 1.15 * A0
    @test !isempty(hist.area)
end

@testset "Portela dual HBIE on insulated hole is finite" begin
    d = portela_heat_hole(; a=0.3, ne=4, nint=4, degree=1)
    dad = bemdata_from_loops(d)
    prepare_design_dual!(dad, d)
    @test count(==(3), dad.eq_type) > 0
    BEM.Crack.assemble_dual_laplace!(dad; npg=8, threaded=false)
    solve(dad)
    @test all(isfinite, dad.T)
    @test isfinite(thermal_conductance(dad))
end

@testset "3D DT of T=z is ~3/2" begin
    include(joinpath(@__DIR__, "..", "data", "Laplace", "cube_mesh.jl"))
    msh = mesh_cube(; L=1.0, ndiv=2, nome="t_topo3d_z")
    dad = format3d(msh, Laplace(1.0); pontointerno=false)
    set_internal_nodes!(dad, vec(cube_interior_grid(1.0, 2)))
    assemble!(dad; npg=8, threaded=false)
    solve(dad)
    gb = boundary_grad_T(dad)
    gi = interior_grad_T(dad)
    @test median(getindex.(gb, 3)) ≈ 1.0 atol=0.15
    @test median(getindex.(gi, 3)) ≈ 1.0 atol=0.25
    DTb, DTi = topological_derivative(dad)
    @test median(DTb) ≈ 1.5 atol=0.45
    @test median(DTi) ≈ 1.5 atol=0.55
    @test thermal_conductance(dad) > 0
end

@testset "3D DIBEM-SIMP K=1 matches homogeneous" begin
    dad = heat_cube_3d(; ndiv=2, nint=2, degree=1)
    assemble!(dad; npg=8, threaded=false)
    solve(dad)
    C0 = thermal_conductance(dad)
    dad2 = heat_cube_3d(; ndiv=2, nint=2, degree=1)
    assemble!(dad2; npg=8, threaded=false)
    ρ = ones(dad2.nt)
    opt = DibemSimpOptions(p=3.0, K0=1.0, Kmin=1e-3, rbf=PHS(1; poly_deg=-1))
    K = simp_conductivity(ρ, opt)
    solve_heterogeneous!(dad2, K; rbf=opt.rbf)
    @test all(isfinite, dad2.T)
    @test abs(thermal_conductance(dad2) - C0) / max(abs(C0), 1e-12) < 0.35
end

@testset "3D DIBEM-SIMP heat one OC step is finite" begin
    dad = heat_cube_3d(; ndiv=2, nint=2, degree=1)
    assemble!(dad; npg=8, threaded=false)
    opt = DibemSimpOptions(volfrac=0.5, rmin=0.25, n_simp=1, cut=false, pacheco=false,
        verbose=false, npg=8)
    w = dibem_volume_weights(dad, opt.rbf)
    ρ = fill(opt.volfrac, dad.nt)
    st = dibem_simp_step!(dad, ρ, w, opt)
    @test isfinite(st.C)
    @test 0.15 < st.V < 0.85
    @test all(x -> 0 < x ≤ 1, ρ)
end

@testset "3D elastic heterogeneous E=E0 matches homogeneous" begin
    include(joinpath(@__DIR__, "..", "data", "Laplace", "cube_mesh.jl"))
    props = Elasticity(1.0, 0.3, 1.0; plane_strain=true)
    msh = mesh_unit_cube(; L=1.0, ndiv=2, nome="t_topo3d_el", bc="0;0;0;0;0;0")
    dad = format3d(msh, props; pontointerno=false)
    set_internal_nodes!(dad, vec(cube_interior_grid(1.0, 2)))
    ana = ana_elasticity_patch(; E=1.0, ν=0.3, εxx=0.01, dim=3)
    neumann = [i for i in 1:dad.n if abs(dad.Normal[i][1]) < 0.5]
    apply_analytical_bc!(dad, ana, neumann)
    assemble!(dad; npg=8, threaded=false)
    solve(dad)
    J0 = elastic_compliance(dad)
    dad2 = format3d(msh, props; pontointerno=false)
    set_internal_nodes!(dad2, vec(cube_interior_grid(1.0, 2)))
    apply_analytical_bc!(dad2, ana, neumann)
    assemble!(dad2; npg=8, threaded=false)
    solve_heterogeneous!(dad2, fill(dad2.properties.E, dad2.nt))
    J1 = elastic_compliance(dad2)
    @test all(isfinite, dad2.u)
    @test abs(J1 - J0) / max(abs(J0), 1e-12) < 0.08
    σi, _ = interior_stress_strain(dad)
    @test !isempty(σi)
    @test all(isfinite, σi)
end

@testset "3D DIBEM-SIMP elasticity one OC step is finite" begin
    dad = cantilever_cube_3d(; ndiv=2, nint=2, degree=1, E=1.0, ν=0.3)
    assemble!(dad; npg=8, threaded=false)
    opt = DibemSimpOptions(volfrac=0.4, rmin=0.4, n_simp=1, cut=false, pacheco=false,
        verbose=false, npg=8)
    w = dibem_volume_weights(dad, opt.rbf)
    ρ = fill(opt.volfrac, dad.nt)
    st = dibem_simp_step!(dad, ρ, w, opt)
    @test isfinite(st.C)
    @test st.C > 0
    @test 0.1 < st.V < 0.8
    @test all(x -> 0 < x ≤ 1, ρ)
end

@testset "3D DT-ρ one shot is finite and hits volume" begin
    dad = heat_cube_3d(; ndiv=2, nint=2, degree=1)
    assemble!(dad; npg=8, threaded=false)
    solve(dad)
    w = dibem_volume_weights(dad, PHS(1; poly_deg=-1))
    frozen = BEM.Topology._freeze_design_nodes(dad)
    DT = nodal_DT(dad)
    ρ = density_from_dt(DT, w, 0.5, frozen)
    @test all(x -> 0 < x ≤ 1, ρ)
    @test 0.2 < dot(ρ, w) / sum(w) < 0.8
    @test maximum(DT) > minimum(DT)
end

@testset "marching cubes of a ball is a closed-ish surface" begin
    n = 17
    xs = range(-1.0, 1.0; length=n)
    ys = copy(xs)
    zs = copy(xs)
    R = 0.55
    Z = [R - sqrt(x^2 + y^2 + z^2) for x in xs, y in ys, z in zs]
    tris = marching_cubes(xs, ys, zs, Z, 0.0)
    @test length(tris) ≥ 50
    @test all(t -> all(isfinite, t[1]) && all(isfinite, t[2]) && all(isfinite, t[3]), tris)
    V = grid_solid_volume(xs, ys, zs, Z, 0.0)
    @test 0.4 < V < 1.2   # 4/3 π R³ ≈ 0.70
end

@testset "3D iso-cut caches triangles" begin
    dad = heat_cube_3d(; ndiv=2, nint=2, degree=1)
    assemble!(dad; npg=8, threaded=false)
    solve(dad)
    ρ = fill(0.05, dad.nt)
    for i in 1:dad.nt
        point(dad, i)[1] < 0.5 && (ρ[i] = 1.0)
    end
    opt = DibemSimpOptions(volfrac=0.5, ngrid=13, cut=true, match_area=false,
        ρ_cut=0.4, rmin=0.0, verbose=false, β_end=0.0)
    tris, ρh, lev = cut_density_3d!(dad, ρ, opt)
    @test 0.12 ≤ lev ≤ 0.78
    @test length(tris) ≥ 1
    @test has_cache(dad, :simp_iso)
    vtk = tempname() * ".vtk"
    export_vtk_isosurface(tris, vtk)
    @test isfile(vtk)
    rm(vtk; force=true)
end

@testset "3D closed ball iso is one cavity component" begin
    n = 15
    xs = range(0.0, 1.0; length=n)
    Z = [0.18 - sqrt((x - 0.5)^2 + (y - 0.5)^2 + (z - 0.5)^2) for x in xs, y in xs, z in xs]
    tris = marching_cubes(xs, xs, xs, Z, 0.0)
    dad = heat_cube_3d(; ndiv=2, nint=2, degree=1)
    cavs = extract_closed_cavities(tris, dad; min_area=0.05, min_dist=0.05)
    @test length(cavs) == 1
    @test cavs[1].area > 0.05
end

@testset "3D bemdata_from_iso inserts a cubic cavity" begin
    dad = heat_cube_3d(; ndiv=2, nint=2, degree=1)
    n0 = dad.n
    ne0 = length(dad.elements)
    lo, hi = 0.35, 0.65
    corners = (
        Point3D(lo, lo, lo), Point3D(hi, lo, lo), Point3D(hi, hi, lo), Point3D(lo, hi, lo),
        Point3D(lo, lo, hi), Point3D(hi, lo, hi), Point3D(hi, hi, hi), Point3D(lo, hi, hi),
    )
    # 12 triangles, outward for the cube (will be flipped to hole orientation)
    faces = (
        (1, 3, 2), (1, 4, 3), (5, 6, 7), (5, 7, 8),
        (1, 2, 6), (1, 6, 5), (4, 8, 7), (4, 7, 3),
        (1, 5, 8), (1, 8, 4), (2, 3, 7), (2, 7, 6),
    )
    tris = NTuple{3,Point3D}[(corners[a], corners[b], corners[c]) for (a, b, c) in faces]
    dad2, nadd = bemdata_from_iso(dad, tris; min_area=1e-4, min_dist=0.05)
    @test nadd == 1
    @test dad2.n > n0
    @test length(dad2.elements) == ne0 + 12
    @test dad2.n == n0 + 12 * 4   # collapsed quad, tipo=1 → 4 colloc / tri
    # new nodes are Neumann
    dof = 1
    @test all(==(1), dad2.BC[(dof * n0 + 1):end])
    assemble!(dad2; npg=6, threaded=false)
    solve(dad2)
    @test all(isfinite, dad2.T)
    @test thermal_conductance(dad2) > 0
    @test has_cache(dad2, :n_cavities)
    @test dad2.n_cavities == 1
end

@testset "3D open iso (half-space) adds no cavity" begin
    dad = heat_cube_3d(; ndiv=2, nint=2, degree=1)
    ρ = fill(0.05, dad.nt)
    for i in 1:dad.nt
        point(dad, i)[1] < 0.5 && (ρ[i] = 1.0)
    end
    opt = DibemSimpOptions(ngrid=13, match_area=false, ρ_cut=0.4, rmin=0.0,
        β_end=0.0, verbose=false)
    tris, _, _ = cut_density_3d!(dad, ρ, opt)
    _, nadd = bemdata_from_iso(dad, tris; min_area=1e-4, min_dist=0.05)
    @test nadd == 0
end

@testset "Portela plate hole: finite, no fold, area near A0" begin
    d = portela_plate_hole(; ratio=1.0, ne_outer=3, ne_hole=3, nint=4, degree=1, E=1.0)
    A0 = design_area(d)
    opt = PortelaOptions(maxiter=3, npg=8, state=:cbie, param=:radial,
        origin=SVector(0.0, 0.0), vmax=0.02, α=0.25, verbose=false, nsearch=3)
    d, dad, hist = solve_portela!(d, opt)
    J = elastic_compliance(dad)
    A = design_area(d)
    @test isfinite(J)
    @test J > 0
    @test !BEM.Topology._design_folded(d)
    @test 0.80 * A0 < A < 1.15 * A0
end
