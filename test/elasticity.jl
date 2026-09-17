# Elasticity: patch test + local (n,t) frame.
using Test
using LinearAlgebra
using StaticArrays
using BEM

const HAVE_CUDA = try
    using CUDA
    CUDA.functional()
catch
    false
end

@testset "Kelvin HBIE interp vs Richardson H" begin
    mk(name) = format2d(quadrado_elasticity(ndiv=4, show=false, nome=name),
        Elasticity(1.0, 0.3, 1.0; plane_strain=true); pontointerno=false, tipo=1)
    Hi, Gi = H_G_hyper(mk("t_ehbie_i"); npg=8, threaded=false, laurent=:interp)
    Hr, Gr = H_G_hyper(mk("t_ehbie_r"); npg=8, threaded=false, laurent=:richardson)
    Ha, Ga = H_G_hyper(mk("t_ehbie_a"); npg=8, threaded=false, laurent=:auto)
    @test norm(Hi - Hr) / max(norm(Hr), 1e-16) < 5e-4
    @test norm(Hi - Ha) / max(norm(Ha), 1e-16) < 5e-4
    @test norm(Gi - Gr) / max(norm(Gr), 1e-16) < 5e-4
    @test all(isfinite, Hi) && all(isfinite, Hr)
end

@testset "elasticity patch" begin
    props = Elasticity(1.0, 0.3, 1.0)
    dad = format2d(quadrado_elasticity(ndiv=8, show=false, nome="t_el_p"), props;
        pontointerno=false)
    ana = ana_elasticity_patch(; E=1.0, ν=0.3, εxx=0.01)
    apply_analytical_bc!(dad, ana)
    assemble!(dad, 12)
    solve(dad)
    @test rel_error(dad) < 0.15
end

@testset "cube 3D elasticity patch" begin
    include(joinpath(@__DIR__, "..", "data", "Laplace", "cube_mesh.jl"))
    props = Elasticity(1.0, 0.3, 1.0; plane_strain=true)
    msh = mesh_unit_cube(; L=1.0, ndiv=2, nome="t_el3d", bc="0;0;0;0;0;0")
    dad = format3d(msh, props; pontointerno=false)
    ana = ana_elasticity_patch(; E=1.0, ν=0.3, εxx=0.01, dim=3)
    neumann = [i for i in 1:dad.n if abs(dad.Normal[i][1]) < 0.5]
    apply_analytical_bc!(dad, ana, neumann)
    assemble!(dad; npg=8, threaded=false)
    solve(dad)
    @test rel_error(dad) < 1e-3
    @test rel_error_flux(dad) < 1e-3
end

@testset "cube 3D elasticity body force" begin
    include(joinpath(@__DIR__, "..", "data", "Laplace", "cube_mesh.jl"))
    props = Elasticity(1.0, 0.3, 1.0; plane_strain=true)
    λ, μ = props.lambda, props.mu
    bval = -2 * (λ + 2μ)   # Navier of u = (x², 0, 0)
    msh = mesh_unit_cube(; L=1.0, ndiv=2, nome="t_el3dbf", bc="0;0;0;0;0;0")
    dad = format3d(msh, props; pontointerno=false)
    set_internal_nodes!(dad, vec(cube_interior_grid(1.0, 2)))
    for i in 1:dad.n
        p = dad.Nodes[i]
        dad.BC[3*(i-1)+1:3*i] .= 0
        dad.BV[3*(i-1)+1] = p[1]^2
        dad.BV[3*(i-1)+2] = 0.0
        dad.BV[3*(i-1)+3] = 0.0
    end
    assemble!(dad; npg=8, threaded=false)
    DIBEM(dad; rbf=PHS(3; poly_deg=1), npg=8)
    solve_thermoelastic!(dad; bodyforce=p -> SVector(bval, 0.0, 0.0), θ=0.0)
    @test length(dad.uint) == 3 * dad.ni
    err = 0.0
    den = 0.0
    for i in 1:dad.ni
        p = dad.internalNodes[i]
        ui = dad.uint[3*(i-1)+1:3*i]
        ue = SA[p[1]^2, 0.0, 0.0]
        err += sum(abs2, ui .- ue)
        den += sum(abs2, ue)
    end
    @test sqrt(err / max(den, eps())) < 0.15
end

@testset "cube 3D DIBEM hmatrix elasticity" begin
    include(joinpath(@__DIR__, "..", "data", "Laplace", "cube_mesh.jl"))
    props = Elasticity(1.0, 0.3, 1.0; plane_strain=true)
    msh = mesh_unit_cube(; L=1.0, ndiv=2, nome="t_el3dh", bc="0;0;0;0;0;0")
    dad = format3d(msh, props; pontointerno=false)
    set_internal_nodes!(dad, vec(cube_interior_grid(1.0, 2)))
    assemble!(dad; npg=8, threaded=false)
    rbf = PHS(3; poly_deg=1)
    Md = DIBEM(deepcopy(dad); method=:dense, rbf=rbf, npg=8)
    Mh = DIBEM(dad; method=:hmatrix, rbf=rbf, npg=8, nmax=16, atol=1e-5,
        rtol=1e-5, threads=false)
    ndof = 3 * dad.nt
    @test size(Mh) == (ndof, ndof)
    x = randn(ndof)
    @test norm(Mh * x - Md * x) / (norm(Md * x) + 1e-14) < 0.08
    @test eltype(dad.dibem_D) <: SMatrix
end

@testset "cube 3D DIBEM h2 elasticity" begin
    include(joinpath(@__DIR__, "..", "data", "Laplace", "cube_mesh.jl"))
    props = Elasticity(1.0, 0.3, 1.0; plane_strain=true)
    msh = mesh_unit_cube(; L=1.0, ndiv=2, nome="t_el3dh2", bc="0;0;0;0;0;0")
    dad = format3d(msh, props; pontointerno=false)
    set_internal_nodes!(dad, vec(cube_interior_grid(1.0, 2)))
    assemble!(dad; npg=8, threaded=false)
    rbf = PHS(3; poly_deg=1)
    Md = DIBEM(deepcopy(dad); method=:dense, rbf=rbf, npg=8)
    M2 = DIBEM(dad; method=:h2, rbf=rbf, npg=8, nmax=16, rtol=1e-5, threads=false)
    ndof = 3 * dad.nt
    @test size(M2) == (ndof, ndof)
    @test dad.dibem_D isa BEM.HMatrices.NNCAMatrix
    x = randn(ndof)
    @test norm(M2 * x - Md * x) / (norm(Md * x) + 1e-14) < 0.08
end

@testset "cube 3D DIBEM FMM elasticity" begin
    include(joinpath(@__DIR__, "..", "data", "Laplace", "cube_mesh.jl"))
    props = Elasticity(1.0, 0.3, 1.0; plane_strain=true)
    msh = mesh_unit_cube(; L=1.0, ndiv=2, nome="t_el3df")
    dad = format3d(msh, props; pontointerno=false)
    set_internal_nodes!(dad, vec(cube_interior_grid(1.0, 2)))
    assemble!(dad; npg=8, threaded=false)
    rbf = PHS(3; poly_deg=1)
    Md = DIBEM(deepcopy(dad); method=:dense, rbf=rbf, npg=8)
    Mf = DIBEM(dad; method=:fmm, rbf=rbf, npg=8, nmax=16, eps=1e-6)
    @test dad.dibem_D isa BEM.FMM.KelvinFMMMatrix3D
    ndof = 3 * dad.nt
    x = randn(ndof)
    @test norm(Mf * x - Md * x) / (norm(Md * x) + 1e-14) < 0.05
end

@testset "local (n,t) rotation" begin
    n̂, t̂ = local_basis2d(Point2D(3.0, 4.0))
    @test n̂ ≈ Point2D(0.6, 0.8) atol=1e-14
    @test t̂ ≈ Point2D(-0.8, 0.6) atol=1e-14
    R = node_rotation2d(Point2D(0.0, 1.0))
    @test R ≈ [0.0 -1.0; 1.0 0.0] atol=1e-14
end

@testset "thermoelasticity constrained" begin
    E, ν, α, Δθ = 1000.0, 0.3, 1e-5, 50.0
    props = Elasticity(E, ν, 1.0; plane_strain=true, α=α)
    k̂ = thermal_modulus(props)
    @test analytical_constrained_thermal_stress(props, Δθ) ≈ -k̂ * Δθ
    dad = format2d(quadrado_elasticity(ndiv=6, show=false, nome="t_th"), props;
        pontointerno=false)
    fill!(dad.BC, 0)
    fill!(dad.BV, 0.0)
    assemble!(dad; npg=10, threaded=false)
    u = solve_thermoelastic!(dad; θ=Δθ)
    @test maximum(abs, u) < 1e-8
end

@testset "3D anisotropic FS isotropic limit" begin
    E, ν = 1.0, 0.3
    kel = Elasticity(E, ν, 1.0; plane_strain=true)
    ani = aniso3d_isotropic(E, ν; nψ=64)
    r = SVector(0.3, -0.2, 0.5)
    n = SVector(0.0, 0.0, 1.0)
    Uk, Tk = fundamental(kel, r, n)
    Ua, Ta = fundamental(ani, r, n)
    @test norm(Ua - Uk) / norm(Uk) < 1e-6
    @test norm(Ta - Tk) / norm(Tk) < 5e-5
end

@testset "cube 3D elasticity triangles" begin
    include(joinpath(@__DIR__, "..", "data", "Laplace", "cube_mesh.jl"))
    props = Elasticity(1.0, 0.3, 1.0; plane_strain=true)
    msh = mesh_unit_cube(; L=1.0, ndiv=2, nome="t_el3d_tri", bc="0;0;0;0;0;0",
        recombine=false)
    dad = format3d(msh, props; pontointerno=false)
    ana = ana_elasticity_patch(; E=1.0, ν=0.3, εxx=0.01, dim=3)
    neumann = [i for i in 1:dad.n if abs(dad.Normal[i][1]) < 0.5]
    apply_analytical_bc!(dad, ana, neumann)
    assemble!(dad; npg=8, threaded=false)
    solve(dad)
    @test rel_error(dad) < 0.15
end

@testset "cube 3D anisotropic patch" begin
    include(joinpath(@__DIR__, "..", "data", "Laplace", "cube_mesh.jl"))
    props = aniso3d_cubic(230.0, 135.0, 117.0; nψ=24)
    msh = mesh_unit_cube(; L=1.0, ndiv=1, nome="t_el3d_ani", bc="0;0;0;0;0;0")
    dad = format3d(msh, props; pontointerno=false)
    ana = ana_aniso3d_patch(props; εxx=0.01)
    neumann = [i for i in 1:dad.n if abs(dad.Normal[i][1]) < 0.5]
    apply_analytical_bc!(dad, ana, neumann)
    assemble!(dad; npg=4, threaded=false)
    solve(dad)
    @test rel_error(dad) < 5e-2
end

@testset "Kane strain-stress 2D/3D patch" begin
    props2 = Elasticity(1.0, 0.3, 1.0)
    dad2 = format2d(quadrado_elasticity(ndiv=8, show=false, nome="t_ss2"), props2;
        pontointerno=false)
    ana2 = ana_elasticity_patch(; E=1.0, ν=0.3, εxx=0.01)
    apply_analytical_bc!(dad2, ana2)
    assemble!(dad2, 12)
    solve(dad2)
    ε2, σ2 = recover_strain_stress!(dad2)
    @test mean(abs.(ε2[:, 1] .- 0.01)) < 0.05
    λ, μ = props2.lambda, props2.mu
    σxx = (λ + 2μ) * 0.01
    @test mean(abs.(σ2[:, 1] .- σxx)) / abs(σxx) < 0.08

    include(joinpath(@__DIR__, "..", "data", "Laplace", "cube_mesh.jl"))
    props3 = Elasticity(1.0, 0.3, 1.0; plane_strain=true)
    msh = mesh_unit_cube(; L=1.0, ndiv=2, nome="t_ss3", bc="0;0;0;0;0;0")
    dad3 = format3d(msh, props3; pontointerno=false)
    ana3 = ana_elasticity_patch(; E=1.0, ν=0.3, εxx=0.01, dim=3)
    neumann = [i for i in 1:dad3.n if abs(dad3.Normal[i][1]) < 0.5]
    apply_analytical_bc!(dad3, ana3, neumann)
    assemble!(dad3; npg=8, threaded=false)
    solve(dad3)
    ε3, σ3 = recover_strain_stress!(dad3)
    @test mean(abs.(ε3[:, 1] .- 0.01)) < 0.05
    vtk = export_vtk(dad3, joinpath(tempdir(), "t_el3d.vtk"))
    txt = read(vtk, String)
    @test occursin("DATASET POLYDATA", txt)
    @test occursin("POINTS $(dad3.n)", txt)
    @test occursin("POLYGONS", txt)
    @test occursin("VECTORS u", txt)
    @test occursin("TENSORS strain", txt)
end

@testset "cube 3D DRM body force" begin
    include(joinpath(@__DIR__, "..", "data", "Laplace", "cube_mesh.jl"))
    props = Elasticity(1.0, 0.3, 1.0; plane_strain=true)
    λ, μ = props.lambda, props.mu
    bval = -2 * (λ + 2μ)
    msh = mesh_unit_cube(; L=1.0, ndiv=2, nome="t_el3ddrm", bc="0;0;0;0;0;0")
    dad = format3d(msh, props; pontointerno=false)
    set_internal_nodes!(dad, vec(cube_interior_grid(1.0, 2)))
    for i in 1:dad.n
        p = dad.Nodes[i]
        dad.BC[3*(i-1)+1:3*i] .= 0
        dad.BV[3*(i-1)+1] = p[1]^2
        dad.BV[3*(i-1)+2] = 0.0
        dad.BV[3*(i-1)+3] = 0.0
    end
    assemble!(dad; npg=8, threaded=false)
    build_drm_matrices(dad)
    solve_thermoelastic!(dad; bodyforce=p -> SVector(bval, 0.0, 0.0), θ=0.0)
    err = 0.0
    den = 0.0
    for i in 1:dad.ni
        p = dad.internalNodes[i]
        ui = dad.uint[3*(i-1)+1:3*i]
        ue = SA[p[1]^2, 0.0, 0.0]
        err += sum(abs2, ui .- ue)
        den += sum(abs2, ue)
    end
    @test sqrt(err / max(den, eps())) < 0.35
end

@testset "elasticity KA CPU kernel patch" begin
    props = Elasticity(1.0, 0.3, 1.0)
    dad = format2d(quadrado_elasticity(ndiv=8, show=false, nome="t_el_gpu_cpu"), props;
        pontointerno=false)
    dad_ref = format2d(quadrado_elasticity(ndiv=8, show=false, nome="t_el_gpu_ref"), props;
        pontointerno=false)
    ana = ana_elasticity_patch(; E=1.0, ν=0.3, εxx=0.01)
    apply_analytical_bc!(dad, ana)
    apply_analytical_bc!(dad_ref, ana)
    assemble!(dad_ref, 12; threaded=false)
    H_G_gpu(dad; T=Float64, npg=12, near=:cpu, device=:cpu, threaded=false)
    @test norm(dad.H - dad_ref.H) / norm(dad_ref.H) < 1e-10
    @test norm(dad.G - dad_ref.G) / norm(dad_ref.G) < 1e-10
    solve(dad)
    @test rel_error(dad) < 0.15
end

@testset "elasticity CUDA patch" begin
    if !HAVE_CUDA
        @info "CUDA not functional; skipping GPU elasticity assembly test"
    else
        props = Elasticity(1.0, 0.3, 1.0)
        dad = format2d(quadrado_elasticity(ndiv=8, show=false, nome="t_el_gpu"), props;
            pontointerno=false)
        dad_ref = format2d(quadrado_elasticity(ndiv=8, show=false, nome="t_el_gpu_d"), props;
            pontointerno=false)
        ana = ana_elasticity_patch(; E=1.0, ν=0.3, εxx=0.01)
        apply_analytical_bc!(dad, ana)
        apply_analytical_bc!(dad_ref, ana)
        assemble!(dad_ref, 12; threaded=false)
        assemble!(dad; method=:gpu, T=Float64, npg=12, near=:cpu, threaded=false)
        @test norm(dad.H - dad_ref.H) / norm(dad_ref.H) < 1e-10
        solve(dad)
        @test rel_error(dad) < 0.15
        rbf = PHS(3; poly_deg=1)
        Md = DIBEM(deepcopy(dad_ref); rbf=rbf, npg=12)
        Mg = DIBEM(dad; method=:gpu, rbf=rbf, T=Float64, npg=12)
        @test norm(Mg - Md) / norm(Md) < 1e-8
    end
end
