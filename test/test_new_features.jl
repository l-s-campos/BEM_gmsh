# Fast tests: geometric props, orthotropic Laplace, buckling, axisym FS, shell
using Test
using DrWatson
@quickactivate :BEM
using .ThinPlate
using LinearAlgebra
using StaticArrays

@testset "geometric properties 2D" begin
    # unit square via polygon helper (builds temporary dad + Equispaced shapes)
    verts = [SA[0.0, 0.0], SA[1.0, 0.0], SA[1.0, 1.0], SA[0.0, 1.0]]
    g = geometric_props_2d_polygon(verts; npg=16)
    @test g.perimeter ≈ 4.0 rtol=1e-3
    @test g.area ≈ 1.0 rtol=1e-2
    @test g.centroid ≈ SA[0.5, 0.5] rtol=1e-2

    # same via format2d BEMdata + geometric_props(dad)
    include(datadir("Laplace", "Laplace_dad.jl"))
    msh = quadrado(ndiv=8, show=false, nome="geo_sq")
    dad = format2d(msh, Laplace(1.0); pontointerno=false)
    g2 = geometric_props(dad)
    @test g2.perimeter ≈ 4.0 rtol=0.05
    @test g2.area ≈ 1.0 rtol=0.05
    @test g2.centroid ≈ SA[0.5, 0.5] atol=0.05
    # re-integrate with shapefun + denser Gauss
    g3 = geometric_props(dad; npg_boundary=12)
    @test g3.area ≈ 1.0 rtol=0.05
    @info "geo2d" g.area g2.area g3.area g2.centroid
end

@testset "geometric properties 3D cube" begin
    # unit cube — bilinear faces with explicit ∂r/∂ξ mapping (matches Input.jl style)
    # corners of each face in CCW order when viewed from outside
    faces_xyz = [
        # bottom z=0 outward -z
        (SA[0.0,0.0,0.0], SA[0.0,1.0,0.0], SA[1.0,1.0,0.0], SA[1.0,0.0,0.0]),
        # top z=1 outward +z
        (SA[0.0,0.0,1.0], SA[1.0,0.0,1.0], SA[1.0,1.0,1.0], SA[0.0,1.0,1.0]),
        # y=0 outward -y
        (SA[0.0,0.0,0.0], SA[1.0,0.0,0.0], SA[1.0,0.0,1.0], SA[0.0,0.0,1.0]),
        # y=1 outward +y
        (SA[1.0,1.0,0.0], SA[0.0,1.0,0.0], SA[0.0,1.0,1.0], SA[1.0,1.0,1.0]),
        # x=0 outward -x
        (SA[0.0,0.0,0.0], SA[0.0,0.0,1.0], SA[0.0,1.0,1.0], SA[0.0,1.0,0.0]),
        # x=1 outward +x
        (SA[1.0,0.0,0.0], SA[1.0,1.0,0.0], SA[1.0,1.0,1.0], SA[1.0,0.0,1.0]),
    ]
    poly = BEM.Equispaced(1)
    qsi, wi = gausslegendre(2)
    Nodes = Point3D[]
    Normal = Point3D[]
    elements = BEM.Element[]
    w2d = Float64[wi[i] * wi[j] for j in 1:2 for i in 1:2]
    for Xc in faces_xyz
        idx0 = length(Nodes) + 1
        Jvec = Float64[]
        for j in 1:2, i in 1:2
            ξ, η = qsi[i], qsi[j]
            # standard bilinear N on [-1,1]²
            N = SA[
                0.25(1-ξ)*(1-η), 0.25(1+ξ)*(1-η),
                0.25(1+ξ)*(1+η), 0.25(1-ξ)*(1+η),
            ]
            dNξ = SA[-0.25(1-η), 0.25(1-η), 0.25(1+η), -0.25(1+η)]
            dNη = SA[-0.25(1-ξ), -0.25(1+ξ), 0.25(1+ξ), 0.25(1-ξ)]
            x = sum(N[k] * Xc[k] for k in 1:4)
            dxξ = sum(dNξ[k] * Xc[k] for k in 1:4)
            dxη = sum(dNη[k] * Xc[k] for k in 1:4)
            nvec = cross(dxξ, dxη)
            J = norm(nvec)
            push!(Nodes, Point3D(x))
            push!(Normal, Point3D(nvec / J))
            push!(Jvec, J)
        end
        idx = collect(idx0:(idx0 + 3))
        push!(elements, BEM.Element(idx, Jvec, sum(Jvec .* w2d), 1))
    end
    dad3 = BEM.BEMdata(;
        name="cube", dimension=3, elements=elements, element_type=poly,
        elem_weight=SVector{4,Float64}(w2d),
        Nodes=Nodes, Normal=Normal, internalNodes=Point3D[],
        properties=Laplace(1.0), BC=ones(Int, length(Nodes)),
        BV=zeros(length(Nodes)), n=length(Nodes), ni=0, nt=length(Nodes),
    )
    g3 = geometric_props(dad3)
    @test g3.surface_area ≈ 6.0 rtol=0.05
    @test g3.volume ≈ 1.0 rtol=0.15
    @test g3.centroid ≈ SA[0.5, 0.5, 0.5] atol=0.05
    @info "geo3d" g3.surface_area g3.volume g3.centroid
end

@testset "orthotropic Laplace FS" begin
    p = OrthotropicLaplace(k1=2.0, k2=0.5)
    r = SA[0.3, 0.4]
    n = SA[1.0, 0.0]
    G, H = fundamental(p, r, n)
    @test isfinite(G) && isfinite(H)
    # isotropic limit matches Laplace
    p_iso = OrthotropicLaplace(k1=1.0, k2=1.0)
    Gi, Hi = fundamental(p_iso, r, n)
    G0, H0 = fundamental(Laplace(k=1.0), r, n)
    @test Gi ≈ G0 rtol=1e-10
    @test Hi ≈ H0 rtol=1e-10
end

@testset "orthotropic Laplace BEM square" begin
    include(datadir("Laplace", "Laplace_dad.jl"))
    msh = quadrado(ndiv=6, show=false, nome="test_orto")
    # k1=k2=1 should match isotropic
    dad = format2d(msh, OrthotropicLaplace(k1=1.0, k2=1.0); pontointerno=false)
    H_G_full_direct(dad; npg=10, threaded=false)
    applyBC(dad)
    x = dad.A \ dad.b
    # Dirichlet left T=0, right flux etc. — just check solve runs
    @test length(x) == dad.n
    @test all(isfinite, x)
end

@testset "axisymmetric fundamental" begin
    U, T = axisym_fundamental(1.0, 0.0, 1.2, 0.1, 1.0, 0.0, 1.0, 0.3)
    @test size(U) == (2, 2) && size(T) == (2, 2)
    @test all(isfinite, U) && all(isfinite, T)
    # on-axis source limit shouldn't explode for separated points
    U2, T2 = axisym_fundamental(0.05, 0.0, 1.0, 0.5, 0.0, 1.0, 210e3, 0.3)
    @test all(isfinite, U2)
end

@testset "plate buckling analytical k" begin
    D = 1.0
    a = 1.0
    Ncr = analytical_Ncr_ss_uniaxial(; a=a, D=D)
    @test Ncr ≈ 4π^2 rtol=1e-12
    # thermal force formula finite
    props = ThinPlateProps(E=1e5, ν=0.3, h=0.01, q_c=0.0)
    plate = build_square_plate(; a=a, n_el=3, bc="SSSS", props=props,
        corner_bc='F', n_internal=4)
    assemble_plate!(plate; npg=8)
    # compression reference Nxx = -1
    res = plate_buckling(plate; Nxx=-1.0, Nyy=0.0, nmodes=1, a=a)
    @test length(res.λ) >= 1
    @test res.λ[1] > 0
    @info "buckling" λ1=res.λ[1] k=res.k_factor[1] Ncr_ana=Ncr
end

@testset "thermal buckling API" begin
    props = ThinPlateProps(E=1e5, ν=0.3, h=0.01)
    plate = build_square_plate(; a=1.0, n_el=3, bc="SSSS", props=props,
        corner_bc='F', n_internal=4)
    assemble_plate!(plate; npg=8)
    res = thermal_buckling(plate; α=1e-5, ΔT=1.0, nmodes=1, a=1.0)
    @test length(res.λ) >= 1
    @test isfinite(res.λ[1])
end

@testset "shallow shell API" begin
    props = ThinPlateProps(E=1e5, ν=0.3, h=0.01, q_c=10.0)
    plate = build_square_plate(; a=1.0, n_el=3, bc="SSSS", props=props,
        corner_bc='F', n_internal=1)
    assemble_plate!(plate; npg=8)
    include(datadir("Laplace", "Laplace_dad.jl"))
    msh = quadrado_elasticity(ndiv=4, show=false, nome="shell_pe")
    dad = format2d(msh, Elasticity(1e5, 0.3, 1.0; plane_strain=false); pontointerno=false)
    fill!(dad.BC, 0); fill!(dad.BV, 0.0)
    shell = ShallowShell(plate, dad; R11=50.0, R22=50.0)
    assemble_shell_coupling(shell; npg=8)
    out = solve_shallow_shell!(shell; niter=4)
    @test isfinite(out.w_center)
    @info "shell" out.w_center
end
