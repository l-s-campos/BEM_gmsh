# Constant-cell elastoplasticity: kernels, patch, uniaxial, thick cylinder.
using Test
using LinearAlgebra
using StaticArrays
using BEM

_vmises_q(σx, σy, τ) = sqrt(max(σx * σx - σx * σy + σy * σy + 3 * τ * τ, 0.0))

@testset "initial-stress kernels match ∂U" begin
    props = Elasticity(2.1, 0.3, 1.0; plane_strain=true)
    r = SVector(0.35, -0.22)
    n = SVector(1.0, 0.0)
    Ux, _, Uy, _ = fundamental_grad(props, r, n)
    E = initial_strain_kernel(props, r)
    @test E[1, 1] ≈ Ux[1, 1] rtol=1e-10
    @test E[2, 1] ≈ Ux[2, 1] rtol=1e-10
    @test E[1, 2] ≈ Uy[1, 2] rtol=1e-10
    @test E[2, 2] ≈ Uy[2, 2] rtol=1e-10
    @test E[1, 3] ≈ Uy[1, 1] + Ux[1, 2] rtol=1e-10
    @test E[2, 3] ≈ Uy[2, 1] + Ux[2, 2] rtol=1e-10
    F = initial_stress_free_term(props)
    @test F[1, 3] == 0 && F[3, 1] == 0
    @test F[3, 3] ≈ -1 / (4 * (1 - 0.3))
end

@testset "cell stress op is d(displacement BIE)/dp" begin
    props = Elasticity(1.0, 0.3, 1.0; plane_strain=true)
    verts = [SVector(0.0, 0.0), SVector(1.0, 0.0), SVector(1.0, 1.0), SVector(0.0, 1.0)]
    cell = DomainCell(verts, SVector(0.5, 0.5), 1.0)
    Ein = cell_integral_Estress(props, cell.centroid, cell; npg=20, free_term=true)
    Gin = cell_integral_Estress_gao(props, cell.centroid, cell; npg=20, free_term=true)
    @test norm(Ein - Gin) / (norm(Gin) + 1e-16) < 1e-8
    xout = SVector(2.0, 0.5)
    Eout = cell_integral_Estress(props, xout, cell; npg=20, free_term=false)
    Gout = cell_integral_Estress_gao(props, xout, cell; npg=20, free_term=false)
    @test norm(Eout - Gout) / (norm(Gout) + 1e-16) < 1e-8
    h = 1e-6
    Q = cell_integral_Estrain(props, cell.centroid, cell; npg=20)
    Qx = cell_integral_Estrain(props, cell.centroid + SVector(h, 0.0), cell; npg=20)
    Qy = cell_integral_Estrain(props, cell.centroid + SVector(0.0, h), cell; npg=20)
    dQdx = (Qx - Q) / h
    dQdy = (Qy - Q) / h
    M = @SMatrix [
        dQdx[1, 1] dQdx[1, 2] dQdx[1, 3]
        dQdy[2, 1] dQdy[2, 2] dQdy[2, 3]
        (dQdy[1, 1] + dQdx[2, 1]) (dQdy[1, 2] + dQdx[2, 2]) (dQdy[1, 3] + dQdx[2, 3])
    ]
    Efd = inplane_stiffness(props) * M - I
    @test norm(Ein - Efd) / (norm(Efd) + 1e-16) < 1e-6
end

@testset "σp=0 recovers elastic solve" begin
    props = Elasticity(1.0, 0.3, 1.0; plane_stress=true)
    dad = format2d(quadrado_elasticity(ndiv=4, show=false, nome="t_pl_el"), props;
        pontointerno=true, tipo=1)
    assemble!(dad, 8)
    solve(dad)
    u0 = copy(dad.u)
    mat = VonMises(σY=1e6, H′=0.0)
    solve_elastoplastic!(dad, mat; nsteps=1, maxiter=3, tol=1e-8, npg=8,
        npg_stress=10, threaded=false)
    @test norm(dad.u - u0) / (norm(u0) + 1e-16) < 1e-8
    @test maximum(dad.plastic_strain) < 1e-14
end

@testset "uniform eigenstrain → near-zero stress" begin
    props = Elasticity(1.0, 0.25, 1.0; plane_stress=true)
    dad = format2d(quadrado_elasticity(ndiv=4, show=false, nome="t_pl_patch"), props;
        pontointerno=true, tipo=1)
    for i in 1:dad.n
        p = dad.Nodes[i]
        dad.BC[2i-1] = 1
        dad.BC[2i] = 1
        dad.BV[2i-1] = 0.0
        dad.BV[2i] = 0.0
        if p[1] < 1e-8
            dad.BC[2i-1] = 0
            dad.BV[2i-1] = 0.0
        end
        if p[2] < 1e-8
            dad.BC[2i] = 0
            dad.BV[2i] = 0.0
        end
    end
    assemble!(dad, 8)
    assemble_plastic_ops!(dad; npg=8, npg_stress=10, threaded=false)
    applyBC(dad)
    C = inplane_stiffness(props)
    εp = SVector(0.01, 0.0, 0.0)
    σp = C * εp
    nc = length(extract_domain_cells(dad))
    σp_vec = repeat(collect(σp), nc)
    x = dad.A \ (dad.b + dad.plastic_Q * σp_vec)
    u = zeros(2 * dad.n)
    t = zeros(2 * dad.n)
    split_sol!(dad, view(x, 1:2*dad.n), u, t)
    σB = dad.plastic_Su * u + dad.plastic_St * t + dad.plastic_Sσ * σp_vec
    @test maximum(abs, σB) < 0.15 * props.E * 0.01
    ux = [u[2i-1] for i in 1:dad.n]
    xs = [dad.Nodes[i][1] for i in 1:dad.n]
    @test cor(ux, xs) > 0.9
end

@testset "DIBEM plasticity remainder and elastic limit" begin
    props = Elasticity(1.0, 0.3, 1.0; plane_stress=true)
    dad = format2d(quadrado_elasticity(ndiv=4, show=false, nome="t_pl_dibem"), props;
        pontointerno=true, tipo=1)
    assemble!(dad, 8)
    assemble_plastic_ops!(dad; domain=:dibem, npg=8, npg_stress=10, threaded=false)
    nc = dad.plastic_nc
    e1 = zeros(3nc)
    @inbounds for k in 1:nc
        e1[3k-2] = 1.0
    end
    @test dad.ni == dad.plastic_nc > 0
    uQ = dad.plastic_Q * e1
    nb = 2 * dad.n
    IE = vcat((BEM._boundary_integral_Estrain(dad, point(dad, i); npg=8)[:, 1] for i in 1:dad.n)...)
    @test norm(uQ[1:nb] - IE) / (norm(IE) + 1e-16) < 1e-8
    solve(dad)
    u0 = copy(dad.u)
    solve_elastoplastic!(dad, VonMises(σY=1e6); nsteps=1, maxiter=2, tol=1e-8,
        npg=8, npg_stress=10, threaded=false, domain=:dibem)
    @test norm(dad.u - u0) / (norm(u0) + 1e-16) < 1e-8
end

@testset "uniaxial bar yields above σY" begin
    props = Elasticity(100.0, 0.25, 1.0; plane_stress=true)
    dad = format2d(quadrado_elasticity(ndiv=4, Lx=2.0, Ly=1.0, show=false,
            nome="t_pl_bar"), props; pontointerno=true, tipo=1)
    σY = 1.0
    function _bar_bc!(d, tx)
        for i in 1:d.n
            p = d.Nodes[i]
            d.BC[2i-1] = 1
            d.BC[2i] = 1
            d.BV[2i-1] = 0.0
            d.BV[2i] = 0.0
            if p[1] < 1e-8
                d.BC[2i-1] = 0
                d.BV[2i-1] = 0.0
            end
            if p[2] < 1e-8
                d.BC[2i] = 0
                d.BV[2i] = 0.0
            end
            if p[1] > 2.0 - 1e-8
                d.BV[2i-1] = tx
            end
        end
        return d
    end
    _bar_bc!(dad, 0.4 * σY)
    assemble!(dad, 8)
    mat = VonMises(σY=σY, H′=0.0)
    solve_elastoplastic!(dad, mat; nsteps=2, maxiter=15, tol=1e-4, npg=8,
        npg_stress=10, threaded=false)
    @test maximum(dad.plastic_strain) < 1e-10

    dad2 = format2d(quadrado_elasticity(ndiv=4, Lx=2.0, Ly=1.0, show=false,
            nome="t_pl_bar2"), props; pontointerno=true, tipo=1)
    _bar_bc!(dad2, 1.3 * σY)
    assemble!(dad2, 8)
    solve_elastoplastic!(dad2, mat; nsteps=4, maxiter=40, tol=1e-4, npg=8,
        npg_stress=10, threaded=false, inner=:broyden, relax=0.4)
    @test maximum(dad2.plastic_strain) > 0
    @test all(isfinite, dad2.stress)
    q = [_vmises_q(dad2.stress[k, :]...) for k in 1:size(dad2.stress, 1)]
    @test median(q) < 1.15 * σY
end

@testset "thick cylinder elastic then plastic" begin
    include(joinpath(@__DIR__, "..", "data", "elastico", "iso", "pressurized_tube.jl"))
    a, b = 50.0, 100.0
    E, ν, σY = 200_000.0, 0.3, 240.0
    props = Elasticity(E, ν, 1.0; plane_strain=true)
    k = σY / sqrt(3)
    pel = k * (1 - (a / b)^2)
    msh = mesh_pressurized_tube(; ndiv=6, nome="t_pl_cyl")
    dad = format2d(msh, props; pontointerno=true, tipo=1)
    apply_radius_pressure!(dad, 0.6 * pel; R=a, tol=4.0)
    assemble!(dad, 8)
    mat = VonMises(σY=σY, H′=0.0)
    solve_elastoplastic!(dad, mat; nsteps=2, maxiter=8, tol=1e-4, npg=8,
        npg_stress=10, threaded=false)
    @test maximum(dad.plastic_strain) < 1e-8
    ana = ana_thick_cylinder_plastic(b; a=a, b=b, p=0.6 * pel, σY=σY, E=E, ν=ν)
    @test ana.elastic
    ub = [hypot(dad.u[2i-1], dad.u[2i]) for i in 1:dad.n
          if abs(hypot(dad.Nodes[i][1], dad.Nodes[i][2]) - b) < 4.0]
    !isempty(ub) && @test median(ub) / abs(ana.u) ≈ 1 atol=0.35

    dadp = format2d(msh, props; pontointerno=true, tipo=1)
    apply_radius_pressure!(dadp, 1.35 * pel; R=a, tol=4.0)
    assemble!(dadp, 8)
    solve_elastoplastic!(dadp, mat; nsteps=4, maxiter=20, tol=1e-3, npg=8,
        npg_stress=10, threaded=false)
    @test maximum(dadp.plastic_strain) > 0
    @test all(isfinite, dadp.u)
    @test all(isfinite, dadp.stress)
end
