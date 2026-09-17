# Leonardo (2026) §4.7 — local (n, t) elasticity BEM
using Test
using LinearAlgebra
using Statistics: mean
using BEM

include(datadir("Laplace", "Laplace_dad.jl"))

@testset "local basis / rotation" begin
    n̂, t̂ = local_basis2d(Point2D(3.0, 4.0))
    @test n̂ ≈ Point2D(0.6, 0.8) atol = 1e-14
    @test t̂ ≈ Point2D(-0.8, 0.6) atol = 1e-14
    @test abs(dot(n̂, t̂)) < 1e-14
    R = node_rotation2d(Point2D(0.0, 1.0))
    # n=(0,1), t=(-1,0) → R = [0 -1; 1 0]
    @test R ≈ [0.0 -1.0; 1.0 0.0] atol = 1e-14
    # u_g = R u_ℓ : un=1, ut=0 → u = n
    @test R * SVector(1.0, 0.0) ≈ SVector(0.0, 1.0) atol = 1e-14
end

@testset "HG local transform orthogonality" begin
    msh = quadrado_elasticity(ndiv=4, show=false, nome="loc_R")
    dad = format2d(msh, Elasticity(1.0, 0.3, 1.0; plane_strain=true); pontointerno=false)
    H_G_full_direct(dad, 10)
    H, G = dad.H, dad.G
    Hhat, Ghat = transform_HG_local(H, G, dad)
    R = rotation_block2d(dad)
    # Ĥ = H R  (boundary square)
    n2 = 2 * dad.n
    @test Hhat[1:n2, 1:n2] ≈ H[1:n2, 1:n2] * R rtol = 1e-12
    @test Ghat ≈ G * R rtol = 1e-12
    # R is block-orthogonal: R'R = I
    @test R' * R ≈ I(n2) atol = 1e-12
end

@testset "global vs local frame agreement" begin
    E, ν = 1.0, 0.3
    props = Elasticity(E, ν, 1.0; plane_strain=true)
    msh = quadrado_elasticity(ndiv=6, show=false, nome="loc_cmp")
    dad_g = format2d(msh, props; pontointerno=false)
    # Patch-like: left fixed, right uniform tx, other sides free
    # Override BCs node-wise from geometry
    for i in 1:dad_g.n
        x, y = dad_g.Nodes[i]
        if x < 1e-9
            dad_g.BC[2i-1] = 0; dad_g.BV[2i-1] = 0.0
            dad_g.BC[2i] = 0; dad_g.BV[2i] = 0.0
        elseif x > 1.0 - 1e-9
            dad_g.BC[2i-1] = 1; dad_g.BV[2i-1] = 1.0   # tx = 1
            dad_g.BC[2i] = 1; dad_g.BV[2i] = 0.0
        else
            dad_g.BC[2i-1] = 1; dad_g.BV[2i-1] = 0.0
            dad_g.BC[2i] = 1; dad_g.BV[2i] = 0.0
        end
    end
    H_G_full_direct(dad_g, 12)
    u_g = solve(dad_g; frame=:global)

    dad_l = deepcopy(dad_g)
    # clear solution cache but keep H,G
    bc_global_to_local!(dad_l)
    u_l = solve(dad_l; frame=:local)

    @test all(isfinite, u_g)
    @test all(isfinite, u_l)
    @test norm(u_l - u_g) / norm(u_g) < 5e-2
    # mean right-edge ux should be positive under tension
    ux_right = [u_g[2i-1] for i in 1:dad_g.n if dad_g.Nodes[i][1] > 1 - 1e-9]
    @test mean(ux_right) > 0
end

@testset "local roller BC (u_n = 0)" begin
    # Bottom rollers in local frame: u_n=0, t_t=0; top pressure t_n = -p
    E, ν = 100.0, 0.3
    props = Elasticity(E, ν, 1.0; plane_strain=true)
    msh = quadrado_elasticity(ndiv=5, show=false, nome="loc_roll")
    dad = format2d(msh, props; pontointerno=false)
    p = 1.0
    for i in 1:dad.n
        x, y = dad.Nodes[i]
        n = dad.Normal[i]
        if y < 1e-9 && abs(n[2]) > 0.5
            # bottom: outward n ≈ (0,-1) → u_n=0, t_t=0
            dad.BC[2i-1] = 0; dad.BV[2i-1] = 0.0
            dad.BC[2i] = 1; dad.BV[2i] = 0.0
        elseif y > 1 - 1e-9 && n[2] > 0.5
            # top: t_n = -p (compression), t_t = 0
            dad.BC[2i-1] = 1; dad.BV[2i-1] = -p
            dad.BC[2i] = 1; dad.BV[2i] = 0.0
        elseif x < 1e-9 || x > 1 - 1e-9
            # sides free in local frame
            dad.BC[2i-1] = 1; dad.BV[2i-1] = 0.0
            dad.BC[2i] = 1; dad.BV[2i] = 0.0
        else
            dad.BC[2i-1] = 1; dad.BV[2i-1] = 0.0
            dad.BC[2i] = 1; dad.BV[2i] = 0.0
        end
    end
    # pin one corner tangentially in global sense via an extra Dirichlet on a
    # side node u_t to kill rigid motion horizontally: set left mid ux via
    # converting one node to u_t=0 when n is horizontal
    for i in 1:dad.n
        x, y = dad.Nodes[i]
        if x < 1e-9 && 0.4 < y < 0.6
            dad.BC[2i] = 0; dad.BV[2i] = 0.0   # u_t = 0 on left (t ≈ (0,-1) or (0,1))
            break
        end
    end
    H_G_full_direct(dad, 12)
    u = solve(dad; frame=:local)
    @test all(isfinite, u)
    @test has_cache(dad, :u_local)
    # bottom nodes: u_n ≈ 0
    for i in 1:dad.n
        y = dad.Nodes[i][2]
        n = dad.Normal[i]
        if y < 1e-9 && abs(n[2]) > 0.5
            @test abs(dad.u_local[2i-1]) < 1e-6
        end
    end
    # compression: top moves down (global uy < 0 on average at top)
    uy_top = [u[2i] for i in 1:dad.n if dad.Nodes[i][2] > 1 - 1e-9]
    @test mean(uy_top) < 0
end
