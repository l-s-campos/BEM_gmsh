# Manufactured solutions: Poisson RBF-BEM + transient DRM
using Test
using LinearAlgebra
using Statistics: mean
using BEM

include(joinpath(@__DIR__, "..", "data", "Laplace", "Laplace_dad.jl"))

function _square_dad(ndiv; pontointerno=true, nome="pois_mfg")
    msh = Base.invokelatest(quadrado; ndiv=ndiv, show=false, nome=nome)
    return format2d(msh, Laplace(1.0); pontointerno=pontointerno)
end

function _set_dirichlet!(dad, ufun)
    for i in 1:dad.n
        dad.BC[i] = 0
        dad.BV[i] = ufun(dad.Nodes[i])
    end
end

function _set_mixed_square!(dad, ufun, qfun)
    # left/right Dirichlet, top/bottom Neumann (by normal)
    for i in 1:dad.n
        p = dad.Nodes[i]
        n = dad.Normal[i]
        if abs(abs(n[1]) - 1) < 0.5  # vertical sides
            dad.BC[i] = 0
            dad.BV[i] = ufun(p)
        else
            dad.BC[i] = 1
            dad.BV[i] = qfun(p, n)  # q = -k ∂u/∂n
        end
    end
end

@testset "Poisson manufactured u=x²+y² (f=4)" begin
    ufun(p) = p[1]^2 + p[2]^2
    dad = _square_dad(8; nome="pois_xy2")
    _set_dirichlet!(dad, ufun)
    u = solve_poisson_rbf_bem!(dad, 4.0; method=:global, basis=PHS(3; poly_deg=1), npg=14)
    pts = [dad.Nodes; dad.internalNodes]
    uex = ufun.(pts)
    rmse = sqrt(mean(abs2, u .- uex))
    @info "u=x²+y² RMSE" rmse
    @test rmse < 0.15
end

@testset "Poisson manufactured u=x³ (f=6x)" begin
    # smoother polynomial source — good accuracy test for PHS particular
    ufun(p) = p[1]^3
    fsrc(p) = 6 * p[1]
    dad = _square_dad(10; nome="pois_x3")
    _set_dirichlet!(dad, ufun)
    u = solve_poisson_rbf_bem!(dad, fsrc; method=:global, basis=PHS(5; poly_deg=2), npg=14)
    pts = [dad.Nodes; dad.internalNodes]
    uex = ufun.(pts)
    rmse = sqrt(mean(abs2, u .- uex))
    @info "u=x³ RMSE" rmse
    @test all(isfinite, u)
    @test rmse < 0.25
end

@testset "Poisson mixed BC" begin
    ufun(p) = p[1]^2 + p[2]^2
    qfun(p, n) = -2 * (p[1] * n[1] + p[2] * n[2])  # k=1
    dad = _square_dad(8; nome="pois_mix")
    _set_mixed_square!(dad, ufun, qfun)
    u = solve_poisson_rbf_bem!(dad, 4.0; method=:global, basis=PHS(3; poly_deg=1), npg=14)
    pts = [dad.Nodes; dad.internalNodes]
    uex = ufun.(pts)
    rmse = sqrt(mean(abs2, u .- uex))
    @info "mixed BC RMSE" rmse
    @test all(isfinite, u)
    @test rmse < 0.4
end

@testset "compare Poisson methods" begin
    ufun(p) = p[1]^2 + p[2]^2
    dad = _square_dad(6; nome="pois_cmp")
    _set_dirichlet!(dad, ufun)
    res = compare_poisson_rbf_bem(dad, 4.0, ufun; methods=(:global, :local), npg=12)
    @test any(r -> isfinite(r.rmse) && r.rmse < 0.2, res)
    @info "compare" res
end

@testset "transient DRM manufactured" begin
    # u = exp(-2π² κ t) sin(πx) sin(πy), f=0, κ=1
    κ = 1.0
    ufun(p, t) = exp(-2 * π^2 * κ * t) * sin(π * p[1]) * sin(π * p[2])
    dad = _square_dad(6; nome="drm_heat", pontointerno=true)
    # Dirichlet time-dependent: update each step via BV — for simplicity use
    # homogeneous Dirichlet (compatible with IC=0 at t large) — use IC at t=0
    # with Dirichlet u=0 on boundary (exact satisfies).
    for i in 1:dad.n
        dad.BC[i] = 0
        dad.BV[i] = 0.0
    end
    pts = [dad.Nodes; dad.internalNodes]
    u0 = [ufun(p, 0.0) for p in pts]
    # boundary of exact is 0, interior nonzero
    t_hist, U = solve_transient_drm!(dad, u0; κ=κ, Δt=0.002, t_end=0.02,
        f=0.0, θ=1.0, basis=PHS(3; poly_deg=1), npg=12)
    # error at final time on interior nodes
    t_end = t_hist[end]
    uex = [ufun(p, t_end) for p in pts]
    rmse = sqrt(mean(abs2, U[:, end] .- uex))
    @info "DRM heat RMSE" rmse t_end length(t_hist)
    @test rmse < 0.35   # coarse mesh + BE; should be finite and reasonable
    @test all(isfinite, U)
end

@testset "build_drm_matrices" begin
    dad = _square_dad(5; nome="drm_mat")
    for i in 1:dad.n
        dad.BC[i] = 0
        dad.BV[i] = 0.0
    end
    drm = build_drm_matrices(dad, PHS(3; poly_deg=1); npg=10)
    @test size(drm.M) == (dad.nt, dad.nt)
    @test all(isfinite, drm.M)
end
