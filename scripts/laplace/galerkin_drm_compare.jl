# Collocation DRM vs Galerkin DRM: Laplace (Poisson), diffusion, wave.
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Statistics, Printf

include(datadir("Laplace", "Laplace_dad.jl"))
include(datadir("Laplace", "potencial_problems.jl"))
include(datadir("Laplace", "wave_propagation.jl"))

pts_of(dad) = all_points(dad)
rmse(a, b) = sqrt(mean(abs2, a .- b))


function _square(ndiv, nome; internals=true)
    msh = quadrado(ndiv=ndiv, show=false, nome=nome)
    return format2d(msh, Laplace(1.0); tipo=1, pontointerno=internals)
end

function poisson_pair()
    ufun(p) = p[1]^2 + p[2]^2
    rows = []
    for (gal, tag) in ((false, "collocation"), (true, "galerkin"))
        dad = _square(8, "pois_$tag")
        for i in 1:dad.n
            dad.BC[i] = 0
            dad.BV[i] = ufun(dad.Nodes[i])
        end
        t0 = time()
        u = solve_drm_poisson!(dad, 4.0; basis=PHS(3; poly_deg=1), npg=10, galerkin=gal)
        dt = time() - t0
        uex = ufun.(pts_of(dad))

        push!(rows, (; problem="Poisson u=x²+y²", method=tag, n=dad.n, nt=dad.nt,
            rmse=rmse(u, uex), time=dt))
    end
    return rows
end

function laplace_Tx_pair()
    rows = []
    for (gal, tag) in ((false, "collocation"), (true, "galerkin"))
        dad = _square(10, "Tx_$tag"; internals=false)
        apply_analytical_bc!(dad, ana_laplace_linear(; direction=SA[1.0, 0.0]))
        t0 = time()
        if gal
            H_G_galerkin(dad; npg=10, threaded=false)
        else
            H_G_full_direct(dad; npg=10, threaded=false)
        end
        solve(dad)
        dt = time() - t0
        qana = -getindex.(dad.Normal, 1)
        push!(rows, (; problem="Laplace T=x (flux)", method=tag, n=dad.n, nt=dad.nt,
            rmse=rmse(dad.q, qana), time=dt))
    end
    return rows
end

function heat_pair()
    κ = 1.0
    ufun(p, t) = exp(-2 * π^2 * κ * t) * sin(π * p[1]) * sin(π * p[2])
    rows = []
    for (gal, tag) in ((false, "collocation"), (true, "galerkin"))
        dad = _square(6, "heat_$tag")
        for i in 1:dad.n
            dad.BC[i] = 0
            dad.BV[i] = 0.0
        end
        pts = pts_of(dad)

        u0 = [ufun(p, 0.0) for p in pts]
        t0 = time()
        t_hist, U = solve_transient_drm!(dad, u0; κ=κ, Δt=0.002, t_end=0.02,
            f=0.0, θ=1.0, basis=PHS(3; poly_deg=1), npg=10, galerkin=gal)
        dt = time() - t0
        uex = [ufun(p, t_hist[end]) for p in pts]
        push!(rows, (; problem="diffusion sinπx sinπy", method=tag, n=dad.n, nt=dad.nt,
            rmse=rmse(U[:, end], uex), time=dt))
    end
    return rows
end

function wave_pair()
    rows = []
    Δt, tf = 0.05, 1.0
    probe = Point2D(1.0, 0.5)
    ana = ana_bar_sudden()
    for (gal, tag) in ((false, "collocation"), (true, "galerkin"))
        dad, _ = wave_problem(:bar_sudden; ndiv=8, n_int=4)
        t0 = time()
        drm = build_drm_matrices(dad, PHS(3; poly_deg=1); npg=8, galerkin=gal)
        set_cache!(dad; H=drm.H, G=drm.G, M=drm.M)
        if has_cache(dad, :A)
            dad.cache.A = nothing
        end
        solve_mmm!(dad, Δt, tf)
        dt = time() - t0
        pts = pts_of(dad)
        ip = argmin(norm(p - probe) for p in pts)
        t = dad.t
        u_num = dad.T[ip, :]
        u_ana = [ana.u(pts[ip]; t=tt) for tt in t]
        push!(rows, (; problem="wave bar_sudden MMM", method=tag, n=dad.n, nt=dad.nt,
            rmse=rmse(u_num, u_ana), time=dt,
            umax=maximum(abs, u_num), uana_max=maximum(abs, u_ana)))
    end
    return rows
end


function main()
    rows = vcat(laplace_Tx_pair(), poisson_pair(), heat_pair(), wave_pair())
    println()
    @printf("%-28s %-12s %4s %4s %11s %8s %s\n",
        "problem", "method", "n", "nt", "RMSE", "time(s)", "notes")
    println("-"^90)
    for r in rows
        extra = haskey(r, :umax) ? @sprintf("umax=%.3f ana=%.3f", r.umax, r.uana_max) : ""
        @printf("%-28s %-12s %4d %4d %11.4e %8.2f %s\n",
            r.problem, r.method, r.n, r.nt, r.rmse, r.time, extra)
    end
    return rows
end

main()
