# Schultz et al., Tribol. Int. 212:110967 (2025), Interpretation I (CFP)
# with P(θ) = heterogeneous DIBEM. Parabolic slider vs Elrod FVM.
#
#   julia --project=. scripts/laplace/schultz_cfp_dibem.jl
#
using DrWatson
@quickactivate :BEM
using LinearAlgebra
using Printf
using Plots

const OUTDIR = projectdir("plots", "schultz_cfp")
mkpath(OUTDIR)

function main()
    println("="^72)
    println("CFP (Schultz 2025 I)  —  P(θ) = heterogeneous DIBEM")
    println("="^72)
    film = film_parabolic()
    L = film.L
    μ = 39e-3
    U = 4.57
    pleft = 3.36414e3
    pright = 0.0
    pc = 0.0

    # --- 1-D FVM reference (Elrod–Adams) ---
    n1 = 301
    x1 = collect(range(0.0, L; length=n1))
    h1 = film.h.(x1)
    dx1 = x1[2] - x1[1]
    p_fvm, θ_fvm = solve_elrod_1d(h1, dx1; U=U, rheo=ConstRheology(; ρ=580.0, μ=μ),
        pleft=pleft, pright=pright,
        opt=ElrodOptions(; pcav=pc, maxiter=80, tol=1e-8, verbose=true))
    @printf("  FVM 1D   pmax=%.3f MPa  n_cav=%d\n",
        maximum(p_fvm) / 1e6, count(<(0.999), θ_fvm))

    # --- DIBEM strip, q=0 on sides ---
    B = L / 8
    msh = quadrado(ndiv=14, Lx=L, Ly=B, show=false, nome="cfp_para")
    dad = format2d(msh, Laplace(1.0); pontointerno=true, tipo=1)
    @inbounds for i in 1:dad.n
        x = dad.Nodes[i][1]
        if x < 1e-4 * L
            dad.BC[i] = 0; dad.BV[i] = pleft
        elseif x > L - 1e-4 * L
            dad.BC[i] = 0; dad.BV[i] = pright
        else
            dad.BC[i] = 1; dad.BV[i] = 0.0
        end
    end
    println("  DIBEM  nΓ=", dad.n, "  ni=", dad.ni)
    t0 = time()
    p_cfp, θ_cfp = solve_reynolds_cfp!(dad, film; μ=μ, U=U, pc=pc, α=0.66,
        ε=5e-4, maxiter=25, verbose=true, ambient=false)
    t_cfp = time() - t0
    @printf("  CFP-DIBEM  pmax=%.3f MPa  pmin=%.3f MPa  %.2f s  iters=%d\n",
        maximum(p_cfp) / 1e6, minimum(p_cfp) / 1e6, t_cfp, length(dad.cfp_hist))

    y0 = B / 2
    xs = Float64[]; ps = Float64[]; ths = Float64[]
    @inbounds for i in 1:dad.nt
        pt = point(dad, i)
        abs(pt[2] - y0) <= 0.15 * B || continue
        push!(xs, pt[1]); push!(ps, dad.T[i]); push!(ths, dad.theta[i])
    end
    perm = sortperm(xs)
    xs, ps, ths = xs[perm], ps[perm], ths[perm]

    ax1 = plot(x1 ./ L, p_fvm ./ 1e6; color=:black, lw=2, label="FVM Elrod",
        xlabel="x / L", ylabel="p (MPa)",
        title="Parabolic slider — CFP-DIBEM vs Elrod FVM", legend=:topleft)
    scatter!(ax1, xs ./ L, ps ./ 1e6; marker=:diamond, markersize=8, label="CFP-DIBEM")
    ax2 = plot(x1 ./ L, θ_fvm; color=:black, lw=2, label="FVM",
        xlabel="x / L", ylabel="θ", title="Liquid ratio", legend=:bottomleft, ylim=(0, 1.05))
    scatter!(ax2, xs ./ L, ths; marker=:diamond, markersize=8, label="CFP-DIBEM")
    fig = plot(ax1, ax2; layout=(2, 1), size=(720, 640))
    savefig(fig, joinpath(OUTDIR, "cfp_parabolic.png"))
    println("  wrote ", joinpath(OUTDIR, "cfp_parabolic.png"))
end

main()
