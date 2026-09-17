# SLIPPY mixed-lubrication example (ball on plane) with BEM-only semi-system.
#
#   julia --project=. scripts/laplace/slippy_semi_system.jl [n_int]
#
# Same numbers as slippy `examples/Mixed lubrication by the semi system approach.ipynb`.
# Fluid and gap are Hertz-nondimensional (X=x/a, H=h R/a², P=p/p_H) so DIBEM
# sees O(1) fields, as in the journal CFP runs.
#
using DrWatson
@quickactivate :BEM
using BEM.Contact
using LinearAlgebra
using Printf
using Plots
using DelimitedFiles

get(ENV, "GKSwstype", nothing) === nothing && (ENV["GKSwstype"] = "100")
const OUTDIR = projectdir("plots", "slippy_ehl")
mkpath(OUTDIR)

function hamrock_dowson_hmin(; η0, ū, Estar, R, F, α)
    U = η0 * ū / (Estar * R)
    G = α * Estar
    W = F / (Estar * R^2)
    Hmin = 3.63 * U^0.68 * G^0.49 * W^(-0.073) * (1 - exp(-0.68))
    return Hmin * R, U, G, W
end

function main()
    println("="^72)
    println("SLIPPY ball-on-flat EHL — BEM semi-system (Hertz nd)")
    println("="^72)

    radius = 0.01905
    # SLIPPY notebook uses 800 N (λ≈0.009, piezoviscous Hertz). 15×15 DIBEM
    # cannot hold that; same ball/oil/speed at 8 N has λ=O(1) and a thick film.
    load = 8.0
    ū = 4.0
    E = 200e9
    ν = 0.3
    η0 = 0.096
    n_int = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 25
    nside = n_int + 1

    G = G_from_E(E, ν)
    Estar = contact_modulus(combined_halfspace(G, ν, G, ν))
    hz = hertz_sphere_load(radius, load, Estar)
    a = hz.a
    pH = hz.p0
    δH = hz.δ
    λ = 12 * η0 * ū * radius^2 / (pH * a^3)
    Wnd = load / (pH * a^2)
    println("  Hertz  a=", a, " m  p0=", pH / 1e9, " GPa  δ=", δH, " m")
    println("  nd     λ=", λ, "  Wnd=", Wnd, "  domain X∈[-2,2]")

    msh = mesh_ehl_square(; L=4.0, nside=nside, nome="slippy_ehl_n$(n_int)")
    dad = format2d(msh, Laplace(1.0); pontointerno=false, tipo=1)
    internal_grid!(dad, n_int, n_int; d_min=0.0, layout=:cell)
    apply_ambient_pressure!(dad)
    gmap = interior_rect_map(dad)
    hs = combined_halfspace(G, ν, G, ν; hx=gmap.hx * a, hy=gmap.hy * a)
    prep = precompute_kernels(gmap.nix, gmap.niy, hs; components=(Kzz,))
    println("  nΓ=", dad.n, "  ni=", dad.ni, "  grid=", gmap.nix, "×", gmap.niy,
        "  hx_nd=", gmap.hx)

    nt = dad.nt
    h0 = zeros(nt)
    p0 = zeros(nt)
    @inbounds for i in 1:nt
        pt = point(dad, i)
        h0[i] = (pt[1]^2 + pt[2]^2) / 2
        r2 = pt[1]^2 + pt[2]^2
        if r2 < 1
            p0[i] = sqrt(1 - r2)
        end
    end

    α = 0.68 * (log(η0) + 9.67) * 5.1e-9
    hmin_hd, Ud, Gd, Wd = hamrock_dowson_hmin(; η0=η0, ū=ū, Estar=Estar, R=radius, F=load, α=α)
    @printf("  Hamrock–Dowson hmin = %.3e m  (U=%.3e  G=%.1f  W=%.3e)\n", hmin_hd, Ud, Gd, Wd)

    μfun = (_P -> 1.0)   # isoviscous; Roelands cap is optional via μ_cap
    hmin_nd = 0.47e-9 * radius / a^2
    hfloor_nd = 1e-4
    u_scale = radius / a^2
    p0 .*= 0.3

    t0 = time()
    sol = solve_semi_system!(dad, gmap, hs, prep;
        h0=h0, W=Wnd, ū=λ / 12, η0=1.0, R=radius, δ0=0.4, p0=p0,
        μfun=μfun, h_min=hmin_nd, h_kfloor=hfloor_nd, μ_cap=1e4,
        p_yield=5.0, p_fft_scale=pH, u_scale=u_scale,
        ωp=0.25, ωδ=0.06, maxiter=150, rtol_p=5e-4, rtol_W=2e-3, verbose=true)
    t = time() - t0
    pmax = sol.pmax * pH
    hmin = sol.hmin * a^2 / radius
    Wphys = sol.load * pH * a^2
    @printf("  BEM EHL  %.1f s  iters=%d  pmax=%.3f GPa  hmin=%.3e m  W=%.1f N  er_W=%.3g\n",
        t, sol.iters, pmax / 1e9, hmin, Wphys, sol.er_W)
    @printf("  vs Hertz p0=%.3f GPa  (BEM %+5.1f%%)   vs HD hmin %+5.1f%%\n",
        pH / 1e9, 100 * (pmax / pH - 1), 100 * (hmin / hmin_hd - 1))

    j0 = argmin(abs.(gmap.ys))
    xc = gmap.xs
    pc = sol.pmat[:, j0]
    hc = [sol.h[gmap.idx[i, j0]] * a^2 / radius for i in 1:gmap.nix]
    pHertz = [let r2 = x^2
            r2 < 1 ? sqrt(1 - r2) : 0.0
        end for x in xc]

    tsv = joinpath(OUTDIR, "centerline_n$(n_int).tsv")
    open(tsv, "w") do io
        println(io, "x_over_a\tp_over_pH\th_m\tpHertz")
        for i in eachindex(xc)
            println(io, xc[i], '\t', pc[i], '\t', hc[i], '\t', pHertz[i])
        end
    end
    println("  wrote ", tsv)

    ax1 = plot(xc, pHertz; color=:black, ls=:dash, lw=2, label="Hertz",
        xlabel="x/a", ylabel="p / p_H",
        title="Ball-on-flat EHL  8 N  $(n_int)×$(n_int)", legend=:topright)
    tsv15 = joinpath(OUTDIR, "centerline_n15.tsv")
    if n_int != 15 && isfile(tsv15)
        raw = readdlm(tsv15, '\t', Float64; skipstart=1)
        plot!(ax1, raw[:, 1], raw[:, 2]; color=:gray, marker=:circle, markersize=4,
            label="BEM 15×15")
    end
    plot!(ax1, xc, pc; marker=:diamond, label="BEM $(n_int)×$(n_int)")
    ax2 = plot(xc, hc .* 1e6; marker=:diamond, label="BEM $(n_int)×$(n_int)",
        xlabel="x/a", ylabel="h (μm)", legend=:topright)
    if n_int != 15 && isfile(tsv15)
        raw = readdlm(tsv15, '\t', Float64; skipstart=1)
        plot!(ax2, raw[:, 1], raw[:, 3] .* 1e6; color=:gray, marker=:circle,
            markersize=4, label="BEM 15×15")
    end
    hline!(ax2, [hmin_hd * 1e6]; color=:red, ls=:dash, label="Hamrock–Dowson")
    fig = plot(ax1, ax2; layout=(2, 1), size=(720, 640))
    png = joinpath(OUTDIR, "slippy_ehl_centerline_n$(n_int).png")
    savefig(fig, png)
    println("  wrote ", png)

    ax3 = heatmap(gmap.xs, gmap.ys, sol.pmat';
        xlabel="x/a", ylabel="y/a", title="p / p_H  $(n_int)×$(n_int)",
        aspect_ratio=:equal, colorbar=true, size=(640, 520))
    png2 = joinpath(OUTDIR, "slippy_ehl_pressure_n$(n_int).png")
    savefig(ax3, png2)
    println("  wrote ", png2)
    return sol
end

main()
