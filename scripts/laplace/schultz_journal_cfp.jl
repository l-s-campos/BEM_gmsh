# Schultz et al. 2025 Table 1 — eccentric journal, CFP with periodic DIBEM.
#
#   julia --project=. scripts/laplace/schultz_journal_cfp.jl
#
# Dimensionless: X = x/L, Y = y/L, H = h/c,
#   P = p c² / (6 η |u| L),   ∇·(H³ ∇P) = ∂(θ H)/∂X
#
# Paper Fig. 5 (centerline x₂ = 0.015 m): p_max ≈ 3.2 MPa at x₁ ≈ 0.065 m,
# cavitation x₁ ≳ 0.09 m, reformation near x₁ ≈ 0.14 m.
#
using DrWatson
@quickactivate :BEM
using LinearAlgebra
using Printf
using Statistics: median
using Plots

get(ENV, "GKSwstype", nothing) === nothing && (ENV["GKSwstype"] = "100")

const OUTDIR = projectdir("plots", "schultz_cfp")
mkpath(OUTDIR)

function mesh_journal_rect(; Lx=1.0, Ly=0.2, nx=31, ny=9, ordem=1, nome="journal_rect")
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(nome)
    lc = min(Lx, Ly) / 5
    p1 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, lc)
    p2 = gmsh.model.geo.addPoint(Lx, 0.0, 0.0, lc)
    p3 = gmsh.model.geo.addPoint(Lx, Ly, 0.0, lc)
    p4 = gmsh.model.geo.addPoint(0.0, Ly, 0.0, lc)
    bot = gmsh.model.geo.addLine(p1, p2)
    rgt = gmsh.model.geo.addLine(p2, p3)
    top = gmsh.model.geo.addLine(p3, p4)
    lft = gmsh.model.geo.addLine(p4, p1)
    cl = gmsh.model.geo.addCurveLoop([bot, rgt, top, lft])
    s = gmsh.model.geo.addPlaneSurface([cl])
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.setTransfiniteCurve(bot, nx)
    gmsh.model.mesh.setTransfiniteCurve(top, nx)
    gmsh.model.mesh.setTransfiniteCurve(rgt, ny)
    gmsh.model.mesh.setTransfiniteCurve(lft, ny)
    gmsh.model.mesh.setTransfiniteSurface(s)
    gmsh.model.mesh.setRecombine(2, s)
    gmsh.model.addPhysicalGroup(1, [bot, top], -1, "0;0")
    gmsh.model.addPhysicalGroup(1, [lft, rgt], -1, "1;0")
    gmsh.model.addPhysicalGroup(2, [s], -1, "pad")
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(ordem)
    out = datadir("Laplace", nome * ".msh")
    mkpath(dirname(out))
    gmsh.write(out)
    gmsh.finalize()
    return out
end

in_oil_supply(x, y; L, W, Ls, Ws) = begin
    yc = W / 2
    xseam = min(abs(x), abs(x - L))
    return xseam <= Ls / 2 && abs(y - yc) <= Ws / 2
end

function apply_eccentric_bcs!(dad; L=1.0, W=0.2, pa=0.0, ps=0.0,
        Ls=0.04, Ws=0.04, tol=2e-3)
    yc = W / 2
    @inbounds for i in 1:dad.n
        x, y = dad.Nodes[i]
        if y < tol || y > W - tol
            dad.BC[i] = 0
            dad.BV[i] = pa
        else
            dad.BC[i] = 1
            dad.BV[i] = 0.0
        end
        if in_oil_supply(x, y; L=L, W=W, Ls=Ls, Ws=Ws)
            dad.BC[i] = 0
            dad.BV[i] = ps
        end
    end
    pairs = mark_periodic_x!(dad; x0=0.0, x1=L, tol=tol)
    # Dirichlet wins on ambient edges and on the oil-supply patch (seam).
    @inbounds for i in 1:dad.n
        x, y = dad.Nodes[i]
        if y < tol || y > W - tol
            dad.BC[i] = 0
            dad.BV[i] = pa
        end
        if in_oil_supply(x, y; L=L, W=W, Ls=Ls, Ws=Ws)
            dad.BC[i] = 0
            dad.BV[i] = ps
        end
    end
    keep = Tuple{Int,Int}[]
    @inbounds for (i, j) in pairs
        dad.BC[i] == 4 && dad.BC[j] == 4 && push!(keep, (i, j))
    end
    set_cache!(dad; periodic_pairs=keep)
    return keep
end

function write_tsv(path, header, cols...)
    open(path, "w") do io
        println(io, join(header, '\t'))
        n = length(cols[1])
        for i in 1:n
            print(io, cols[1][i])
            for c in cols[2:end]
                print(io, '\t', c[i])
            end
            println(io)
        end
    end
    return path
end

function try_plot_centerline(xs, pMPa, ths, pc_MPa, xfvm, pfvm, thfvm)
    ax1 = plot(xs, pMPa; marker=:diamond, markersize=7, label="CFP-DIBEM",
        xlabel="x₁ (m)", ylabel="p (MPa)",
        title="Eccentric journal — centerline  x₂ = 0.015 m", legend=:topleft)
    if !isempty(xfvm)
        plot!(ax1, xfvm, pfvm; color=:black, lw=2, label="FVM Elrod")
    end
    hline!(ax1, [pc_MPa]; color=:gray, linestyle=:dot, label="p_c")
    hline!(ax1, [3.2]; color=:red, linestyle=:dash, label="paper ≈ 3.2 MPa")
    ax2 = plot(xs, ths; marker=:diamond, markersize=7, label="CFP-DIBEM",
        xlabel="x₁ (m)", ylabel="θ", title="Liquid ratio", ylim=(0, 1.05),
        legend=:bottomleft)
    if !isempty(xfvm)
        plot!(ax2, xfvm, thfvm; color=:black, lw=2, label="FVM Elrod")
    end
    fig = plot(ax1, ax2; layout=(2, 1), size=(760, 640))
    png = joinpath(OUTDIR, "journal_eccentric.png")
    savefig(fig, png)
    println("  wrote ", png)
    return nothing
end

function elrod_journal_fvm(; L=0.15, W=0.03, c=30e-6, ε=0.6, η=0.01, U=6.0,
        pa=0.1e6, ps=0.3e6, pc=0.08e6, Ls=6e-3, Ws=6e-3, nx=121, ny=25)
    x = collect(range(0.0, L; length=nx))
    y = collect(range(0.0, W; length=ny))
    dx = x[2] - x[1]
    dy = y[2] - y[1]
    h = [c * (1 + ε * cos(2π * xi / L)) for xi in x, _yj in y]
    p = fill(pa, nx, ny)
    θ = ones(nx, ny)
    fixed = falses(nx, ny)
    yc = W / 2
    @inbounds for j in 1:ny, i in 1:nx
        if in_oil_supply(x[i], y[j]; L=L, W=W, Ls=Ls, Ws=Ws)
            p[i, j] = ps
            θ[i, j] = 1.0
            fixed[i, j] = true
        end
    end
    rheo = ConstRheology(; ρ=820.0, μ=η)
    opt = ElrodOptions(; pcav=pc, maxiter=8000, tol=1e-6, ωp=0.85, ωθ=0.6, verbose=true)
    t0 = time()
    solve_elrod_2d!(p, θ, h, dx, dy; U=U, rheo=rheo, opt=opt,
        bc=(left=nothing, right=nothing, bottom=pa, top=pa),
        periodic_x=true, fixed=fixed)
    t = time() - t0
    jmid = argmin(abs.(y .- yc))
    return (; t, x, y, p, θ, jmid,
        pmax=maximum(p), pmin=minimum(p),
        xc=x, pc=p[:, jmid], thc=θ[:, jmid])
end

function main()
    println("="^72)
    println("Schultz 2025 Table 1 — eccentric journal, CFP + periodic DIBEM")
    println("="^72)

    L = 0.15; W = 0.03; c = 30e-6; ε = 0.6
    η = 0.01; Uphys = 6.0
    pa = 0.1e6; psup = 0.3e6; pc = 0.08e6
    Ls = 6e-3; Ws = 6e-3
    scale = 6 * η * Uphys * L / c^2          # p_dim = P * scale
    Lnd = 1.0
    Wnd = W / L
    pa_nd = pa / scale; ps_nd = psup / scale; pc_nd = pc / scale
    Ls_nd = Ls / L; Ws_nd = Ws / L
    @printf("  scale = %.4e Pa / nd   pa_nd=%.5f  pc_nd=%.5f  ps_nd=%.5f\n",
        scale, pa_nd, pc_nd, ps_nd)

    film = film_journal_unfolded(; L=Lnd, c=1.0, ε=ε)
    msh = mesh_journal_rect(; Lx=Lnd, Ly=Wnd, nx=31, ny=9, nome="schultz_ecc")
    dad = format2d(msh, Laplace(1.0); pontointerno=true, tipo=1)
    pairs = apply_eccentric_bcs!(dad; L=Lnd, W=Wnd, pa=pa_nd, ps=ps_nd,
        Ls=Ls_nd, Ws=Ws_nd, tol=4e-3)
    n_dir = count(==(0), dad.BC)
    n_per = count(==(4), dad.BC)
    n_neu = count(==(1), dad.BC)
    println("  nΓ=", dad.n, "  ni=", dad.ni, "  periodic pairs=", length(pairs),
        "  Dirichlet=", n_dir, "  Neumann=", n_neu, "  periodic=", n_per)

    prescribe = function (pt)
        in_oil_supply(pt[1], pt[2]; L=Lnd, W=Wnd, Ls=Ls_nd, Ws=Ws_nd) ? ps_nd : nothing
    end

    t0 = time()
    p_nd, θ = solve_reynolds_cfp!(dad, film; μ=1.0, U=1 / 6, pc=pc_nd, α=0.66,
        ε=5e-4, maxiter=60, verbose=true, ambient=false, periodic_x=true,
        prescribe=prescribe)
    t_cfp = time() - t0
    p_dim = p_nd .* scale
    imax = argmax(p_dim)
    ptmax = point(dad, imax)
    @printf("  CFP  %.2f s  pmax=%.3f MPa at (x,y)=(%.4f, %.4f) m  pmin=%.3f MPa  θmin=%.3f  iters=%d\n",
        t_cfp, maximum(p_dim) / 1e6, ptmax[1] * L, ptmax[2] * L,
        minimum(p_dim) / 1e6, minimum(θ), length(dad.cfp_hist))

    y0 = Wnd / 2
    # one sample per x-bin, closest to the mid-plane y = W/2
    bins = Dict{Float64,Tuple{Float64,Float64,Float64,Float64}}()
    @inbounds for i in 1:dad.nt
        pt = point(dad, i)
        abs(pt[2] - y0) <= 0.12 * Wnd || continue
        xk = round(pt[1]; digits=4)
        dy = abs(pt[2] - y0)
        if !haskey(bins, xk) || dy < bins[xk][4]
            bins[xk] = (pt[1] * L, dad.T[i] * scale / 1e6, dad.theta[i], dy)
        end
    end
    xs = Float64[]; pcline = Float64[]; ths = Float64[]
    for xk in sort(collect(keys(bins)))
        x, pv, th, _ = bins[xk]
        push!(xs, x); push!(pcline, pv); push!(ths, th)
    end
    if !isempty(pcline)
        kpk = argmax(pcline)
        kcav = findfirst(t -> t < 0.99, ths)
        @printf("  centerline pmax=%.3f MPa at x₁=%.4f m\n", pcline[kpk], xs[kpk])
        if kcav !== nothing
            @printf("  centerline cavitation onset θ<0.99 at x₁=%.4f m\n", xs[kcav])
        end
    end

    xf = Float64[]; pf = Float64[]; thf = Float64[]; pmax_fvm = NaN
    println("  FVM Elrod 2-D reference …")
    try
        fvm = elrod_journal_fvm(; L=L, W=W, c=c, ε=ε, η=η, U=Uphys,
            pa=pa, ps=psup, pc=pc, Ls=Ls, Ws=Ws)
        xf, pf, thf = fvm.xc, fvm.pc ./ 1e6, fvm.thc
        pmax_fvm = fvm.pmax / 1e6
        kpk = argmax(pf)
        @printf("  FVM  %.2f s  pmax=%.3f MPa at x₁=%.4f m  pmin=%.3f MPa\n",
            fvm.t, pmax_fvm, xf[kpk], fvm.pmin / 1e6)
        write_tsv(joinpath(OUTDIR, "journal_eccentric_fvm.tsv"),
            ("x_m", "p_MPa", "theta"), xf, pf, thf)
    catch e
        @warn "FVM reference failed" exception = e
    end

    write_tsv(joinpath(OUTDIR, "journal_eccentric.tsv"),
        ("x_m", "p_MPa", "theta"), xs, pcline, ths)
    xt = Float64[]; yt = Float64[]; pt = Float64[]; tt = Float64[]
    @inbounds for i in 1:dad.nt
        q = point(dad, i)
        push!(xt, q[1] * L); push!(yt, q[2] * L)
        push!(pt, dad.T[i] * scale / 1e6); push!(tt, dad.theta[i])
    end
    write_tsv(joinpath(OUTDIR, "journal_eccentric_field.tsv"),
        ("x_m", "y_m", "p_MPa", "theta"), xt, yt, pt, tt)
    println("  wrote ", joinpath(OUTDIR, "journal_eccentric.tsv"))
    println("  paper Fig. 5: p_max ≈ 3.2 MPa at x₁ ≈ 0.065 m, cavitation x₁ ≳ 0.09 m")
    if isfinite(pmax_fvm) && !isempty(pcline)
        @printf("  vs paper 3.2 MPa:  CFP %+5.1f%%   FVM %+5.1f%%\n",
            100 * (maximum(pcline) / 3.2 - 1), 100 * (pmax_fvm / 3.2 - 1))
    end
    try_plot_centerline(xs, pcline, ths, pc / 1e6, xf, pf, thf)
    return nothing
end

main()
