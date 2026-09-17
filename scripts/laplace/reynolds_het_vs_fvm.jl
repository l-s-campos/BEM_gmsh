# Full-film Reynolds: heterogeneous DIBEM vs FVM (cavitation off).
#
#   julia --project=. scripts/laplace/reynolds_het_vs_fvm.jl
#
using DrWatson
@quickactivate :BEM
using LinearAlgebra
using Printf
using Statistics: median
using Plots

const OUTDIR = projectdir("plots", "reynolds_het_vs_fvm")
mkpath(OUTDIR)

const RHEO = ConstRheology(; ρ=1.0, μ=1.0)
const U = 1.0
const μ = 1.0
const ρ0 = 1.0
const OPT = ElrodOptions(; pcav=-1e30, maxiter=80, tol=1e-10)

"""Direct 2-D FVM for full-film Reynolds (θ ≡ 1). Same stencil as Elrod–Adams."""
function solve_fvm2d_fullfilm(h, dx, dy; U=1.0, μ=1.0, ρ=1.0,
        bc=(left=0.0, right=0.0, bottom=nothing, top=nothing))
    nx, ny = size(h)
    n = nx * ny
    id(i, j) = i + (j - 1) * nx
    A = zeros(n, n)
    rhs = zeros(n)
    @inbounds for j in 1:ny, i in 1:nx
        k = id(i, j)
        if (i == 1 && bc.left !== nothing) || (i == nx && bc.right !== nothing) ||
           (j == 1 && bc.bottom !== nothing) || (j == ny && bc.top !== nothing)
            A[k, k] = 1.0
            rhs[k] = i == 1 && bc.left !== nothing ? bc.left :
                     i == nx && bc.right !== nothing ? bc.right :
                     j == 1 && bc.bottom !== nothing ? bc.bottom : bc.top
            continue
        end
        ie = min(i + 1, nx); iw = max(i - 1, 1)
        jn = min(j + 1, ny); js = max(j - 1, 1)
        hf = 0.5 * (h[i, j] + h[ie, j]); hb = 0.5 * (h[i, j] + h[iw, j])
        hn = 0.5 * (h[i, j] + h[i, jn]); hs = 0.5 * (h[i, j] + h[i, js])
        Γf = ρ * hf^3 / (12μ); Γb = ρ * hb^3 / (12μ)
        Γn = ρ * hn^3 / (12μ); Γs = ρ * hs^3 / (12μ)
        ke = Γf * dy / dx; kw = Γb * dy / dx
        kn = Γn * dx / dy; ks = Γs * dx / dy
        i == nx && (ke = 0.0); i == 1 && (kw = 0.0)
        j == ny && (kn = 0.0); j == 1 && (ks = 0.0)
        A[k, k] = ke + kw + kn + ks
        ke != 0 && (A[k, id(ie, j)] -= ke)
        kw != 0 && (A[k, id(iw, j)] -= kw)
        kn != 0 && (A[k, id(i, jn)] -= kn)
        ks != 0 && (A[k, id(i, js)] -= ks)
        coup = (ρ * U / 2) * dy * (hf - hb)
        if i == 1; coup = (ρ * U / 2) * dy * hf; end
        if i == nx; coup = -(ρ * U / 2) * dy * hb; end
        rhs[k] = -coup
    end
    p = reshape(A \ rhs, nx, ny)
    return p
end

function _centerline_dibem(dad, y0; atol=0.03)
    xs = Float64[]; ps = Float64[]
    @inbounds for i in 1:dad.nt
        pt = point(dad, i)
        abs(pt[2] - y0) <= atol || continue
        push!(xs, pt[1]); push!(ps, dad.T[i])
    end
    perm = sortperm(xs)
    return xs[perm], ps[perm]
end

function _rel(a, b)
    return norm(a .- b) / (norm(b) + 1e-14)
end

function case_wedge_1d()
    println("\n## 1-D linear wedge  (p=0 at ends, no side leakage)")
    film = film_linear(; a=2.0, hi=2.0, L=1.0)
    n = 201
    x = collect(range(0.0, 1.0; length=n))
    h = film.h.(x)
    dx = x[2] - x[1]
    p_fvm, θ = solve_elrod_1d(h, dx; U=U, rheo=RHEO, pleft=0.0, pright=0.0, opt=OPT)
    p_ex = infinite_bearing_pressure(film, x; μ=μ, U=U)
    @printf("  FVM vs exact   rel=%.3e  pmax FVM=%.4f  exact=%.4f  θmin=%.3f\n",
        _rel(p_fvm, p_ex), maximum(p_fvm), maximum(p_ex), minimum(θ))
    return x, p_fvm, p_ex, film
end

function case_strip(film; B=0.75, ndiv=16, nx=81, ny=21, nome="rey_strip")
    println("\n## 2-D strip  B/L=$(B)  film=$(film.name)  p=0 on x=0,L; q=0 on y-sides")
    msh = quadrado(ndiv=ndiv, Lx=1.0, Ly=B, show=false, nome=nome)
    dad = format2d(msh, Laplace(1.0); pontointerno=true, tipo=2)
    @inbounds for i in 1:dad.n
        x = dad.Nodes[i][1]
        if x < 1e-8 || x > 1 - 1e-8
            dad.BC[i] = 0; dad.BV[i] = 0.0
        else
            dad.BC[i] = 1; dad.BV[i] = 0.0
        end
    end
    assemble!(dad; npg=12, threaded=false)
    t0 = time()
    solve_reynolds_het!(dad, film; μ=μ, U=U, ambient=false)
    t_het = time() - t0
    y0 = B / 2
    xh, ph = _centerline_dibem(dad, y0; atol=max(0.04, 0.6 * B / ndiv))

    x = collect(range(0.0, 1.0; length=nx))
    y = collect(range(0.0, B; length=ny))
    dx = x[2] - x[1]; dy = y[2] - y[1]
    h = [film.h(xi) for xi in x, _yj in y]
    t0 = time()
    p = solve_fvm2d_fullfilm(h, dx, dy; U=U, μ=μ, ρ=ρ0,
        bc=(left=0.0, right=0.0, bottom=nothing, top=nothing))
    t_fvm = time() - t0
    jmid = (ny + 1) ÷ 2
    pf = p[:, jmid]
    p_ex = infinite_bearing_pressure(film, x; μ=μ, U=U)
    # interpolate het onto FVM x
    ph_i = similar(x)
    for (i, xi) in enumerate(x)
        k = argmin(abs.(xh .- xi))
        ph_i[i] = isempty(ph) ? NaN : ph[k]
    end
    @printf("  het DIBEM  nΓ=%d ni=%d  pmax=%.4f  %.3f s\n", dad.n, dad.ni, maximum(dad.T), t_het)
    @printf("  FVM 2D     %d×%d        pmax=%.4f  %.3f s\n", nx, ny, maximum(pf), t_fvm)
    @printf("  1-D exact               pmax=%.4f\n", maximum(p_ex))
    @printf("  FVM vs exact  rel=%.3e   het vs FVM (nearest y=B/2) rel=%.3e\n",
        _rel(pf, p_ex), _rel(ph_i, pf))
    return (; x, pf, ph=xh, phe=ph, p_ex, dad, t_het, t_fvm)
end

function case_pad_h2(; nx=81, ny=31)
    println("\n## Guiggiani pad  film h2  p=0 on all Γ")
    film = film_h2(; a=2.0, hi=2.0)
    msh = mesh_guiggiani_pad(; nome="rey_cmp_pad", show=false)
    dad0 = format2d(msh, Laplace(1.0); pontointerno=false, tipo=2)
    internal_grid!(dad0, 11, 7; d_min=0.02, layout=:cell)
    assemble!(dad0; npg=12, threaded=false)
    d_het = deepcopy(dad0)
    d_tr = deepcopy(dad0)
    t0 = time(); solve_reynolds_het!(d_het, film; μ=μ, U=U); t_het = time() - t0
    t0 = time(); solve_reynolds_dibem!(d_tr, film; μ=μ, U=U); t_tr = time() - t0

    B = 0.75
    x = collect(range(0.0, 1.0; length=nx))
    y = collect(range(-B / 2, B / 2; length=ny))
    dx = x[2] - x[1]; dy = y[2] - y[1]
    h = [film.h(xi) for xi in x, _yj in y]
    t0 = time()
    p = solve_fvm2d_fullfilm(h, dx, dy; U=U, μ=μ, ρ=ρ0,
        bc=(left=0.0, right=0.0, bottom=0.0, top=0.0))
    t_fvm = time() - t0
    j0 = argmin(abs.(y))
    pf0 = p[:, j0]
    xh, ph = _centerline_dibem(d_het, 0.0; atol=0.04)
    xt, pt = _centerline_dibem(d_tr, 0.0; atol=0.04)
    pmax_h = maximum(d_het.T[(d_het.n + 1):end])
    pmax_t = maximum(d_tr.T[(d_tr.n + 1):end])
    pmax_f = maximum(p)
    @printf("  transform DIBEM  pmax=%.4f  %.3f s\n", pmax_t, t_tr)
    @printf("  het DIBEM        pmax=%.4f  %.3f s   vs transform rel=%.3e\n",
        pmax_h, t_het, norm(d_het.T - d_tr.T) / (norm(d_tr.T) + 1e-14))
    @printf("  FVM rect pad     pmax=%.4f  %.3f s   (rounded Γ vs rectangle)\n",
        pmax_f, t_fvm)
    p_inf = maximum(infinite_bearing_pressure(film, range(0.05, 0.95; length=40); μ=μ, U=U))
    @printf("  1-D infinite     pmax=%.4f\n", p_inf)
    return (; x, pf0, xh, ph, xt, pt, p_inf, y, p)
end

function main()
    println("="^72)
    println("Full-film Reynolds: heterogeneous DIBEM vs FVM  (no cavitation)")
    println("="^72)
    x1, p_fvm1, p_ex1, film = case_wedge_1d()
    s75 = case_strip(film; B=0.75, ndiv=16, nome="rey_b075")
    pad = case_pad_h2()

    ax1 = plot(x1, p_ex1; color=:black, lw=2, label="exact",
        xlabel="x", ylabel="p", title="1-D linear wedge", legend=:topleft)
    plot!(ax1, x1, p_fvm1; color=:red, linestyle=:dash, lw=2, label="FVM")

    ax2 = plot(s75.x, s75.p_ex; color=:black, lw=2, label="1-D exact (no leak)",
        xlabel="x", ylabel="p", title="Linear wedge, strip B/L=0.75, q=0 on sides",
        legend=:topleft)
    plot!(ax2, s75.x, s75.pf; color=:red, lw=2, label="FVM 2-D")
    scatter!(ax2, s75.ph, s75.phe; marker=:diamond, markersize=8, label="het DIBEM")

    ax3 = plot(pad.x, pad.p_inf .* ones(length(pad.x)); color=:gray, linestyle=:dot,
        label="1-D inf (ref)", xlabel="x", ylabel="p",
        title="Guiggiani pad, film h2, p=0 on Γ", legend=:topleft)
    plot!(ax3, pad.x, pad.pf0; color=:red, lw=2, label="FVM y=0")
    plot!(ax3, pad.xh, pad.ph; marker=:diamond, markersize=8, label="het DIBEM y≈0")
    plot!(ax3, pad.xt, pad.pt; marker=:circle, markersize=6, label="transform DIBEM y≈0")
    fig = plot(ax1, ax2, ax3; layout=(3, 1), size=(780, 900))
    savefig(fig, joinpath(OUTDIR, "compare.png"))
    println("\nWrote ", joinpath(OUTDIR, "compare.png"))
end

main()
