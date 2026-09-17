# Time-domain Burton–Miller correlato on the wave bar
#   (H + α H′) u − (G + α G′) q = (M + α M′) ü
# Mass: DRM / DIBEM / cells. Interior rows stay CBIE.
#
#   julia --project=. scripts/burton_miller_wave.jl
using DrWatson
@quickactivate :BEM
using LinearAlgebra
using Printf

include(datadir("Laplace", "Laplace_dad.jl"))
include(datadir("Laplace", "potencial_problems.jl"))
include(datadir("Laplace", "wave_propagation.jl"))

const NDIV = parse(Int, get(ENV, "BM_NDIV", "8"))
const NINT = parse(Int, get(ENV, "BM_NINT", "4"))
const NPG = parse(Int, get(ENV, "BM_NPG", "10"))
const TRES = parse(Float64, get(ENV, "BM_TRES", "0.5"))
const DT = parse(Float64, get(ENV, "BM_DT", "0.05"))
const TF = parse(Float64, get(ENV, "BM_TF", "2.0"))

rel(a, b) = norm(a - b) / (norm(b) + 1e-30)

function _relres(r, parts...)
    den = sum(norm, parts) + 1e-30
    return norm(r) / den
end

function fill_bar!(u, q, ddu, dad, t; N=400)
    pts = all_points(dad)
    @inbounds for i in eachindex(pts)
        f = bar_sudden_fields(pts[i], t; N=N, c=1.0, L=1.0)
        u[i] = f.u
        ddu[i] = f.ddu
    end
    @inbounds for i in 1:dad.n
        f = bar_sudden_fields(dad.Nodes[i], t; N=N, c=1.0, L=1.0)
        q[i] = -f.dudx * dad.Normal[i][1]
    end
    return u, q, ddu
end

function probe_end(dad)
    pts = all_points(dad)
    return argmin(norm(p - Point2D(1.0, 0.5)) for p in pts)
end

function report_residual(tag, α, dad, u, q, ddu)
    n = dad.n
    H, G, M = dad.H_cbie, dad.G_cbie, dad.M_cbie
    Hp, Gp, Mp = dad.H_hyper, dad.G_hyper, dad.M_hyper
    Hu, Gq, Md = H * u, G * q, M * ddu
    r = Hu .- Gq .- Md
    Hpu, Gpq, Mpd = Hp * u, Gp * q, view(Mp, 1:n, :) * ddu
    rp = Hpu .- Gpq .- Mpd
    rα = r[1:n] .+ α .* rp
    Hc, Gc, Mc = dad.H, dad.G, dad.M
    rc = Hc * u .- Gc * q .- Mc * ddu
    @printf("  %-8s α=%4.1f  CBIE ||r||/Σ=%.3e  HBIE ||r′||/Σ=%.3e  BM-row=%.3e  combined=%.3e\n",
        tag, α,
        _relres(r, Hu, Gq, Md),
        _relres(rp, Hpu, Gpq, Mpd),
        _relres(rα, Hu[1:n], Gq[1:n], Md[1:n], α .* Hpu, α .* Gpq, α .* Mpd),
        _relres(rc, Hc * u, Gc * q, Mc * ddu))
    return nothing
end

println(" Burton–Miller correlato  (H+αH′)u − (G+αG′)q = (M+αM′)ü")
println(" bar_sudden  ndiv=$NDIV  n_int=$NINT  npg=$NPG")
println()

dad0, meta = wave_problem(:bar_sudden; ndiv=NDIV, n_int=NINT)
u = zeros(dad0.nt)
q = zeros(dad0.n)
ddu = zeros(dad0.nt)
fill_bar!(u, q, ddu, dad0, TRES)

println(" Analytical residual at t=$TRES")
for mass in (:drm, :dibem, :cells)
    for α in (0.0, 1.0)
        dad = deepcopy(dad0)
        assemble_wave_burton_miller!(dad; mass=mass, α=α, npg=NPG,
            rbf=PHS(3; poly_deg=1), basis=PHS(3; poly_deg=1), threaded=true)
        report_residual(String(mass), α, dad, u, q, ddu)
    end
end

println()
println(" Houbolt vs ana_bar_sudden  Δt=$DT  tf=$TF")
ana = meta.ana
@printf("  %-8s %6s  %10s  %10s  %10s  %10s\n",
    "mass", "α", "u_end_err", "max|u|", "max|ana|", "finite")
for mass in (:drm, :dibem, :cells)
    for α in (0.0, 0.5, 1.0)
        dad = deepcopy(dad0)
        assemble_wave_burton_miller!(dad; mass=mass, α=α, npg=NPG,
            rbf=PHS(3; poly_deg=1), basis=PHS(3; poly_deg=1), threaded=true)
        solve_Houbolt(dad, DT, TF)
        ip = probe_end(dad)
        tgrid = dad.time
        u_num = dad.T[ip, :]
        u_ana = [float(ana.u(Point2D(1.0, 0.5); t=ti)) for ti in tgrid]
        err = rel(u_num, u_ana)
        ok = all(isfinite, dad.T)
        @printf("  %-8s %6.1f  %10.3e  %10.3e  %10.3e  %10s\n",
            mass, α, err, maximum(abs, u_num), maximum(abs, u_ana), ok)
    end
end
