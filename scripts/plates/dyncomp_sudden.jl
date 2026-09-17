# Sudden (step) pressure on Dynamic_Composite_Plate EjemploDin_10.
# MATLAB Dynsolver: Houbolt from rest, FQ = Q (constant), dt=1e-3.
# Gold: Navier 2 w_stat (undamped SDOF) + shipped ANSYS SOLFEM + BEM Aprfun3.
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf, DelimitedFiles
using BEM.Plate

println("="^72)
println(" Sudden load  SS [0/90]s  Wang FSDT  Houbolt + DIBEM")
println(" MATLAB Dynamic_Composite_Plate  EjemploDin_10 / Dynsolver FQ=Q")
println("="^72)

E1, E2, ν12 = 4e6, 2e6, 0.25
G12, G13, G23 = 1e6, 1e6, 5e5
a, h, q, ρ = 1.0, 0.1, 1.0, 4000.0
plies = [(E1, E2, ν12, G12, θ, h / 4) for θ in (0.0, 90.0, 90.0, 0.0)]
props = laminate_fsdt_props(plies; Ks=5 / 6, G13=G13, G23=G23, q_c=q, ρ=ρ, nθ=8)
wN = navier_w_ss_fsdt(a / 2, a / 2, props; a=a, q=q)
D11, D22, D12, D66 = props.D[1, 1], props.D[2, 2], props.D[1, 2], props.D[3, 3]
A44, A55 = props.AT[1, 1], props.AT[2, 2]
α = π / a
KK = [D11*α^2+D66*α^2+A55  (D12+D66)*α^2  A55*α
      (D12+D66)*α^2        D66*α^2+D22*α^2+A44  A44*α
      A55*α                A44*α          A55*α^2+A44*α^2]
Kred = KK[3, 3] - dot(KK[1:2, 3], KK[1:2, 1:2] \ KK[1:2, 3])
T11 = 2π / sqrt(Kred / (ρ * h))

@printf("  Navier w_stat=%.6e  2 w_stat=%.6e  T11=%.4f s  (first peak ~ T11/2=%.3f)\n",
    wN, 2wN, T11, T11 / 2)

function parse_tw(path)
    txt = read(path, String)
    tw = Tuple{Float64,Float64}[]
    for m in eachmatch(r"([+-]?(?:\d+\.?\d*|\.\d+)(?:[Ee][+-]?\d+)?)\s+([+-]?(?:\d+\.?\d*|\.\d+)(?:[Ee][+-]?\d+)?)", txt)
        t = parse(Float64, m.captures[1])
        w = parse(Float64, m.captures[2])
        (0.0 <= t <= 1.2 && abs(w) < 1e-3) && push!(tw, (t, w))
    end
    return tw
end

function peak_of(tw)
    i = argmax(abs(w) for (_, w) in tw)
    return tw[i]
end

mat = "/home/lsc/Downloads/BEM_Plate_Shell_Book_Juseche-main/Dynamic_Composite_Plate/Lam_0_90_90_0_Din"
fem = parse_tw(joinpath(mat, "SOLFEM.m"))
bem3 = parse_tw(joinpath(mat, "Aprfun3.dat"))
tf, wf = peak_of(fem)
tb, wb = peak_of(bem3)
@printf("  ANSYS SOLFEM     peak t=%.3f  w=%.6e  /2Nav=%.3f\n", tf, wf, abs(wf) / (2wN))
@printf("  MATLAB Aprfun3   peak t=%.3f  w=%.6e  /2Nav=%.3f\n", tb, wb, abs(wb) / (2wN))

function run_houbolt(; n_el, n_int=1, dt=1e-3, tmax=0.5, rbf=PHS(2; poly_deg=1))
    mesh = build_square_fsdt(; a=a, n_el=n_el, bc="SSSS", props=props,
        n_internal=n_int, p=2)
    assemble_fsdt!(mesh; npg=8, nsub=6)
    dibem_fsdt!(mesh; npg=8, rbf=rbf)
    solve_fsdt!(mesh)
    wstat = fsdt_w_int(mesh, 1)
    res = solve_fsdt_houbolt!(mesh; dt=dt, tmax=tmax, mass=:raw)  # q(t)=1, rest IC
    imax = argmax(abs.(res.w_center))
    return (n_el=n_el, n_int=n_int, dt=dt, wstat=wstat,
        peak=res.w_center[imax], tpeak=res.t[imax], t=res.t, w=res.w_center)
end

println("\n  Julia Houbolt  q(t)=H(t)  (sudden). n_int=1 DIBEM M is unstable; use 9.")
@printf("  %-8s %4s %6s %12s %12s %8s %10s %10s\n",
    "n_el", "nint", "dt", "w_stat", "peak", "t_peak", "peak/2Nav", "peak/2stat")
rows = []
for (n_el, n_int, dt, tmax) in ((2, 9, 1e-3, 0.5), (2, 9, 5e-3, 0.5),
        (4, 9, 1e-3, 0.5), (4, 9, 5e-3, 0.5))
    r = run_houbolt(; n_el=n_el, n_int=n_int, dt=dt, tmax=tmax)
    @printf("  %-8d %4d %6.1e %12.4e %12.4e %8.3f %10.3f %10.3f\n",
        n_el, n_int, dt, r.wstat, r.peak, r.tpeak,
        abs(r.peak) / (2wN), abs(r.peak) / (2 * abs(r.wstat) + eps()))
    push!(rows, r)
end

r0 = rows[1]  # n_el=2, 9 internals, dt=1e-3  (MATLAB EjemploDin_10 dt)
out = joinpath(@__DIR__, "dyncomp_sudden.csv")
open(out, "w") do io
    println(io, "t,w_julia,w_fem,w_matlab_aprfun3")
    for (t, w) in zip(r0.t, r0.w)
        wf_ = wf_interp = 0.0
        # nearest FEM / BEM samples
        function nearest(tw, t0)
            isempty(tw) && return 0.0
            j = argmin(abs(s[1] - t0) for s in tw)
            return tw[j][2]
        end
        println(io, join((t, w, nearest(fem, t), nearest(bem3, t)), ","))
    end
end
@printf("\n  vs ANSYS peak  Julia n_el=2 dt=1e-3  %.2f %%\n",
    100 * abs(abs(r0.peak) - abs(wf)) / abs(wf))
@printf("  vs MATLAB Aprfun3 peak                    %.2f %%\n",
    100 * abs(abs(r0.peak) - abs(wb)) / abs(wb))
@printf("  vs Navier 2 w_stat                        %.2f %%\n",
    100 * abs(abs(r0.peak) - 2wN) / (2wN))
println("  wrote ", out)
println("Done.")
