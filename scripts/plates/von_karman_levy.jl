# Levy clamped isotropic square plate, von Kármán, Kirchhoff DIBEM.
# Twin: fenics experimentos/von_karman_plate/von_karman_isotropic.py
# Levy, NACA Report 740 / TN 847, Table 5 (ν = 0.316).
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf, Statistics
using Plots
using BEM.Plate

const Point2D = SVector{2,Float64}

const A = 1.0
const H = 0.01
const E = 1.0e6
const NU = 0.316
const N_EL = 8
const N_INT = 81
const NPG = 8
const RBF = PHS(2; poly_deg=1)

const LEVY_Q = [17.79, 38.3, 63.4, 95.0, 134.9, 184.0, 245.0, 318.0, 402.0]
const LEVY_W = [0.237, 0.471, 0.695, 0.912, 1.121, 1.323, 1.521, 1.714, 1.902]
const W_OVER_QA4_D = 0.001263

q_from_Q(Q) = Q * E * H^4 / A^4
function linear_wc_over_h(Q)
    D = E * H^3 / (12 * (1 - NU^2))
    return W_OVER_QA4_D * q_from_Q(Q) * A^4 / (D * H)
end

function center_grid(a, nint)
    g = ceil(Int, sqrt(nint))
    xs = range(a / (g + 1), a * g / (g + 1); length=g)
    pts = Point2D[Point2D(a / 2, a / 2)]
    for y in xs, x in xs
        hypot(x - a / 2, y - a / 2) < 1e-12 && continue
        push!(pts, Point2D(x, y))
        length(pts) >= nint && break
    end
    return pts
end

function plate_center_k(dad)
    xc, yc = A / 2, A / 2
    best, bd = 1, Inf
    for k in 1:dad.ni
        p = dad.internalNodes[k]
        d = (p[1] - xc)^2 + (p[2] - yc)^2
        if d < bd
            bd = d
            best = k
        end
    end
    return best
end

function levy_membrane_bc!(dad, a)
    fill!(dad.BC, 1)
    fill!(dad.BV, 0.0)
    @inbounds for i in 1:dad.n
        p = dad.Nodes[i]
        if p[1] < 1e-8 * a || p[1] > a * (1 - 1e-8)
            dad.BC[2i - 1] = 0
            dad.BV[2i - 1] = 0.0
        end
        if p[2] < 1e-8 * a || p[2] > a * (1 - 1e-8)
            dad.BC[2i] = 0
            dad.BV[2i] = 0.0
        end
    end
    return dad
end

function wc_over_h(prob)
    k = plate_center_k(prob.plate)
    return plate_w_int(prob.plate, k) / H
end

println("="^72)
println(" Levy clamped square  von Kármán  Kirchhoff DIBEM")
println("  a/h=$(A / H)  ν=$NU  n_el/edge=$N_EL  n_internal=$N_INT")
println("="^72)

Qmax = LEVY_Q[end]
q0 = q_from_Q(Qmax)
props = ThinPlateProps(; E=E, ν=NU, h=H, q_c=q0)
internal = center_grid(A, N_INT)
plate = build_square_plate(; a=A, n_el=N_EL, bc="CCCC", props=props,
    n_internal=length(internal), internal=internal, corner_bc='C', p=2)
assemble_plate!(plate; npg=NPG)
dibem_plate!(plate; npg=NPG, rbf=RBF, apply_load=true)

include(datadir("Laplace", "Laplace_dad.jl"))
msh = quadrado_elasticity(; ndiv=N_EL + 1, show=false, nome="levy_pe",
    Lx=A, Ly=A, ordem=1)
dad_pe = format2d(msh, Elasticity(E, NU, 1.0; plane_strain=false);
    pontointerno=true)
levy_membrane_bc!(dad_pe, A)
set_internal_nodes!(dad_pe, plate.internalNodes)
H_G_full_direct(dad_pe; npg=NPG, threaded=true)
dibem_elasticity!(dad_pe; npg=NPG, rbf=RBF)

prob = build_large_plate_problem(plate, dad_pe; npg_plate=NPG, npg_pe=NPG,
    plate_dibem=true, rbf=RBF)

k0 = plate_center_k(plate)
w_lin = linear_wmax_reference(prob; λ=LEVY_Q[1] / Qmax)
w_lin_h = abs(w_lin) / H
w_kir = linear_wc_over_h(LEVY_Q[1])
@printf("Linear at Q=%.2f:  BEM w_c/h=%.4f  Kirchhoff Levy=%.4f  e=%.2f %%\n",
    LEVY_Q[1], w_lin_h, w_kir, 100 * abs(w_lin_h - w_kir) / w_kir)
@printf("  centre internal k=%d  (%.4f, %.4f)\n", k0,
    plate.internalNodes[k0][1], plate.internalNodes[k0][2])
@printf("  ‖q_load‖=%.3e  ndof_plate=%d  ndof_mem=%d\n",
    norm(prob.q_load), size(prob.A_pl, 1), size(prob.A_pe, 1))

println("\nNonlinear continuation (Levy Table 5 loads)")
λ_path = LEVY_Q ./ Qmax
t = @elapsed res = solve_large_plate!(prob; λ_path=λ_path, e_relax=0.4,
    abstol=1e-6, reltol=1e-6, maxiters=25, nonlinear=:newton)

wcs = abs.(res.w_center) ./ H
finite = isfinite.(wcs) .& (wcs .< 10)
@printf("\n  %8s  %9s  %9s  %8s\n", "Q", "BEM w/h", "Levy w/h", "error %")
err = fill(NaN, length(wcs))
for i in eachindex(LEVY_Q)
    if finite[i]
        err[i] = 100 * (wcs[i] - LEVY_W[i]) / LEVY_W[i]
        @printf("  %8.2f  %9.4f  %9.3f  %8.2f\n", LEVY_Q[i], wcs[i], LEVY_W[i], err[i])
    else
        @printf("  %8.2f  %9s  %9.3f  %8s\n", LEVY_Q[i], "diverged", LEVY_W[i], "—")
    end
end
okerr = filter(isfinite, err)
@printf("max |error| = %.2f %%   (%.1f s)\n",
    isempty(okerr) ? NaN : maximum(abs.(okerr)), t)

Q_fine = range(0.0, Qmax; length=80)
w_k_line = linear_wc_over_h.(Q_fine)
plt = plot(Q_fine, w_k_line; color=:grey, linestyle=:dash, lw=1.4,
    label="linear Kirchhoff (Levy 0.001263)",
    xlabel=raw"$Q = q a^4 / (E h^4)$", ylabel=raw"$w_c / h$",
    title="Clamped isotropic square, von Kármán DIBEM",
    xlims=(0, 430), ylims=(0, 2.35), legend=:topleft, size=(720, 520),
    legend_foreground_color=nothing)
plot!(plt, [0.0; LEVY_Q[finite]], [0.0; wcs[finite]]; color=:steelblue, lw=1.9,
    marker=:circle, markersize=5, label="von Kármán BEM (Kirchhoff DIBEM)")
scatter!(plt, LEVY_Q, LEVY_W; marker=:square, markersize=7, markercolor=:white,
    markerstrokecolor=:firebrick, markerstrokewidth=1.4,
    label="Levy, NACA R-740, Table 5")
for i in eachindex(LEVY_Q)
    finite[i] || continue
    annotate!(plt, LEVY_Q[i], wcs[i] - 0.09,
        text(@sprintf("%.3f", wcs[i]), 7, :steelblue))
end
out = joinpath(@__DIR__, "von_karman_levy.png")
savefig(plt, out)
csv = joinpath(@__DIR__, "von_karman_levy.csv")
open(csv, "w") do io
    println(io, "Q,w_bem_over_h,w_levy_over_h,error_pct")
    for i in eachindex(LEVY_Q)
        @printf(io, "%.2f,%.6f,%.4f,%.3f\n",
            LEVY_Q[i], wcs[i], LEVY_W[i], err[i])
    end
end
println("wrote ", out)
println("wrote ", csv)
println("done")
