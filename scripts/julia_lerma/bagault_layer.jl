# Short Bagault Fig. 1 / Fig. 3 trends. Full paper figures: bagault_paper.jl
using DrWatson
@quickactivate :BEM
using BEM.Contact
using LinearAlgebra, Printf, FFTW, Plots
gr()
default(size=(640, 420), linewidth=1.6, legendfontsize=8, guidefontsize=11,
        tickfontsize=9, titlefontsize=11, grid=false, framestyle=:box)

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const FIG = joinpath(ROOT, "plots", "julia_lerma", "layered")
mkpath(FIG)

function sphere_setup(; N=32, a=1.0, R=10.0, G=82000.0, ν=0.28)
    E = 2G * (1 + ν)
    δ = a^2 / R
    Estar = 2G / (1 - ν)
    p0 = 2Estar * a / (π * R)
    L = 2.4 * a
    hx = 2L / N
    x = collect(range(-L + hx / 2, L - hx / 2; length=N))
    gap = [(xi^2 + yj^2) / (2R) for xi in x, yj in x]
    return (; N, a, R, E, G, ν, δ, p0, hx, x, gap)
end

function p_profile(sol, x, ymid)
    j = argmin(abs.(x .- ymid))
    return x, sol.p[:, j]
end

s = sphere_setup()
println(@sprintf("Hertz p0=%.1f  δ=%.4f  a=%.2f", s.p0, s.δ, s.a))

# --- Fig. 1 trend: isotropic coating, Zc = a/2 ---
plt1 = plot(xlabel="x / a", ylabel="p / p_Hertz", title="isotropic coating (Zc = a/2)")
for (ratio, col) in ((0.25, :cyan), (1.0, :black), (4.0, :red))
    hs = isotropic_coated(ratio * s.E, s.ν, s.a / 2, s.E, s.ν; hx=s.hx, hy=s.hx)
    sol = solve_normal_contact(s.gap, s.δ, hs; tol=1e-5)
    xv, pv = p_profile(sol, s.x, 0.0)
    plot!(plt1, xv ./ s.a, pv ./ s.p0; color=col, label="Ec/Es=$(ratio)")
    println(@sprintf("  Ec/Es=%4.2f  pmax/pH=%.3f", ratio, maximum(sol.p) / s.p0))
end
savefig(plt1, joinpath(FIG, "bagault_fig1_coating.png"))
println("  wrote ", joinpath(FIG, "bagault_fig1_coating.png"))

# --- Fig. 3: homogeneous orthotropic E3 = 2 E ---
Co = orthotropic_C(s.E, s.E, 2s.E, s.ν, s.ν, s.ν, s.G, s.G, s.G)
hsO = homogeneous(Co; hx=s.hx, hy=s.hx)
solO = solve_normal_contact(s.gap, s.δ, hsO; tol=1e-5)
hsI = isotropic_halfspace(s.E, s.ν; hx=s.hx, hy=s.hx)
solI = solve_normal_contact(s.gap, s.δ, hsI; tol=1e-5)
xv, pv = p_profile(solO, s.x, 0.0)
xi, pi_ = p_profile(solI, s.x, 0.0)
plt3 = plot(xi ./ s.a, pi_ ./ s.p0; label="isotropic", xlabel="y / a",
            ylabel="p / p_Hertz", title="orthotropic E3 = 2 E1")
plot!(plt3, xv ./ s.a, pv ./ s.p0; label="E3=2E")
savefig(plt3, joinpath(FIG, "bagault_fig3_orthotropic.png"))
println(@sprintf("  Fig.3  pmax/pH=%.3f (paper ~1.33)", maximum(solO.p) / s.p0))
println("  wrote ", joinpath(FIG, "bagault_fig3_orthotropic.png"))
