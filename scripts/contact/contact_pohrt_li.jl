# Pohrt & Li (2014) — half-space contact BEM demo
# Hertz normal contact + Mindlin-like partial slip on a sphere
using DrWatson
@quickactivate :BEM
using BEM.Contact

using LinearAlgebra

println("="^60)
println(" Pohrt–Li half-space contact BEM")
println("="^60)

# Material: steel-like, ν=0.3
E = 1.0
ν = 0.3
G = E / (2(1 + ν))
hs = ElasticHalfSpace(G, ν)

# Grid
N = 64
L = 2.0
x = range(-L, L; length=N)
y = range(-L, L; length=N)
hx = x[2] - x[1]
hy = y[2] - y[1]
hs = ElasticHalfSpace(G, ν; hx=hx, hy=hy)

# Spherical indenter gap: g0 = r²/(2R)
R_sphere = 1.0
gap0 = zeros(N, N)
@inbounds for j in 1:N, i in 1:N
    gap0[i, j] = (x[i]^2 + y[j]^2) / (2R_sphere)
end

δ = 0.05   # indentation
println("\n[1] Normal frictionless contact  δ=$δ  grid=$N×$N")
sol = solve_normal_contact(gap0, δ, hs; tol=1e-7)
println("  contact nodes : ", count(sol.contact))
println("  normal force  : ", sol.force)

# Analytical Hertz
Estar = contact_modulus(hs)
a_hz = sqrt(R_sphere * δ)
F_hz = 4/3 * Estar * sqrt(R_sphere) * δ^(3/2)
p0_hz = 1.5 * F_hz / (π * a_hz^2)
println("  Hertz a, F, p0: ", a_hz, ", ", F_hz, ", ", p0_hz)
println("  F_num / F_hz  : ", sol.force / F_hz)
println("  max p / p0    : ", maximum(sol.p) / p0_hz)

# Influence sanity: Kzz(0,0) > 0
K0 = influence_coeff(Kzz, 0, 0, hs)
println("\n[2] Influence Kzz(0,0) = ", K0, " (must be > 0)")

# Partial slip
μf = 0.3
d = 0.2 * μf * δ   # moderate tangential displacement
println("\n[3] Partial slip  μ=$μf  d=$d")
ps = solve_partial_slip(sol.p, sol.contact, d, μf, hs; direction=:x, tol=1e-6)
println("  stick nodes   : ", count(ps.stick))
println("  slip  nodes   : ", count(ps.slip))
println("  tangential F  : ", ps.force_t)
println("  |Ft|/(μ Fn)   : ", abs(ps.force_t) / (μf * sol.force))

# Optional plot
try
    using Plots
    ax1 = heatmap(x, y, sol.p'; title="normal pressure p", aspect_ratio=:equal,
        xlabel="x", ylabel="y")
    zone = zeros(N, N)
    zone[ps.stick] .= 1
    zone[ps.slip] .= 2
    ax2 = heatmap(x, y, zone'; title="stick (blue) / slip (red)", aspect_ratio=:equal,
        xlabel="x", ylabel="y", color=cgrad([:white, :dodgerblue, :crimson]),
        clims=(0, 2))
    fig = plot(ax1, ax2; layout=(1, 2), size=(900, 400))
    display(fig)
catch e
    @warn "plot skipped" e
end

println("\nDone.")
