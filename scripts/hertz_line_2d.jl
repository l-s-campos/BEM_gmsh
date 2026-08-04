# 2D line contact vs Hertz cylinder theory
using DrWatson
@quickactivate :BEM

println("="^60)
println(" 2D Flamant BEM vs Hertz line contact")
println("="^60)

G, ν = 1.0, 0.3
E = 2G * (1 + ν)
N = 256
L = 2.0
x = range(-L, L; length=N) |> collect
h = x[2] - x[1]
hp = ElasticHalfPlane2D(G, ν; h=h)

R = 1.0
F_target = 0.05
hz = hertz_line(F_target, R, hp)
println("Hertz: a=$(hz.a), p0=$(hz.p0), E*=$(hz.Estar)")

# gap for cylinder: g0 = x²/(2R)
gap0 = @. x^2 / (2R)
# find δ such that force ≈ F_target by bisection on indentation
function force_of(δ)
    sol = solve_line_contact(gap0, δ, hp; tol=1e-12)
    return sol.force, sol
end

# Hertz δ is not unique on infinite half-plane (log); match via contact half-width
# Impose δ so that contact radius ≈ a_Hertz
δ_lo, δ_hi = 1e-6, 0.5
sol = nothing
for _ in 1:40
    δ_mid = 0.5 * (δ_lo + δ_hi)
    F, sol = force_of(δ_mid)
    a_num = 0.5 * h * count(sol.contact)   # rough half-width
    if a_num < hz.a
        δ_lo = δ_mid
    else
        δ_hi = δ_mid
    end
end
F_num = sol.force
a_num = begin
    idx = findall(sol.contact)
    isempty(idx) ? 0.0 : 0.5 * (x[maximum(idx)] - x[minimum(idx)])
end
p0_num = maximum(sol.p)

println("BEM:   a=$a_num, p0=$p0_num, F=$F_num")
println("a_num/a_hz  = ", a_num / hz.a)
println("p0_num/p0   = ", p0_num / hz.p0)
println("F_num/F_hz  = ", F_num / F_target)

# pressure shape comparison at Hertz force level
p_hz, _ = hertz_line_pressure(F_num, R, hp, x)

fig = Figure(size=(800, 400))
ax = Axis(fig[1, 1]; xlabel="x", ylabel="p", title="Line contact pressure")
lines!(ax, x, sol.p; label="BEM Flamant")
lines!(ax, x, p_hz; label="Hertz", linestyle=:dash)
axislegend(ax)
display(fig)
println("Done.")
