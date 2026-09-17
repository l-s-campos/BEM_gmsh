# Quarter thick cylinder, constant-cell von Mises. Compare outer radial
# displacement to Hill / Hodge.
using DrWatson
@quickactivate :BEM

include(datadir("elastico", "iso", "pressurized_tube.jl"))

a, b = Ra, Rb
E, ν, σY = 200_000.0, 0.3, 240.0
k = σY / sqrt(3)
pel = k * (1 - (a / b)^2)
p = 1.4 * pel
println("p_el = $pel  p = $p  p_lim = $(2k * log(b / a))")

props = Elasticity(E, ν, 1.0; plane_strain=true)
msh = mesh_pressurized_tube(; ndiv=10, nome="cyl_plastic")
dad = format2d(msh, props; pontointerno=true, tipo=1)
apply_radius_pressure!(dad, p; R=a, tol=4.0)
assemble!(dad, 12)
solve_elastoplastic!(dad, VonMises(σY=σY, H′=0.0); nsteps=8, maxiter=30, tol=1e-4)

ana = ana_thick_cylinder_plastic(b; a=a, b=b, p=p, σY=σY, E=E, ν=ν)
ub = Float64[]
for i in 1:dad.n
    r = hypot(dad.Nodes[i][1], dad.Nodes[i][2])
    abs(r - b) < 4.0 || continue
    push!(ub, hypot(dad.u[2i-1], dad.u[2i]))
end
println("cells = ", length(extract_domain_cells(dad)),
    "  plastic cells = ", count(>(1e-12), dad.plastic_strain))
println("u(b) BEM median = ", median(ub), "  analytic = ", ana.u,
    "  plastic front c = ", ana.c)
