# Large-deflection SS square plate (von Kármán) — cf. placa_large.jl / large1
using DrWatson
@quickactivate :BEM
using BEM.Plate
using .ThinPlate

println("="^60)
println(" Large plate (von Kármán) — NonlinearSolve Newton")
println("="^60)

E, ν, h, a = 1e7, 0.3, 1.0, 1.0
# large1 load scale: C = -40 h^4 E  → q = 40 E h^4 (sign → downward + in our conv.)
q0 = 40 * E * h^4   # magnitude; plate uses q_c positive → check sign vs ana
props = ThinPlateProps(; E=E, ν=ν, h=h, q_c=q0)

# plate mesh (SS, free corners, centre internal point)
plate = build_square_plate(; a=a, n_el=4, bc="SSSS", props=props,
    corner_bc='F', n_internal=1)
assemble_plate!(plate; npg=8)

# membrane: square with fixed edges (immovable)
include(datadir("Laplace", "Laplace_dad.jl"))
msh = quadrado_elasticity(ndiv=6, show=false, nome="large_pe", Lx=a, Ly=a)
# fix all in-plane displacements
dad_pe = format2d(msh, Elasticity(E, ν, 1.0; plane_strain=false); pontointerno=true)
fill!(dad_pe.BC, 0)
fill!(dad_pe.BV, 0.0)
H_G_full_direct(dad_pe; npg=8, threaded=false)

prob = build_large_plate_problem(plate, dad_pe; npg_plate=8, npg_pe=8)
w_lin = linear_wmax_reference(prob; λ=1.0)
println("linear |w_c| / h = ", abs(w_lin) / h)

res = solve_large_plate!(prob; nsteps=8, λ_max=1.0, e_relax=0.5,
    abstol=1e-5, reltol=1e-5, maxiters=25, nonlinear=:anm)

println("load factor λ     w_c/h")
for (λ, wc) in zip(res.λ, res.w_center)
    println("  ", round(λ; digits=3), "   ", round(wc / h; sigdigits=4))
end
println("final |w_c|/h = ", abs(res.w_center[end]) / h,
    "   linear |w_c|/h = ", abs(w_lin) / h)
println("membrane stiffening ratio = ",
    abs(res.w_center[end]) / (abs(w_lin) + eps()))
println("Done.")
