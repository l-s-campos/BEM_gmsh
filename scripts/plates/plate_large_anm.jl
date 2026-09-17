# ANM + Padé smoke vs linear tangent (large1 scaling)
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf
using BEM.Plate
using .ThinPlate

E, ν, h, a = 1e7, 0.3, 1.0, 1.0
q0 = 40 * E * h^4
props = ThinPlateProps(; E=E, ν=ν, h=h, q_c=q0)
plate = build_square_plate(; a=a, n_el=3, bc="SSSS", props=props,
    corner_bc='F', n_internal=1)
assemble_plate!(plate; npg=8, singular=:analytic)

include(datadir("Laplace", "Laplace_dad.jl"))
msh = quadrado_elasticity(ndiv=4, show=false, nome="anm_pe", Lx=a, Ly=a)
dad_pe = format2d(msh, Elasticity(E, ν, 1.0; plane_strain=false); pontointerno=true)
fill!(dad_pe.BC, 0)
fill!(dad_pe.BV, 0.0)
H_G_full_direct(dad_pe; npg=6, threaded=false)

prob = build_large_plate_problem(plate, dad_pe; npg_plate=8, npg_pe=6)
wlin = linear_wmax_reference(prob; λ=1.0)
println("linear w/h @ Q=40 = ", abs(wlin) / h)
println("nfree = ", count(!, prob.is_kin))

x0 = zeros(length(prob.is_kin))
X, Λ = BEM.Plate._anm_series(prob, x0, 0.0; order=2, ε=0.05)
println("λ1 = ", Λ[2], "  ||x1|| = ", norm(X[2]), "  (λ1 should be 1)")
xlin = prob.A_pl \ (prob.b_bc .+ prob.q_load)
dxdλ = X[2] ./ Λ[2]
println("rel ||x1/λ1 - x_lin|| = ", norm(dxdλ - xlin) / (norm(xlin) + eps()))

xp, λp = BEM.Plate._anm_eval_pade(X, Λ, 0.25)
xt, λt = BEM.Plate._anm_eval_taylor(X, Λ, 0.25)
n = length(plate.nodes)
up = BEM.Plate.pack_plate_u(prob, xp)
ut = BEM.Plate.pack_plate_u(prob, xt)
println("Padé  a=0.25  ||x||=", norm(xp), "  w/h=", abs(up[2n+1])/h, "  λ=", λp)
println("Taylor a=0.25  ||x||=", norm(xt), "  w/h=", abs(ut[2n+1])/h, "  λ=", λt)
println("X2,X3,X4 norms ", norm(X[2]), " ", length(X)>3 ? norm(X[3]) : NaN, " ",
    length(X)>4 ? norm(X[4]) : NaN)

println("ANM–Padé to λ=0.25 (Q=10)...")
t = @elapsed res = solve_large_plate_anm!(prob; λ_max=0.25, order=3, ε=0.1,
    rtol=0.5, maxsteps=6, corrector=false)
println("  time ", round(t; digits=2), " s  nsteps ", length(res.λ) - 1)
for (λ, wc) in zip(res.λ, res.w_center)
    @printf("  λ=%6.3f  Q=%5.1f  w/h=%8.4f  (lin %8.4f)\n",
        λ, λ * 40, abs(wc) / h, abs(wlin) * λ / h)
end
println("Done.")
