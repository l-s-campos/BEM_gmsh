# Smoke: ForwardDiff Jacobian of mixed residual vs A at the origin
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf
using BEM.Plate
using .ThinPlate

E, ν, h, a = 1e7, 0.3, 1.0, 1.0
q0 = 40 * E * h^4
props = ThinPlateProps(; E=E, ν=ν, h=h, q_c=q0)
plate = build_square_plate(; a=a, n_el=2, bc="SSSS", props=props,
    corner_bc='F', n_internal=1)
assemble_plate!(plate; npg=6, singular=:analytic)

include(datadir("Laplace", "Laplace_dad.jl"))
msh = quadrado_elasticity(ndiv=3, show=false, nome="ad_pe", Lx=a, Ly=a)
dad_pe = format2d(msh, Elasticity(E, ν, 1.0; plane_strain=false); pontointerno=true)
fill!(dad_pe.BC, 0)
fill!(dad_pe.BV, 0.0)
H_G_full_direct(dad_pe; npg=6, threaded=false)

prob = build_large_plate_problem(plate, dad_pe; npg_plate=6, npg_pe=6)
x0 = zeros(length(prob.is_kin))
λ0 = 0.0
t = @elapsed J = BEM.Plate._ad_jacobian(prob, x0, λ0)
R0 = BEM.Plate._residual_xλ(prob, x0, λ0)
dxJ = J \ prob.q_load
dxA = prob.A_pl \ prob.q_load
@printf("ndof = %d  J %d×%d  finite = %s  time = %.2f s\n",
    length(x0), size(J, 1), size(J, 2), string(all(isfinite, J)), t)
@printf("||R(0,0)|| = %.3e\n", norm(R0))
@printf("||J-A||/||A|| = %.3e\n", norm(J - prob.A_pl) / (norm(prob.A_pl) + eps()))
@printf("||J\\q - A\\q|| / ||A\\q|| = %.3e\n",
    norm(dxJ - dxA) / (norm(dxA) + eps()))

# One Newton step at λ=0.1 from the linear guess (AD J, not A)
λ = 0.1
xlin = prob.A_pl \ (prob.b_bc .+ λ .* prob.q_load)
xN = BEM.Plate._newton_ad!(prob, xlin, λ; maxiters=3, atol=1e-10)
n = length(plate.nodes)
uN = BEM.Plate.pack_plate_u(prob, xN)
@printf("Newton AD λ=0.1  finite = %s  w/h = %.4f  ||R|| = %.3e\n",
    string(all(isfinite, xN)), abs(uN[2n+1]) / h,
    norm(BEM.Plate._residual_xλ(prob, xN, λ)))
println("Done.")
