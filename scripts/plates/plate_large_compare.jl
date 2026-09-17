# large1 (BEM.jl atual): SS square, immovable edges, Q = q a⁴/(E h⁴)
# Pala w/h at Q = 5,10,15,20,25. Picard (placa_grande) vs Newton-AD.
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf
using NonlinearSolve
using ADTypes: AutoForwardDiff
using BEM.Plate
using .ThinPlate

const PALA_Q = [5.0, 10.0, 15.0, 20.0, 25.0]
const PALA_W = [0.5214, 0.8521, 1.080, 1.255, 1.399]

E, ν, h, a = 1e7, 0.3, 1.0, 1.0
# large1: C = -40 h⁴ E  →  Q_max = 40 at λ=1
q0 = 40 * E * h^4
props = ThinPlateProps(; E=E, ν=ν, h=h, q_c=q0)

println("="^64)
println(" large1 SS immovable — Pala vs Picard vs Newton")
println("="^64)

plate = build_square_plate(; a=a, n_el=6, bc="SSSS", props=props,
    corner_bc='F', n_internal=1)
assemble_plate!(plate; npg=10, singular=:guiggiani)

include(datadir("Laplace", "Laplace_dad.jl"))
msh = quadrado_elasticity(ndiv=8, show=false, nome="large_pe_cmp", Lx=a, Ly=a)
dad_pe = format2d(msh, Elasticity(E, ν, 1.0; plane_strain=false); pontointerno=true)
fill!(dad_pe.BC, 0)
fill!(dad_pe.BV, 0.0)
H_G_full_direct(dad_pe; npg=8, threaded=false)

prob = build_large_plate_problem(plate, dad_pe; npg_plate=10, npg_pe=8)
w_lin = linear_wmax_reference(prob; λ=1.0)
println("linear |w_c|/h at Q=40 = ", abs(w_lin) / h)

λ_max = 25 / 40   # stop at Pala's last Q
nsteps = 5

function run_method(label, nonlinear)
    plate.u .= 0
    p = build_large_plate_problem(plate, dad_pe; npg_plate=10, npg_pe=8)
    t = @elapsed res = solve_large_plate!(p; nsteps=nsteps, λ_max=λ_max,
        e_relax=0.5, abstol=1e-6, reltol=1e-6, maxiters=20, nonlinear=nonlinear)
    println("\n", label, "  (", round(t; digits=2), " s)")
    @printf("  %6s %10s %10s %10s\n", "Q", "w/h", "Pala", "rel")
    for (λ, wc) in zip(res.λ, res.w_center)
        Q = λ * 40
        wh = abs(wc) / h
        # nearest Pala
        k = argmin(abs.(PALA_Q .- Q))
        rel = abs(wh - PALA_W[k]) / PALA_W[k]
        @printf("  %6.1f %10.4f %10.4f %10.3f\n", Q, wh, PALA_W[k], rel)
    end
    return res
end

res_p = run_method("Picard", :picard)
res_n = run_method("Newton", :newton)

function residual_study(prob)
    println("\nResidual ‖R_free‖ at Q=10 from linear predictor")
    λ = 10 / 40
    x_lin = prob.A_pl \ (prob.b_bc .+ λ .* prob.q_load)
    resid_norm = function (x)
        y = BEM.Plate._gather_free(prob, x)
        p = (prob=prob, λ=λ, x_base=x)
        return norm(BEM.Plate.plate_residual_free(y, p))
    end
    u = pack_plate_u(prob, x_lin)
    w = extract_w_field(prob, u)
    Nxx, Nyy, Nxy = membrane_N_from_w(prob, w)
    qg = geometric_load(prob, w, Nxx, Nyy, Nxy)
    n = length(prob.plate.nodes)
    println("  w_c=", w[n+1], "  qg_c=", qg[n+1], "  max|Nxx|=", maximum(abs.(Nxx)))
    println("  ‖q_load‖=", norm(prob.q_load), "  ‖rhs_geo‖=",
        norm(BEM.Plate.geo_rhs_vector(prob, qg)))
    xp = copy(x_lin)
    println("  Picard:")
    for it in 0:8
        it > 0 && (xp = BEM.Plate._picard_plate_step(prob, λ, xp; niter=1, e=0.5))
        @printf("    %2d  %12.3e\n", it, resid_norm(xp))
    end
    xn = BEM.Plate._picard_plate_step(prob, λ, x_lin; niter=2, e=0.5)
    y0 = BEM.Plate._gather_free(prob, xn)
    p = (prob=prob, λ=λ, x_base=copy(xn))
    nlprob = NonlinearProblem(BEM.Plate.plate_residual_free, y0, p)
    sol = solve(nlprob, NewtonRaphson(; autodiff=AutoForwardDiff());
        abstol=1e-10, reltol=1e-10, maxiters=15)
    xn2 = BEM.Plate._scatter_free(prob, sol.u, p.x_base)
    @printf("  Newton after 2 Picard: ret=%s  ‖R‖=%12.3e\n",
        string(sol.retcode), resid_norm(xn2))
end
residual_study(prob)
println("Done.")
