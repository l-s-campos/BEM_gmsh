# Multibody elasticity contact — NTN vs NTS on two blocks
using DrWatson
@quickactivate :BEM

using LinearAlgebra
using Statistics
using Printf

include(datadir("elastico", "two_blocks_contact.jl"))

println("="^64)
println(" Multibody elasticity contact: NTN vs NTS")
println("="^64)

E, ν = 100.0, 0.3
props = Elasticity(E, ν, 1.0; plane_strain=true)
gap = 0.02
δ = 0.035
kn = 20 * E / 0.4
kt = 0.5 * kn

function run_case(method; frame=:local, ndiv_bot=6, ndiv_top=6)
    prob = load_two_blocks_contact(props;
        W=1.0, H=0.4, gap=gap, μ=0.3,
        ndiv_bot=ndiv_bot, ndiv_top=ndiv_top, ndiv_y=3,
        nome="mb_$(method)_$(frame)_$(ndiv_bot)x$(ndiv_top)")
    t0 = time()
    solve_multibody_elasticity_contact!(prob; method=method, frame=frame, kn=kn, kt=kt,
        δ=δ, maxiter=60, ω=0.45, npg=10,
        slave_reg=1, master_reg=2)
    dt = time() - t0
    n_open = count(cp -> cp.state == 1, prob.contacts)
    n_slip = count(cp -> cp.state == 2, prob.contacts)
    n_stick = count(cp -> cp.state == 3, prob.contacts)
    tn = [cp.tn for cp in prob.contacts]
    @printf("  %s/%s  pairs=%2d  open/slip/stick=%d/%d/%d  mean tn=%8.3f  |u|_max=%.4f  (%.2fs)\n",
        rpad(String(method), 4), rpad(String(frame), 6), length(prob.contacts),
        n_open, n_slip, n_stick,
        mean(tn), max(maximum(abs, prob.regions[1].u), maximum(abs, prob.regions[2].u)), dt)
    return prob
end

println("\nMatching meshes (ndiv=6/6), local frame (Leonardo §4.7):")
run_case(:ntn; frame=:local, ndiv_bot=6, ndiv_top=6)
run_case(:nts; frame=:local, ndiv_bot=6, ndiv_top=6)

println("\nMatching meshes, global frame (legacy):")
run_case(:ntn; frame=:global, ndiv_bot=6, ndiv_top=6)

println("\nNon-matching (ndiv_bot=10, ndiv_top=6) — NTS natural:")
run_case(:ntn; frame=:local, ndiv_bot=10, ndiv_top=6)
run_case(:nts; frame=:local, ndiv_bot=10, ndiv_top=6)

println("\nDone.")
