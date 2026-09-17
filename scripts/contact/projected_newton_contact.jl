# =============================================================================
# Projected Coulomb BEM–BEM frictional contact demo
# (algorithms after Rodríguez-Tembleque & Abascal, IJNME 2013)
# Two elastic blocks, NTN pairing — compare active-set / SSN / proj-GNM / proj-Newton
# =============================================================================
using DrWatson
@quickactivate :BEM

using LinearAlgebra
using Statistics
using Printf

include(datadir("elastico", "two_blocks_contact.jl"))

E, ν = 100.0, 0.3
props = Elasticity(E, ν, 1.0; plane_strain=true)
gap0 = 0.02
δ_end = 0.04
kwargs = (W=1.0, H=0.4, gap=gap0, μ=0.3, ndiv_bot=6, ndiv_top=6, ndiv_y=3)

println("="^64)
println(" Projected Newton BEM–BEM contact demo  (δ_end=$δ_end)")
println("="^64)

function run_solver(name, solver; kw...)
    prob = load_two_blocks_contact(props; kwargs..., nome="proj_$name")
    t0 = time()
    solve_contact_friction_stepped!(prob; δ_end=δ_end, nsteps=10, tol=1e-7,
        maxiter=100, npg=10, method=:ntn, solver=solver, kw...)
    dt = time() - t0
    closed = filter(cp -> abs(cp.state) != 1, prob.contacts)
    tn = isempty(closed) ? 0.0 : mean(abs(cp.tn) for cp in closed)
    tt = isempty(closed) ? 0.0 : mean(abs(cp.tt) for cp in closed)
    @printf("  %-14s  closed=%2d/%2d  mean|tn|=%8.4f  mean|tt|=%8.4f  time=%.2fs\n",
        String(solver), length(closed), length(prob.contacts), tn, tt, dt)
    return (; tn, tt, n=length(closed), dt, prob)
end

r_as = run_solver("as", :activeset)
r_ss = run_solver("ssn", :ssn)
r_pg = run_solver("pgnm", :proj_gnm)
r_pn = run_solver("pnew", :proj_newton)
r_gl = run_solver("gnmls", :gnmls)

println("\nRelative mean|tn| vs active-set:")
for (lab, r) in (("ssn", r_ss), ("proj_gnm", r_pg), ("proj_newton", r_pn), ("gnmls", r_gl))
    @printf("  %-12s  tn/tn_AS = %.3f\n", lab, r.tn / max(r_as.tn, eps()))
end

println("\nDone.")
