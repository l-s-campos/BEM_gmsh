# Compare mesh order / density on raw nodal shear (no postprocessing)
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Statistics, Printf

include(datadir("elastico", "two_blocks_contact.jl"))
include(datadir("elastico", "dad_5d_contact.jl"))

par = loyola_dad5d_params()
μ = par.μ

function contato_newton!(prob; tol=1e-9, maxiter=80, npg=12)
    ctx = BEM._contact_friction_setup(prob; method=:ntn, npg=npg)
    prep, pairs = ctx.prep, ctx.pairs
    h = [cp.gap0 for cp in pairs]
    x0 = has_cache(prob.regions[1], :contact_x) &&
         length(prob.regions[1].contact_x) == ctx.N ?
         collect(Float64, prob.regions[1].contact_x) : zeros(ctx.N)
    if !has_cache(prob.regions[1], :contact_x) || length(prob.regions[1].contact_x) != ctx.N
        for cp in pairs; cp.state = 3; cp.ut_lock = 0.0; end
        x0 = zeros(ctx.N)
    end
    ok = false
    for it in 1:maxiter
        it > 1 && BEM._verify_contact_states!(pairs, prep, h, x0; epsc=1e-7)
        A, b = BEM._assemble_contact_system(prep, pairs, h, x0)
        x = A \ b
        dist = norm(x - x0); x0 = x
        dist < tol && (ok = true; break)
    end
    BEM._verify_contact_states!(pairs, prep, h, x0; epsc=1e-7)
    BEM._update_contact_ut_locks!(pairs, prep, h, x0)
    BEM._scatter_contact_solution!(prob, prep, pairs, x0)
    set_cache!(prob.regions[1]; contact_x=copy(x0))
    return ok
end

function qstats(prob)
    fr = contact_interface_xyτ(prob)
    cl = findall(s -> abs(s) != 1, fr.state)
    isempty(cl) && return (; p0=0.0, Q=0.0, P=0.0, rms_q=0.0, odd=0.0, even=0.0, d2=NaN, ncl=0, L2q=NaN)
    x = fr.x[cl]; q = .-fr.tt[cl]; p = .-fr.tn[cl]; st = fr.state[cl]
    w = ones(length(x))
    if length(x) >= 2
        w[1] = abs(x[2]-x[1]); w[end] = abs(x[end]-x[end-1])
        for i in 2:length(x)-1; w[i] = 0.5*abs(x[i+1]-x[i-1]); end
    end
    P = sum(p .* w); Q = sum(q .* w); p0 = maximum(p)
    a = p0 > 0 ? 2P/(π*p0) : 0.0
    odd = even = 0.0; n = 0
    for i in eachindex(x)
        x[i] < -1e-9 || continue
        j = argmin(abs.(x .+ x[i]))
        qi, qj = q[i], q[j]
        even += ((qi + qj)/2)^2
        odd  += ((qi - qj)/2)^2
        n += 1
    end
    d2m = length(q) >= 3 ? mean(abs, [q[i+1] - 2q[i] + q[i-1] for i in 2:length(q)-1]) : NaN
    L2q = NaN
    if abs(Q) > 1 && a > 0
        qA = cattaneo_shear(collect(x), a, p0, Q, μ, abs(P))
        L2q = sqrt(sum(abs2, q .- qA)) / max(sqrt(sum(abs2, qA)), eps())
    end
    return (; p0, Q, P, a, rms_q=sqrt(mean(abs2, q)),
        odd=sqrt(odd/max(n,1)), even=sqrt(even/max(n,1)), d2=d2m,
        ncl=length(cl), nsl=count(s -> abs(s)==2, st), L2q)
end

function run_case(lab; tipo=2, ndiv_c=16, ndiv_f=8, u_max=0.03, μA=nothing, reset_lock_A=true)
    println("\n", "="^60)
    println(lab)
    prob, _ = load_dad_5d_contact(; μ=μ, ndiv_c=ndiv_c, ndiv_f=ndiv_f, ndiv_s=6,
        ndiv_top=12, tipo=tipo, nome="m"*string(hash(lab)%10000))
    # A
    if μA !== nothing
        for cp in prob.contacts; cp.μ = float(μA); end
    end
    contato_newton!(prob)
    if μA !== nothing
        for cp in prob.contacts; cp.μ = μ; end
    end
    if reset_lock_A
        for cp in prob.contacts; cp.ut_lock = 0.0; end
    end
    sA = qstats(prob)
    @printf("  A  P=%.1f p0=%.1f ncl=%d rms_q=%.2f odd=%.2f even=%.2f d2=%.2f\n",
        sA.P, sA.p0, sA.ncl, sA.rms_q, sA.odd, sA.even, sA.d2)

    _, uyA = dad5d_top_u_mean(prob)
    apply_dad5d_bulk_ux!(prob, 0.0; uy=uyA)
    contato_newton!(prob)

    # B
    for s in 1:6
        apply_dad5d_bulk_ux!(prob, u_max * s/6; uy=uyA)
        contato_newton!(prob)
    end
    sB = qstats(prob)
    @printf("  B  P=%.1f Q=%.1f |Q|/fP=%.3f nsl=%d odd=%.2f even=%.2f d2=%.2f L2q=%.3f\n",
        sB.P, sB.Q, abs(sB.Q)/max(μ*sB.P,eps()), sB.nsl, sB.odd, sB.even, sB.d2, sB.L2q)

    # C
    for s in 1:6
        apply_dad5d_bulk_ux!(prob, u_max * (1 - s/6); uy=uyA)
        contato_newton!(prob)
    end
    sC = qstats(prob)
    @printf("  C  P=%.1f Q=%.1f odd=%.2f even=%.2f d2=%.2f nsl=%d\n",
        sC.P, sC.Q, sC.odd, sC.even, sC.d2, sC.nsl)
    return (; sA, sB, sC)
end

function main()
    run_case("linear tipo=1 nd16"; tipo=1, ndiv_c=16, ndiv_f=8)
    run_case("quad tipo=2 nd16"; tipo=2, ndiv_c=16, ndiv_f=8)
    run_case("quad tipo=2 nd24"; tipo=2, ndiv_c=24, ndiv_f=10)
    run_case("quad tipo=2 nd16 μA=0"; tipo=2, ndiv_c=16, ndiv_f=8, μA=0.0)
    run_case("quad tipo=2 nd16 no lock reset"; tipo=2, ndiv_c=16, ndiv_f=8, reset_lock_A=false)
    run_case("quad tipo=2 nd16 u_max=0.05"; tipo=2, ndiv_c=16, ndiv_f=8, u_max=0.05)
end
main()
