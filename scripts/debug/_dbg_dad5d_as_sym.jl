# Track when L/R active-set symmetry breaks during fretting ramp
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Statistics, Printf

include(datadir("elastico", "two_blocks_contact.jl"))
include(datadir("elastico", "dad_5d_contact.jl"))

par = loyola_dad5d_params()
μ = par.μ

function contato_newton!(prob; tol=1e-9, maxiter=100, npg=12, epsc=1e-7)
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
    ok = false; nit = 0
    for it in 1:maxiter
        nit = it
        it > 1 && BEM._verify_contact_states!(pairs, prep, h, x0; epsc=epsc)
        A, b = BEM._assemble_contact_system(prep, pairs, h, x0)
        x = A \ b
        dist = norm(x - x0); x0 = x
        dist < tol && (ok = true; break)
    end
    BEM._verify_contact_states!(pairs, prep, h, x0; epsc=epsc)
    BEM._update_contact_ut_locks!(pairs, prep, h, x0)
    BEM._scatter_contact_solution!(prob, prep, pairs, x0)
    set_cache!(prob.regions[1]; contact_x=copy(x0))
    return ok, nit
end

function sym_report(prob)
    fr = contact_interface_xyτ(prob)
    da = prob.regions[1]
    cl = findall(i -> abs(fr.state[i]) != 1, eachindex(fr.state))
    x = fr.x[cl]; q = .-fr.tt[cl]; p = .-fr.tn[cl]; st = fr.state[cl]
    # match L/R
    n_mismatch = 0
    max_dq = 0.0
    for i in eachindex(x)
        x[i] < -1e-8 || continue
        j = argmin(abs.(x .+ x[i]))
        if abs(st[i]) != abs(st[j]) || sign(st[i]) != sign(st[j]) && abs(st[i])==2
            n_mismatch += 1
        end
        max_dq = max(max_dq, abs(q[i] - q[j]))
    end
    w = ones(length(x))
    if length(x) >= 2
        w[1] = abs(x[2]-x[1]); w[end] = abs(x[end]-x[end-1])
        for i in 2:length(x)-1; w[i] = 0.5*abs(x[i+1]-x[i-1]); end
    end
    P = sum(p .* w); Q = sum(q .* w)
    odd = even = 0.0; n = 0
    for i in eachindex(x)
        x[i] < -1e-8 || continue
        j = argmin(abs.(x .+ x[i]))
        even += ((q[i]+q[j])/2)^2; odd += ((q[i]-q[j])/2)^2; n += 1
    end
    return (; P, Q, nsl=count(s->abs(s)==2, st), nst=count(==(3), st),
        n_mismatch, max_dq, odd=sqrt(odd/max(n,1)), even=sqrt(even/max(n,1)))
end

function run(; tipo=2, ndiv_c=24, ndiv_f=10, u_max=0.05, n_leg=20, μA=0.0, epsc=1e-7)
    @printf("\n>>> tipo=%d ndiv_c=%d u_max=%.3f n_leg=%d μA=%.2f epsc=%.1e\n",
        tipo, ndiv_c, u_max, n_leg, μA, epsc)
    prob, _ = load_dad_5d_contact(; μ=μ, ndiv_c=ndiv_c, ndiv_f=ndiv_f, ndiv_s=6,
        ndiv_top=12, tipo=tipo, nome="as$(ndiv_c)")
    for cp in prob.contacts; cp.μ = μA; end
    contato_newton!(prob; epsc=epsc)
    for cp in prob.contacts; cp.μ = μ; cp.ut_lock = 0.0; end
    r = sym_report(prob)
    @printf("A(μ=%.2f→μ) P=%.1f Q=%.2f odd=%.2f mm=%d\n", μA, r.P, r.Q, r.odd, r.n_mismatch)

    _, uyA = dad5d_top_u_mean(prob)
    apply_dad5d_bulk_ux!(prob, 0.0; uy=uyA)
    contato_newton!(prob; epsc=epsc)

    for s in 1:n_leg
        ux = u_max * s / n_leg
        apply_dad5d_bulk_ux!(prob, ux; uy=uyA)
        ok, nit = contato_newton!(prob; epsc=epsc)
        r = sym_report(prob)
        if s == 1 || s == n_leg || r.n_mismatch > 0 || s % 5 == 0
            @printf("  s=%02d ux=%.4f it=%d Q=%+.1f st/sl=%d/%d odd=%.2f even=%.2f mm=%d max|dq|=%.1f ok=%s\n",
                s, ux, nit, r.Q, r.nst, r.nsl, r.odd, r.even, r.n_mismatch, r.max_dq, ok)
        end
    end
    # print edge states
    fr = contact_interface_xyτ(prob)
    println("edge states (|x| near a):")
    for i in eachindex(fr.x)
        abs(fr.x[i]) < 0.7 || abs(fr.x[i]) > 1.4 && continue
        @printf("  x=%+.4f st=%2d p=%.1f q=%.1f\n", fr.x[i], fr.state[i], -fr.tn[i], -fr.tt[i])
    end
end

run(tipo=2, ndiv_c=24, u_max=0.05, n_leg=20, μA=0.0)
run(tipo=2, ndiv_c=24, u_max=0.05, n_leg=40, μA=0.0)
run(tipo=2, ndiv_c=24, u_max=0.05, n_leg=20, μA=0.0, epsc=1e-5)
