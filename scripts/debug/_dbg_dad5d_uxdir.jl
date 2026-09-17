# dad_5d fretting: freeze top uy from A, ramp ux (full Dirichlet far field)
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Statistics, Printf

include(datadir("elastico", "two_blocks_contact.jl"))
include(datadir("elastico", "dad_5d_contact.jl"))

par = loyola_dad5d_params()
μ = par.μ

function contato_newton!(prob; tol=1e-9, maxiter=80, npg=10)
    ctx = BEM._contact_friction_setup(prob; method=:ntn, npg=npg)
    prep, pairs = ctx.prep, ctx.pairs
    h = [cp.gap0 for cp in pairs]
    x0 = has_cache(prob.regions[1], :contact_x) &&
         length(prob.regions[1].contact_x) == ctx.N ?
         collect(Float64, prob.regions[1].contact_x) : zeros(ctx.N)
    if !has_cache(prob.regions[1], :contact_x)
        for cp in pairs; cp.state = 3; cp.ut_lock = 0.0; end
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
    return ok, prep, pairs, x0
end

function top_uy_mean(prob)
    dad = prob.regions[1]
    ymax = maximum(pt[2] for pt in dad.Nodes)
    # after scatter, displacement is in dad.u (global)
    us = Float64[]
    for i in 1:dad.n
        abs(dad.Nodes[i][2] - ymax) < 1e-9 || continue
        if has_cache(dad, :u)
            push!(us, dad.u[2i])  # uy
        end
    end
    isempty(us) && return 0.0
    return mean(us)
end

function metrics(prob)
    fr = contact_interface_xyτ(prob)
    xs = fr.x; n = length(xs)
    w = ones(n)
    if n >= 2
        w[1] = abs(xs[2]-xs[1]); w[end] = abs(xs[end]-xs[end-1])
        for i in 2:n-1; w[i] = 0.5*abs(xs[i+1]-xs[i-1]); end
    end
    p = .-fr.tn; q = .-fr.tt; cl = abs.(fr.state) .!= 1
    P = any(cl) ? sum(p[cl].*w[cl]) : 0.0
    Q = any(cl) ? sum(q[cl].*w[cl]) : 0.0
    p0 = any(cl) ? maximum(p[cl]) : 0.0
    xc = any(cl) ? sum(xs[cl].*p[cl])/max(sum(p[cl]),eps()) : 0.0
    a = p0 > 0 ? 2P/(π*p0) : 0.0
    n_st = count(==(3), fr.state); n_sl = count(s->abs(s)==2, fr.state)
    L2q = 0.0
    if any(cl) && abs(Q) > 1
        qA = cattaneo_shear(xs .- xc, a, p0, Q, μ, abs(P))
        L2q = sqrt(sum(abs2, q[cl].-qA[cl])) / max(sqrt(sum(abs2, qA[cl])), eps())
    end
    return (; P, Q, p0, a, xc, n_st, n_sl, L2q,
        n_in=count(i->abs(xs[i])<=par.a_H && abs(fr.state[i])!=1, eachindex(xs)))
end

function apply_top_dirichlet!(prob, ux, uy)
    dad = prob.regions[1]
    ymax = maximum(pt[2] for pt in dad.Nodes)
    nset = 0
    for i in 1:dad.n
        if abs(dad.Nodes[i][2] - ymax) <= 1e-9 * max(abs(ymax), 1.0)
            dad.BC[2i-1] = 0; dad.BV[2i-1] = float(ux)
            dad.BC[2i]   = 0; dad.BV[2i]   = float(uy)
            nset += 1
        end
    end
    return nset
end

function main()
    prob, _ = load_dad_5d_contact(; μ=μ, ndiv_c=16, ndiv_f=8, ndiv_s=6, ndiv_top=12, nome="uxd")
    println("[A] force-controlled")
    contato_newton!(prob)
    mA = metrics(prob)
    uyA = top_uy_mean(prob)
    @printf("A  P=%.1f Q=%.2f p0=%.1f xc=%.4f st/sl=%d/%d uy_top=%.6f\n",
        mA.P, mA.Q, mA.p0, mA.xc, mA.n_st, mA.n_sl, uyA)

    println("\nux sweep with uy frozen at uy_A:")
    for ux in (1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2)
        prob2, _ = load_dad_5d_contact(; μ=μ, ndiv_c=16, ndiv_f=8, ndiv_s=6, ndiv_top=12, nome="uxd")
        contato_newton!(prob2)
        uyA2 = top_uy_mean(prob2)
        nleg = 4
        local m
        for s in 1:nleg
            apply_top_dirichlet!(prob2, ux * s/nleg, uyA2)
            # BC type changed → drop warm x (layout/meaning change)
            if has_cache(prob2.regions[1], :contact_x)
                # keep contact_x but zero exterior-ish: safest clear
                # actually only first switch matters; keep warm for contact
            end
            contato_newton!(prob2)
            m = metrics(prob2)
        end
        @printf("  ux=%8.1e  P=%.1f Q=%+.1f |Q|/fP=%.3f xc=%.4f st/sl=%d/%d inH=%d L2q=%.3f\n",
            ux, m.P, m.Q, abs(m.Q)/max(μ*m.P,eps()), m.xc, m.n_st, m.n_sl, m.n_in, m.L2q)
    end
end
main()
