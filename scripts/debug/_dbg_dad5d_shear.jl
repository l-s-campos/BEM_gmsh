# Diagnostic: dad_5d AS fretting — dump states / tn / tt at A and B
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
    niter = 0
    for it in 1:maxiter
        niter = it
        it > 1 && BEM._verify_contact_states!(pairs, prep, h, x0; epsc=1e-7)
        A, b = BEM._assemble_contact_system(prep, pairs, h, x0)
        x = A \ b
        dist = norm(x - x0)
        x0 = x
        if dist < tol
            ok = true
            break
        end
    end
    # capture pre-final-verify states
    st_before = [cp.state for cp in pairs]
    BEM._verify_contact_states!(pairs, prep, h, x0; epsc=1e-7)
    st_after = [cp.state for cp in pairs]
    nchg = count(st_before .!= st_after)
    BEM._update_contact_ut_locks!(pairs, prep, h, x0)
    BEM._scatter_contact_solution!(prob, prep, pairs, x0)
    set_cache!(prob.regions[1]; contact_x=copy(x0))
    return (; ok, prep, pairs, x=x0, h, niter, nchg)
end

function dump_pairs(label, sol; aH=par.a_H)
    prep, pairs, x, h = sol.prep, sol.pairs, sol.x, sol.h
    da = prep[1].dad
    db = prep[2].dad
    nx = sum(p.ndof for p in prep)
    println("\n==== $label  ok=$(sol.ok) iters=$(sol.niter) final_verify_chgs=$(sol.nchg) ====")
    @printf("%3s %8s %6s %10s %10s %10s %10s %10s %8s\n",
        "k", "x", "st", "tn1", "tt1", "gn", "gt", "ut_lock", "|tt|/μ|tn|")
    rows = Any[]
    for (k, cp) in enumerate(pairs)
        kin = BEM._contact_pair_kinematics(prep, cp, h[k], x, k, nx)
        x1 = da.Nodes[cp.node_a][1]
        util = abs(kin.tt1) / (μ * abs(kin.tn1) + 1e-30)
        push!(rows, (x1, k, cp.state, kin, util, cp.ut_lock))
    end
    sort!(rows; by=r -> r[1])
    n_st = count(r -> r[3] == 3, rows)
    n_sl = count(r -> abs(r[3]) == 2, rows)
    n_op = count(r -> abs(r[3]) == 1, rows)
    # integrate with spacing
    xs = [r[1] for r in rows]
    w = ones(length(xs))
    if length(xs) >= 2
        w[1] = abs(xs[2]-xs[1]); w[end] = abs(xs[end]-xs[end-1])
        for i in 2:length(xs)-1; w[i] = 0.5*abs(xs[i+1]-xs[i-1]); end
    end
    P = Q = 0.0
    for (i, r) in enumerate(rows)
        st, kin = r[3], r[4]
        if abs(st) != 1
            P += -kin.tn1 * w[i]
            Q += -kin.tt1 * w[i]
        end
        if abs(r[1]) < 2.5aH || abs(st) != 1
            @printf("%3d %8.3f %6d %10.3f %10.3f %10.5f %10.5f %10.5f %8.3f\n",
                r[2], r[1], st, kin.tn1, kin.tt1, kin.gn, kin.gt, r[6], r[5])
        end
    end
    @printf("counts st/sl/op=%d/%d/%d  P=%.2f Q=%.2f\n", n_st, n_sl, n_op, P, Q)
    # How many closed in |x|<aH?
    n_cl_in = count(r -> abs(r[1]) <= aH && abs(r[3]) != 1, rows)
    n_op_in = count(r -> abs(r[1]) <= aH && abs(r[3]) == 1, rows)
    @printf("inside |x|<aH: closed=%d open=%d\n", n_cl_in, n_op_in)
end

function main()
    prob, _ = load_dad_5d_contact(; μ=μ, ndiv_c=16, ndiv_f=8, ndiv_s=6, ndiv_top=12, nome="dbg2")
    println("n contacts = ", length(prob.contacts))

    println("\n[A] vertical only")
    solA = contato_newton!(prob)
    dump_pairs("A", solA)

    println("\n[B] ramp tx")
    solB = solA
    for s in 1:6
        tx = s / 6 * par.cargah
        apply_dad5d_bulk_tx!(prob, tx; cargav=par.cargav)
        solB = contato_newton!(prob)
        nx = sum(p.ndof for p in solB.prep)
        n_st = n_sl = n_op = 0
        P = Q = 0.0
        da = solB.prep[1].dad
        xs = Float64[]; tns = Float64[]; tts = Float64[]; sts = Int[]
        for (k, cp) in enumerate(solB.pairs)
            kin = BEM._contact_pair_kinematics(solB.prep, cp, solB.h[k], solB.x, k, nx)
            push!(xs, da.Nodes[cp.node_a][1]); push!(tns, kin.tn1); push!(tts, kin.tt1); push!(sts, cp.state)
        end
        perm = sortperm(xs)
        xs, tns, tts, sts = xs[perm], tns[perm], tts[perm], sts[perm]
        w = ones(length(xs))
        w[1] = abs(xs[2]-xs[1]); w[end] = abs(xs[end]-xs[end-1])
        for i in 2:length(xs)-1; w[i] = 0.5*abs(xs[i+1]-xs[i-1]); end
        for i in eachindex(xs)
            abs(sts[i]) == 1 && (n_op += 1; continue)
            abs(sts[i]) == 2 && (n_sl += 1)
            sts[i] == 3 && (n_st += 1)
            P += -tns[i]*w[i]; Q += -tts[i]*w[i]
        end
        @printf("  s=%d tx=%+.1f it=%d chg=%d st/sl/op=%d/%d/%d P=%.1f Q=%+.1f ok=%s\n",
            s, tx, solB.niter, solB.nchg, n_st, n_sl, n_op, P, Q, solB.ok)
    end
    dump_pairs("B", solB)

    # Check R at center
    _, k0 = findmin(k -> abs(solB.prep[1].dad.Nodes[solB.pairs[k].node_a][1]),
        eachindex(solB.pairs))
    cp = solB.pairs[k0]
    na, nb = cp.node_a, cp.node_b
    n1 = solB.prep[1].dad.Normal[na]
    n2 = solB.prep[2].dad.Normal[nb]
    R1 = Matrix(node_rotation2d(n1)); R2 = Matrix(node_rotation2d(n2))
    println("\ncenter pair: n1=$n1 n2=$n2")
    println("R1=$R1")
    println("R2=$R2")
    println("R=R2*R1' = $(R2*R1')")

    # Compare cp.tn (from verify/scatter path) vs kinematics from x
    fr = contact_interface_xyτ(prob)
    println("\nfr vs solB.x (first closed-ish nodes):")
    for i in 1:length(fr.x)
        abs(fr.x[i]) > 1.5 && continue
        @printf("  x=%7.3f st=%2d fr.tn=%9.3f fr.tt=%9.3f\n",
            fr.x[i], fr.state[i], fr.tn[i], fr.tt[i])
    end
end
main()
