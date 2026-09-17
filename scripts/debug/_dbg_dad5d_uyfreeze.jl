using DrWatson
@quickactivate :BEM
using LinearAlgebra, Statistics, Printf

include(datadir("elastico", "two_blocks_contact.jl"))
include(datadir("elastico", "dad_5d_contact.jl"))

par = loyola_dad5d_params()

function solveA!(prob)
    ctx = BEM._contact_friction_setup(prob; method=:ntn, npg=10)
    prep, pairs = ctx.prep, ctx.pairs
    for cp in pairs; cp.state = 3; cp.ut_lock = 0.0; end
    h = [cp.gap0 for cp in pairs]
    x = zeros(ctx.N)
    for it in 1:60
        it > 1 && BEM._verify_contact_states!(pairs, prep, h, x; epsc=1e-7)
        A, b = BEM._assemble_contact_system(prep, pairs, h, x)
        xnew = A \ b
        dist = norm(xnew - x); x = xnew
        dist < 1e-9 && break
    end
    BEM._verify_contact_states!(pairs, prep, h, x; epsc=1e-7)
    BEM._update_contact_ut_locks!(pairs, prep, h, x)
    BEM._scatter_contact_solution!(prob, prep, pairs, x)
    set_cache!(prob.regions[1]; contact_x=copy(x))
    return prep, pairs, x, h
end

function metrics(prob)
    fr = contact_interface_xyτ(prob)
    cl = abs.(fr.state) .!= 1
    p = .-fr.tn
    @printf("  cl=%d st=%d sl=%d  p0=%.1f mean_tn=%.1f mean_tt=%.2f\n",
        count(cl), count(==(3), fr.state), count(s->abs(s)==2, fr.state),
        any(cl) ? maximum(p[cl]) : 0.0,
        any(cl) ? mean(fr.tn[cl]) : 0.0,
        any(cl) ? mean(fr.tt[cl]) : 0.0)
end

function top_u(prob)
    dad = prob.regions[1]
    ymax = maximum(pt[2] for pt in dad.Nodes)
    uxs = Float64[]; uys = Float64[]
    for i in 1:dad.n
        abs(dad.Nodes[i][2]-ymax)<1e-9 || continue
        push!(uxs, dad.u[2i-1]); push!(uys, dad.u[2i])
    end
    return mean(uxs), mean(uys), extrema(uys)
end

function set_top!(prob, ux, uy)
    dad = prob.regions[1]
    ymax = maximum(pt[2] for pt in dad.Nodes)
    for i in 1:dad.n
        abs(dad.Nodes[i][2]-ymax)<1e-9 || continue
        dad.BC[2i-1]=0; dad.BV[2i-1]=float(ux)
        dad.BC[2i]=0; dad.BV[2i]=float(uy)
    end
end

function main()
    prob, _ = load_dad_5d_contact(; μ=par.μ, ndiv_c=12, ndiv_f=6, ndiv_s=4, ndiv_top=8, nome="uyf")
    println("A force")
    solveA!(prob)
    metrics(prob)
    uxA, uyA, uyext = top_u(prob)
    @printf("top ux_mean=%.6f uy_mean=%.6f uy_ext=%s\n", uxA, uyA, uyext)

    # B0: Dirichlet ux=0, uy=uyA, clear warm start, all stick
    println("\nB0: Dirichlet ux=0 uy=uyA, zero start, all-stick")
    set_top!(prob, 0.0, uyA)
    for cp in prob.contacts; cp.state = 3; cp.ut_lock = 0.0; end
    # clear warm
    set_cache!(prob.regions[1]; contact_x=zeros(0))
    ctx = BEM._contact_friction_setup(prob; method=:ntn, npg=10)
    prep, pairs = ctx.prep, ctx.pairs
    h = [cp.gap0 for cp in pairs]
    x = zeros(ctx.N)
    for it in 1:60
        it > 1 && BEM._verify_contact_states!(pairs, prep, h, x; epsc=1e-7)
        A, b = BEM._assemble_contact_system(prep, pairs, h, x)
        xnew = A \ b
        dist = norm(xnew - x); x = xnew
        if it == 1 || it == 2 || dist < 1e-9
            nst = count(cp->cp.state==3, pairs)
            nsl = count(cp->abs(cp.state)==2, pairs)
            # sample tn
            nx = sum(p.ndof for p in prep)
            tns = [BEM._contact_pair_kinematics(prep, cp, h[k], x, k, nx).tn1 for (k,cp) in enumerate(pairs)]
            @printf("  it=%d dist=%.2e st/sl=%d/%d tn[min,max]=[%.1f,%.1f]\n",
                it, dist, nst, nsl, minimum(tns), maximum(tns))
        end
        dist < 1e-9 && break
    end
    BEM._verify_contact_states!(pairs, prep, h, x; epsc=1e-7)
    BEM._scatter_contact_solution!(prob, prep, pairs, x)
    metrics(prob)

    # B1: also try uy more compressive
    for fac in (1.0, 1.05, 1.1, 0.95, 0.9)
        println("\nDirichlet ux=0 uy=$(fac)*uyA")
        set_top!(prob, 0.0, fac * uyA)
        for cp in prob.contacts; cp.state = 3; cp.ut_lock = 0.0; end
        set_cache!(prob.regions[1]; contact_x=zeros(0))
        ctx = BEM._contact_friction_setup(prob; method=:ntn, npg=10)
        prep, pairs = ctx.prep, ctx.pairs
        h = [cp.gap0 for cp in pairs]
        x = zeros(ctx.N)
        for it in 1:60
            it > 1 && BEM._verify_contact_states!(pairs, prep, h, x; epsc=1e-7)
            A, b = BEM._assemble_contact_system(prep, pairs, h, x)
            xnew = A \ b
            dist = norm(xnew - x); x = xnew
            dist < 1e-9 && break
        end
        BEM._verify_contact_states!(pairs, prep, h, x; epsc=1e-7)
        BEM._scatter_contact_solution!(prob, prep, pairs, x)
        metrics(prob)
    end

    # B2: ux ramp with best uy
    println("\nux=0.01 uy=1.05*uyA")
    set_top!(prob, 0.01, 1.05 * uyA)
    for cp in prob.contacts; cp.state = 3; cp.ut_lock = 0.0; end
    ctx = BEM._contact_friction_setup(prob; method=:ntn, npg=10)
    prep, pairs = ctx.prep, ctx.pairs
    h = [cp.gap0 for cp in pairs]
    x = zeros(ctx.N)
    for it in 1:60
        it > 1 && BEM._verify_contact_states!(pairs, prep, h, x; epsc=1e-7)
        A, b = BEM._assemble_contact_system(prep, pairs, h, x)
        xnew = A \ b
        dist = norm(xnew - x); x = xnew
        dist < 1e-9 && break
    end
    BEM._verify_contact_states!(pairs, prep, h, x; epsc=1e-7)
    BEM._scatter_contact_solution!(prob, prep, pairs, x)
    metrics(prob)
    fr = contact_interface_xyτ(prob)
    cl = abs.(fr.state) .!= 1
    if any(cl)
        xs=fr.x; w=ones(length(xs))
        w[1]=abs(xs[2]-xs[1]); w[end]=abs(xs[end]-xs[end-1])
        for i=2:length(xs)-1; w[i]=0.5*abs(xs[i+1]-xs[i-1]); end
        Q = sum((.-fr.tt[cl]).*w[cl]); P = sum((.-fr.tn[cl]).*w[cl])
        @printf("  P=%.1f Q=%.1f |Q|/fP=%.3f xc=%.4f\n", P, Q, abs(Q)/(par.μ*P),
            sum(xs[cl].*(-fr.tn[cl]))/sum(-fr.tn[cl]))
    end
end
main()
