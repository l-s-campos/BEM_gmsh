using DrWatson
@quickactivate :BEM
using LinearAlgebra, Statistics, Printf

include(datadir("elastico", "two_blocks_contact.jl"))
include(datadir("elastico", "dad_5d_contact.jl"))

function main()
    par = loyola_dad5d_params()
    prob, _ = load_dad_5d_contact(; μ=par.μ, ndiv_c=12, ndiv_f=6, ndiv_s=4, ndiv_top=8, nome="bcchk")

    # Step A
    ctx = BEM._contact_friction_setup(prob; method=:ntn, npg=10)
    prep, pairs = ctx.prep, ctx.pairs
    for cp in pairs; cp.state = 3; cp.ut_lock = 0.0; end
    h = [cp.gap0 for cp in pairs]
    x = zeros(ctx.N)
    for it in 1:40
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
    fr = contact_interface_xyτ(prob)
    cl = abs.(fr.state) .!= 1
    println("A: n_cl=$(count(cl)) mean tn=$(mean(fr.tn[cl])) mean tt=$(mean(fr.tt[cl]))")

    ux = 0.02
    apply_dad5d_bulk_ux!(prob, ux; cargav=par.cargav)
    dad = prob.regions[1]
    ymax = maximum(pt[2] for pt in dad.Nodes)
    println("\nGlobal BC on top after apply_dad5d_bulk_ux!(ux=$ux):")
    for i in 1:dad.n
        abs(dad.Nodes[i][2] - ymax) < 1e-9 || continue
        @printf("  node %3d x=%7.3f BC=(%d,%d) BV=(%g,%g) n=(%.3f,%.3f)\n",
            i, dad.Nodes[i][1], dad.BC[2i-1], dad.BC[2i], dad.BV[2i-1], dad.BV[2i],
            dad.Normal[i][1], dad.Normal[i][2])
    end

    ctx = BEM._contact_friction_setup(prob; method=:ntn, npg=10)
    prep, pairs = ctx.prep, ctx.pairs
    pr = prep[1]
    println("\nLocal exterior BC on top nodes:")
    for i in 1:dad.n
        abs(dad.Nodes[i][2] - ymax) < 1e-9 || continue
        n̂, t̂ = local_basis2d(dad.Normal[i])
        @printf("  node %3d n=(%.3f,%.3f) t=(%.3f,%.3f) BCl=(%d,%d) BVl=(%g,%g)\n",
            i, n̂[1], n̂[2], t̂[1], t̂[2],
            pr.BC_ext[2i-1], pr.BC_ext[2i], pr.BV_ext[2i-1], pr.BV_ext[2i])
    end

    x0 = has_cache(prob.regions[1], :contact_x) ? collect(Float64, prob.regions[1].contact_x) : zeros(ctx.N)
    for (label, xstart) in (("warm", copy(x0)), ("zero", zeros(ctx.N)))
        x = copy(xstart)
        for it in 1:40
            it > 1 && BEM._verify_contact_states!(pairs, prep, h, x; epsc=1e-7)
            A, b = BEM._assemble_contact_system(prep, pairs, h, x)
            xnew = A \ b
            dist = norm(xnew - x); x = xnew
            dist < 1e-9 && (@printf("%s converged it=%d\n", label, it); break)
        end
        BEM._verify_contact_states!(pairs, prep, h, x; epsc=1e-7)
        nx = sum(p.ndof for p in prep)
        Q = 0.0; P = 0.0; nst=0; nsl=0
        tts = Float64[]
        for (k, cp) in enumerate(pairs)
            kin = BEM._contact_pair_kinematics(prep, cp, h[k], x, k, nx)
            push!(tts, kin.tt1)
            abs(cp.state)==1 && continue
            cp.state==3 && (nst+=1)
            abs(cp.state)==2 && (nsl+=1)
            P += -kin.tn1; Q += -kin.tt1
        end
        @printf("%s: st/sl=%d/%d sum(-tn)=%.1f sum(-tt)=%.1f  tt range [%.1f,%.1f]\n",
            label, nst, nsl, P, Q, minimum(tts), maximum(tts))
        println("  top local unknowns in x:")
        for i in 1:dad.n
            abs(dad.Nodes[i][2] - ymax) < 1e-9 || continue
            iu = pr.off + 2i - 1
            @printf("    node %d x_loc=(%g,%g)  BC=(%d,%d)\n",
                i, x[iu], x[iu+1], pr.BC_ext[2i-1], pr.BC_ext[2i])
        end
    end
end
main()
