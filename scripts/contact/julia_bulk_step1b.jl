# Corrected mesh (45 els + NOS_RES pins): step-1 Newtons at npg=10 and 30.
ENV["GKSwstype"] = "100"
using DrWatson
@quickactivate :BEM
using BEM.MultiRegion
using LinearAlgebra, Printf

include(datadir("elastico", "dad_contato_bulk.jl"))
const OUT = joinpath(projectdir(), "plots", "cattaneo_mindlin", "octave_bulk")
const MR = BEM.MultiRegion
mkpath(OUT)

function gself(pr, node)
    cols = pr.Gc_cols[node]
    iu = 2node - 1
    G = pr.G_local
    return (G[iu, cols[1]], G[iu, cols[2]], G[iu+1, cols[1]], G[iu+1, cols[2]])
end

function run_npg(npg)
    prob, _ = load_dad_contato_bulk(; ndiv_c=45, ndiv_s=15, tipo=2,
        gap=:euclidean, nome="jstep1_npg$(npg)")
    dad1, dad2 = prob.regions
    xs = [dad1.Nodes[cp.node_a][1] for cp in prob.contacts]
    @printf("\n===== npg=%d  pad n=%d els=%d  spec n=%d els=%d  pairs=%d  min|x|=%.3e  n_h0=%d =====\n",
        npg, dad1.n, length(dad1.elements), dad2.n, length(dad2.elements),
        length(prob.contacts), minimum(abs, xs), count(cp -> cp.gap0 == 0, prob.contacts))
    ctx = MR._contact_friction_setup(prob; method=:ntn, npg=npg, common_normal=false)
    prep, pairs = ctx.prep, ctx.pairs
    h = [cp.gap0 for cp in pairs]
    nx = sum(p.ndof for p in prep)
    nsteps = 50
    k0 = argmin(abs(dad1.Nodes[cp.node_a][1]) for cp in pairs)
    cp0 = pairs[k0]
    Gn = gself(prep[1], cp0.node_a)
    Gs = gself(prep[2], cp0.node_b)
    @printf("centre x=%.4e  Gpad(row xy, col nt)=[%.4e %.4e; %.4e %.4e]\n",
        dad1.Nodes[cp0.node_a][1], Gn...)
    @printf("           Gspec=[%.4e %.4e; %.4e %.4e]  ||A_pad||=%.4e ||b_pad||=%.4e\n",
        Gs..., norm(prep[1].A), norm(prep[1].b))

    x0 = zeros(ctx.N)
    MR._verify_contact_states!(pairs, prep, h, x0; epsc=1e-7)
    @printf("step0  o/s/l=%d/%d/%d  stick x=%s\n",
        count(==(1), (cp.state for cp in pairs)),
        count(==(3), (cp.state for cp in pairs)),
        count(s -> abs(s)==2, (cp.state for cp in pairs)),
        [round(dad1.Nodes[cp.node_a][1]; digits=5) for cp in pairs if abs(cp.state)!=1])

    xtot = zeros(ctx.N); x = zeros(ctx.N)
    for it in 1:8
        x_tot = xtot .+ x
        MR._verify_contact_states!(pairs, prep, h, x_tot; epsc=1e-7)
        st_pre = [cp.state for cp in pairs]
        A, b = MR._assemble_contact_system(prep, pairs, h, x_tot)
        b[1:nx] ./= nsteps
        for (k, cp) in enumerate(pairs)
            abs(cp.state) == 1 && continue
            ot = nx + 4(k - 1)
            b[ot + 1] = h[k] - MR._contato_deltaun(prep, cp, xtot, nx)
            if abs(cp.state) == 2
                sμ = cp.μ * (cp.state >= 0 ? 1.0 : -1.0)
                b[ot + 2] = -sμ * xtot[ot + 1] - xtot[ot + 2]
            end
        end
        # dump stick-pair constraint rows + G col norms on first 3-stick assemble
        nstick = count(==(3), st_pre)
        if nstick >= 3 && it <= 3
            println("  3-stick (or more) assemble it=$it  closed pairs:")
            for (k, cp) in enumerate(pairs)
                abs(cp.state) == 1 && continue
                ot = nx + 4(k - 1)
                rows = ot+1:ot+4
                iu1 = prep[cp.reg_a].off + (2cp.node_a - 1)
                iu2 = prep[cp.reg_b].off + (2cp.node_b - 1)
                @printf("    x=%+.5f st=%d  gap-row u: un1=%.1f ut1=%.1f un2=%.1f ut2=%.1f  b=%.3e\n",
                    dad1.Nodes[cp.node_a][1], cp.state,
                    A[rows[1], iu1], A[rows[1], iu1+1], A[rows[1], iu2], A[rows[1], iu2+1],
                    b[rows[1]])
                @printf("             stick-row u: un1=%.1f ut1=%.1f un2=%.1f ut2=%.1f  b=%.3e\n",
                    A[rows[2], iu1], A[rows[2], iu1+1], A[rows[2], iu2], A[rows[2], iu2+1],
                    b[rows[2]])
                @printf("             eq-n t: tn1=%.1f tt1=%.1f tn2=%.1f tt2=%.1f\n",
                    A[rows[3], ot+1], A[rows[3], ot+2], A[rows[3], ot+3], A[rows[3], ot+4])
                @printf("             eq-t t: tn1=%.1f tt1=%.1f tn2=%.1f tt2=%.1f\n",
                    A[rows[4], ot+1], A[rows[4], ot+2], A[rows[4], ot+3], A[rows[4], ot+4])
                pr1, pr2 = prep[cp.reg_a], prep[cp.reg_b]
                c1, c2 = pr1.Gc_cols[cp.node_a], pr2.Gc_cols[cp.node_b]
                @printf("             ||-G pad tn,tt||=%.3e,%.3e  spec=%.3e,%.3e\n",
                    norm(A[pr1.off+1:pr1.off+pr1.ndof, ot+1]),
                    norm(A[pr1.off+1:pr1.off+pr1.ndof, ot+2]),
                    norm(A[pr2.off+1:pr2.off+pr2.ndof, ot+3]),
                    norm(A[pr2.off+1:pr2.off+pr2.ndof, ot+4]))
            end
        end
        x_new = A \ b
        dist = norm(x_new - x)
        x .= x_new
        MR._verify_contact_states!(pairs, prep, h, xtot .+ x; epsc=1e-7)
        p0 = 0.0; qmax = 0.0
        for cp in pairs
            abs(cp.state) == 1 && continue
            p0 = max(p0, -cp.tn); qmax = max(qmax, abs(cp.tt))
        end
        n_o = count(==(1), (cp.state for cp in pairs))
        n_s = count(==(3), (cp.state for cp in pairs))
        n_l = count(s -> abs(s)==2, (cp.state for cp in pairs))
        @printf("it=%d  dist=%.3e  pre o/s/l=%d/%d/%d  post o/s/l=%d/%d/%d  p0=%.4f qmax=%.4f\n",
            it, dist,
            count(==(1), st_pre), count(==(3), st_pre), count(s -> abs(s)==2, st_pre),
            n_o, n_s, n_l, p0, qmax)
        for (k, cp) in enumerate(pairs)
            abs(cp.state) == 1 && abs(cp.tn) < 1e-12 && continue
            kin = MR._contact_pair_kinematics(prep, cp, h[k], xtot .+ x, k, nx)
            util = (cp.μ*abs(kin.tn1) > 1e-14) ? abs(kin.tt1)/(cp.μ*abs(kin.tn1)) : 0.0
            @printf("  x=%+.5f st=%d tn=%8.3f tt=%8.4f util=%.3f dun=%.3e dut=%.3e h=%.3e\n",
                dad1.Nodes[cp.node_a][1], cp.state, kin.tn1, kin.tt1, util,
                kin.dun, kin.dut, h[k])
        end
        dist < 1e-8 && it > 1 && break
    end
end

run_npg(10)
run_npg(30)
println("Done.")
