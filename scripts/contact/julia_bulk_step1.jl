# Step-1 Newton-by-Newton dump: current mesh vs MATLAB-element-count mesh.
ENV["GKSwstype"] = "100"
using DrWatson
@quickactivate :BEM
using BEM.MultiRegion
using LinearAlgebra, Printf

include(datadir("elastico", "dad_contato_bulk.jl"))
const OUT = joinpath(projectdir(), "plots", "cattaneo_mindlin", "octave_bulk")
const MR = BEM.MultiRegion
mkpath(OUT)

function pin_face_center_ux!(dad; face::Symbol=:top)
    ys = [pt[2] for pt in dad.Nodes]
    yref = face === :top ? maximum(ys) : minimum(ys)
    thr = 1e-9 * max(abs(yref), maximum(abs, ys), 1.0)
    best, bx = 0, Inf
    for i in 1:dad.n
        abs(dad.Nodes[i][2] - yref) <= thr || continue
        ax = abs(dad.Nodes[i][1]); ax < bx && (bx = ax; best = i)
    end
    dad.BC[2best-1] = 0; dad.BV[2best-1] = 0.0
    return best
end

function report_mesh(label, prob)
    dad1, dad2 = prob.regions
    xs = sort([dad1.Nodes[cp.node_a][1] for cp in prob.contacts])
    hs = [cp.gap0 for cp in prob.contacts]
    @printf("\n======== %s ========\n", label)
    @printf("pad n=%d els=%d  spec n=%d els=%d  pairs=%d\n",
        dad1.n, length(dad1.elements), dad2.n, length(dad2.elements), length(prob.contacts))
    @printf("contact x: n=%d  xmin=%.6f xmax=%.6f  min|x|=%.6e  has_x0=%s\n",
        length(xs), xs[1], xs[end], minimum(abs, xs), any(abs(x) < 1e-12 for x in xs))
    @printf("h [%.3e, %.3e]  n_h0=%d  n_h<1e-7=%d\n",
        extrema(hs)..., count(==(0.0), hs), count(h -> h < 1e-7, hs))
    cps = sort(prob.contacts; by=cp -> abs(dad1.Nodes[cp.node_a][1]))
    println("  closest pairs:")
    for cp in cps[1:min(5, length(cps))]
        n1, n2 = dad1.Normal[cp.node_a], dad2.Normal[cp.node_b]
        R1 = Matrix(BEM.node_rotation2d(n1)); R2 = Matrix(BEM.node_rotation2d(n2))
        R = R2 * R1'
        @printf("    xa=%+.6f xb=%+.6f  h=%.6e  nA=(%+.3f,%+.3f) nB=(%+.3f,%+.3f)  R=[%.1f %.1f; %.1f %.1f]\n",
            dad1.Nodes[cp.node_a][1], dad2.Nodes[cp.node_b][1], cp.gap0,
            n1[1], n1[2], n2[1], n2[2], R[1,1], R[1,2], R[2,1], R[2,2])
    end
    return nothing
end

function dump_closed!(io, prep, pairs, h, x, dad1, tag)
    nx = sum(p.ndof for p in prep)
    for (k, cp) in enumerate(pairs)
        kin = MR._contact_pair_kinematics(prep, cp, h[k], x, k, nx)
        abs(cp.state) == 1 && abs(kin.tn1) < 1e-14 && continue
        pr1, pr2 = prep[cp.reg_a], prep[cp.reg_b]
        R1 = BEM.node_rotation2d(pr1.dad.Normal[cp.node_a])
        R2 = BEM.node_rotation2d(pr2.dad.Normal[cp.node_b])
        ug1 = R1 * [kin.un1, kin.ut1]
        ug2 = R2 * [kin.un2, kin.ut2]
        util = (cp.μ * abs(kin.tn1) > 1e-14) ? abs(kin.tt1) / (cp.μ * abs(kin.tn1)) : 0.0
        @printf("%s  k=%3d x=%+.5f st=%d tn=%8.3f tt=%8.4f util=%.3f  un=(%.3e,%.3e) ut=(%.3e,%.3e) dun=%.3e dut=%.3e ux=(%.3e,%.3e)\n",
            tag, k, dad1.Nodes[cp.node_a][1], cp.state, kin.tn1, kin.tt1, util,
            kin.un1, kin.un2, kin.ut1, kin.ut2, kin.dun, kin.dut, ug1[1], ug2[1])
        println(io, join((tag, k, dad1.Nodes[cp.node_a][1], h[k], cp.state,
            kin.tn1, kin.tt1, util, kin.un1, kin.ut1, kin.un2, kin.ut2,
            kin.dun, kin.dut, kin.gn, kin.gt, ug1[1], ug1[2], ug2[1], ug2[2]), ","))
    end
end

function step1_newtons!(prob; nsteps=50, maxiter=12, label="jul", pin_spec::Bool=true)
    ipad = pin_face_center_ux!(prob.regions[1]; face=:top)
    if pin_spec
        ispec = pin_face_center_ux!(prob.regions[2]; face=:bottom)
        @printf("pins: pad-top %d (x=%.4f)  spec-bottom %d (x=%.4f)\n",
            ipad, prob.regions[1].Nodes[ipad][1],
            ispec, prob.regions[2].Nodes[ispec][1])
    else
        @printf("pins: pad-top %d (x=%.4f)  spec-bottom NONE\n",
            ipad, prob.regions[1].Nodes[ipad][1])
    end
    ctx = MR._contact_friction_setup(prob; method=:ntn, npg=10, common_normal=false)
    prep, pairs = ctx.prep, ctx.pairs
    h = [cp.gap0 for cp in pairs]
    nx = sum(p.ndof for p in prep)
    dad1 = prob.regions[1]
    x0 = zeros(ctx.N)
    MR._verify_contact_states!(pairs, prep, h, x0; epsc=1e-7)
    n_o = count(==(1), (cp.state for cp in pairs))
    n_s = count(==(3), (cp.state for cp in pairs))
    n_l = count(s -> abs(s) == 2, (cp.state for cp in pairs))
    @printf("step0 verify  o/s/l=%d/%d/%d  stick x=%s  h=%s\n", n_o, n_s, n_l,
        [round(dad1.Nodes[cp.node_a][1]; digits=5) for cp in pairs if cp.state == 3],
        [cp.gap0 for cp in pairs if cp.state == 3])

    xtot = zeros(ctx.N)
    x = zeros(ctx.N)
    csv = joinpath(OUT, "$(label)_step1_newton.csv")
    open(csv, "w") do io
        println(io, "tag,k,x,h,state,tn,tt,util,un1,ut1,un2,ut2,dun,dut,gn,gt,ux1,uy1,ux2,uy2")
        dump_closed!(io, prep, pairs, h, x0, dad1, "$(label)_it0")
        prevst = [cp.state for cp in pairs]
        for it in 1:maxiter
            x_tot = xtot .+ x
            MR._verify_contact_states!(pairs, prep, h, x_tot; epsc=1e-7)
            st = [cp.state for cp in pairs]
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
            x_new = A \ b
            dist = norm(x_new - x)
            x .= x_new
            MR._verify_contact_states!(pairs, prep, h, xtot .+ x; epsc=1e-7)
            n_o = count(==(1), (cp.state for cp in pairs))
            n_s = count(==(3), (cp.state for cp in pairs))
            n_l = count(s -> abs(s) == 2, (cp.state for cp in pairs))
            p0 = 0.0; qmax = 0.0
            for cp in pairs
                abs(cp.state) == 1 && continue
                p0 = max(p0, -cp.tn); qmax = max(qmax, abs(cp.tt))
            end
            @printf("%s it=%2d  dist=%.3e  o/s/l=%d/%d/%d  p0=%.4f  qmax=%.4f\n",
                label, it, dist, n_o, n_s, n_l, p0, qmax)
            dump_closed!(io, prep, pairs, h, xtot .+ x, dad1, "$(label)_it$(it)")
            n_flip = count(i -> [cp.state for cp in pairs][i] != prevst[i], eachindex(prevst))
            prevst = [cp.state for cp in pairs]
            (dist < 1e-8 && n_flip == 0) && break
        end
    end
    println("wrote $csv")
    return nothing
end

# --- A: current mesh (ndiv_c=45 transfinite nodes = 44 elements) ---
probA, _ = load_dad_contato_bulk(; ndiv_c=45, ndiv_s=15, tipo=2,
    gap=:euclidean, nome="jstep1A")
report_mesh("A current ndiv=45 nodes", probA)
step1_newtons!(probA; label="julA", pin_spec=false)

probA2, _ = load_dad_contato_bulk(; ndiv_c=45, ndiv_s=15, tipo=2,
    gap=:euclidean, nome="jstep1A2")
report_mesh("A2 current + specimen pin", probA2)
step1_newtons!(probA2; label="julA2", pin_spec=true)

# --- B: MATLAB MALHA 45 elements (Gmsh transfinite 46 nodes) + side 16 ---
probB, _ = load_dad_contato_bulk(; ndiv_c=46, ndiv_s=16, tipo=2,
    gap=:euclidean, nome="jstep1B")
report_mesh("B MATLAB 45 els (transfinite 46) + specimen pin", probB)
step1_newtons!(probB; label="julB", pin_spec=true)

println("Done Julia step-1 Newtons.")
