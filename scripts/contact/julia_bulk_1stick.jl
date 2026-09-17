# 1-stick Newton: default far-lumping vs full quadrature (MATLAB-like).
ENV["GKSwstype"] = "100"
using DrWatson
@quickactivate :BEM
using BEM.MultiRegion
using LinearAlgebra, Printf, FastGaussQuadrature

include(datadir("elastico", "dad_contato_bulk.jl"))
const MR = BEM.MultiRegion

function traction_resultant(dad)
    Fx = Fy = 0.0
    has_cache(dad, :traction) || return (Fx, Fy)
    t = dad.traction
    η, w = gausslegendre(8)
    for el in dad.elements
        X = dad.Nodes[el.index]
        N, dN = BEM.shapefun(dad.element_type, η)
        pg_dx = dN * X
        tx = [t[2i - 1] for i in el.index]
        ty = [t[2i] for i in el.index]
        Tgx = N * tx
        Tgy = N * ty
        for k in eachindex(η)
            J = hypot(pg_dx[k][1], pg_dx[k][2])
            Fx += Tgx[k] * J * w[k]
            Fy += Tgy[k] * J * w[k]
        end
    end
    return (Fx, Fy)
end

function prescribed_top_load(dad)
    ymax = maximum(pt[2] for pt in dad.Nodes)
    Fy = 0.0; ntop = 0
    for i in 1:dad.n
        abs(dad.Nodes[i][2] - ymax) <= 1e-9 * max(ymax, 1.0) || continue
        ntop += 1
        Fy += dad.BV[2i]  # global ty before local (still as format2d stored)
    end
    return ntop, Fy
end

function run_mode(label; npg=10, near_factor=2.0, nsteps=50)
    prob, par = load_dad_contato_bulk(; ndiv_c=45, ndiv_s=15, tipo=2,
        gap=:euclidean, nome="j1s_$(label)")
    dad1 = prob.regions[1]
    ntop, Fyt = prescribed_top_load(dad1)
    @printf("\n===== %s  npg=%s  near_factor=%s  pairs=%d  top nodes=%d  sum BV_ty=%.4f  ΔP=%.4f =====\n",
        label, npg, near_factor, length(prob.contacts), ntop, Fyt, par.P / nsteps)
    ctx = MR._contact_friction_setup(prob; method=:ntn, npg=npg,
        common_normal=false, near_factor=near_factor)
    prep, pairs = ctx.prep, ctx.pairs
    h = [cp.gap0 for cp in pairs]
    nx = sum(p.ndof for p in prep)
    x = zeros(ctx.N)
    MR._verify_contact_states!(pairs, prep, h, x; epsc=1e-7)
    @printf("step0 o/s/l=%d/%d/%d  ||b_pad||=%.4e  ||A_pad||=%.4e\n",
        count(==(1), (cp.state for cp in pairs)),
        count(==(3), (cp.state for cp in pairs)),
        count(s -> abs(s)==2, (cp.state for cp in pairs)),
        norm(prep[1].b), norm(prep[1].A))
    for it in 1:4
        st_pre = [cp.state for cp in pairs]
        A, b = MR._assemble_contact_system(prep, pairs, h, x)
        b[1:nx] ./= nsteps
        for (k, cp) in enumerate(pairs)
            abs(cp.state) == 1 && continue
            ot = nx + 4(k - 1)
            b[ot + 1] = h[k]
        end
        x = A \ b
        MR._verify_contact_states!(pairs, prep, h, x; epsc=1e-7)
        p0 = 0.0; qmax = 0.0
        for cp in pairs
            abs(cp.state) == 1 && continue
            p0 = max(p0, -cp.tn); qmax = max(qmax, abs(cp.tt))
        end
        @printf("it=%d  pre o/s/l=%d/%d/%d  post=%d/%d/%d  p0=%.4f qmax=%.4f\n",
            it,
            count(==(1), st_pre), count(==(3), st_pre), count(s -> abs(s)==2, st_pre),
            count(==(1), (cp.state for cp in pairs)),
            count(==(3), (cp.state for cp in pairs)),
            count(s -> abs(s)==2, (cp.state for cp in pairs)),
            p0, qmax)
        ord = sortperm(1:length(pairs); by=k -> abs(dad1.Nodes[pairs[k].node_a][1]))
        for k in ord[1:min(5, length(ord))]
            cp = pairs[k]
            kin = MR._contact_pair_kinematics(prep, cp, h[k], x, k, nx)
            util = (cp.μ * abs(kin.tn1) > 1e-14) ? abs(kin.tt1) / (cp.μ * abs(kin.tn1)) : 0.0
            @printf("  x=%+.5f st=%d tn=%8.3f tt=%8.4f util=%.3f dun=%.3e dut=%.3e h=%.3e un1=%.3e\n",
                dad1.Nodes[cp.node_a][1], cp.state, kin.tn1, kin.tt1, util,
                kin.dun, kin.dut, h[k], kin.un1)
        end
        if it == 1
            MR._scatter_contact_solution!(prob, prep, pairs, x)
            Fx, Fy = traction_resultant(dad1)
            @printf("  pad ∫t dΓ = (Fx,Fy)=(%.4f, %.4f)  expect Fy≈%.4f\n",
                Fx, Fy, -par.P / nsteps)
        end
        it > 1 && count(s -> abs(s)==2, (cp.state for cp in pairs)) > 0 && break
        it > 1 && p0 > 0 && count(==(3), (cp.state for cp in pairs)) == count(==(3), st_pre) &&
            it >= 2 && break
    end
end

run_mode("full10"; npg=10, near_factor=Inf)
println("Done.")
