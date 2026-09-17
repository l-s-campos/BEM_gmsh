# Gauss–Legendre collocation: full integration vs nodal lumping (near_factor=2).
ENV["GKSwstype"] = "100"
using DrWatson
@quickactivate :BEM
using BEM.MultiRegion
using LinearAlgebra, Printf

include(datadir("elastico", "dad_contato_bulk.jl"))
const MR = BEM.MultiRegion

function run50(label; collocation=:legendre, near_factor=2.0, npg=10, nsteps=50)
    prob, par = load_dad_contato_bulk(; ndiv_c=45, ndiv_s=15, tipo=2,
        collocation=collocation, gap=:euclidean, nome="jgl_$(label)")
    dad1 = prob.regions[1]
    ξ = dad1.element_type.nodes
    xs = [dad1.Nodes[cp.node_a][1] for cp in prob.contacts]
    @printf("\n===== %s  col=%s  ξ=%s  pairs=%d  min|x|=%.3e  near_factor=%s =====\n",
        label, collocation, ξ, length(prob.contacts), minimum(abs, xs), near_factor)
    ctx = MR._contact_friction_setup(prob; method=:ntn, npg=npg,
        common_normal=false, near_factor=near_factor)
    prep, pairs = ctx.prep, ctx.pairs
    h = [cp.gap0 for cp in pairs]
    nx = sum(p.ndof for p in prep)
    xtot = zeros(ctx.N)
    x = zeros(ctx.N)
    MR._verify_contact_states!(pairs, prep, h, xtot; epsc=1e-7)
    @printf("step0  o/s/l=%d/%d/%d  n_h0=%d\n",
        count(==(1), (cp.state for cp in pairs)),
        count(==(3), (cp.state for cp in pairs)),
        count(s -> abs(s)==2, (cp.state for cp in pairs)),
        count(==(0.0), h))
    p0 = 0.0; qmax = 0.0; n_o = 0; n_s = 0; n_l = 0; nit = 0
    for s in 1:nsteps
        prev = fill(0, length(pairs))
        nit = 0
        for it in 1:80
            nit = it
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
            n_flip = count(i -> st[i] != prev[i], eachindex(st))
            x .= x_new
            prev .= st
            (dist < 1e-8 && n_flip == 0) && break
        end
        xtot .+= x
        MR._verify_contact_states!(pairs, prep, h, xtot; epsc=1e-7)
        p0 = 0.0; qmax = 0.0
        for cp in pairs
            abs(cp.state) == 1 && continue
            p0 = max(p0, -cp.tn)
            qmax = max(qmax, abs(cp.tt))
        end
        n_o = count(==(1), (cp.state for cp in pairs))
        n_s = count(==(3), (cp.state for cp in pairs))
        n_l = count(s -> abs(s)==2, (cp.state for cp in pairs))
        if s == 1 || s == nsteps || s in (2, 10, 25)
            @printf("  s=%2d nit=%2d  o/s/l=%d/%d/%d  p0=%.3f  qmax=%.4f  Hertz=%.2f%%\n",
                s, nit, n_o, n_s, n_l, p0, qmax, abs(p0 - par.p0_H) / par.p0_H * 100)
        end
    end
    util = [abs(cp.tt) / (par.μ * abs(cp.tn)) for cp in pairs
            if abs(cp.state) != 1 && par.μ * abs(cp.tn) > 1e-12]
    @printf("DONE %s  p0=%.3f (Hertz %.2f%%)  qmax=%.4f  util=%.3f  closed=%d\n",
        label, p0, abs(p0 - par.p0_H) / par.p0_H * 100, qmax,
        isempty(util) ? 0.0 : sum(util) / length(util), n_s + n_l)
    return (; p0, qmax, n_o, n_s, n_l)
end

println("Hertz two-body p0=", round(dad_contato_bulk_params().p0_H; digits=2))
run50("gl_full"; collocation=:legendre, near_factor=Inf)
run50("gl_lump"; collocation=:legendre, near_factor=2.0)
println("Done.")
