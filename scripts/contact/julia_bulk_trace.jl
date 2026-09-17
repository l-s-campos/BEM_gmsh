# Step-by-step Julia twin of dad_Contato_Bulk (same dumps as octave_bulk_trace.m)
ENV["GKSwstype"] = "100"
using DrWatson
@quickactivate :BEM
using BEM.MultiRegion
using LinearAlgebra, Statistics, Printf

include(datadir("elastico", "dad_contato_bulk.jl"))
const OUT = joinpath(projectdir(), "plots", "cattaneo_mindlin", "octave_bulk")
const MR = BEM.MultiRegion
mkpath(OUT)

function pin_top_center_ux!(prob)
    dad = prob.regions[1]
    ymax = maximum(pt[2] for pt in dad.Nodes)
    best, bx = 0, Inf
    for i in 1:dad.n
        abs(dad.Nodes[i][2] - ymax) <= 1e-9 * max(abs(ymax), 1.0) || continue
        ax = abs(dad.Nodes[i][1]); ax < bx && (bx = ax; best = i)
    end
    dad.BC[2best-1] = 0; dad.BV[2best-1] = 0.0
    return best
end

prob, par = load_dad_contato_bulk(; ndiv_c=45, ndiv_s=15, tipo=2,
    collocation=:legendre, gap=:euclidean, nome="jtrace")
pin_top_center_ux!(prob)
ctx = MR._contact_friction_setup(prob; method=:ntn, npg=10, common_normal=false)
prep, pairs = ctx.prep, ctx.pairs
h = [cp.gap0 for cp in pairs]
nx = sum(p.ndof for p in prep)
N = ctx.N
dad1 = prob.regions[1]

open(joinpath(OUT, "jul_gap.csv"), "w") do io
    println(io, "x,y,nx,ny,h")
    for cp in pairs
        n = dad1.Nodes[cp.node_a]; nv = dad1.Normal[cp.node_a]
        println(io, join((n[1], n[2], nv[1], nv[2], cp.gap0), ","))
    end
end

# G_nn at centre pair
k0 = argmin(abs(dad1.Nodes[cp.node_a][1]) for cp in pairs)
cp0 = pairs[k0]
pr1 = prep[cp0.reg_a]
cols = pr1.Gc_cols[cp0.node_a]
iu = 2cp0.node_a - 1
Gnn = pr1.G_local[iu, first(cols)]
@printf("Julia nodes pad=%d spec=%d NPc=%d  h[%.3e, %.3e] n_h0=%d\n",
    prob.regions[1].n, prob.regions[2].n, length(pairs), extrema(h)..., count(==(0.0), h))
@printf("centre x=%.4f  G_nn=%.6e  ||A_r||=%.4e ||b||=%.4e\n",
    dad1.Nodes[cp0.node_a][1], Gnn, norm(pr1.A), norm(pr1.b))

x0 = zeros(N)
MR._verify_contact_states!(pairs, prep, h, x0; epsc=1e-7)
@printf("step0 verify  open=%d stick=%d slip=%d\n",
    count(==(1), (cp.state for cp in pairs)),
    count(==(3), (cp.state for cp in pairs)),
    count(s -> abs(s)==2, (cp.state for cp in pairs)))

nsteps = 50
xtot = zeros(N)
x = zeros(N)
open(joinpath(OUT, "jul_steps.csv"), "w") do io
    println(io, "step,n_open,n_stick,n_slip,p0,qmax,nit")
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
        @printf("jul s=%2d  nit=%2d  o/s/l=%d/%d/%d  p0=%.3f  qmax=%.4f\n", s, nit, n_o, n_s, n_l, p0, qmax)
        println(io, join((s, n_o, n_s, n_l, p0, qmax, nit), ","))
        if s == 1 || s == nsteps
            open(joinpath(OUT, "jul_iface_s$(s).csv"), "w") do f2
                println(f2, "x,tn,tt,state,h")
                for cp in pairs
                    xa = dad1.Nodes[cp.node_a][1]
                    println(f2, join((xa, cp.tn, cp.tt, cp.state, cp.gap0), ","))
                end
            end
        end
    end
end
println("Done Julia trace.")
