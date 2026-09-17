ENV["GKSwstype"] = "100"
using DrWatson
@quickactivate :BEM
using BEM.MultiRegion
using LinearAlgebra, Printf, Statistics
include(datadir("elastico", "two_blocks_contact.jl"))
include(datadir("elastico", "loyola_bulk_contact.jl"))
const MR = BEM.MultiRegion

function contato_as!(prob)
    ctx = MR._contact_friction_setup(prob; method=:ntn, npg=10)
    prep, pairs = ctx.prep, ctx.pairs
    h = [cp.gap0 for cp in pairs]
    for cp in pairs; cp.state = 3; cp.ut_lock = 0.0; end
    x0 = zeros(ctx.N)
    prev = fill(3, length(pairs))
    for it in 1:40
        it > 1 && MR._verify_contact_states!(pairs, prep, h, x0; epsc=1e-7)
        st = [cp.state for cp in pairs]
        n_flip = count(i -> st[i] != prev[i], eachindex(st))
        A, b = MR._assemble_contact_system(prep, pairs, h, x0)
        x = A \ b
        dist = norm(x - x0) / max(1.0, norm(x))
        x0 = x; prev = st
        (dist < 1e-9 && n_flip == 0) && break
    end
    MR._verify_contact_states!(pairs, prep, h, x0; epsc=1e-7)
    MR._scatter_contact_solution!(prob, prep, pairs, x0)
    return prep, pairs, x0
end

function run()
prob, par = load_loyola_bulk_contact(; μ=0.3, ndiv_c=12, ndiv_f=6, ndiv_s=4, ndiv_top=5,
    tipo=2, nome="ux932")
dad1 = prob.regions[1]
ymax = maximum(pt[2] for pt in dad1.Nodes)
best, bx = 0, Inf
for i in 1:dad1.n
    abs(dad1.Nodes[i][2]-ymax) <= 1e-9*max(abs(ymax),1.0) || continue
    ax = abs(dad1.Nodes[i][1]); ax < bx && (bx = ax; best = i)
end
dad1.BC[2best-1] = 0; dad1.BV[2best-1] = 0.0
prep, pairs, xsol = contato_as!(prob)
nx = sum(p.ndof for p in prep)
dad2 = prob.regions[2]
println("x        ux_pad    ux_spec   Δux      uy_pad    uy_spec   tn      tt     util  st")
for cp in sort(prob.contacts; by=c -> dad1.Nodes[c.node_a][1])
    abs(cp.state)==1 && continue
    abs(dad1.Nodes[cp.node_a][1]) > 1.8 && continue
    i, j = cp.node_a, cp.node_b
    uxp, uyp = dad1.u[2i-1], dad1.u[2i]
    uxs, uys = dad2.u[2j-1], dad2.u[2j]
    Δux = uxp - uxs
    util = abs(cp.tt)/max(0.3*abs(cp.tn), 1e-12)
    @printf("%+7.4f  %+8.5f  %+8.5f  %+8.5f  %+8.5f  %+8.5f  %+7.1f %+7.1f  %.3f  %2d\n",
        dad1.Nodes[i][1], uxp, uxs, Δux, uyp, uys, cp.tn, cp.tt, util, cp.state)
end
# kinematics gt vs Δux
println("\ncompare gt (contact frame) vs Δux (global):")
for (k, cp) in enumerate(pairs)
    abs(cp.state)==1 && continue
    x = dad1.Nodes[cp.node_a][1]
    abs(x) > 0.4 && continue
    kin = MR._contact_pair_kinematics(prep, cp, cp.gap0, xsol, k, nx)
    @printf("  x=%+6.3f  gt=%+.5e  gn=%+.5e  dut=%+.5e  un1=%+.5f ut1=%+.5f un2=%+.5f ut2=%+.5f\n",
        x, kin.gt, kin.gn, kin.dut, kin.un1, kin.ut1, kin.un2, kin.ut2)
end
end
run()
