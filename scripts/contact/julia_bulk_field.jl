# 1-stick displacement field: global uy and nearfield=:plain vs :euclid.
ENV["GKSwstype"] = "100"
using DrWatson
@quickactivate :BEM
using BEM.MultiRegion
using LinearAlgebra, Printf

include(datadir("elastico", "dad_contato_bulk.jl"))
const MR = BEM.MultiRegion

function run(label; nearfield=:euclid, npg=30)
    prob, par = load_dad_contato_bulk(; ndiv_c=45, ndiv_s=15, tipo=2,
        gap=:euclidean, nome="jfield_$(label)")
    for dad in prob.regions
        set_cache!(dad; nearfield=nearfield)
    end
    ctx = MR._contact_friction_setup(prob; method=:ntn, npg=npg,
        common_normal=false, near_factor=Inf, singular=:telles)
    prep, pairs = ctx.prep, ctx.pairs
    dad1, dad2 = prob.regions
    h = [cp.gap0 for cp in pairs]
    nx = sum(p.ndof for p in prep)
    nsteps = 50
    x = zeros(ctx.N)
    MR._verify_contact_states!(pairs, prep, h, x; epsc=1e-7)
    A, b = MR._assemble_contact_system(prep, pairs, h, x)
    b[1:nx] ./= nsteps
    for (k, cp) in enumerate(pairs)
        abs(cp.state) == 1 && continue
        b[nx + 4(k - 1) + 1] = h[k]
    end
    x = A \ b
    MR._verify_contact_states!(pairs, prep, h, x; epsc=1e-7)
    ymax = maximum(pt[2] for pt in dad1.Nodes)
    itop = argmin(i -> (abs(dad1.Nodes[i][2] - ymax), abs(dad1.Nodes[i][1])), 1:dad1.n)
    @printf("\n===== %s  nearfield=%s =====\n", label, nearfield)
    @printf("pad top node %d x=%.4f  BC=(%d,%d) BV=(%.3f,%.3f)\n",
        itop, dad1.Nodes[itop][1], dad1.BC[2itop-1], dad1.BC[2itop],
        dad1.BV[2itop-1], dad1.BV[2itop])
    # G top → contact centre
    k0 = argmin(abs(dad1.Nodes[cp.node_a][1]) for cp in pairs)
    ic = pairs[k0].node_a
    G = dad1.G
    @printf("G(top,centre) Gxx=%.4e Gxy=%.4e Gyx=%.4e Gyy=%.4e\n",
        G[2ic-1, 2itop-1], G[2ic-1, 2itop], G[2ic, 2itop-1], G[2ic, 2itop])
    ord = sortperm(1:length(pairs); by=k -> abs(dad1.Nodes[pairs[k].node_a][1]))
    p0 = maximum(-cp.tn for cp in pairs if abs(cp.state) != 1; init=0.0)
    @printf("1-stick p0=%.4f  o/s/l=%d/%d/%d\n", p0,
        count(==(1), (cp.state for cp in pairs)),
        count(==(3), (cp.state for cp in pairs)),
        count(s -> abs(s)==2, (cp.state for cp in pairs)))
    for k in ord[1:5]
        cp = pairs[k]
        kin = MR._contact_pair_kinematics(prep, cp, h[k], x, k, nx)
        R1 = BEM.node_rotation2d(dad1.Normal[cp.node_a])
        R2 = BEM.node_rotation2d(dad2.Normal[cp.node_b])
        ug1 = R1 * [kin.un1, kin.ut1]
        ug2 = R2 * [kin.un2, kin.ut2]
        @printf("  x=%+.5f st=%d tn=%7.3f  un=(%+.3e,%+.3e)  uy=(%+.3e,%+.3e)  dun=%.3e dut=%.3e\n",
            dad1.Nodes[cp.node_a][1], cp.state, kin.tn1,
            kin.un1, kin.un2, ug1[2], ug2[2], kin.dun, kin.dut)
    end
end

run("euclid"; nearfield=:euclid)
run("plain"; nearfield=:plain)
println("Done.")
