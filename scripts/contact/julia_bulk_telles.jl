# 1-stick centre element: Guiggiani vs Telles on-element (Contato calc_gh).
ENV["GKSwstype"] = "100"
using DrWatson
@quickactivate :BEM
using BEM.MultiRegion
using LinearAlgebra, Printf

include(datadir("elastico", "dad_contato_bulk.jl"))
const MR = BEM.MultiRegion
const OUT = joinpath(projectdir(), "plots", "cattaneo_mindlin", "octave_bulk")
mkpath(OUT)

function center_nodes(dad, pairs)
    ks = sortperm(pairs; by=cp -> abs(dad.Nodes[cp.node_a][1]))
    na = [pairs[k].node_a for k in ks[1:3]]
    # sort by x
    sort!(na; by=i -> dad.Nodes[i][1])
    return na
end

function dump_HG_block(io, tag, dad, nodes)
    H, G = dad.H, dad.G
    println(io, "tag,i,j,Hxx,Hxy,Hyx,Hyy,Gxx,Gxy,Gyx,Gyy")
    for (a, ia) in enumerate(nodes), (b, ib) in enumerate(nodes)
        ri, ci = 2ia - 1, 2ib - 1
        println(io, join((tag, a, b,
            H[ri, ci], H[ri, ci+1], H[ri+1, ci], H[ri+1, ci+1],
            G[ri, ci], G[ri, ci+1], G[ri+1, ci], G[ri+1, ci+1]), ","))
    end
end

function run_mode(label; singular=:guiggiani, npg=30)
    prob, par = load_dad_contato_bulk(; ndiv_c=45, ndiv_s=15, tipo=2,
        gap=:euclidean, nome="jtelles_$(label)")
    dad1 = prob.regions[1]
    ctx = MR._contact_friction_setup(prob; method=:ntn, npg=npg,
        common_normal=false, near_factor=Inf, singular=singular)
    prep, pairs = ctx.prep, ctx.pairs
    na = center_nodes(dad1, pairs)
    @printf("\n===== %s  singular=%s  npg=%d  centre nodes x=%s =====\n",
        label, singular, npg, [round(dad1.Nodes[i][1]; digits=5) for i in na])
    # H coupling centre→neighbour (global xy, before local)
    ic, il, ir = na[2], na[1], na[3]
    H = dad1.H
    @printf("  H(left,centre)  Hxx=%.4e Hxy=%.4e Hyx=%.4e Hyy=%.4e\n",
        H[2il-1, 2ic-1], H[2il-1, 2ic], H[2il, 2ic-1], H[2il, 2ic])
    @printf("  G(left,centre)  Gxx=%.4e Gxy=%.4e Gyx=%.4e Gyy=%.4e\n",
        dad1.G[2il-1, 2ic-1], dad1.G[2il-1, 2ic], dad1.G[2il, 2ic-1], dad1.G[2il, 2ic])
    open(joinpath(OUT, "jul_$(label)_Hcenter.csv"), "w") do io
        dump_HG_block(io, label, dad1, na)
    end

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
    ord = sortperm(1:length(pairs); by=k -> abs(dad1.Nodes[pairs[k].node_a][1]))
    n_o = count(==(1), (cp.state for cp in pairs))
    n_s = count(==(3), (cp.state for cp in pairs))
    n_l = count(s -> abs(s)==2, (cp.state for cp in pairs))
    p0 = maximum(-cp.tn for cp in pairs if abs(cp.state) != 1; init=0.0)
    @printf("  after 1-stick  o/s/l=%d/%d/%d  p0=%.4f\n", n_o, n_s, n_l, p0)
    for k in ord[1:5]
        cp = pairs[k]
        kin = MR._contact_pair_kinematics(prep, cp, h[k], x, k, nx)
        @printf("  x=%+.5f st=%d tn=%8.3f tt=%8.4f dun=%.4e dut=%.4e h=%.4e un1=%.4e\n",
            dad1.Nodes[cp.node_a][1], cp.state, kin.tn1, kin.tt1,
            kin.dun, kin.dut, h[k], kin.un1)
    end
    return p0
end

run_mode("guig"; singular=:guiggiani, npg=30)
run_mode("telles"; singular=:telles, npg=30)
println("Done.")
