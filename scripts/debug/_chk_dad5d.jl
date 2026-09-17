using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf
include(datadir("elastico", "two_blocks_contact.jl"))
include(datadir("elastico", "dad_5d_contact.jl"))

function main()
    prob, par = load_dad_5d_contact(; μ=0.0, ndiv_c=8, ndiv_f=4, ndiv_s=4, ndiv_top=8, nome="chk")
    dad = prob.regions[1]
    ymax = maximum(pt[2] for pt in dad.Nodes)
    println("upper n=$(dad.n) ymax=$ymax")
    n_top = 0
    for i in 1:dad.n
        if abs(dad.Nodes[i][2] - ymax) < 1e-6
            n_top += 1
            if n_top == 1
                @printf("top BC=(%d,%d) BV=(%.2f,%.2f) n=(%.2f,%.2f)\n",
                    dad.BC[2i-1], dad.BC[2i], dad.BV[2i-1], dad.BV[2i],
                    dad.Normal[i][1], dad.Normal[i][2])
            end
        end
    end
    println("top nodes=$n_top  pairs=$(length(prob.contacts))")
    println("gap0 ", extrema(cp.gap0 for cp in prob.contacts))
    for cp in prob.contacts[1:min(5, end)]
        n1 = prob.regions[1].Normal[cp.node_a]
        n2 = prob.regions[2].Normal[cp.node_b]
        @printf("x=%7.3f n1=(%5.2f,%5.2f) n2=(%5.2f,%5.2f) gap=%.5f\n",
            dad.Nodes[cp.node_a][1], n1[1], n1[2], n2[1], n2[2], cp.gap0)
    end

    ctx = BEM._contact_friction_setup(prob; method=:ntn, npg=8)
    prep, pairs = ctx.prep, ctx.pairs
    h = [cp.gap0 for cp in pairs]
    x = zeros(ctx.N)
    for cp in pairs; cp.state = 3; end
    A, b = BEM._assemble_contact_system(prep, pairs, h, x)
    println("cond(A)=", cond(A))
    x = A \ b
    println("||x||=", norm(x), " max|x|=", maximum(abs, x))
    BEM._verify_contact_states!(pairs, prep, h, x; epsc=1e-7)
    println("after 1 solve: open=$(count(cp->cp.state==1,pairs)) stick=$(count(cp->cp.state==3,pairs))")
    nx = sum(p.ndof for p in prep)
    tn = [x[nx + 4(k-1) + 1] for k in 1:length(pairs)]
    println("tn range ", extrema(tn), " mean closed tn ",
        mean(tn[i] for i in 1:length(pairs) if pairs[i].state != 1; init=0.0))
end
main()
