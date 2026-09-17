# Original P3 BCs: residual of HBIE operators on the CBIE solution (not a polynomial field).
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf
include(datadir("Laplace", "Laplace_dad.jl"))

MAT1 = (E1=124.04e3, E2=10.09e3, G12=6.03e3, ν12=0.334, η12_1=1.255, η12_2=-0.031)
props = AnisotropicElasticity(lekhnitskii_engineering(
    MAT1.E1, MAT1.E2, MAT1.G12, MAT1.ν12; η12_1=MAT1.η12_1, η12_2=MAT1.η12_2))

function group_of(p)
    r = hypot(p[1], p[2])
    p[1] < 1.0 && return :left
    p[2] < 1.0 && return :bottom
    r < 450 && return :inner
    return :outer
end

function report_r(tag, r, dad)
    n = dad.n
    dir_n = 0.0; neu_n = 0.0; nd = 0; nn = 0
    acc = Dict(:left=>0.0, :bottom=>0.0, :inner=>0.0, :outer=>0.0)
    cnt = Dict(:left=>0, :bottom=>0, :inner=>0, :outer=>0)
    @inbounds for i in 1:n
        ri = hypot(r[2i-1], r[2i])
        g = group_of(dad.Nodes[i])
        acc[g] += ri; cnt[g] += 1
        if dad.BC[2i-1] == 0 && dad.BC[2i] == 0
            dir_n += ri; nd += 1
        else
            neu_n += ri; nn += 1
        end
    end
    @printf("  %-28s  ||r||=%.3e  max|r|=%.3e\n", tag, norm(r), maximum(abs, r))
    @printf("    mean|r|  Dir-nodes=%.3e (n=%d)  Neu-nodes=%.3e (n=%d)\n",
        dir_n / max(nd, 1), nd, neu_n / max(nn, 1), nn)
    @printf("    mean|r|  left=%.2e  bottom=%.2e  inner=%.2e  outer=%.2e\n",
        acc[:left]/max(cnt[:left],1), acc[:bottom]/max(cnt[:bottom],1),
        acc[:inner]/max(cnt[:inner],1), acc[:outer]/max(cnt[:outer],1))
end

function run(msh, near)
    println("\n", "="^72)
    println(msh, "  near=", near)
    dad = format2d(msh, props; tipo=2, pontointerno=false)
    assemble!(dad; npg=16, threaded=false); solve(dad)
    uc, tc = copy(dad.u), copy(dad.traction)
    Hc, Gc = copy(dad.H), copy(dad.G)
    @printf("  CBIE  max|u|=%.3f  ||Hc uc - Gc tc||=%.3e\n",
        maximum(abs, uc), norm(Hc * uc - Gc * tc))

    set_cache!(dad; nearfield=near)
    H_G_hyper(dad; npg=50, threaded=false)
    Hh, Gh = copy(dad.H), copy(dad.G)
    Hu = Hh * uc
    Gt = Gh * tc
    r = Hu .- Gt
    @printf("  HBIE on CBIE field  ||H'uc||=%.3e  ||G'tc||=%.3e  ||r||/scale=%.3e\n",
        norm(Hu), norm(Gt), norm(r) / (norm(Hu) + norm(Gt) + 1e-30))
    report_r("H'uc − G'tc", r, dad)
    report_r("H'uc only", Hu, dad)
    report_r("G'tc only", Gt, dad)

    # mixed-system residual of the CBIE (u,t) in HBIE unknowns
    applyBC(dad)
    x = zeros(2 * dad.n)
    @inbounds for dof in 1:2*dad.n
        x[dof] = dad.BC[dof] == 0 ? tc[dof] : uc[dof]
    end
    rb = dad.A * x - dad.b
    report_r("A x_cbie − b  (mixed)", rb, dad)

    solve(dad)
    @printf("  HBIE solve  max|u|=%.3f  relCBIE=%.3e  cond=%.2e\n",
        maximum(abs, dad.u), norm(dad.u .- uc) / (norm(uc) + 1e-30), cond(Matrix(dad.A)))
end

run(datadir("elastico", "p3_cmp_q2.msh"), :euclid)
run(datadir("elastico", "p3_cmp_q2.msh"), :tangent)
run(datadir("elastico", "p3_cmp_lin.msh"), :euclid)
println("\ndone")
