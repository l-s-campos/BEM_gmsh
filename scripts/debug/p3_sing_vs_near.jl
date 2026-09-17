# Split HBIE residual of the CBIE solution: on-element vs off-element.
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf
include(datadir("Laplace", "Laplace_dad.jl"))

MAT1 = (E1=124.04e3, E2=10.09e3, G12=6.03e3, ν12=0.334, η12_1=1.255, η12_2=-0.031)
props = AnisotropicElasticity(lekhnitskii_engineering(
    MAT1.E1, MAT1.E2, MAT1.G12, MAT1.ν12; η12_1=MAT1.η12_1, η12_2=MAT1.η12_2))

function assemble_split(dad; npg, nearfield=:euclid)
    set_cache!(dad; nearfield=nearfield)
    BEM._init_quadrature!(dad, npg)
    dim = 2; n = dad.n
    Hon = zeros(2n, 2n); Gon = zeros(2n, 2n)
    Hoff = zeros(2n, 2n); Goff = zeros(2n, 2n)
    for i in 1:n
        pf = dad.Nodes[i]; nf = dad.Normal[i]
        ii = BEM.expand(i, dim)
        f = (d, r, nrm) -> fundamental_hyper(d, r, nrm, nf)
        for el in dad.elements
            xj = dad.Nodes[el.index]
            jj = BEM.expand(el.index, dim)
            hloc = zeros(dim, length(jj)); gloc = zeros(dim, length(jj))
            BEM.integrate_element(dad, el, xj, pf, hloc, gloc, f;
                orders=(-1, -2), source=i)
            if i in el.index
                Hon[ii, jj] .+= hloc; Gon[ii, jj] .+= gloc
            else
                Hoff[ii, jj] .+= hloc; Goff[ii, jj] .+= gloc
            end
        end
    end
    return Hon, Gon, Hoff, Goff
end

function run(msh, label; nearfield=:euclid)
    dad = format2d(msh, props; tipo=2, pontointerno=false)
    assemble!(dad; npg=16, threaded=false); solve(dad)
    uc, tc = copy(dad.u), copy(dad.traction)
    Hon, Gon, Hoff, Goff = assemble_split(dad; npg=50, nearfield=nearfield)
    # same jump as H_G_hyper, on the on-element G only (collocation is on-element)
    n = dad.n
    for i in 1:n
        Gon[2i-1, 2i-1] -= 0.5
        Gon[2i, 2i] -= 0.5
    end
    r_on  = Hon * uc - Gon * tc
    r_off = Hoff * uc - Goff * tc
    r_all = r_on + r_off
    gt = (Gon + Goff) * tc
    println(label, "  n=$(dad.n)  nearfield=$nearfield")
    @printf("  ||H_on u - G_on t|| / ||Gt|| = %.3e\n", norm(r_on)/(norm(gt)+1e-30))
    @printf("  ||H_off u - G_off t|| / ||Gt|| = %.3e\n", norm(r_off)/(norm(gt)+1e-30))
    @printf("  ||H_all u - G_all t|| / ||Gt|| = %.3e\n", norm(r_all)/(norm(gt)+1e-30))
    @printf("  ||H_on||=%.3e  ||H_off||=%.3e  ||G_on||=%.3e  ||G_off||=%.3e\n",
        norm(Hon), norm(Hoff), norm(Gon), norm(Goff))
    # Dirichlet vs Neumann rows of the split residual
    nd = 0; nn = 0; eon_d=0.0; eon_n=0.0; eoff_d=0.0; eoff_n=0.0
    for i in 1:2n
        if dad.BC[i] == 0
            nd += 1
            eon_d = max(eon_d, abs(r_on[i])); eoff_d = max(eoff_d, abs(r_off[i]))
        else
            nn += 1
            eon_n = max(eon_n, abs(r_on[i])); eoff_n = max(eoff_n, abs(r_off[i]))
        end
    end
    @printf("  max |res_on|  Dir=%.3e  Neu=%.3e\n", eon_d, eon_n)
    @printf("  max |res_off| Dir=%.3e  Neu=%.3e\n", eoff_d, eoff_n)
end

println("CBIE solution plugged into split HBIE operator")
run(datadir("elastico", "p3_cmp_q2.msh"), "circular (nodes on CAD arc)")
run(datadir("elastico", "p3_cmp_lin.msh"), "chordal (mid-nodes on secant)")
println("done")
