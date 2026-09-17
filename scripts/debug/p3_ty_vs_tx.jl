using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf, StaticArrays
include(datadir("Laplace", "Laplace_dad.jl"))

MAT1 = (E1=124.04e3, E2=10.09e3, G12=6.03e3, ν12=0.334, η12_1=1.255, η12_2=-0.031)
P = 1000.0
Ri, Ro = 300.0, 600.0
props = AnisotropicElasticity(lekhnitskii_engineering(
    MAT1.E1, MAT1.E2, MAT1.G12, MAT1.ν12; η12_1=MAT1.η12_1, η12_2=MAT1.η12_2))

function apply_p3_bc!(dad; load::Symbol=:ty)
    # Fig. 13: u=0 on vertical (x≈0); traction on bottom (y≈0); arcs free.
    for i in 1:dad.n
        x, y = dad.Nodes[i][1], dad.Nodes[i][2]
        if x < 1.0
            dad.BC[2i-1] = 0; dad.BV[2i-1] = 0.0
            dad.BC[2i] = 0; dad.BV[2i] = 0.0
        elseif y < 1.0 && x > Ri - 1
            dad.BC[2i-1] = 1; dad.BC[2i] = 1
            if load === :ty
                dad.BV[2i-1] = 0.0
                dad.BV[2i] = -P
            else
                dad.BV[2i-1] = P
                dad.BV[2i] = 0.0
            end
        else
            dad.BC[2i-1] = 1; dad.BV[2i-1] = 0.0
            dad.BC[2i] = 1; dad.BV[2i] = 0.0
        end
    end
    return dad
end

function run(msh, load)
    dad = format2d(msh, props; tipo=2, pontointerno=false)
    apply_p3_bc!(dad; load=load)
    assemble!(dad; npg=16, threaded=false); solve(dad)
    uc = copy(dad.u)
    ux = [dad.u[2i-1] for i in 1:dad.n]
    uy = [dad.u[2i] for i in 1:dad.n]
    H_G_hyper(dad; npg=50, threaded=false); solve(dad)
    hx = [dad.u[2i-1] for i in 1:dad.n]
    hy = [dad.u[2i] for i in 1:dad.n]
    rel = norm(dad.u .- uc) / (norm(uc) + 1e-30)
    @printf("%-16s %-4s  CBIE Ux=%.2f cm  Uy=%.2f cm  |u|=%.2f cm   HBIE Ux=%.2f Uy=%.2f |u|=%.2f cm  rel=%.3e  κ=%.2e\n",
        basename(msh), load,
        extrema(ux)[1]/10, extrema(uy)[1]/10, maximum(abs, uc)/10,
        extrema(hx)[1]/10, extrema(hy)[1]/10, maximum(abs, dad.u)/10,
        rel, cond(Matrix(dad.A)))
end

for msh in (datadir("elastico", "p3_cmp_q2.msh"), datadir("elastico", "p3_cmp_lin.msh"))
    run(msh, :tx)
    run(msh, :ty)
end
