using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf, StaticArrays
include(datadir("Laplace", "Laplace_dad.jl"))

P = 1000.0
Ri = 300.0
function apply_ty!(dad)
    for i in 1:dad.n
        x, y = dad.Nodes[i][1], dad.Nodes[i][2]
        if x < 1.0
            dad.BC[2i-1] = 0; dad.BV[2i-1] = 0.0
            dad.BC[2i] = 0; dad.BV[2i] = 0.0
        elseif y < 1.0 && x > Ri - 1
            dad.BC[2i-1] = 1; dad.BV[2i-1] = 0.0
            dad.BC[2i] = 1; dad.BV[2i] = -P
        else
            dad.BC[2i-1] = 1; dad.BV[2i-1] = 0.0
            dad.BC[2i] = 1; dad.BV[2i] = 0.0
        end
    end
end

function run(lab, props)
    dad = format2d(datadir("elastico", "p3_cmp_q2.msh"), props; tipo=2, pontointerno=false)
    apply_ty!(dad)
    assemble!(dad; npg=16, threaded=false); solve(dad)
    ux = extrema(dad.u[1:2:end])
    uy = extrema(dad.u[2:2:end])
    @printf("%-28s  Ux=[%.1f, %.1f] cm  Uy=[%.1f, %.1f] cm  |u|=%.1f cm\n",
        lab, ux[1]/10, ux[2]/10, uy[1]/10, uy[2]/10, maximum(abs, dad.u)/10)
end

E1, E2, G, ν = 124.04e3, 10.09e3, 6.03e3, 0.334
η1, η2 = 1.255, -0.031
function rotC(C, θdeg)
    θ = θdeg * π / 180
    m, n = cos(θ), sin(θ)
    Rot = @SMatrix [m^2 n^2 2m*n; n^2 m^2 -2m*n; -m*n m*n m^2-n^2]
    return inv(Rot) * C * inv(Rot')
end
C0 = lekhnitskii_engineering(E1, E2, G, ν; η12_1=η1, η12_2=η2).C
run("MAT1 aligned", AnisotropicElasticity(lekhnitskii_engineering(E1, E2, G, ν; η12_1=η1, η12_2=η2)))
run("MAT1 θ=90 +η", AnisotropicElasticity(lekhnitskii_params(rotC(C0, 90))))
run("MAT1 θ=-90 +η", AnisotropicElasticity(lekhnitskii_params(rotC(C0, -90))))
run("ortho θ=90", AnisotropicElasticity(lekhnitskii_params(E1, E2, G, ν; θ_deg=90)))
run("iso E2", Elasticity(E2, ν, 1.0; plane_stress=true))
run("iso E1", Elasticity(E1, ν, 1.0; plane_stress=true))
