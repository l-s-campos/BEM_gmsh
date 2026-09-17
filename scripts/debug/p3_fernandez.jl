# Thesis §7.2 / Fernández 2012 constants on the 90° ring (Fig. 7.7–7.10).
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf, StaticArrays
include(datadir("Laplace", "Laplace_dad.jl"))

# Ex, Ey, Gxy, νyx, ηxy,x, ηxy,y  and EPD: νzy, νzx, ηxy,z
Ex, Ey, Gxy = 124.04e3, 10.09e3, 6.03e3
νyx = 0.344
ηx, ηy = 1.255, -0.031
νzy, νzx, ηz = 0.25, 0.40, 0.50
P, Ri = 1000.0, 300.0

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

function rotC(C, θdeg)
    θ = θdeg * π / 180
    m, n = cos(θ), sin(θ)
    Rot = @SMatrix [m^2 n^2 2m*n; n^2 m^2 -2m*n; -m*n m*n m^2-n^2]
    return inv(Rot) * C * inv(Rot')
end

function makeD(; D12, plane_strain=false)
    D11, D22, D66 = 1/Ex, 1/Ey, 1/Gxy
    D16, D26 = ηx/Ex, ηy/Ey
    D = @SMatrix [D11 D12 D16; D12 D22 D26; D16 D26 D66]
    if plane_strain
        E3 = Ey
        D33 = 1/E3
        D13 = -νzx / E3          # νzx = 0.40
        D23 = -νzy / E3          # νzy = 0.25
        D36 = ηz / E3
        v3 = SVector(D13, D23, D36)
        D = D - (v3 * v3') / D33
    end
    ev = eigvals(Symmetric(Matrix(D)))
    return D, ev
end

function run(lab, props)
    dad = format2d(datadir("elastico", "p3_cmp_q2.msh"), props; tipo=2, pontointerno=false)
    apply_ty!(dad)
    assemble!(dad; npg=16, threaded=false); solve(dad)
    uc = copy(dad.u)
    ux, uy = extrema(uc[1:2:end]), extrema(uc[2:2:end])
    H_G_hyper(dad; npg=50, threaded=false); solve(dad)
    hx, hy = extrema(dad.u[1:2:end]), extrema(dad.u[2:2:end])
    rel = norm(dad.u .- uc) / (norm(uc) + 1e-30)
    @printf("%-32s  CBIE Ux=%.1f  Uy=%.1f  |u|=%.1f cm   HBIE Ux=%.1f Uy=%.1f  rel=%.2e\n",
        lab, ux[1]/10, uy[1]/10, maximum(abs, uc)/10, hx[1]/10, hy[1]/10, rel)
end

println("=== D12 conventions ===")
for (lab, D12) in (("−νyx/Ey (literal)", -νyx/Ey), ("−νyx/Ex (large Poisson)", -νyx/Ex))
    D, ev = makeD(; D12=D12)
    @printf("  %-28s  D12=%.4e  eig(D)=%s  PD(C)=%s\n",
        lab, D12, string(round.(ev; sigdigits=3)), string(all(>(0), eigvals(inv(Matrix(D))))))
end

println("\n=== EPT (plane stress), ty=−P ===")
# A: D12 = -νyx/Ex  (same as current code with ν12=0.344)
pA = AnisotropicElasticity(lekhnitskii_engineering(Ex, Ey, Gxy, νyx; η12_1=ηx, η12_2=ηy))
run("ν=0.344 as ν12, aligned", pA)
run("ν=0.344, θ=45", AnisotropicElasticity(lekhnitskii_params(rotC(pA.params.C, 45))))
run("ν=0.344, θ=90", AnisotropicElasticity(lekhnitskii_params(rotC(pA.params.C, 90))))

# B: D12 = -νyx/Ey
Dlit, ev = makeD(; D12=-νyx/Ey)
if all(>(0), eigvals(inv(Matrix(Dlit))))
    pB = AnisotropicElasticity(lekhnitskii_params(inv(Dlit)))
    run("D12=−νyx/Ey aligned", pB)
    run("D12=−νyx/Ey θ=90", AnisotropicElasticity(lekhnitskii_params(rotC(pB.params.C, 90))))
else
    println("  D12=−νyx/Ey: C not PD, skip BEM")
end

println("\n=== EPD (plane strain), ty=−P ===")
# Maxwell: D13=-νzx/Ez = -νxz/Ex. νzx=0.40 cannot sit on Ez=Ey (implied νxz≈4.9).
# Use D13=-νzx/Ex, D23=-νzy/Ey (0.40 and 0.25 as in-plane Poisson with Ex, Ey).
function epd_props(; E3=Ey, d13_over=Ex, d23_over=Ey)
    D11, D22, D66 = 1/Ex, 1/Ey, 1/Gxy
    D12 = -νyx/Ex
    D16, D26 = ηx/Ex, ηy/Ey
    D = @SMatrix [D11 D12 D16; D12 D22 D26; D16 D26 D66]
    D33 = 1/E3
    D13 = -νzx / d13_over
    D23 = -νzy / d23_over
    D36 = ηz / E3
    v3 = SVector(D13, D23, D36)
    Ds = D - (v3 * v3') / D33
    ev = eigvals(Symmetric(Matrix(Ds)))
    @printf("  E3=%.0f  D13=%.3e D23=%.3e  eig(D*)=%s  PD=%s\n",
        E3, D13, D23, string(round.(ev; sigdigits=3)), string(all(>(0), ev)))
    all(>(0), ev) || return nothing
    return AnisotropicElasticity(lekhnitskii_params(inv(Ds)))
end
for (lab, kw) in (
        ("E3=Ey, D13=-νzx/Ey (code default)", (; E3=Ey, d13_over=Ey, d23_over=Ey)),
        ("E3=Ey, D13=-νzx/Ex", (; E3=Ey, d13_over=Ex, d23_over=Ey)),
        ("E3=Ex, D13=-νzx/Ex", (; E3=Ex, d13_over=Ex, d23_over=Ey)),
    )
    println("  -- ", lab)
    p = epd_props(; kw...)
    p === nothing && continue
    run("EPD aligned "*lab[1:15], p)
    run("EPD θ=90  "*lab[1:15], AnisotropicElasticity(lekhnitskii_params(rotC(p.params.C, 90))))
end
println("done")
