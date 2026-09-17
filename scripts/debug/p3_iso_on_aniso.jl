# Cordeiro P3: isotropic constants on the Lekhnitskii kernel vs native Kelvin.
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf, StaticArrays
include(datadir("Laplace", "Laplace_dad.jl"))

E, ν = 124.04e3, 0.334
Giso = E / (2 * (1 + ν))
kel = Elasticity(E, ν, 1.0; plane_stress=true)

function try_lek(E1, E2, G12, ν12; η12_1=0.0, η12_2=0.0)
    p = lekhnitskii_engineering(E1, E2, G12, ν12; η12_1=η12_1, η12_2=η12_2)
    return AnisotropicElasticity(p)
end

println("=== Lekhnitskii roots ===")
for (lab, args) in (
        ("exact iso", (E, E, Giso, ν)),
        ("iso E2=1.001 E", (E, 1.001E, Giso, ν)),
        ("iso E2=1.01 E", (E, 1.01E, Giso, ν)),
        ("MAT1", (E, 10.09e3, 6.03e3, ν)),
    )
    try
        lek = try_lek(args...)
        μ = lek.params.mi
        @printf("%-20s  μ1=%.6f%+.6fi  μ2=%.6f%+.6fi  |μ1-μ2|=%.3e  ||A||=%.3e\n",
            lab, real(μ[1]), imag(μ[1]), real(μ[2]), imag(μ[2]),
            abs(μ[1] - μ[2]), norm(lek.params.A))
    catch e
        println(lab, "  ERROR: ", e)
    end
end

lek_iso = try_lek(E, E, Giso, ν)
r = Point2D(0.3, 0.4); n = Point2D(0.0, 1.0); nf = Point2D(1.0, 0.0)
Uk, Tk = let kp = fundamental(kel, r, n); BEM._to_smat(kp.U), BEM._to_smat(kp.T) end
Ul, Tl = let kp = fundamental(lek_iso, r, zero(r), n); BEM._to_smat(kp.U), BEM._to_smat(kp.T) end
hk = fundamental_hyper(kel, r, n, nf)
hl = fundamental_hyper(lek_iso, r, zero(r), n, nf)
@printf("\nkernel rel  U=%.3e  T=%.3e  Uh=%.3e  Th=%.3e\n",
    norm(Ul - Uk) / (norm(Uk) + 1e-30),
    norm(Tl - Tk) / (norm(Tk) + 1e-30),
    norm(BEM._to_smat(hl.U) - BEM._to_smat(hk.U)) / (norm(BEM._to_smat(hk.U)) + 1e-30),
    norm(BEM._to_smat(hl.T) - BEM._to_smat(hk.T)) / (norm(BEM._to_smat(hk.T)) + 1e-30))

function run(lab, msh, props; npg_h=50)
    dad = format2d(msh, props; tipo=2, pontointerno=false)
    assemble!(dad; npg=16, threaded=false); solve(dad)
    uc = copy(dad.u)
    cc = cond(Matrix(dad.A))
    H_G_hyper(dad; npg=npg_h, threaded=false); solve(dad)
    uh = copy(dad.u)
    ch = cond(Matrix(dad.A))
    rel = norm(uh .- uc) / (norm(uc) + 1e-30)
    @printf("%-28s  CBIE %10.3f  (κ=%.2e)  HBIE %10.3f  (κ=%.2e)  rel=%.3e\n",
        lab, maximum(abs, uc), cc, maximum(abs, uh), ch, rel)
    return (; uc, uh, rel, cc, ch)
end

msh_q2 = datadir("elastico", "p3_cmp_q2.msh")
msh_lin = datadir("elastico", "p3_cmp_lin.msh")
lek_mat1 = try_lek(E, 10.09e3, 6.03e3, ν; η12_1=1.255, η12_2=-0.031)
lek_pert = try_lek(E, 1.001E, Giso, ν)

println("\n=== P3 on-circle q2 ===")
run("Kelvin iso", msh_q2, kel)
run("Lekhnitskii iso", msh_q2, lek_iso)
run("Lekhnitskii iso+0.1%", msh_q2, lek_pert)
run("Lekhnitskii MAT1", msh_q2, lek_mat1)

println("\n=== P3 chordal ===")
run("Kelvin iso", msh_lin, kel)
run("Lekhnitskii iso", msh_lin, lek_iso)
run("Lekhnitskii MAT1", msh_lin, lek_mat1)
println("done")
