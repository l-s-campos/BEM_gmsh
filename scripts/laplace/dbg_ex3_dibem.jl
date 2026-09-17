# Diagnose Example 3 iso-DIBEM vs anisotropic FS (k1=5, k2=0.5).
using BEM
using LinearAlgebra
using StaticArrays
using Printf
include(joinpath(@__DIR__, "..", "..", "data", "Laplace", "Laplace_dad.jl"))
include(joinpath(@__DIR__, "orthotropic_dibem_examples.jl"))

K = @SMatrix [5.0 0.0; 0.0 0.5]
k1, k2 = 5.0, 0.5
FENICS_MID = 2.769439  # h=0.02

function tmid(dad)
    ys, Ts, _ = right_edge(dad)
    isempty(Ts) && return NaN
    return Ts[argmin(abs.(ys .- 0.5))]
end

function rel_profile(a, b)
    ya, Ta, _ = right_edge(a)
    yb, Tb, _ = right_edge(b)
    # interpolate b onto a
    Tbi = similar(Ta)
    for (k, z) in enumerate(ya)
        if z <= yb[1]
            Tbi[k] = Tb[1]
        elseif z >= yb[end]
            Tbi[k] = Tb[end]
        else
            j = searchsortedlast(yb, z)
            t = (z - yb[j]) / (yb[j+1] - yb[j] + eps())
            Tbi[k] = (1 - t) * Tb[j] + t * Tb[j+1]
        end
    end
    return norm(Ta .- Tbi) / (norm(Tbi) + eps())
end

function run_one(label; ordem=1, tipo=1, rbf=PHS(3; poly_deg=1), strat=:hess,
        mesh=:placa, lc=0.05, nlocal=21, ref=nothing)
    slug = replace(label, r"[^A-Za-z0-9]+" => "_")
    if mesh === :placa
        path = placa_furo_orto(; lc=lc, nome="dbg_e3_$(slug)", show=false,
            ordem=ordem, qright=-k1)
        dad = format2d(path, Laplace(1.0); pontointerno=true, tipo=tipo)
    else
        path = mesh_square_hole(; nome="dbg_e3c_$(slug)", lc=lc, ordem=ordem)
        dad = format2d(path, Laplace(1.0); pontointerno=true, tipo=tipo)
        apply_ex3!(dad; k1=k1, k2=k2)
    end
    assemble!(dad; npg=12, threaded=true)
    DIBEM(dad; rbf=rbf, threaded=true)
    t0 = time()
    if strat === :hess
        solve_anisotropic_dibem!(dad, K; rbf=rbf, npg=12, nlocal=nlocal)
    else
        solve_anisotropic_ibp!(dad, K; rbf=rbf, npg=12, nlocal=nlocal)
    end
    dt = time() - t0
    tm = tmid(dad)
    vs = ref === nothing ? NaN : rel_profile(dad, ref)
    @printf("%-30s n=%4d ni=%4d  Tmid=%.4f  ΔFEn=%+.4f  vsFS=%.3e  t=%.2fs\n",
        label, dad.n, dad.ni, tm, tm - FENICS_MID, vs, dt)
    flush(stdout)
    return dad
end

println("FEniCS T(1,0.5) = ", FENICS_MID)

refs = Dict{String,Any}()
for (ord, tip, lab) in ((1, 1, "aniso FS linear"), (2, 2, "aniso FS quad"))
    path = placa_furo_orto(; lc=0.05, nome="dbg_e3_fs_$(tip)", show=false,
        ordem=ord, qright=-k1)
    dad = format2d(path, AnisotropicLaplace(K); pontointerno=false, tipo=tip)
    assemble!(dad; npg=12, threaded=true)
    solve(dad)
    tm = tmid(dad)
    @printf("%-30s n=%4d ni=%4d  Tmid=%.4f  ΔFEn=%+.4f\n",
        lab, dad.n, dad.ni, tm, tm - FENICS_MID)
    refs[lab] = dad
end
fs1 = refs["aniso FS linear"]

println("\n--- DIBEM variants ---")
r1 = PHS(3; poly_deg=1)
r2 = PHS(3; poly_deg=2)
run_one("S1 lin p1 nloc21 placa"; rbf=r1, strat=:hess, ref=fs1)
run_one("S1 lin p2 nloc21 placa"; rbf=r2, strat=:hess, ref=fs1)
run_one("S3 lin p2 nloc21 placa"; rbf=r2, strat=:ibp, ref=fs1)
run_one("S3 lin p1 nloc21 placa"; rbf=r1, strat=:ibp, ref=fs1)
run_one("S1 quad p2 nloc21 placa"; ordem=2, tipo=2, rbf=r2, strat=:hess,
    ref=refs["aniso FS quad"])
run_one("S3 quad p2 nloc21 placa"; ordem=2, tipo=2, rbf=r2, strat=:ibp,
    ref=refs["aniso FS quad"])
run_one("S1 lin p1 custom apply"; rbf=r1, strat=:hess, mesh=:custom, ref=fs1)
run_one("S1 lin p2 custom apply"; rbf=r2, strat=:hess, mesh=:custom, ref=fs1)
run_one("S3 lin p2 custom apply"; rbf=r2, strat=:ibp, mesh=:custom, ref=fs1)
run_one("S1 lin p2 nloc41 placa"; rbf=r2, strat=:hess, nlocal=41, ref=fs1)
run_one("S3 lin p2 nloc41 placa"; rbf=r2, strat=:ibp, nlocal=41, ref=fs1)

