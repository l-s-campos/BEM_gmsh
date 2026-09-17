# Static BEM vs Timoshenko Theory of Elasticity cantilever (parabolic end shear).
#   julia --project=. scripts/check_toe_cantilever.jl
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf, Statistics

include(datadir("Laplace", "Laplace_dad.jl"))
include(datadir("Laplace", "potencial_problems.jl"))
include(datadir("elastico", "iso", "analytical_elastodynamics.jl"))
include(datadir("elastico", "iso", "elastodynamics_problems.jl"))

function rmse_u(dad, meta)
    u = dad.u
    n = dad.n
    e2 = 0.0
    nrm = 0.0
    @inbounds for i in 1:n
        ua, va = meta.ana(dad.Nodes[i])
        un, vn = u[2i-1], u[2i]
        e2 += (un - ua)^2 + (vn - va)^2
        nrm += ua^2 + va^2
    end
    return sqrt(e2 / n), sqrt(e2) / (sqrt(nrm) + eps())
end

function main()
    dad, meta = elastodynamics_problem(:toe_cantilever; mesh_tag=200)
    H_G_full_direct(dad; npg=12, threaded=false)
    BEM.solve(dad)
    abserr, rel = rmse_u(dad, meta)
    ip, p = _elasto_probe_id(dad, meta.probe)
    ua, va = meta.ana(p)
    un, vn = dad.u[2ip-1], dad.u[2ip]
    @printf("nt=%d n=%d ni=%d  tip ana u2=%.6f  BEM u2=%.6f  rel_u2=%.3e\n",
        dad.nt, dad.n, dad.ni, va, vn, abs(vn - va) / (abs(va) + eps()))
    @printf("RMSE |u-u_ana|=%.3e  rel_L2=%.3e  δ_tip=%.6f\n", abserr, rel, meta.δ)

    # same mesh, uniform end ty = P/D (wrong BC)
    dadU, metaU = elastodynamics_problem(:toe_cantilever; mesh_tag=200)
    _apply_right_ty!(dadU, meta.P)   # overwrite right face with uniform
    H_G_full_direct(dadU; npg=12, threaded=false)
    BEM.solve(dadU)
    _, relU = rmse_u(dadU, metaU)
    @printf("uniform end shear  rel_L2=%.3e  (parabolic was %.3e)\n", relU, rel)
    return nothing
end

main()
