# Static BEM vs 1-D SS / Airy camber for the uniformly loaded strip.
#   julia --project=. scripts/check_toe_beam_uniform.jl
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf

include(datadir("Laplace", "Laplace_dad.jl"))
include(datadir("Laplace", "potencial_problems.jl"))
include(datadir("elastico", "iso", "analytical_elastodynamics.jl"))
include(datadir("elastico", "iso", "elastodynamics_problems.jl"))

function _boundary_probe(dad, p)
    n = dad.n
    ip = argmin(norm(dad.Nodes[i] - p) for i in 1:n)
    return ip, dad.Nodes[ip]
end

function _v(dad, ip)
    return dad.u[2ip]
end

function main()
    dad, meta = elastodynamics_problem(:toe_beam_uniform; mesh_tag=200)
    H_G_full_direct(dad; npg=12, threaded=false)
    BEM.solve(dad)
    l, c = meta.l, meta.c
    iL, pL = _boundary_probe(dad, Point2D(0.0, c))
    iR, pR = _boundary_probe(dad, Point2D(2l, c))
    it, pt = _boundary_probe(dad, Point2D(l, 2c))
    ib, pb = _boundary_probe(dad, Point2D(l, 0.0))
    vL, vR, vt, vb = _v(dad, iL), _v(dad, iR), _v(dad, it), _v(dad, ib)
    @printf("nt=%d n=%d  pins iL=%d (%.3f,%.3f)  iR=%d (%.3f,%.3f)\n",
        dad.nt, dad.n, iL, pL[1], pL[2], iR, pR[1], pR[2])
    @printf("  pin v_L=%.3e  pin v_R=%.3e  (expect 0)\n", vL, vR)
    @printf("  top mid v=%.6f  bot mid v=%.6f\n", vt, vb)
    @printf("  δ_EB   =%.6f\n", meta.δ_eb)
    @printf("  δ_Airy =%.6f  top/δ_Airy=%.4f\n", meta.δ_airy, vt / meta.δ_airy)
    @printf("  δ_Timo =%.6f  top/δ_Timo=%.4f  (1-D SS target)\n",
        meta.δ, vt / meta.δ)
    return nothing
end

main()
