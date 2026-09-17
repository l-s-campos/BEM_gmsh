# Houbolt: MATLAB DRM twin vs Laplace-style DIBEM (boundary + cell centroids).
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf
using BEM.Plate

Ed, hd, ad, qd, ρd = 200e3, 0.1, 2.0, 1e3, 0.7853
pd = FSDTProps(; E=Ed, ν=0.3, h=hd, q_c=qd, ρ=ρd)

println("="^64)
println(" MATLAB example01  Houbolt  DRM vs DIBEM (centroids)")
println("="^64)

function run_h(method)
    m = build_square_fsdt(; a=ad, n_el=4, bc="SSSS", props=pd, n_internal=9)
    assemble_fsdt!(m; npg=8, nsub=6)
    dibem_fsdt!(m; method=method)
    solve_fsdt!(m)
    wstat = abs(fsdt_w_int(m, 1))
    res = solve_fsdt_houbolt!(m; dt=5e-3, tmax=0.15)
    imax = argmax(abs.(res.w_center))
    peak = abs(res.w_center[imax])
    @printf("  %-6s  n_int=%d  first=%s\n", method, length(m.internal), string(m.internal[1]))
    @printf("  %-6s  static=%.6e  peak=%.6e t=%.3f  2×static=%.6e  ratio=%.3f\n",
        method, wstat, peak, res.t[imax], 2 * wstat, peak / (2 * wstat + eps()))
    return peak
end
p_drm = run_h(:drm)
p_dib = run_h(:dibem)
@printf("  peak DIBEM/DRM = %.3f\n", p_dib / (p_drm + eps()))
println("  MATLAB step-load peak ~ 2×static.")
