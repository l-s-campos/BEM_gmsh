using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf, Statistics
using Plots
gr()
default(fontfamily="Computer Modern", linewidth=1.6, framestyle=:box, grid=false, dpi=160, size=(640,380), legendfontsize=9)

include(datadir("Laplace", "Laplace_dad.jl"))
include(datadir("Laplace", "potencial_problems.jl"))
include(datadir("Laplace", "wave_propagation.jl"))
include(datadir("elastico", "iso", "bar_sudden.jl"))

function probe_ux(U, dad, probe)
    pts = vcat(dad.Nodes, dad.internalNodes)
    ip = argmin(norm(p - probe) for p in pts)
    return U[2*(ip-1)+1, :]
end
rel(a,b) = norm(a.-b) / (norm(b)+eps())

tf = 16.0
nT = 101
Δt = tf / (nT-1)
ndiv, n_int = 8, 4
ν = 0.0

dad0, meta = elasticity_bar_sudden(; ndiv=ndiv, n_int=n_int, ν=ν)
H_G_full_direct(dad0; npg=8, threaded=false)

# DIBEM
dadD = deepcopy(dad0)
DIBEM(dadD; method=:dense, rbf=PHS(1; poly_deg=-1))
sysD = build_modal_system(dadD)
nnegD = count(<( -1e-8), real.(eigvals(sysD.M \ sysD.K)))
UD, t, bD = solve_mmm!(dadD, Δt, tf)
uxD = probe_ux(UD, dadD, meta.probe)

# DRM
dadR = deepcopy(dad0)
drm = build_drm_matrices(dadR; npg=8)
sysR = build_modal_system(dadR)
nnegR = count(<( -1e-8), real.(eigvals(sysR.M \ sysR.K)))
UR, _, bR = solve_mmm!(dadR, Δt, tf)
uxR = probe_ux(UR, dadR, meta.probe)

ua = [meta.ana.u(meta.probe; t=ti) for ti in t]

@printf("DIBEM: nneg=%d  nmodes=%d  ω1=%.4f  rel=%.3e  max=%.3f\n", nnegD, length(bD.ω), bD.ω[1], rel(uxD,ua), maximum(abs,uxD))
@printf("DRM  : nneg=%d  nmodes=%d  ω1=%.4f  rel=%.3e  max=%.3f\n", nnegR, length(bR.ω), bR.ω[1], rel(uxR,ua), maximum(abs,uxR))

plt = plot(t, ua; color=:black, ls=:dash, label="1D series")
plot!(plt, t, uxD; color=:darkorange, label="MMM + DIBEM")
plot!(plt, t, uxR; color=:steelblue, label="MMM + DRM")
plot!(plt; xlabel="t", ylabel="u_x(L, L/2)", title="elast bar sudden  tf=16  nT=101  ndiv=8  ν=0")
out = abspath("elast_bar_sudden_tf16_drm.png")
savefig(plt, out)
savefig(plt, replace(out, ".png"=>".pdf"))
println("wrote ", out)
