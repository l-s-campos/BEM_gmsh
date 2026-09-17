# Is the laminated-shell membrane (Lekhnitskii) elastic problem right?
# 1) uniaxial tension patch  2) Table 3 NL: t_m vs N_L·n vs (N_L+N_vk)·n
# julia --project=. scripts/debug/membrane_elastic.jl
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf, Statistics, StaticArrays
using BEM.Plate

a = 1.0
E, ν, hth = 1.0, 0.25, 0.1
Aiso = E * hth / (1 - ν^2) * @SMatrix [1.0 ν 0.0; ν 1.0 0.0; 0.0 0.0 (1 - ν) / 2]
props = FSDTProps(; E=E, ν=ν, h=hth, q_c=0.0, ρ=1.0)
mesh = build_square_fsdt(; a=a, n_el=4, bc="SSSS", props=props, n_internal=9)
shell = LaminatedShell(mesh, Aiso, FlatShell(); mem_bc=:free)
assemble_laminated_shell!(shell; npg=6, nsub=4, rbf=PHS(2; poly_deg=1),
    rbf_grad=PHS(3; poly_deg=1))
n = BEM.Plate._n(shell.plate)
pts = Point2D[BEM.Plate._plate_nodes(shell.plate); BEM.Plate._plate_internal(shell.plate)]
σ = 1.0
# pin node 1 (u,v) and v of a y-shifted node; traction tx=σ on x=a
fill!(shell.BCm, 1)
fill!(shell.BVm, 0.0)
shell.BCm[1] = 0
shell.BCm[2] = 0
jpin = findfirst(i -> abs(pts[i][1]) < 1e-12 && abs(pts[i][2] - a) < 1e-12, 1:n)
jpin === nothing && (jpin = 2)
shell.BCm[2jpin] = 0
for i in 1:n
    if abs(pts[i][1] - a) < 1e-9
        shell.BVm[2i - 1] = σ   # tx
    end
end
solve_laminated_shell!(shell)
um = [shell.u_m[2i - 1] for i in 1:n]
ux_ana = [σ / E * (pts[i][1] - pts[1][1]) for i in 1:n]  # plane stress, ν terms if uy free
# better: εx = σ/E, εy = -ν σ/E, origin at pinned node
x0, y0 = pts[1][1], pts[1][2]
ux_ana = [σ / E * (pts[i][1] - x0) for i in 1:n]
uy_ana = [-ν * σ / E * (pts[i][2] - y0) for i in 1:n]
vm = [shell.u_m[2i] for i in 1:n]
@printf("=== 1. membrane uniaxial σx=1, plane stress ===\n")
@printf("  rel ux = %.3e  rel uy = %.3e  max|ux|=%.3e  ana=%.3e\n",
    norm(um - ux_ana) / (norm(ux_ana) + 1e-30),
    norm(vm - uy_ana) / (norm(uy_ana) + 1e-30),
    maximum(abs, um), maximum(abs, ux_ana))

# --- Table 3 NL traction vs N ---
println("\n=== 2. Table 3 sine+Navier: elastic t vs total N·n ===")
Lh, E2 = 10, 1.0
hh = a / Lh
E1 = 25 * E2
pl = [(E1, E2, 0.25, 0.5 * E2, Float64(θ), hh / 4) for θ in (0, 90, 90, 0)]
G13, G23 = 0.5 * E2, 0.2 * E2
A, _, _, _, _ = BEM.Plate._laminate_ABD_AT(pl; Ks=5 / 6, G13=G13, G23=G23)
P = 50.0
q = P * E2 * hh^4 / a^4
pr = laminate_fsdt_props(pl; Ks=5 / 6, G13=G13, G23=G23, q_c=q, ρ=1.0)
m3 = build_square_fsdt(; a=a, n_el=4, bc="SSSS", props=pr, n_internal=9)
sh = LaminatedShell(m3, A, FlatShell(); mem_bc=:navier_ss)
assemble_laminated_shell!(sh; npg=6, nsub=4, rbf=PHS(2; poly_deg=1),
    rbf_grad=PHS(3; poly_deg=1))
pts3 = Point2D[BEM.Plate._plate_nodes(m3); BEM.Plate._plate_internal(m3)]
n3 = BEM.Plate._n(sh.plate)
nt3 = n3 + BEM.Plate._ni(sh.plate)
qp = [q * sin(π * p[1] / a) * sin(π * p[2] / a) for p in pts3]
BEM.set_cache!(sh.plate; fsdt_q=sh.Mw * qp, q=sh.Mw * qp)
solve_laminated_shell!(sh)
solve_laminated_shell!(sh; large=true, nsteps=6, λ_max=1.0, nonlinear=:newton,
    e_relax=0.4, maxiters=8, atol=1e-8)
u = vcat(sh.plate.u[1:3nt3], sh.u_m)
w = [u[3i] for i in 1:nt3]
umv = [u[3nt3 + 2i - 1] for i in 1:nt3]
vmv = [u[3nt3 + 2i] for i in 1:nt3]
Dx, Dy = sh.Dx, sh.Dy
wx, wy = Dx * w, Dy * w
ux, uy = Dx * umv, Dy * umv
vx, vy = Dx * vmv, Dy * vmv
function Nof(εx, εy, γ)
    return (A[1, 1] .* εx .+ A[1, 2] .* εy .+ A[1, 3] .* γ,
        A[1, 2] .* εx .+ A[2, 2] .* εy .+ A[2, 3] .* γ,
        A[1, 3] .* εx .+ A[2, 3] .* εy .+ A[3, 3] .* γ)
end
NxxL, NyyL, NxyL = Nof(ux, vy, uy .+ vx)
NxxV, NyyV, NxyV = Nof(0.5 .* wx .^ 2, 0.5 .* wy .^ 2, wx .* wy)
nrm = BEM.Plate._plate_normals(sh.plate)
txL = zeros(n3)
tyL = zeros(n3)
txV = zeros(n3)
tyV = zeros(n3)
@inbounds for i in 1:n3
    nx, ny = nrm[i][1], nrm[i][2]
    txL[i] = NxxL[i] * nx + NxyL[i] * ny
    tyL[i] = NxyL[i] * nx + NyyL[i] * ny
    txV[i] = NxxV[i] * nx + NxyV[i] * ny
    tyV[i] = NxyV[i] * nx + NyyV[i] * ny
end
tmx = [sh.t_m[2i - 1] for i in 1:n3]
tmy = [sh.t_m[2i] for i in 1:n3]
free_u = [sh.BCm[2i - 1] == 1 for i in 1:n3]
free_v = [sh.BCm[2i] == 1 for i in 1:n3]
rms(z) = sqrt(mean(abs2, z))
@printf("  rms t_m (BEM)     tx=%.3e ty=%.3e\n", rms(tmx), rms(tmy))
@printf("  rms N_L · n       tx=%.3e ty=%.3e\n", rms(txL), rms(tyL))
@printf("  rms N_vk · n      tx=%.3e ty=%.3e\n", rms(txV), rms(tyV))
@printf("  rms (N_L+N_vk)·n  tx=%.3e ty=%.3e\n", rms(txL .+ txV), rms(tyL .+ tyV))
@printf("  on free-u nodes (should be N_nn=0 if total traction BC):\n")
@printf("    rms t_m x=%.3e  N_L nx=%.3e  N_vk nx=%.3e  total=%.3e  n=%d\n",
    rms(tmx[free_u]), rms(txL[free_u]), rms(txV[free_u]),
    rms((txL .+ txV)[free_u]), count(free_u))
@printf("  on free-v nodes:\n")
@printf("    rms t_m y=%.3e  N_L ny=%.3e  N_vk ny=%.3e  total=%.3e  n=%d\n",
    rms(tmy[free_v]), rms(tyL[free_v]), rms(tyV[free_v]),
    rms((tyL .+ tyV)[free_v]), count(free_v))
@printf("  BEM t_m vs N_L·n  rel tx=%.3e ty=%.3e\n",
    norm(tmx - txL) / (norm(txL) + 1e-30),
    norm(tmy - tyL) / (norm(tyL) + 1e-30))
println("done")
