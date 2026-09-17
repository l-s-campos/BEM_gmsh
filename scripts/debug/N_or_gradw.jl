# Is N wrong or grad w wrong on Table 3 (sine + Navier)?
# julia --project=. scripts/debug/N_or_gradw.jl
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf, Statistics, StaticArrays
using BEM.Plate

a, Lh, E2 = 1.0, 10, 1.0
h = a / Lh
E1 = 25 * E2
pl = [(E1, E2, 0.25, 0.5 * E2, Float64(θ), h / 4) for θ in (0, 90, 90, 0)]
G13, G23 = 0.5 * E2, 0.2 * E2
A, _, _, _, _ = BEM.Plate._laminate_ABD_AT(pl; Ks=5 / 6, G13=G13, G23=G23)
P, q = 50.0, 50.0 * E2 * h^4 / a^4
props = laminate_fsdt_props(pl; Ks=5 / 6, G13=G13, G23=G23, q_c=q, ρ=1.0)
mesh = build_square_fsdt(; a=a, n_el=4, bc="SSSS", props=props, n_internal=9)
shell = LaminatedShell(mesh, A, FlatShell(); mem_bc=:navier_ss)
assemble_laminated_shell!(shell; npg=6, nsub=4, rbf=PHS(2; poly_deg=1),
    rbf_grad=PHS(3; poly_deg=1))
pts = Point2D[BEM.Plate._plate_nodes(mesh); BEM.Plate._plate_internal(mesh)]
n = BEM.Plate._n(shell.plate)
nt = n + BEM.Plate._ni(shell.plate)
qp = [q * sin(π * p[1] / a) * sin(π * p[2] / a) for p in pts]
BEM.set_cache!(shell.plate; fsdt_q=shell.Mw * qp, q=shell.Mw * qp)
solve_laminated_shell!(shell)
res = solve_laminated_shell!(shell; large=true, nsteps=6, λ_max=1.0,
    nonlinear=:newton, e_relax=0.4, maxiters=8, atol=1e-8)

u = vcat(shell.plate.u[1:3nt], shell.u_m)
w = [u[3i] for i in 1:nt]
um = [u[3nt + 2i - 1] for i in 1:nt]
vm = [u[3nt + 2i] for i in 1:nt]
Dx, Dy = shell.Dx, shell.Dy
wx, wy = Dx * w, Dy * w
ux, uy = Dx * um, Dy * um
vxu, vyu = Dx * vm, Dy * vm
W = w[n + 1]
wx_a = [W * (π / a) * cos(π * p[1] / a) * sin(π * p[2] / a) for p in pts]
wy_a = [W * (π / a) * sin(π * p[1] / a) * cos(π * p[2] / a) for p in pts]
nrm = BEM.Plate._plate_normals(shell.plate)

function N_of(εx, εy, γ)
    Nxx = A[1, 1] .* εx .+ A[1, 2] .* εy .+ A[1, 3] .* γ
    Nyy = A[1, 2] .* εx .+ A[2, 2] .* εy .+ A[2, 3] .* γ
    Nxy = A[1, 3] .* εx .+ A[2, 3] .* εy .+ A[3, 3] .* γ
    return Nxx, Nyy, Nxy
end
Nxx_L, Nyy_L, Nxy_L = N_of(ux, vyu, uy .+ vxu)
Nxx_vk, Nyy_vk, Nxy_vk = N_of(0.5 .* wx .^ 2, 0.5 .* wy .^ 2, wx .* wy)
Nxx_a, Nyy_a, Nxy_a = N_of(0.5 .* wx_a .^ 2, 0.5 .* wy_a .^ 2, wx_a .* wy_a)
Nxx, Nyy, Nxy = Nxx_L .+ Nxx_vk, Nyy_L .+ Nyy_vk, Nxy_L .+ Nxy_vk

function edge_Nn(Nxx, Nyy, Nxy, wx, wy)
    Nn = zeros(n)
    Nt = zeros(n)
    vn = zeros(n)   # (N ∇w)·n
    wn = zeros(n)
    @inbounds for i in 1:n
        nx, ny = nrm[i][1], nrm[i][2]
        tx, ty = -ny, nx
        Nn[i] = Nxx[i] * nx * nx + Nyy[i] * ny * ny + 2 * Nxy[i] * nx * ny
        Nt[i] = Nxx[i] * nx * tx + Nxy[i] * (nx * ty + ny * tx) + Nyy[i] * ny * ty
        wn[i] = wx[i] * nx + wy[i] * ny
        vn[i] = (Nxx[i] * wx[i] + Nxy[i] * wy[i]) * nx +
                (Nxy[i] * wx[i] + Nyy[i] * wy[i]) * ny
    end
    return Nn, Nt, vn, wn
end

rel(x, y) = norm(x - y) / (norm(y) + 1e-30)
rms(x) = sqrt(mean(abs2, x))

println("after NL  λ=$(res.λ[end])  w/h=$(abs(W)/h)  paper NL≈0.324  lin≈0.331")
@printf("  max|u|=%.3e  max|v|=%.3e  max|w|=%.3e\n",
    maximum(abs, um), maximum(abs, vm), maximum(abs, w))
@printf("  grad w:  RBF vs sine-fit  all=%.3f  Γ=%.3f  Ω=%.3f\n",
    rel(wx, wx_a), rel(wx[1:n], wx_a[1:n]), rel(wx[n+1:end], wx_a[n+1:end]))
@printf("  max|wx|_RBF=%.3e  max|wx|_sine=%.3e\n", maximum(abs, wx), maximum(abs, wx_a))

println("\nN split (‖·‖_rms)")
@printf("  Nxx:  from u (linear ε)  %.3e\n", rms(Nxx_L))
@printf("        from ½(grad w)² RBF %.3e\n", rms(Nxx_vk))
@printf("        from ½(grad w)² sine %.3e\n", rms(Nxx_a))
@printf("        total N              %.3e\n", rms(Nxx))
@printf("  cancel ratio  ‖N_L+N_vk‖/‖N_vk‖ = %.3f  (0 = u fully cancelled stretching)\n",
    rms(Nxx) / (rms(Nxx_vk) + 1e-30))

Nn, Nt, vn, wn = edge_Nn(Nxx, Nyy, Nxy, wx, wy)
Nn_vk, _, vn_vk, wn_r = edge_Nn(Nxx_vk, Nyy_vk, Nxy_vk, wx, wy)
Nn_a, _, vn_a, wn_a = edge_Nn(Nxx_a, Nyy_a, Nxy_a, wx_a, wy_a)
Nn_L, _, vn_L, _ = edge_Nn(Nxx_L, Nyy_L, Nxy_L, wx, wy)

println("\non Γ  (SSSS1 wants N_nn ≈ 0, then (N∇w)·n = N_nn ∂w/∂n ≈ 0)")
@printf("  rms N_nn  total=%.3e  from u=%.3e  from ½w,RBF=%.3e  from ½w,sine=%.3e\n",
    rms(Nn), rms(Nn_L), rms(Nn_vk), rms(Nn_a))
@printf("  rms N_nt  total=%.3e\n", rms(Nt))
@printf("  rms ∂w/∂n RBF=%.3e  sine=%.3e  rel=%.3f\n",
    rms(wn), rms(wn_a), rel(wn, wn_a))
@printf("  rms (N∇w)·n  total=%.3e  N_L only=%.3e  N_vk RBF=%.3e  N_vk sine=%.3e\n",
    rms(vn), rms(vn_L), rms(vn_vk), rms(vn_a))
@printf("  if N_nn=0 but keep ∂w/∂n: flux would be 0;  Γ density is (N∇w)·n\n")

# mid-side x=0
i0 = argmin(norm(pts[i] - Point2D(0.0, a / 2)) for i in 1:n)
@printf("\nmid-side x=0  (%.3f,%.3f)  n=(%.2f,%.2f)\n",
    pts[i0][1], pts[i0][2], nrm[i0][1], nrm[i0][2])
@printf("  ∂w/∂n  RBF=%.4e  sine=%.4e\n", wn[i0], wn_a[i0])
@printf("  N_nn   total=%.4e  u=%.4e  ½w RBF=%.4e  ½w sine=%.4e\n",
    Nn[i0], Nn_L[i0], Nn_vk[i0], Nn_a[i0])
@printf("  (N∇w)·n total=%.4e  u*wn=%.4e  vk RBF=%.4e  sine=%.4e\n",
    vn[i0], vn_L[i0], vn_vk[i0], vn_a[i0])

# counterfactual: analytic grad w, but BEM u
Nxx_mix, Nyy_mix, Nxy_mix = N_of(ux, vyu, uy .+ vxu)
Nxx_mix = Nxx_mix .+ Nxx_a
Nyy_mix = Nyy_mix .+ Nyy_a
Nxy_mix = Nxy_mix .+ Nxy_a
_, _, vn_mix, _ = edge_Nn(Nxx_mix, Nyy_mix, Nxy_mix, wx_a, wy_a)
@printf("\ncounterfactual rms (N∇w)·n on Γ:\n")
@printf("  live (RBF w, BEM u)     %.3e\n", rms(vn))
@printf("  sine grad w, BEM u      %.3e\n", rms(vn_mix))
@printf("  sine grad w, u=0        %.3e\n", rms(vn_a))
@printf("  live but drop N_L (u=0) %.3e\n", rms(vn_vk))
println("done")
