# Impose FEniCS in-plane (u,v) as BEM membrane Dirichlet BCs, then NL solve.
# julia --project=. scripts/debug/bem_fenics_u_bc.jl
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf, Statistics, StaticArrays, DelimitedFiles
using BEM.Plate

const OUT = joinpath(@__DIR__, "fenics_tab3")
Lh, P, a, E2 = 10.0, 50.0, 1.0, 1.0
h = a / Lh
NEL, NINT, NPG, NSUB = 4, 9, 6, 4
pl = [(25 * E2, E2, 0.25, 0.5 * E2, Float64(θ), h / 4) for θ in (0, 90, 90, 0)]
A, _, _, _, _ = BEM.Plate._laminate_ABD_AT(pl; Ks=5 / 6, G13=0.5 * E2, G23=0.2 * E2)
q = P * E2 * h^4 / a^4
props = laminate_fsdt_props(pl; Ks=5 / 6, G13=0.5 * E2, G23=0.2 * E2, q_c=q, ρ=1.0)
mesh = build_square_fsdt(; a=a, n_el=NEL, bc="SSSS", props=props, n_internal=NINT)
shell = LaminatedShell(mesh, A, FlatShell(); mem_bc=:navier_ss)
BCm0 = copy(shell.BCm)
BVm0 = copy(shell.BVm)
assemble_laminated_shell!(shell; npg=NPG, nsub=NSUB, rbf=PHS(2; poly_deg=1),
    rbf_grad=PHS(3; poly_deg=1))
n = BEM.Plate._n(shell.plate)
nt = n + BEM.Plate._ni(shell.plate)
pts = Point2D[BEM.Plate._plate_nodes(shell.plate); BEM.Plate._plate_internal(shell.plate)]

femf = joinpath(OUT, "fem_Lh$(Int(Lh))_P$(Int(P)).csv")
raw = readdlm(femf, ','; comments=true, header=true)
data, hdr = raw
hdr = string.(vec(hdr))
col = Dict(hdr[j] => j for j in eachindex(hdr))
isn = [string(data[i, col["tag"]]) == "nl" for i in 1:size(data, 1)]
fem(name) = Float64.(data[isn, col[name]])
u_f, v_f, w_f = fem("u"), fem("v"), fem("w")
Nxx_f, Nyy_f = fem("Nxx"), fem("Nyy")
@assert length(u_f) == nt

qp = [q * sin(π * p[1] / a) * sin(π * p[2] / a) for p in pts]
BEM.set_cache!(shell.plate; fsdt_q=shell.Mw * qp, q=shell.Mw * qp)

function apply_u_bc!(mode::Symbol)
    shell.BCm .= BCm0
    shell.BVm .= BVm0
    mode === :navier && return shell
    mode === :pullin || error(mode)
    BEM.Plate.set_membrane_dirichlet!(shell, u_f, v_f; only_free=true)
    return shell
end

function run_mode(mode)
    apply_u_bc!(mode)
    solve_laminated_shell!(shell)
    wL = abs(fsdt_w_int(shell.plate, 1))
    res = solve_laminated_shell!(shell; large=true, nsteps=8, λ_max=1.0,
        nonlinear=:newton, e_relax=0.4, maxiters=8, atol=1e-8)
    wN = abs(res.w_center[end])
    um = [shell.u_m[2i - 1] for i in 1:nt]
    vm = [shell.u_m[2i] for i in 1:nt]
    w = [shell.plate.u[3i] for i in 1:nt]
    Dx, Dy = shell.Dx, shell.Dy
    wx, wy = Dx * w, Dy * w
    ux, uy = Dx * um, Dy * um
    vx, vy = Dx * vm, Dy * vm
    Nxx = A[1, 1] .* (ux .+ 0.5 .* wx .^ 2) .+ A[1, 2] .* (vy .+ 0.5 .* wy .^ 2) .+
          A[1, 3] .* (uy .+ vx .+ wx .* wy)
    i0 = argmin(norm(p - Point2D(0.0, 0.5)) for p in pts[1:n])
    @printf("%-8s  lin=%.4f  NL=%.4f  λ=%.2f  u(x=0)=%+.3e (FEM %+.3e)  Nxx_c=%+.3e (FEM %+.3e)\n",
        mode, wL / h, wN / h, res.λ[end], um[i0], u_f[i0],
        Nxx[n + 1], Nxx_f[n + 1])
    @printf("         rel u_Γ=%.3f  rel v_Γ=%.3f  rel Nxx=%.3f  rel w=%.3f\n",
        norm(um[1:n] .- u_f[1:n]) / (norm(u_f[1:n]) + 1e-30),
        norm(vm[1:n] .- v_f[1:n]) / (norm(v_f[1:n]) + 1e-30),
        norm(Nxx .- Nxx_f) / (norm(Nxx_f) + 1e-30),
        norm(w .- w_f) / (norm(w_f) + 1e-30))
    return wN / h
end

@printf("FEM NL w/h=%.4f  u(x=0 mid)=%+.3e\n", maximum(abs, w_f) / h, u_f[argmin(norm(p - Point2D(0.0, 0.5)) for p in pts[1:n])])
run_mode(:navier)
run_mode(:pullin)
println("done")
