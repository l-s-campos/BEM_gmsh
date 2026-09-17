# Inject FEniCS N (and optionally ∇w) into the BEM plate geometric residual.
# julia --project=. scripts/debug/bem_with_fenics_N.jl
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf, Statistics, StaticArrays, DelimitedFiles
using BEM.Plate

const OUT = joinpath(@__DIR__, "fenics_tab3")
const FENICS = "/home/lsc/Projects/fenics experimentos/von_karman_plate"
Lh, P, a, E2 = 10.0, 50.0, 1.0, 1.0
h = a / Lh
NEL, NINT, NPG, NSUB = 4, 9, 6, 4

pl = [(25 * E2, E2, 0.25, 0.5 * E2, Float64(θ), h / 4) for θ in (0, 90, 90, 0)]
A, _, _, _, _ = BEM.Plate._laminate_ABD_AT(pl; Ks=5 / 6, G13=0.5 * E2, G23=0.2 * E2)
q = P * E2 * h^4 / a^4
props = laminate_fsdt_props(pl; Ks=5 / 6, G13=0.5 * E2, G23=0.2 * E2, q_c=q, ρ=1.0)
mesh = build_square_fsdt(; a=a, n_el=NEL, bc="SSSS", props=props, n_internal=NINT)
shell = LaminatedShell(mesh, A, FlatShell(); mem_bc=:navier_ss)
assemble_laminated_shell!(shell; npg=NPG, nsub=NSUB, rbf=PHS(2; poly_deg=1),
    rbf_grad=PHS(3; poly_deg=1))
n = BEM.Plate._n(shell.plate)
nt = n + BEM.Plate._ni(shell.plate)
pts = Point2D[BEM.Plate._plate_nodes(shell.plate); BEM.Plate._plate_internal(shell.plate)]
ptsf = joinpath(OUT, "pts.csv")
mkpath(OUT)
open(ptsf, "w") do io
    println(io, "x,y")
    for p in pts
        println(io, "$(p[1]),$(p[2])")
    end
end
femf = joinpath(OUT, "fem_Lh$(Int(Lh))_P$(Int(P)).csv")
if !isfile(femf)
    py = joinpath(FENICS, "dump_tran2015_fields.py")
    run(`micromamba run -n fenicsx-env python $py --pts $ptsf --out $femf --Lh $Lh --P $P`)
end

raw = readdlm(femf, ','; comments=true, header=true)
data, hdr = raw
hdr = string.(vec(hdr))
col = Dict(hdr[j] => j for j in eachindex(hdr))
isn = [string(data[i, col["tag"]]) == "nl" for i in 1:size(data, 1)]
fem(name) = Float64.(data[isn, col[name]])
Nxx_f, Nyy_f, Nxy_f = fem("Nxx"), fem("Nyy"), fem("Nxy")
wx_f, wy_f = fem("wx"), fem("wy")
w_f = fem("w")
@assert length(Nxx_f) == nt

qp = [q * sin(π * p[1] / a) * sin(π * p[2] / a) for p in pts]
BEM.set_cache!(shell.plate; fsdt_q=shell.Mw * qp, q=shell.Mw * qp)
solve_laminated_shell!(shell)
w_lin = abs(fsdt_w_int(shell.plate, 1))

sys = BEM.Plate._laminated_shell_system(shell)
tknown = BEM.Plate._laminated_apply_bc!(sys, shell.BVm)
H, q0, Gt = sys.H, sys.q, sys.G * tknown
ndp = sys.ndp

function geo_plate(Nxx, Nyy, Nxy, wx, wy)
    vx = Nxx .* wx .+ Nxy .* wy
    vy = Nxy .* wx .+ Nyy .* wy
    return BEM.Plate.ibp_div_Uv(shell.Mx, shell.My, BEM.Plate._plate_G(shell.plate),
        BEM.Plate._plate_normals(shell.plate), vx, vy, 3)
end

function solve_with_geo(fplate)
    f = zeros(sys.ndof)
    f[1:ndp] .= fplate
    x = H \ (q0 .+ Gt .+ f)
    BEM.Plate._shell_write_sol!(shell, sys, x, tknown)
    return abs(fsdt_w_int(shell.plate, 1))
end

function picard_Nfreeze(Nxx, Nyy, Nxy, niter=12)
    w = [shell.plate.u[3i] for i in 1:nt]
    wc = abs(w[n + 1])
    for k in 1:niter
        wx, wy = shell.Dx * w, shell.Dy * w
        wc = solve_with_geo(geo_plate(Nxx, Nyy, Nxy, wx, wy))
        w = [shell.plate.u[3i] for i in 1:nt]
    end
    return wc
end

# restore linear
xlin = H \ (q0 .+ Gt)
BEM.Plate._shell_write_sol!(shell, sys, xlin, tknown)
w0 = abs(fsdt_w_int(shell.plate, 1))

wA = picard_Nfreeze(Nxx_f, Nyy_f, Nxy_f)
BEM.Plate._shell_write_sol!(shell, sys, xlin, tknown)
wB = solve_with_geo(geo_plate(Nxx_f, Nyy_f, Nxy_f, wx_f, wy_f))
BEM.Plate._shell_write_sol!(shell, sys, xlin, tknown)

# BEM N from linear w (u=0) as control one-shot
wlin_vec = [xlin[3i] for i in 1:nt]  # mixed x may not be u on kin dofs
u_lin = BEM.Plate._shell_disp_from_mixed(sys, xlin)
wlin_vec = [u_lin[3i] for i in 1:nt]
wx_b, wy_b = shell.Dx * wlin_vec, shell.Dy * wlin_vec
Nxx_b, Nyy_b, Nxy_b = begin
    εx, εy, γ = 0.5 .* wx_b .^ 2, 0.5 .* wy_b .^ 2, wx_b .* wy_b
    (A[1, 1] .* εx .+ A[1, 2] .* εy .+ A[1, 3] .* γ,
        A[1, 2] .* εx .+ A[2, 2] .* εy .+ A[2, 3] .* γ,
        A[1, 3] .* εx .+ A[2, 3] .* εy .+ A[3, 3] .* γ)
end
wC = solve_with_geo(geo_plate(Nxx_b, Nyy_b, Nxy_b, wx_b, wy_b))

fp_f = geo_plate(Nxx_f, Nyy_f, Nxy_f, wx_f, wy_f)
fp_b = geo_plate(Nxx_b, Nyy_b, Nxy_b, wx_b, wy_b)
ic = 3(n + 1)
@printf("L/h=%.0f P̄=%.0f  FEM NL w/h=%.4f  FEM lin w/h=%.4f\n",
    Lh, P, maximum(abs, w_f) / h, w_lin / h)
@printf("  BEM linear                    w/h=%.4f\n", w0 / h)
@printf("  Picard, N=FEM, ∇w=BEM RBF     w/h=%.4f\n", wA / h)
@printf("  one-shot N=FEM, ∇w=FEM        w/h=%.4f\n", wB / h)
@printf("  one-shot N=BEM ½(∇w)², ∇w=RBF w/h=%.4f\n", wC / h)
@printf("  geo RHS centre (w-eq)  FEM N,∇w=%.3e  BEM N,∇w=%.3e  ratio=%.2f\n",
    fp_f[ic], fp_b[ic], fp_b[ic] / (fp_f[ic] + 1e-30))
@printf("  ‖geo‖ FEM fields=%.3e  BEM fields=%.3e  ratio=%.2f\n",
    norm(fp_f), norm(fp_b), norm(fp_b) / (norm(fp_f) + 1e-30))
@printf("  rms Nxx FEM=%.3e  BEM(lin w)=%.3e\n",
    sqrt(mean(abs2, Nxx_f)), sqrt(mean(abs2, Nxx_b)))
println("done")
