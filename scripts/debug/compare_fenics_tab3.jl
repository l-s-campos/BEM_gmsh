# Compare BEM Table-3 fields to FEniCS dump at BEM collocation points.
# julia --project=. scripts/debug/compare_fenics_tab3.jl [Lh] [P]
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf, Statistics, StaticArrays, DelimitedFiles
using BEM.Plate

const FENICS = "/home/lsc/Projects/fenics experimentos/von_karman_plate"
const OUT = joinpath(@__DIR__, "fenics_tab3")
mkpath(OUT)

Lh = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 10.0
P = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 50.0
a, E2 = 1.0, 1.0
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
open(ptsf, "w") do io
    println(io, "x,y,kind")
    for (i, p) in enumerate(pts)
        println(io, "$(p[1]),$(p[2]),$(i <= n ? "G" : "I")")
    end
end
femf = joinpath(OUT, "fem_Lh$(Int(Lh))_P$(Int(P)).csv")
py = joinpath(FENICS, "dump_tran2015_fields.py")
cmd = `micromamba run -n fenicsx-env python $py --pts $ptsf --out $femf --Lh $Lh --P $P`
println("FEM: ", cmd)
run(cmd)

qp = [q * sin(π * p[1] / a) * sin(π * p[2] / a) for p in pts]
BEM.set_cache!(shell.plate; fsdt_q=shell.Mw * qp, q=shell.Mw * qp)
solve_laminated_shell!(shell)
wL = abs(fsdt_w_int(shell.plate, 1))
res = solve_laminated_shell!(shell; large=true, nsteps=8, λ_max=1.0,
    nonlinear=:newton, e_relax=0.4, maxiters=8, atol=1e-8)
wN = abs(res.w_center[end])
u = vcat(shell.plate.u[1:3nt], shell.u_m)
w = [u[3i] for i in 1:nt]
um = [u[3nt + 2i - 1] for i in 1:nt]
vm = [u[3nt + 2i] for i in 1:nt]
βx = [u[3i - 2] for i in 1:nt]
βy = [u[3i - 1] for i in 1:nt]
Dx, Dy = shell.Dx, shell.Dy
wx, wy = Dx * w, Dy * w
ux, uy = Dx * um, Dy * um
vxu, vyu = Dx * vm, Dy * vm
Nof(εx, εy, γ) = (
    A[1, 1] .* εx .+ A[1, 2] .* εy .+ A[1, 3] .* γ,
    A[1, 2] .* εx .+ A[2, 2] .* εy .+ A[2, 3] .* γ,
    A[1, 3] .* εx .+ A[2, 3] .* εy .+ A[3, 3] .* γ,
)
NxxL, NyyL, NxyL = Nof(ux, vyu, uy .+ vxu)
NxxV, NyyV, NxyV = Nof(0.5 .* wx .^ 2, 0.5 .* wy .^ 2, wx .* wy)
Nxx, Nyy, Nxy = NxxL .+ NxxV, NyyL .+ NyyV, NxyL .+ NxyV
Nxxp, Nyyp, Nxyp = copy(Nxx), copy(Nyy), copy(Nxy)
BEM.Plate._project_Nnn_free!(shell, Nxxp, Nyyp, Nxyp)

raw = readdlm(femf, ','; comments=true, header=true)
data, hdr = raw
hdr = string.(vec(hdr))
col = Dict(hdr[j] => j for j in eachindex(hdr))
isl = [string(data[i, col["tag"]]) == "lin" for i in 1:size(data, 1)]
isn = .!isl
fem(tagmask, name) = Float64.(data[tagmask, col[name]])

function rel(b, f)
    return norm(b - f) / (norm(f) + 1e-30)
end
function blk(name, b, f; mask=:)
    bb, ff = b[mask], f[mask]
    @printf("  %-8s  rel=%.3f  maxB=%.3e  maxF=%.3e  max|Δ|=%.3e\n",
        name, rel(bb, ff), maximum(abs, bb), maximum(abs, ff), maximum(abs, bb .- ff))
end

G = 1:n
I = (n + 1):nt
onx = [abs(pts[i][1]) < 1e-9 || abs(pts[i][1] - a) < 1e-9 for i in 1:n]
ony = [abs(pts[i][2]) < 1e-9 || abs(pts[i][2] - a) < 1e-9 for i in 1:n]

fw = fem(isn, "w")
fu = fem(isn, "u")
fv = fem(isn, "v")
fwx = fem(isn, "wx")
fwy = fem(isn, "wy")
fNxx = fem(isn, "Nxx")
fNyy = fem(isn, "Nyy")
fNxy = fem(isn, "Nxy")
fNnn = fem(isn, "Nnn")
fthx = fem(isn, "thx")
fthy = fem(isn, "thy")
fex = fem(isn, "ex")

@printf("\n=== L/h=%.0f  P̄=%.0f  BEM NEL=%d ni=%d ===\n", Lh, P, NEL, NINT)
@printf("  BEM  w_lin/h=%.4f  w_nl/h=%.4f\n", wL / h, wN / h)
@printf("  FEM  (see dump header)\n")
println("NL fields at collocation (BEM vs FEM):")
blk("w", w, fw)
blk("  Ω", w, fw; mask=I)
blk("u", um, fu)
blk("v", vm, fv)
blk("wx", wx, fwx)
blk("  Γ", wx, fwx; mask=G)
blk("  Ω", wx, fwx; mask=I)
blk("wy", wy, fwy)
blk("Nxx", Nxx, fNxx)
blk("  Ω", Nxx, fNxx; mask=I)
blk("  Γ", Nxx, fNxx; mask=G)
blk("Nyy", Nyy, fNyy)
blk("Nxy", Nxy, fNxy)
blk("βx vs θx", βx, fthx)
blk("βy vs θy", βy, fthy)

# N_nn on Γ (unprojected RBF vs FEM)
nrm = BEM.Plate._plate_normals(shell.plate)
Nn = zeros(n)
@inbounds for i in 1:n
    nx, ny = nrm[i][1], nrm[i][2]
    Nn[i] = Nxx[i] * nx * nx + Nyy[i] * ny * ny + 2 * Nxy[i] * nx * ny
end
@printf("\n  N_nn on Γ  rms BEM=%.3e  FEM=%.3e  rel=%.3f\n",
    sqrt(mean(abs2, Nn)), sqrt(mean(abs2, fNnn[G])), rel(Nn, fNnn[G]))
@printf("  N_nn x-edges rms BEM=%.3e FEM=%.3e\n",
    sqrt(mean(abs2, Nn[onx])), sqrt(mean(abs2, fNnn[G][onx])))
@printf("  N_nn y-edges rms BEM=%.3e FEM=%.3e\n",
    sqrt(mean(abs2, Nn[ony])), sqrt(mean(abs2, fNnn[G][ony])))

# mid-side and centre
function nearest(xy)
    return argmin(norm(p - Point2D(xy...)) for p in pts)
end
for (lab, xy) in (("centre", (0.5, 0.5)), ("x=0 mid", (0.0, 0.5)),
    ("x=1 mid", (1.0, 0.5)), ("y=0 mid", (0.5, 0.0)), ("qtr", (0.25, 0.25)))
    i = nearest(xy)
    @printf("\n  %s  BEM i=%d (%.3f,%.3f)  FEM same row\n", lab, i, pts[i][1], pts[i][2])
    @printf("    w    B=%+.4e F=%+.4e\n", w[i], fw[i])
    @printf("    u,v  B=(%+.3e,%+.3e) F=(%+.3e,%+.3e)\n", um[i], vm[i], fu[i], fv[i])
    @printf("    wx,wy B=(%+.3e,%+.3e) F=(%+.3e,%+.3e)\n", wx[i], wy[i], fwx[i], fwy[i])
    @printf("    Nxx  B=%+.3e F=%+.3e   Nyy B=%+.3e F=%+.3e\n",
        Nxx[i], fNxx[i], Nyy[i], fNyy[i])
    if i <= n
        @printf("    N_nn B=%+.3e F=%+.3e\n", Nn[i], fNnn[i])
    end
    @printf("    β/θ  B=(%+.3e,%+.3e) F=(%+.3e,%+.3e)\n", βx[i], βy[i], fthx[i], fthy[i])
end
println("done")
