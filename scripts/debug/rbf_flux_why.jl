# Why RBF flux N∇w disagrees with analytic sine-w flux.
# julia --project=. scripts/debug/rbf_flux_why.jl
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf, Statistics, StaticArrays
using BEM.Plate

const a = 1.0
const NEL, NINT, NPG, NSUB = 4, 9, 6, 4
const RBF_G = PHS(3; poly_deg=1)

function matIII()
    E2 = 1.0
    E1 = 25 * E2
    return (E1, E2, 0.25, 0.5 * E2, 0.5 * E2, 0.2 * E2)
end
function plies_cross(angles, h, mat)
    E1, E2, ν12, G12, G13, G23 = mat
    t = h / length(angles)
    return [(E1, E2, ν12, G12, Float64(θ), t) for θ in angles], G13, G23
end

h100 = a / 100
pl, G13, G23 = plies_cross([0, 90, 90, 0], h100, matIII())
lam = laminate_fsdt_props(pl; Ks=5 / 6, G13=G13, G23=G23, q_c=1.0, ρ=1.0)
A, _, _, _, _ = BEM.Plate._laminate_ABD_AT(pl; Ks=5 / 6, G13=G13, G23=G23)
mesh = build_square_fsdt(; a=a, n_el=NEL, bc="SSSS", props=lam, n_internal=NINT)
shell = LaminatedShell(mesh, A, FlatShell(); mem_bc=:navier_ss)
assemble_laminated_shell!(shell; npg=NPG, nsub=NSUB, rbf=PHS(2; poly_deg=1),
    rbf_grad=RBF_G)
n = BEM.Plate._n(shell.plate)
ni = BEM.Plate._ni(shell.plate)
nt = n + ni
pts = Point2D[BEM.Plate._plate_nodes(shell.plate); BEM.Plate._plate_internal(shell.plate)]
q0 = 300 * h100^4
q_pts = [q0 * sin(π * p[1] / a) * sin(π * p[2] / a) for p in pts]
BEM.set_cache!(shell.plate; fsdt_q=shell.Mw * q_pts, q=shell.Mw * q_pts)
solve_laminated_shell!(shell)
w = [shell.plate.u[3i] for i in 1:nt]
W = w[n + 1]
w_sin = [W * sin(π * p[1] / a) * sin(π * p[2] / a) for p in pts]
wx_a = [W * (π / a) * cos(π * p[1] / a) * sin(π * p[2] / a) for p in pts]
wy_a = [W * (π / a) * sin(π * p[1] / a) * cos(π * p[2] / a) for p in pts]
Dx, Dy = shell.Dx, shell.Dy

function rel(u, v)
    return norm(u - v) / (norm(v) + 1e-30)
end
function stats(name, e, ib, ii)
    @printf("  %-22s  all=%.3f  Γ=%.3f  Ω=%.3f  max|e|_Γ=%.3e  max|e|_Ω=%.3e\n",
        name, rel(e, e),  # placeholder
        NaN, NaN, NaN, NaN)
end

ib = 1:n
ii = (n + 1):nt
on_side = falses(nt)
@inbounds for i in 1:n
    p = pts[i]
    on_side[i] = (abs(p[1]) < 1e-12 || abs(p[1] - a) < 1e-12) ⊻
                 (abs(p[2]) < 1e-12 || abs(p[2] - a) < 1e-12)
    # mid-side: exactly one coord on edge. XOR above is wrong for corners
end
# redo edge flags
xedge = falses(nt)
yedge = falses(nt)
@inbounds for i in 1:n
    p = pts[i]
    xedge[i] = abs(p[1]) < 1e-12 || abs(p[1] - a) < 1e-12
    yedge[i] = abs(p[2]) < 1e-12 || abs(p[2] - a) < 1e-12
end
corner = xedge .& yedge
midside = (xedge .| yedge) .& .!corner

function blk(tag, u, v, mask)
    um, vm = u[mask], v[mask]
    isempty(um) && return
    @printf("  %-28s  rel=%.3f  max|u|=%.3e  max|v|=%.3e  max|u-v|=%.3e\n",
        tag, rel(um, vm), maximum(abs, um), maximum(abs, vm), maximum(abs, um .- vm))
end

println("=== cloud ===")
@printf("  nΓ=%d  ni=%d  nt=%d  W=w_c=%.4e  h=%.4e\n", n, ni, nt, W, h100)
hs = [minimum(norm(pts[i] - pts[j]) for j in 1:nt if j != i) for i in 1:nt]
@printf("  min spacing  Γ: %.3e  Ω: %.3e  ratio Γ/Ω = %.2f\n",
    minimum(hs[ib]), minimum(hs[ii]), minimum(hs[ib]) / minimum(hs[ii]))
@printf("  cond(Dx)=%.3e  cond(Dy)=%.3e\n", cond(Dx), cond(Dy))

println("\n=== 1. BEM w vs fitted sine W sin(πx)sin(πy) ===")
@printf("  w=0 on Γ?  max|w_Γ|=%.3e\n", maximum(abs, w[ib]))
blk("w BEM vs sine, all", w, w_sin, 1:nt)
blk("w BEM vs sine, Ω", w, w_sin, ii)
blk("w BEM vs sine, Γ", w, w_sin, ib)

println("\n=== 2. RBF ∇(BEM w) vs analytic ∇(sine)  [shape + interpolant] ===")
wx_b, wy_b = Dx * w, Dy * w
blk("wx BEM-RBF vs sine", wx_b, wx_a, 1:nt)
blk("  Ω interior", wx_b, wx_a, ii)
blk("  Γ all", wx_b, wx_a, ib)
blk("  x=0 or a (should be max wx)", wx_b, wx_a, xedge)
blk("  y=0 or a (wx should be 0)", wx_b, wx_a, yedge .& .!xedge)
blk("wy BEM-RBF vs sine", wy_b, wy_a, 1:nt)
blk("  Ω interior", wy_b, wy_a, ii)
blk("  y=0 or a (max wy)", wy_b, wy_a, yedge)

println("\n=== 3. RBF ∇(sampled sine) vs analytic ∇(sine)  [interpolant only] ===")
wx_s, wy_s = Dx * w_sin, Dy * w_sin
blk("wx RBF(sine) vs ∂sine", wx_s, wx_a, 1:nt)
blk("  Ω", wx_s, wx_a, ii)
blk("  Γ", wx_s, wx_a, ib)
blk("  x-edges (true max)", wx_s, wx_a, xedge)
blk("  y-edges (true wx=0)", wx_s, wx_a, yedge .& .!xedge)

println("\n=== 4. RBF ∇(BEM w) vs RBF ∇(sine)  [BEM shape only, same Dx] ===")
blk("wx BEM vs sine through Dx", wx_b, wx_s, 1:nt)
blk("  Ω", wx_b, wx_s, ii)
blk("  Γ", wx_b, wx_s, ib)

println("\n=== 5. Flux v = N∇w, N = A:(½∇w⊗∇w), u=0 ===")
function flux_of(wx, wy)
    vx = similar(wx)
    vy = similar(wy)
    @inbounds for i in eachindex(wx)
        εx, εy, γ = 0.5 * wx[i]^2, 0.5 * wy[i]^2, wx[i] * wy[i]
        Nxx = A[1, 1] * εx + A[1, 2] * εy + A[1, 3] * γ
        Nyy = A[1, 2] * εx + A[2, 2] * εy + A[2, 3] * γ
        Nxy = A[1, 3] * εx + A[2, 3] * εy + A[3, 3] * γ
        vx[i] = Nxx * wx[i] + Nxy * wy[i]
        vy[i] = Nxy * wx[i] + Nyy * wy[i]
    end
    return vx, vy
end
vx_a, vy_a = flux_of(wx_a, wy_a)
vx_b, vy_b = flux_of(wx_b, wy_b)
vx_s, vy_s = flux_of(wx_s, wy_s)
blk("v_x analytic vs BEM-RBF", vx_b, vx_a, 1:nt)
blk("  Ω", vx_b, vx_a, ii)
blk("  Γ", vx_b, vx_a, ib)
blk("v_x analytic vs RBF(sine)", vx_s, vx_a, 1:nt)
blk("  Ω", vx_s, vx_a, ii)
blk("  Γ", vx_s, vx_a, ib)

# cubic scaling: if ∇w error is ε, v error ~ 3ε for small ε
@printf("\n  ‖∇w‖_∞ sine=%.3e  BEM-RBF=%.3e  RBF(sine)=%.3e\n",
    maximum(hypot.(wx_a, wy_a)), maximum(hypot.(wx_b, wy_b)),
    maximum(hypot.(wx_s, wy_s)))
@printf("  ‖v‖_∞ sine=%.3e  BEM-RBF=%.3e  RBF(sine)=%.3e\n",
    maximum(hypot.(vx_a, vy_a)), maximum(hypot.(vx_b, vy_b)),
    maximum(hypot.(vx_s, vy_s)))

println("\n=== 6. Where |v_x| lives (analytic) vs RBF error ===")
ord = sortperm(abs.(vx_a); rev=true)
@printf("  top |v_x| analytic nodes (i, x, y, v_a, v_BEM, v_RBFsine):\n")
for k in 1:min(8, nt)
    i = ord[k]
    p = pts[i]
    tag = i <= n ? "Γ" : "Ω"
    @printf("    %s i=%3d  (%.3f,%.3f)  ana=%+.3e  BEM=%+.3e  sRBF=%+.3e\n",
        tag, i, p[1], p[2], vx_a[i], vx_b[i], vx_s[i])
end
ordb = sortperm(abs.(vx_b .- vx_a); rev=true)
@printf("  top |v_x BEM-RBF − ana| :\n")
for k in 1:min(6, nt)
    i = ordb[k]
    p = pts[i]
    tag = i <= n ? "Γ" : "Ω"
    @printf("    %s i=%3d  (%.3f,%.3f)  Δv=%+.3e  wx_a=%+.3e  wx_b=%+.3e\n",
        tag, i, p[1], p[2], vx_b[i] - vx_a[i], wx_a[i], wx_b[i])
end

println("\n=== 7. Other RBF operators on the same sine samples ===")
function trial(label, rbf)
    ops = rbf_gradient_ops(pts; rbf=rbf)
    wx = ops.Fx * w_sin
    wy = ops.Fy * w_sin
    vx, vy = flux_of(wx, wy)
    @printf("  %-22s  condFx=%.2e  wx_all=%.3f wx_Ω=%.3f wx_Γ=%.3f  vx_all=%.3f vx_Ω=%.3f\n",
        label, cond(ops.Fx), rel(wx, wx_a), rel(wx[ii], wx_a[ii]), rel(wx[ib], wx_a[ib]),
        rel(vx, vx_a), rel(vx[ii], vx_a[ii]))
end
trial("PHS3 poly1 (live)", PHS(3; poly_deg=1))
trial("PHS3 poly2", PHS(3; poly_deg=2))
trial("PHS5 poly1", PHS(5; poly_deg=1))
trial("PHS5 poly2", PHS(5; poly_deg=2))
trial("PHS2 poly1", PHS(2; poly_deg=1))

println("\n=== 8. Exact sine interpolant: ∂w/∂n on SS ===")
# analytic |wx| max on x=0,y=a/2
@printf("  true wx(0,a/2)=%.4e  (W π/a)\n", W * π / a)
# find node nearest (0, 0.5)
i0 = argmin(norm(p - Point2D(0.0, a / 2)) for p in pts[1:n])
p0 = pts[i0]
@printf("  nearest Γ node (%.3f,%.3f)  wx_a=%.3e  wx_BEM=%.3e  wx_sRBF=%.3e\n",
    p0[1], p0[2], wx_a[i0], wx_b[i0], wx_s[i0])
println("done")
