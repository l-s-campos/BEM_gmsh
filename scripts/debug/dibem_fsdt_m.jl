# Diagnose FSDT / membrane DIBEM M vs true polar RIM of U* q.
# julia --project=. scripts/debug/dibem_fsdt_m.jl
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf, Statistics, StaticArrays
using BEM.Plate
using FastGaussQuadrature: gausslegendre

const a = 1.0
const NEL = 4
const NINT = 9
const NPG = 6
const NSUB = 4
const RBF = PHS(2; poly_deg=1)
const Nρ = 8

function matIII(E2=1.0)
    E1 = 25 * E2
    return (E1, E2, 0.25, 0.5 * E2, 0.5 * E2, 0.2 * E2)
end
function plies_cross(angles, h, mat)
    E1, E2, ν12, G12, G13, G23 = mat
    t = h / length(angles)
    return [(E1, E2, ν12, G12, Float64(θ), t) for θ in angles], G13, G23
end

function num_Fρ(Ufun, pf, ê, R; nρ=48)
    ρg, ρw = gausslegendre(nρ)
    Acc = zeros(3, 3)
    dummy = Point2D(1.0, 0.0)
    for (ir, γ) in enumerate(ρg)
        ρ = (γ + 1) / 2 * R
        ρ < 1e-14 && continue
        pg = pf + ρ * ê
        U, = Ufun(pg, pf, dummy)
        Acc .+= Matrix(U) .* (ρ * (R / 2) * ρw[ir])
    end
    return Acc
end

function polar_rim_q(mesh, qfun; npg=NPG, nρ=Nρ, srcs=nothing)
    n = BEM.Plate._n(mesh)
    ni = BEM.Plate._ni(mesh)
    pts = Point2D[BEM.Plate._plate_nodes(mesh); BEM.Plate._plate_internal(mesh)]
    idx = srcs === nothing ? collect(1:(n + ni)) : collect(srcs)
    qsi, w = gausslegendre(npg)
    ρg, ρw = gausslegendre(nρ)
    dummy = Point2D(1.0, 0.0)
    props = BEM.Plate._plate_props(mesh)
    out = zeros(3, length(idx))
    for (k, i) in enumerate(idx)
        pf = pts[i]
        qe = zeros(3)
        for el in mesh.elements
            for (ig, ξ) in enumerate(qsi)
                pg, J, n̂ = BEM.Plate.elem_geom(el, ξ)
                RX, RY = pg[1] - pf[1], pg[2] - pf[2]
                R = hypot(RX, RY)
                R < 1e-14 && continue
                nr = (n̂[1] * RX + n̂[2] * RY) / R
                F = zeros(3)
                for (ir, γ) in enumerate(ρg)
                    ρ = (γ + 1) / 2 * R
                    ρ < 1e-14 && continue
                    pgρ = pf + (ρ / R) * (pg - pf)
                    U, = fsdt_kernels(props, pgρ, pf, dummy)
                    qX = qfun(pgρ)
                    F .+= U[:, 3] .* (qX * ρ * (R / 2) * ρw[ir])
                end
                qe .+= F .* (nr / R * J * w[ig])
            end
        end
        out[:, k] = qe
    end
    return idx, out
end

function rebuild_c(mesh, rbf)
    pts = Point2D[BEM.Plate._plate_nodes(mesh); BEM.Plate._plate_internal(mesh)]
    nt = length(pts)
    IF = BEM.Plate._fsdt_IF(mesh, pts, rbf; npg=NPG)
    Frbf = zeros(nt, nt)
    @inbounds for j in 1:nt, i in 1:nt
        i == j && continue
        Frbf[i, j] = rbf(norm(pts[i] - pts[j]))
    end
    BEM._dibem_ridge_F!(Frbf)
    IP = BEM.Plate._fsdt_monomial_IP(mesh, rbf; npg=NPG)
    c = BEM._dibem_poly_c(Frbf, IF, pts, rbf; IP=IP)
    return Frbf, IF, c
end

function wblock(Mw, nt)
    W = zeros(nt, nt)
    @inbounds for j in 1:nt, i in 1:nt
        W[i, j] = Mw[3i, j]
    end
    return W
end

function report_vec(name, v)
    @printf("  %-22s  min=%+.3e  max=%+.3e  mean=%+.3e  %%neg=%.1f  ‖v‖=%.3e\n",
        name, minimum(v), maximum(v), mean(v), 100 * count(<(0), v) / length(v),
        norm(v))
end

println("=== 1. Fρ vs numerical ∫ U* ρ dρ ===")
pf = Point2D(0.0, 0.0)
pg = Point2D(0.25, 0.1)
R = norm(pg - pf)
ê = (pg - pf) / R
iso = FSDTProps(; E=1e5, ν=0.3, h=0.01, q_c=1.0)
Uiso(pg, pf, n) = fsdt_kernels(iso, pg, pf, n)
Fana = fsdt_Fρ(iso, pg, pf)
Fnum = num_Fρ(Uiso, pf, ê, R)
@printf("  isotropic Reissner  rel‖Fρ-num‖ = %.3e\n",
    norm(Fana - Fnum) / (norm(Fana) + 1e-30))
@printf("    Fρ_33=%.6e  num_33=%.6e\n", Fana[3, 3], Fnum[3, 3])

h100 = a / 100
pl, G13, G23 = plies_cross([0, 90, 90, 0], h100, matIII())
lam = laminate_fsdt_props(pl; Ks=5 / 6, G13=G13, G23=G23, q_c=1.0, ρ=1.0)
Ulam(pg, pf, n) = fsdt_kernels(lam, pg, pf, n)
Flana = fsdt_Fρ(lam, pg, pf)
Flnum = num_Fρ(Ulam, pf, ê, R; nρ=64)
@printf("  Wang laminate L/h=100  rel‖Fρ-num‖ = %.3e\n",
    norm(Flana - Flnum) / (norm(Flana) + 1e-30))
@printf("    Fρ_33=%.6e  num_33=%.6e  Fρ_13=%.6e  num_13=%.6e\n",
    Flana[3, 3], Flnum[3, 3], Flana[1, 3], Flnum[1, 3])

h4 = a / 4
pl4, G13b, G23b = plies_cross([0, 90, 90, 0], h4, matIII())
lam4 = laminate_fsdt_props(pl4; Ks=5 / 6, G13=G13b, G23=G23b, q_c=1.0, ρ=1.0)
U4(pg, pf, n) = fsdt_kernels(lam4, pg, pf, n)
F4a = fsdt_Fρ(lam4, pg, pf)
F4n = num_Fρ(U4, pf, ê, R; nρ=64)
@printf("  Wang laminate L/h=4    rel‖Fρ-num‖ = %.3e\n",
    norm(F4a - F4n) / (norm(F4a) + 1e-30))

println("\n=== 2. Build Table-3-like shell (NEL=$NEL NINT=$NINT L/h=100) ===")
A, _, _, _, _ = BEM.Plate._laminate_ABD_AT(pl; Ks=5 / 6, G13=G13, G23=G23)
mesh = build_square_fsdt(; a=a, n_el=NEL, bc="SSSS", props=lam, n_internal=NINT)
shell = LaminatedShell(mesh, A, FlatShell(); mem_bc=:navier_ss)
assemble_laminated_shell!(shell; npg=NPG, nsub=NSUB, rbf=RBF,
    rbf_grad=PHS(3; poly_deg=1))
mesh = FSDTMesh(shell.plate)
nt = BEM.Plate._n(mesh) + BEM.Plate._ni(mesh)
n = BEM.Plate._n(mesh)
pts = Point2D[mesh.nodes; mesh.internal]
Mw = shell.Mw
ID = BEM.Plate._fsdt_ID(mesh; npg=NPG)
ones_w = ones(nt)
Mw1 = Mw * ones_w
IDcol3 = reduce(vcat, [ID[3i-2:3i, 3] for i in 1:nt])
IDw = [ID[3i, 3] for i in 1:nt]
@printf("  nt=%d nΓ=%d ni=%d  ndof_p=%d\n", nt, n, BEM.Plate._ni(mesh), 3nt)
@printf("  remainder ‖Mw*1 - ID[:,3]‖/‖ID[:,3]‖ = %.3e\n",
    norm(Mw1 - IDcol3) / (norm(IDcol3) + 1e-30))

W = wblock(Mw, nt)
dW = [W[i, i] for i in 1:nt]
offW = copy(W)
@inbounds for i in 1:nt
    offW[i, i] = 0
end
@printf("  W-block  max|diag|=%.3e  max|off|=%.3e  off/diag=%.1f  cond(W)=%.3e\n",
    maximum(abs, dW), maximum(abs, offW),
    maximum(abs, offW) / (maximum(abs, dW) + 1e-30), cond(W))
@printf("  W row-sum (should = IDw) rel = %.3e\n",
    norm(vec(sum(W; dims=2)) - IDw) / (norm(IDw) + 1e-30))
@printf("  center IDw=%.3e  W_11=%.3e  mean|W_1j|=%.3e\n",
    IDw[n + 1], W[n + 1, n + 1], mean(abs, W[n + 1, :]))

Frbf, IF, c = rebuild_c(mesh, RBF)
report_vec("c (PHS2+lin)", c)
report_vec("IF", IF)
@printf("  cond(F+ridge)=%.3e  rank≈ %d / %d\n",
    cond(Frbf), rank(Frbf), size(Frbf, 1))
F0 = copy(Frbf)
@inbounds for i in 1:nt
    F0[i, i] = 0
end
@printf("  rank(F off-diag r²) = %d  (PHS2 = |x-y|² is rank ≤ 4)\n", rank(F0))

println("\n=== 3. Polar RIM of U* q  vs  Mw*q  vs  lumped ID*q(ξ) ===")
ic = n + 1
srcs = unique([ic, 1, n, min(nt, ic + 1), max(1, ic - 1)])
q_one(X) = 1.0
q_sine(X) = sin(π * X[1] / a) * sin(π * X[2] / a)
q_xx(X) = (X[1] - a / 2)^2
idx, rim1 = polar_rim_q(mesh, q_one; srcs=srcs)
_, rims = polar_rim_q(mesh, q_sine; srcs=srcs)
_, rimx = polar_rim_q(mesh, q_xx; srcs=srcs)
q1 = ones(nt)
qs = [q_sine(p) for p in pts]
qx = [q_xx(p) for p in pts]
Mwq1 = Mw * q1
Mwqs = Mw * qs
Mwqx = Mw * qx
lmp1 = IDcol3
lmps = reduce(vcat, [ID[3i-2:3i, 3] * qs[i] for i in 1:nt])
lmpx = reduce(vcat, [ID[3i-2:3i, 3] * qx[i] for i in 1:nt])

function cmp_src(label, idx, rim, Mwq, lmp)
    println("  -- $label --")
    println("     src     rim_w          Mw_w           lmp_w         Mw/rim   lmp/rim")
    for (k, i) in enumerate(idx)
        rw, mw, lw = rim[3, k], Mwq[3i], lmp[3i]
        @printf("     %3d  %+.6e  %+.6e  %+.6e  %7.3f  %7.3f\n",
            i, rw, mw, lw, mw / (rw + 1e-30), lw / (rw + 1e-30))
    end
end
cmp_src("q=1", idx, rim1, Mwq1, lmp1)
cmp_src("q=sinπx sinπy", idx, rims, Mwqs, lmps)
cmp_src("q=(x-a/2)²", idx, rimx, Mwqx, lmpx)

println("\n=== 4. Linear sine solve + geometric qg ===")
q0 = 300 * 1.0 * h100^4 / a^4
q_pts = [q0 * q_sine(p) for p in pts]
BEM.set_cache!(shell.plate; fsdt_q=Mw * q_pts, q=Mw * q_pts)
solve_laminated_shell!(shell)
up = shell.plate.u
w = [up[3i] for i in 1:nt]
um = [shell.u_m[2i - 1] for i in 1:nt]
vm = [shell.u_m[2i] for i in 1:nt]
@printf("  w_c/h = %.4f  (Pagano FSDT lin P̄=300 → 1.297)\n", abs(w[ic]) / h100)
@printf("  max|u|=%.3e  max|v|=%.3e  max|w|=%.3e\n",
    maximum(abs, um), maximum(abs, vm), maximum(abs, w))
qκ, fvk, vx, vy, = BEM.Plate._shell_vk_loads(shell,
    vcat(shell.plate.u[1:3nt], shell.u_m))
qg = shell.Dx * vx .+ shell.Dy * vy
report_vec("w", w)
report_vec("qg = div(N∇w)", qg)
report_vec("vx", vx)
report_vec("fvk_x", fvk[1:2:end])
Nscale = maximum(abs, vx) / (maximum(abs, w) + 1e-30)
@printf("  |qg_c|=%.3e  qg_c/q0=%.3e  sine_c=1  Mw*sine_c/Mw*1 = %.3f\n",
    qg[ic], qg[ic] / q0, (Mw * qs)[3ic] / (Mw * q1)[3ic])

q_qg(X) = begin
    # inverse-distance interpolant of nodal qg
    s, sw = 0.0, 0.0
    @inbounds for j in 1:nt
        d2 = sum(abs2, X - pts[j])
        wi = 1 / (d2 + 1e-16)
        s += wi * qg[j]
        sw += wi
    end
    return s / sw
end
_, rimqg = polar_rim_q(mesh, q_qg; srcs=srcs)
Mwqg = Mw * qg
lmpqg = reduce(vcat, [ID[3i-2:3i, 3] * qg[i] for i in 1:nt])
cmp_src("q=qg (IDW interp)", idx, rimqg, Mwqg, lmpqg)

if !isempty(shell.Mx)
    Ibp = shell.Mx * vx .+ shell.My * vy
    println("  IBP Mx*vx+My*vy vs Mw*qg vs lumped vs RIM (w-eq):")
    println("     src     RIM            Mw*qg          lumped         IBP")
    for (k, i) in enumerate(idx)
        @printf("     %3d  %+.6e  %+.6e  %+.6e  %+.6e\n",
            i, rimqg[3, k], Mwqg[3i], lmpqg[3i], Ibp[3i])
    end
end

println("\n=== 5. Membrane Mm ===")
Mm = shell.Mm
IDm = BEM.Plate._membrane_ID(shell; npg=NPG)
e1 = zeros(2nt)
e2 = zeros(2nt)
@inbounds for i in 1:nt
    e1[2i - 1] = 1
    e2[2i] = 1
end
M1 = Mm * e1
M2 = Mm * e2
ID1 = reduce(vcat, [IDm[2i-1:2i, 1] for i in 1:nt])
ID2 = reduce(vcat, [IDm[2i-1:2i, 2] for i in 1:nt])
@printf("  ‖Mm e1 - ID[:,1]‖/‖ID‖ = %.3e\n", norm(M1 - ID1) / (norm(ID1) + 1e-30))
@printf("  ‖Mm e2 - ID[:,2]‖/‖ID‖ = %.3e\n", norm(M2 - ID2) / (norm(ID2) + 1e-30))
Mu = zeros(nt, nt)
@inbounds for j in 1:nt, i in 1:nt
    Mu[i, j] = Mm[2i - 1, 2j - 1]
end
dMu = [Mu[i, i] for i in 1:nt]
offMu = copy(Mu)
@inbounds for i in 1:nt
    offMu[i, i] = 0
end
@printf("  Mu-block max|diag|=%.3e  max|off|=%.3e  off/diag=%.1f  cond=%.3e\n",
    maximum(abs, dMu), maximum(abs, offMu),
    maximum(abs, offMu) / (maximum(abs, dMu) + 1e-30), cond(Mu))
rhs_m = Mm * fvk
@printf("  ‖Mm*fvk‖=%.3e  ‖fvk‖=%.3e  max|Mm*fvk|=%.3e\n",
    norm(rhs_m), norm(fvk), maximum(abs, rhs_m))

println("\n=== 6. PHS3+lin c (same cloud) vs PHS2 ===")
F3, IF3, c3 = rebuild_c(mesh, PHS(3; poly_deg=1))
report_vec("c (PHS3+lin)", c3)
@printf("  cond(F3)=%.3e  rank=%d / %d\n", cond(F3), rank(F3), size(F3, 1))

println("\n=== 7. Isotropic Reissner M (same mesh density) W-block ===")
iso_m = FSDTProps(; E=1e5, ν=0.3, h=h100, q_c=1.0, ρ=1.0)
mesh_i = build_square_fsdt(; a=a, n_el=NEL, bc="SSSS", props=iso_m, n_internal=NINT)
assemble_fsdt!(mesh_i; npg=NPG, nsub=NSUB)
dibem_fsdt!(mesh_i; npg=NPG, rbf=RBF)
fm_i = FSDTMesh(mesh_i)
I0i = iso_m.ρ * iso_m.h
Mwi = zeros(3nt, nt)
@inbounds for j in 1:nt
    Mwi[:, j] .= fm_i.M[:, 3j] ./ I0i
end
Wi = wblock(Mwi, nt)
dWi = [Wi[i, i] for i in 1:nt]
offWi = copy(Wi)
@inbounds for i in 1:nt
    offWi[i, i] = 0
end
@printf("  iso W  max|diag|=%.3e  max|off|=%.3e  off/diag=%.1f  cond=%.3e\n",
    maximum(abs, dWi), maximum(abs, offWi),
    maximum(abs, offWi) / (maximum(abs, dWi) + 1e-30), cond(Wi))
Fiso, _, ciso = rebuild_c(fm_i, RBF)
report_vec("c iso PHS2", ciso)

println("\n=== 8. Analytic sine-w qg vs RBF qg; Mw on 2π harmonic ===")
W = w[ic]
Amat = shell.A
function ana_grad(X)
    sx, cx = sin(π * X[1] / a), cos(π * X[1] / a)
    sy, cy = sin(π * X[2] / a), cos(π * X[2] / a)
    wx = W * (π / a) * cx * sy
    wy = W * (π / a) * sx * cy
    wxx = -W * (π / a)^2 * sx * sy
    wyy = -W * (π / a)^2 * sx * sy
    wxy = W * (π / a)^2 * cx * cy
    return wx, wy, wxx, wyy, wxy
end
function ana_N(wx, wy)
    εx, εy, γ = 0.5 * wx^2, 0.5 * wy^2, wx * wy
    Nxx = Amat[1, 1] * εx + Amat[1, 2] * εy + Amat[1, 3] * γ
    Nyy = Amat[1, 2] * εx + Amat[2, 2] * εy + Amat[2, 3] * γ
    Nxy = Amat[1, 3] * εx + Amat[2, 3] * εy + Amat[3, 3] * γ
    return Nxx, Nyy, Nxy
end
function ana_qg2(X)
    hfd = 1e-5
    function vpair(P)
        wx, wy, _, _, _ = ana_grad(P)
        Nxx, Nyy, Nxy = ana_N(wx, wy)
        return Nxx * wx + Nxy * wy, Nxy * wx + Nyy * wy
    end
    vxp, _ = vpair(X + Point2D(hfd, 0.0))
    vxm, _ = vpair(X + Point2D(-hfd, 0.0))
    _, vyp = vpair(X + Point2D(0.0, hfd))
    _, vym = vpair(X + Point2D(0.0, -hfd))
    return (vxp - vxm) / (2hfd) + (vyp - vym) / (2hfd)
end
qg_ana = [ana_qg2(p) for p in pts]
report_vec("qg analytic", qg_ana)
report_vec("qg RBF     ", qg)
@printf("  qg_c ana=%.3e  RBF=%.3e  (sine-w centre should be ~0)\n",
    qg_ana[ic], qg[ic])
@printf("  rel‖qg_RBF-ana‖/‖ana‖ = %.3e\n",
    norm(qg - qg_ana) / (norm(qg_ana) + 1e-30))
wx_r = shell.Dx * w
wy_r = shell.Dy * w
wx_a = [ana_grad(p)[1] for p in pts]
wy_a = [ana_grad(p)[2] for p in pts]
@printf("  rel‖∇w_RBF-ana‖ = %.3e  (wx)  %.3e (wy)\n",
    norm(wx_r - wx_a) / (norm(wx_a) + 1e-30),
    norm(wy_r - wy_a) / (norm(wy_a) + 1e-30))

_, rim_ana = polar_rim_q(mesh, ana_qg2; srcs=srcs)
Mwana = Mw * qg_ana
lmpana = reduce(vcat, [ID[3i-2:3i, 3] * qg_ana[i] for i in 1:nt])
cmp_src("q=qg analytic", idx, rim_ana, Mwana, lmpana)

q2(X) = sin(2π * X[1] / a) * sin(2π * X[2] / a)
_, rim2 = polar_rim_q(mesh, q2; srcs=srcs)
q2n = [q2(p) for p in pts]
Mw2 = Mw * q2n
lmp2 = reduce(vcat, [ID[3i-2:3i, 3] * q2n[i] for i in 1:nt])
cmp_src("q=sin2πx sin2πy", idx, rim2, Mw2, lmp2)

vx_ana = zeros(nt)
vy_ana = zeros(nt)
@inbounds for i in 1:nt
    wx, wy, _, _, _ = ana_grad(pts[i])
    Nxx, Nyy, Nxy = ana_N(wx, wy)
    vx_ana[i] = Nxx * wx + Nxy * wy
    vy_ana[i] = Nxy * wx + Nyy * wy
end
Ibp_ana = BEM.Plate.ibp_div_Uv(shell.Mx, shell.My, mesh.G, mesh.Normal,
    vx_ana, vy_ana, 3)
Ibp_rbf = BEM.Plate.ibp_div_Uv(shell.Mx, shell.My, mesh.G, mesh.Normal, vx, vy, 3)
cmp_src("IBP+Γ vs RIM (analytic flux)", idx, rim_ana, Ibp_ana, Ibp_rbf)

load_c = (Mw * q_pts)[3ic]
kic = findfirst(==(ic), idx)
@printf("\n  centre w-eq magnitudes at P̄=300 linear state:\n")
@printf("    Mw*(q0 sine)     = %+.4e\n", load_c)
@printf("    RIM U* qg_ana    = %+.4e   ratio to load %.3f\n",
    rim_ana[3, kic], rim_ana[3, kic] / load_c)
@printf("    Mw*qg_ana        = %+.4e   ratio to load %.3f\n",
    Mwana[3ic], Mwana[3ic] / load_c)
@printf("    IBP+Γ analytic v = %+.4e   ratio to load %.3f  IBP/RIM=%.3f\n",
    Ibp_ana[3ic], Ibp_ana[3ic] / load_c, Ibp_ana[3ic] / (rim_ana[3, kic] + 1e-30))
@printf("    IBP+Γ RBF v      = %+.4e   ratio to load %.3f\n",
    Ibp_rbf[3ic], Ibp_rbf[3ic] / load_c)
@printf("    Mw*qg_RBF        = %+.4e   ratio to load %.3f\n",
    Mwqg[3ic], Mwqg[3ic] / load_c)
@printf("    lumped ID*qg_RBF = %+.4e   ratio to load %.3f\n",
    lmpqg[3ic], lmpqg[3ic] / load_c)

println("\ndone")
