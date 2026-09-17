# Useche (2025) Ch. 7–8 numerical examples
# Kirchhoff composite (Ch. 7) and FSDT composite (Ch. 8).
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf, StaticArrays
using BEM.Plate
using .ThinPlate

# ---------------------------------------------------------------------------
# CLPT / FSDT helpers
# ---------------------------------------------------------------------------

function ply_Q(E1, E2, ν12, G12)
    ν21 = ν12 * E2 / E1
    den = 1 - ν12 * ν21
    return (E1 / den, E2 / den, ν12 * E2 / den, G12)
end

function Qbar(E1, E2, ν12, G12, θ)
    Q11, Q22, Q12, Q66 = ply_Q(E1, E2, ν12, G12)
    m, n = cos(θ), sin(θ)
    m2, n2, m4, n4 = m^2, n^2, m^4, n^4
    mn = m * n
    Q11b = Q11 * m4 + Q22 * n4 + 2 * (Q12 + 2Q66) * m2 * n2
    Q22b = Q11 * n4 + Q22 * m4 + 2 * (Q12 + 2Q66) * m2 * n2
    Q12b = (Q11 + Q22 - 4Q66) * m2 * n2 + Q12 * (m4 + n4)
    Q66b = (Q11 + Q22 - 2Q12 - 2Q66) * m2 * n2 + Q66 * (m4 + n4)
    Q16b = (Q11 - Q12 - 2Q66) * m^3 * n + (Q12 - Q22 + 2Q66) * m * n^3
    Q26b = (Q11 - Q12 - 2Q66) * m * n^3 + (Q12 - Q22 + 2Q66) * m^3 * n
    return @SMatrix [Q11b Q12b Q16b; Q12b Q22b Q26b; Q16b Q26b Q66b]
end

"""plies: Vector of (E1, E2, ν12, G12, θ_deg, t). Midplane at z=0."""
function laminate_ABD(plies; Ks=5 / 6, G13=nothing, G23=nothing)
    h = sum(p[6] for p in plies)
    z = -h / 2
    A = zeros(3, 3)
    B = zeros(3, 3)
    D = zeros(3, 3)
    As = zeros(2, 2)
    for p in plies
        E1, E2, ν12, G12, θdeg, t = p
        θ = deg2rad(θdeg)
        Qb = Qbar(E1, E2, ν12, G12, θ)
        zb, zt = z, z + t
        A .+= Qb .* (zt - zb)
        B .+= Qb .* (zt^2 - zb^2) / 2
        D .+= Qb .* (zt^3 - zb^3) / 3
        g13 = G13 === nothing ? G12 : G13
        g23 = G23 === nothing ? G12 : G23
        m, n = cos(θ), sin(θ)
        Q55 = g13 * m^2 + g23 * n^2          # x-z
        Q44 = g13 * n^2 + g23 * m^2          # y-z
        Q45 = (g13 - g23) * m * n
        As .+= Ks * (zt - zb) * [Q55 Q45; Q45 Q44]
        z = zt
    end
    return (A=A, B=B, D=D, As=As, h=h)
end

function ortho_D(E1, E2, ν12, G12, h)
    ν21 = ν12 * E2 / E1
    den = 1 - ν12 * ν21
    D11 = E1 * h^3 / (12 * den)
    D22 = E2 * h^3 / (12 * den)
    D12 = ν12 * E2 * h^3 / (12 * den)
    D66 = G12 * h^3 / 12
    return D11, D22, D12, D66
end

"""Navier SS rectangular Kirchhoff (D16=D26=0) under uniform q."""
function navier_w_ortho(x, y; a, b=a, q, D11, D22, D12, D66, nterms=80)
    H = D12 + 2 * D66
    w = 0.0
    @inbounds for m in 1:2:nterms, n in 1:2:nterms
        α = m * π / a
        β = n * π / b
        den = D11 * α^4 + 2 * H * α^2 * β^2 + D22 * β^4
        qmn = 16q / (π^2 * m * n)
        w += (qmn / den) * sin(α * x) * sin(β * y)
    end
    return w
end

"""FSDT Navier SS cross-ply (symmetric, D16=D26=0) uniform q → w."""
function navier_w_fsdt(x, y; a, b=a, q, D11, D22, D12, D66, A44, A55, nterms=40)
    w = 0.0
    @inbounds for m in 1:2:nterms, n in 1:2:nterms
        α = m * π / a
        β = n * π / b
        K11 = D11 * α^2 + D66 * β^2 + A55
        K12 = (D12 + D66) * α * β
        K13 = A55 * α
        K22 = D66 * α^2 + D22 * β^2 + A44
        K23 = A44 * β
        K33 = A55 * α^2 + A44 * β^2
        K = [K11 K12 K13; K12 K22 K23; K13 K23 K33]
        qmn = 16q / (π^2 * m * n)
        Δ = K \ [0.0, 0.0, qmn]
        w += Δ[3] * sin(α * x) * sin(β * y)
    end
    return w
end

"""Cantilever FSDT beam (unit width): tip load q (force/length)."""
cantilever_w_tip(q, L, D11, A55) = q * L^3 / (3 * D11) + q * L / A55

println("="^72)
println(" Useche 2025  Ch. 7–8 examples")
println("="^72)

# ===========================================================================
# 7.5.1  Orthotropic SS square plate
# ===========================================================================
println("\n## 7.5.1  Orthotropic SS square (Kirchhoff)")
a = 1.0
h = 0.01
Ex = 2.068e11
Ey = Ex / 15
νxy = 0.3
Gxy = 6.055e8
q = 1e4
D11, D22, D12, D66 = ortho_D(Ex, Ey, νxy, Gxy, h)
wA = navier_w_ortho(a / 2, a / 2; a=a, q=q, D11=D11, D22=D22, D12=D12, D66=D66)
wB_c4 = navier_w_ortho(a / 4, a / 2; a=a, q=q, D11=D11, D22=D22, D12=D12, D66=D66)
wB_q = navier_w_ortho(a / 4, a / 4; a=a, q=q, D11=D11, D22=D22, D12=D12, D66=D66)
@printf("  D11=%.4e  D22=%.4e  D12=%.4e  D66=%.4e\n", D11, D22, D12, D66)
@printf("  series w(a/2,a/2) = %.6e m   book A 8.1258e-3\n", wA)
@printf("  series w(a/4,a/4) = %.6e m   book B 4.5211e-3\n", wB_q)
@printf("  series w(a/4,a/2) = %.6e m\n", wB_c4)
@printf("  rel vs book A: %.4f %%\n", 100 * abs(wA - 8.1258e-3) / 8.1258e-3)
@printf("  rel vs book B @ (a/4,a/4): %.4f %%\n", 100 * abs(wB_q - 4.5211e-3) / 4.5211e-3)

props = aniso_thin_plate_props(; D11=D11, D22=D22, D12=D12, D66=D66, q_c=q, h=h)
plate = build_square_plate(; a=a, n_el=6, bc="SSSS", props=props,
    corner_bc='F', internal=[SVector(a / 2, a / 2), SVector(a / 4, a / 4)])
assemble_plate!(plate; npg=10, singular=:guiggiani)
solve_plate!(plate)
wc_bem = plate_w_int(plate, 1)
wB_bem = plate_w_int(plate, 2)
@printf("  BEM aniso Kirchhoff n_el=6  w_c=%.6e m  (%.2f %% vs series A)\n",
    wc_bem, 100 * abs(wc_bem - wA) / abs(wA))
@printf("  BEM aniso Kirchhoff         w(a/4,a/4)=%.6e m  (%.2f %% vs series B)\n",
    wB_bem, 100 * abs(wB_bem - wB_q) / abs(wB_q))

# ===========================================================================
# 7.5.2  Cross-ply graphite-epoxy SS square
# ===========================================================================
println("\n## 7.5.2  Cross-ply graphite-epoxy SS square")
# Book OCR: Ex=2.07e9, Ey=5.17e9 (Ey>Ex). Standard high-modulus graphite-epoxy
# is E1=2.07e11, E2=5.17e9, G12=3.10e9, ν12=0.25 (Whitney/Pagano).
E1 = 2.07e11
E2 = 5.17e9
ν12 = 0.25
G12 = 3.10e9
q2 = 6.9e3
a2 = 1.0
h2 = 0.01
book_nd = 4.4718e-3   # w E22 h^3 / (q a^4)

function nd_w(w) 
    return w * E2 * h2^3 / (q2 * a2^4)
end

for (name, angs) in (
        ("5-layer [0/90/0/90/0]", [0, 90, 0, 90, 0]),
        ("9-layer [0/90]_4/0", [0, 90, 0, 90, 0, 90, 0, 90, 0]),
        ("[0/90/0/90/0]s  = 9 ply", [0, 90, 0, 90, 0, 90, 0, 90, 0]),
    )
    t = h2 / length(angs)
    plies = [(E1, E2, ν12, G12, float(θ), t) for θ in angs]
    lam = laminate_ABD(plies)
    w = navier_w_ortho(a2 / 2, a2 / 2; a=a2, q=q2,
        D11=lam.D[1, 1], D22=lam.D[2, 2], D12=lam.D[1, 2], D66=lam.D[3, 3])
    nd = nd_w(w)
    @printf("  %-28s  w=%.5e  nd=%.5f  book 4.4718e-3  err=%.2f %%\n",
        name, w, nd * 1e3, 100 * abs(nd - book_nd) / book_nd)
end

t9 = h2 / 9
plies9 = [(E1, E2, ν12, G12, float(θ), t9) for θ in [0, 90, 0, 90, 0, 90, 0, 90, 0]]
lam9 = laminate_ABD(plies9)
w9 = navier_w_ortho(a2 / 2, a2 / 2; a=a2, q=q2,
    D11=lam9.D[1, 1], D22=lam9.D[2, 2], D12=lam9.D[1, 2], D66=lam9.D[3, 3])
props9 = aniso_thin_plate_props(lam9.D; q_c=q2, h=h2)
plate9 = build_square_plate(; a=a2, n_el=6, bc="SSSS", props=props9,
    corner_bc='F', n_internal=1)
assemble_plate!(plate9; npg=10, singular=:guiggiani)
solve_plate!(plate9)
w9_bem = plate_w_int(plate9, 1)
@printf("  9-ply BEM aniso Kirchhoff  w=%.5e  nd=%.5f  (%.2f %% vs series)\n",
    w9_bem, nd_w(w9_bem) * 1e3, 100 * abs(w9_bem - w9) / abs(w9))

# OCR E1=2.07e9 (as printed)
E1b = 2.07e9
E2b = 5.17e9
angs = [0, 90, 0, 90, 0, 90, 0, 90, 0]
t = h2 / 9
plies = [(E1b, E2b, ν12, G12, float(θ), t) for θ in angs]
lam = laminate_ABD(plies)
w = navier_w_ortho(a2 / 2, a2 / 2; a=a2, q=q2,
    D11=lam.D[1, 1], D22=lam.D[2, 2], D12=lam.D[1, 2], D66=lam.D[3, 3])
@printf("  printed E1=2.07e9 9-ply   w=%.5e  nd×10³=%.5f\n", w, nd_w(w) * 1e3)

# ===========================================================================
# 7.7.1  SS cylindrical laminated shell
# ===========================================================================
println("\n## 7.7.1  SS cylindrical cross-ply shell  [90/0/0/90]s")
# κ11=1/50, κ22=0, a/h=10, a/b=1, E11=25 E22, ν12=0.25, G12=0.5 E22 (Ψ in OCR)
# Take a=10 so a/h=10 ⇒ h=1, R=1/κ=50 (a/R=0.2, shallow but not flat).
E22 = 1.0e9
E11 = 25 * E22
νs = 0.25
G12s = 0.5 * E22
aa = 10.0
hh = aa / 10
R11 = 50.0
q3 = 1.0
angs7 = [90.0, 0.0, 0.0, 90.0, 90.0, 0.0, 0.0, 90.0]
t7 = hh / 8
plies7 = [(E11, E22, νs, G12s, θ, t7) for θ in angs7]
lam7 = laminate_ABD(plies7)
@printf("  a=%.2f  h=%.3f  R11=%.1f  D11=%.4e  D22=%.4e  A11=%.4e\n",
    aa, hh, R11, lam7.D[1, 1], lam7.D[2, 2], lam7.A[1, 1])
w_flat = navier_w_ortho(aa / 2, aa / 2; a=aa, q=q3,
    D11=lam7.D[1, 1], D22=lam7.D[2, 2], D12=lam7.D[1, 2], D66=lam7.D[3, 3])
@printf("  flat-plate series w_c = %.6e  (no curvature)\n", w_flat)
ctr7 = navier_ss_laminate_shell(aa / 2, aa / 2; a=aa, q=q3, κ1=1 / R11, κ2=0,
    A=lam7.A, D=lam7.D, As=lam7.As)
@printf("  5-DOF Navier cyl w_c = %.6e  flat/cyl=%.3f\n", ctr7.w, w_flat / ctr7.w)
println("  coupled DIBEM (Wang + membrane u,v): scripts/plates/usech_771_dibem.jl")

# ===========================================================================
# 8.6.1  Cantilever [0/90/90/0]  (MATLAB GitHub = Ch. 8 code)
# ===========================================================================
println("\n## 8.6.1  Cantilever [0/90/90/0]  end shear")
# MATLAB Dynamic_Composite_Plate/test/Ejemplo_Laminado.m
# 10×5, h=0.1 (4×0.025), E1=4e6, E2=2e6, G12=1e6, G23=0.5e6, ν=0.25
# free-edge Vz = -100  (force/length)
E1c, E2c, νc = 4e6, 2e6, 0.25
G12c, G13c, G23c = 1e6, 1e6, 5e5
Lc, bc_w, hc = 10.0, 5.0, 0.1
q_end = 100.0
plies8 = [(E1c, E2c, νc, G12c, θ, 0.025) for θ in (0.0, 90.0, 90.0, 0.0)]
lam8 = laminate_ABD(plies8; Ks=5 / 6, G13=G13c, G23=G23c)
w_tip = cantilever_w_tip(q_end, Lc, lam8.D[1, 1], lam8.As[1, 1])
w_bend = q_end * Lc^3 / (3 * lam8.D[1, 1])
@printf("  MATLAB 10×5×0.1  V=100  D11=%.4e  A55=%.4e\n", lam8.D[1, 1], lam8.As[1, 1])
@printf("  FSDT beam w_tip = %.6e  (bending %.6e, shear %.1f %%)\n",
    w_tip, w_bend, 100 * (w_tip - w_bend) / w_tip)
props8 = laminate_fsdt_props(plies8; Ks=5 / 6, G13=G13c, G23=G23c, q_c=0.0, nθ=10)
mesh8 = build_rect_fsdt(; Lx=Lc, Ly=bc_w, n_el=(8, 4, 8, 4), bc="FFFC",
    vn=(0.0, -q_end, 0.0, 0.0), props=props8, n_internal=4, p=2)
assemble_fsdt!(mesh8; npg=10, nsub=8)
solve_fsdt!(mesh8)
itip = argmin(i -> begin
        p = mesh8.nodes[i]
        abs(p[1] - Lc) > 0.25 ? Inf : abs(p[2] - bc_w / 2)
    end, eachindex(mesh8.nodes))
w8 = fsdt_w(mesh8, itip)
@printf("  Wang BEM tip (%.2f, %.2f) w=%.6e  vs beam %.2f %%\n",
    mesh8.nodes[itip][1], mesh8.nodes[itip][2], w8,
    100 * abs(abs(w8) - w_tip) / w_tip)
println("  book: BEM <1% vs Reddy on 6–16 elements (Fig. 8.4); beam is 1-D.")

# book-like inch units if Table 3.1 were the MATLAB moduli in psi-scale
# skip; GitHub file is the runnable statement of the example

# ===========================================================================
# 8.6.2  SS [0/90/90/0] impulsive pressure
# ===========================================================================
println("\n## 8.6.2  SS square [0/90/90/0]  impulsive q=1")
# book: a=1, h=0.1, E11=4e6, E22=0.5 E11, G12=G31=1e6, G32=0.5 G12,
# ν=0.5 (MATLAB codes use 0.25), ρ=4000
a8 = 1.0
h8 = 0.1
E18, E28 = 4e6, 2e6
ν8_book, ν8_m = 0.5, 0.25
G128, G138, G238 = 1e6, 1e6, 5e5
ρ8 = 4000.0
q8 = 1.0
t8 = h8 / 4
for (tag, ν8) in (("MATLAB ν=0.25", ν8_m), ("book ν=0.5", ν8_book))
    let ν8 = ν8, tag = tag
        pl = [(E18, E28, ν8, G128, θ, t8) for θ in (0.0, 90.0, 90.0, 0.0)]
        lam8d = laminate_ABD(pl; Ks=5 / 6, G13=G138, G23=G238)
        wK = navier_w_ortho(a8 / 2, a8 / 2; a=a8, q=q8,
            D11=lam8d.D[1, 1], D22=lam8d.D[2, 2], D12=lam8d.D[1, 2], D66=lam8d.D[3, 3])
        wF = navier_w_fsdt(a8 / 2, a8 / 2; a=a8, q=q8,
            D11=lam8d.D[1, 1], D22=lam8d.D[2, 2], D12=lam8d.D[1, 2], D66=lam8d.D[3, 3],
            A44=lam8d.As[2, 2], A55=lam8d.As[1, 1])
        I0 = ρ8 * h8
        α8 = π / a8
        β8 = π / a8
        K11 = lam8d.D[1, 1] * α8^2 + lam8d.D[3, 3] * β8^2 + lam8d.As[1, 1]
        K12 = (lam8d.D[1, 2] + lam8d.D[3, 3]) * α8 * β8
        K13 = lam8d.As[1, 1] * α8
        K22 = lam8d.D[3, 3] * α8^2 + lam8d.D[2, 2] * β8^2 + lam8d.As[2, 2]
        K23 = lam8d.As[2, 2] * β8
        K33 = lam8d.As[1, 1] * α8^2 + lam8d.As[2, 2] * β8^2
        KK = [K11 K12 K13; K12 K22 K23; K13 K23 K33]
        Krr = KK[1:2, 1:2]
        Krw = KK[1:2, 3]
        Kred = KK[3, 3] - dot(Krw, Krr \ Krw)
        ω11 = sqrt(Kred / I0)
        T11 = 2π / ω11
        @printf("  %-16s  w_CLPT=%.4e  w_FSDT=%.4e  T11=%.4f s  (book plots w(t))\n",
            tag, wK, wF, T11)
    end
end
println("  Houbolt+DIBEM / Figs 8.5–8.7: scripts/plates/usech_86_dibem.jl")

println("\n## What this tree can / cannot BEM")
println("  Ch.7.5 Kirchhoff composite: AnisoThinPlateProps (Lekhnitskii μ, no iso smear).")
println("  Ch.7.7 shell: LaminatedShell (Wang + anisotropic membrane DIBEM).")
println("  Ch.9 dynamics Houbolt: scripts/plates/usech_96_houbolt.jl.")
println("  Ch.8 FSDT / Wang kernels: `wang_kernels` / `LaminateFSDTProps` in BEM.Plate.")
println("  https://github.com/jfuseche/BEM_Plate_Shell_Book_Juseche")
println("  (Dynamic_Composite_Plate = Ch.8).")
println("Done.")
