# Useche (2025) Ch. 9–10 numerical examples
# Ch.9 FSDT laminated shallow shells; Ch.10 cracked thick plates (DBEM).
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf, StaticArrays
using BEM.Crack

# ---------------------------------------------------------------------------
# CLPT / FSDT (same as usech_ch78.jl)
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
    Q11b = Q11 * m4 + Q22 * n4 + 2 * (Q12 + 2Q66) * m2 * n2
    Q22b = Q11 * n4 + Q22 * m4 + 2 * (Q12 + 2Q66) * m2 * n2
    Q12b = (Q11 + Q22 - 4Q66) * m2 * n2 + Q12 * (m4 + n4)
    Q66b = (Q11 + Q22 - 2Q12 - 2Q66) * m2 * n2 + Q66 * (m4 + n4)
    Q16b = (Q11 - Q12 - 2Q66) * m^3 * n + (Q12 - Q22 + 2Q66) * m * n^3
    Q26b = (Q11 - Q12 - 2Q66) * m * n^3 + (Q12 - Q22 + 2Q66) * m^3 * n
    return @SMatrix [Q11b Q12b Q16b; Q12b Q22b Q26b; Q16b Q26b Q66b]
end
function laminate_ABD(plies; Ks=5 / 6, G13=nothing, G23=nothing)
    h = sum(p[6] for p in plies)
    z = -h / 2
    A = zeros(3, 3); B = zeros(3, 3); D = zeros(3, 3); As = zeros(2, 2)
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
        Q55 = g13 * m^2 + g23 * n^2
        Q44 = g13 * n^2 + g23 * m^2
        Q45 = (g13 - g23) * m * n
        As .+= Ks * (zt - zb) * [Q55 Q45; Q45 Q44]
        z = zt
    end
    return (A=A, B=B, D=D, As=As, h=h)
end

"""Reddy FSDT Navier SS-1 cross-ply shallow spherical shell, uniform q.
Returns (w, Nx, Ny, Mx, My) at (x,y). 19 terms = book."""
function navier_ss_sphere(x, y; a, b=a, q, κ1, κ2, A, D, As, nterms=19)
    w = Nx = Ny = Mx = My = 0.0
    @inbounds for m in 1:2:nterms, n in 1:2:nterms
        α = m * π / a
        β = n * π / b
        A11, A22, A12, A66 = A[1, 1], A[2, 2], A[1, 2], A[3, 3]
        D11, D22, D12, D66 = D[1, 1], D[2, 2], D[1, 2], D[3, 3]
        A44, A55 = As[2, 2], As[1, 1]
        K = zeros(5, 5)
        K[1, 1] = A11 * α^2 + A66 * β^2
        K[1, 2] = (A12 + A66) * α * β
        K[1, 3] = -(A11 * κ1 + A12 * κ2) * α
        K[2, 1] = K[1, 2]
        K[2, 2] = A22 * β^2 + A66 * α^2
        K[2, 3] = -(A12 * κ1 + A22 * κ2) * β
        K[3, 1] = K[1, 3]
        K[3, 2] = K[2, 3]
        K[3, 3] = A55 * α^2 + A44 * β^2 +
                  (A11 * κ1 + A12 * κ2) * κ1 + (A12 * κ1 + A22 * κ2) * κ2
        K[3, 4] = A55 * α
        K[3, 5] = A44 * β
        K[4, 3] = K[3, 4]
        K[4, 4] = D11 * α^2 + D66 * β^2 + A55
        K[4, 5] = (D12 + D66) * α * β
        K[5, 3] = K[3, 5]
        K[5, 4] = K[4, 5]
        K[5, 5] = D22 * β^2 + D66 * α^2 + A44
        qmn = 16q / (π^2 * m * n)
        Δ = K \ [0.0, 0.0, qmn, 0.0, 0.0]
        U, V, W, X, Y = Δ
        s = sin(α * x) * sin(β * y)
        w += W * s
        εx = -α * U + κ1 * W
        εy = -β * V + κ2 * W
        Nx += (A11 * εx + A12 * εy) * s
        Ny += (A12 * εx + A22 * εy) * s
        Mx += (-D11 * α * X - D12 * β * Y) * s
        My += (-D12 * α * X - D22 * β * Y) * s
    end
    return (w=w, Nx=Nx, Ny=Ny, Mx=Mx, My=My)
end

println("="^72)
println(" Useche 2025  Ch. 9–10 examples")
println("="^72)

# ===========================================================================
# 9.6.1  SS spherical laminated shell  [0/90]s
# ===========================================================================
println("\n## 9.6.1  SS double-curved shell  κ11=κ22=1/100, a/h=100, q=1")
# h/4=0.25 ⇒ h=1, four plies [0/90]s. Material = Reddy E1/E2=25 (Ch. 7.7.1).
E1, E2, ν12 = 25.0, 1.0, 0.25
G12 = G13 = 0.5 * E2
G23 = 0.2 * E2
h = 1.0
a = 100.0
R = 100.0
κ = 1 / R
q = 1.0
plies = [(E1, E2, ν12, G12, θ, h / 4) for θ in (0.0, 90.0, 90.0, 0.0)]
lam = laminate_ABD(plies; Ks=5 / 6, G13=G13, G23=G23)
@printf("  A11=%.4e  A22=%.4e  D11=%.4e  D22=%.4e  A55=%.4e\n",
    lam.A[1, 1], lam.A[2, 2], lam.D[1, 1], lam.D[2, 2], lam.As[1, 1])
ctr = navier_ss_sphere(a / 2, a / 2; a=a, q=q, κ1=κ, κ2=κ,
    A=lam.A, D=lam.D, As=lam.As, nterms=19)
flat = navier_ss_sphere(a / 2, a / 2; a=a, q=q, κ1=0, κ2=0,
    A=lam.A, D=lam.D, As=lam.As, nterms=19)
nd = ctr.w * E2 * h^3 / (q * a^4)
@printf("  19-term FSDT  w_c=%.6e  w E2 h³/(q a⁴)×10³=%.4f\n", ctr.w, nd * 1e3)
@printf("  Nx=%.4e  Ny=%.4e  Mx=%.4e  My=%.4e\n", ctr.Nx, ctr.Ny, ctr.Mx, ctr.My)
@printf("  flat-plate w_c=%.6e  shell/flat=%.3f\n", flat.w, ctr.w / flat.w)
println("  book: 16 BE + 98 RIM vs Reddy 19-term Navier (Figs. 9.2–9.3, no table).")

# Shallow scaling: a=1, h=0.01, R=100 (a/h=100, κ=1/100, a/R=0.01).
# Literal h/4=0.25 above is a=R=100 (a/R=1, not shallow).
a1, h1, R1 = 1.0, 0.01, 100.0
plies1 = [(E1, E2, ν12, G12, θ, h1 / 4) for θ in (0.0, 90.0, 90.0, 0.0)]
lam1 = laminate_ABD(plies1; Ks=5 / 6, G13=G13, G23=G23)
ctr1 = navier_ss_sphere(a1 / 2, a1 / 2; a=a1, q=q, κ1=1 / R1, κ2=1 / R1,
    A=lam1.A, D=lam1.D, As=lam1.As, nterms=19)
flat1 = navier_ss_sphere(a1 / 2, a1 / 2; a=a1, q=q, κ1=0, κ2=0,
    A=lam1.A, D=lam1.D, As=lam1.As, nterms=19)
@printf("  shallow a=1,h=0.01,R=100  w_c=%.6e  flat=%.6e  ratio=%.3f\n",
    ctr1.w, flat1.w, ctr1.w / flat1.w)

println("  BEM: scripts/plates/usech_96_dibem.jl (static DIBEM) and")
println("       scripts/plates/usech_96_houbolt.jl (Ch.9 Houbolt + DIBEM mass).")

# ===========================================================================
# 9.6.2  Clamped circular laminated shell
# ===========================================================================
println("\n## 9.6.2  Clamped circular shell  a=5, h=0.1, p=pmax(1+r²)")
println("  Table 9.1–9.2 at r=a/2 for [90/0/90/90/0]. pmax not printed.")
println("  Column '1/R' matches membrane N~pR/2 only if it is the radius R.")
angs = [90.0, 0.0, 90.0, 90.0, 0.0]
h9 = 0.1
ac = 5.0
plies9 = [(E1, E2, ν12, G12, θ, h9 / 5) for θ in angs]
lam9 = laminate_ABD(plies9; Ks=5 / 6, G13=G13, G23=G23)
A11, A22, A12 = lam9.A[1, 1], lam9.A[2, 2], lam9.A[1, 2]
@printf("  ABD  A11=%.3f  A22=%.3f  A12=%.3f  D11=%.4f  D22=%.4f\n",
    A11, A22, A12, lam9.D[1, 1], lam9.D[2, 2])
# book Table 9.1 BEM
bookN = (
    (20, -83.597, -111.230, -9.279),
    (50, -219.667, -145.927, -4.812),
    (80, -264.382, -174.137, -0.807),
    (100, -272.045, -179.895, -0.873),
)
bookM = (
    (20, -0.191, -0.811, 0.102),
    (50, -3.471, -0.591, 0.204),
    (80, -5.710, -1.142, 0.200),
    (100, -7.003, -1.389, 0.176),
)
println("  membrane estimate Nα+Nβ = -p R, split by Aαα/(A11+A22); pmax=1, r=a/2:")
@printf("  %6s %10s %10s %10s %10s %10s\n",
    "R", "N11_mem", "N11_BEM", "N22_mem", "N22_BEM", "pR/2")
for (R, N11, N22, _) in bookN
    p = 1.0 * (1 + (ac / 2)^2)          # pmax=1, r=a/2
    pR = p * R
    # N11+N22 = -p R  (spherical membrane); split by A
    sA = A11 + A22
    N11m = -pR * A11 / sA
    N22m = -pR * A22 / sA
    @printf("  %6.0f %10.2f %10.2f %10.2f %10.2f %10.2f\n",
        R, N11m, N11, N22m, N22, pR / 2)
end
println("  book BEM vs FEM in Tables 9.1–9.2 already agree to ~0.3% (N) / ~2% (M).")
println("  No circular FSDT-shell BEM in this tree (square Kirchhoff ShallowShell only).")

# ===========================================================================
# 10.5.1  Rectangular plate, centre crack, bending + tension
# ===========================================================================
println("\n## 10.5.1  Centre crack  b/h=2, c/b=2, Mo=1, t=1, E=2.1e5, ν=0.3")
println("  Book Table 10.1  K1b/(Mo √(πa))  [Dirgantara 2002 Reissner DBEM]")
@printf("  %6s %12s %12s %8s\n", "a/b", "BEM book", "Ref [2]", "err %")
for (ab, kb, kr, e) in (
        (0.1, 0.993, 0.995, 0.20),
        (0.2, 0.992, 0.990, 0.20),
        (0.4, 0.845, 0.850, 0.59),
        (0.6, 0.095, 0.100, 0.50),
        (0.8, 0.134, 0.135, 0.74),
    )
    @printf("  %6.1f %12.3f %12.3f %8.2f\n", ab, kb, kr, e)
end
println("  a/b=0.6–0.8 drop to ~0.1 is almost certainly OCR (expect ~0.9–1.1).")
println("  Infinite Reissner F≈1 for small a/h. Membrane KI (t=σ=1) is Isida")
println("  (finite width 2b, c/b=2 ≈ infinite height). In-plane dual BEM COD")
println("  is currently high vs Griffith (not Reissner K1b).")
Wb = 1.0
@printf("  %6s %12s %12s %12s\n", "a/b", "Isida F", "KI_m /σ√(πa)", "book K1b F")
for (ab, kb) in ((0.1, 0.993), (0.2, 0.992), (0.4, 0.845), (0.6, 0.095), (0.8, 0.134))
    aa = ab * Wb
    KIana = analytical_KI_center_crack(1.0, aa; W=Wb)
    F = KIana / sqrt(π * aa)
    @printf("  %6.1f %12.4f %12.4f %12.3f\n", ab, F, F, kb)
end

# ===========================================================================
# 10.5.2  Clamped square orthotropic cracked plate (harmonic)
# ===========================================================================
println("\n## 10.5.2  Clamped square central crack, harmonic, θ=0–45°")
println("  h=25.4 mm, q=1 N/m², 24 BE + 288 RIM. Book Fig. 10.7 vs ANSYS.")
println("  Figures only (FRF + first five frequencies vs fibre angle).")
println("  No cracked-plate modal solver in BEM.Plate (isotropic ThinPlate only).")

# ===========================================================================
# 10.5.3  Double-curved laminated shell with central crack, pulse
# ===========================================================================
println("\n## 10.5.3  SS? encastré spherical [0/90]s, central crack 2a, p(t) pulse")
println("  κ=1/100, a/h=100, p0/E22=500, Δt=1e-4 s, tmax=5e-3 s.")
p0 = 500 * E2
@printf("  p0 = 500 E22 = %.4e  (static FSDT w_c at q=p0 would be %.3e)\n",
    p0, ctr.w * p0)
println("  Book Fig. 10.8: RIM 9/18/32/64 pts vs FEM 1580 shells; 0.01% at 64 pts.")
println("  No cracked FSDT-shell DBEM here.")

println("\n## What this tree can / cannot BEM")
println("  Ch.9: LaminatedShell (Wang plate + Lekhnitskii membrane, DIBEM for")
println("  curvature and inertia). Static 9.6.1 / 9.6.2: usech_96_dibem.jl.")
println("  Dynamics (9.2)/(9.5)/(9.17) Houbolt: usech_96_houbolt.jl.")
println("  Ch.10 Reissner DBEM: scripts/plates/usech_1051.jl (same twins as in-plane dual).")
println("  MATLAB: Static_Thick_Shell (Ch.9), Static_Thick_Cracked_Plate (Ch.10)")
println("  in https://github.com/jfuseche/BEM_Plate_Shell_Book_Juseche")
println("Done.")
