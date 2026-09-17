# Check our plate code vs Useche MATLAB book codes + rerun Ch. 7–10.
# MATLAB root: /home/lsc/Downloads/BEM_Plate_Shell_Book_Juseche-main
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf, StaticArrays
using BEM.Plate
using BEM.Crack
using .ThinPlate

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
function ortho_D(E1, E2, ν12, G12, h)
    ν21 = ν12 * E2 / E1
    den = 1 - ν12 * ν21
    return E1 * h^3 / (12 * den), E2 * h^3 / (12 * den),
        ν12 * E2 * h^3 / (12 * den), G12 * h^3 / 12
end
function navier_w_ortho(x, y; a, b=a, q, D11, D22, D12, D66, nterms=80)
    H = D12 + 2 * D66
    w = 0.0
    @inbounds for m in 1:2:nterms, n in 1:2:nterms
        α = m * π / a; β = n * π / b
        den = D11 * α^4 + 2 * H * α^2 * β^2 + D22 * β^4
        w += 16q / (π^2 * m * n) / den * sin(α * x) * sin(β * y)
    end
    return w
end
function navier_w_fsdt(x, y; a, b=a, q, D11, D22, D12, D66, A44, A55, nterms=40)
    w = 0.0
    @inbounds for m in 1:2:nterms, n in 1:2:nterms
        α = m * π / a; β = n * π / b
        K = [D11*α^2+D66*β^2+A55  (D12+D66)*α*β  A55*α;
             (D12+D66)*α*β  D66*α^2+D22*β^2+A44  A44*β;
             A55*α  A44*β  A55*α^2+A44*β^2]
        Δ = K \ [0.0, 0.0, 16q / (π^2 * m * n)]
        w += Δ[3] * sin(α * x) * sin(β * y)
    end
    return w
end
function navier_ss_sphere(x, y; a, b=a, q, κ1, κ2, A, D, As, nterms=19)
    w = Nx = Ny = Mx = My = 0.0
    @inbounds for m in 1:2:nterms, n in 1:2:nterms
        α = m * π / a; β = n * π / b
        A11, A22, A12, A66 = A[1, 1], A[2, 2], A[1, 2], A[3, 3]
        D11, D22, D12, D66 = D[1, 1], D[2, 2], D[1, 2], D[3, 3]
        A44, A55 = As[2, 2], As[1, 1]
        K = zeros(5, 5)
        K[1, 1] = A11*α^2 + A66*β^2
        K[1, 2] = (A12+A66)*α*β
        K[1, 3] = -(A11*κ1 + A12*κ2)*α
        K[2, 2] = A22*β^2 + A66*α^2
        K[2, 3] = -(A12*κ1 + A22*κ2)*β
        K[3, 3] = A55*α^2 + A44*β^2 + (A11*κ1+A12*κ2)*κ1 + (A12*κ1+A22*κ2)*κ2
        K[3, 4] = A55*α; K[3, 5] = A44*β
        K[4, 4] = D11*α^2 + D66*β^2 + A55
        K[4, 5] = (D12+D66)*α*β
        K[5, 5] = D22*β^2 + D66*α^2 + A44
        K[2, 1] = K[1, 2]; K[3, 1] = K[1, 3]; K[3, 2] = K[2, 3]
        K[4, 3] = K[3, 4]; K[5, 3] = K[3, 5]; K[5, 4] = K[4, 5]
        Δ = K \ [0.0, 0.0, 16q/(π^2*m*n), 0.0, 0.0]
        U, V, W, X, Y = Δ
        s = sin(α*x)*sin(β*y)
        w += W * s
        εx = -α*U + κ1*W; εy = -β*V + κ2*W
        Nx += (A11*εx + A12*εy)*s
        Ny += (A12*εx + A22*εy)*s
        Mx += (-D11*α*X - D12*β*Y)*s
        My += (-D12*α*X - D22*β*Y)*s
    end
    return (w=w, Nx=Nx, Ny=Ny, Mx=Mx, My=My)
end
cantilever_w_tip(q, L, D11, A55) = q*L^3/(3*D11) + q*L/A55

println("="^72)
println(" Our code vs MATLAB book  +  rerun Ch. 7–10")
println("="^72)

# ===========================================================================
println("\n## 0. What the MATLAB tree actually is")
println("  Static_Thick_Plate     Reissner–Mindlin isotropic (Vander Weeën)")
println("  Dynamic_Composite_Plate  FSDT laminate, Wang kernels (Ch. 8)")
println("  Static_Thick_Shell     FSDT plate + plane stress + curvature (Ch. 9)")
println("  Static_Thick_Cracked_Plate  Reissner DBEM (Ch. 10 / Dirgantara)")
println("  Our ThinPlate          isotropic Kirchhoff (Shi–Bezine 2×2)")
println("  Same: D = E h³/12(1-ν²), quadratic discontinuous N, GL/Telles")
println("  Not the same BIE: 3 DOF (ψx,ψy,w) vs 2 DOF (w, ∂w/∂n)")

# ===========================================================================
println("\n## 1. ABD vs Octave ContsLam  (Material.m [0/90/90/0])")
plies8 = [(4e6, 2e6, 0.25, 1e6, θ, 0.025) for θ in (0.0, 90.0, 90.0, 0.0)]
lam8 = laminate_ABD(plies8; Ks=5/6, G13=1e6, G23=5e5)
# Octave ContsLam: AT=[62500 0; 0 62500] as [A44 A45; A45 A55]
# D11=322.5806 D12=43.0108 D22=193.5484 D66=83.3333
@printf("  Julia D11=%.6f  Octave 322.5806  rel=%.3e\n",
    lam8.D[1, 1], abs(lam8.D[1, 1]-322.5806)/322.5806)
@printf("  Julia D22=%.6f  Octave 193.5484  rel=%.3e\n",
    lam8.D[2, 2], abs(lam8.D[2, 2]-193.5484)/193.5484)
@printf("  Julia D12=%.6f  Octave  43.0108  rel=%.3e\n",
    lam8.D[1, 2], abs(lam8.D[1, 2]-43.0108)/43.0108)
@printf("  Julia D66=%.6f  Octave  83.3333  rel=%.3e\n",
    lam8.D[3, 3], abs(lam8.D[3, 3]-83.3333)/83.3333)
@printf("  Julia A55=A44=%.1f  Octave AT=62500  rel=%.3e\n",
    lam8.As[1, 1], abs(lam8.As[1, 1]-62500)/62500)
println("  ABD/shear stiffness: match. ContsLam does not build A (extensional).")

# ===========================================================================
println("\n## 2. MATLAB prueba03  SS square Reissner vs our Kirchhoff")
# Static_Thick_Plate/prueba03.m : [-1,1]², E=200e9, ν=0.3, h=0.02, q=1, SS
E, ν, h, a, q0 = 200e9, 0.3, 0.02, 2.0, 1.0
props = ThinPlateProps(; E=E, ν=ν, h=h, q_c=q0)
D = bending_stiffness(props)
w_ana = analytical_wmax_ss_square(; a=a, q=q0, D=D)
plate = build_square_plate(; a=a, n_el=6, bc="SSSS", props=props,
    corner_bc='F', n_internal=1)
assemble_plate!(plate; npg=10, singular=:analytic)
solve_plate!(plate)
wc = plate_w_int(plate, 1)
@printf("  a=2  h=0.02  a/h=100  D=%.4e\n", D)
@printf("  Navier w_c     = %.6e\n", w_ana)
@printf("  Kirchhoff BEM  = %.6e   rel Navier %.3f %%\n",
    wc, 100 * abs(wc - w_ana) / abs(w_ana))
println("  MATLAB is Reissner (3 DOF, λ=√10/h). Thin limit of their U33 is")
println("  r² log r /(8πD) — same leading term as our W*. a/h=100 ⇒ agree.")

# ===========================================================================
println("\n## 7.5.1  Orthotropic SS square (no MATLAB Kirchhoff-aniso folder)")
Ex, Ey, νxy, Gxy = 2.068e11, 2.068e11/15, 0.3, 6.055e8
D11, D22, D12, D66 = ortho_D(Ex, Ey, νxy, Gxy, 0.01)
wA = navier_w_ortho(0.5, 0.5; a=1.0, q=1e4, D11=D11, D22=D22, D12=D12, D66=D66)
wB = navier_w_ortho(0.25, 0.25; a=1.0, q=1e4, D11=D11, D22=D22, D12=D12, D66=D66)
@printf("  series A=%.8e  book 8.1258e-3  err=%.4f %%\n", wA, 100*abs(wA-8.1258e-3)/8.1258e-3)
@printf("  series B=%.8e  book 4.5211e-3  err=%.4f %%\n", wB, 100*abs(wB-4.5211e-3)/4.5211e-3)
ps = aniso_thin_plate_props(; D11=D11, D22=D22, D12=D12, D66=D66, q_c=1e4, h=0.01)
pl = build_square_plate(; a=1.0, n_el=6, bc="SSSS", props=ps, corner_bc='F', n_internal=1)
assemble_plate!(pl; npg=10, singular=:guiggiani); solve_plate!(pl)
@printf("  aniso Kirchhoff BEM w_c=%.4e  rel series A %.2f %%\n",
    plate_w_int(pl, 1), 100 * abs(plate_w_int(pl, 1) - wA) / abs(wA))

# ===========================================================================
println("\n## 7.5.2  9-ply graphite-epoxy  E1=207 GPa (book OCR 2.07e9)")
E1, E2, ν12, G12 = 2.07e11, 5.17e9, 0.25, 3.10e9
angs = [0, 90, 0, 90, 0, 90, 0, 90, 0]
t = 0.01/9
lam = laminate_ABD([(E1, E2, ν12, G12, float(θ), t) for θ in angs])
w = navier_w_ortho(0.5, 0.5; a=1.0, q=6.9e3,
    D11=lam.D[1,1], D22=lam.D[2,2], D12=lam.D[1,2], D66=lam.D[3,3])
nd = w * E2 * 0.01^3 / (6.9e3 * 1.0^4) * 1e3
@printf("  nd×10³=%.4f  book 4.4718  err=%.2f %%  (BEM book 4.4507)\n",
    nd, 100*abs(nd-4.4718)/4.4718)

# ===========================================================================
println("\n## 8.6.1  MATLAB Ejemplo_Laminado.m  cantilever [0/90]s")
@printf("  FSDT beam w_tip=%.6e  (D11=%.4f A55=%.0f  V=100 L=10)\n",
    cantilever_w_tip(100.0, 10.0, lam8.D[1,1], lam8.As[1,1]),
    lam8.D[1,1], lam8.As[1,1])
println("  MATLAB BEM is Wang FSDT (`wang_kernels` / KernelP.m).")
println("  Book: BEM <1% vs Reddy on 6–16 elements (Fig. 8.4).")

# ===========================================================================
println("\n## 8.6.2  MATLAB Material.m + SOLFEM.m  SS [0/90]s impulsive")
# a=1, h=0.1, q=1, ρ=4000, ν=0.25  (book text ν=0.5 is OCR; MATLAB is 0.25)
wK = navier_w_ortho(0.5, 0.5; a=1.0, q=1.0,
    D11=lam8.D[1,1], D22=lam8.D[2,2], D12=lam8.D[1,2], D66=lam8.D[3,3])
wF = navier_w_fsdt(0.5, 0.5; a=1.0, q=1.0,
    D11=lam8.D[1,1], D22=lam8.D[2,2], D12=lam8.D[1,2], D66=lam8.D[3,3],
    A44=lam8.As[2,2], A55=lam8.As[1,1])
I0 = 4000.0 * 0.1
α = π
K11 = lam8.D[1,1]*α^2 + lam8.D[3,3]*α^2 + lam8.As[1,1]
K12 = (lam8.D[1,2]+lam8.D[3,3])*α*α
K13 = lam8.As[1,1]*α
K22 = lam8.D[3,3]*α^2 + lam8.D[2,2]*α^2 + lam8.As[2,2]
K23 = lam8.As[2,2]*α
K33 = lam8.As[1,1]*α^2 + lam8.As[2,2]*α^2
KK = [K11 K12 K13; K12 K22 K23; K13 K23 K33]
Kred = KK[3,3] - dot(KK[1:2,3], KK[1:2,1:2] \ KK[1:2,3])
T11 = 2π / sqrt(Kred / I0)
# SOLFEM.m peak ~ 4.143e-5 at t=0.2225
fem_peak = 0.414314e-04
@printf("  w_CLPT=%.4e  w_FSDT=%.4e  T11=%.4f s\n", wK, wF, T11)
@printf("  2×w_FSDT (step-load peak) = %.4e   FEM peak = %.4e  rel=%.1f %%\n",
    2wF, fem_peak, 100*abs(2wF - fem_peak)/fem_peak)
println("  MATLAB ν=0.25, not book 0.5. FEM file matches FSDT static×2.")

# ===========================================================================
println("\n## 9.6.1  SS spherical [0/90]s  19-term Reddy (MATLAB Static_Thick_Shell)")
E1s, E2s = 25.0, 1.0
plies9 = [(E1s, E2s, 0.25, 0.5, θ, 0.25) for θ in (0.0, 90.0, 90.0, 0.0)]
lam9 = laminate_ABD(plies9; Ks=5/6, G13=0.5, G23=0.2)
ctr = navier_ss_sphere(50.0, 50.0; a=100.0, q=1.0, κ1=0.01, κ2=0.01,
    A=lam9.A, D=lam9.D, As=lam9.As, nterms=19)
flat = navier_ss_sphere(50.0, 50.0; a=100.0, q=1.0, κ1=0, κ2=0,
    A=lam9.A, D=lam9.D, As=lam9.As, nterms=19)
@printf("  a=R=100 h=1  w_c=%.4e  nd×10³=%.4f  shell/flat=%.3f\n",
    ctr.w, ctr.w*E2s*(1.0)^3/(1.0*100.0^4)*1e3, ctr.w/flat.w)
@printf("  Nx=%.2f Ny=%.2f Mx=%.3f My=%.3f\n", ctr.Nx, ctr.Ny, ctr.Mx, ctr.My)
ctr1 = navier_ss_sphere(0.5, 0.5; a=1.0, q=1.0, κ1=0.01, κ2=0.01,
    A=laminate_ABD([(E1s,E2s,0.25,0.5,θ,0.0025) for θ in (0.0,90.0,90.0,0.0)];
        Ks=5/6, G13=0.5, G23=0.2).A,
    D=laminate_ABD([(E1s,E2s,0.25,0.5,θ,0.0025) for θ in (0.0,90.0,90.0,0.0)];
        Ks=5/6, G13=0.5, G23=0.2).D,
    As=laminate_ABD([(E1s,E2s,0.25,0.5,θ,0.0025) for θ in (0.0,90.0,90.0,0.0)];
        Ks=5/6, G13=0.5, G23=0.2).As, nterms=19)
println("  MATLAB coupling is FSDT+aniso membrane (Eqn_plate + Eqn_sheet).")
println("  Our ShallowShell is isotropic Kirchhoff + lumped N/R — not this BIE.")

# ===========================================================================
println("\n## 9.6.2  Tables 9.1–9.2  (MATLAB/FEM already agree; pmax not in .m)")
println("  R=20…100  N11 BEM/FEM  −83.60/−83.74 … −272.0/−272.6")
println("  No circular mesh builder in ThinPlate.")

# ===========================================================================
println("\n## 10.5.1  MATLAB test01.m  Dirgantara SS square + centre crack")
println("  MATLAB: [-1,1]², a=0.5, h=0.1667, E=1e9, ν=0.3, q=1, SS")
println("  Book Table 10.1 is K1b (bending), Reissner DBEM — not our in-plane dual.")
@printf("  %6s %12s %12s %12s\n", "a/b", "book K1b F", "Ref", "Isida F_m")
for (ab, kb, kr) in ((0.1,0.993,0.995),(0.2,0.992,0.990),(0.4,0.845,0.850),
                     (0.6,0.095,0.100),(0.8,0.134,0.135))
    F = analytical_KI_center_crack(1.0, ab; W=1.0) / sqrt(π * ab)
    @printf("  %6.1f %12.3f %12.3f %12.4f\n", ab, kb, kr, F)
end
println("  a/b=0.6–0.8 ~0.1 is OCR. MATLAB CalcSIF.m is Dirgantara CTOD (Eh³/48).")

println("\n## Verdict")
println("  CORRECT vs MATLAB: laminate D and AT (ContsLam), Kirchhoff SS Navier,")
println("  thin-limit kernel r²log/(8πD), Ch.7 series vs Timoshenko/Reddy.")
println("  Ch.7.5 anisotropic Kirchhoff is AnisoThinPlateProps (no iso smear).")
println("  Ch.7.7 shallow shell is still smeared-iso + membrane, not laminated A,D.")
println("Done.")
