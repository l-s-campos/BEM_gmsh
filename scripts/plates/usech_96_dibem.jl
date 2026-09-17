# Useche 9.6 — Wang FSDT + anisotropic membrane, DIBEM curvature (Ch.9).
# MATLAB twin: Static_Thick_Shell coupling (Hpw / Hu / Hsw) with RIM → DIBEM.
# 9.6.1 SS spherical [0/90]s; 9.6.2 clamped circular (straight-chord).
# Dynamics (Houbolt + DIBEM mass): scripts/plates/usech_96_houbolt.jl.
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf, StaticArrays
using BEM.Plate

function navier_fsdt_kmem(p, kmem; a, q, nterms=40)
    D11, D22, D12, D66 = p.D[1, 1], p.D[2, 2], p.D[1, 2], p.D[3, 3]
    A44, A55 = p.AT[1, 1], p.AT[2, 2]
    w = 0.0
    @inbounds for m in 1:2:nterms, n in 1:2:nterms
        α = m * π / a
        β = n * π / a
        K11 = D11 * α^2 + D66 * β^2 + A55
        K12 = (D12 + D66) * α * β
        K13 = A55 * α
        K22 = D66 * α^2 + D22 * β^2 + A44
        K23 = A44 * β
        K33 = A55 * α^2 + A44 * β^2 + kmem
        Δ = [K11 K12 K13; K12 K22 K23; K13 K23 K33] \ [0.0, 0.0, 16q / (π^2 * m * n)]
        w += Δ[3] * sin(α * a / 2) * sin(β * a / 2)
    end
    return w
end

println("="^72)
println(" Useche 9.6  laminated shallow shells — Wang FSDT + DIBEM")
println("="^72)

# ===========================================================================
# 9.6.1  SS spherical [0/90]s  κ=1/100, a/h=100, q=1
# Scaled a=1, h=0.01, R=1  (same a/h, a/R as a=100, h=1, R=100).
# ===========================================================================
println("\n## 9.6.1  SS spherical [0/90]s")
E1, E2, ν12 = 25.0, 1.0, 0.25
G12 = G13 = 0.5 * E2
G23 = 0.2 * E2
a, h, R, q = 1.0, 0.01, 1.0, 1.0
κ = 1 / R
plies = [(E1, E2, ν12, G12, θ, h / 4) for θ in (0.0, 90.0, 90.0, 0.0)]
props = laminate_fsdt_props(plies; Ks=5 / 6, G13=G13, G23=G23, q_c=q, ρ=1.0)
A, Bmat, D, AT, _ = BEM.Plate._laminate_ABD_AT(plies; Ks=5 / 6, G13=G13, G23=G23)
As = @SMatrix [AT[2, 2] AT[1, 2]; AT[1, 2] AT[1, 1]]   # [A55 A45; A45 A44] for Navier
kmem = (A[1, 1] + 2 * A[1, 2] + A[2, 2]) * κ^2
@printf("  a=%.2f  h=%.3f  R=%.2f  a/h=%.0f  a/R=%.2f\n", a, h, R, a / h, a / R)
@printf("  A11=%.4e  A22=%.4e  A12=%.4e  D11=%.4e  kmem=%.4e\n",
    A[1, 1], A[2, 2], A[1, 2], D[1, 1], kmem)

ctr = navier_ss_laminate_shell(a / 2, a / 2; a=a, q=q, κ1=κ, κ2=κ, A=A, D=D, As=As)
flat = navier_ss_laminate_shell(a / 2, a / 2; a=a, q=q, κ1=0, κ2=0, A=A, D=D, As=As)
w3 = navier_fsdt_kmem(props, kmem; a=a, q=q)
nd(w) = w * E2 * h^3 / (q * a^4)
@printf("  series 5-DOF Navier  flat w_c=%.6e  nd×10³=%.4f\n", flat.w, nd(flat.w) * 1e3)
@printf("  series 5-DOF Navier  sph  w_c=%.6e  nd×10³=%.4f  flat/sph=%.3f\n",
    ctr.w, nd(ctr.w) * 1e3, flat.w / ctr.w)
@printf("  series 3-DOF+kmem    sph  w_c=%.6e  (vs 5-DOF %.2f %%)\n",
    w3, 100 * abs(w3 - ctr.w) / abs(ctr.w))
@printf("  Nx=%.4e  Ny=%.4e  Mx=%.4e  My=%.4e\n", ctr.Nx, ctr.Ny, ctr.Mx, ctr.My)

n_el, n_int = 4, 81
println("\n  coupled Wang+membrane DIBEM  n_el=$n_el (16 BE, book)  n_int=$n_int (~MATLAB 98 RIM)")
mesh = build_square_fsdt(; a=a, n_el=n_el, bc="SSSS", props=props, n_internal=n_int)
shell = LaminatedShell(mesh, A, κ, κ; mem_bc=:navier_ss)
assemble_laminated_shell!(shell; npg=8, nsub=6)
# flat: same operators with κ=0 (re-assemble plate-only for the flat check)
mesh_f = build_square_fsdt(; a=a, n_el=n_el, bc="SSSS", props=props, n_internal=n_int)
assemble_fsdt!(mesh_f; npg=8, nsub=6)
dibem_fsdt!(mesh_f)
solve_fsdt!(mesh_f)
w_flat = fsdt_w_int(mesh_f, 1)
@printf("  DIBEM FSDT flat w_c=%.6e  rel 5-DOF flat %.2f %%\n",
    w_flat, 100 * abs(w_flat - flat.w) / abs(flat.w))

solve_laminated_shell!(shell)
w_sph = fsdt_w_int(mesh, 1)
res = shell_resultants(shell; method=:ss1)
nb = length(mesh.nodes)
ic = nb + 1
@printf("  DIBEM coupled 5-DOF w_c=%.6e  rel 5-DOF sph %.2f %%  rel 3-DOF+kmem %.2f %%\n",
    w_sph, 100 * abs(w_sph - ctr.w) / abs(ctr.w), 100 * abs(w_sph - w3) / abs(w3))
@printf("  BEM flat/sph=%.3f  series 5-DOF flat/sph=%.3f\n",
    w_flat / w_sph, flat.w / ctr.w)
@printf("  BEM centre Nx=%.4e  Ny=%.4e  Mx=%.4e  My=%.4e\n",
    res.Nx[ic], res.Ny[ic], res.Mx[ic], res.My[ic])
@printf("  rel Nx %.1f %%  Ny %.1f %%  Mx %.1f %%  My %.1f %%\n",
    100 * abs(res.Nx[ic] - ctr.Nx) / max(abs(ctr.Nx), 1e-30),
    100 * abs(res.Ny[ic] - ctr.Ny) / max(abs(ctr.Ny), 1e-30),
    100 * abs(res.Mx[ic] - ctr.Mx) / max(abs(ctr.Mx), 1e-30),
    100 * abs(res.My[ic] - ctr.My) / max(abs(ctr.My), 1e-30))

println("\n  Fig. 9.3  N, M along y=a/2 (x1 through centre) vs 19-term Navier")
@printf("  %8s %12s %12s %8s %12s %12s %8s %12s %12s %8s %12s %12s %8s\n",
    "x", "Nx BEM", "Nx ser", "e%", "Ny BEM", "Ny ser", "e%",
    "Mx BEM", "Mx ser", "e%", "My BEM", "My ser", "e%")
cl = shell_centreline(shell; dir=:x)
for i in eachindex(cl.s)
    g = navier_ss_laminate_shell(cl.pts[i][1], cl.pts[i][2];
        a=a, q=q, κ1=κ, κ2=κ, A=A, D=D, As=As)
    e(b, t) = 100 * abs(b - t) / max(abs(t), 1e-30)
    @printf("  %8.3f %12.4e %12.4e %7.1f %12.4e %12.4e %7.1f %12.4e %12.4e %7.1f %12.4e %12.4e %7.1f\n",
        cl.s[i], cl.Nx[i], g.Nx, e(cl.Nx[i], g.Nx),
        cl.Ny[i], g.Ny, e(cl.Ny[i], g.Ny),
        cl.Mx[i], g.Mx, e(cl.Mx[i], g.Mx),
        cl.My[i], g.My, e(cl.My[i], g.My))
end

# ===========================================================================
# 9.6.2  Clamped circular  a=5, h=0.1, p=pmax(1+r²), [90/0/90/90/0]
# Book Tables 9.1–9.2 are N,M at r=a/2 (no w table).
# ===========================================================================
println("\n## 9.6.2  Clamped circular [90/0/90/90/0]  p=pmax(1+r²)")
ac, hc, pmax = 5.0, 0.1, 1.0
angs = [90.0, 0.0, 90.0, 90.0, 0.0]
plies2 = [(E1, E2, ν12, G12, θ, hc / 5) for θ in angs]
A2, _, D2, AT2, _ = BEM.Plate._laminate_ABD_AT(plies2; Ks=5 / 6, G13=G12, G23=G12)
@printf("  a=%.1f  h=%.2f  A11=%.3f  A22=%.3f  D11=%.4f  D22=%.4f\n",
    ac, hc, A2[1, 1], A2[2, 2], D2[1, 1], D2[2, 2])
println("  16-chord circle, n_int=81, coupled DIBEM (book 16 curved BE + 120 RIM)")

@printf("  %8s %12s %12s %12s %12s\n", "R", "w_c", "Nx(a/2)", "Mx(a/2)", "kmem")
for Rv in (20.0, 50.0, 100.0)
    κ2 = 1 / Rv
    kmem2 = (A2[1, 1] + 2 * A2[1, 2] + A2[2, 2]) * κ2^2
    props2 = laminate_fsdt_props(plies2; Ks=5 / 6, G13=G12, G23=G12, q_c=pmax, ρ=1.0)
    mesh2 = build_circle_fsdt(; R=ac, n_el=16, bc='C', props=props2, n_internal=120)
    shell2 = LaminatedShell(mesh2, A2, κ2, κ2; mem_bc=:clamped)
    assemble_laminated_shell!(shell2; npg=8, nsub=6)
    I0 = props2.ρ * props2.h
    pts = [mesh2.nodes; mesh2.internal]
    qv = zeros(length(mesh2.q))
    @inbounds for j in eachindex(pts)
        pj = pmax * (1 + (pts[j][1]^2 + pts[j][2]^2))
        qv .+= mesh2.M[:, 3j] .* (pj / I0)
    end
    mesh2.q = qv
    solve_laminated_shell!(shell2)
    wc = fsdt_w_int(mesh2, 1)
    r2 = shell_resultants(shell2)
    ih = argmin(i -> abs(hypot(r2.pts[i][1], r2.pts[i][2]) - ac / 2),
        eachindex(r2.pts))
    @printf("  %8.0f %12.4e %12.4e %12.4e %12.4e\n",
        Rv, wc, r2.Nx[ih], r2.Mx[ih], kmem2)
end
println("  book Tables 9.1–9.2: N,M at r=a/2 (R as radius).")
println("Done.")
