# Compare SST lead tensors to Cordeiro eqs. 23–24, 35, 37 and to ρ→0 of the kernel.
using DrWatson
@quickactivate :BEM
using LinearAlgebra, StaticArrays, Printf

pars = lekhnitskii_engineering(124.04, 10.09, 6.03, 0.334; η12_1=1.255, η12_2=-0.031)
props = AnisotropicElasticity(pars)
p = pars
A, q, g, C, μ = p.A, p.q, p.g, p.C, p.mi
XT = transpose

function paper_DS_lead(n)
    # R = [1 1; μ1 μ2]  (eq. 24)
    # U_{lj,m} = 2 Re[ R_{m p} q_{j p} A_{l p} / z_p ]
    # freeze 1/z_p → 1/α_p so that D_actual = D_lead / ((ξ-ξ0) J0)
    # T_{lj,m} = -2 Re[ R_{m p} g_{j p} α_p A_{l p} / z_p² ]
    # freeze 1/z² → 1/α², but α/z² = 1/(z²/α) = 1/(δ² J² α) so lead uses 1/α
    α1 = μ[1] * n[1] - n[2]
    α2 = μ[2] * n[1] - n[2]
    # FIELD derivatives (eqs 23–24). Traction BIE wants ∇_source = −∇_field.
    # U_field,x = +2 Re(A * (1/z) q^T) → source ux = −that.
    # After freeze 1/z = 1/(δ J α), lead of source ux is −2 Re(A (1/α) q^T).
    invα = @SMatrix [1/α1 0; 0 1/α2]
    μinvα = @SMatrix [μ[1]/α1 0; 0 μ[2]/α2]
    # eq 24 FIELD: U_,1 = 2 Re(A invz q^T), U_,2 = 2 Re(A μ/z q^T)
    Ux_field = 2 * real(A * invα * XT(q))
    Uy_field = 2 * real(A * μinvα * XT(q))
    # eq 23 FIELD: T_,m = −2 Re(R_m g α A / z²); α/z² lead 1/α after / (δ² J²)
    # T_field,1 = −2 Re(A (α/z²) g^T) with α/z² → 1/(δ²J² α)
    Tx_field = -2 * real(A * invα * XT(g))
    Ty_field = -2 * real(A * μinvα * XT(g))
    # source derivatives (what we need): minus field
    ux, uy = -Ux_field, -Uy_field
    px, py = -Tx_field, -Ty_field
    D1 = C * SVector(ux[1, 1], uy[2, 1], uy[1, 1] + ux[2, 1])
    D2 = C * SVector(ux[1, 2], uy[2, 2], uy[1, 2] + ux[2, 2])
    S1 = C * SVector(px[1, 1], py[2, 1], py[1, 1] + px[2, 1])
    S2 = C * SVector(px[1, 2], py[2, 2], py[1, 2] + px[2, 2])
    n1, n2 = n[1], n[2]
    # η_i D_ijk  with D_ijk = σ_ik due to force j   (paper C_iklm, free j)
    # and η_i D_kij with D_kij = σ_ij due to force k (our hyper)
    Uh_paper = @SMatrix [  # M_jk = η_i D_ijk = t_k^(j)  → row j (force), col k (traction)
        n1 * D1[1] + n2 * D1[3]   n1 * D1[3] + n2 * D1[2]
        n1 * D2[1] + n2 * D2[3]   n1 * D2[3] + n2 * D2[2]
    ]
    Uh_ours = @SMatrix [   # U^h_ik = t_i^(k)  row traction, col force
        n1 * D1[1] + n2 * D1[3]   n1 * D2[1] + n2 * D2[3]
        n1 * D1[3] + n2 * D1[2]   n1 * D2[3] + n2 * D2[2]
    ]
    Th_ours = @SMatrix [
        n1 * S1[1] + n2 * S1[3]   n1 * S2[1] + n2 * S2[3]
        n1 * S1[3] + n2 * S1[2]   n1 * S2[3] + n2 * S2[2]
    ]
    Th_paper = @SMatrix [
        n1 * S1[1] + n2 * S1[3]   n1 * S1[3] + n2 * S1[2]
        n1 * S2[1] + n2 * S2[3]   n1 * S2[3] + n2 * S2[2]
    ]
    return Uh_ours, Th_ours, Uh_paper, Th_paper
end

n = SVector(0.0, -1.0)
Uo, To, Up, Tp = paper_DS_lead(n)
Ul, Tl = BEM._lekh_Uh_Th_lead(p, n)
println("=== straight n=(0,-1) lead tensors ===")
println("Uh _lekh vs eqs23-24 source-deriv:\n  rel=$(norm(Uo-Ul)/norm(Ul))")
println("Th _lekh vs eqs23-24 source-deriv:\n  rel=$(norm(To-Tl)/norm(Tl))")
println("Uh ours (t_i due to force k):\n$Uo")
println("Uh paper-index (t_k due to force j) = ours^T:\n$Up")
println("ours^T == paper? ", Uo' ≈ Up)
println("Th ours:\n$To")
println("Th paper-index = ours^T?:\n$Tp")
println("ours^T == paper Th? ", To' ≈ Tp)

# kernel limit on a straight element
poly = BEM.Equispaced(1)
nodes = [Point2D(0.0, 0.0), Point2D(1.0, 0.0)]
a, s, ρ = 0.0, 1.0, 1e-8
N, dN = BEM.shapefun(poly, a)
J = 0.5
φ = N[1, 1]  # 0.5
pf = Point2D(0.5, 0.0)
pg = pf + Point2D(s * ρ * J, 0.0)
nrm = Point2D(0.0, -1.0)
Uh, Th = fundamental_hyper(props, pg, pf, nrm, nrm)
Uh = BEM._to_smat(Uh); Th = BEM._to_smat(Th)
Fg = Uh * (φ * J)
Fh = Th * (φ * J)
println("\n=== kernel limit vs SST K* (straight, φ=$φ, J=$J) ===")
println("eq35:  F_G* = Uh_lead φ / δ,  δ=sρ=$(s*ρ)")
println("  ρ Fg vs Uh_lekh φ: rel=$(norm(ρ*Fg - Ul*φ)/norm(Ul*φ))")
println("eq37:  F_H* = Th_lead φ / (δ² J)")
println("  ρ² Fh vs Th_lekh φ / J: rel=$(norm(ρ^2*Fh - Tl*φ/J)/norm(Tl*φ/J))")
println("  ρ² Fh vs Th_lekh φ / J² (literal (Jα)²): rel=$(norm(ρ^2*Fh - Tl*φ/J^2)/max(norm(Tl*φ/J^2),1e-30))")

# curved quadratic
polyq = BEM.Legendre(2)
θ = (0.0, π/8, π/4)
Xarc = [Point2D(cos(t), sin(t)) for t in θ]
ac = polyq.nodes[2]
Nc, dNc = BEM.shapefun(polyq, ac)
g = BEM._geom_1d(polyq, Xarc, ac)
Nrow, Jc, t, nc = g
Ul, Tl = BEM._lekh_Uh_Th_lead(p, nc)
pfc = (Nc * Xarc)[1]
ρ = 1e-8
ξ = ac + ρ
Nξ, dNξ = BEM.shapefun(polyq, ξ)
pg = (Nξ * Xarc)[1]
dx = (dNξ * Xarc)[1]
Jξ = norm(dx)
nrm = Point2D(dx[2], -dx[1]) / Jξ
Uh, Th = fundamental_hyper(props, pg, pfc, nrm, nc)
Uh = BEM._to_smat(Uh); Th = BEM._to_smat(Th)
φ0 = Nc[1, 2]  # middle node
Fg = Uh * (Nξ[1, 2] * Jξ)
Fh = Th * (Nξ[1, 2] * Jξ)
δ = ρ
println("\n=== curved R=1, mid node, J0=$Jc n=$nc ===")
println("  δ Fg vs Uh φ0: rel=$(norm(δ*Fg - Ul*φ0)/norm(Ul*φ0))")
println("  δ² Fh vs Th φ0 / J0: rel=$(norm(δ^2*Fh - Tl*φ0/Jc)/norm(Tl*φ0/Jc))")
println("  δ² Fh vs Th φ0 / J0²: rel=$(norm(δ^2*Fh - Tl*φ0/Jc^2)/max(norm(Tl*φ0/Jc^2),1e-30))")
println("done")
