using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf, Statistics, StaticArrays
using BEM.Plate
using FastGaussQuadrature: gausslegendre
const TP = BEM.Plate.ThinPlate

E, ν, h = 2.1e5, 0.3, 0.5
props = ThinPlateProps(; E=E, ν=ν, h=h)
mesh = build_rect_plate_crack(; W=1.0, H=2.0, a=0.2, props=props, Mo=1.0,
    ndiv_b=4, ndiv_h=4, ndiv_crack=6, nome="diag_sqrt_n")
tips = TP._plate_geometric_tips(mesh)
poly = mesh.element_type
eq = mesh.eq_type

println("=== √ρ shape identity on tip elements ===")
let n_tip = 0
    for el in mesh.elements
        eq[el.index[1]] in (2, 3) || continue
        tip = TP._el_tip(el, tips)
        tip === nothing && continue
        n_tip += 1
        qsi, _ = gausslegendre(length(el.index))
        Ierr = 0.0
        for (k, ξ) in enumerate(qsi)
            Nw, Nθ = TP._plate_Nwt(el, poly, ξ, tip)
            Ierr = max(Ierr, abs(Nθ[k] - 1), maximum(abs, Nθ) - 1)
            for j in eachindex(Nθ)
                j == k || (Ierr = max(Ierr, abs(Nθ[j]), abs(Nw[j])))
            end
            Ierr = max(Ierr, abs(Nw[k] - 1))
        end
        sN = [sqrt(norm(mesh.nodes[j] - tip)) for j in el.index]
        ξm = 0.0
        _, Nθm = TP._plate_Nwt(el, poly, ξm, tip)
        pg, _, _ = TP.elem_geom(el, ξm)
        s_mid = sqrt(norm(pg - tip))
        s_interp = dot(Nθm, sN)
        @printf("  el nodes=%s  Ierr=%.2e  s_mid=%.4f interp=%.4f rel=%.2e\n",
            string(el.index), Ierr, s_mid, s_interp, abs(s_interp - s_mid) / max(s_mid, 1e-15))
    end
    println("  n_tip_els = ", n_tip)
end
flush(stdout)

println("=== Dual assemble + Δθ profile ===")
assemble_plate_dual!(mesh; npg=6, nsub=4)
solve_plate!(mesh)
K1b, _, _, _, _ = sif_ctod_plate(mesh; tip=:right, method=:band)
K1t, _, _, _, _ = sif_ctod_plate(mesh; tip=:right, method=:tip)
@printf("  Dual band F=%.4f  tip F=%.4f  (Sih 1)\n", abs(K1b) / sqrt(0.2), abs(K1t) / sqrt(0.2))
flush(stdout)

xm = maximum(p[1] for p in tips)
cands = [p for p in tips if abs(p[1] - xm) < 1e-9]
tippos = cands[argmax(p[2] for p in cands)]
Cθ = TP._Cθ_plate(props)
println("  ρ/a     Δθ          F(ρ)")
for i in 1:length(mesh.nodes)
    eq[i] == 2 || continue
    ρ = norm(mesh.nodes[i] - tippos)
    ρ < 1e-14 && continue
    Δθ = TP._dtheta_e2(mesh, i, SVector(0.0, 1.0))
    Fρ = Cθ * Δθ / sqrt(ρ) / sqrt(0.2)
    @printf("  %.4f  %11.4e  %8.3f\n", ρ / 0.2, Δθ, Fρ)
end
flush(stdout)

println("=== XBEM extra ===")
Hε, Cu, Cε, tps, _ = assemble_plate_xbem!(mesh; n_enr=2, npg=6, nsub=4, n_v=3)
A, b, _, _ = TP.apply_bc_plate(mesh)
@printf("  ||A||∞=%.3e  ||Hε||∞=%.3e  ratio=%.3e  cond(Cε)=%.3e\n",
    opnorm(A, Inf), opnorm(Hε, Inf), opnorm(Hε, Inf) / max(opnorm(A, Inf), 1e-30),
    cond(Cε))
flush(stdout)
_, K1x, K2x, tps, _ = solve_plate_xbem!(mesh; n_enr=2, npg=6, nsub=4, n_v=3)
den = sqrt(0.2)
for (k, p) in enumerate(tps)
    @printf("  tip %d (%.3f,%.3f)  F1=%.4f  F2=%.4f\n",
        k, p[1], p[2], abs(K1x[k]) / den, abs(K2x[k]) / den)
end
ir = argmax(p[1] for p in tps)
@printf("  extra F1=%.4f  F2=%.4f  (right)  K1=%s  K2=%s\n",
    abs(K1x[ir]) / den, abs(K2x[ir]) / den, string(K1x), string(K2x))
flush(stdout)
println("DONE")
