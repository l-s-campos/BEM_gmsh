using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf, Statistics, StaticArrays
using BEM.Plate
const TP = BEM.Plate.ThinPlate

function report(ndiv_c; n_near=2, ρ_near=0.3, interior=false)
    # n_near: Dual HBIE rows replaced by u_reg=0 (physical = Williams)
    props = ThinPlateProps(; E=2.1e5, ν=0.3, h=0.5)
    mesh = build_rect_plate_crack(; W=1.0, H=2.0, a=0.2, props=props, Mo=1.0,
        ndiv_b=4, ndiv_h=4, ndiv_crack=ndiv_c, nome="wtie_$ndiv_c")
    assemble_plate_dual!(mesh; npg=6, nsub=4)
    solve_plate!(mesh)
    den = sqrt(0.2)
    Kb, _, _, _, _ = sif_ctod_plate(mesh; tip=:right, method=:band)
    _, K1x, K2x, tips, frames = solve_plate_xbem!(mesh; n_enr=2, npg=6, nsub=4,
        n_v=3, n_near=n_near, ρ_near=ρ_near, interior=interior)
    ir = argmax(p[1] for p in tips)
    @printf("ndiv_c=%d  Dual band=%.4f  extra F1=%.4f F2=%.4f  K1=%s\n",
        ndiv_c, abs(Kb) / den, abs(K1x[ir]) / den, abs(K2x[ir]) / den, string(K1x))
    tip = tips[ir]
    e1, e2 = frames[ir]
    Cθ = TP._Cθ_plate(props)
    eq = mesh.eq_type
    K1 = K1x[ir]
    println("  ρ/a     Δθ_phys     Δθ_W(K)    Δθ/Δθ_W")
    for i in 1:length(mesh.nodes)
        eq[i] == 2 || continue
        ρ = norm(mesh.nodes[i] - tip)
        ρ < 1e-14 && continue
        Δθ = TP._dtheta_e2(mesh, i, e2)
        # unshifted: physical = u_reg + ψ K; Dual mesh.u after XBEM is u_reg
        W = TP.kirchhoff_wgrad_global(props, tip, e1, e2, mesh.nodes[i]; ω=π)
        θW = e2[1] * W[2, 1] + e2[2] * W[3, 1]
        ΔθW = 2 * abs(θW) * sign(K1 == 0 ? 1.0 : K1)
        # Hui–Zehnder |Δθ| for this K1
        ΔθW = (K1 / Cθ) * sqrt(ρ)   # K1 = Cθ Δθ / √ρ  ⇒  Δθ_W = K1 √ρ / Cθ
        # physical θ jump: u_reg jump + Williams
        uA, uB, nA, nB = TP.crack_opening_plate(mesh, i)
        sA = dot(nA, e2)
        sB = dot(nB, e2)
        θr = (abs(sA) > 1e-14 ? uA[2] / sA : uA[2]) - (abs(sB) > 1e-14 ? uB[2] / sB : uB[2])
        Δθp = θr + (K1 / Cθ) * sqrt(ρ) * sign(θr == 0 ? 1.0 : sign(θr))
        # sign: Williams Δθ for +K1 is positive opening
        Δθp = θr + K1 / Cθ * sqrt(ρ)
        @printf("  %.4f  %10.3e  %10.3e  %8.3f\n", ρ / 0.2, Δθp, K1 / Cθ * sqrt(ρ),
            Δθp / max(abs(K1 / Cθ * sqrt(ρ)), 1e-16))
    end
    flush(stdout)
end

println("=== replace Dual rows at near-tip with u_reg=0 ===")
report(4; n_near=2, interior=false)
report(6; n_near=2, interior=false)
println("=== baseline n_near=0 (no row replace) ===")
report(4; n_near=0, interior=false)
println("=== + interior w ===")
report(4; n_near=2, interior=true)
println("DONE")
