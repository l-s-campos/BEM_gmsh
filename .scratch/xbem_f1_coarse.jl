using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf, StaticArrays
using BEM.Plate
const TP = BEM.Plate.ThinPlate

function run(ndiv_c; ndiv_b=4, ndiv_h=4, npg=6, nsub=4)
    props = ThinPlateProps(; E=2.1e5, ν=0.3, h=0.5)
    mesh = build_rect_plate_crack(; W=1.0, H=2.0, a=0.2, props=props, Mo=1.0,
        ndiv_b=ndiv_b, ndiv_h=ndiv_h, ndiv_crack=ndiv_c, nome="c$(ndiv_c)")
    tips = TP._plate_geometric_tips(mesh)
    xm = maximum(p[1] for p in tips)
    tippos = [p for p in tips if abs(p[1] - xm) < 1e-9][1]
    ρmin = minimum(norm(mesh.nodes[i] - tippos)
        for i in 1:length(mesh.nodes) if mesh.eq_type[i] == 2 &&
            norm(mesh.nodes[i] - tippos) > 1e-14)
    assemble_plate_dual!(mesh; npg=npg, nsub=nsub)
    solve_plate!(mesh)
    den = sqrt(0.2)
    Kb, _, _, _, _ = sif_ctod_plate(mesh; tip=:right, method=:band)
    Kt, _, _, _, _ = sif_ctod_plate(mesh; tip=:right, method=:tip)
    _, K1x, K2x, tps, _ = solve_plate_xbem!(mesh; n_enr=2, npg=npg, nsub=nsub, n_v=3)
    ir = argmax(p[1] for p in tps)
    @printf("  ndiv_c=%2d  ρmin/a=%.4f  Dual band=%.4f  tip=%.4f  extra F1=%.4f F2=%.4f\n",
        ndiv_c, ρmin / 0.2, abs(Kb) / den, abs(Kt) / den,
        abs(K1x[ir]) / den, abs(K2x[ir]) / den)
    flush(stdout)
    return abs(K1x[ir]) / den
end

println("=== Dual/XBEM vs ndiv_crack (first collocation ρ/a) ===")
for nc in (4, 6, 8)
    run(nc)
end
println("DONE")
