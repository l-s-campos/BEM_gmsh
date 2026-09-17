using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf, StaticArrays
using BEM.Plate
const TP = BEM.Plate.ThinPlate

function run(; ndiv_c=6, bump=0.45, ndiv_b=4, ndiv_h=4)
    props = ThinPlateProps(; E=2.1e5, ν=0.3, h=0.5)
    mesh = build_rect_plate_crack(; W=1.0, H=2.0, a=0.2, props=props, Mo=1.0,
        ndiv_b=ndiv_b, ndiv_h=ndiv_h, ndiv_crack=ndiv_c,
        nome="bump_$(ndiv_c)_$(bump)", bump=bump)
    tips = TP._plate_geometric_tips(mesh)
    xm = maximum(p[1] for p in tips)
    tippos = [p for p in tips if abs(p[1] - xm) < 1e-9][1]
    ρs = [norm(mesh.nodes[i] - tippos)
          for i in 1:length(mesh.nodes)
          if mesh.eq_type[i] == 2 && norm(mesh.nodes[i] - tippos) > 1e-14]
    ρmin = minimum(ρs)
    assemble_plate_dual!(mesh; npg=6, nsub=4)
    solve_plate!(mesh)
    den = sqrt(0.2)
    Kb, _, _, _, _ = sif_ctod_plate(mesh; tip=:right, method=:band)
    _, K1x, K2x, tps, _ = solve_plate_xbem!(mesh; n_enr=2, npg=6, nsub=4, n_v=3)
    ir = argmax(p[1] for p in tps)
    @printf("  bump=%.2f ndiv_c=%d  ρmin/a=%.4f  Dual=%.4f  extra F1=%.4f F2=%.4f  K1=%s\n",
        bump, ndiv_c, ρmin / 0.2, abs(Kb) / den, abs(K1x[ir]) / den,
        abs(K2x[ir]) / den, string(K1x))
    flush(stdout)
end

println("=== bump sweep ===")
for b in (1.0, 0.6, 0.45, 0.3)
    run(; bump=b, ndiv_c=6)
end
run(; bump=0.45, ndiv_c=4)
run(; bump=1.0, ndiv_c=4)
println("DONE")
