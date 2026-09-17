using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf, Statistics, StaticArrays, ForwardDiff
using BEM.Plate
const TP = BEM.Plate.ThinPlate

function dual_F(; h=0.5, ndiv_b=4, ndiv_h=4, ndiv_crack=6, npg=6, nsub=4)
    props = ThinPlateProps(; E=2.1e5, ν=0.3, h=h)
    mesh = build_rect_plate_crack(; W=1.0, H=2.0, a=0.2, props=props, Mo=1.0,
        ndiv_b=ndiv_b, ndiv_h=ndiv_h, ndiv_crack=ndiv_crack,
        nome="f1_$(h)_$(ndiv_crack)")
    assemble_plate_dual!(mesh; npg=npg, nsub=nsub)
    solve_plate!(mesh)
    den = sqrt(0.2)
    Kb, _, _, _, _ = sif_ctod_plate(mesh; tip=:right, method=:band)
    Kt, _, _, _, _ = sif_ctod_plate(mesh; tip=:right, method=:tip)
    return mesh, abs(Kb) / den, abs(Kt) / den
end

function xbem_F(mesh; npg=6, nsub=4)
    den = sqrt(0.2)
    _, K1x, K2x, tips, _ = solve_plate_xbem!(mesh; n_enr=2, npg=npg, nsub=nsub, n_v=3)
    ir = argmax(p[1] for p in tips)
    return abs(K1x[ir]) / den, abs(K2x[ir]) / den, K1x, K2x
end

println("=== kernel 6th vs FD of 3rd ===")
props = ThinPlateProps(; E=2.1e5, ν=0.3, h=0.5)
for r in (1e-1, 1e-2, 1e-3)
    rel = SVector(r, 0.1 * r)
    d = TP._wderivs6(rel, props)
    d3(u) = TP._iso_d3(u, 1 / (8π * bending_stiffness(props)))
    J = ForwardDiff.jacobian(d3, collect(rel))
    # J rows: wxxx,wxxy,wxyy,wyyy; cols x,y → wxxxx=J[1,1]
    @printf("  r=%.1e  wxxxx FD=%.4e  AD6=%.4e  rel=%.2e\n",
        r, J[1, 1], d.wxxxx, abs(J[1, 1] - d.wxxxx) / max(abs(J[1, 1]), 1e-30))
    J5 = ForwardDiff.jacobian(u -> ForwardDiff.jacobian(d3, u)[:, 1], collect(rel))
    @printf("         wxxxxx FD=%.4e  AD6=%.4e  rel=%.2e\n",
        J5[1, 1], d.wxxxxx, abs(J5[1, 1] - d.wxxxxx) / max(abs(J5[1, 1]), 1e-30))
end
flush(stdout)

println("=== Dual F vs mesh / h ===")
for (h, nb, nh, nc) in ((0.5, 4, 4, 6), (0.05, 4, 4, 6), (0.5, 5, 5, 10))
    mesh, Fb, Ft = dual_F(; h=h, ndiv_b=nb, ndiv_h=nh, ndiv_crack=nc)
    @printf("  h=%.2f  %d/%d/%d  band=%.4f  tip=%.4f\n", h, nb, nh, nc, Fb, Ft)
    if nb == 4 && h == 0.5
        Fx, Fy, K1x, K2x = xbem_F(mesh)
        @printf("         extra F1=%.4f  F2=%.4f  K1=%s\n", Fx, Fy, string(K1x))
    end
    flush(stdout)
end

println("=== Reissner Dual F (same 4/4/6) ===")
propsR = FSDTProps(; E=2.1e5, ν=0.3, h=0.5)
meshR = build_rect_fsdt_crack(; W=1.0, H=2.0, a=0.2, props=propsR, Mo=1.0,
    ndiv_b=4, ndiv_h=4, ndiv_crack=6, nome="f1_fsdt")
assemble_fsdt_dual!(meshR; npg=6, nsub=4)
solve_fsdt!(meshR)
K1b, _, _, _, _, _ = sif_ctod_fsdt(meshR; tip=:right)
@printf("  Reissner Dual F=K1b/(Mo√(πa))=%.4f   Dolbow F=K1/(Mo√a)=%.4f\n",
    abs(K1b) / sqrt(π * 0.2), abs(K1b) / sqrt(π) / sqrt(0.2))
flush(stdout)
println("DONE")
