# SS square plate under uniform load — Navier series (Gmsh path for geometry note)
using DrWatson
@quickactivate :BEM
using .ThinPlate

println("="^60)
println(" Example: SS plate Navier w_max (discontinuous plate elements)")
println("="^60)

a, E, ν, h = 1.0, 1e5, 0.3, 0.01
q0 = 1.0
props = ThinPlateProps(; E=E, ν=ν, h=h, q_c=q0)
D = bending_stiffness(props)
w_ana = analytical_wmax_ss_square(; a=a, q=q0, D=D)

# ThinPlate uses discontinuous quadratic collocation (ξ=±2/3,0) — same as format2d tipo=2
mesh = build_square_plate(; a=a, n_el=8, bc="SSSS", props=props,
    corner_bc='F', n_internal=1)
assemble_plate!(mesh; npg=10)
solve_plate!(mesh)
w_c = plate_w_int(mesh, 1)
err = abs(w_c - w_ana) / abs(w_ana)
println("  D            = ", D)
println("  w_num / w_ana = ", w_c, " / ", w_ana)
println("  rel_error     = ", err)
@assert err < 0.10
println("OK.")
