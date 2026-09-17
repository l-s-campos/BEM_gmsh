# Phase 0 printout: julia --project=. scripts/check_analytical_elastodynamics.jl
using Printf
using Statistics: mean
using BEM

include(joinpath(@__DIR__, "..", "..", "data", "elastico", "iso", "analytical_elastodynamics.jl"))
include(joinpath(@__DIR__, "..", "..", "data", "Laplace", "Laplace_dad.jl"))
include(joinpath(@__DIR__, "..", "..", "data", "Laplace", "potencial_problems.jl"))
include(joinpath(@__DIR__, "..", "..", "data", "Laplace", "wave_propagation.jl"))

t = range(0, 4; length=81)
w_eb = euler_bernoulli_ss_step.(0.5, t)
@printf("EB mid-span  min=%.4e  max=%.4e  mean=%.4e  static=%.4e\n",
    minimum(w_eb), maximum(w_eb), mean(w_eb), 1 / 48)

w_ti = timoshenko_ss_step.(0.5, t; L=1.0, F0=1e3)
@printf("Timoshenko SS mid  min=%.4e  max=%.4e\n", minimum(w_ti), maximum(w_ti))

w_c = timoshenko_cantilever_tip_step.(1.0, t)
@printf("Cantilever tip  min=%.4e  max=%.4e  w(0,0.2)=%.4e\n",
    minimum(w_c), maximum(w_c), timoshenko_cantilever_tip_step(0.0, 0.2))

u_cyl = cylinder_step_pressure.(1.5, t)
ust = cylinder_u_static(1.5)
@printf("Cylinder r=1.5  t=0=%.4e  max=%.4e  mean=%.4e  Lamé=%.4e\n",
    cylinder_step_pressure(1.5, 0.0), maximum(u_cyl), mean(u_cyl), ust)

u_pl = plate_hole_pressure.(1.5, t)
@printf("Plate-hole r=1.5  t=0=%.4e  max=%.4e\n", u_pl[1], maximum(abs, u_pl))

σ = [transient_kirsch(1.0, π / 2, ti) for ti in t]
_, σθθ, _ = kirsch_static(1.0, π / 2)
@printf("Kirsch σθθ(a,π/2)  t=0=%.4e  last=%.4e  static=%.4e\n", σ[1], σ[end], σθθ)

ana = ana_bar_sudden(; N=200, c=1.0, L=1.0)
p = Point2D(1.0, 0.5)
@printf("Bar u(L,0.5)=%.4e  u(L,1.5)=%.4e\n", ana.u(p; t=0.5), ana.u(p; t=1.5))
