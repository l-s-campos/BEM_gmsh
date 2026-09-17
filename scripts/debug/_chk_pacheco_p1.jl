using DrWatson
@quickactivate :BEM
using BEM.Topology

d, opt = pacheco_problem(1; ne=10, nint=10, degree=1)
opt.maxiter = 8
opt.verbose = true
opt.nucleate_every = 1
A0 = design_area(d)
println("A0=", A0, " Ap=", (1 - opt.ΔA) * A0)
d, dad, hist = solve_topology!(d, opt)
println("final A=", design_area(d), " holes=", n_holes(d))
println("area hist=", hist.area)
println("J hist=", hist.J)
