# DIBEM PHS centers = cell centroids vs collocation (wave-bar mesh).
# julia --project=. scripts/dibem_volume_centers.jl
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf

include(datadir("Laplace", "Laplace_dad.jl"))
include(datadir("Laplace", "potencial_problems.jl"))
include(datadir("Laplace", "wave_propagation.jl"))
include(datadir("elastico", "iso", "bar_sudden.jl"))

nnegc(c) = count(<(0), c)
nnegM(M) = count(<( -1e-8), real.(eigvals(Matrix(M))))
rel(a, b) = norm(a - b) / (norm(b) + 1e-30)

function probe_ux(U, dad, probe)
    pts = vcat(dad.Nodes, dad.internalNodes)
    ip = argmin(norm(p - probe) for p in pts)
    nd = size(U, 1) ÷ dad.nt
    return U[nd * (ip - 1) + 1, :]
end

function report(tag, dad)
    c = dad.dibem_c
    ev = real.(eigvals(Matrix(dad.M)))
    @printf("%-22s  nc=%3d  c_neg=%3d  c_min=%+.3e  nneg(M)=%3d  λmin=%+.3e\n",
        tag, length(c), nnegc(c), minimum(c), count(<( -1e-8), ev), minimum(ev))
end

Δt, tf = 0.05, 2.0
PROBE = Point2D(1.0, 0.5)
rbf = PHS(3; poly_deg=1)

println("="^72)
println(" Volume-center DIBEM  (centers=:cells)")
println("="^72)

dadE, metaE = elasticity_bar_sudden(; ndiv=8, n_int=4, ν=0.0)
H_G_full_direct(dadE; npg=8, threaded=false)
println("\nElasticity bar  n=$(dadE.n) ni=$(dadE.ni) cells=$(length(extract_domain_cells(dadE)))")

d0 = deepcopy(dadE)
DIBEM(d0; method=:dense, rbf=rbf, centers=:collocation)
report("collocation", d0)

d1 = deepcopy(dadE)
DIBEM(d1; method=:dense, rbf=rbf, centers=:cells)
report("cell centroids", d1)

dC = deepcopy(dadE)
build_cell_mass(dC; npg=8)
evC = real.(eigvals(Matrix(dC.M)))
@printf("%-22s  nc=%3d  c_neg=%3d  c_min=%+.3e  nneg(M)=%3d  λmin=%+.3e\n",
    "cells (RIM)", length(dC.cells), 0, NaN, count(<( -1e-8), evC), minimum(evC))

ua = [metaE.ana.u(metaE.probe; t=ti) for ti in 0:Δt:tf]

function try_step(tag, dad, stepper)
    ok, err, mx = true, NaN, NaN
    try
        stepper(dad)
        U = dad.u
        ok = all(isfinite, U)
        un = probe_ux(U, dad, metaE.probe)
        err = rel(un, ua)
        mx = maximum(abs, un)
    catch e
        ok = false
        @printf("  FAIL %s: %s\n", tag, sprint(showerror, e))
    end
    @printf("  %-22s  rel=%.3e  max=%.3e  %s\n", tag, err, mx, ok ? "ok" : "FAIL")
end

println("\nTime stepping Δt=$Δt tf=$tf")
try_step("El colloc Houbolt", deepcopy(d0), d -> solve_Houbolt(d, Δt, tf))
try_step("El volume Houbolt", deepcopy(d1), d -> solve_Houbolt(d, Δt, tf))
try_step("El colloc Newmark", deepcopy(d0), d -> solve_Newmark(d, Δt, tf))
try_step("El volume Newmark", deepcopy(d1), d -> solve_Newmark(d, Δt, tf))
try_step("El cells Houbolt", deepcopy(dC), d -> solve_Houbolt(d, Δt, tf))

dadL, metaL = wave_problem(:bar_sudden; ndiv=8, n_int=4)
H_G_full_direct(dadL; npg=8, threaded=false)
println("\nLaplace bar  n=$(dadL.n) ni=$(dadL.ni) cells=$(length(extract_domain_cells(dadL)))")
L0 = deepcopy(dadL)
DIBEM(L0; method=:dense, rbf=rbf, centers=:collocation)
report("collocation", L0)
L1 = deepcopy(dadL)
DIBEM(L1; method=:dense, rbf=rbf, centers=:cells)
report("cell centroids", L1)

uaL = [metaL.ana.u(PROBE; t=ti) for ti in 0:Δt:tf]
function try_stepL(tag, dad, stepper)
    ok, err, mx = true, NaN, NaN
    try
        stepper(dad)
        T = dad.T
        ok = all(isfinite, T)
        pts = vcat(dad.Nodes, dad.internalNodes)
        ip = argmin(norm(p - PROBE) for p in pts)
        un = T[ip, :]
        err = rel(un, uaL)
        mx = maximum(abs, un)
    catch e
        ok = false
        @printf("  FAIL %s: %s\n", tag, sprint(showerror, e))
    end
    @printf("  %-22s  rel=%.3e  max=%.3e  %s\n", tag, err, mx, ok ? "ok" : "FAIL")
end
println("\nLaplace time stepping")
try_stepL("Lap colloc Houbolt", deepcopy(L0), d -> solve_Houbolt(d, Δt, tf))
try_stepL("Lap volume Houbolt", deepcopy(L1), d -> solve_Houbolt(d, Δt, tf))
try_stepL("Lap colloc Newmark", deepcopy(L0), d -> solve_Newmark(d, Δt, tf))
try_stepL("Lap volume Newmark", deepcopy(L1), d -> solve_Newmark(d, Δt, tf))
println("\nDone.")
