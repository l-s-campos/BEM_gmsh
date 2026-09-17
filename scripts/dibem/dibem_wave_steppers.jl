# Full-system wave steppers (no MMM): Houbolt vs Newmark vs DiffEq.
# Laplace + elasticity sudden bar vs 1-D series.
# julia --project=. scripts/dibem_wave_steppers.jl
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf, Statistics

include(datadir("Laplace", "Laplace_dad.jl"))
include(datadir("Laplace", "potencial_problems.jl"))
include(datadir("Laplace", "wave_propagation.jl"))
include(datadir("elastico", "iso", "bar_sudden.jl"))

rel(a, b) = norm(a - b) / (norm(b) + 1e-30)

function nneg(M)
    ev = real.(eigvals(Matrix(M)))
    return count(<( -1e-8), ev)
end

function probe_hist(U, dad, probe; comp=1)
    pts = vcat(dad.Nodes, dad.internalNodes)
    ip = argmin(norm(p - probe) for p in pts)
    nd = size(U, 1) ÷ dad.nt
    return U[nd * (ip - 1) + comp, :]
end

function run_one(tag, dad, meta, stepper; Δt, tf, probe, comp=1)
    t0 = time()
    ok = true
    un = Float64[]
    try
        stepper(dad)
        U = has_cache(dad, :u) ? dad.u : dad.T
        un = probe_hist(U, dad, probe; comp=comp)
        ok = all(isfinite, U)
    catch e
        ok = false
        @printf("  FAIL %s: %s\n", tag, sprint(showerror, e))
    end
    dt = time() - t0
    t = collect(0:Δt:tf)
    ua = [meta.ana.u(probe; t=ti) for ti in t]
    err = (ok && length(un) == length(ua)) ? rel(un, ua) : NaN
    mx = (ok && !isempty(un)) ? maximum(abs, un) : NaN
    return (; tag, err, mx, dt, ok, nneg=nneg(dad.M))
end

function print_row(r)
    @printf("  %-28s  rel=%.3e  max=%.3f  nneg=%3d  t=%.2fs  %s\n",
        r.tag, r.err, r.mx, r.nneg, r.dt, r.ok ? "ok" : "FAIL")
end

Δt, tf = 0.05, 2.0
PROBE = Point2D(1.0, 0.5)

println("="^72)
println(" Wave steppers  Δt=$Δt  tf=$tf  (no MMM)")
println("="^72)

# ---------- Laplace ----------
dadL, metaL = wave_problem(:bar_sudden; ndiv=8, n_int=4)
H_G_full_direct(dadL; npg=8, threaded=false)
DIBEM(dadL; method=:dense, rbf=PHS(3; poly_deg=1))
println("\nLaplace bar  n=$(dadL.n) ni=$(dadL.ni)  nneg(M)=$(nneg(dadL.M))")

rows = []
push!(rows, run_one("Lap DIBEM Houbolt", deepcopy(dadL), metaL,
    d -> solve_Houbolt(d, Δt, tf); Δt, tf, probe=PROBE))
push!(rows, run_one("Lap DIBEM Newmark 1/4", deepcopy(dadL), metaL,
    d -> solve_Newmark(d, Δt, tf; β=1 / 4, γ=1 / 2); Δt, tf, probe=PROBE))
push!(rows, run_one("Lap DIBEM Newmark 1/6", deepcopy(dadL), metaL,
    d -> solve_Newmark(d, Δt, tf; β=1 / 6, γ=1 / 2); Δt, tf, probe=PROBE))
# DiffEq on indefinite DIBEM M diverges; run those algs on cell M below.

dadLd = deepcopy(dadL)
build_drm_matrices(dadLd, PHS(3; poly_deg=1); npg=8)
push!(rows, run_one("Lap DRM Houbolt", dadLd, metaL,
    d -> solve_Houbolt(d, Δt, tf); Δt, tf, probe=PROBE))
push!(rows, run_one("Lap DRM Newmark 1/4", deepcopy(dadLd), metaL,
    d -> solve_Newmark(d, Δt, tf); Δt, tf, probe=PROBE))

if has_cache(dadL, :cells) && !isempty(extract_domain_cells(dadL))
    dadC = deepcopy(dadL)
    build_cell_mass(dadC; npg=8)
    push!(rows, run_one("Lap cells Houbolt", dadC, metaL,
        d -> solve_Houbolt(d, Δt, tf); Δt, tf, probe=PROBE))
    push!(rows, run_one("Lap cells Newmark 1/4", deepcopy(dadC), metaL,
        d -> solve_Newmark(d, Δt, tf); Δt, tf, probe=PROBE))
end

for r in rows
    print_row(r)
end
okL = filter(r -> r.ok && isfinite(r.err), rows)
if !isempty(okL)
    best = okL[argmin(getfield.(okL, :err))]
    println("  best Laplace: $(best.tag)  rel=$(best.err)")
end

# ---------- Elasticity ----------
dadE, metaE = elasticity_bar_sudden(; ndiv=8, n_int=4, ν=0.0)
H_G_full_direct(dadE; npg=8, threaded=false)
DIBEM(dadE; method=:dense, rbf=PHS(3; poly_deg=1))
println("\nElasticity bar ν=0  n=$(dadE.n) ni=$(dadE.ni)  nneg(M)=$(nneg(dadE.M))")

rowsE = []
push!(rowsE, run_one("El DIBEM Houbolt", deepcopy(dadE), metaE,
    d -> solve_Houbolt(d, Δt, tf); Δt, tf, probe=metaE.probe))
push!(rowsE, run_one("El DIBEM Newmark 1/4", deepcopy(dadE), metaE,
    d -> solve_Newmark(d, Δt, tf; β=1 / 4); Δt, tf, probe=metaE.probe))
push!(rowsE, run_one("El DIBEM Newmark 1/6", deepcopy(dadE), metaE,
    d -> solve_Newmark(d, Δt, tf; β=1 / 6); Δt, tf, probe=metaE.probe))

dadEd = deepcopy(dadE)
build_drm_matrices(dadEd; npg=8, kernel=:r, poly_deg=1)
push!(rowsE, run_one("El DRM Houbolt", dadEd, metaE,
    d -> solve_Houbolt(d, Δt, tf); Δt, tf, probe=metaE.probe))
push!(rowsE, run_one("El DRM Newmark 1/4", deepcopy(dadEd), metaE,
    d -> solve_Newmark(d, Δt, tf); Δt, tf, probe=metaE.probe))

if has_cache(dadE, :cells) && !isempty(extract_domain_cells(dadE))
    dadC = deepcopy(dadE)
    build_cell_mass(dadC; npg=8)
    push!(rowsE, run_one("El cells Houbolt", dadC, metaE,
        d -> solve_Houbolt(d, Δt, tf); Δt, tf, probe=metaE.probe))
    push!(rowsE, run_one("El cells Newmark 1/4", deepcopy(dadC), metaE,
        d -> solve_Newmark(d, Δt, tf); Δt, tf, probe=metaE.probe))
end

for r in rowsE
    print_row(r)
end
okE = filter(r -> r.ok && isfinite(r.err), rowsE)
if !isempty(okE)
    best = okE[argmin(getfield.(okE, :err))]
    println("  best elasticity: $(best.tag)  rel=$(best.err)")
end
println("\nDone.")
