# Portela shape vs Pacheco / SIMP / DT-ρ / level-set.
#
# Track C (default): methods from the empty design (Portela cannot nucleate).
# Track B (`--polish`): SIMP iso-cut then Portela vs Pacheco shape-only.
#
#   julia --project=. scripts/topology/portela_compare.jl
#   julia --project=. scripts/topology/portela_compare.jl --ids=1 --maxiter=8
#   julia --project=. scripts/topology/portela_compare.jl --elast --ids=3
#   julia --project=. scripts/topology/portela_compare.jl --polish --ls

using DrWatson
@quickactivate :BEM
using BEM.Topology
using Printf
using Statistics
using LinearAlgebra
using Dates
using StaticArrays

ids = [1]
maxiter = 8
n_simp = 8
ne = 8
nint = 8
degree = 1
do_elast = false
do_ls = false
do_polish = false
do_hole = true
area_rtol = 0.08

for a in ARGS
    if startswith(a, "--ids=")
        global ids = parse.(Int, split(split(a, "=")[2], ","; keepempty=false))
    elseif startswith(a, "--maxiter=")
        global maxiter = parse(Int, split(a, "=")[2])
    elseif startswith(a, "--nsimp=")
        global n_simp = parse(Int, split(a, "=")[2])
    elseif startswith(a, "--ne=")
        global ne = parse(Int, split(a, "=")[2])
    elseif a == "--elast"
        global do_elast = true
    elseif a == "--ls"
        global do_ls = true
    elseif a == "--polish"
        global do_polish = true
    elseif a == "--no-hole"
        global do_hole = false
    end
end

const HEAT_NAMES = ("inverted-V", "asymmetric", "bridge", "cross")

_J0_heat(d) = (dad = bemdata_from_loops(d); H_G_full_direct(dad; npg=8, threaded=false);
    solve(dad); thermal_conductance(dad))
_J0_elast(d) = (dad = bemdata_from_loops(d); H_G_full_direct(dad; npg=8, threaded=false);
    solve(dad); elastic_compliance(dad))

function _row(name, method, A, Ap, J, J0, holes, folded, s; heat=true)
    ratio = J0 == 0 ? NaN : J / J0
    @printf("  %-14s %-16s %8.4f %8.4f %10.4g %8.3f %5d %6s %6.1f\n",
        name, method, A, Ap, J, ratio, holes, folded ? "yes" : "no", s)
end

function _pacheco_run(d, opt; nucleate=true)
    o = deepcopy(opt)
    o.maxiter = maxiter
    o.verbose = false
    o.area_rtol = area_rtol
    if !nucleate
        o.nucleate_first = false
        o.nucleate_every = typemax(Int)
    end
    t0 = time()
    d, dad, hist = solve_topology!(d, o)
    return d, dad, time() - t0
end

function _simp_run(d, volfrac)
    opt = DibemSimpOptions(; volfrac=volfrac, n_simp=n_simp, rmin=0.12, ngrid=41,
        cut=true, pacheco=true, verbose=false, npg=8,
        pacheco_opt=PachecoOptions(maxiter=maxiter, verbose=false,
            nucleate_first=false, nucleate_every=typemax(Int), area_rtol=area_rtol))
    t0 = time()
    d, dad, _, _ = solve_dibem_simp!(d, opt)
    return d, dad, time() - t0
end

function _dt_run(d, volfrac)
    opt = DibemSimpOptions(; volfrac=volfrac, n_simp=n_simp, rmin=0.12, ngrid=41,
        cut=true, pacheco=true, verbose=false, npg=8, method=:dt,
        pacheco_opt=PachecoOptions(maxiter=maxiter, verbose=false,
            nucleate_first=false, nucleate_every=typemax(Int), area_rtol=area_rtol))
    t0 = time()
    d, dad, _, _ = solve_dt_density!(d, opt)
    return d, dad, time() - t0
end

function _portela_run(d; ΔA=0.0, param=:normal, origin=nothing)
    o = PortelaOptions(maxiter=maxiter, npg=8, state=:cbie, param=param,
        vmax=0.03, α=0.3, verbose=false, ΔA=ΔA, area_rtol=area_rtol)
    origin !== nothing && (o.origin = origin)
    t0 = time()
    d, dad, _ = solve_portela!(d, o)
    return d, dad, time() - t0
end

function _ls_run(d, ΔA)
    o = LevelSetOptions(ΔA=ΔA, maxiter=maxiter, ngrid=48, verbose=false, npg=8)
    t0 = time()
    d, dad, _, _ = solve_levelset!(d, o)
    return d, dad, time() - t0
end

println("="^72)
println(" Portela vs other strategies   $(Dates.now())")
println(" heat ids=$ids  elast=$(do_elast)  polish=$(do_polish)  ls=$(do_ls)")
println(" ne=$ne  nint=$nint  maxiter=$maxiter  n_simp=$n_simp")
println(" J/J0: heat higher is better; elasticity lower is better")
println(" Portela is shape-only (no nucleation).")
println("="^72)
@printf("  %-14s %-16s %8s %8s %10s %8s %5s %6s %6s\n",
    "problem", "method", "A", "Ap", "J", "J/J0", "holes", "fold", "s")
println("  ", "-"^70)

if do_hole
    let
        d0 = portela_heat_hole(; a=0.30, ne=ne, nint=nint, degree=degree)
        J0 = _J0_heat(d0)
        Ap = design_area(d0)
        d, dad, s = _portela_run(copy_design(d0); ΔA=0.0, param=:radial,
            origin=SVector{2,Float64}(0.5, 0.5))
        _row("heat-hole", "portela", design_area(d), Ap, thermal_conductance(dad), J0,
            n_holes(d), BEM.Topology._design_folded(d), s)
        d, dad, s = _pacheco_run(copy_design(d0), PachecoOptions(ΔA=0.0, vmax=0.03, pct=0.9);
            nucleate=false)
        _row("heat-hole", "pacheco-shape", design_area(d), Ap, thermal_conductance(dad), J0,
            n_holes(d), BEM.Topology._design_folded(d), s)
    end
end

for id in ids
    let name = HEAT_NAMES[id]
        dP, optP = pacheco_problem(id; ne=ne, nint=nint, degree=degree)
        J0 = _J0_heat(dP)
        Ap = (1 - optP.ΔA) * design_area(dP)
        d, dad, s = _portela_run(copy_design(dP); ΔA=optP.ΔA)
        _row(name, "portela", design_area(d), Ap, thermal_conductance(dad), J0,
            n_holes(d), BEM.Topology._design_folded(d), s)
        d, dad, s = _pacheco_run(copy_design(dP), optP; nucleate=false)
        _row(name, "pacheco-shape", design_area(d), Ap, thermal_conductance(dad), J0,
            n_holes(d), BEM.Topology._design_folded(d), s)
        d, dad, s = _pacheco_run(copy_design(dP), optP; nucleate=true)
        _row(name, "pacheco", design_area(d), Ap, thermal_conductance(dad), J0,
            n_holes(d), BEM.Topology._design_folded(d), s)
        d, dad, s = _simp_run(copy_design(dP), 1 - optP.ΔA)
        _row(name, "simp+cut", design_area(d), Ap, thermal_conductance(dad), J0,
            n_holes(d), BEM.Topology._design_folded(d), s)
        d, dad, s = _dt_run(copy_design(dP), 1 - optP.ΔA)
        _row(name, "dt-ρ+cut", design_area(d), Ap, thermal_conductance(dad), J0,
            n_holes(d), BEM.Topology._design_folded(d), s)
        if do_ls
            d, dad, s = _ls_run(copy_design(dP), optP.ΔA)
            _row(name, "levelset", design_area(d), Ap, thermal_conductance(dad), J0,
                n_holes(d), BEM.Topology._design_folded(d), s)
        end
        if do_polish
            ds, _, _ = _simp_run(copy_design(dP), 1 - optP.ΔA)
            dp, dadp, s = _portela_run(copy_design(ds); ΔA=0.0)
            _row(name, "simp→portela", design_area(dp), Ap, thermal_conductance(dadp), J0,
                n_holes(dp), BEM.Topology._design_folded(dp), s)
        end
    end
end

if do_elast
    println()
    let
        d0 = portela_plate_hole(; ratio=1.0, ne_outer=4, ne_hole=4, nint=nint, degree=degree, E=1.0)
        J0 = _J0_elast(d0)
        A0 = design_area(d0)
        d, dad, s = _portela_run(copy_design(d0); ΔA=0.0, param=:radial,
            origin=SVector{2,Float64}(0.0, 0.0))
        _row("plate-hole", "portela", design_area(d), A0, elastic_compliance(dad), J0,
            n_holes(d), BEM.Topology._design_folded(d), s; heat=false)
    end
    for id in (3,)
        let name = id == 3 ? "cantilever" : id == 4 ? "cc-beam" : "ss-beam"
            dC, optC = coelho_problem(id; ne=ne, nint=nint, degree=degree)
            J0 = _J0_elast(dC)
            Ap = (1 - optC.ΔA) * design_area(dC)
            d, dad, s = _portela_run(copy_design(dC); ΔA=optC.ΔA)
            _row(name, "portela", design_area(d), Ap, elastic_compliance(dad), J0,
                n_holes(d), BEM.Topology._design_folded(d), s; heat=false)
            d, dad, s = _pacheco_run(copy_design(dC), optC; nucleate=false)
            _row(name, "pacheco-shape", design_area(d), Ap, elastic_compliance(dad), J0,
                n_holes(d), BEM.Topology._design_folded(d), s; heat=false)
            d, dad, s = _pacheco_run(copy_design(dC), optC; nucleate=true)
            _row(name, "pacheco", design_area(d), Ap, elastic_compliance(dad), J0,
                n_holes(d), BEM.Topology._design_folded(d), s; heat=false)
        end
    end
end
