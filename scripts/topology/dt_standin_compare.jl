# DT stand-in vs JuMP (linear LP / MMA) vs Pacheco `move_boundary!`.
#
#   julia --project=. scripts/topology/dt_standin_compare.jl
#   julia --project=. scripts/topology/dt_standin_compare.jl --ids=1 --maxiter=8 --mma

using DrWatson
@quickactivate :BEM
using BEM.Topology
using Printf
using Dates
using StaticArrays

ids = [1]
maxiter = 8
ne = 8
nint = 8
degree = 1
do_elast = false
do_hole = true
do_mma = false
area_rtol = 0.08

for a in ARGS
    if startswith(a, "--ids=")
        global ids = parse.(Int, split(split(a, "=")[2], ","; keepempty=false))
    elseif startswith(a, "--maxiter=")
        global maxiter = parse(Int, split(a, "=")[2])
    elseif startswith(a, "--ne=")
        global ne = parse(Int, split(a, "=")[2])
    elseif a == "--elast"
        global do_elast = true
    elseif a == "--mma"
        global do_mma = true
    elseif a == "--no-hole"
        global do_hole = false
    end
end

const HEAT_NAMES = ("inverted-V", "asymmetric", "bridge", "cross")

_J0_heat(d) = (dad = bemdata_from_loops(d); H_G_full_direct(dad; npg=8, threaded=false);
    solve(dad); thermal_conductance(dad))
_J0_elast(d) = (dad = bemdata_from_loops(d); H_G_full_direct(dad; npg=8, threaded=false);
    solve(dad); elastic_compliance(dad))

function _row(name, method, A, Ap, J, J0, holes, folded, s)
    ratio = J0 == 0 ? NaN : J / J0
    @printf("  %-14s %-18s %8.4f %8.4f %10.4g %8.3f %5d %6s %6.1f\n",
        name, method, A, Ap, J, ratio, holes, folded ? "yes" : "no", s)
end

function _run(d0, opt; nucleate, motion, inward=false, jump_mode=:linear)
    o = deepcopy(opt)
    o.maxiter = maxiter
    o.verbose = false
    o.area_rtol = area_rtol
    o.motion = motion
    o.standin_inward = inward
    o.jump_mode = jump_mode
    o.jump_maxeval = 6
    o.vmax = motion in (:standin, :jump) ? 0.04 : o.vmax
    if !nucleate
        o.nucleate_first = false
        o.nucleate_every = typemax(Int)
    end
    t0 = time()
    d, dad, _ = solve_topology!(copy_design(d0), o)
    return d, dad, time() - t0
end

println("="^72)
println(" DT stand-in vs move_boundary! (quantile)   $(Dates.now())")
println(" ids=$ids  elast=$do_elast  ne=$ne  maxiter=$maxiter")
println(" J/J0: heat higher is better; elasticity lower is better")
println(" :quantile = current inward low-DT recession")
println(" :standin  = vn ∝ (DT − λ), both ways, area projection")
println(" :jump     = JuMP linearized MMFD LP (HiGHS)")
println(" :jump-mma = JuMP + NLopt LD_MMA (BEM in f; --mma)")
println("="^72)
@printf("  %-14s %-18s %8s %8s %10s %8s %5s %6s %6s\n",
    "problem", "method", "A", "Ap", "J", "J/J0", "holes", "fold", "s")
println("  ", "-"^72)

if do_hole
    let
        d0 = portela_heat_hole(; a=0.30, ne=ne, nint=nint, degree=degree)
        J0 = _J0_heat(d0)
        opt = PachecoOptions(ΔA=0.15, vmax=0.04, pct=0.9, volume_step=0.25)
        Ap = (1 - opt.ΔA) * design_area(d0)
        for (lab, motion, nuc, inward, jmode) in (
                ("quantile-shape", :quantile, false, false, :linear),
                ("standin", :standin, false, false, :linear),
                ("jump-lp", :jump, false, false, :linear),
            )
            d, dad, s = _run(d0, opt; nucleate=nuc, motion=motion, inward=inward,
                jump_mode=jmode)
            _row("heat-hole", lab, design_area(d), Ap, thermal_conductance(dad), J0,
                n_holes(d), BEM.Topology._design_folded(d), s)
        end
        if do_mma
            d, dad, s = _run(d0, opt; nucleate=false, motion=:jump, jump_mode=:mma)
            _row("heat-hole", "jump-mma", design_area(d), Ap, thermal_conductance(dad), J0,
                n_holes(d), BEM.Topology._design_folded(d), s)
        end
    end
end

for id in ids
    let name = HEAT_NAMES[id]
        dP, optP = pacheco_problem(id; ne=ne, nint=nint, degree=degree)
        J0 = _J0_heat(dP)
        Ap = (1 - optP.ΔA) * design_area(dP)
        for (lab, motion, nuc, inward, jmode) in (
                ("quantile-shape", :quantile, false, false, :linear),
                ("standin", :standin, false, false, :linear),
                ("jump-lp", :jump, false, false, :linear),
                ("quantile+nuc", :quantile, true, false, :linear),
            )
            d, dad, s = _run(dP, optP; nucleate=nuc, motion=motion, inward=inward,
                jump_mode=jmode)
            _row(name, lab, design_area(d), Ap, thermal_conductance(dad), J0,
                n_holes(d), BEM.Topology._design_folded(d), s)
        end
        if do_mma
            d, dad, s = _run(dP, optP; nucleate=false, motion=:jump, jump_mode=:mma)
            _row(name, "jump-mma", design_area(d), Ap, thermal_conductance(dad), J0,
                n_holes(d), BEM.Topology._design_folded(d), s)
        end
    end
end

if do_elast
    println()
    let
        dC, optC = coelho_problem(3; ne=ne, nint=nint, degree=degree)
        J0 = _J0_elast(dC)
        Ap = (1 - optC.ΔA) * design_area(dC)
        for (lab, motion, nuc, inward, jmode) in (
                ("quantile-shape", :quantile, false, false, :linear),
                ("standin", :standin, false, false, :linear),
                ("jump-lp", :jump, false, false, :linear),
            )
            d, dad, s = _run(dC, optC; nucleate=nuc, motion=motion, inward=inward,
                jump_mode=jmode)
            _row("cantilever", lab, design_area(d), Ap, elastic_compliance(dad), J0,
                n_holes(d), BEM.Topology._design_folded(d), s)
        end
    end
end
