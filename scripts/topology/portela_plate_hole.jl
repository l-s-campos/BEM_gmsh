# Portela (2012) plate-with-hole: equal biaxial → circle; ratio 0.75 → ellipse.
#
#   julia --project=. scripts/topology/portela_plate_hole.jl
#   julia --project=. scripts/topology/portela_plate_hole.jl --state=dual --maxiter=20

using DrWatson
@quickactivate :BEM
using BEM.Topology
using Printf
using Statistics
using LinearAlgebra
using StaticArrays

state = :cbie
maxiter = 12
ratio = 1.0
do_heat = true

for a in ARGS
    if startswith(a, "--state=")
        global state = Symbol(split(a, "=")[2])
    elseif startswith(a, "--maxiter=")
        global maxiter = parse(Int, split(a, "=")[2])
    elseif startswith(a, "--ratio=")
        global ratio = parse(Float64, split(a, "=")[2])
    elseif a == "--no-heat"
        global do_heat = false
    end
end

function _hole_radii(d::TopologyDesign, origin)
    rs = Float64[]
    for segs in d.loops, s in segs
        BEM.Topology._is_fixed_segment(s) && continue
        all(s.frozen) && continue
        for p in s.verts
            push!(rs, norm(p - origin))
        end
    end
    return rs
end

function _roundness(d, origin)
    rs = _hole_radii(d, origin)
    isempty(rs) && return NaN, NaN, NaN
    return mean(rs), std(rs), maximum(rs) / max(minimum(rs), 1e-12)
end

println("="^72)
println(" Portela 2012 plate-with-hole  state=$state  maxiter=$maxiter")
println("="^72)

for rat in (ratio == 1.0 ? (1.0, 0.75) : (ratio,))
    d = portela_plate_hole(; ratio=rat, ne_outer=4, ne_hole=4, nint=6, degree=1, E=1.0)
    A0 = design_area(d)
    origin = SVector(0.0, 0.0)
    μ0, σ0, mm0 = _roundness(d, origin)
    opt = PortelaOptions(maxiter=maxiter, npg=10, state=state, param=:radial,
        origin=origin, vmax=0.025, α=0.3, verbose=true, ΔA=0.0)
    t0 = time()
    d, dad, hist = solve_portela!(d, opt)
    dt = time() - t0
    J = elastic_compliance(dad)
    A = design_area(d)
    μ, σ, mm = _roundness(d, origin)
    folded = BEM.Topology._design_folded(d)
    @printf("  ratio=%g  A/A0=%.4f  J=%.4g  r̄=%.4f  std/mean=%.4f→%.4f  max/min=%.3f→%.3f  folded=%s  %.1fs\n",
        rat, A / A0, J, μ, σ0 / μ0, σ / μ, mm0, mm, folded, dt)
    println("    Banichuk: ratio=1 → circle (std/mean → 0); ratio=0.75 → ellipse axis ratio 0.75")
end

if do_heat
    println()
    println(" Laplace analogue: insulated square hole")
    d = portela_heat_hole(; a=0.30, ne=5, nint=6, degree=1)
    A0 = design_area(d)
    origin = SVector(0.5, 0.5)
    μ0, σ0, mm0 = _roundness(d, origin)
    dad0 = bemdata_from_loops(d)
    H_G_full_direct(dad0; npg=8, threaded=false)
    solve(dad0)
    J0 = thermal_conductance(dad0)
    opt = PortelaOptions(maxiter=maxiter, npg=10, state=state, param=:radial,
        origin=origin, vmax=0.03, α=0.3, verbose=true, ΔA=0.0)
    t0 = time()
    d, dad, hist = solve_portela!(d, opt)
    dt = time() - t0
    J = thermal_conductance(dad)
    A = design_area(d)
    μ, σ, mm = _roundness(d, origin)
    folded = BEM.Topology._design_folded(d)
    @printf("  heat  A/A0=%.4f  J/J0=%.4f  std/mean=%.4f→%.4f  max/min=%.3f→%.3f  folded=%s  %.1fs\n",
        A / A0, J / J0, σ0 / μ0, σ / μ, mm0, mm, folded, dt)
end
