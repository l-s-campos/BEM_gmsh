# Quick beam check: pin-roller SS, 2×static amplitude and T1.
#   julia --project=. scripts/check_beam_amp_period.jl
using DrWatson
@quickactivate :BEM
using LinearAlgebra, Printf

include(datadir("Laplace", "Laplace_dad.jl"))
include(datadir("Laplace", "potencial_problems.jl"))
include(datadir("Laplace", "wave_propagation.jl"))
include(datadir("elastico", "iso", "bar_sudden.jl"))
include(datadir("elastico", "iso", "analytical_elastodynamics.jl"))
include(datadir("elastico", "iso", "elastodynamics_problems.jl"))

const NPG = 12
const NTS = 160
const MESH = 200

function main()
    @printf("%-32s %8s %8s %8s %8s %8s %8s %8s\n",
        "case", "δ", "2δ", "u_pk", "amp/2δ", "T1", "T_num", "T/T1")
    cases = get(ENV, "BEAM_CASES", "cantilever")
    names = if cases == "all"
        BEAM_PROBLEMS
    elseif cases == "ss"
        filter(n -> occursin("ss_", string(n)), BEAM_PROBLEMS)
    else
        filter(n -> occursin("cantilever", string(n)), BEAM_PROBLEMS)
    end
    for pname in names
        dad, meta = elastodynamics_problem(pname; mesh_tag=MESH)
        H_G_full_direct(dad; npg=NPG, threaded=false)
        build_cell_mass(dad; npg=NPG)
        dt = meta.tf / NTS
        U = solve_Houbolt(dad, dt, meta.tf)
        ip, _ = _elasto_probe_id(dad, meta.probe)
        un = -collect(_elasto_hist(U, dad, ip; comp=meta.comp))  # downward +
        t = collect(dad.time)
        ap = beam_amp_period(t, un)
        rA = ap.u_peak / meta.amp
        rT = ap.T_num / meta.T
        @printf("nt=%d  max|u|=%.3e  %s\n", dad.nt, maximum(abs, un), meta.notes)
        @printf("%-32s %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f\n",
            string(pname), meta.δ, meta.amp, ap.u_peak, rA, meta.T, ap.T_num, rT)
    end
    return nothing
end

main()
