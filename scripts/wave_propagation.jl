# Scalar wave-propagation benchmarks (DIBEM + full-system Houbolt / DiffEq)
#
#   julia --project=. scripts/wave_propagation.jl
#   WAVE_CASE=bar_sudden WAVE_NDIV=12 julia --project=. scripts/wave_propagation.jl
#   WAVE_SOLVER=diffeq WAVE_CASE=bar_sudden julia --project=. scripts/wave_propagation.jl
using DrWatson
@quickactivate :BEM
using LinearAlgebra
using Printf

include(datadir("Laplace", "Laplace_dad.jl"))
include(datadir("Laplace", "potencial_problems.jl"))
include(datadir("Laplace", "wave_propagation.jl"))

const CASE = Symbol(get(ENV, "WAVE_CASE", "bar_sudden"))
const NDIV = parse(Int, get(ENV, "WAVE_NDIV", "12"))
const NINT = parse(Int, get(ENV, "WAVE_NINT", "6"))
const TF = parse(Float64, get(ENV, "WAVE_TF", "2.0"))
const DT = parse(Float64, get(ENV, "WAVE_DT", "0.05"))
const SOLVER = get(ENV, "WAVE_SOLVER", "houbolt")  # houbolt | diffeq | both

function run_case(case::Symbol; ndiv=NDIV, n_int=NINT, tf=TF, Δt=DT, ω=1.0, solver=SOLVER)
    println("="^64)
    println(" wave propagation — $case  solver=$solver")
    println(" ndiv=$ndiv  n_int=$n_int  Δt=$Δt  tf=$tf")
    println("="^64)

    dad, meta = wave_problem(case; ndiv=ndiv, n_int=n_int, ω=ω)
    println(dad)
    println("  note: ", meta.note)

    t_asm = @elapsed begin
        H_G_full_direct(dad; npg=10, threaded=false)
        DIBEM(dad; rbf=PHS(3; poly_deg=0))
    end
    println("  assembly+DIBEM: $(round(t_asm; digits=2)) s")

    if case === :bar_periodic
        @warn "bar_periodic: time-dependent Neumann not in drivers yet (static q)."
    elseif case === :ricker
        @warn "ricker: body force not wired; free response smoke only."
    end

    results = Dict{Symbol,Any}()
    solvers = solver == "both" ? ("houbolt", "diffeq") : (solver,)

    for s in solvers
        dad_s = deepcopy(dad)
        # re-attach operators (deepcopy may drop cache methods — reassemble cheaply if needed)
        if !has_cache(dad_s, :H)
            H_G_full_direct(dad_s; npg=10, threaded=false)
            DIBEM(dad_s; rbf=PHS(3; poly_deg=0))
        end
        t_sol = @elapsed begin
            if s == "houbolt"
                solve_Houbolt(dad_s, Δt, tf)
            elseif s == "diffeq"
                du0 = case === :membrane_v0 ? wave_initial_velocity!(dad_s, meta) : nothing
                solve_transient_o2(dad_s, Δt, tf; du0=du0, abstol=1e-5, reltol=1e-5, progress=false)
            else
                error("WAVE_SOLVER must be houbolt|diffeq|both (got $s)")
            end
        end
        println("  $s: $(round(t_sol; digits=2)) s, steps=$(size(dad_s.T, 2))")

        err = NaN
        if meta.ana !== nothing && case !== :ricker
            pts = vcat(dad_s.Nodes, dad_s.internalNodes)
            ip = argmin(norm(p - meta.probe) for p in pts)
            tgrid = dad_s.t isa AbstractRange ? collect(dad_s.t) : dad_s.t
            unum = dad_s.T[ip, :]
            uana = [meta.ana.u(pts[ip]; t=tt) for tt in tgrid]
            i0 = min(length(tgrid), max(2, length(tgrid) ÷ 10))
            den = norm(uana[i0:end])
            err = den > 0 ? norm(unum[i0:end] .- uana[i0:end]) / den : Inf
            @printf("  probe (%.3f,%.3f)  rel_err≈%.3e  u_end=%.4g  ana=%.4g\n",
                pts[ip][1], pts[ip][2], err, unum[end], uana[end])
        end
        results[Symbol(s)] = (; dad=dad_s, err)
    end
    return results
end

function main()
    if CASE === :all
        for c in wave_problem_names()
            try
                run_case(c)
            catch e
                @error "case $c failed" exception = (e, catch_backtrace())
            end
        end
    else
        run_case(CASE)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
