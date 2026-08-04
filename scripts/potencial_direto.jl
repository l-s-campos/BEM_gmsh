# Classic Laplace / potential benchmarks from BEM.jl atual
# `scripts/potencial/potencial_direto.jl`, rewritten for Gmsh + format2d.
#
# Problems:
#   1. potencial1d   — unit square, T = x
#   2. laquini1      — Fourier series (top Neumann)
#   3. laquini2      — Fourier series (mixed Neumann)
#   4. laquini3      — top Dirichlet 1
#   5. quarto_circ    — quarter annulus radial conduction
#   6. placa_moulton — √r cos(θ/2) singularity
#
# Run from project root:
#   julia --project=. scripts/potencial_direto.jl
# Optional: ENV["POT_NDIV"]="20"  ENV["POT_TIPO"]="2"  ENV["POT_NPG"]="16"

using DrWatson
@quickactivate :BEM
using LinearAlgebra
using Printf
using Dates

include(datadir("Laplace", "Laplace_dad.jl"))
include(datadir("Laplace", "potencial_problems.jl"))

const NDIV = parse(Int, get(ENV, "POT_NDIV", "16"))
const TIPO = parse(Int, get(ENV, "POT_TIPO", "2"))   # format2d: n_gauss = tipo+1
const NPG = parse(Int, get(ENV, "POT_NPG", "16"))
const NOME = get(ENV, "POT_NOME", "all")             # or potencial1d, laquini1, ...

"""Solve one Laplace problem and return relative L2 error on all collocation nodes."""
function run_problem(name::String, msh_path::String, ana::AnalyticalSolution;
                     keep_type_bc::Bool=false, npg=NPG, tipo=TIPO)
    dad = format2d(msh_path, Laplace(1.0); tipo=tipo, pontointerno=true)
    if keep_type_bc
        apply_bc_keep_type!(dad, ana)
    else
        # mesh already carries constant BC values; still attach ana for error
        attach_analytical!(dad, ana)
    end
    t_asm = @elapsed H_G_full_direct(dad; npg=npg, threaded=false)
    t_sol = @elapsed solve(dad)
    err = rel_error(dad)
    # boundary-only error
    Tana_b = [float(ana.u(p)) for p in dad.Nodes]
    err_b = norm(dad.T[1:dad.n] .- Tana_b) / max(norm(Tana_b), eps())
    return (;
        name,
        n=dad.n,
        ni=length(dad.internalNodes),
        tipo,
        t_asm,
        t_sol,
        err,
        err_b,
        dad,
        ana,
    )
end

function make_cases(; ndiv=NDIV)
    return [
        (
            name="potencial1d",
            mesh=() -> potencial1d_mesh(; ndiv=ndiv, nome="pot_p1d"),
            ana=ana_potencial1d(),
            keep=false,
        ),
        (
            name="laquini1",
            mesh=() -> laquini1_mesh(; ndiv=ndiv, nome="pot_laq1"),
            ana=ana_laquini1(),
            keep=false,
        ),
        (
            name="laquini2",
            mesh=() -> laquini2_mesh(; ndiv=ndiv, nome="pot_laq2"),
            ana=ana_laquini2(),
            keep=false,
        ),
        (
            name="laquini3",
            mesh=() -> laquini3_mesh(; ndiv=ndiv, nome="pot_laq3"),
            ana=ana_laquini3(),
            keep=false,
        ),
        (
            name="quarto_circ",
            mesh=() -> quarto_circ_mesh(; ndiv=ndiv, nome="pot_qcirc"),
            ana=ana_quarto_circ(),
            keep=false,
        ),
        (
            name="placa_moulton",
            mesh=() -> placa_moulton_mesh(; ndiv=ndiv, nome="pot_moulton"),
            ana=ana_moulton(),
            keep=true,   # fill non-constant Neumann from analytical
        ),
    ]
end

function main()
    println("="^72)
    println(" potencial_direto — Laplace benchmarks (Gmsh + format2d + H_G_full_direct)")
    println(" ndiv=$NDIV  tipo=$TIPO  npg=$NPG  filter=$NOME")
    println("="^72)

    cases = make_cases()
    if NOME != "all"
        cases = filter(c -> c.name == NOME, cases)
        isempty(cases) && error("Unknown POT_NOME=$NOME; choose one of: " *
                                join(getfield.(make_cases(), :name), ", "))
    end

    rows = NamedTuple[]
    for c in cases
        print("▸ $(c.name) … ")
        flush(stdout)
        try
            msh = c.mesh()
            r = run_problem(c.name, msh, c.ana; keep_type_bc=c.keep)
            push!(rows, r)
            @printf("n=%4d  ni=%4d  err=%.3e  err_b=%.3e  t_asm=%.3fs  t_sol=%.3fs\n",
                r.n, r.ni, r.err, r.err_b, r.t_asm, r.t_sol)
        catch e
            println("FAILED")
            showerror(stdout, e, catch_backtrace())
            println()
        end
    end

    println("-"^72)
    @printf("%-14s %6s %8s %10s %10s\n", "problem", "n", "err", "t_asm", "t_sol")
    for r in rows
        @printf("%-14s %6d %8.2e %10.3f %10.3f\n", r.name, r.n, r.err, r.t_asm, r.t_sol)
    end
    println("="^72)
    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
