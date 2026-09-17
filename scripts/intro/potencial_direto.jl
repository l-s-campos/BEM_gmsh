# potencial_direto — Laplace benchmarks (port of BEM.jl atual)
# `scripts/potencial/potencial_direto.jl`
#
# Improvements over the legacy script:
#   - Gmsh + format2d (no hand-rolled NOS/ELEM)
#   - Dumont near-field via H_G_full_direct
#   - optional H-matrix path (H_G_Hmat)
#   - order × h-refinement sweep with boundary / interior errors
#   - TSV results (no extra deps)
#
# Problems (same set as the paper script):
#   potencial1d, laquini1, laquini2, laquini3, quarto_circ, placa_moulton
#
# Run from project root:
#   julia --project=. scripts/potencial_direto.jl
#
# ENV knobs:
#   POT_NOME=all|potencial1d|…     problem filter
#   POT_NDIVS=8,16,32              comma-separated boundary divisions
#   POT_TIPOS=1,2,3                collocation orders (tipo = p, n_gauss=p+1)
#   POT_NPG=16                     near-field GL order
#   POT_METHODS=dense,hmat         assembly backends
#   POT_ATOL=1e-6                  H-matrix ACA tolerance
#   POT_INTERIOR=1                 1 → pontointerno + interior error
#   POT_OUT=plots/potencial_direto.tsv

using DrWatson
@quickactivate :BEM
using LinearAlgebra
using Printf
using Dates

include(datadir("Laplace", "Laplace_dad.jl"))
include(datadir("Laplace", "potencial_problems.jl"))

const NOME     = get(ENV, "POT_NOME", "all")
const NPG      = parse(Int, get(ENV, "POT_NPG", "16"))
const ATOL     = parse(Float64, get(ENV, "POT_ATOL", "1e-6"))
const INTERIOR = parse(Bool, get(ENV, "POT_INTERIOR", "1"))
const OUTPATH  = get(ENV, "POT_OUT", projectdir("plots", "potencial_direto.tsv"))

_parse_ints(s) = parse.(Int, split(s, ','; keepempty=false))
const NDIVS   = _parse_ints(get(ENV, "POT_NDIVS", "8,16,32"))
const TIPOS   = _parse_ints(get(ENV, "POT_TIPOS", "1,2,3"))
const METHODS = Symbol.(split(get(ENV, "POT_METHODS", "dense"), ','; keepempty=false))

# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------

function make_case_builders()
    return [
        (name="potencial1d",
         mesh=(ndiv, tipo) -> potencial1d_mesh(; ndiv=ndiv, ordem=1, nome="pot_p1d_$(ndiv)_$(tipo)"),
         ana=ana_potencial1d(), keep=false),
        (name="laquini1",
         mesh=(ndiv, tipo) -> laquini1_mesh(; ndiv=ndiv, ordem=1, nome="pot_laq1_$(ndiv)_$(tipo)"),
         ana=ana_laquini1(), keep=false),
        (name="laquini2",
         mesh=(ndiv, tipo) -> laquini2_mesh(; ndiv=ndiv, ordem=1, nome="pot_laq2_$(ndiv)_$(tipo)"),
         ana=ana_laquini2(), keep=false),
        (name="laquini3",
         mesh=(ndiv, tipo) -> laquini3_mesh(; ndiv=ndiv, ordem=1, nome="pot_laq3_$(ndiv)_$(tipo)"),
         ana=ana_laquini3(), keep=false),
        (name="quarto_circ",
         mesh=(ndiv, tipo) -> quarto_circ_mesh(; ndiv=ndiv, ordem=1, nome="pot_qcirc_$(ndiv)_$(tipo)"),
         ana=ana_quarto_circ(), keep=false),
        (name="placa_moulton",
         mesh=(ndiv, tipo) -> placa_moulton_mesh(; ndiv=ndiv, ordem=1, nome="pot_mou_$(ndiv)_$(tipo)"),
         ana=ana_moulton(), keep=true),
    ]
end

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

function _rel(a, b)
    nb = norm(b)
    return nb > 0 ? norm(a .- b) / nb : (norm(a) == 0 ? 0.0 : Inf)
end

function _mean_rel(a, b)
    sb = sum(abs, b)
    return sb > 0 ? sum(abs, a .- b) / sb : (sum(abs, a) == 0 ? 0.0 : Inf)
end

function metrics(dad, ana)
    n, nt = dad.n, dad.nt
    T = dad.T
    Tb = @view T[1:n]
    Tana_b = [float(ana.u(p)) for p in dad.Nodes]
    err_b = _rel(Tb, Tana_b)
    em_b  = _mean_rel(Tb, Tana_b)

    err_i = NaN
    em_i  = NaN
    if nt > n && !isempty(dad.internalNodes)
        Ti = @view T[n+1:nt]
        Tana_i = [float(ana.u(p)) for p in dad.internalNodes]
        err_i = _rel(Ti, Tana_i)
        em_i  = _mean_rel(Ti, Tana_i)
    end
    err_all = rel_error(dad)
    return (; err_all, err_b, em_b, err_i, em_i)
end

# ---------------------------------------------------------------------------
# One solve
# ---------------------------------------------------------------------------

function run_one(name, msh_path, ana; method::Symbol=:dense, tipo::Int=2,
                 npg::Int=NPG, keep::Bool=false, interior::Bool=INTERIOR, atol=ATOL)
    dad = format2d(msh_path, Laplace(1.0); tipo=tipo, pontointerno=interior)
    if keep
        apply_bc_keep_type!(dad, ana)
    else
        attach_analytical!(dad, ana)
    end

    t_asm = @elapsed begin
        if method === :dense
            H_G_full_direct(dad; npg=npg)
        elseif method === :hmat
            H_G_Hmat(dad; atol=atol, nmax=32, format=:H)
        else
            error("unknown method=$method (use :dense or :hmat)")
        end
    end
    t_sol = @elapsed solve(dad)
    m = metrics(dad, ana)
    return (;
        name, method=String(method), n=dad.n, ni=dad.ni, nt=dad.nt, tipo, npg,
        t_asm, t_sol, t_total=t_asm + t_sol,
        err_all=m.err_all, err_b=m.err_b, em_b=m.em_b, err_i=m.err_i, em_i=m.em_i,
    )
end

# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

const HEADER = ("prob", "method", "ndiv", "tipo", "n", "ni", "nt",
                "t_asm", "t_sol", "t_total",
                "err_all", "err_b", "em_b", "err_i", "em_i")

function write_tsv(path, rows)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, join(HEADER, '\t'))
        for r in rows
            @printf(io,
                "%s\t%s\t%d\t%d\t%d\t%d\t%d\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\n",
                r.name, r.method, r.ndiv, r.tipo, r.n, r.ni, r.nt,
                r.t_asm, r.t_sol, r.t_total,
                r.err_all, r.err_b, r.em_b,
                isnan(r.err_i) ? NaN : r.err_i,
                isnan(r.em_i)  ? NaN : r.em_i)
        end
    end
    return path
end

function print_table(rows)
    println("-"^96)
    @printf("%-14s %-6s %4s %4s %6s %10s %10s %10s %10s\n",
        "prob", "meth", "ndiv", "tipo", "n", "err_b", "err_i", "t_asm", "t_sol")
    for r in rows
        @printf("%-14s %-6s %4d %4d %6d %10.2e %10.2e %10.3f %10.3f\n",
            r.name, r.method, r.ndiv, r.tipo, r.n, r.err_b,
            isnan(r.err_i) ? NaN : r.err_i, r.t_asm, r.t_sol)
    end
    println("="^96)
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

function main()
    println("="^96)
    println(" potencial_direto — Laplace benchmarks (Gmsh + Dumont / optional Hmat)")
    println(" ndivs=$NDIVS  tipos=$TIPOS  methods=$METHODS  npg=$NPG  interior=$INTERIOR")
    println(" filter=$NOME  out=$OUTPATH")
    println("="^96)

    builders = make_case_builders()
    if NOME != "all"
        builders = filter(c -> c.name == NOME, builders)
        isempty(builders) && error("Unknown POT_NOME=$NOME; choose: " *
            join(getfield.(make_case_builders(), :name), ", "))
    end

    rows = NamedTuple[]
    for c in builders, ndiv in NDIVS, tipo in TIPOS, meth in METHODS
        tag = "$(c.name) ndiv=$ndiv tipo=$tipo meth=$meth"
        print("▸ $tag … "); flush(stdout)
        try
            msh = c.mesh(ndiv, tipo)
            r0 = run_one(c.name, msh, c.ana; method=meth, tipo=tipo, keep=c.keep)
            r = merge(r0, (; ndiv=ndiv))
            push!(rows, r)
            @printf("n=%4d ni=%4d err_b=%.2e err_i=%.2e t=%.2fs\n",
                r.n, r.ni, r.err_b, isnan(r.err_i) ? NaN : r.err_i, r.t_total)
        catch e
            println("FAILED")
            showerror(stdout, e, catch_backtrace()); println()
        end
    end

    print_table(rows)
    if !isempty(rows)
        p = write_tsv(OUTPATH, rows)
        println(" wrote $p  ($(length(rows)) rows)  $(Dates.now())")
    end
    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
