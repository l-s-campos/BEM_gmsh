# helmholtz_H_paper — frequency-domain Helmholtz via Laplace correlato
# Port of BEM.jl atual `scripts/helmholtz/helmholtz_H_paper.jl`
#
# Legacy paper compared three assembly styles (calc_HeG / calc_HeGd / H-matrix)
# of the correlato system
#
#     (H + κ² M) T = G q ,   κ = ω/c
#
# with Laplace fundamentals + DIBEM mass M. Native Complex Helmholtz FS exists
# in BEM_gmsh but dense/Hmat assembly is still Float64 — this script uses the
# supported correlato path (same math as the paper).
#
# Methods
#   :bem    — H_G_full_direct + DIBEM(:dense)   + dense factorize
#   :dihbem — H_G_Hmat        + DIBEM(:hmatrix) + dense factorize of materialized ops
#
# Run:
#   julia --project=. scripts/helmholtz_H_paper.jl
#
# ENV:
#   HELM_PROB=helm1d|helmdirichlet|helmcirculo|all
#   HELM_FR=4                  frequency (κ = FR/c, c=1)
#   HELM_NDIVS=4,8,12,16
#   HELM_TIPO=2
#   HELM_NPG=12
#   HELM_NINT_FAC=1.5          internal grid ≈ fac·nelem per side
#   HELM_METHODS=bem,dihbem
#   HELM_ATOL=1e-6
#   HELM_OUT=plots/helmholtz_H_paper.tsv

using DrWatson
@quickactivate :BEM
using LinearAlgebra
using Printf
using Dates

include(datadir("Laplace", "Laplace_dad.jl"))
include(datadir("Laplace", "potencial_problems.jl"))
include(datadir("Laplace", "wave_propagation.jl"))   # set_internal_grid!
include(datadir("Laplace", "helmholtz_problems.jl"))

const PROB     = get(ENV, "HELM_PROB", "helm1d")
const FR       = parse(Float64, get(ENV, "HELM_FR", "4.0"))
const TIPO     = parse(Int, get(ENV, "HELM_TIPO", "2"))
const NPG      = parse(Int, get(ENV, "HELM_NPG", "12"))
const NINT_FAC = parse(Float64, get(ENV, "HELM_NINT_FAC", "1.5"))
const ATOL     = parse(Float64, get(ENV, "HELM_ATOL", "1e-6"))
const OUTPATH  = get(ENV, "HELM_OUT", projectdir("plots", "helmholtz_H_paper.tsv"))
const C_WAVE   = 1.0

_parse_ints(s) = parse.(Int, split(s, ','; keepempty=false))
const NDIVS    = _parse_ints(get(ENV, "HELM_NDIVS", "4,8,12,16"))
const METHODS  = Symbol.(split(get(ENV, "HELM_METHODS", "bem,dihbem"), ','; keepempty=false))

# ---------------------------------------------------------------------------
# Problem catalogue
# ---------------------------------------------------------------------------

function _problem_spec(name::Symbol, κ::Float64)
    if name === :helm1d
        return (
            name="helm1d",
            mesh=(ndiv) -> helm1d_mesh(; ndiv=ndiv, nome="helm1d_$ndiv"),
            setup! = (dad) -> attach_analytical!(dad, ana_helm1d(κ)),
            ana = ana_helm1d(κ),
            bbox = (0.0, 1.0, 0.0, 1.0),
        )
    elseif name === :helmdirichlet
        κ > π || error("helmdirichlet needs FR=κ > π (got $κ)")
        return (
            name="helmdirichlet",
            mesh=(ndiv) -> helmdirichlet_mesh(; ndiv=ndiv, nome="helmdir_$ndiv"),
            setup! = (dad) -> begin
                apply_helmdirichlet_bc!(dad)
                attach_analytical!(dad, ana_helmdirichlet(κ))
            end,
            ana = ana_helmdirichlet(κ),
            bbox = (0.0, 1.0, 0.0, 1.0),
        )
    elseif name === :helmcirculo
        return (
            name="helmcirculo",
            mesh=(ndiv) -> helmcirculo_mesh(; ndiv=ndiv, nome="helmcirc_$ndiv"),
            setup! = (dad) -> attach_analytical!(dad, ana_helmcirculo(κ)),
            ana = ana_helmcirculo(κ),
            bbox = (0.0, 1.0, 0.0, 1.0),
        )
    else
        error("unknown problem $name")
    end
end

function selected_problems(κ)
    names = if PROB == "all"
        collect(helmholtz_problem_names())
    else
        [Symbol(PROB)]
    end
    return [_problem_spec(n, κ) for n in names]
end

# ---------------------------------------------------------------------------
# Correlato solve: (H + κ² M) T = G q
# ---------------------------------------------------------------------------

function solve_correlato!(dad::BEMdata{<:Laplace}, κ::Float64;
                          method::Symbol=:bem, npg::Int=NPG, atol::Float64=ATOL,
                          rbf=PHS(3; poly_deg=0))
    t_HG = 0.0
    t_M  = 0.0
    t_sol = 0.0

    if method === :bem
        t_HG = @elapsed H_G_full_direct(dad; npg=npg)
        t_M  = @elapsed DIBEM(dad; method=:dense, rbf=rbf)
    elseif method === :dihbem
        # H_G_Hmat near-field correction needs dad.qsi/dad.w
        qq, ww = gausslegendre(npg)
        set_cache!(dad; qsi=qq, w=ww)
        t_HG = @elapsed H_G_Hmat(dad; atol=atol, nmax=32, format=:H)
        t_M  = @elapsed DIBEM(dad; method=:hmatrix, rbf=rbf, atol=atol, nmax=32)
    else
        error("unknown method=$method (use :bem or :dihbem)")
    end

    # Packed BC blocks: A = [Huu+κ²Muu  −Guq; Hqu+κ²Mqu  −Gqq], x=[T_u; q_q]
    # No densification — whole subblocks via matvec (Hmat/FMM friendly).
    t_sol = @elapsed solve(dad; blocks=true, M=dad.M, κ2=κ^2)

    return (; t_HG, t_M, t_bc=0.0, t_sol, t_total=t_HG + t_M + t_sol, cr_H=NaN)
end

function _rel(a, b)
    nb = norm(b)
    return nb > 0 ? norm(a .- b) / nb : (norm(a) == 0 ? 0.0 : Inf)
end

function _mean_rel(a, b)
    sb = sum(abs, b)
    return sb > 0 ? sum(abs, a .- b) / sb : Inf
end

function helm_errors(dad, ana)
    n, nt = dad.n, dad.nt
    Tb = @view dad.T[1:n]
    Tana_b = [float(ana.u(p)) for p in dad.Nodes]
    err_b = _rel(Tb, Tana_b)
    em_b  = _mean_rel(Tb, Tana_b)

    err_i = NaN; em_i = NaN
    if nt > n
        Ti = @view dad.T[n+1:nt]
        Tana_i = [float(ana.u(p)) for p in dad.internalNodes]
        err_i = _rel(Ti, Tana_i)
        em_i  = _mean_rel(Ti, Tana_i)
    end
    # paper reported interior L2 primarily
    err = isnan(err_i) ? err_b : err_i
    em  = isnan(em_i)  ? em_b  : em_i
    return (; err, em, err_b, em_b, err_i, em_i)
end

# ---------------------------------------------------------------------------
# One (problem × ndiv × method) run
# ---------------------------------------------------------------------------

function run_one(spec; ndiv::Int, method::Symbol, κ::Float64,
                 tipo::Int=TIPO, npg::Int=NPG, nint_fac::Float64=NINT_FAC)
    msh = spec.mesh(ndiv)
    dad = format2d(msh, Laplace(1.0); tipo=tipo, pontointerno=false)

    # controlled internal grid (paper: NPX ≈ 1.5·nelem)
    n_side = max(2, Int(round(nint_fac * ndiv)))
    xmin, xmax, ymin, ymax = spec.bbox
    set_internal_grid!(dad; nx=n_side, ny=n_side,
        x=(xmin, xmax), y=(ymin, ymax), pad=1e-3)

    spec.setup!(dad)

    times = solve_correlato!(dad, κ; method=method, npg=npg)
    errs  = helm_errors(dad, spec.ana)

    return (;
        name=spec.name, method=String(method), ndiv, tipo,
        n=dad.n, ni=dad.ni, nt=dad.nt, κ,
        t_HG=times.t_HG, t_M=times.t_M, t_bc=times.t_bc, t_sol=times.t_sol,
        t_total=times.t_total,
        err=errs.err, em=errs.em, err_b=errs.err_b, err_i=errs.err_i,
    )
end

# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

const HEADER = ("prob", "method", "ndiv", "tipo", "n", "ni", "nt", "kappa",
                "t_HG", "t_M", "t_bc", "t_sol", "t_total",
                "err", "em", "err_b", "err_i")

function write_tsv(path, rows)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, join(HEADER, '\t'))
        for r in rows
            @printf(io,
                "%s\t%s\t%d\t%d\t%d\t%d\t%d\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\n",
                r.name, r.method, r.ndiv, r.tipo, r.n, r.ni, r.nt, r.κ,
                r.t_HG, r.t_M, r.t_bc, r.t_sol, r.t_total,
                r.err, r.em, r.err_b, isnan(r.err_i) ? NaN : r.err_i)
        end
    end
    return path
end

function print_table(rows)
    println("-"^100)
    @printf("%-14s %-7s %4s %6s %6s %10s %10s %10s %10s\n",
        "prob", "method", "ndiv", "n", "ni", "err", "err_i", "t_HG+M", "t_tot")
    for r in rows
        @printf("%-14s %-7s %4d %6d %6d %10.2e %10.2e %10.3f %10.3f\n",
            r.name, r.method, r.ndiv, r.n, r.ni, r.err,
            isnan(r.err_i) ? NaN : r.err_i, r.t_HG + r.t_M, r.t_total)
    end
    println("="^100)
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

function main()
    κ = FR / C_WAVE
    println("="^100)
    println(" helmholtz_H_paper — Laplace correlato  (H + κ² M) T = G q")
    println(" κ=FR/c = $κ   (FR=$FR, c=$C_WAVE)")
    println(" ndivs=$NDIVS  tipo=$TIPO  methods=$METHODS  npg=$NPG  nint_fac=$NINT_FAC")
    println(" filter=$PROB  out=$OUTPATH")
    println("="^100)

    specs = selected_problems(κ)
    rows = NamedTuple[]

    for spec in specs, ndiv in NDIVS, meth in METHODS
        tag = "$(spec.name) ndiv=$ndiv meth=$meth"
        print("▸ $tag … "); flush(stdout)
        try
            r = run_one(spec; ndiv=ndiv, method=meth, κ=κ)
            push!(rows, r)
            @printf("n=%4d ni=%4d err=%.2e err_i=%.2e t=%.2fs\n",
                r.n, r.ni, r.err, isnan(r.err_i) ? NaN : r.err_i, r.t_total)
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
