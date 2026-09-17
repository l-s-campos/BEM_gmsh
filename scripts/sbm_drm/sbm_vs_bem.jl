# sbm_vs_bem — SBM (BEMdata + Guiggiani OIFs) vs classical dense BEM
#
# Laplace benchmarks on the unit square.
#
# Run from project root:
#   julia --project=. scripts/sbm_vs_bem.jl
#
# ENV knobs:
#   SBM_NDIVS=8,12,16,24
#   SBM_PROBS=Tx,quad          # Tx = T=x, quad = T=x²−y²
#   SBM_NPG=16
#   SBM_OUT=plots/sbm_vs_bem.tsv

using DrWatson
@quickactivate :BEM
using LinearAlgebra
using Printf
using StaticArrays
using Statistics

include(datadir("Laplace", "Laplace_dad.jl"))

const NDIVS = parse.(Int, split(get(ENV, "SBM_NDIVS", "8,12,16,24"), ','; keepempty=false))
const PROBS = Symbol.(split(get(ENV, "SBM_PROBS", "Tx,quad"), ','; keepempty=false))
const NPG   = parse(Int, get(ENV, "SBM_NPG", "16"))
const OUT   = get(ENV, "SBM_OUT", projectdir("plots", "sbm_vs_bem.tsv"))

function _ana(prob::Symbol)
    if prob === :Tx
        return ana_laplace_linear(; direction=SA[1.0, 0.0], k=1.0)
    elseif prob === :quad
        return ana_laplace_quadratic(; k=1.0)
    else
        error("unknown problem $prob (use Tx or quad)")
    end
end

"""Keep mesh BC types; fill values from analytics."""
function _apply_mixed_ana!(dad, ana)
    for i in 1:dad.n
        if dad.BC[i] == 0
            dad.BV[i] = float(ana.u(dad.Nodes[i]))
        else
            dad.BV[i] = float(ana.q(dad.Nodes[i], dad.Normal[i]))
        end
    end
    attach_analytical!(dad, ana)
    return dad
end

function run_row(prob::Symbol, ndiv::Int; npg::Int=NPG)
    ana = _ana(prob)
    msh = quadrado(ndiv=ndiv, show=false, nome="sbm_$(prob)_$ndiv", ordem=1)
    dad = format2d(msh, Laplace(1.0); tipo=1, pontointerno=false)
    _apply_mixed_ana!(dad, ana)

    cmp = compare_sbm_bem(dad, ana; npg=npg, threaded=false)
    return (;
        prob=String(prob),
        ndiv,
        n=cmp.n,
        err_bem=cmp.err_bem,
        err_sbm=cmp.err_sbm,
        diff_T=cmp.diff_T,
        t_bem=cmp.t_bem,
        t_sbm=cmp.t_sbm,
        t_bem_asm=cmp.t_bem_asm,
        t_sbm_asm=cmp.t_sbm_asm,
        Gii_mean=mean(cmp.u_ii),
        Gii_std=std(cmp.u_ii),
    )
end

const HEADER = ("prob", "ndiv", "n",
                "err_bem", "err_sbm", "diff_T",
                "t_bem", "t_sbm", "t_bem_asm", "t_sbm_asm",
                "Gii_mean", "Gii_std")

function main()
    rows = NamedTuple[]
    for prob in PROBS, ndiv in NDIVS
        r = run_row(prob, ndiv)
        push!(rows, r)
        @printf("%-6s ndiv=%2d  n=%3d  err_BEM=%.3e  err_SBM=%.3e  |ΔT|=%.3e  t_BEM=%.3fs  t_SBM=%.3fs\n",
                r.prob, r.ndiv, r.n, r.err_bem, r.err_sbm, r.diff_T, r.t_bem, r.t_sbm)
    end

    mkpath(dirname(OUT))
    open(OUT, "w") do io
        println(io, join(HEADER, '\t'))
        for r in rows
            println(io, join((r.prob, r.ndiv, r.n,
                              r.err_bem, r.err_sbm, r.diff_T,
                              r.t_bem, r.t_sbm, r.t_bem_asm, r.t_sbm_asm,
                              r.Gii_mean, r.Gii_std), '\t'))
        end
    end
    println("wrote $OUT")
    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
