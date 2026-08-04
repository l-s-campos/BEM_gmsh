#!/usr/bin/env julia
# Compare RBF particular-solution methods for Poisson + BEM
#   julia --project=. scripts/poisson_rbf_bem_compare.jl

using Printf
using LinearAlgebra
using Statistics: mean
using BEM

include(joinpath(@__DIR__, "..", "data", "Laplace", "Laplace_dad.jl"))

function main()
    println("="^72)
    println("Poisson RBF–BEM manufactured-solution comparison")
    println("  ∇²u = f,  u = g on ∂Ω  (unit square)")
    println("="^72)

    cases = [
        (name = "u=x²+y² (f=4)", u = p -> p[1]^2 + p[2]^2, f = 4.0,
            basis = PHS(3; poly_deg = 1)),
        (name = "u=sin(πx)sin(πy)", u = p -> sin(π * p[1]) * sin(π * p[2]),
            f = p -> -2π^2 * sin(π * p[1]) * sin(π * p[2]),
            basis = PHS(5; poly_deg = 1)),
    ]
    methods = (:global, :local, :pu)
    ndivs = (6, 8, 10)

    for case in cases
        println("\n## $(case.name)")
        @printf("%-6s %-10s %12s %12s %10s\n", "ndiv", "method", "RMSE", "max|e|", "time[s]")
        println("-"^56)
        for nd in ndivs
            msh = quadrado(ndiv = nd, show = false, nome = "cmp_$(nd)")
            dad0 = format2d(msh, Laplace(1.0); pontointerno = true)
            for i in 1:dad0.n
                dad0.BC[i] = 0
                dad0.BV[i] = case.u(dad0.Nodes[i])
            end
            res = compare_poisson_rbf_bem(dad0, case.f, case.u;
                methods = methods, basis = case.basis, npg = 14)
            for r in res
                @printf("%-6d %-10s %12.3e %12.3e %10.3f\n",
                    nd, r.method, r.rmse, r.maxerr, r.time)
            end
        end
    end
    println("\nDone.")
end

main()
