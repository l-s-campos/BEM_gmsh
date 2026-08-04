#!/usr/bin/env julia
# Pinheiro thesis §8.2.1 — DIBEM alternative, variable velocity (div v = 0)
#   julia --project=. scripts/dibem_c8e1_demo.jl
#
#   α∇²u = v·∇u,  v = (m y, m x),  u = exp(m x y) on unit square

using Printf
using LinearAlgebra
using BEM

include(joinpath(@__DIR__, "..", "data", "Laplace", "Laplace_dad.jl"))

function main()
    println("="^72)
    println("DIBEM Alternative — C8E1 (Pinheiro Ch.8 §8.2.1)")
    println("  v = (m y, m x),  u = exp(m x y),  all-Dirichlet")
    println("="^72)

    println("\n--- Convergence in NPI (ndiv=16, m=1) ---")
    @printf("  %6s  %6s  %12s  %12s\n", "ndiv", "NPI", "flux err %", "mean |Δu|")
    for n_int in (3, 5, 7, 9)
        msh = Base.invokelatest(quadrado; ndiv=16, show=false, nome="c8e1_c$n_int")
        dad = setup_dibem_c8e1(msh; m=1.0, n_int=n_int)
        res = test_dibem_c8e1(dad; m=1.0, npg=14, verbose=false)
        @printf("  %6d  %6d  %12.4f  %12.4e\n", 16, dad.ni, res.flux_err_pct, res.err_u)
    end

    println("\n--- Parametric m (ndiv=20, NPI=7²) ---")
    @printf("  %6s  %12s  %12s\n", "m", "flux err %", "mean |Δu|")
    for m in 1:6
        msh = Base.invokelatest(quadrado; ndiv=20, show=false, nome="c8e1_m$m")
        dad = setup_dibem_c8e1(msh; m=float(m), n_int=7)
        res = test_dibem_c8e1(dad; m=float(m), npg=14, verbose=false)
        @printf("  %6d  %12.4f  %12.4e\n", m, res.flux_err_pct, res.err_u)
    end

    println("\nDone.")
end

main()
