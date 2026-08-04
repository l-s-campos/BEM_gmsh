#!/usr/bin/env julia
# Diffuse–advective DIBEM — Pinheiro §8.2.1
#   julia --project=. scripts/diffuse_advective_c8e1.jl
#
#   α∇²u = v·∇u,  v = (m y, m x),  u = exp(m x y)

using Printf
using BEM

include(joinpath(@__DIR__, "..", "data", "Laplace", "Laplace_dad.jl"))

function main()
    println("="^72)
    println("Diffuse–advective DIBEM — C8E1 (Pinheiro Ch.8 §8.2.1)")
    println("  v = (m y, m x),  u = exp(m x y),  all-Dirichlet")
    println("="^72)

    println("\n--- NPI convergence (ndiv=16, m=1) ---")
    @printf("  %6s  %6s  %12s  %12s\n", "ndiv", "NPI", "flux err %", "mean |Δu|")
    for n_int in (3, 5, 7, 9)
        msh = Base.invokelatest(quadrado; ndiv=16, show=false, nome="da_c$n_int")
        dad = setup_da_c8e1(msh; m=1.0, n_int=n_int)
        res = test_da_c8e1(dad; m=1.0, npg=14, verbose=false)
        @printf("  %6d  %6d  %12.4f  %12.4e\n", 16, dad.ni, res.flux_err_pct, res.err_u)
    end

    println("\n--- Parametric m (ndiv=20, NPI=49) ---")
    @printf("  %6s  %12s  %12s\n", "m", "flux err %", "mean |Δu|")
    for m in 1:6
        msh = Base.invokelatest(quadrado; ndiv=20, show=false, nome="da_m$m")
        dad = setup_da_c8e1(msh; m=float(m), n_int=7)
        res = test_da_c8e1(dad; m=float(m), npg=14, verbose=false)
        @printf("  %6d  %12.4f  %12.4e\n", m, res.flux_err_pct, res.err_u)
    end
    println("\nDone.")
end

main()
