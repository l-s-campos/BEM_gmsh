#!/usr/bin/env julia
# Método Modal Modificado (MMM) demo — fixed membrane free vibration + forced edge
#   julia --project=. scripts/mmm_membrane_demo.jl
#
# Reference: Áquila Santos thesis, Ch.4 §4.5–4.6 (Prodonoff–Zepka MMM)

using Printf
using LinearAlgebra
using Statistics: mean
using StaticArrays
using BEM

include(joinpath(@__DIR__, "..", "data", "Laplace", "Laplace_dad.jl"))

function membrane_dad(; ndiv=10, n_int=5, left_u=0.0, nome="mmm_demo")
    msh = Base.invokelatest(quadrado; ndiv=ndiv, show=false, nome=nome)
    dad = format2d(msh, Laplace(1.0); tipo=1, pontointerno=false)
    dad.BC .= 0
    dad.BV .= 0.0
    # left edge x≈0 → Dirichlet left_u (sudden load case)
    for (i, p) in enumerate(dad.Nodes)
        if p[1] < 1e-9
            dad.BV[i] = left_u
        end
    end
    xs = range(0.12, 0.88; length=n_int)
    empty!(dad.internalNodes)
    append!(dad.internalNodes, [SVector(x, y) for y in xs for x in xs])
    dad.ni = length(dad.internalNodes)
    dad.nt = dad.n + dad.ni
    H_G_full_direct(dad, 14)
    DIBEM(dad)
    return dad
end

function main()
    println("="^72)
    println("MMM — Método Modal Modificado (thesis §4.5)")
    println("="^72)

    # --- free vibration spectrum ---
    dad0 = membrane_dad(; ndiv=12, n_int=6, left_u=0.0, nome="mmm_free")
    sys = build_modal_system(dad0)
    b_mmm = modal_analysis_mmm(sys; nmodes=10)
    b_mmc = modal_analysis_mmc(sys; nmodes=10)
    println("\nNatural frequencies (free membrane, free DOFs = internals):")
    println("  analytical ω_{mn} = π √(m²+n²);  ω11=π√2≈4.4429")
    @printf("  %4s  %12s  %12s  %12s\n", "mode", "ω MMM", "ω MMC", "ω² MMM")
    for k in 1:min(6, length(b_mmm.ω))
        @printf("  %4d  %12.5f  %12.5f  %12.5f\n", k, b_mmm.ω[k], b_mmc.ω[k], b_mmm.ω²[k])
    end
    Gbi = b_mmm.Φ̃' * b_mmm.Φ
    @printf("  bi-orthogonality ‖Φ̃ᵀΦ − I‖_F = %.3e\n", norm(Gbi - I))

    # --- pluck IC transient ---
    println("\nTransient free vibration (pluck IC, MMM Houbolt):")
    u0 = zeros(dad0.nt)
    for (k, p) in enumerate(dad0.internalNodes)
        u0[dad0.n + k] = sin(π * p[1]) * sin(π * p[2])
    end
    U, t, basis = solve_mmm!(dad0, 0.01, 2.0; nmodes=8, u0=u0)
    # center-ish internal
    ic = argmin(i -> begin
        p = dad0.internalNodes[i]
        (p[1] - 0.5)^2 + (p[2] - 0.5)^2
    end, eachindex(dad0.internalNodes))
    j = dad0.n + ic
    @printf("  center DOF peak |u| = %.4e  at t=%.3f\n",
        maximum(abs, U[j, :]), t[argmax(abs.(U[j, :]))])

    # --- forced edge + amplitude selection ---
    println("\nForced left edge u=1 (constant) — mode amplitudes §4.6:")
    dadF = membrane_dad(; ndiv=12, n_int=6, left_u=1.0, nome="mmm_forced")
    sysF = build_modal_system(dadF)
    bF = modal_analysis_mmm(sysF; nmodes=12)
    A = mode_amplitudes(bF)
    a_rel = mode_relative_amplitudes(A)
    println("  mode    A_i         a_i %")
    for k in 1:min(8, length(A))
        @printf("  %4d  %10.4e  %8.2f\n", k, A[k], a_rel[k])
    end
    Uf, tf, bf = solve_mmm!(dadF, 0.01, 1.5; nmodes=12, select=:amplitude, nkeep=6)
    @printf("  response with %d amplitude-selected modes, max|U|=%.4e\n",
        size(bf.Φ, 2), maximum(abs, Uf))

    println("\nDone.")
end

main()
