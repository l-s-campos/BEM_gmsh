#!/usr/bin/env julia
# =============================================================================
# Compare HBS/HSS formats and direct-solver backends (Martinsson Ch. 14–18)
#
# HBS (Martinsson) and HSS (Xia/Gu) are the *same nested weak-admissibility
# format*. Differences that matter in practice are the *factorization backend*:
#   - :scattering  — Ch. 18 discrete scattering matrices (native nested)
#   - :hodlr       — expand nested generators → HODLR ULV (legacy)
#   - HODLR itself — non-nested weak admissibility (independent bases / block)
#   - dense LU     — reference
#
# Run from repo root:
#   julia --project=. scripts/compare_hbs_hss.jl
# =============================================================================

using LinearAlgebra
using Printf
using StaticArrays
using Random

using BEM  # reexports HMatrices

Random.seed!(0)

function kernel_matrix(pts)
    n = length(pts)
    A = zeros(n, n)
    @inbounds for j in 1:n, i in 1:n
        A[i, j] = i == j ? 0.0 : log(norm(pts[i] - pts[j]) + 1e-15)
    end
    return A + 5.0 * I
end

points_1d(n) = [SVector(i / (n + 1)) for i in 1:n]

function bench_one(n; rtol = 1e-8, nmax = 32)
    pts = points_1d(n)
    A = kernel_matrix(pts)
    tree = ClusterTree(pts, CardinalitySplitter(; nmax))
    b = A * ones(n)
    xref = A \ b

    t_hss_asm = @elapsed Hss = assemble_hss(A, tree; rtol, method = :dense)
    t_hbs_asm = @elapsed Hhbs = assemble_hbs(A, tree; rtol, method = :dense)
    t_hod_asm = @elapsed Hhod = assemble_hodlr(A, tree; comp = PartialACA(; rtol))

    # matvec
    x = randn(n)
    t_hss_mv = @elapsed mul!(zeros(n), Hss, x)
    t_hod_mv = @elapsed mul!(zeros(n), Hhod, x)
    err_hss_mv = norm(Hss * x - A * x) / norm(A * x)
    err_hod_mv = norm(Hhod * x - A * x) / norm(A * x)

    # factor + solve
    t_sc_fac = @elapsed Fsc = lu(Hss; method = :scattering)
    t_sc_sol = @elapsed xsc = ldiv!(Fsc, copy(b))
    err_sc = norm(xsc - xref) / norm(xref)

    t_ulv_fac = @elapsed Fulv = lu(Hss; method = :hodlr, rtol)
    t_ulv_sol = @elapsed xulv = ldiv!(Fulv, copy(b))
    err_ulv = norm(xulv - xref) / norm(xref)

    t_hod_fac = @elapsed Fhod = lu(Hhod; rtol)
    t_hod_sol = @elapsed xhod = ldiv!(Fhod, copy(b))
    err_hod = norm(xhod - xref) / norm(xref)

    t_dense = @elapsed xd = A \ b

    return (;
        n,
        maxrank_hss = maxrank(Hss),
        maxrank_hod = maxrank(Hhod),
        cr_hss = compression_ratio(Hss),
        cr_hod = compression_ratio(Hhod),
        t_hss_asm,
        t_hbs_asm,
        t_hod_asm,
        t_hss_mv,
        t_hod_mv,
        err_hss_mv,
        err_hod_mv,
        t_sc_fac,
        t_sc_sol,
        err_sc,
        t_ulv_fac,
        t_ulv_sol,
        err_ulv,
        t_hod_fac,
        t_hod_sol,
        err_hod,
        t_dense,
        alias_same = typeof(Hss) === typeof(Hhbs),
    )
end

function main()
    println("="^78)
    println("HBS vs HSS — Martinsson Fast Direct Solvers (Ch. 14–18)")
    println("="^78)
    println()
    println("Theory (Martinsson §5.8, §15.8, Ch. 18):")
    println("  • HBS (hierarchically block separable) and HSS (hierarchically")
    println("    semi-separable) are the same nested weak-admissibility class.")
    println("  • Nested bases U,V + sibling couplings B only (not full HODLR).")
    println("  • Ch. 16: interpolative decomposition / recursive skeletonization.")
    println("  • Ch. 18: discrete scattering solve ≡ Ch. 14 Woodbury, but uses")
    println("    S = D̂⁻¹ as the primitive → fewer inversions, better stability.")
    println()
    println("This package: HBSMatrix === HSSMatrix; assemble_hbs === assemble_hss.")
    println("Difference under test = factorization backend, plus HODLR baseline.")
    println()

    ns = [128, 256, 512, 1024]
    results = [bench_one(n) for n in ns]

    @printf("%-6s %8s %8s %10s %10s %10s %10s %10s %10s %10s\n",
        "N", "k_HSS", "k_HOD", "asm_HSS", "fac_scat", "fac_ULV", "fac_HOD",
        "err_scat", "err_ULV", "err_HOD")
    println("-"^96)
    for r in results
        @printf("%-6d %8d %8d %10.3f %10.3f %10.3f %10.3f %10.2e %10.2e %10.2e\n",
            r.n, r.maxrank_hss, r.maxrank_hod,
            r.t_hss_asm, r.t_sc_fac, r.t_ulv_fac, r.t_hod_fac,
            r.err_sc, r.err_ulv, r.err_hod)
    end

    println()
    @printf("%-6s %10s %10s %10s %10s %10s %10s %10s\n",
        "N", "mv_HSS", "mv_HOD", "sol_scat", "sol_ULV", "sol_HOD",
        "dense\\\\", "cr_HSS")
    println("-"^88)
    for r in results
        @printf("%-6d %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f %10.1f\n",
            r.n, r.t_hss_mv, r.t_hod_mv,
            r.t_sc_sol, r.t_ulv_sol, r.t_hod_sol, r.t_dense, r.cr_hss)
    end

    println()
    println("Notes:")
    println("  • asm_HSS ≈ asm_HBS (identical code; alias_same=$(results[1].alias_same))")
    println("  • fac_scat  = Ch. 18 scattering on nested HBS/HSS generators")
    println("  • fac_ULV   = expand HSS → HODLR then recursive ULV")
    println("  • fac_HOD   = assemble+factor plain HODLR (independent off-diagonal bases)")
    println("  • Nested HSS/HBS usually stores less than HODLR (see cr_HSS vs cr_HOD).")
    println("  • Scattering factor cost should beat HSS→HODLR ULV (no basis expansion).")
    r = results[end]
    @printf("  • At N=%d: scat/ULV factor speedup ≈ %.2fx,  scat err=%.2e, ULV err=%.2e\n",
        r.n, r.t_ulv_fac / max(r.t_sc_fac, 1e-12), r.err_sc, r.err_ulv)
    @printf("  • matvec err HSS=%.2e  HODLR=%.2e\n", r.err_hss_mv, r.err_hod_mv)
    return results
end

main()
