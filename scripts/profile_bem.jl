#!/usr/bin/env julia
# =============================================================================
# BEM.jl detailed profiling — time, memory, type stability
#   julia --project=. scripts/profile_bem.jl
#   julia --project=. scripts/profile_bem.jl --ndiv=40
# =============================================================================

using Pkg
Pkg.activate(dirname(@__DIR__))

using BEM
using LinearAlgebra
using SparseArrays
using Statistics
using Printf
using InteractiveUtils: @code_warntype, code_warntype
using BenchmarkTools

const PROJECT = dirname(@__DIR__)
include(joinpath(PROJECT, "data", "Laplace", "Laplace_dad.jl"))

function _parse_ndiv(args)
    ndiv = 20
    for a in args
        startswith(a, "--ndiv=") && (ndiv = parse(Int, split(a, "="; limit=2)[2]))
    end
    return ndiv
end

bytes(x) = Base.summarysize(x)

function fmt_bytes(b)
    b < 1024 && return @sprintf("%d B", b)
    b < 1024^2 && return @sprintf("%.2f KiB", b / 1024)
    b < 1024^3 && return @sprintf("%.2f MiB", b / 1024^2)
    return @sprintf("%.2f GiB", b / 1024^3)
end

function section(title)
    println()
    println("="^72)
    println(title)
    println("="^72)
end

function timed(label, f)
    # warm once
    f()
    GC.gc(false)
    t0 = time_ns()
    b0 = Base.gc_bytes()
    r = f()
    t1 = time_ns()
    b1 = Base.gc_bytes()
    dt = (t1 - t0) / 1e9
    db = b1 - b0
    @printf("  %-28s  %8.3f s   alloc %+s\n", label, dt, fmt_bytes(db))
    return r, dt, db
end

"""Infer eltype / concrete-ness of a few hot expressions via @code_warntype capture."""
function warntype_summary(io, f, args...)
    # redirect code_warntype text
    buf = IOBuffer()
    code_warntype(buf, f, typeof.(args))
    txt = String(take!(buf))
    red = count(==("::Any"), split(txt, r"\s+")) + count(x -> occursin("::Any", x), split(txt, '\n'))
    # simpler: count Any and Union in output
    n_any = length(collect(eachmatch(r"\bAny\b", txt)))
    n_union = length(collect(eachmatch(r"\bUnion\{", txt)))
    n_core = length(collect(eachmatch(r"Core\.Box", txt)))
    println(io, "    @code_warntype flags: Any≈$n_any  Union{≈$n_union  Core.Box≈$n_core")
    # show last body lines if suspicious
    if n_any > 15 || n_core > 0
        println(io, "    (elevated — inspect with InteractiveUtils.@code_warntype)")
    end
    return (; n_any, n_union, n_core, txt)
end

function matrix_stats(name, A)
    println("  $name:")
    println("    type     = ", typeof(A))
    println("    size     = ", size(A))
    println("    eltype   = ", eltype(A))
    println("    sizeof   = ", fmt_bytes(bytes(A)))
    if A isa AbstractMatrix{<:Number} && !(A isa HMatrices.HMatrix)
        dens = count(!iszero, A) / length(A)
        @printf("    nnz dens = %.2f%%\n", 100dens)
        if size(A, 1) == size(A, 2) && size(A, 1) ≤ 2000
            κ = try
                cond(Array(A))
            catch
                NaN
            end
            @printf("    cond(A)  = %.3e\n", κ)
        end
    elseif A isa HMatrices.HMatrix
        # leaf stats
        leaves_ = collect(HMatrices.leaves(A))
        nd = count(b -> !HMatrices.isadmissible(b), leaves_)
        nl = count(b -> HMatrices.isadmissible(b), leaves_)
        println("    leaves   = ", length(leaves_), " (dense=$nd, low-rank=$nl)")
    end
end

function profile_pipeline(; ndiv::Int=20)
    section("1. Problem setup  (ndiv=$ndiv)")
    props = Laplace(1.0)
    msh_ref = Ref{String}()
    dad_ref = Ref{Any}()
    (_, t_mesh, _) = timed("quadrado mesh", () -> begin
        msh_ref[] = quadrado(ndiv=ndiv, show=false, nome="prof_quad_$ndiv")
        msh_ref[]
    end)
    msh = msh_ref[]
    (_, t_fmt, _) = timed("format2d", () -> begin
        dad_ref[] = format2d(msh, props; pontointerno=true)
        dad_ref[]
    end)
    dad = dad_ref[]::BEMdata

    println()
    println("  DOFs:")
    println("    boundary nodes n = ", dad.n)
    println("    internal ni      = ", dad.ni)
    println("    total nt         = ", dad.nt)
    println("    elements         = ", length(dad.elements))
    println("    BEMdata size     = ", fmt_bytes(bytes(dad)))

    section("2. Dense assembly H_G_full_direct")
    npg = 20
    # clear any prior
    has_cache(dad, :H) && (dad.cache.H = nothing)
    (_, t_dense, a_dense) = timed("H_G_full_direct(npg=$npg)", () -> H_G_full_direct(dad, npg))
    H = dad.H
    G = dad.G
    matrix_stats("H", H)
    matrix_stats("G", G)
    @printf("  dense assembly rate: %.1f kDOF²/s\n", (dad.n^2) / t_dense / 1e3)
    @printf("  memory H+G: %s (theory Float64 dense 2N² = %s)\n",
        fmt_bytes(bytes(H) + bytes(G)),
        fmt_bytes(2 * dad.n^2 * 8))

    section("3. BC apply + dense solve")
    (_, t_bc, _) = timed("applyBC", () -> applyBC(dad))
    matrix_stats("A (after BC)", dad.A)
    (_, t_sol, a_sol) = timed("solve (dense \\ )", () -> solve(dad))
    println("  T range = ", extrema(dad.T[1:dad.n]))

    section("4. H-matrix assembly (same mesh)")
    dad2 = format2d(msh, props; pontointerno=true)
    try
        (_, t_hmat, a_hmat) = timed("H_G_Hmat", () -> H_G_Hmat(dad2; nmax=40, atol=1e-6))
        if has_cache(dad2, :H)
            matrix_stats("H (Hmat)", dad2.H)
        end
        if has_cache(dad2, :A) || has_cache(dad2, :H)
            (_, t_hs, _) = timed("solve Hmat/GMRES", () -> solve(dad2))
            println("  T range (H) = ", extrema(dad2.T[1:dad2.n]))
        end
    catch e
        println("  H-matrix path skipped/failed: ", e)
    end

    section("5. Scaling scan (dense assembly only)")
    println("  ndiv   N      t_assemble[s]   mem_H[MiB]   t/N² [ns]")
    for nd in (8, 12, 16, 24, 32)
        m = quadrado(ndiv=nd, show=false, nome="scale_$nd")
        d = format2d(m, props; pontointerno=false)
        N = d.n
        H_G_full_direct(d, 12)  # warmup compile
        GC.gc(false)
        t0 = time_ns()
        d3 = format2d(m, props; pontointerno=false)
        H_G_full_direct(d3, 12)
        dt = (time_ns() - t0) / 1e9
        mem = bytes(d3.H) / 1024^2
        @printf("  %4d %5d   %10.3f   %10.2f   %8.1f\n",
            nd, N, dt, mem, 1e9 * dt / N^2)
    end

    section("6. Type stability (hot functions)")
    # Build minimal typed args
    d = dad
    # sample fundamental / kernel if available
    try
        x = d.Nodes[1]
        y = d.Nodes[min(2, d.n)]
        nrm = d.Normal[min(2, d.n)]
        println("  fundamental(Laplace):")
        warntype_summary(stdout, fundamental, d.properties, x, y, nrm)
    catch e
        println("  fundamental warntype skipped: ", e)
    end

    println("  applyBC:")
    try
        warntype_summary(stdout, applyBC, d)
    catch e
        println("  skipped: ", e)
    end

    println("  solve (dense Laplace):")
    try
        warntype_summary(stdout, solve, d)
    catch e
        println("  skipped: ", e)
    end

    # Element Jacobian sample
    if !isempty(d.elements)
        el = d.elements[1]
        println("  element fields:")
        println("    typeof(element) = ", typeof(el))
        println("    typeof(Nodes)   = ", typeof(d.Nodes))
        println("    typeof(Normal)  = ", typeof(d.Normal))
        println("    typeof(BC/BV)   = ", typeof(d.BC), " / ", typeof(d.BV))
        println("    typeof(H)       = ", typeof(d.H))
        # abstract eltype checks
        for (name, v) in (
            "Nodes" => d.Nodes,
            "elements" => d.elements,
            "Normal" => d.Normal,
            "BC" => d.BC,
            "BV" => d.BV,
        )
            et = eltype(v)
            concrete = isconcretetype(et)
            println("    eltype($name) = $et  concrete=$concrete")
        end
    end

    section("7. Allocation micro-benchmarks (BenchmarkTools)")
    d = format2d(msh, props; pontointerno=false)
    H_G_full_direct(d, 12)
    applyBC(d)
    A = d.A
    b = d.b
    println("  A\\b  (N=$(size(A,1))):")
    display(@benchmark $A \ $b samples=5 evals=1 seconds=15)
    println()
    println("  mul! residual-like:")
    x = rand(size(A, 1))
    y = similar(x)
    display(@benchmark mul!($y, $A, $x) samples=20 evals=10)
    println()

    section("8. Summary & recommendations")
    N = dad.n
    println("""
  Observed structure
  ------------------
  • Mesh + format2d: geometry/BC ingest (Gmsh API + collocation setup).
  • Dense assembly: O(N²) kernel evaluations + singular integration — usually
    the dominant CPU term for moderate N.
  • Storage: dense H,G each N×N Float64 ⇒ ~16 N² bytes together.
      N=$(N) → ~$(fmt_bytes(2*N*N*8)) for H+G alone.
  • Solve: LU O(N³) for dense A; GMRES for H-matrix / MixedBCOperator.

  Likely bottlenecks (typical BEM)
  --------------------------------
  1. Dense H/G assembly (kernel + quadrature) — CPU & memory bandwidth.
  2. Dense H/G RAM (limits N ≲ few·10³ on laptops).
  3. Package load / precompile (GLMakie, DiffEq) — not runtime of solve.
  4. Type instability in hot loops (if Any/abstract eltypes) → extra allocs.

  Improvements
  ------------
  • Prefer H_G_Hmat / HSS for N ≳ 1–2·10³; keep dense for verification.
  • Cut npg on far interactions; keep high-order only near singular.
  • Ensure Nodes::Vector{SVector{2,Float64}}, BC::Vector{Int}, matrices
    Matrix{Float64} — avoid Vector{Any} / abstract Problem in inner loops.
  • Preallocate element quadrature buffers; reuse workspace in assembly.
  • Thread assembly over collocation rows (Polyester/Threads) if not already.
  • Lazy-load GLMakie/DifferentialEquations to cut precompile & `using` time.
  • For production scripts: PackageCompiler sysimage after API stabilizes.
""")
    return dad
end

function main(args=ARGS)
    ndiv = _parse_ndiv(args)
    println("BEM profiling  ndiv=$ndiv  threads=$(Threads.nthreads())")
    profile_pipeline(; ndiv)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
